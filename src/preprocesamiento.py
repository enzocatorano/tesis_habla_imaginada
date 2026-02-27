import os
import numpy as np
import json
from scipy.signal import resample_poly

def preprocess_subjects(start_sub=1, end_sub=15):
    # --- Configuración de parámetros ---
    fs_original = 1024
    fs_target = 128
    n_channels = 6
    duration_sec = 4
    n_samples_orig = fs_original * duration_sec  # 4096 muestras por canal
    total_signal_features = n_channels * n_samples_orig # 24576
    
    down_factor = fs_original // fs_target # 8
    
    input_dir = os.path.join(".", "data", "original")
    output_dir = os.path.join(".", "data", "preprocessed")
    os.makedirs(output_dir, exist_ok=True)

    metadata = {
        "pipeline": "Extraction -> Reshape -> Downsampling",
        "input_structure": "trials x (signals + 3 labels)",
        "fs_original": fs_original,
        "fs_target": fs_target,
        "final_shape": f"(trials, {n_channels}, {fs_target * duration_sec})",
        "subjects_processed": []
    }

    print(f"--- Iniciando procesamiento de {start_sub} a {end_sub} ---")

    for i in range(start_sub, end_sub + 1):
        filename = f"S{i:02d}_EEG.npz"
        path = os.path.join(input_dir, filename)
        
        if not os.path.exists(path):
            continue

        with np.load(path) as loader:
            # Extraemos el array (asumiendo que es el primero en el .npz)
            raw_data = loader[loader.files[0]]
        
        # 1. Separar Datos de Etiquetas
        # Señales: todas las filas, hasta la columna 24576
        signals = raw_data[:, :total_signal_features]
        # Etiquetas: todas las filas, las últimas 3 columnas
        labels = raw_data[:, -3:]

        # 2. Reshape de las señales
        # (trials, 24576) -> (trials, 6, 4096)
        signals_reshaped = signals.reshape(-1, n_channels, n_samples_orig)

        # 3. Submuestreo (eje de tiempo es el último)
        # resample_poly aplica filtro anti-aliasing automáticamente
        signals_preprocessed = resample_poly(signals_reshaped, 1, down_factor, axis=-1)

        # 4. Guardar en .npz (incluimos datos y etiquetas por separado)
        out_name = f"S{i:02d}_preprocessed.npz"
        np.savez_compressed(
            os.path.join(output_dir, out_name), 
            x=signals_preprocessed, 
            y=labels
        )
        
        metadata["subjects_processed"].append(out_name)
        print(f"Sujeto {i:02d} OK | Señal: {signals_preprocessed.shape} | Labels: {labels.shape}")

    # Guardar bitácora de procesamiento
    with open(os.path.join(output_dir, "process_log.json"), "w") as f:
        json.dump(metadata, f, indent=4)

if __name__ == "__main__":
    preprocess_subjects()