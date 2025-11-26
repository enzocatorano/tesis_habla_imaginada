# Script: carga, resampleo, segmentación y augmentación (Butterworth) para 15 sujetos
# - Lee archivos 'data/processed/S01_EEG.npz', ... 'S15_EEG.npz' (ajustá path si hace falta)
# - Reestructura datos a (trials, channels, samples) según el ejemplo del usuario
# - Resamplea de 1024 -> 128 Hz con resample_poly (scipy)
# - Segmenta cada trial en ventanas de 1.5 s con hop 0.5 s (6 ventanas por trial si trial = 4s)
# - Para cada ventana genera 5 augmentaciones (una por banda: delta..gamma) usando Butterworth + sosfiltfilt
# - A cada ventana aumentada se le asignan las etiquetas originales del trial y una cuarta etiqueta = índice de banda (0..4)
# - Guarda los resultados por sujeto en 'data/processed_aug/SXX_augmented.npz'
#
# Requisitos: scipy, numpy. Ejecutar en el entorno donde estén los .npz originales.

import os
import numpy as np
from pathlib import Path

# --- Configurable por el usuario ---
# carpeta donde está este script (src/)
SCRIPT_DIR = Path(__file__).resolve().parent
# directorio raíz del proyecto (una carpeta arriba de src/)
PROJECT_ROOT = SCRIPT_DIR.parent
# ruta a data/processed desde el script
DATA_DIR_DEFAULT = PROJECT_ROOT / "data" / "processed"
OUT_DIR_DEFAULT = PROJECT_ROOT / "data" / "processed_aug"
# convertir a string si es necesario por compatibilidad con APIs antiguas
DATA_DIR = str(DATA_DIR_DEFAULT)
OUT_DIR = str(OUT_DIR_DEFAULT)
SUBJECTS = [f"S{str(i).zfill(2)}" for i in range(1, 16)]  # S01 .. S15
FILE_TEMPLATE = "{subj}_EEG.npz"
KEY_IN_NPZ = "data"  # si el .npz usa otra key, ajustá o el script lo detectará
FS_ORIG = 1024
FS_TARGET = 128
WIN_SEC = 1.5
HOP_SEC = 0.5
BUTTER_ORDER = 4
NOISE_FACTOR = 0.5   # energía del ruido = energía de banda limpia * NOISE_FACTOR
PER_CHANNEL_SCALE = True  # escala por canal

# -----------------------------------
def require_scipy():
    try:
        import scipy.signal as sps
        return sps
    except Exception as e:
        raise ImportError("Este script requiere scipy (scipy.signal). Instalalo con `pip install scipy`.") from e

# --- Filtrado Butterworth (SOS + zero-phase) ---
def band_defs():
    return {
        'delta': (0.5, 4.0),
        'theta': (4.0, 8.0),
        'alpha': (8.0, 12.0),
        'beta' : (12.0, 32.0),
        'gamma': (32.0, 63.5)
    }

def bandpass_sosfiltfilt(x, low, high, fs, order=4, axis=1):
    """
    Filtra x (n_channels, n_samples) en [low, high] usando butter(order) con sosfiltfilt (zero-phase).
    Reduce orden si filtfilt falla por padlen.
    """
    sps = require_scipy()
    cur_order = int(order)
    last_exc = None
    while cur_order >= 1:
        try:
            sos = sps.butter(cur_order, [low, high], btype='band', fs=fs, output='sos')
            y = sps.sosfiltfilt(sos, x, axis=axis)
            return y, cur_order
        except Exception as e:
            last_exc = e
            cur_order -= 1
    raise RuntimeError(f"sosfiltfilt falló (intentadas órdenes hasta 1). Error: {last_exc}")

def generate_band_limited_noise_butter(n_channels, n_samples, low, high, fs, order=4, seed=None):
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, 1, size=(n_channels, n_samples)).astype(float)
    noise_filt, used_order = bandpass_sosfiltfilt(noise, low, high, fs, order=order, axis=1)
    return noise_filt, used_order

def compute_energy(x):
    # energía por canal: suma de cuadrados
    return np.sum(x.astype(float)**2, axis=1)

def augment_window_bands(window, fs=128, factor=0.5, order=4, seed=None, per_channel_scale=True):
    """
    window: (channels, samples)
    Devuelve dict band_name -> dict with 'augmented' and metadata
    """
    bands = band_defs()
    n_channels, n_samples = window.shape
    augmented = {}
    for i, (bname, (low, high)) in enumerate(bands.items()):
        # 1) extraer banda limpia
        band_clean, used_order = bandpass_sosfiltfilt(window, low, high, fs, order=order, axis=1)
        band_energy = compute_energy(band_clean)
        target_energy = band_energy * float(factor)

        # 2) generar ruido banda-limitado
        noise, _ = generate_band_limited_noise_butter(n_channels, n_samples, low, high, fs, order=order, seed=(None if seed is None else int(seed)+i))
        noise_energy = compute_energy(noise)

        # 3) escalar ruido
        eps = 1e-12
        if per_channel_scale:
            scales = np.sqrt((target_energy + eps) / (noise_energy + eps))
            scales = np.clip(scales, 0.0, 1e6)
            noise_scaled = noise * scales[:, None]
        else:
            total_target = np.sum(target_energy)
            total_noise = np.sum(noise_energy) + eps
            global_scale = np.sqrt(total_target / total_noise)
            noise_scaled = noise * global_scale

        # 4) sumar ruido
        augmented_trial = window.astype(float) + noise_scaled

        info = {
            'augmented': augmented_trial,
            'band_clean_energy_per_channel': band_energy,
            'noise_energy_before_scaling_per_channel': noise_energy,
            'noise_energy_after_scaling_per_channel': compute_energy(noise_scaled),
            'factor': factor,
            'band': (low, high),
            'butter_order_used': used_order,
            'band_index': i  # 0..4
        }
        augmented[bname] = info
    return augmented

# --- Resampling util (usa resample_poly) ---
def resample_subject_signals(signal_flat, orig_fs=1024, target_fs=128):
    """
    signal_flat: (n_trials, 6*original_samples_per_trial) as in tu ejemplo
    Esto reestructura a (n_trials, channels=6, samples_orig) y resamplea por canal a target_fs.
    """
    sps = require_scipy()
    from scipy.signal import resample_poly

    n_trials = signal_flat.shape[0]
    # despiezamos: asumimos que signal_flat.shape[1] es divisible por 6
    samples_orig = signal_flat.shape[1] // 6
    signal = signal_flat.reshape(n_trials, 6, samples_orig)
    # nuevo array
    new_samples = int(round(samples_orig * float(target_fs) / float(orig_fs)))
    resampled = np.zeros((n_trials, 6, new_samples), dtype=float)
    for i in range(n_trials):
        for ch in range(6):
            resampled[i, ch, :] = resample_poly(signal[i, ch, :], up=target_fs, down=orig_fs)
    return resampled

# --- Segmentación en ventanas ---
def segment_trial_windows(trial_signal, fs=128, win_sec=1.5, hop_sec=0.5):
    """
    trial_signal: (channels, samples)
    devuelve lista/array de ventanas shape (n_windows, channels, win_samples)
    """
    n_channels, n_samples = trial_signal.shape
    win_samples = int(round(win_sec * fs))
    hop_samples = int(round(hop_sec * fs))
    starts = list(range(0, n_samples - win_samples + 1, hop_samples))
    windows = np.stack([trial_signal[:, s:s+win_samples] for s in starts], axis=0) if len(starts)>0 else np.empty((0, n_channels, win_samples))
    return windows, starts

# --- Procesado completo por sujeto ---
def process_subject(filepath, out_dir=OUT_DIR, fs_orig=FS_ORIG, fs_target=FS_TARGET, win_sec=WIN_SEC, hop_sec=HOP_SEC,
                    butter_order=BUTTER_ORDER, noise_factor=NOISE_FACTOR, per_channel_scale=PER_CHANNEL_SCALE):
    print(f"\nProcesando {filepath} ...")
    if not os.path.exists(filepath):
        print("  Archivo no encontrado:", filepath)
        return None
    data = np.load(filepath)
    # encontrar key con array principal (por defecto 'data')
    if KEY_IN_NPZ in data:
        eeg_data = data[KEY_IN_NPZ]
    else:
        # tomar la primera matriz encontrada
        found = False
        for k in data.files:
            if isinstance(data[k], np.ndarray) and data[k].ndim >= 2:
                eeg_data = data[k]
                found = True
                print(f"  Key '{KEY_IN_NPZ}' no encontrada. Usando key '{k}' del .npz")
                break
        if not found:
            raise RuntimeError("No se encontró un array válido en el archivo .npz")

    # separar labels y señales según ejemplo del usuario
    labels = eeg_data[:, -3:]  # (n_trials, 3)
    eeg_signals_flat = eeg_data[:, :-3]  # (n_trials, 6*samples_orig)
    # resample y reestructura
    signals_res = resample_subject_signals(eeg_signals_flat, orig_fs=fs_orig, target_fs=fs_target)  # (n_trials, 6, new_samples)
    n_trials, n_ch, n_samples = signals_res.shape
    print(f"  Señales reestructuradas a (trials, ch, samples): {signals_res.shape} (fs={fs_target})")

    # preparar contenedores para augmented data
    augmented_list = []
    augmented_labels = []  # shape (N_aug, 4) -> original 3 labels + band_index
    band_names = list(band_defs().keys())

    total_windows = 0
    total_aug = 0

    for t in range(n_trials):
        trial_sig = signals_res[t]  # (6, samples)
        windows, starts = segment_trial_windows(trial_sig, fs=fs_target, win_sec=win_sec, hop_sec=hop_sec)
        total_windows += windows.shape[0]
        # para cada ventana, generar 5 augmentaciones (una por banda)
        for w_idx in range(windows.shape[0]):
            w = windows[w_idx]
            aug_dict = augment_window_bands(w, fs=fs_target, factor=noise_factor, order=butter_order, seed=(t*1000 + w_idx), per_channel_scale=per_channel_scale)
            # conservar etiquetas originales y añadir cuarta etiqueta = band_index
            for bname, info in aug_dict.items():
                band_index = info['band_index']  # 0..4
                new_label = np.concatenate([labels[t].astype(float), np.array([band_index], dtype=float)])
                augmented_list.append(info['augmented'].astype(np.float32))
                augmented_labels.append(new_label.astype(np.float32))
                total_aug += 1

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    # guardar arrays
    augmented_array = np.stack(augmented_list, axis=0) if len(augmented_list)>0 else np.empty((0, n_ch, int(round(win_sec*fs_target))), dtype=np.float32)
    labels_array = np.stack(augmented_labels, axis=0) if len(augmented_labels)>0 else np.empty((0, 4), dtype=np.float32)

    subj_base = os.path.basename(filepath).replace(".npz", "")
    out_path = os.path.join(out_dir, f"{subj_base}_augmented.npz")
    np.savez_compressed(out_path, data=augmented_array, labels=labels_array, band_names=np.array(band_names))
    print(f"  Guardado {augmented_array.shape[0]} muestras aumentadas en {out_path}")
    print(f"  Ventanas generadas por trial (total): {total_windows}, augmentaciones totales: {total_aug}")
    return out_path

# --- Ejecutar para todos los sujetos ---
def run_all():
    sps = require_scipy()  # fallará si scipy no está disponible
    results = {}
    for subj in SUBJECTS:
        filename = FILE_TEMPLATE.format(subj=subj)
        filepath = os.path.join(DATA_DIR, filename)
        out = process_subject(filepath, out_dir=OUT_DIR, fs_orig=FS_ORIG, fs_target=FS_TARGET,
                              win_sec=WIN_SEC, hop_sec=HOP_SEC, butter_order=BUTTER_ORDER,
                              noise_factor=NOISE_FACTOR, per_channel_scale=PER_CHANNEL_SCALE)
        results[subj] = out
        print(f"  Resultados para {subj} listos.")
    return results

if __name__ == "__main__":
    print("Iniciando procesamiento de sujetos...")
    res = run_all()
    print("Hecho. Archivos guardados:", res)
