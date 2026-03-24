"""
preprocess.py
============
Preprocesamiento de datos EEG según el método de Bolaños y Rufiner.

Replica exactamente el pipeline de MATLAB:
  1. Datos crudos downsampeados (128 Hz, 512 muestras por canal)
  2. Ventana deslizante con solapamiento + Hamming
  3. Descomposición en 6 bandas EEG (DWT+SWT con pywt, Daubechies-4)
  4. Extracción de Modos de Correlación Wavelet (WCM):
     autovalores de A @ A.T donde A es la matriz 6×N de coeficientes por banda
  5. Aplanado: 6 canales × n_ventanas × 6 features = n_features totales

Uso:
  python preprocess.py

Todos los parámetros se configuran en la sección CABEZAL.
"""

import os
import json
import math
from pathlib import Path

import numpy as np
import pywt
from scipy.signal import butter, sosfiltfilt

########################################################################################
########################################################################################
# CABEZAL: Parámetros de preprocesamiento
########################################################################################

DATA_INPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "preprocessed"
DATA_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "preprocesamiento_segun_bolanos_rufiner"
SUFIJO_ENTRADA = "_preprocessed"
SUFIJO_SALIDA = "_preprocessed"

SUBJECTS = list(range(1, 16))

WINDOW_SIZE = 0.2
OVERLAP = 0.25
FS = 128
WAVELET = "db4"

CHANNEL_NAMES = ["F3", "F4", "C3", "C4", "P3", "P4"]

########################################################################################
########################################################################################
# Filtros butterworth pre-calculados (para speedup)
########################################################################################

_WAVELET_OBJ = pywt.Wavelet(WAVELET)
_WAVELET_LO = np.asarray(_WAVELET_OBJ.dec_lo) 
_WAVELET_HI = np.asarray(_WAVELET_OBJ.dec_hi)

########################################################################################
########################################################################################
# Funciones de preprocesamiento
########################################################################################

def hamming_window(n):
    """Ventana de Hamming de n muestras."""
    n = int(n)
    return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n) / (n - 1))


def dilate_filter(filt, factor):
    """
    Dilata el filtro insertando ceros (Algoritmo a trous).
    Ej: si factor=2, inserta 1 cero entre cada coeficiente.
    """
    if factor == 1:
        return filt
    dilated = np.zeros(len(filt) * factor)
    dilated[::factor] = filt
    return dilated


def conv_same_len(signal, filt):
    """
    Realiza una convolución que SIEMPRE devuelve la longitud de 'signal',
    evitando el comportamiento por defecto de np.convolve(mode='same').
    """
    full_conv = np.convolve(signal, filt, mode='full')
    shift = (len(filt) - 1) // 2
    return full_conv[shift : shift + len(signal)]


def mdwt_decomposition(vector):
    """
    Descomposición en 6 bandas EEG usando un árbol Wavelet Packet estacionario.
    Mantiene la longitud estricta de N/2 en todas las bandas.
    """
    vector = np.asarray(vector, dtype=np.float64)
    if len(vector) % 2 == 1:
        vector = np.append(vector, 0.0)

    # --- NIVEL 1 (Entrada 128 Hz -> Salida 64 Hz) ---
    # Aquí sí diezmamos para fijar la longitud de salida en N/2 [cite: 186-188, 191]
    gamma_band = conv_same_len(vector, _WAVELET_HI)[::2] # 32 - 64 Hz [cite: 185, 222-223]
    L1 = conv_same_len(vector, _WAVELET_LO)[::2]         # 0 - 32 Hz [cite: 184, 195-197]

    # --- NIVEL 2 (Fs efectiva = 64 Hz, Filtros originales) ---
    beta_band = conv_same_len(L1, _WAVELET_HI) # 16 - 32 Hz [cite: 203-204]
    L2 = conv_same_len(L1, _WAVELET_LO)        # 0 - 16 Hz [cite: 200]

    # --- NIVEL 3 (Dilatación x2) ---
    lo_x2 = dilate_filter(_WAVELET_LO, 2)
    hi_x2 = dilate_filter(_WAVELET_HI, 2)
    
    L3_low = conv_same_len(L2, lo_x2)  # 0 - 8 Hz
    L3_high = conv_same_len(L2, hi_x2) # 8 - 16 Hz

    # --- NIVEL 4 (Dilatación x4) ---
    lo_x4 = dilate_filter(_WAVELET_LO, 4)
    hi_x4 = dilate_filter(_WAVELET_HI, 4)

    # Separamos 0-8 Hz
    delta_band = conv_same_len(L3_low, lo_x4)  # 0 - 4 Hz [cite: 164]
    theta_band = conv_same_len(L3_low, hi_x4)  # 4 - 8 Hz [cite: 166]
    
    # Separamos 8-16 Hz
    alpha_band = conv_same_len(L3_high, lo_x4) # 8 - 12 Hz [cite: 168]
    sigma_band = conv_same_len(L3_high, hi_x4) # 12 - 16 Hz [cite: 170]

    return np.vstack([
        delta_band,
        theta_band,
        alpha_band,
        sigma_band,
        beta_band,
        gamma_band
    ])


def extract_features_single_channel(channel_data, window_size, overlap, fs):
    """
    Extrae features WCM de un solo canal.

    Input:
      channel_data: (n_trials, n_samples) — un canal, varios trials
      window_size: float — segundos
      overlap: float — fracción de solapamiento [0, 1)
      fs: int — frecuencia de muestreo

    Output:
      feat_out: (n_trials, n_windows * 6)

    Cada trial se segmenta en ventanas deslizantes.
    Cada ventana: Hamming × datos → mdwt → autovalores de A @ A.T
    Optimizado: filtros pre-calculados + sosfiltfilt.
    """
    n_trials, n_samples = channel_data.shape

    int_size = int(round(window_size * fs))

    if overlap >= 1:
        advance = 1
    else:
        advance = math.ceil((1 - overlap) * int_size)

    indexes = list(range(1, n_samples - int_size + 1, advance))
    I = len(indexes)

    ham = hamming_window(int_size)

    feat_out = np.zeros((n_trials, I * 6))

    for k in range(I):
        start = indexes[k] - 1
        end = start + int_size

        chunk = channel_data[:, start:end] * ham[np.newaxis, :]

        lambda_features = np.zeros((n_trials, 6))

        for i in range(n_trials):
            mat = mdwt_decomposition(chunk[i, :])
            corr_matrix = mat @ mat.T
            eigenvalues = np.linalg.eigvalsh(corr_matrix)
            lambda_features[i, :] = eigenvalues

        feat_out[:, k * 6:(k + 1) * 6] = lambda_features

    return feat_out


def preprocess_subject(x_raw, window_size, overlap, fs):
    """
    Preprocesa todos los trials de un sujeto para los 6 canales.

    Input:
      x_raw: (n_trials, 6, 512) — 6 canales × 512 muestras
      window_size: float — segundos
      overlap: float — fracción
      fs: int — frecuencia

    Output:
      X: (n_trials, 6 * n_windows * 6) — features aplanadas
      n_windows: int — número de ventanas por canal
    """
    n_trials, n_channels, n_samples = x_raw.shape

    feat_per_channel = []

    for ch in range(n_channels):
        channel_data = x_raw[:, ch, :]
        feat_ch = extract_features_single_channel(channel_data, window_size, overlap, fs)
        feat_per_channel.append(feat_ch)

    X = np.hstack(feat_per_channel)
    n_windows = feat_per_channel[0].shape[1] // 6

    return X, n_windows


def preprocess_all_subjects(subjects, data_input_dir, data_output_dir,
                            suffix_in, suffix_out,
                            window_size, overlap, fs):
    """
    Preprocesa todos los sujetos y guarda los resultados en cache.

    Input:
      subjects: list[int] — lista de sujetos, ej [1,2,3]
      data_input_dir: Path
      data_output_dir: Path
      suffix_in: str
      suffix_out: str
      window_size: float
      overlap: float
      fs: int
    """
    data_output_dir = Path(data_output_dir)
    data_output_dir.mkdir(parents=True, exist_ok=True)

    n_features_total = None
    n_windows_total = None

    for subj in subjects:
        fixed_subj = f"{subj:02d}"
        output_path = data_output_dir / f"S{fixed_subj}{suffix_out}.npz"

        if output_path.exists():
            print(f"  Sujeto {subj:02d} ya procesado, saltando.")
            continue

        print(f"  Procesando sujeto {subj:02d}...", end=" ", flush=True)

        input_path = data_input_dir / f"S{fixed_subj}{suffix_in}.npz"

        if not input_path.exists():
            print(f"ADVERTENCIA: {input_path} no encontrado, saltando.")
            continue

        data = np.load(input_path, allow_pickle=True)
        x_raw = data["x"]
        y = data["y"]

        X, n_windows = preprocess_subject(x_raw, window_size, overlap, fs)

        if n_features_total is None:
            n_features_total = X.shape[1]
            n_windows_total = n_windows

        if X.shape[1] != n_features_total:
            print(f"\n  ERROR: Sujeto {subj} tiene {X.shape[1]} features, "
                  f"esperado {n_features_total}. Saltando.")
            continue

        output_path = data_output_dir / f"S{fixed_subj}{suffix_out}.npz"
        np.savez_compressed(output_path, x=X, y=y)

        print(f"OK — {X.shape[0]} trials, {X.shape[1]} features, "
              f"{n_windows} ventanas/canal.")

    details = {
        "description": "Preprocesamiento según Bolaños y Rufiner (Wavelet Correlation Modes)",
        "feature_type": "WCM",
        "feature_extraction": "Autovalores de A @ A.T donde A es la matriz 6×N/2 de coeficientes wavelet por banda",
        "window_size_sec": window_size,
        "overlap": overlap,
        "fs_hz": fs,
        "wavelet": "db4",
        "n_windows_per_channel": n_windows_total,
        "n_features_per_channel": n_windows_total * 6,
        "n_channels": 6,
        "n_features_total": n_features_total,
        "channels": CHANNEL_NAMES,
        "bands": ["delta", "theta", "alpha", "sigma", "beta", "gamma"],
        "subjects_processed": subjects,
        "input_dir": str(data_input_dir),
        "output_dir": str(data_output_dir),
        "notes": (
            "Cada trial produce: 6 canales × n_ventanas × 6 features = n_features_total. "
            "Labels preservadas: [modalidad, estímulo, artefacto] sin modificaciones."
        )
    }

    details_path = data_output_dir / "details.json"
    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)

    print(f"\nPreprocesamiento completo. Datos en: {data_output_dir}")
    print(f"Detalles guardados en: {details_path}")


########################################################################################
########################################################################################
# main
########################################################################################

if __name__ == "__main__":

    print("=" * 60)
    print("PREPROCESAMIENTO SEGÚN BOLAÑOS Y RUFINER")
    print("=" * 60)
    print(f"Entrada:   {DATA_INPUT_DIR}")
    print(f"Salida:    {DATA_OUTPUT_DIR}")
    print(f"Sujetos:   {SUBJECTS}")
    print(f"Ventana:   {WINDOW_SIZE} s")
    print(f"Overlap:   {OVERLAP * 100:.0f}%")
    print(f"Fs:        {FS} Hz")
    print(f"Ondita:    {WAVELET}")
    print("=" * 60)

    preprocess_all_subjects(
        subjects=SUBJECTS,
        data_input_dir=DATA_INPUT_DIR,
        data_output_dir=DATA_OUTPUT_DIR,
        suffix_in=SUFIJO_ENTRADA,
        suffix_out=SUFIJO_SALIDA,
        window_size=WINDOW_SIZE,
        overlap=OVERLAP,
        fs=FS,
    )
