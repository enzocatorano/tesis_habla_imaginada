#!/usr/bin/env python3
"""
Genera dataset donde cada trial es la matriz de covarianza espectro-espacial
(descompuesto en bandas EEG clasicas via DWT a-trous estilo bolanos_rufiner).

Fuente: data/clean_preprocessed/SXX_clean.npz (x: (trials, 6, 512) a 128 Hz)
Salida: data/clean_preprocessed_corr_bandas/SXX_clean.npz (x: (trials, 30, 30))

Por cada trial (C x T):
  1. Cada canal se descompone en 5 bandas (db4, arbol a-trous estacionario,
      al estilo de preprocesamiento_segun_bolanos_rufiner):
        delta (0-4 Hz), theta (4-8 Hz), alpha (8-12 Hz), beta (12-32 Hz), gamma (32-64 Hz)
      Beta (12-32) se reconstruye sumando los nodos 12-16 y 16-32 del arbol.
  2. Se apila el trial en una senal (n_bandas * C, N/2) con orden [delta..gamma] x [F3,F4,C3,C4,P3,P4].
  3. Se resta la media por fila y se calcula la covarianza espectral completa:
       R = (Xc @ Xc.T) / (T - 1)  ->  (30, 30)
     Los bloques 6x6 de la diagonal son las covarianzas por banda; los bloques
     fuera de diagonal son la covarianza entre bandas. Toda la informacion queda
     disponible y separar por banda se hace en el experimento.

Las etiquetas (y) se conservan sin cambios.
"""

import numpy as np
import json
import pywt
from pathlib import Path

INPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "clean_preprocessed"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "clean_preprocessed_corr_bandas"

N_CHANNELS = 6
SUBJ_IDS = [f"S{i:02d}" for i in range(1, 16)]

WAVELET = "db4"
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
CHANNEL_NAMES = ["F3", "F4", "C3", "C4", "P3", "P4"]

_WAVELET_OBJ = pywt.Wavelet(WAVELET)
_WAVELET_LO = np.asarray(_WAVELET_OBJ.dec_lo)
_WAVELET_HI = np.asarray(_WAVELET_OBJ.dec_hi)


####################################################################################################################################
# Filtros / descomposicion DWT a-trous
####################################################################################################################################

def dilate_filter(filt, factor):
    """Dilata el filtro insertando ceros (algoritmo a trous)."""
    if factor == 1:
        return filt
    dilated = np.zeros(len(filt) * factor)
    dilated[::factor] = filt
    return dilated


def conv_same_len(signal, filt):
    """Convolucion que devuelve la misma longitud que 'signal'."""
    full_conv = np.convolve(signal, filt, mode='full')
    shift = (len(filt) - 1) // 2
    return full_conv[shift: shift + len(signal)]


def dwt_a_trous_5_bandas(vector):
    """
    Descomposicion en 5 bandas EEG usando un arbol wavelet estacionario a-trous.
    Mantiene la longitud N/2 (256) en todas las bandas para entrada de 512 muestras.
    Bandas: delta (0-4), theta (4-8), alpha (8-12), beta (12-32), gamma (32-64).
    Beta (12-32) se reconstruye sumando los nodos 12-16 y 16-32 del arbol a-trous.

    Retorna: (5, N/2)
    """
    vector = np.asarray(vector, dtype=np.float64)
    if len(vector) % 2 == 1:
        vector = np.append(vector, 0.0)

    gamma_band = conv_same_len(vector, _WAVELET_HI)[::2]  # 32 - 64 Hz
    L1 = conv_same_len(vector, _WAVELET_LO)[::2]           # 0 - 32 Hz

    beta_16_32 = conv_same_len(L1, _WAVELET_HI)            # 16 - 32 Hz
    L2 = conv_same_len(L1, _WAVELET_LO)                    # 0 - 16 Hz

    lo_x2 = dilate_filter(_WAVELET_LO, 2)
    hi_x2 = dilate_filter(_WAVELET_HI, 2)

    L3_low = conv_same_len(L2, lo_x2)                      # 0 - 8 Hz
    L3_high = conv_same_len(L2, hi_x2)                     # 8 - 16 Hz

    lo_x4 = dilate_filter(_WAVELET_LO, 4)
    hi_x4 = dilate_filter(_WAVELET_HI, 4)

    delta_band = conv_same_len(L3_low, lo_x4)              # 0 - 4 Hz
    theta_band = conv_same_len(L3_low, hi_x4)              # 4 - 8 Hz
    alpha_band = conv_same_len(L3_high, lo_x4)             # 8 - 12 Hz
    sigma_band = conv_same_len(L3_high, hi_x4)             # 12 - 16 Hz
    beta_band = sigma_band + beta_16_32                    # 12 - 32 Hz

    return np.vstack([delta_band, theta_band, alpha_band, beta_band, gamma_band])


def trial_to_spectral_cov(trial):
    """
    Calcula la matriz de covarianza espectro-espacial para un trial.
    Input: (n_channels, n_timepoints) -> Output: (n_bands * n_channels, n_bands * n_channels).
    Orden de filas/cols: [delta..gamma] x [F3,F4,C3,C4,P3,P4] (banda-major).
    """
    n_ch, n_tp = trial.shape
    A = np.zeros((len(BANDS) * n_ch, n_tp // 2), dtype=float)

    for b in range(len(BANDS)):
        for ch in range(n_ch):
            A[b * n_ch + ch, :] = dwt_a_trous_5_bandas(trial[ch])[b]

    Xc = A - A.mean(axis=1, keepdims=True)
    return (Xc @ Xc.T) / (A.shape[1] - 1)


def process_subject(subj_id: str) -> dict:
    """Procesa un sujeto: convierte cada trial a matriz de covarianza espectro-espacial."""
    src_path = INPUT_DIR / f"{subj_id}_clean.npz"
    if not src_path.exists():
        raise FileNotFoundError(f"No encontrado: {src_path}")

    data = np.load(src_path, allow_pickle=True)
    X = data["x"]  # (trials, n_channels, n_timepoints)
    Y = data["y"]  # (trials, 3)

    n_trials, n_ch, n_tp = X.shape
    if n_ch != N_CHANNELS:
        raise ValueError(f"[{subj_id}] Esperaba {N_CHANNELS} canales, se encontraron {n_ch}")

    n_bands = len(BANDS)
    X_cov = np.empty((n_trials, n_bands * n_ch, n_bands * n_ch), dtype=float)
    for i in range(n_trials):
        X_cov[i] = trial_to_spectral_cov(X[i])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{subj_id}_clean.npz"
    np.savez_compressed(out_path, x=X_cov, y=Y)

    print(f"  {subj_id}: {n_trials} trials -> {X_cov.shape}")

    return {"id": subj_id, "trials": int(n_trials)}


def process_all_subjects():
    """Procesa todos los sujetos S01-S15 y genera process_log.json."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    subjects_data = []
    for subj_id in SUBJ_IDS:
        print(f"Procesando {subj_id}...")
        subjects_data.append(process_subject(subj_id))

    total_trials = sum(s["trials"] for s in subjects_data)
    n_bands = len(BANDS)

    band_blocks = {
        band: f"filas/cols {b * N_CHANNELS} a {b * N_CHANNELS + N_CHANNELS - 1} ({band})"
        for b, band in enumerate(BANDS)
    }

    log = {
        "pipeline": "Covarianza espectro-espacial por trial (5 bandas DWT a-trous + centrado por fila, sin normalizar)",
        "description": "Cada trial EEG se descompone en 5 bandas por canal (db4, arbol wavelet estacionario a-trous "
                       "como en preprocesamiento_segun_bolanos_rufiner), se apila banda-major en una senal "
                       f"({n_bands}x{n_bands} bandas x canales, N/2 puntos por fila), se resta la media por fila (Xc) "
                       "y se computa R = (Xc @ Xc.T) / (T - 1). "
                       f"La matriz ({n_bands * N_CHANNELS}x{n_bands * N_CHANNELS}) es simetrica y sus valores NO estan "
                       "acotados (-inf a inf). Los bloques 6x6 de la diagonal son las covarianzas por banda.",
        "source": "data/clean_preprocessed/",
        "fs": 128,
        "wavelet": WAVELET,
        "decomposition": "DWT a-trous estacionario (dvdb4), al estilo de preprocesamiento_segun_bolanos_rufiner",
        "bands": BANDS,
        "bands_hz": {"delta": "0-4", "theta": "4-8", "alpha": "8-12", "beta": "12-32 (reconstruido: nodos 12-16 + 16-32)", "gamma": "32-64"},
        "band_blocks": band_blocks,
        "shape": f"(trials, {n_bands * N_CHANNELS}, {n_bands * N_CHANNELS})",
        "row_order": "banda-major: [delta, theta, alpha, beta, gamma] x [F3, F4, C3, C4, P3, P4]",
        "trials": {"total": total_trials},
        "labels": {
            "column_0_modalidad": {
                "description": "Modalidad del habla",
                "values": {"1": "Imaginada (unico valor en este dataset)"},
                "note": "Solo trials de habla imaginada - se eliminan los pronunciados"
            },
            "column_1_estimulo": {
                "description": "Estimulo",
                "values": {
                    "1": "A (vocal)", "2": "E (vocal)", "3": "I (vocal)", "4": "O (vocal)", "5": "U (vocal)",
                    "6": "Arriba (comando)", "7": "Abajo (comando)", "8": "Adelante (comando)",
                    "9": "Atras (comando)", "10": "Derecha (comando)", "11": "Izquierda (comando)"
                }
            },
            "column_2_artefacto": {
                "description": "Presencia de artefacto",
                "values": {"1": "Limpio (sin parpadeo - unico valor en este dataset)"},
                "note": "Solo trials limpios incluidos - se eliminan los de parpadeo"
            }
        },
        "channels": {
            "count": N_CHANNELS,
            "names": CHANNEL_NAMES,
            "layout": "Sistema internacional 10-20"
        },
        "time": {
            "duration_seconds": 4,
            "samples_per_trial": 512,
            "fs": 128,
            "note": "El trial completo se usa para la descomposicion y la covarianza (sin ventaneo)"
        },
        "covariance": {
            "type": "espectro-espacial (sin normalizar por desviacion estandar)",
            "formula": "R = (Xc @ Xc.T) / (T - 1) sobre la senal (30, 256) descompuesta en bandas",
            "centering": "se resta la media por fila (axis=1) antes del producto",
            "estimated_over": "tiempo (256 muestras por banda del trial)",
            "per_trial_matrix": f"({n_bands * N_CHANNELS}x{n_bands * N_CHANNELS}) simetrica",
            "range": "(-inf, inf) - valores no acotados",
            "note": "No se divide por la desviacion estandar a diferencia de la correlacion de Pearson"
        },
        "subjects": subjects_data,
        "subjects_processed": [f"{s['id']}_clean.npz" for s in subjects_data],
        "data_format": {
            "x": f"(trials, {n_bands * N_CHANNELS}, {n_bands * N_CHANNELS}) - matriz de covarianza espectro-espacial (sin normalizar)",
            "y": "(trials, 3) - etiquetas [modalidad, estimulo, artefacto] (sin cambios)"
        },
        "note": "Version de data/clean_preprocessed descompuesta en 5 bandas EEG clasicas (db4 a-trous) donde cada "
                "trial es la covarianza espectro-espacial completa entre (banda, canal). Separar por banda equivale a "
                "extraer los bloques 6x6 de la diagonal."
    }

    log_path = OUTPUT_DIR / "process_log.json"
    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2, ensure_ascii=False)

    print(f"\n=== COMPLETADO ===")
    print(f"Directorio: {OUTPUT_DIR}")
    print(f"Total trials: {total_trials}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    process_all_subjects()