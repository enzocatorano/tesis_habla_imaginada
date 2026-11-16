#!/usr/bin/env python3
"""
Pipeline Unificado: Augmentación, Segmentación Temporal y Clasificación Riemanniana.

Basado en:
1. generate_clean_corr_bandas.py (Descomposición DWT a-trous en tiempo).
2. ML_TS_Riemman_experiment.py (Proyección Riemanniana y Clasificación L1).

Flujo por fold:
  1. División Train/Val/Test por índices de TRIALS ORIGINALES.
  2. Augmentación en el dominio del tiempo (solo Train).
  3. DWT a-trous multicanal (todas las particiones).
  4. Segmentación en ventanas temporales superpuestas.
  5. Cálculo de matrices de covarianza (5 bandas 6x6) por ventana.
  6. Proyección al espacio tangente Riemanniano (ref de media calculada solo con Train).
  7. Concatenación de features (105 dims) y entrenamiento de Regresión Logística L1.
"""

import os
import re
import json
import time
import random
import platform
import traceback
from pathlib import Path

import numpy as np
import pywt
from scipy.fft import rfft, irfft
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression

from pyriemann.utils.mean import mean_riemann
from pyriemann.tangentspace import tangent_space

# -----------------------------
# CONFIGURACIÓN GENERAL
# -----------------------------
EXPERIMENTS_ROOT = Path(__file__).resolve().parent.parent / "experiments"
EXPERIMENT_NAME = "ML_TS_Riemman_OvR_FTS_1TWS0.5_0.5_4.0"

# Rutas a los datos RAW (Series de tiempo 128Hz, sin procesar covarianzas aún)
INPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "clean_preprocessed"
SUFIJO_DATOS = "_clean"

# Tarea a resolver: "separado", "unificado", "binario", "one_vs_rest"
EXPERIMENT_MODE = "one_vs_rest"
SUBJECT = None # None = todos, o lista ej: [1, 2, 3]

MASTER_SEED = 42
N_SEEDS = 3
K_FOLDS = 5
VAL_FRAC = 0.2

# -----------------------------
# PARÁMETROS DE SEÑAL Y VENTANEO
# -----------------------------
FS = 128
TEMPORAL_WINDOW = (0.5, 4.0)   # Recorte inicial del trial en segundos
WINDOW_SIZE_SEC = 1.0          # Tamaño de la sub-ventana
WINDOW_STRIDE_SEC = 0.5        # Avance (solapamiento) de la sub-ventana
N_CANALES = 6
CHANNEL_NAMES = ["F3", "F4", "C3", "C4", "P3", "P4"]
BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]

# -----------------------------
# PARÁMETROS DE AUGMENTACIÓN (SOLO TRAIN)
# -----------------------------
AUG_FT_FRAC = 0.7          # % de trials de train a generar vía FT Surrogate
AUG_NOISE_FRAC = 0.7       # % de trials de train a generar vía Inyección de Ruido
NOISE_ENERGY_RATIO = 0.025 # Varianza del ruido como fracción de la varianza del trial
 
# -----------------------------
# CLASIFICADOR
# -----------------------------
LOGREG_L1_KWARGS_BASE = dict(penalty="l1", C=1.0, max_iter=2000, class_weight="balanced")
if EXPERIMENT_MODE == "separado":
    LOGREG_L1_KWARGS = dict(LOGREG_L1_KWARGS_BASE, solver="saga", multi_class="multinomial")
else:
    LOGREG_L1_KWARGS = dict(LOGREG_L1_KWARGS_BASE, solver="liblinear")


# =============================================================================
# FUNCIONES AUXILIARES: DWT A-TROUS
# =============================================================================
_WAVELET_OBJ = pywt.Wavelet("db4")
_WAVELET_LO = np.asarray(_WAVELET_OBJ.dec_lo)
_WAVELET_HI = np.asarray(_WAVELET_OBJ.dec_hi)

def dilate_filter(filt, factor):
    if factor == 1: return filt
    dilated = np.zeros(len(filt) * factor)
    dilated[::factor] = filt
    return dilated

def conv_same_len(signal, filt):
    full_conv = np.convolve(signal, filt, mode='full')
    shift = (len(filt) - 1) // 2
    return full_conv[shift: shift + len(signal)]

def dwt_a_trous_5_bandas(vector):
    """Retorna las 5 bandas (T/2 muestras cada una)."""
    vector = np.asarray(vector, dtype=np.float64)
    if len(vector) % 2 == 1:
        vector = np.append(vector, 0.0)

    gamma_band = conv_same_len(vector, _WAVELET_HI)[::2]
    L1 = conv_same_len(vector, _WAVELET_LO)[::2]
    beta_16_32 = conv_same_len(L1, _WAVELET_HI)
    L2 = conv_same_len(L1, _WAVELET_LO)

    lo_x2 = dilate_filter(_WAVELET_LO, 2)
    hi_x2 = dilate_filter(_WAVELET_HI, 2)
    L3_low = conv_same_len(L2, lo_x2)
    L3_high = conv_same_len(L2, hi_x2)

    lo_x4 = dilate_filter(_WAVELET_LO, 4)
    hi_x4 = dilate_filter(_WAVELET_HI, 4)
    delta_band = conv_same_len(L3_low, lo_x4)
    theta_band = conv_same_len(L3_low, hi_x4)
    alpha_band = conv_same_len(L3_high, lo_x4)
    sigma_band = conv_same_len(L3_high, hi_x4)
    beta_band = sigma_band + beta_16_32

    return np.vstack([delta_band, theta_band, alpha_band, beta_band, gamma_band])

def apply_dwt_multichannel(X_trials):
    """Aplica DWT a todos los trials: (N, 6, T) -> (N, 5, 6, T/2)"""
    N, C, T = X_trials.shape
    # La salida temporal es T/2 (asumiendo T par)
    X_dwt = np.zeros((N, len(BAND_NAMES), C, T // 2))
    for i in range(N):
        for c in range(C):
            X_dwt[i, :, c, :] = dwt_a_trous_5_bandas(X_trials[i, c, :])
    return X_dwt


# =============================================================================
# FUNCIONES AUXILIARES: AUGMENTACIÓN EN TIEMPO
# =============================================================================
def ft_surrogate_multichannel(trial):
    """Conserva la amplitud pero aleatoriza la fase (misma fase para todos los canales)."""
    fft_X = rfft(trial, axis=-1)
    phases = np.random.uniform(0, 2 * np.pi, size=fft_X.shape[-1])
    phases[0] = 0.0 # Componente DC
    if trial.shape[-1] % 2 == 0:
        phases[-1] = 0.0 # Frecuencia Nyquist
    
    fft_X_shifted = fft_X * np.exp(1j * phases)
    return irfft(fft_X_shifted, n=trial.shape[-1], axis=-1)

def noise_injection(trial, energy_ratio):
    """Inyecta ruido gaussiano escalado a un % de la varianza temporal de cada canal."""
    noise = np.random.normal(0, 1, size=trial.shape)
    var_signal = np.var(trial, axis=-1, keepdims=True)
    scaled_noise = noise * np.sqrt(var_signal * energy_ratio)
    return trial + scaled_noise

def apply_augmentations(X, y):
    """Genera y concatena copias aumentadas a partir del subset provisto."""
    N = len(X)
    n_ft = int(AUG_FT_FRAC * N)
    n_noise = int(AUG_NOISE_FRAC * N)
    
    X_aug, y_aug = [], []
    
    # FT Surrogate
    if n_ft > 0:
        idx_ft = np.random.choice(N, n_ft, replace=False)
        for idx in idx_ft:
            X_aug.append(ft_surrogate_multichannel(X[idx]))
            y_aug.append(y[idx])
            
    # Noise
    if n_noise > 0:
        idx_noise = np.random.choice(N, n_noise, replace=False)
        for idx in idx_noise:
            X_aug.append(noise_injection(X[idx], NOISE_ENERGY_RATIO))
            y_aug.append(y[idx])
            
    if X_aug:
        return np.vstack([X, np.array(X_aug)]), np.concatenate([y, np.array(y_aug)])
    return X, y


# =============================================================================
# FUNCIONES AUXILIARES: VENTANEO Y COVARIANZA
# =============================================================================
def segment_into_windows(X_dwt, y, fs_dwt, win_sec, stride_sec):
    """
    Desliza ventanas temporales sobre los trials DWT.
    X_dwt: (N, Bands, Channels, Time)
    Retorna X_wins: (W_total, Bands, Channels, T_win) y etiquetas correspondientes.
    """
    N, B, C, T = X_dwt.shape
    win_len = int(win_sec * fs_dwt)
    stride = int(stride_sec * fs_dwt)
    
    windows_X, windows_y = [], []
    for i in range(N):
        start = 0
        while start + win_len <= T:
            windows_X.append(X_dwt[i, :, :, start:start+win_len])
            windows_y.append(y[i])
            start += stride
            
    return np.array(windows_X), np.array(windows_y)

def calculate_covariances(X_wins):
    """
    Calcula covarianzas. X_wins: (W, Bands, Channels, Time)
    Retorna C: (W, Bands, Channels, Channels)
    """
    # Centrado por canal en cada ventana/banda
    Xc = X_wins - X_wins.mean(axis=-1, keepdims=True)
    T = Xc.shape[-1]
    
    # Producto Xc @ Xc.T (vectorizado sobre W y Bands)
    # Xc: (W, B, C, T) -> np.matmul entre (W,B,C,T) y (W,B,T,C) = (W,B,C,C)
    C = np.matmul(Xc, np.swapaxes(Xc, -1, -2)) / (T - 1)
    
    # Regularización de seguridad para la métrica SPD de Riemman
    I = np.eye(C.shape[-1])
    C = C + 1e-6 * I
    return C


# =============================================================================
# FUNCIONES AUXILIARES: GENERALES
# =============================================================================
def now_timestamp(): return time.strftime("%Y%m%d-%H%M%S", time.gmtime())

def set_global_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def save_json(path: Path, obj):
    with open(path, "w", encoding="utf8") as fh:
        json.dump(obj, fh, indent=2, default=lambda o: (o.tolist() if isinstance(o, np.ndarray) else str(o)))

def define_tarea(mode: str, Y_all: np.ndarray):
    """Retorna (subset_name, mask, y_target, n_classes)."""
    targets = []
    if mode in ("separado", "unificado"):
        subsets = [("vocales", 1, 5, 5), ("comandos", 6, 11, 6)] if mode == "separado" else [("unificado", 1, 11, 11)]
        for nombre, s_min, s_max, n_cl in subsets:
            mask = (Y_all[:, 1].astype(int) >= s_min) & (Y_all[:, 1].astype(int) <= s_max)
            targets.append((nombre, mask, (Y_all[:, 1].astype(int) - s_min), n_cl))
    elif mode == "binario":
        mask = np.ones(Y_all.shape[0], dtype=bool)
        targets.append(("vocal_vs_comando", mask, (Y_all[:, 1].astype(int) >= 6).astype(int), 2))
    elif mode == "one_vs_rest":
        mask_voc = (Y_all[:, 1].astype(int) >= 1) & (Y_all[:, 1].astype(int) <= 5)
        for v_id, v_name in {1:"A", 2:"E", 3:"I", 4:"O", 5:"U"}.items():
            targets.append((f"OvR_vocal_{v_name}", mask_voc, (Y_all[:, 1].astype(int) == v_id).astype(int), 2))
        mask_cmd = (Y_all[:, 1].astype(int) >= 6) & (Y_all[:, 1].astype(int) <= 11)
        for c_id, c_name in {6:"Arriba", 7:"Abajo", 8:"Izquierda", 9:"Derecha", 10:"Adelante", 11:"Atras"}.items():
            targets.append((f"OvR_comando_{c_name}", mask_cmd, (Y_all[:, 1].astype(int) == c_id).astype(int), 2))
    return targets


# =============================================================================
# BUCLE PRINCIPAL
# =============================================================================
if __name__ == '__main__':
    set_global_seed(MASTER_SEED)
    rng = np.random.default_rng(MASTER_SEED)
    SEEDS_LIST = rng.integers(low=0, high=2**31 - 1, size=N_SEEDS).tolist()

    ts = now_timestamp()
    host = platform.node().replace(" ", "_")
    EXPERIMENT_ROOT = EXPERIMENTS_ROOT / f"{EXPERIMENT_NAME}_{ts}_{host}"
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"=== SCRIPT DE AUGMENTACIÓN + VENTANEO Y RIEMANN ===")
    print(f"Root: {EXPERIMENT_ROOT}")

    # Exploración de archivos del dataset original RAW (Series temporales)
    subject_files = sorted([p for p in INPUT_DIR.iterdir() if p.is_file() and p.suffix == ".npz" and SUFIJO_DATOS in p.name])
    if SUBJECT is not None:
        wanted = [SUBJECT] if isinstance(SUBJECT, int) else SUBJECT
        subject_files = [p for p in subject_files if int(re.match(r"^[sS](\d+)", p.name).group(1)) in wanted]

    if not subject_files: raise FileNotFoundError(f"No hay archivos en {INPUT_DIR}")

    # Metadata de la configuración
    config = {
        "experiment_name": EXPERIMENT_NAME, "mode": EXPERIMENT_MODE,
        "temporal_window_raw": TEMPORAL_WINDOW, "window_size_sec": WINDOW_SIZE_SEC, "stride_sec": WINDOW_STRIDE_SEC,
        "augmentations": {"ft_frac": AUG_FT_FRAC, "noise_frac": AUG_NOISE_FRAC, "noise_energy": NOISE_ENERGY_RATIO},
        "classifier": "RegLog L1", "clf_kwargs": LOGREG_L1_KWARGS,
        "seeds": SEEDS_LIST, "k_folds": K_FOLDS, "val_frac": VAL_FRAC
    }
    save_json(EXPERIMENT_ROOT / "config.json", config)

    fs_dwt = FS // 2 # Porque el a-trous decima por 2 las muestras (efectivo en `conv_same_len(..)[::2]`)

    for subj_idx, subj_path in enumerate(subject_files, start=1):
        subj_name = subj_path.stem
        print(f"\n--- SUJETO {subj_idx}/{len(subject_files)}: {subj_name} ---")

        data = np.load(subj_path, allow_pickle=True)
        X_all, Y_all = data["x"], data["y"]

        # Recorte temporal inicial a cada Trial completo
        if TEMPORAL_WINDOW is not None:
            s_start, s_end = int(TEMPORAL_WINDOW[0] * FS), int(TEMPORAL_WINDOW[1] * FS)
            s_start, s_end = max(0, min(s_start, X_all.shape[-1])), max(s_start, min(s_end, X_all.shape[-1]))
            X_all = X_all[:, :, s_start:s_end]

        tareas = define_tarea(EXPERIMENT_MODE, Y_all)
        subj_out = EXPERIMENT_ROOT / subj_name
        subj_out.mkdir(exist_ok=True)

        for subset_name, mask, y_target, n_classes_target in tareas:
            if mask.sum() == 0: continue
            
            X_subset = X_all[mask]
            y_subset = y_target[mask]
            
            subset_out = subj_out / subset_name
            subset_out.mkdir(exist_ok=True)
            print(f"  [Tarea: {subset_name}] - Trials Originales: {len(X_subset)}")

            for seed_i, seed_val in enumerate(SEEDS_LIST):
                seed_out = subset_out / f"seed_{seed_i}"
                seed_out.mkdir(exist_ok=True)
                set_global_seed(seed_val)

                skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=seed_val)
                for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_subset, y_subset), start=1):
                    fold_out = seed_out / f"fold_{fold_idx}"
                    fold_out.mkdir(exist_ok=True)
                    
                    # 1. PARTICIÓN SOBRE TRIALS ORIGINALES (Previniendo Leakage)
                    X_tr_orig, y_tr_orig = X_subset[train_idx], y_subset[train_idx]
                    X_te_orig, y_te = X_subset[test_idx], y_subset[test_idx]
                    
                    if VAL_FRAC > 0.0:
                        idx_tr_f, idx_va_f = train_test_split(
                            np.arange(len(y_tr_orig)), test_size=VAL_FRAC,
                            stratify=y_tr_orig, random_state=seed_val + fold_idx, shuffle=True
                        )
                        X_tr, y_tr = X_tr_orig[idx_tr_f], y_tr_orig[idx_tr_f]
                        X_va, y_va = X_tr_orig[idx_va_f], y_tr_orig[idx_va_f]
                    else:
                        X_tr, y_tr = X_tr_orig, y_tr_orig
                        X_va, y_va = np.empty((0, *X_tr.shape[1:])), np.empty((0,))

                    # 2. AUGMENTACIÓN (Solo sobre Entrenamiento)
                    X_tr_aug, y_tr_aug = apply_augmentations(X_tr, y_tr)

                    # 3. DWT (A todos por separado)
                    # Output: (Trials, Bands, Channels, T/2)
                    X_tr_dwt = apply_dwt_multichannel(X_tr_aug)
                    X_va_dwt = apply_dwt_multichannel(X_va) if len(X_va) > 0 else np.empty((0, 5, 6, X_tr_dwt.shape[-1]))
                    X_te_dwt = apply_dwt_multichannel(X_te_orig)

                    # 4. SEGMENTACIÓN TEMPORAL (Ventaneo)
                    # Output: (Windows, Bands, Channels, Time_Win)
                    X_tr_win, y_tr_win = segment_into_windows(X_tr_dwt, y_tr_aug, fs_dwt, WINDOW_SIZE_SEC, WINDOW_STRIDE_SEC)
                    X_va_win, y_va_win = segment_into_windows(X_va_dwt, y_va, fs_dwt, WINDOW_SIZE_SEC, WINDOW_STRIDE_SEC) if len(X_va) > 0 else (np.empty((0, 5, 6, X_tr_win.shape[-1])), np.empty((0,)))
                    X_te_win, y_te_win = segment_into_windows(X_te_dwt, y_te, fs_dwt, WINDOW_SIZE_SEC, WINDOW_STRIDE_SEC)

                    # 5. COVARIANZA ESPACIAL
                    # Output: (Windows, Bands, 6, 6)
                    C_tr = calculate_covariances(X_tr_win)
                    C_va = calculate_covariances(X_va_win) if len(X_va_win) > 0 else np.empty((0, 5, 6, 6))
                    C_te = calculate_covariances(X_te_win)

                    # 6. ESPACIO TANGENTE RIEMANNIANO (Band-wise)
                    F_tr_list, F_va_list, F_te_list = [], [], []
                    try:
                        for b in range(len(BAND_NAMES)):
                            # Media Riemanniana solo con datos de TRAIN de esa banda específica
                            ref_b = mean_riemann(C_tr[:, b, :, :])
                            
                            # Proyectar usando pyriemann (devuelve vector plano, triangular superior + diag, con ponderación sqrt(2))
                            F_tr_list.append(tangent_space(C_tr[:, b, :, :], ref_b))
                            F_te_list.append(tangent_space(C_te[:, b, :, :], ref_b))
                            if len(C_va) > 0: F_va_list.append(tangent_space(C_va[:, b, :, :], ref_b))
                        
                        # 7. CONCATENACIÓN DE BANDAS (5 bandas x 21 features = 105 dims)
                        F_tr = np.hstack(F_tr_list)
                        F_te = np.hstack(F_te_list)
                        F_va = np.hstack(F_va_list) if F_va_list else np.empty((0, F_tr.shape[1]))

                        # ENTRENAMIENTO L1
                        clf = LogisticRegression(**LOGREG_L1_KWARGS)
                        t0 = time.time()
                        clf.fit(F_tr, y_tr_win)
                        t1 = time.time()

                        # PREDICCIÓN (A nivel ventana)
                        y_pred_te = clf.predict(F_te)
                        acc_test = accuracy_score(y_te_win, y_pred_te)
                        
                        y_pred_va = clf.predict(F_va) if len(F_va) > 0 else []
                        acc_val = accuracy_score(y_va_win, y_pred_va) if len(F_va) > 0 else None

                        valid_labels = np.arange(n_classes_target)
                        cm_test = confusion_matrix(y_te_win, y_pred_te, labels=valid_labels)
                        report_test = classification_report(y_te_win, y_pred_te, labels=valid_labels, output_dict=True, zero_division=0)

                        # METADATOS Y EXPORTACIÓN
                        meta = {
                            "subject": subj_name, "subset": subset_name, "seed": seed_val, "fold": fold_idx,
                            "trials_orig": {"train": len(y_tr), "train_aug": len(y_tr_aug), "val": len(y_va), "test": len(y_te)},
                            "windows_qty": {"train": len(y_tr_win), "val": len(y_va_win), "test": len(y_te_win)},
                            "features_dim": F_tr.shape[1],
                            "test_accuracy": float(acc_test),
                            "val_accuracy": float(acc_val) if acc_val is not None else None,
                            "train_time_s": t1 - t0,
                            "status": "success"
                        }
                        
                        np.save(fold_out / "confusion_matrix.npy", cm_test)
                        save_json(fold_out / "classification_report.json", report_test)
                        save_json(fold_out / "metadata.json", meta)
                        print(f"    Fold {fold_idx}: TrainWins={len(y_tr_win)}, AccTest={acc_test:.4f}")

                    except Exception as e:
                        print(f"    [Error] Fold {fold_idx}: {str(e)}")
                        save_json(fold_out / "metadata.json", {"status": "error", "error": traceback.format_exc()})

    # Resumen final
    all_meta = []
    for f in EXPERIMENT_ROOT.rglob("metadata.json"):
        try: all_meta.append(json.loads(f.read_text(encoding="utf8")))
        except Exception: pass
    save_json(EXPERIMENT_ROOT / "summary_runs.json", {"n_runs_indexed": len(all_meta), "runs": all_meta})
    print(f"\nFinalizado. Resumen en: {EXPERIMENT_ROOT / 'summary_runs.json'}")