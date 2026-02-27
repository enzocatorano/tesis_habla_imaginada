#!/usr/bin/env python3
"""
Pipeline de augmentación para EEG (resample, segmentación, band-noise Butterworth, FT-surrogate)
Guarda por sujeto en data/processed_aug/SXX_EEG_augmented.npz

Requisitos: numpy, scipy
"""

import os
import numpy as np
from pathlib import Path
import json

# --- Configurable por el usuario (ajustá aquí) ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = str(PROJECT_ROOT / "data" / "original")
OUT_DIR = str(PROJECT_ROOT / "data" / "preproc_aug_segm_gnperband_fts")

SUBJECTS = [f"S{str(i).zfill(2)}" for i in range(1, 16)]
FILE_TEMPLATE = "{subj}_EEG.npz"
KEY_IN_NPZ = "data"

# Frecuencias
FS_ORIG = 1024
FS_TARGET = 128

# Segmentación (si DO_SEGMENTATION=False, se usará la señal completa del trial como una única ventana)
DO_SEGMENTATION = False
WIN_SEC = 1.5
HOP_SEC = 0.5

# Band-noise (gaussian noise banda-limitada por Butterworth)
DO_BAND_NOISE = True
BUTTER_ORDER = 4
F_ENERGY = 3  # fracción de energía objetivo
ENERGY_USE_WEAKEST_BAND = True
PER_CHANNEL_SCALE = True

# FT surrogate
DO_FT_SURROGATE = True
N_FTS = 3  # cantidad de surrogates por ventana/banda (además se guarda la versión sin FT con ft_flag=0)
FT_PHASE_MIN = 0.0
FT_PHASE_MAX = 2.0 * np.pi
# semilla base para reproducibilidad (None = aleatorio)
GLOBAL_SEED = 17

# ----------------------

def require_scipy():
    try:
        import scipy.signal as sps
        return sps
    except Exception as e:
        raise ImportError("Este script requiere scipy (scipy.signal). Instalalo con `pip install scipy`.") from e

# definiciones de banda
def band_defs():
    return {
        'delta': (0.5, 4.0),
        'theta': (4.0, 8.0),
        'alpha': (8.0, 12.0),
        'beta' : (12.0, 32.0),
        'gamma': (32.0, 63.5)
    }

# butterworth bandpass con sosfiltfilt robusto reduciendo orden si hace falta
def bandpass_sosfiltfilt(x, low, high, fs, order=4, axis=1):
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
    # energía por canal: suma de cuadrados a lo largo de samples
    return np.sum(x.astype(float)**2, axis=1)

# Resample usando resample_poly
def resample_subject_signals(signal_flat, orig_fs=1024, target_fs=128):
    sps = require_scipy()
    from scipy.signal import resample_poly
    n_trials = signal_flat.shape[0]
    if signal_flat.shape[1] % 6 != 0:
        raise RuntimeError("Formato inesperado: el número de columnas no es divisible por 6 (canales).")
    samples_orig = signal_flat.shape[1] // 6
    signal = signal_flat.reshape(n_trials, 6, samples_orig)
    new_samples = int(round(samples_orig * float(target_fs) / float(orig_fs)))
    resampled = np.zeros((n_trials, 6, new_samples), dtype=float)
    for i in range(n_trials):
        for ch in range(6):
            resampled[i, ch, :] = resample_poly(signal[i, ch, :], up=target_fs, down=orig_fs)
    return resampled

# Segmentación
def segment_trial_windows(trial_signal, fs=128, win_sec=1.5, hop_sec=0.5):
    n_channels, n_samples = trial_signal.shape
    win_samples = int(round(win_sec * fs))
    hop_samples = int(round(hop_sec * fs))
    if win_samples > n_samples:
        # si la ventana es mayor que la trial, cortamos o hacemos padding? Aquí devolvemos una única ventana con padding por ceros.
        pad = win_samples - n_samples
        w = np.pad(trial_signal, ((0,0),(0,pad)), mode='constant')
        return np.expand_dims(w, axis=0), [0]
    starts = list(range(0, n_samples - win_samples + 1, hop_samples))
    windows = np.stack([trial_signal[:, s:s+win_samples] for s in starts], axis=0) if len(starts)>0 else np.empty((0, n_channels, win_samples))
    return windows, starts

# Augmentación por bandas (gaussian noise banda-limitada)
def augment_window_bands(window, fs=128, factor=0.5, order=4, seed=None, per_channel_scale=True, energy_use_weakest=False):
    """
    window: (channels, samples)
    Devuelve dict band_name -> dict with 'augmented' and metadata
    """
    bands = band_defs()
    n_channels, n_samples = window.shape
    augmented = {}

    # calcular energías por banda de la señal original (para la opción weakest)
    band_clean_energies = {}
    for bname, (low, high) in bands.items():
        band_clean, _ = bandpass_sosfiltfilt(window, low, high, fs, order=order, axis=1)
        band_clean_energies[bname] = compute_energy(band_clean)  # array canales

    # si usamos weakest_band, calculamos la energía por canal mínima entre bandas
    if energy_use_weakest:
        # energy per channel: min across bands
        energies_stack = np.stack([band_clean_energies[b] for b in band_clean_energies.keys()], axis=0)  # (n_bands, n_channels)
        weakest_per_channel = np.min(energies_stack, axis=0)  # (n_channels,)
    else:
        weakest_per_channel = None

    for i, (bname, (low, high)) in enumerate(bands.items()):
        # 1) extraer banda limpia (se recalcula para asegurar orden usado)
        band_clean, used_order = bandpass_sosfiltfilt(window, low, high, fs, order=order, axis=1)
        band_energy = compute_energy(band_clean)  # per channel

        if energy_use_weakest:
            target_energy = weakest_per_channel * float(factor)
        else:
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
            global_scale = np.sqrt(total_target / total_noise) if total_noise>0 else 0.0
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

# FT surrogate: devuelve una lista de ventanas FT-perturbadas (N) y posibilita reproducibilidad
def ft_surrogates(window, n_aug=3, phase_min=0.0, phase_max=2*np.pi, seed=None):
    """
    window: (channels, samples) float
    Retorna list_of_augmented_windows (length n_aug), each (channels, samples)
    Nota: usa rfft/irfft para mantener realidad de la señal. La misma phi(f) para todos los canales, aleatorizada por surrogate.
    """
    rng = np.random.RandomState(seed)
    n_channels, n_samples = window.shape
    # rfft length
    X = np.fft.rfft(window, axis=1)  # shape (channels, n_freqs)
    n_freqs = X.shape[1]
    results = []
    for a in range(n_aug):
        phi = rng.uniform(phase_min, phase_max, size=(n_freqs,))
        phasor = np.exp(1j * phi)  # (n_freqs,)
        Xp = X * phasor[None, :]  # same phasor for all channels
        xp = np.fft.irfft(Xp, n=n_samples, axis=1)
        results.append(xp.astype(float))
    return results

# Procesado por sujeto (pipeline completo)
def process_subject(filepath, out_dir=OUT_DIR, fs_orig=FS_ORIG, fs_target=FS_TARGET,
                    do_seg=DO_SEGMENTATION, win_sec=WIN_SEC, hop_sec=HOP_SEC,
                    do_band_noise=DO_BAND_NOISE, butter_order=BUTTER_ORDER, f_energy=F_ENERGY,
                    energy_use_weakest=ENERGY_USE_WEAKEST_BAND, per_channel_scale=PER_CHANNEL_SCALE,
                    do_ft=DO_FT_SURROGATE, n_fts=N_FTS, ft_phase_min=FT_PHASE_MIN, ft_phase_max=FT_PHASE_MAX,
                    global_seed=GLOBAL_SEED):
    print(f"\nProcesando {filepath} ...")
    if not os.path.exists(filepath):
        print("  Archivo no encontrado:", filepath)
        return None
    data = np.load(filepath)
    # encontrar key con array principal (por defecto 'data')
    if KEY_IN_NPZ in data:
        eeg_data = data[KEY_IN_NPZ]
    else:
        found = False
        for k in data.files:
            if isinstance(data[k], np.ndarray) and data[k].ndim >= 2:
                eeg_data = data[k]
                found = True
                print(f"  Key '{KEY_IN_NPZ}' no encontrada. Usando key '{k}' del .npz")
                break
        if not found:
            raise RuntimeError("No se encontró un array válido en el archivo .npz")

    # separar labels y señales según ejemplo del usuario (últimas 3 columnas = labels)
    if eeg_data.shape[1] < 4:
        raise RuntimeError("El array cargado parece no contener señales + 3 etiquetas al final.")
    labels = eeg_data[:, -3:]  # (n_trials, 3)
    eeg_signals_flat = eeg_data[:, :-3]  # (n_trials, 6*samples_orig)

    # resample y reestructura (OBLIGATORIO)
    signals_res = resample_subject_signals(eeg_signals_flat, orig_fs=fs_orig, target_fs=fs_target)  # (n_trials, 6, new_samples)
    n_trials, n_ch, n_samples = signals_res.shape
    print(f"  Señales reestructuradas a (trials, ch, samples): {signals_res.shape} (fs={fs_target})")

    augmented_list = []
    augmented_labels = []  # (N, 5): [orig3, band_index (-1 si no aplica), ft_flag]
    band_names = list(band_defs().keys())

    rng_master = np.random.RandomState(global_seed)

    total_windows = 0
    total_aug = 0

    for t in range(n_trials):
        trial_sig = signals_res[t]  # (6, samples)
        seed_trial = None if global_seed is None else int(rng_master.randint(0, 2**31-1))
        # 1) segmentación (si corresponde) -> lista de ventanas
        if do_seg:
            windows, starts = segment_trial_windows(trial_sig, fs=fs_target, win_sec=win_sec, hop_sec=hop_sec)
        else:
            # ventana única: la trial completa (si hay diferencia en tamaño con win_sec, lo ignoramos y usamos full length)
            windows = np.expand_dims(trial_sig, axis=0)
            starts = [0]
        total_windows += windows.shape[0]

        for w_idx in range(windows.shape[0]):
            w = windows[w_idx]  # (ch, samples)
            base_seed_for_window = None if seed_trial is None else seed_trial + w_idx * 1000

            # 2) Band-noise (si corresponde) -> genera N_bands augmentaciones por ventana.
            band_augmented_items = []  # cada elemento = (aug_window, band_index)
            if do_band_noise:
                aug_dict = augment_window_bands(w, fs=fs_target, factor=f_energy, order=butter_order,
                                                seed=base_seed_for_window, per_channel_scale=per_channel_scale,
                                                energy_use_weakest=energy_use_weakest)
                # itera en orden de band_names para mantener band_index consistente
                for bname in band_names:
                    info = aug_dict[bname]
                    band_idx = info['band_index']
                    band_augmented_items.append((info['augmented'], int(band_idx)))
            else:
                # no se hacen augmentaciones por banda: conservar la ventana original con band_index = -1
                band_augmented_items.append((w.astype(float), -1))

            # 3) FT surrogates: para cada item en band_augmented_items aplicar FT (si corresponde).
            for (aug_w, band_idx) in band_augmented_items:
                if do_ft:
                    # incluir la versión sin FT (ft_flag=0)
                    lbl = np.concatenate([labels[t].astype(float), np.array([float(band_idx)], dtype=float), np.array([0.0], dtype=float)])
                    augmented_list.append(aug_w.astype(np.float32))
                    augmented_labels.append(lbl.astype(np.float32))
                    total_aug += 1
                    # generar N_FTS surrogates
                    ft_seed_base = None if base_seed_for_window is None else base_seed_for_window + band_idx + 777
                    surs = ft_surrogates(aug_w, n_aug=n_fts, phase_min=ft_phase_min, phase_max=ft_phase_max, seed=(ft_seed_base))
                    for surr in surs:
                        lbls = np.concatenate([labels[t].astype(float), np.array([float(band_idx)], dtype=float), np.array([1.0], dtype=float)])
                        augmented_list.append(surr.astype(np.float32))
                        augmented_labels.append(lbls.astype(np.float32))
                        total_aug += 1
                else:
                    # no FT -> solo guardamos la ventana tal cual con ft_flag=0
                    lbl = np.concatenate([labels[t].astype(float), np.array([float(band_idx)], dtype=float), np.array([0.0], dtype=float)])
                    augmented_list.append(aug_w.astype(np.float32))
                    augmented_labels.append(lbl.astype(np.float32))
                    total_aug += 1

    # preparar salida
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    if len(augmented_list) > 0:
        augmented_array = np.stack(augmented_list, axis=0)  # (N_aug, ch, win_samples)
        labels_array = np.stack(augmented_labels, axis=0)  # (N_aug, 5)
    else:
        # fallback shapes
        win_samples = int(round((WIN_SEC if DO_SEGMENTATION else (n_samples / FS_TARGET)) * FS_TARGET))
        augmented_array = np.empty((0, n_ch, win_samples), dtype=np.float32)
        labels_array = np.empty((0, 5), dtype=np.float32)

    subj_base = os.path.basename(filepath).replace(".npz", "")
    out_path = os.path.join(out_dir, f"{subj_base}_augmented.npz")
    np.savez_compressed(out_path,
                        data=augmented_array.astype(np.float32),
                        labels=labels_array.astype(np.float32),
                        band_names=np.array(band_names),
                        params=np.array([['FS_TARGET', fs_target],
                                         ['DO_SEGMENTATION', int(do_seg)],
                                         ['DO_BAND_NOISE', int(do_band_noise)],
                                         ['DO_FT_SURROGATE', int(do_ft)]]))
    print(f"  Guardado {augmented_array.shape[0]} muestras aumentadas en {out_path}")
    print(f"  Ventanas generadas por trial (total): {total_windows}, augmentaciones totales: {total_aug}")
    n_orig_trials = n_trials
    n_aug_trials = augmented_array.shape[0]
    return out_path, n_orig_trials, n_aug_trials

# Ejecutar para todos los sujetos
def run_all(subjects=SUBJECTS):
    _ = require_scipy()  # validar dependencia temprano
    results = {}

    # --- parámetros globales del pipeline (se guardan en el JSON) ---
    global_params = {
        "fs_orig": FS_ORIG,
        "fs_target": FS_TARGET,

        "do_segmentation": DO_SEGMENTATION,
        "win_sec": WIN_SEC,
        "hop_sec": HOP_SEC,

        "do_band_noise": DO_BAND_NOISE,
        "butter_order": BUTTER_ORDER,
        "f_energy": F_ENERGY,
        "energy_use_weakest_band": ENERGY_USE_WEAKEST_BAND,
        "per_channel_scale": PER_CHANNEL_SCALE,

        "do_ft_surrogate": DO_FT_SURROGATE,
        "n_fts": N_FTS,
        "ft_phase_min": FT_PHASE_MIN,
        "ft_phase_max": FT_PHASE_MAX,

        "global_seed": GLOBAL_SEED,
    }

    # estructura del JSON
    summary = {
        "global_params": global_params,
        "subjects": {}
    }

    for subj in subjects:
        filename = FILE_TEMPLATE.format(subj=subj)
        filepath = os.path.join(DATA_DIR, filename)
        out, n_orig, n_aug = process_subject(
            filepath,
            out_dir=OUT_DIR,
            fs_orig=FS_ORIG,
            fs_target=FS_TARGET,
            do_seg=DO_SEGMENTATION,
            win_sec=WIN_SEC,
            hop_sec=HOP_SEC,
            do_band_noise=DO_BAND_NOISE,
            butter_order=BUTTER_ORDER,
            f_energy=F_ENERGY,
            energy_use_weakest=ENERGY_USE_WEAKEST_BAND,
            per_channel_scale=PER_CHANNEL_SCALE,
            do_ft=DO_FT_SURROGATE,
            n_fts=N_FTS,
            ft_phase_min=FT_PHASE_MIN,
            ft_phase_max=FT_PHASE_MAX,
            global_seed=GLOBAL_SEED
        )

        results[subj] = out
        summary["subjects"][subj] = {
            "original_trials": int(n_orig),
            "augmented_trials": int(n_aug)
        }
        print(f"  Resultados para {subj} listos.")

    # guardar JSON con resumen global en el mismo directorio que los .npz aumentados
    os.makedirs(OUT_DIR, exist_ok=True)
    summary_path = os.path.join(OUT_DIR, "augmentation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nResumen de augmentación guardado en {summary_path}")
    return results

if __name__ == "__main__":
    print("Iniciando procesamiento de sujetos...")
    res = run_all()
    print("Hecho. Archivos guardados:", res)
