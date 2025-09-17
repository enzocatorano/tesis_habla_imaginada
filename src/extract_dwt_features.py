#!/usr/bin/env python3
"""
src/extract_dwt_features.py

Extrae features basados en DWT (PyWavelets) desde los .npz en data/processed.
Para cada trial y canal:
 - resample a target_fs (resample_poly)
 - descomposición DWT de L niveles (wavedec)
 - por cada vector de coeficientes: mean, mean_abs, var, std, skew, kurtosis

Guarda:
 - per-subject: <SUBJ>_features.npy  shape (n_trials, n_channels, L+1, n_stats)
 - per-subject flat:    <SUBJ>_features_flat.npy shape (n_trials, n_features)
 - per-subject labels:  <SUBJ>_labels.npy  (n_trials, 3)
 - per-subject task:    <SUBJ>_task.npy    (0=vowel,1=command)
 - global: X.npy, y.npy, task.npy, meta.json

Uso:
 python src/extract_dwt_features.py --data-root ../data/processed --out-dir features/preproc_dwt_L5_db4 --L 5 --wavelet db4 --target-fs 128 --n-channels 6
"""
import argparse
from pathlib import Path
import numpy as np
import json
import pywt
from scipy import stats
from scipy.signal import resample_poly
from glob import glob
from fractions import Fraction
import math
import sys

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def resample_signal(sig, orig_fs, target_fs):
    """Resample 1D signal using resample_poly with integer up/down."""
    if orig_fs == target_fs:
        return sig
    # compute rational approx for target_fs/orig_fs
    frac = Fraction(target_fs, orig_fs).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator
    return resample_poly(sig, up, down)

def compute_stats(vec):
    """Return [mean, mean_abs, var, std, skewness, kurtosis] for 1D numpy array"""
    if vec.size == 0:
        return [0.0]*6
    a_mean = float(np.mean(vec))
    a_mean_abs = float(np.mean(np.abs(vec)))
    a_var = float(np.var(vec, ddof=0))
    a_std = float(np.std(vec, ddof=0))
    a_skew = float(stats.skew(vec, bias=False)) if vec.size > 2 else 0.0
    # Using Fisher=False to return Pearson kurtosis; change if prefer excess kurtosis.
    a_kurt = float(stats.kurtosis(vec, fisher=False, bias=False)) if vec.size > 3 else 0.0
    return [a_mean, a_mean_abs, a_var, a_std, a_skew, a_kurt]

def label_task_from_stimulus(stim):
    """Stimulus codes 1-5 vowels (A,E,I,O,U), 6-11 commands (directions)"""
    if 1 <= stim <= 5:
        return 0  # vowel
    if 6 <= stim <= 11:
        return 1  # command
    return -1  # unknown

def process_file(npz_path, out_subj_dir, cfg, verbose=False):
    """Process a single subject .npz file and save features"""
    if verbose:
        print("Processing:", npz_path)
    arr = np.load(npz_path)
    if 'data' not in arr:
        raise ValueError(f"{npz_path} does not contain 'data' key")
    data = arr['data']  # shape (n_trials, total_samples + 3 labels at end)
    meta_p = Path(str(npz_path).replace('.npz', '.meta.json'))
    # read fs from meta if exists
    orig_fs = cfg['orig_fs_default']
    if meta_p.exists():
        try:
            m = json.load(open(meta_p, 'r', encoding='utf8'))
            if 'fs' in m:
                orig_fs = int(m['fs'])
        except Exception:
            pass

    n_trials, total_cols = data.shape
    # labels: last 3 entries
    labels = data[:, -3:].astype(int)  # (n_trials,3)
    X_raw = data[:, :-3]  # (n_trials, total_samples)
    n_channels = cfg['n_channels']
    if total_cols - 3 <= 0:
        raise ValueError(f"No signal columns found in {npz_path}")
    samples_per_channel = (total_cols - 3) // n_channels
    if (total_cols - 3) % n_channels != 0:
        print(f"Warning: total samples ({total_cols-3}) not divisible by n_channels ({n_channels}) in {npz_path}", file=sys.stderr)

    L = cfg['L']
    wavelet = cfg['wavelet']
    target_fs = cfg['target_fs']

    # prepare output containers
    n_stats = 6
    decomps = L + 1
    subj_feat = np.zeros((n_trials, n_channels, decomps, n_stats), dtype=np.float32)
    subj_flat = []  # will become (n_trials, n_features)
    subj_task = np.zeros((n_trials,), dtype=np.int8)
    for t in range(n_trials):
        # reshape trial into channels: (n_channels, samples_per_channel)
        trial = X_raw[t, :n_channels * samples_per_channel]
        trial_ch = trial.reshape(n_channels, samples_per_channel)
        for ch in range(n_channels):
            sig = trial_ch[ch].astype(np.float64)
            # resample to target_fs
            sig_rs = resample_signal(sig, orig_fs, target_fs)
            # DWT decomposition (returns list length L+1)
            coeffs = pywt.wavedec(sig_rs, wavelet, level=L)
            # coeffs: [cA_L, cD_L, cD_{L-1}, ..., cD_1] -> length L+1
            # compute stats for each coeff vector
            ch_feats = []
            for c in coeffs:
                st = compute_stats(np.asarray(c, dtype=np.float64))
                ch_feats.append(st)
            ch_feats = np.array(ch_feats, dtype=np.float32)  # shape (L+1, n_stats)
            subj_feat[t, ch, :, :] = ch_feats
        # flatten per trial
        flat = subj_feat[t].reshape(-1)  # channel-major then decomposition then stats
        subj_flat.append(flat)
        # task label from stimulus (2nd column of labels? need to check: labels are [modalidad, estimulo, artefactos])
        # as user specified, labels order is: Modalidad, Estímulo, Artefactos
        stimulus = int(labels[t, 1])
        subj_task[t] = label_task_from_stimulus(stimulus)

    subj_flat = np.vstack(subj_flat)  # (n_trials, n_features)
    # save outputs
    subj_name = Path(npz_path).stem  # e.g., S01_EEG -> S01_EEG
    np.save(out_subj_dir / f"{subj_name}_features.npy", subj_feat)
    np.save(out_subj_dir / f"{subj_name}_features_flat.npy", subj_flat)
    np.save(out_subj_dir / f"{subj_name}_labels.npy", labels)
    np.save(out_subj_dir / f"{subj_name}_task.npy", subj_task)
    return subj_feat.shape, subj_flat.shape, labels.shape

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="../data/processed", help="Ruta a los .npz canonicos")
    parser.add_argument("--out-dir", type=str, default="features/preproc_dwt_L5_db4", help="Salida de features")
    parser.add_argument("--L", type=int, default=5, help="Niveles DWT")
    parser.add_argument("--wavelet", type=str, default="db4", help="Wavelet name")
    parser.add_argument("--target-fs", type=int, default=128, help="Frecuencia objetivo [Hz]")
    parser.add_argument("--n-channels", type=int, default=6, help="Número de canales por trial")
    parser.add_argument("--orig-fs-default", type=int, default=1024, help="fs por defecto si no está en meta.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    per_subject_dir = out_dir / "per_subject"
    ensure_dir(per_subject_dir)
    ensure_dir(out_dir / "meta")

    cfg = {
        "L": args.L,
        "wavelet": args.wavelet,
        "target_fs": args.target_fs,
        "n_channels": args.n_channels,
        "orig_fs_default": args.orig_fs_default
    }

    # find files *_EEG.npz
    files = sorted(glob(str(data_root / "*_EEG.npz")))
    if args.verbose:
        print("Found", len(files), "EEG .npz files in", data_root)
    all_flat = []
    all_labels = []
    all_tasks = []
    subj_summaries = {}
    for f in files:
        subj_dir = per_subject_dir
        feat_shape, flat_shape, labels_shape = process_file(f, subj_dir, cfg, verbose=args.verbose)
        if args.verbose:
            print("Saved subject features:", f)
        # load saved flat and labels to aggregate
        stem = Path(f).stem
        flat = np.load(subj_dir / f"{stem}_features_flat.npy")
        labels = np.load(subj_dir / f"{stem}_labels.npy")
        task = np.load(subj_dir / f"{stem}_task.npy")
        all_flat.append(flat)
        all_labels.append(labels)
        all_tasks.append(task.reshape(-1,1))
        subj_summaries[stem] = {
            "n_trials": labels.shape[0],
            "features_shape": flat.shape
        }

    if len(all_flat) == 0:
        print("No files processed. Abort.")
        return

    X = np.vstack(all_flat)
    y = np.vstack(all_labels)
    task = np.vstack(all_tasks).ravel()
    # Save global
    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)
    np.save(out_dir / "task.npy", task)
    meta = {
        "preproc_id": out_dir.name,
        "L": args.L,
        "wavelet": args.wavelet,
        "target_fs": args.target_fs,
        "n_channels": args.n_channels,
        "n_subjects": len(files),
        "n_total_windows": X.shape[0],
        "feature_shape_per_trial": list(flat_shape) if 'flat_shape' in locals() else None,
        "per_subject_summary": subj_summaries
    }
    json.dump(meta, open(out_dir / "meta" / "meta.json", "w"), indent=2)
    if args.verbose:
        print("Saved global X.npy, y.npy, task.npy and meta.json in", out_dir)

if __name__ == "__main__":
    main()
