#!/usr/bin/env python3
# extract_results_and_temporal_window.py
"""
Extractor y visualizador unificado:
 - visualizaciones por sujeto y global (learning curves, confusion matrices, metrics boxplots)
 - análisis por banda / FT Surrogate (cuando sea posible)
 - análisis por ventana temporal (cuando sea posible; intenta reconstruir usando metadata)
 
Funciona con la estructura de salida del lanzador:
 experiments/<EXPERIMENT_NAME>_<timestamp>_<host>/<Sxx>/<subset>/seed_<s>/fold_<f>/
 y con los ficheros generados: metadata.json, train_metrics.json, test_preds.npz,
 confusion_matrix.npy, classification_report.json, augmentation_metadata.json (si existe).

El script intenta usar tanto archivos 'data/preprocessed/S##_preprocessed.npz' (si existen)
como la información de 'augmentation_metadata.json' para recuperar etiquetas de banda/fts/ventana.
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import math
import warnings

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

# ---------------- CONFIG ----------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
# prefijo del experimento que querés procesar (busca el más reciente que empiece por este prefijo)
EXPERIMENT_NAME_PREFIX = "EEGNet_full_baseline_onlineAug"
# Ruta donde están los preprocessed (no-augmented) .npz (para poder leer etiquetas originales si hace falta)
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"
OUTPUT_SUBDIR = "visualization_results_and_temporal"

BAND_LABELS = ["delta", "theta", "alpha", "beta", "gamma"]
VOCAL_CLASS_NAMES = ['A', 'E', 'I', 'O', 'U']
COMANDO_CLASS_NAMES = ['Arriba', 'Abajo', 'Izquierda', 'Derecha', 'Adelante', 'Atras']
SUBSETS = ["vocales", "comandos"]
METRICS_NAMES = ["Precision", "Recall", "F1-Score"]

# Parámetros por defecto para inferencia de ventanas si no hay metadata completa
DEFAULT_N_WINDOWS = 6
DEFAULT_AUG_PER_WINDOW = 20  # hipótesis: cantidad de augmentaciones por ventana (si no está en metadata)

# ----------------------------------------

def find_latest_experiment(root: Path, prefix: str) -> Path:
    candidates = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not candidates:
        raise FileNotFoundError(f"No experiment folders starting with '{prefix}' found under {root}")
    return sorted(candidates, key=lambda p: p.name)[-1]

def load_json_safe(path: Path):
    if not path or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf8"))
    except Exception:
        return None

def save_json_safe(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf8") as fh:
        json.dump(obj, fh, indent=2, default=lambda o: (o.tolist() if isinstance(o, np.ndarray) else str(o)))

def pad_and_aggregate_series(list_of_lists: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
    if not list_of_lists:
        return np.array([]), np.array([])
    lengths = [len(l) for l in list_of_lists]
    max_len = max(lengths)
    arr = np.full((len(list_of_lists), max_len), np.nan, dtype=float)
    for i, l in enumerate(list_of_lists):
        if len(l) > 0:
            arr[i, :len(l)] = np.array(l, dtype=float)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    return mean, std

# ---------------- Data helpers ----------------

def load_subject_preprocessed(subject_name: str, preproc_dir: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Intenta cargar el fichero preprocesado del sujeto (no-augmentado).
    Soporta archivos que contengan 'data'/'labels' o 'x'/'y' como claves.
    Devuelve (X, Y) o (None, None) si no existe.
    """
    candidates = list(preproc_dir.glob(f"{subject_name}*.npz"))
    if not candidates:
        # also try base name without suffix: S01 -> S01_preprocessed
        candidates = list(preproc_dir.glob(f"{subject_name.split('_')[0]}*.npz"))
    if not candidates:
        return None, None
    path = candidates[0]
    try:
        d = np.load(path, allow_pickle=True)
        if 'data' in d and 'labels' in d:
            return d['data'], d['labels']
        if 'x' in d and 'y' in d:
            return d['x'], d['y']
        # try common alternatives
        if 'X' in d and 'Y' in d:
            return d['X'], d['Y']
        # as fallback, return first two arrays in file
        keys = list(d.keys())
        if len(keys) >= 2:
            return d[keys[0]], d[keys[1]]
    except Exception as e:
        warnings.warn(f"Could not load preprocessed file {path}: {e}")
    return None, None

# ---------------- Fold data collection (robust) ----------------

def collect_fold_data(subject_dir: Path, subset: str, preprocessed_X: Optional[np.ndarray], preprocessed_Y: Optional[np.ndarray]):
    """
    Recolecta datos de todos los folds/seeds para un sujeto y subset.
    Estrategias para band/fts/window:
      - Si preprocessed_Y contiene columnas con window/band/fts (ya augmentado offline), se usa directamente.
      - Si existe augmentation_metadata.json por fold, el script intenta usarlo para inferir ventanas (parcial).
      - Si no hay info suficiente, band/fts/window analyses se omiten y se registra en summary.
    """
    subset_dir = subject_dir / subset
    if not subset_dir.exists():
        return None

    # Try to extract per-sample info from preprocessed Y if present:
    preproc_has_window_band_fts = False
    if preprocessed_Y is not None and preprocessed_Y.ndim == 2 and preprocessed_Y.shape[1] >= 6:
        # columns: [modalidad, estímulo, artefacto, ventana, banda_afectada, fts_usado]
        preproc_has_window_band_fts = True

    metrics_list = []
    cm_list = []
    band_acc_list = []
    fts_acc_list = []
    train_loss_list = []
    val_loss_list = []
    val_acc_list = []
    # For temporal window analysis aggregated raw predictions
    window_predictions_raw = []  # list of dicts per fold: {"windows": array, "y_true":, "y_pred":}

    seed_dirs = sorted([p for p in subset_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")])
    if not seed_dirs:
        seed_dirs = [subset_dir]  # fallback

    for seed_dir in seed_dirs:
        fold_dirs = sorted([p for p in seed_dir.iterdir() if p.is_dir() and p.name.startswith("fold_")])
        for fold_dir in fold_dirs:
            meta = load_json_safe(fold_dir / "metadata.json")
            if meta is None or meta.get("status") != "success":
                continue

            # load classification report
            cr = load_json_safe(fold_dir / "classification_report.json")
            if cr:
                macro = cr.get("macro avg") or cr.get("macro_avg") or cr.get("macro-average")
                if macro:
                    metrics_list.append({
                        "precision": float(macro.get("precision", np.nan)),
                        "recall": float(macro.get("recall", np.nan)),
                        "f1": float(macro.get("f1-score", macro.get("f1_score", np.nan)))
                    })

            # confusion matrices
            cm_path = fold_dir / "confusion_matrix.npy"
            if cm_path.exists():
                try:
                    cm = np.load(cm_path)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        row_sums = cm.sum(axis=1, keepdims=True)
                        cm_pct = np.divide(cm, row_sums, where=(row_sums != 0)) * 100.0
                        cm_pct = np.nan_to_num(cm_pct, nan=0.0)
                    cm_list.append(cm_pct)
                except Exception as e:
                    warnings.warn(f"Error loading confusion matrix {cm_path}: {e}")

            # load predictions
            preds_path = fold_dir / "test_preds.npz"
            if not preds_path.exists():
                continue
            try:
                d = np.load(preds_path, allow_pickle=True)
                y_true = np.array(d["y_true"])
                y_pred = np.array(d["y_pred"])
            except Exception as e:
                warnings.warn(f"Could not load preds {preds_path}: {e}")
                continue

            # Attempt strategy 1: if preprocessed_Y contains per-window labels (offline augmented dataset)
            test_idx_global = meta.get("test_idx_global")
            if preproc_has_window_band_fts and test_idx_global is not None:
                try:
                    idxs = np.array(test_idx_global, dtype=int)
                    # preprocessed_Y is per-augmented-sample in offline scenario, so idxs index into preproc arrays
                    bands_test = preprocessed_Y[idxs, 4].astype(int)  # assuming band at col 4 (0-based)
                    fts_test = preprocessed_Y[idxs, 5].astype(int)
                    windows_test = preprocessed_Y[idxs, 3].astype(int)
                    # compute band accuracies
                    band_accs = []
                    for b in range(len(BAND_LABELS)):
                        mask = (bands_test == b)
                        if mask.sum() == 0:
                            band_accs.append(np.nan)
                        else:
                            band_accs.append(float(np.mean(y_true[mask] == y_pred[mask])))
                    band_acc_list.append(np.array(band_accs))
                    # fts acc
                    mask_with = (fts_test == 1)
                    mask_without = (fts_test == 0)
                    acc_with = float(np.mean(y_true[mask_with] == y_pred[mask_with])) if mask_with.sum() > 0 else np.nan
                    acc_without = float(np.mean(y_true[mask_without] == y_pred[mask_without])) if mask_without.sum() > 0 else np.nan
                    fts_acc_list.append({"with_fts": acc_with, "without_fts": acc_without})
                    # store windows raw
                    window_predictions_raw.append({"windows": windows_test, "y_true": y_true, "y_pred": y_pred, "fold_path": str(fold_dir)})
                    continue  # done for this fold
                except Exception as e:
                    warnings.warn(f"Strategy1 (preproc per-window) failed for {fold_dir}: {e}")

            # Attempt strategy 2: try to use augmentation_metadata.json to reconstruct windows for the ORIGINAL segmented block
            aug_meta = load_json_safe(fold_dir / "augmentation_metadata.json")
            if aug_meta is not None:
                # We can try to infer windows for the first block (original segmented windows).
                # Augmentar built X_combined as: [X_seg_originals, X_band_augmented (appended), X_fts_augmented (appended)]
                # So the first block length = n_original_windows = n_test_trials * n_windows_per_trial
                try:
                    # retrieve n_windows_per_trial and augmentation_factors if present
                    window_params = aug_meta.get("window_params", {})
                    n_windows_per_trial = int(window_params.get("n_windows_per_trial", DEFAULT_N_WINDOWS))
                    # original_test_indices should be available in aug_meta.indices
                    indices_info = aug_meta.get("indices", {})
                    original_test_indices = indices_info.get("test_original_indices") or meta.get("test_idx_global")
                    if original_test_indices is None:
                        raise ValueError("No original_test_indices in augmentation metadata nor in fold metadata.")
                    n_test_trials = len(original_test_indices)
                    n_original_windows = n_test_trials * n_windows_per_trial

                    # If predictions length >= n_original_windows, we can analyze the first block
                    if len(y_true) >= n_original_windows:
                        # Build windows array for first block: for trial i in 0..n_test_trials-1, windows 0..n_windows_per_trial-1
                        windows_first_block = np.repeat(np.arange(n_windows_per_trial, dtype=int), n_test_trials).reshape(n_windows_per_trial, n_test_trials).T.flatten()
                        # But ordering produced by segment_sliding_windows in Augmentar is:
                        # for trial_idx in range(n_trials):
                        #   for window_idx in range(n_windows_per_trial): append window
                        # so windows array should be: [0,1,...,n_win-1, 0,1,...] repeated per trial
                        windows_order = np.tile(np.arange(n_windows_per_trial, dtype=int), n_test_trials)
                        windows_first_block = windows_order
                        # compute accuracy per window on the first block
                        y_true_block = y_true[:n_original_windows]
                        y_pred_block = y_pred[:n_original_windows]
                        window_accs = []
                        for w in range(n_windows_per_trial):
                            mask = (windows_first_block == w)
                            if mask.sum() == 0:
                                window_accs.append(np.nan)
                            else:
                                window_accs.append(float(np.mean(y_true_block[mask] == y_pred_block[mask])))
                        window_predictions_raw.append({"windows": windows_first_block, "y_true": y_true_block, "y_pred": y_pred_block, "fold_path": str(fold_dir)})
                        # For band/fts: if aug_meta contains band/fts labels for original segmented windows we could use them.
                        # Check if aug_meta stores 'original_bands' or similar (not guaranteed)
                        bands_info = None
                        fts_info = None
                        # Try to find arrays in augmentation_metadata (some users store detailed arrays under different keys)
                        for k in ["band_labels", "bands_test", "bands", "band_noise"]:
                            if k in aug_meta:
                                bands_info = aug_meta[k]
                                break
                        for k in ["fts_labels", "fts_test", "fts"]:
                            if k in aug_meta:
                                fts_info = aug_meta[k]
                                break
                        # If there's no explicit per-sample band info, we cannot compute band accuracies for appended augmented samples reliably.
                        # Append NaN placeholder so downstream knows we couldn't compute full band/fts accuracies.
                        band_acc_list.append(np.full(len(BAND_LABELS), np.nan))
                        fts_acc_list.append({"with_fts": np.nan, "without_fts": np.nan})
                        continue
                    else:
                        # Not enough predictions to cover original segmented block: fallback
                        band_acc_list.append(np.full(len(BAND_LABELS), np.nan))
                        fts_acc_list.append({"with_fts": np.nan, "without_fts": np.nan})
                        continue
                except Exception as e:
                    warnings.warn(f"Strategy2 (augmentation_metadata reconstruction) failed for {fold_dir}: {e}")
                    band_acc_list.append(np.full(len(BAND_LABELS), np.nan))
                    fts_acc_list.append({"with_fts": np.nan, "without_fts": np.nan})
                    continue

            # If we reached here, we couldn't compute band/fts info for this fold
            band_acc_list.append(np.full(len(BAND_LABELS), np.nan))
            fts_acc_list.append({"with_fts": np.nan, "without_fts": np.nan})

            # Learning curves
            train_metrics = load_json_safe(fold_dir / "train_metrics.json")
            if train_metrics:
                tl = train_metrics.get("train_losses")
                vl = train_metrics.get("val_losses")
                va = train_metrics.get("val_accs")
                if tl:
                    train_loss_list.append(tl)
                if vl:
                    val_loss_list.append(vl)
                if va:
                    val_acc_list.append(va)

    return {
        "metrics": metrics_list,
        "confusion_matrices": cm_list,
        "band_accuracies": band_acc_list,
        "ftsurrogate_accuracies": fts_acc_list,
        "train_losses": train_loss_list,
        "val_losses": val_loss_list,
        "val_accs": val_acc_list,
        "window_predictions_raw": window_predictions_raw
    }

# ---------------- PLOTTING FUNCTIONS ----------------

def plot_learning_curves(data_vocales, data_comandos, output_path: Path, subject_name: str):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, data, subset_name in zip(axes, [data_vocales, data_comandos], ["Vocales", "Comandos"]):
        if data is None:
            ax.text(0.5, 0.5, f"No data for {subset_name}", ha='center', va='center')
            ax.axis('off')
            continue
        tl_mean, tl_std = pad_and_aggregate_series(data["train_losses"])
        vl_mean, vl_std = pad_and_aggregate_series(data["val_losses"])
        va_mean, va_std = pad_and_aggregate_series(data["val_accs"])
        if tl_mean.size == 0 and vl_mean.size == 0 and va_mean.size == 0:
            ax.text(0.5, 0.5, f"No learning curves for {subset_name}", ha='center', va='center')
            ax.axis('off')
            continue
        epochs = np.arange(1, max(tl_mean.size, vl_mean.size, va_mean.size) + 1)
        color_train = "tab:blue"
        color_val = "tab:orange"
        if tl_mean.size > 0:
            ax.plot(epochs[:tl_mean.size], tl_mean, color=color_train, label="Train Loss", linewidth=2)
            ax.fill_between(epochs[:tl_mean.size], tl_mean - tl_std, tl_mean + tl_std, alpha=0.2, color=color_train)
        if vl_mean.size > 0:
            ax.plot(epochs[:vl_mean.size], vl_mean, color=color_val, label="Val Loss", linewidth=2)
            ax.fill_between(epochs[:vl_mean.size], vl_mean - vl_std, vl_mean + vl_std, alpha=0.2, color=color_val)
        ax.set_ylabel("Loss", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax2 = ax.twinx()
        color_acc = "tab:green"
        if va_mean.size > 0:
            ax2.plot(epochs[:va_mean.size], va_mean, color=color_acc, linestyle='--', label="Val Accuracy", linewidth=2)
            ax2.fill_between(epochs[:va_mean.size], va_mean - va_std, va_mean + va_std, alpha=0.15, color=color_acc)
        ax2.set_ylabel("Val Accuracy", fontsize=11)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
        ax.set_title(f"Learning Curves - {subset_name}", fontsize=12, fontweight='bold')
    axes[-1].set_xlabel("Epoch", fontsize=11)
    fig.suptitle(f"Learning Curves - {subject_name}", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_confusion_matrices(data_vocales, data_comandos, output_path: Path, subject_name: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, data, subset_name, class_names in zip(axes, [data_vocales, data_comandos], ["Vocales", "Comandos"], [VOCAL_CLASS_NAMES, COMANDO_CLASS_NAMES]):
        if data is None or not data["confusion_matrices"]:
            ax.text(0.5, 0.5, f"No data for {subset_name}", ha='center', va='center'); ax.axis('off'); continue
        cm_stack = np.stack(data["confusion_matrices"], axis=0)
        cm_mean = np.nanmean(cm_stack, axis=0)
        cm_std = np.nanstd(cm_stack, axis=0)
        n = cm_mean.shape[0]
        annot = np.empty((n, n), dtype=object)
        for r in range(n):
            for c in range(n):
                annot[r, c] = f"{cm_mean[r, c]:.1f}\n±{cm_std[r, c]:.1f}"
        sns.heatmap(cm_mean, annot=annot, fmt="", cmap="cividis", ax=ax,
                   cbar_kws={'label': '% (normalizado por fila)'}, linewidths=0.5, linecolor='gray', annot_kws={"size": 9})
        ax.set_title(f"{subset_name} (media ± std %)", fontsize=12, fontweight='bold')
        ax.set_xlabel("Predicha"); ax.set_ylabel("Verdadera")
        ax.set_xticklabels(class_names, rotation=45, ha='right'); ax.set_yticklabels(class_names, rotation=0)
    fig.suptitle(f"Matrices de confusión - {subject_name}", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_metrics_boxplots(data_vocales, data_comandos, output_path: Path, subject_name: str):
    fig, ax = plt.subplots(figsize=(12, 6))
    all_data = []; labels = []; positions = []; colors = []
    color_map = {"Precision": "#1f77b4", "Recall": "#ff7f0e", "F1-Score": "#2ca02c"}
    pos = 0
    for subset_name, data in [("Vocales", data_vocales), ("Comandos", data_comandos)]:
        if data is None or not data["metrics"]:
            pos += 4; continue
        metrics_arr = np.array([[m["precision"], m["recall"], m["f1"]] for m in data["metrics"]])
        for i, metric_name in enumerate(METRICS_NAMES):
            all_data.append(metrics_arr[:, i])
            labels.append(f"{subset_name}\n{metric_name}")
            positions.append(pos)
            colors.append(color_map[metric_name])
            pos += 1
        pos += 1
    if not all_data:
        ax.text(0.5, 0.5, "No data available", ha='center', va='center'); ax.axis('off')
    else:
        bp = ax.boxplot(all_data, positions=positions, widths=0.6, patch_artist=True, showmeans=True, meanline=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        ax.set_xticks(positions); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Score"); data_concat = np.concatenate(all_data); ax.set_ylim(0.9*data_concat.min(), 1.1*data_concat.max())
        ax.grid(True, alpha=0.3); ax.set_title(f"Distribución de métricas - {subject_name}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_band_accuracy_boxplots(data_vocales, data_comandos, output_path: Path, subject_name: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    band_freq_labels = [f"{lbl}\n" for lbl in BAND_LABELS]
    for ax, data, subset_name, n_classes in zip(axes, [data_vocales, data_comandos], ["Vocales", "Comandos"], [5,6]):
        if data is None or not data["band_accuracies"]:
            ax.text(0.5, 0.5, f"No data for {subset_name}", ha='center', va='center'); ax.axis('off'); continue
        band_arr = np.vstack(data["band_accuracies"])
        # If band_arr contains all NaNs, skip
        if np.all(np.isnan(band_arr)):
            ax.text(0.5,0.5,f"No band info for {subset_name}", ha='center', va='center'); ax.axis('off'); continue
        bp = ax.boxplot([band_arr[:, i] for i in range(len(BAND_LABELS))], labels=band_freq_labels, patch_artist=True, showmeans=True, meanline=True)
        chance_level = 1.0 / n_classes
        data_min = np.nanmin(band_arr); data_max = np.nanmax(band_arr)
        y_min = max(0, min(chance_level * 0.9, data_min * 0.9)); y_max = min(1, max(chance_level * 0.9, data_max * 1.1))
        ax.set_ylim(y_min, y_max); ax.axhline(y=chance_level, color='red', linestyle=':', linewidth=1.5, label=f'Chance ({chance_level:.3f})')
        ax.set_title(f"{subset_name}"); ax.set_ylabel("Accuracy"); ax.legend(loc='best', fontsize=8)
    fig.suptitle(f"Accuracy por banda - {subject_name}", fontsize=14)
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_ftsurrogate_accuracy_boxplots(data_vocales, data_comandos, output_path: Path, subject_name: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, subset_name, n_classes in zip(axes, [data_vocales, data_comandos], ["Vocales", "Comandos"], [5,6]):
        if data is None or not data["ftsurrogate_accuracies"]:
            ax.text(0.5, 0.5, f"No data for {subset_name}", ha='center', va='center'); ax.axis('off'); continue
        with_fts = [d["with_fts"] for d in data["ftsurrogate_accuracies"] if not math.isnan(d["with_fts"])]
        without_fts = [d["without_fts"] for d in data["ftsurrogate_accuracies"] if not math.isnan(d["without_fts"])]
        if not with_fts and not without_fts:
            ax.text(0.5,0.5,f"No FTS data for {subset_name}", ha='center', va='center'); ax.axis('off'); continue
        data_to_plot=[]; labels=[]
        if without_fts: data_to_plot.append(without_fts); labels.append("Without FTS")
        if with_fts: data_to_plot.append(with_fts); labels.append("With FTS")
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, showmeans=True, meanline=True)
        chance_level = 1.0 / n_classes
        all_data = without_fts + with_fts
        y_min = max(0, min(chance_level * 0.9, min(all_data)) * 0.9)
        y_max = min(1, max(chance_level * 0.9, max(all_data)) * 1.1)
        ax.set_ylim(y_min, y_max); ax.axhline(y=chance_level, color='red', linestyle=':', linewidth=1.5)
        ax.set_title(f"{subset_name}"); ax.set_ylabel("Accuracy")
    fig.suptitle(f"Accuracy por FT Surrogate - {subject_name}", fontsize=14)
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

# ---------------- Temporal window analysis ----------------

def infer_temporal_window_from_index_by_heuristic(global_idx: int, augmentation_factor_per_window:int = DEFAULT_AUG_PER_WINDOW, n_windows:int = DEFAULT_N_WINDOWS) -> int:
    """
    Heurística simple para inferir ventana en base a un índice global en dataset augmentado asumiendo
    un patrón repetitivo por trial. Útil si no hay metadata.
    """
    # This is a very rough heuristic and only used as last resort.
    pos_in_block = (global_idx // augmentation_factor_per_window) % n_windows
    return int(pos_in_block)

def collect_window_accuracies(subject_dir: Path, subset: str, n_windows_guess: int = DEFAULT_N_WINDOWS, aug_per_window_guess: int = DEFAULT_AUG_PER_WINDOW):
    subset_dir = subject_dir / subset
    if not subset_dir.exists():
        return None
    window_acc_per_fold = []
    raw_predictions_per_fold = []
    seed_dirs = sorted([p for p in subset_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")])
    if not seed_dirs:
        seed_dirs = [subset_dir]
    for seed_dir in seed_dirs:
        fold_dirs = sorted([p for p in seed_dir.iterdir() if p.is_dir() and p.name.startswith("fold_")])
        for fold_dir in fold_dirs:
            meta = load_json_safe(fold_dir / "metadata.json")
            if meta is None or meta.get("status") != "success":
                continue
            preds_path = fold_dir / "test_preds.npz"
            if not preds_path.exists():
                continue
            try:
                d = np.load(preds_path, allow_pickle=True)
                y_true = np.array(d["y_true"])
                y_pred = np.array(d["y_pred"])
            except Exception as e:
                warnings.warn(f"Could not load preds for {fold_dir}: {e}")
                continue

            # If augmentation metadata exposes mapping or n_windows, prefer that
            aug_meta = load_json_safe(fold_dir / "augmentation_metadata.json")
            if aug_meta is not None:
                window_params = aug_meta.get("window_params", {})
                n_windows = int(window_params.get("n_windows_per_trial", n_windows_guess))
                indices_info = aug_meta.get("indices", {})
                original_test_indices = indices_info.get("test_original_indices")
                if original_test_indices is not None:
                    n_test_trials = len(original_test_indices)
                    n_original_windows = n_test_trials * n_windows
                    if len(y_true) >= n_original_windows:
                        # windows ordering: for trial_idx in range(n_trials): for window_idx in range(n_windows): append
                        windows_order = np.tile(np.arange(n_windows, dtype=int), n_test_trials)
                        y_true_block = y_true[:n_original_windows]
                        y_pred_block = y_pred[:n_original_windows]
                        window_accs = []
                        for w in range(n_windows):
                            mask = (windows_order == w)
                            if mask.sum() == 0:
                                window_accs.append(np.nan)
                            else:
                                window_accs.append(float(np.mean(y_true_block[mask] == y_pred_block[mask])))
                        window_acc_per_fold.append(np.array(window_accs))
                        raw_predictions_per_fold.append({"windows": windows_order, "y_true": y_true_block, "y_pred": y_pred_block, "fold_path": str(fold_dir)})
                        continue
            # Fallback heuristic: infer window from position inside augmented blocks
            n_windows = n_windows_guess
            aug_per_window = aug_per_window_guess
            # Build windows array for the predicted samples (best-effort)
            windows_inferred = np.array([infer_temporal_window_from_index_by_heuristic(i, aug_per_window, n_windows) for i in range(len(y_true))])
            window_accs = []
            for w in range(n_windows):
                mask = (windows_inferred == w)
                if mask.sum() == 0:
                    window_accs.append(np.nan)
                else:
                    window_accs.append(float(np.mean(y_true[mask] == y_pred[mask])))
            window_acc_per_fold.append(np.array(window_accs))
            raw_predictions_per_fold.append({"windows": windows_inferred, "y_true": y_true, "y_pred": y_pred, "fold_path": str(fold_dir)})
    if not window_acc_per_fold:
        return None
    return {
        "window_accuracies": window_acc_per_fold,
        "raw_predictions": raw_predictions_per_fold
    }

def plot_temporal_window_boxplots(data_vocales, data_comandos, output_path: Path, subject_name: str, n_windows:int):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    window_labels = [f"Win {i}" for i in range(n_windows)]
    for ax, data, subset_name, n_classes in zip(axes, [data_vocales, data_comandos], ["Vocales", "Comandos"], [5,6]):
        if data is None or not data["window_accuracies"]:
            ax.text(0.5, 0.5, f"No data for {subset_name}", ha='center', va='center'); ax.axis('off'); continue
        window_arr = np.vstack(data["window_accuracies"])
        bp = ax.boxplot([window_arr[:, i] for i in range(n_windows)], labels=window_labels, patch_artist=True, showmeans=True, meanline=True)
        chance_level = 1.0 / n_classes
        data_min = np.nanmin(window_arr); data_max = np.nanmax(window_arr)
        y_min = max(0, min(chance_level, data_min) * 0.9); y_max = min(1, max(chance_level, data_max) * 1.1)
        ax.set_ylim(y_min, y_max)
        ax.axhline(y=chance_level, color='red', linestyle=':', linewidth=1.5)
        ax.set_title(f"{subset_name}"); ax.set_ylabel("Accuracy")
    fig.suptitle(f"Accuracy by Temporal Window - {subject_name}", fontsize=14)
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

# ---------------- GLOBAL plotting ----------------

def plot_global_metrics_boxplots(all_subjects_data: Dict, output_path: Path):
    fig, ax = plt.subplots(figsize=(12,6))
    all_data=[]; labels=[]; positions=[]; colors=[]
    color_map = {"Precision":"#1f77b4", "Recall":"#ff7f0e", "F1-Score":"#2ca02c"}
    pos=0
    for subset_name in SUBSETS:
        prec_list=[]; rec_list=[]; f1_list=[]
        for subj_name, subj_summary in all_subjects_data.items():
            if subj_summary is None: continue
            subset_data = subj_summary.get(subset_name)
            if subset_data is None: continue
            if subset_data.get("precision_mean") is not None: prec_list.append(subset_data["precision_mean"])
            if subset_data.get("recall_mean") is not None: rec_list.append(subset_data["recall_mean"])
            if subset_data.get("f1_mean") is not None: f1_list.append(subset_data["f1_mean"])
        if not prec_list and not rec_list and not f1_list:
            pos += 4; continue
        for metric_data, metric_name in [(prec_list,"Precision"), (rec_list,"Recall"), (f1_list,"F1-Score")]:
            if metric_data:
                all_data.append(metric_data); labels.append(f"{subset_name.capitalize()}\n{metric_name}"); positions.append(pos); colors.append(color_map[metric_name])
            pos += 1
        pos += 1
    if not all_data:
        ax.text(0.5,0.5,"No data available", ha='center', va='center'); ax.axis('off')
    else:
        bp = ax.boxplot(all_data, positions=positions, widths=0.6, patch_artist=True, showmeans=True, meanline=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        ax.set_xticks(positions); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Score"); data_concat = np.concatenate([np.array(d) for d in all_data]); ax.set_ylim(0.9*data_concat.min(), 1.1*data_concat.max())
        ax.grid(True, alpha=0.3); ax.set_title("Global Metrics Distribution (Across Subjects)")
    fig.tight_layout(); fig.savefig(output_path, dpi=150, bbox_inches='tight'); plt.close(fig)

def plot_global_confusion_matrices(all_subjects_data_raw: Dict, output_path: Path):
    fig, axes = plt.subplots(1,2,figsize=(14,6))
    for ax, subset_name, class_names in zip(axes, SUBSETS, [VOCAL_CLASS_NAMES, COMANDO_CLASS_NAMES]):
        cm_list=[]
        for subj_name, subj_data in all_subjects_data_raw.items():
            subset_data = subj_data.get(subset_name)
            if subset_data is None or not subset_data["confusion_matrices"]:
                continue
            cm_stack = np.stack(subset_data["confusion_matrices"], axis=0)
            cm_mean_subj = np.nanmean(cm_stack, axis=0)
            cm_list.append(cm_mean_subj)
        if not cm_list:
            ax.text(0.5,0.5,f"No data for {subset_name}", ha='center', va='center'); ax.axis('off'); continue
        cm_global_stack = np.stack(cm_list, axis=0); cm_mean = np.nanmean(cm_global_stack, axis=0); cm_std = np.nanstd(cm_global_stack, axis=0)
        n = cm_mean.shape[0]; annot = np.empty((n,n), dtype=object)
        for r in range(n):
            for c in range(n):
                annot[r,c] = f"{cm_mean[r,c]:.1f}\n±{cm_std[r,c]:.1f}"
        sns.heatmap(cm_mean, annot=annot, fmt="", cmap="cividis", ax=ax, cbar_kws={'label':'% (row-normalized)'})
        ax.set_title(f"{subset_name.capitalize()} (mean ± std %)"); ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_xticklabels(class_names, rotation=45, ha='right'); ax.set_yticklabels(class_names)
    fig.suptitle("Global Confusion Matrices (Across Subjects)"); fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(output_path, dpi=150, bbox_inches='tight'); plt.close(fig)

def plot_global_band_accuracy_boxplots(all_subjects_data_raw: Dict, output_path: Path):
    fig, axes = plt.subplots(1,2,figsize=(12,5)); band_freq_labels = [f"{b}" for b in BAND_LABELS]
    for ax, subset_name, n_classes in zip(axes, SUBSETS, [5,6]):
        band_data_per_subject=[]
        for subj_name, subj_data in all_subjects_data_raw.items():
            subset_data = subj_data.get(subset_name)
            if subset_data is None or not subset_data["band_accuracies"]:
                continue
            band_arr = np.vstack(subset_data["band_accuracies"])
            if np.all(np.isnan(band_arr)): continue
            band_mean_subj = np.nanmean(band_arr, axis=0); band_data_per_subject.append(band_mean_subj)
        if not band_data_per_subject:
            ax.text(0.5,0.5,f"No data for {subset_name}", ha='center', va='center'); ax.axis('off'); continue
        band_matrix = np.vstack(band_data_per_subject)
        bp = ax.boxplot([band_matrix[:, i] for i in range(len(BAND_LABELS))], labels=band_freq_labels, patch_artist=True, showmeans=True, meanline=True)
        chance_level = 1.0 / n_classes; data_min=np.nanmin(band_matrix); data_max=np.nanmax(band_matrix)
        y_min=max(0, min(chance_level*0.9, data_min*0.9)); y_max=min(1, max(chance_level*0.9, data_max*1.1))
        ax.set_ylim(y_min,y_max); ax.axhline(y=chance_level, color='red', linestyle=':', linewidth=1.5); ax.set_title(f"{subset_name.capitalize()}"); ax.set_ylabel("Accuracy")
    fig.suptitle("Global Accuracy by Frequency Band"); fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(output_path, dpi=150, bbox_inches='tight'); plt.close(fig)

def plot_global_ftsurrogate_accuracy_boxplots(all_subjects_data_raw: Dict, output_path: Path):
    fig, axes = plt.subplots(1,2,figsize=(12,5))
    for ax, subset_name, n_classes in zip(axes, SUBSETS, [5,6]):
        with_fts_per_subject=[]; without_fts_per_subject=[]
        for subj_name, subj_data in all_subjects_data_raw.items():
            subset_data = subj_data.get(subset_name)
            if subset_data is None or not subset_data["ftsurrogate_accuracies"]: continue
            with_fts = [d["with_fts"] for d in subset_data["ftsurrogate_accuracies"] if not math.isnan(d["with_fts"])]
            without_fts = [d["without_fts"] for d in subset_data["ftsurrogate_accuracies"] if not math.isnan(d["without_fts"])]
            if with_fts: with_fts_per_subject.append(np.nanmean(with_fts))
            if without_fts: without_fts_per_subject.append(np.nanmean(without_fts))
        if not with_fts_per_subject and not without_fts_per_subject:
            ax.text(0.5,0.5,f"No data for {subset_name}", ha='center', va='center'); ax.axis('off'); continue
        data_to_plot=[]; labels=[]
        if without_fts_per_subject: data_to_plot.append(without_fts_per_subject); labels.append("Without FTS")
        if with_fts_per_subject: data_to_plot.append(with_fts_per_subject); labels.append("With FTS")
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, showmeans=True, meanline=True)
        chance_level = 1.0/n_classes; all_data = (without_fts_per_subject + with_fts_per_subject) if (without_fts_per_subject or with_fts_per_subject) else [0]
        y_min = max(0, min(chance_level*0.9, min(all_data))*0.9); y_max=min(1, max(chance_level*0.9, max(all_data))*1.1)
        ax.set_ylim(y_min,y_max); ax.axhline(y=chance_level, color='red', linestyle=':', linewidth=1.5); ax.set_title(f"{subset_name.capitalize()}"); ax.set_ylabel("Accuracy")
    fig.suptitle("Global Accuracy by FT Surrogate Usage"); fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(output_path, dpi=150, bbox_inches='tight'); plt.close(fig)

# ---------------- Main pipeline ----------------

def process_subject(subject_dir: Path, preproc_dir: Path, output_root: Path):
    subject_name = subject_dir.name
    print(f"[Visualizer] Processing subject: {subject_name}")
    # find preprocessed file (non-augmented)
    X_pre, Y_pre = load_subject_preprocessed(subject_name, preproc_dir)
    subject_out = output_root / subject_name
    subject_out.mkdir(parents=True, exist_ok=True)
    # collect data
    data_vocales = collect_fold_data(subject_dir, "vocales", X_pre, Y_pre)
    data_comandos = collect_fold_data(subject_dir, "comandos", X_pre, Y_pre)
    if data_vocales is None and data_comandos is None:
        print(f"  No data found for {subject_name}, skipping...")
        return None

    # save visuals
    plot_learning_curves(data_vocales, data_comandos, subject_out / "learning_curves.png", subject_name)
    plot_confusion_matrices(data_vocales, data_comandos, subject_out / "confusion_matrices.png", subject_name)
    plot_metrics_boxplots(data_vocales, data_comandos, subject_out / "metrics_boxplots.png", subject_name)
    plot_band_accuracy_boxplots(data_vocales, data_comandos, subject_out / "band_accuracy_boxplots.png", subject_name)
    plot_ftsurrogate_accuracy_boxplots(data_vocales, data_comandos, subject_out / "ftsurrogate_accuracy_boxplots.png", subject_name)

    # save summary json per subject
    summary = {}
    for subset_name, data in [("vocales", data_vocales), ("comandos", data_comandos)]:
        if data is None:
            continue
        metrics_arr = np.array([[m["precision"], m["recall"], m["f1"]] for m in data["metrics"]]) if data["metrics"] else np.array([])
        summary[subset_name] = {
            "n_folds": len(data["metrics"]) if data["metrics"] else 0,
            "precision_mean": float(np.nanmean(metrics_arr[:,0])) if metrics_arr.size else None,
            "precision_std": float(np.nanstd(metrics_arr[:,0])) if metrics_arr.size else None,
            "recall_mean": float(np.nanmean(metrics_arr[:,1])) if metrics_arr.size else None,
            "recall_std": float(np.nanstd(metrics_arr[:,1])) if metrics_arr.size else None,
            "f1_mean": float(np.nanmean(metrics_arr[:,2])) if metrics_arr.size else None,
            "f1_std": float(np.nanstd(metrics_arr[:,2])) if metrics_arr.size else None,
            "has_window_predictions": bool(data.get("window_predictions_raw"))
        }
    save_json_safe(subject_out / "summary.json", summary)
    print(f"  Saved summary: {subject_out / 'summary.json'}")

    return {
        "summary": summary,
        "vocales": data_vocales,
        "comandos": data_comandos
    }

def main():
    EXP_ROOT = find_latest_experiment(EXPERIMENTS_ROOT, EXPERIMENT_NAME_PREFIX)
    print(f"[Visualizer] Using experiment root: {EXP_ROOT}")
    OUTPUT_ROOT = EXP_ROOT / OUTPUT_SUBDIR
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    subject_dirs = sorted([p for p in EXP_ROOT.iterdir() if p.is_dir() and p.name.upper().startswith("S")])
    print(f"[Visualizer] Found {len(subject_dirs)} subject directories")

    all_subjects_summaries = {}
    all_subjects_data_raw = {}
    temporal_data_per_subject = {}

    for subject_dir in subject_dirs:
        subject_name = subject_dir.name
        result = process_subject(subject_dir, PREPROCESSED_DIR, OUTPUT_ROOT)
        if result is not None:
            all_subjects_summaries[subject_name] = result["summary"]
            all_subjects_data_raw[subject_name] = {
                "vocales": result["vocales"],
                "comandos": result["comandos"]
            }
            # temporal window collection (use default guesses; if augmentation metadata has better info we use it inside)
            dv = collect_window_accuracies(subject_dir, "vocales")
            dc = collect_window_accuracies(subject_dir, "comandos")
            temporal_data_per_subject[subject_name] = {"vocales": dv, "comandos": dc}
            # plot per-subject temporal windows if available
            if dv is not None or dc is not None:
                n_windows = DEFAULT_N_WINDOWS
                # try to guess n_windows from augmentation_metadata of any fold in vocales subset
                try:
                    # look for any augmentation_metadata.json
                    subset_dir = subject_dir / "vocales"
                    seed_dirs = sorted([p for p in subset_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")]) if subset_dir.exists() else []
                    found = False
                    for sd in seed_dirs:
                        for fd in sd.iterdir():
                            aug_meta = load_json_safe(fd / "augmentation_metadata.json")
                            if aug_meta:
                                wp = aug_meta.get("window_params")
                                if wp and wp.get("n_windows_per_trial"):
                                    n_windows = int(wp.get("n_windows_per_trial"))
                                    found = True
                                    break
                        if found:
                            break
                except Exception:
                    pass
                plot_temporal_window_boxplots(dv, dc, OUTPUT_ROOT / subject_name / "temporal_window_boxplots.png", subject_name, n_windows)
                # save temporal summary
                temp_summary = {}
                for subset_name, data in [("vocales", dv), ("comandos", dc)]:
                    if data is None:
                        continue
                    window_arr = np.vstack(data["window_accuracies"])
                    temp_summary[subset_name] = {
                        "n_folds": len(data["window_accuracies"]),
                        "window_mean": np.nanmean(window_arr, axis=0).tolist(),
                        "window_std": np.nanstd(window_arr, axis=0).tolist()
                    }
                save_json_safe(OUTPUT_ROOT / subject_name / "temporal_window_summary.json", temp_summary)
                print(f"  Saved temporal summary: {OUTPUT_ROOT / subject_name / 'temporal_window_summary.json'}")

    # Global visualizations
    print("\n[Visualizer] Generating global visualizations...")
    global_out = OUTPUT_ROOT / "global"
    global_out.mkdir(parents=True, exist_ok=True)

    plot_global_metrics_boxplots(all_subjects_summaries, global_out / "metrics_boxplots_global.png")
    plot_global_confusion_matrices(all_subjects_data_raw, global_out / "confusion_matrices_global.png")
    plot_global_band_accuracy_boxplots(all_subjects_data_raw, global_out / "band_accuracy_boxplots_global.png")
    plot_global_ftsurrogate_accuracy_boxplots(all_subjects_data_raw, global_out / "ftsurrogate_accuracy_boxplots_global.png")

    # Global temporal window
    # aggregate temporal_data_per_subject into structure usable by global plot
    aggregated_temporal = {}
    for subj_name, d in temporal_data_per_subject.items():
        aggregated_temporal[subj_name] = d
    plot_global_temporal_boxplots = False
    # determine n_windows for global plots using first available augmentation metadata
    n_windows_global = DEFAULT_N_WINDOWS
    for subject_dir in subject_dirs:
        subset_dir = subject_dir / "vocales"
        if subset_dir.exists():
            seed_dirs = sorted([p for p in subset_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")])
            for sd in seed_dirs:
                for fd in sd.iterdir():
                    aug_meta = load_json_safe(fd / "augmentation_metadata.json")
                    if aug_meta:
                        wp = aug_meta.get("window_params")
                        if wp and wp.get("n_windows_per_trial"):
                            n_windows_global = int(wp.get("n_windows_per_trial"))
                            plot_global_temporal_boxplots = True
                            break
                if plot_global_temporal_boxplots:
                    break
        if plot_global_temporal_boxplots:
            break
    # if no augmentation metadata found but at least one subject has temporal data from heuristic, plot global anyway
    any_temporal = any((temporal_data_per_subject.get(s, {}).get("vocales") is not None or temporal_data_per_subject.get(s, {}).get("comandos") is not None) for s in temporal_data_per_subject.keys())
    if any_temporal:
        # build structure for global plotting: for each subject, take mean across folds if available
        global_temporal_agg = {}
        for subj_name, d in temporal_data_per_subject.items():
            subj_entry = {}
            for subset_name in ["vocales", "comandos"]:
                data = d.get(subset_name)
                if data is None:
                    subj_entry[subset_name] = None
                    continue
                window_arr = np.vstack(data["window_accuracies"])
                subj_entry[subset_name] = {
                    "window_mean_subj": np.nanmean(window_arr, axis=0)
                }
            global_temporal_agg[subj_name] = subj_entry
        # prepare data in required format for the global plotting function implemented earlier
        # create data structure similar to earlier functions, but simpler
        # We'll reuse plot_global_temporal_window_boxplots logic by building a minimal structure
        # For simplicity, create a custom global plot here:
        fig, axes = plt.subplots(1,2,figsize=(12,5))
        window_labels = [f"Win {i}" for i in range(n_windows_global)]
        for ax, subset_name, n_classes in zip(axes, ["vocales","comandos"], [5,6]):
            per_subject_values = []
            for subj_name, entry in global_temporal_agg.items():
                subset_entry = entry.get(subset_name)
                if subset_entry is None: continue
                per_subject_values.append(subset_entry["window_mean_subj"])
            if not per_subject_values:
                ax.text(0.5,0.5,f"No temporal data for {subset_name}", ha='center', va='center'); ax.axis('off'); continue
            mat = np.vstack(per_subject_values)
            bp = ax.boxplot([mat[:, i] for i in range(n_windows_global)], labels=window_labels, patch_artist=True, showmeans=True, meanline=True)
            chance_level = 1.0 / (5 if subset_name=="vocales" else 6)
            data_min = np.nanmin(mat); data_max = np.nanmax(mat)
            y_min = max(0, min(chance_level, data_min) * 0.9); y_max = min(1, max(chance_level, data_max) * 1.1)
            ax.set_ylim(y_min,y_max); ax.axhline(y=chance_level, color='gray', linestyle=':', linewidth=1.5)
            ax.set_title(subset_name.capitalize()); ax.set_ylabel("Accuracy")
        fig.suptitle("Global Accuracy by Temporal Window (Across Subjects)"); fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(global_out / "temporal_window_boxplots_global.png", dpi=150, bbox_inches='tight'); plt.close(fig)
        print(f"  Saved global temporal window plot: {global_out / 'temporal_window_boxplots_global.png'}")

    # Save global summary
    global_summary = {
        "experiment_root": str(EXP_ROOT),
        "n_subjects": len(all_subjects_summaries),
        "subjects": list(all_subjects_summaries.keys()),
        "per_subject_summaries": all_subjects_summaries
    }
    save_json_safe(global_out / "global_summary.json", global_summary)
    print(f"Saved global summary: {global_out / 'global_summary.json'}")

    print("\n[Visualizer] Done! Results saved to:", OUTPUT_ROOT)

if __name__ == "__main__":
    main()
