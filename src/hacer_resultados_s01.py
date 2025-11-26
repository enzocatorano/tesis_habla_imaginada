# src/hacer_resultados_s01.py
"""
Genera plots y un resumen agregado para S01 usando las salidas del experimento k-fold.

Salidas (en ../experiments/S01_subject1_kfold/results/):
  - boxplot_vocales_vs_comandos.png
  - learning_curves_vocales.png
  - learning_curves_comandos.png
  - confusion_meanstd_vocales_comandos.png   # ambas matrices en la misma figura (cada celda: mean on first line, std on second)
  - confusion_meanstd_single_vocales.png
  - confusion_meanstd_single_comandos.png
  - band_accuracy_vocales.png
  - band_accuracy_comandos.png
  - summary_results.json
"""
import os
import json
from pathlib import Path
from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix

# --------------------------
# Config (ajustá si hace falta)
# --------------------------
ROOT = Path(__file__).resolve().parents[1]  # carpeta padre de src (proyecto)
DATA_PATH = ROOT / "data" / "processed_aug" / "S01_EEG_augmented.npz"
EXPS_ROOT = ROOT / "experiments" / "S01_subject1_kfold"
RESULTS_DIR = EXPS_ROOT / "results"
K = 6
RANDOM_SEED = 17

# subsets definition (match run_s01_kfold)
SUBSETS = [
    ("vocales", 1, 5, "estimulo_vocal"),    # name, stim_min, stim_max, clase_str
    ("comandos", 6, 11, "estimulo_comando"),
]

# band labels human readable
BAND_LABELS = ["delta 0-4", "theta 4-8", "alpha 8-12", "beta 12-32", "gamma 32-64"]

# class names for plotting confusion matrices
VOCAL_CLASS_NAMES = ['A','E','I','O','U']
COMANDO_CLASS_NAMES = ['Arriba','Abajo','Izquierda','Derecha','Adelante','Atras']

# create results folder
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------
# Helpers
# --------------------------
def load_npz_data(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    X = d["data"]     # (trials, channels, time)
    Y = d["labels"]   # (trials, 4)
    return X, Y

def find_fold_dirs(subset_dir):
    if not subset_dir.exists():
        return []
    fold_dirs = [p for p in subset_dir.iterdir() if p.is_dir() and p.name.startswith("fold_")]
    fold_dirs_sorted = sorted(fold_dirs, key=lambda p: int(p.name.split("_")[1]) if "_" in p.name else p.name)
    return fold_dirs_sorted

def load_metrics_json_from_fold(fold_dir):
    jm = fold_dir / "train_metrics.json"
    if jm.exists():
        try:
            return json.loads(jm.read_text(encoding="utf8"))
        except Exception:
            pass
    for p in fold_dir.iterdir():
        if p.is_dir() and p.name.startswith("run_"):
            candidate = p / "metrics_epochs.json"
            if candidate.exists():
                try:
                    return json.loads(candidate.read_text(encoding="utf8"))
                except Exception:
                    pass
    return None

def load_test_preds_from_fold(fold_dir):
    f = fold_dir / "test_preds.npz"
    if f.exists():
        try:
            d = np.load(f, allow_pickle=True)
            y_true = d["y_true"]
            y_pred = d["y_pred"]
            return np.array(y_true), np.array(y_pred)
        except Exception:
            pass
    return None, None

def load_confusion_from_fold(fold_dir):
    f = fold_dir / "confusion_matrix.npy"
    if f.exists():
        try:
            return np.load(f)
        except Exception:
            pass
    return None

def aggregate_epoch_series(list_of_lists):
    # convert to 2D array with shape (n_folds, max_epochs), pad with nan where shorter
    if not list_of_lists:
        return np.array([]), np.array([])
    lengths = [len(l) for l in list_of_lists]
    max_len = max(lengths)
    arr = np.full((len(list_of_lists), max_len), np.nan, dtype=float)
    for i, l in enumerate(list_of_lists):
        if len(l) > 0:
            arr[i, :len(l)] = np.array(l, dtype=float)
    mean = np.nanmean(arr, axis=0)
    std  = np.nanstd(arr, axis=0)
    return mean, std

# --------------------------
# Cargar datos originales (para mapear bandas + splits)
# --------------------------
print("Cargando datos originales:", DATA_PATH)
X_all, Y_all = load_npz_data(DATA_PATH)
estimulo_all = Y_all[:, 1].astype(int)   # 1..11
banda_all = Y_all[:, 3].astype(int)      # 0..4
indices_total = np.arange(len(estimulo_all))
print("Datos totales:", X_all.shape, Y_all.shape)

# --------------------------
# Procesar cada subset
# --------------------------
summary = {}

for subset_name, stim_min, stim_max, clase_str in SUBSETS:
    print(f"\n=== Procesando subset: {subset_name} ===")
    subset_dir = EXPS_ROOT / subset_name
    fold_dirs = find_fold_dirs(subset_dir)
    if not fold_dirs:
        print(f"No hay folds encontrados en {subset_dir}. Saltando.")
        continue

    # seleccionar indices del sujeto para este subset
    mask = (estimulo_all >= stim_min) & (estimulo_all <= stim_max)
    X = X_all[mask]
    Y_full = estimulo_all[mask]  # original 1..11
    indices_global = np.where(mask)[0]  # indices in the original dataset

    # map labels to 0..n-1
    if subset_name == "vocales":
        Y = (Y_full - 1).astype(int)   # 1..5 -> 0..4
        n_classes = 5
        class_names = VOCAL_CLASS_NAMES
    else:
        Y = (Y_full - 6).astype(int)   # 6..11 -> 0..5
        n_classes = 6
        class_names = COMANDO_CLASS_NAMES

    n_trials, C, T = X.shape
    print(f"Subset trials: {n_trials}, channels: {C}, time: {T}, classes: {n_classes}")

    # recreate stratified KFold splits on the subset (same config used en run)
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=RANDOM_SEED)
    splits = list(skf.split(X, Y))  # list of (train_idx, test_idx) in local indexing

    # containers for aggregation
    fold_accuracies = []
    per_class_precisions = []
    fold_conf_mats = []
    fold_train_losses = []
    fold_val_losses = []
    fold_val_accs = []
    band_acc_per_fold = []

    # iterate through fold directories (assume sorted order corresponds)
    for i, fold_dir in enumerate(fold_dirs):
        fold_idx = i + 1
        print(f"\n-- Fold {fold_idx}: {fold_dir.name}")

        # load saved preds and metrics
        y_true_saved, y_pred_saved = load_test_preds_from_fold(fold_dir)
        metrics = load_metrics_json_from_fold(fold_dir)

        # determine local test indices
        try:
            _, test_idx_local = splits[i]
        except Exception:
            print("Warning: mismatch in splits indexing; usando último split como fallback.")
            _, test_idx_local = splits[-1]

        # global indices of these test samples
        test_global_idx = indices_global[test_idx_local]

        # check preds availability
        if y_true_saved is None or y_pred_saved is None or len(y_true_saved) != len(test_idx_local):
            print(f"No se encontraron preds válidas en {fold_dir}; se salta fold.")
            continue

        y_true = np.array(y_true_saved).astype(int)
        y_pred = np.array(y_pred_saved).astype(int)

        acc = float(accuracy_score(y_true, y_pred))
        fold_accuracies.append(acc)

        precs = precision_score(y_true, y_pred, labels=np.arange(n_classes), average=None, zero_division=0)
        per_class_precisions.append(precs.tolist())

        cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
        # normalize by true class (row-wise) to get percentages per row
        with np.errstate(divide='ignore', invalid='ignore'):
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_pct = np.divide(cm, row_sums, where=(row_sums != 0)) * 100.0  # percentages
            cm_pct = np.nan_to_num(cm_pct, nan=0.0)
        fold_conf_mats.append(cm_pct)

        # training metrics
        if metrics is not None:
            def to_float_array(lst):
                if lst is None:
                    return None
                return np.array([ (float(v) if v is not None else np.nan) for v in lst ], dtype=float)
            tl = to_float_array(metrics.get("train_losses", None))
            vl = to_float_array(metrics.get("val_losses", None))
            va = to_float_array(metrics.get("val_accs", None))
            fold_train_losses.append(tl.tolist() if tl is not None else [])
            fold_val_losses.append(vl.tolist() if vl is not None else [])
            fold_val_accs.append(va.tolist() if va is not None else [])
        else:
            fold_train_losses.append([])
            fold_val_losses.append([])
            fold_val_accs.append([])

        # band-wise accuracy for this fold (use global band labels)
        bands = banda_all[test_global_idx]  # array of band labels for tests in this fold
        n_bands = int(np.max(banda_all) + 1)
        band_accs = []
        for b in range(n_bands):
            mask_b = (bands == b)
            if np.sum(mask_b) == 0:
                band_accs.append(np.nan)
            else:
                y_true_b = y_true[mask_b]
                y_pred_b = y_pred[mask_b]
                band_accs.append(float(accuracy_score(y_true_b, y_pred_b)))
        band_acc_per_fold.append(band_accs)

    # Convert to arrays
    fold_accuracies = np.array(fold_accuracies, dtype=float)
    per_class_precisions = np.array(per_class_precisions, dtype=float) if per_class_precisions else np.zeros((0, n_classes))
    fold_conf_mats = np.array(fold_conf_mats, dtype=float) if fold_conf_mats else np.zeros((0, n_classes, n_classes))
    band_acc_per_fold = np.array(band_acc_per_fold, dtype=float) if band_acc_per_fold else np.zeros((0, int(np.max(banda_all)+1)))

    # Aggregate
    class_prec_mean = np.nanmean(per_class_precisions, axis=0).tolist() if per_class_precisions.size else []
    class_prec_std  = np.nanstd(per_class_precisions, axis=0).tolist() if per_class_precisions.size else []

    if fold_conf_mats.size:
        cm_mean = np.nanmean(fold_conf_mats, axis=0)
        cm_std  = np.nanstd(fold_conf_mats, axis=0)
    else:
        cm_mean = np.zeros((n_classes, n_classes))
        cm_std = np.zeros((n_classes, n_classes))

    if band_acc_per_fold.size:
        band_mean = np.nanmean(band_acc_per_fold, axis=0).tolist()
        band_std = np.nanstd(band_acc_per_fold, axis=0).tolist()
    else:
        band_mean = []
        band_std = []

    # Learning curves aggregated
    train_loss_mean, train_loss_std = aggregate_epoch_series(fold_train_losses)
    val_loss_mean, val_loss_std     = aggregate_epoch_series(fold_val_losses)
    val_acc_mean, val_acc_std       = aggregate_epoch_series(fold_val_accs)

    # Save per-subset summary numeric
    subset_summary = {
        "fold_accuracies": fold_accuracies.tolist(),
        "accuracy_mean": float(np.nanmean(fold_accuracies)) if fold_accuracies.size else None,
        "accuracy_std": float(np.nanstd(fold_accuracies)) if fold_accuracies.size else None,
        "per_class_precision_mean": class_prec_mean,
        "per_class_precision_std": class_prec_std,
        "confusion_mean_pct": cm_mean.tolist(),
        "confusion_std_pct": cm_std.tolist(),
        "band_accuracy_mean": band_mean,
        "band_accuracy_std": band_std,
        "train_loss_mean": train_loss_mean.tolist(),
        "train_loss_std": train_loss_std.tolist(),
        "val_loss_mean": val_loss_mean.tolist(),
        "val_loss_std": val_loss_std.tolist(),
        "val_acc_mean": val_acc_mean.tolist(),
        "val_acc_std": val_acc_std.tolist()
    }
    summary[subset_name] = subset_summary

    # -------------------------
    # Generar y guardar figuras
    # -------------------------

    # 1) Learning curves with twin y-axes
    epochs = np.arange(1, len(train_loss_mean) + 1) if train_loss_mean.size else np.arange(1, len(val_loss_mean)+1)
    fig, ax_loss = plt.subplots(figsize=(10,6))
    color_train = "tab:blue"
    color_val = "tab:orange"
    color_acc = "tab:green"

    # plot train loss
    if train_loss_mean.size:
        ax_loss.plot(epochs, train_loss_mean, label="Train Loss", color=color_train)
        ax_loss.fill_between(epochs, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std, alpha=0.2, color=color_train)
    # plot val loss
    if val_loss_mean.size:
        epochs_val = np.arange(1, len(val_loss_mean) + 1)
        ax_loss.plot(epochs_val, val_loss_mean, label="Val Loss", color=color_val)
        ax_loss.fill_between(epochs_val, val_loss_mean - val_loss_std, val_loss_mean + val_loss_std, alpha=0.2, color=color_val)

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True)

    # twin axis for validation accuracy
    ax_acc = ax_loss.twinx()
    if val_acc_mean.size:
        epochs_acc = np.arange(1, len(val_acc_mean) + 1)
        ax_acc.plot(epochs_acc, val_acc_mean, label="Val Acc", color=color_acc, linestyle='--')
        ax_acc.fill_between(epochs_acc, val_acc_mean - val_acc_std, val_acc_mean + val_acc_std, alpha=0.15, color=color_acc)
    ax_acc.set_ylabel("Validation Accuracy")
    # legend combine both axes
    lines_loss, labels_loss = ax_loss.get_legend_handles_labels()
    lines_acc, labels_acc = ax_acc.get_legend_handles_labels()
    ax_loss.legend(lines_loss + lines_acc, labels_loss + labels_acc, loc='upper right')

    plt.title(f"Learning curves (averaged) - {subset_name}")
    fname_lc = RESULTS_DIR / f"learning_curves_{subset_name}.png"
    plt.savefig(fname_lc, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved learning curves -> {fname_lc}")

    # 2) Confusion matrices mean±std (single image per subset)
    # Build annotation with mean on first line and std on second line for each cell
    n = cm_mean.shape[0]
    annot_lines = np.empty((n, n), dtype=object)
    for r in range(n):
        for c in range(n):
            mean_val = cm_mean[r, c]
            std_val = cm_std[r, c]
            annot_lines[r, c] = f"{mean_val:.1f}\n±{std_val:.1f}"

    # Figure size scaled with number of classes to avoid crowding
    width = max(6, n * 0.9)
    height = max(5, n * 0.9)
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(cm_mean, annot=annot_lines, fmt="", cmap="cividis", ax=ax,
                cbar_kws={'label':'% (row-normalized mean)'},
                linewidths=0.5, linecolor='gray', annot_kws={"size":10})
    ax.set_title(f"Confusion matrix mean (top) / std (bottom) [%] - {subset_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names, rotation=0)
    fname_cm_single = RESULTS_DIR / f"confusion_meanstd_single_{subset_name}.png"
    plt.savefig(fname_cm_single, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved single confusion mean±std -> {fname_cm_single}")

    # 3) Band accuracy barplot with human labels (mean ± std)
    if band_mean:
        bands = list(range(len(band_mean)))
        x = np.arange(len(band_mean))
        fig, ax = plt.subplots(figsize=(8,5))
        ax.bar(x, band_mean, yerr=band_std, capsize=5)
        ax.set_xticks(x)
        labels = BAND_LABELS[:len(band_mean)]
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_xlabel("Banda afectada")
        ax.set_ylabel("Accuracy (test)")
        ax.set_title(f"Accuracy por banda (mean ± std) - {subset_name}")
        fname_band = RESULTS_DIR / f"band_accuracy_{subset_name}.png"
        plt.savefig(fname_band, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved band accuracy plot -> {fname_band}")

# --------------------------
# Combined boxplot: vocales vs comandos
# --------------------------
print("\nGenerando boxplot comparativo (vocales vs comandos).")
boxplot_data = []
labels = []
for subset_name, _, _, _ in SUBSETS:
    if subset_name not in summary:
        continue
    accs = summary[subset_name]["fold_accuracies"]
    if not accs:
        continue
    boxplot_data.append(accs)
    labels.append(subset_name)

if boxplot_data:
    plt.figure(figsize=(8,6))
    sns.boxplot(data=boxplot_data)
    plt.xticks(ticks=np.arange(len(labels)), labels=labels)
    plt.ylabel("Test accuracy per fold")
    plt.title("Comparación de accuracies por fold: vocales vs comandos")
    fname_box = RESULTS_DIR / "boxplot_vocales_vs_comandos.png"
    plt.savefig(fname_box, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved boxplot -> {fname_box}")

# --------------------------
# Confusion matrices: put both subsets in the same figure side-by-side
# Each cell shows mean (top line) and std (bottom line)
# --------------------------
print("\nGenerando figura conjunta de confusion matrices (mean on first line, std on second).")
# Get available subsets for plotting in same order as SUBSETS
available = [s[0] for s in SUBSETS if s[0] in summary]
n_plots = len(available)
if n_plots == 0:
    print("No hay datos para confusion matrices combinadas. Skipping.")
else:
    # compute combined figure width proportional to classes
    widths = []
    for name in available:
        ncls = len(summary[name]["confusion_mean_pct"])
        widths.append(max(6, ncls * 0.9))
    total_width = sum(widths) + (n_plots - 1) * 1.0
    height = max(5, max((len(summary[name]["confusion_mean_pct"]) * 0.9 for name in available)))
    fig, axes = plt.subplots(1, n_plots, figsize=(total_width, height))
    if n_plots == 1:
        axes = [axes]

    for ax, subset_name in zip(axes, available):
        cm_mean = np.array(summary[subset_name]["confusion_mean_pct"])
        cm_std = np.array(summary[subset_name]["confusion_std_pct"])
        n = cm_mean.shape[0]
        annot = np.empty((n, n), dtype=object)
        for r in range(n):
            for c in range(n):
                annot[r, c] = f"{cm_mean[r,c]:.1f}\n±{cm_std[r,c]:.1f}"
        sns.heatmap(cm_mean, annot=annot, fmt="", cmap="cividis", ax=ax,
                    cbar_kws={'label':'% (row-normalized mean)'},
                    linewidths=0.5, linecolor='gray', annot_kws={"size":10})
        ax.set_title(f"{subset_name} (mean ± std %)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        # set class names
        if subset_name == "vocales":
            cls_names = VOCAL_CLASS_NAMES
        else:
            cls_names = COMANDO_CLASS_NAMES
        ax.set_xticklabels(cls_names, rotation=45, ha='right')
        ax.set_yticklabels(cls_names, rotation=0)

    fig.suptitle("Confusion matrices (mean on top line, std on bottom line) - vocales y comandos")
    fname_cm_all = RESULTS_DIR / "confusion_meanstd_vocales_comandos.png"
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(fname_cm_all, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved combined confusion matrices -> {fname_cm_all}")

# --------------------------
# Guardar resumen JSON
# --------------------------
summary_path = RESULTS_DIR / "summary_results.json"
with open(summary_path, "w", encoding="utf8") as fh:
    json.dump(summary, fh, indent=2)
print(f"\nSaved summary JSON -> {summary_path}")

print("\nFIN. Las figuras están en:", RESULTS_DIR)
