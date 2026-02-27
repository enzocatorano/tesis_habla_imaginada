# src/visualize_results.py
"""
Script de visualización y resumen de resultados de experimentos EEG.

Genera dos tipos de análisis:
1. INTRASUJETO: Por cada sujeto individual
   - Learning curves (train/val loss + val acc)
   - Matrices de confusión (mean ± std)
   - Boxplots de P/R/F1
   - Boxplots por banda de frecuencia
   - Boxplots por uso de FT Surrogate

2. INTERSUJETO: Agregado entre todos los sujetos
   - Boxplots de P/R/F1 entre sujetos
   - Matrices de confusión globales
   - Boxplots por banda (global)
   - Boxplots por FT Surrogate (global)

Estructura de salida:
  experiments/<EXPERIMENT>/visualization_results/
    ├── <subject>/
    │   ├── learning_curves.png
    │   ├── confusion_matrices.png
    │   ├── metrics_boxplots.png
    │   ├── band_accuracy_boxplots.png
    │   ├── ftsurrogate_accuracy_boxplots.png
    │   └── summary.json
    └── global/
        ├── metrics_boxplots_global.png
        ├── confusion_matrices_global.png
        ├── band_accuracy_boxplots_global.png
        ├── ftsurrogate_accuracy_boxplots_global.png
        └── global_summary.json
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple

# ---------------- CONFIG ----------------
EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1] / "experiments"
EXPERIMENT_NAME_PREFIX = "EEGNet_full_baseline"
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "preproc_aug_segm_gnperband_fts"
OUTPUT_SUBDIR = "visualization_results"

BAND_LABELS = ["delta", "theta", "alpha", "beta", "gamma"]
VOCAL_CLASS_NAMES = ['A', 'E', 'I', 'O', 'U']
COMANDO_CLASS_NAMES = ['Arriba', 'Abajo', 'Izquierda', 'Derecha', 'Adelante', 'Atras']
SUBSETS = ["vocales", "comandos"]
METRICS_NAMES = ["Precision", "Recall", "F1-Score"]

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 200
# ----------------------------------------


def find_latest_experiment(root: Path, prefix: str) -> Path:
    """Encuentra el experimento más reciente con el prefijo dado."""
    candidates = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not candidates:
        raise FileNotFoundError(f"No experiment folders starting with '{prefix}' found under {root}")
    return sorted(candidates, key=lambda p: p.name)[-1]


def load_json_safe(path: Path):
    """Carga un JSON de forma segura."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf8"))
    except Exception:
        return None


def save_json_safe(path: Path, obj):
    """Guarda un JSON de forma segura."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf8") as fh:
        json.dump(obj, fh, indent=2, default=lambda o: (o.tolist() if isinstance(o, np.ndarray) else str(o)))


def pad_and_aggregate_series(list_of_lists: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Agrega series de diferentes longitudes rellenando con NaN.
    Retorna (mean, std) arrays.
    """
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


def collect_fold_data(subject_dir: Path, subset: str, data_npz_path: Path):
    """
    Recopila datos de todos los folds/seeds para un sujeto y subset específico.
    
    Returns:
        dict con:
            - metrics: lista de dicts {precision, recall, f1}
            - confusion_matrices: lista de matrices (normalizadas por fila)
            - band_accuracies: lista de arrays (n_bands,)
            - ftsurrogate_accuracies: lista de dicts {with_fts, without_fts}
            - train_losses: lista de listas
            - val_losses: lista de listas
            - val_accs: lista de listas
    """
    subset_dir = subject_dir / subset
    if not subset_dir.exists():
        return None
    
    # Cargar datos del sujeto para extraer bandas y ftsurrogate
    subj_data = None
    banda_all = None
    ftsurrogate_all = None
    
    if data_npz_path.exists():
        try:
            subj_data = np.load(data_npz_path, allow_pickle=True)
            Y_all = subj_data['labels']
            banda_all = Y_all[:, 3].astype(int)  # 4ta columna (índice 3)
            ftsurrogate_all = Y_all[:, 4].astype(int)  # 5ta columna (índice 4)
        except Exception as e:
            print(f"  Warning: Could not load subject data: {e}")
    
    metrics_list = []
    cm_list = []
    band_acc_list = []
    fts_acc_list = []
    train_loss_list = []
    val_loss_list = []
    val_acc_list = []
    
    # Iterar por seeds
    seed_dirs = sorted([p for p in subset_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")])
    if not seed_dirs:
        seed_dirs = [subset_dir]  # Fallback si no hay estructura de seeds
    
    for seed_dir in seed_dirs:
        # Iterar por folds
        fold_dirs = sorted([p for p in seed_dir.iterdir() if p.is_dir() and p.name.startswith("fold_")])
        
        for fold_dir in fold_dirs:
            # Cargar metadata
            meta = load_json_safe(fold_dir / "metadata.json")
            if meta is None or meta.get("status") != "success":
                continue
            
            # 1. Métricas de clasificación
            cr = load_json_safe(fold_dir / "classification_report.json")
            if cr:
                macro = cr.get("macro avg") or cr.get("macro_avg") or cr.get("macro-average")
                if macro:
                    metrics_list.append({
                        "precision": float(macro.get("precision", np.nan)),
                        "recall": float(macro.get("recall", np.nan)),
                        "f1": float(macro.get("f1-score", macro.get("f1_score", np.nan)))
                    })
            
            # 2. Matriz de confusión
            cm_path = fold_dir / "confusion_matrix.npy"
            if cm_path.exists():
                try:
                    cm = np.load(cm_path)
                    # Normalizar por filas
                    with np.errstate(divide='ignore', invalid='ignore'):
                        row_sums = cm.sum(axis=1, keepdims=True)
                        cm_pct = np.divide(cm, row_sums, where=(row_sums != 0)) * 100.0
                        cm_pct = np.nan_to_num(cm_pct, nan=0.0)
                    cm_list.append(cm_pct)
                except Exception as e:
                    print(f"  Warning: Error loading confusion matrix: {e}")
            
            # 3. Band accuracy
            if banda_all is not None:
                test_idx_global = meta.get("test_idx_global")
                preds_path = fold_dir / "test_preds.npz"
                
                if test_idx_global is not None and preds_path.exists():
                    try:
                        d = np.load(preds_path, allow_pickle=True)
                        y_true = d["y_true"]
                        y_pred = d["y_pred"]
                        idxs = np.array(test_idx_global, dtype=int)
                        
                        bands_test = banda_all[idxs]
                        band_accs = []
                        
                        for b in range(len(BAND_LABELS)):
                            mask = (bands_test == b)
                            if mask.sum() == 0:
                                band_accs.append(np.nan)
                            else:
                                band_accs.append(float(np.mean(y_true[mask] == y_pred[mask])))
                        
                        band_acc_list.append(np.array(band_accs))
                    except Exception as e:
                        print(f"  Warning: Error computing band accuracy: {e}")
            
            # 4. FT Surrogate accuracy
            if ftsurrogate_all is not None:
                test_idx_global = meta.get("test_idx_global")
                preds_path = fold_dir / "test_preds.npz"
                
                if test_idx_global is not None and preds_path.exists():
                    try:
                        d = np.load(preds_path, allow_pickle=True)
                        y_true = d["y_true"]
                        y_pred = d["y_pred"]
                        idxs = np.array(test_idx_global, dtype=int)
                        
                        fts_test = ftsurrogate_all[idxs]
                        
                        mask_with = (fts_test == 1)
                        mask_without = (fts_test == 0)
                        
                        acc_with = float(np.mean(y_true[mask_with] == y_pred[mask_with])) if mask_with.sum() > 0 else np.nan
                        acc_without = float(np.mean(y_true[mask_without] == y_pred[mask_without])) if mask_without.sum() > 0 else np.nan
                        
                        fts_acc_list.append({"with_fts": acc_with, "without_fts": acc_without})
                    except Exception as e:
                        print(f"  Warning: Error computing FTS accuracy: {e}")
            
            # 5. Learning curves
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
        "val_accs": val_acc_list
    }


def plot_learning_curves(data_vocales, data_comandos, output_path: Path, subject_name: str):
    """Genera figura de learning curves para vocales y comandos."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
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
        
        # Eje izquierdo: Losses
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
        ax.tick_params(axis='y')
        
        # Eje derecho: Accuracy
        ax2 = ax.twinx()
        color_acc = "tab:green"
        
        if va_mean.size > 0:
            ax2.plot(epochs[:va_mean.size], va_mean, color=color_acc, linestyle='--', 
                    label="Val Accuracy", linewidth=2)
            ax2.fill_between(epochs[:va_mean.size], va_mean - va_std, va_mean + va_std, 
                            alpha=0.15, color=color_acc)
        
        ax2.set_ylabel("Val Accuracy", fontsize=11)
        ax2.tick_params(axis='y')
        
        # Leyenda combinada
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
        
        ax.set_title(f"Learning Curves - {subset_name}", fontsize=12, fontweight='bold')
    
    axes[-1].set_xlabel("Epoch", fontsize=11)
    fig.suptitle(f"Learning Curves - {subject_name}", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_confusion_matrices(data_vocales, data_comandos, output_path: Path, subject_name: str):
    """Genera figura de matrices de confusión para vocales y comandos."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, data, subset_name, class_names in zip(
        axes, 
        [data_vocales, data_comandos], 
        ["Vocales", "Comandos"],
        [VOCAL_CLASS_NAMES, COMANDO_CLASS_NAMES]
    ):
        if data is None or not data["confusion_matrices"]:
            ax.text(0.5, 0.5, f"No data for {subset_name}", ha='center', va='center')
            ax.axis('off')
            continue
        
        cm_stack = np.stack(data["confusion_matrices"], axis=0)
        cm_mean = np.nanmean(cm_stack, axis=0)
        cm_std = np.nanstd(cm_stack, axis=0)
        
        n = cm_mean.shape[0]
        annot = np.empty((n, n), dtype=object)
        
        for r in range(n):
            for c in range(n):
                annot[r, c] = f"{cm_mean[r, c]:.1f}\n±{cm_std[r, c]:.1f}"
        
        sns.heatmap(cm_mean, annot=annot, fmt="", cmap="cividis", ax=ax,
                   cbar_kws={'label': '% (normalizado por fila)'}, 
                   linewidths=0.5, linecolor='gray',
                   annot_kws={"size": 9})
        
        ax.set_title(f"{subset_name} (media ± std %)", fontsize=13, fontweight='bold')
        ax.set_xlabel("Predicha", fontsize=11)
        ax.set_ylabel("Verdadera", fontsize=11)
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=11)
        ax.set_yticklabels(class_names, rotation=0, fontsize=11)
    
    fig.suptitle(f"Matrices de confusion - {subject_name}", fontsize=15, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_metrics_boxplots(data_vocales, data_comandos, output_path: Path, subject_name: str):
    """Genera boxplots de P/R/F1 para vocales y comandos."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    all_data = []
    labels = []
    positions = []
    colors = []
    
    color_map = {"Precision": "#1f77b4", "Recall": "#ff7f0e", "F1-Score": "#2ca02c"}
    
    pos = 0
    for subset_name, data in [("Vocales", data_vocales), ("Comandos", data_comandos)]:
        if data is None or not data["metrics"]:
            pos += 4
            continue
        
        metrics_arr = np.array([[m["precision"], m["recall"], m["f1"]] for m in data["metrics"]])
        
        for i, metric_name in enumerate(METRICS_NAMES):
            all_data.append(metrics_arr[:, i])
            labels.append(f"{subset_name}\n{metric_name}")
            positions.append(pos)
            colors.append(color_map[metric_name])
            pos += 1
        
        pos += 1  # Espacio entre vocales y comandos
    
    if not all_data:
        ax.text(0.5, 0.5, "No data available", ha='center', va='center')
        ax.axis('off')
    else:
        bp = ax.boxplot(all_data, positions=positions, widths=0.6, patch_artist=True,
                       showmeans=True, meanline=True,
                       boxprops=dict(linewidth=1.5),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5),
                       medianprops=dict(linewidth=2, color='red'),
                       meanprops=dict(linewidth=2, color='black', linestyle='--'))

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Score", fontsize=11)
        # Límites verticales dinámicos: 0.9 * min, 1.1 * max
        data_concat = np.concatenate(all_data)
        min_val = float(np.min(data_concat))
        max_val = float(np.max(data_concat))
        ax.set_ylim(0.9 * min_val, 1.1 * max_val)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_title(f"Distribucion de metricas - {subject_name}", fontsize=14, fontweight='bold')
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_band_accuracy_boxplots(data_vocales, data_comandos, output_path: Path, subject_name: str):
    """Genera boxplots de accuracy por banda para vocales y comandos."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    band_freq_labels = ["delta\n0-4 Hz", "theta\n4-8 Hz", "alpha\n8-12 Hz", "beta\n12-32 Hz", "gamma\n32-64 Hz"]
    
    for ax, data, subset_name, n_classes in zip(axes, [data_vocales, data_comandos], ["Vocales", "Comandos"], [5, 6]):
        if data is None or not data["band_accuracies"]:
            ax.text(0.5, 0.5, f"No data for {subset_name}", ha='center', va='center')
            ax.axis('off')
            continue
        
        band_arr = np.vstack(data["band_accuracies"])  # (n_folds, n_bands)
        
        bp = ax.boxplot([band_arr[:, i] for i in range(len(BAND_LABELS))],
                       labels=band_freq_labels, patch_artist=True,
                       showmeans=True, meanline=True,
                       boxprops=dict(facecolor='skyblue', alpha=0.7, linewidth=1.5),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5),
                       medianprops=dict(linewidth=2, color='red'),
                       meanprops=dict(linewidth=2, color='black', linestyle='--'))
        
        # Límites dinámicos
        chance_level = 1.0 / n_classes
        data_min = np.nanmin(band_arr)
        data_max = np.nanmax(band_arr)
        y_min = max(0, min(chance_level * 0.9, data_min * 0.9))
        y_max = min(1, max(chance_level * 0.9, data_max * 1.1))
        ax.set_ylim(y_min, y_max)
        
        # Línea de probabilidad base
        ax.axhline(y=chance_level, color='red', linestyle=':', linewidth=2, 
                  label=f'Chance level ({chance_level:.3f})', alpha=0.7)
        
        ax.set_xlabel("Frequency Band", fontsize=10)
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_title(f"{subset_name}", fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=0, labelsize=8)
        ax.legend(loc='best', fontsize=8)
    
    fig.suptitle(f"Accuracy by Frequency Band - {subject_name}", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_ftsurrogate_accuracy_boxplots(data_vocales, data_comandos, output_path: Path, subject_name: str):
    """Genera boxplots de accuracy por uso de FT Surrogate para vocales y comandos."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax, data, subset_name, n_classes in zip(axes, [data_vocales, data_comandos], ["Vocales", "Comandos"], [5, 6]):
        if data is None or not data["ftsurrogate_accuracies"]:
            ax.text(0.5, 0.5, f"No data for {subset_name}", ha='center', va='center')
            ax.axis('off')
            continue
        
        with_fts = [d["with_fts"] for d in data["ftsurrogate_accuracies"] if not np.isnan(d["with_fts"])]
        without_fts = [d["without_fts"] for d in data["ftsurrogate_accuracies"] if not np.isnan(d["without_fts"])]
        
        if not with_fts and not without_fts:
            ax.text(0.5, 0.5, f"No FTS data for {subset_name}", ha='center', va='center')
            ax.axis('off')
            continue
        
        data_to_plot = []
        labels_to_plot = []
        
        if without_fts:
            data_to_plot.append(without_fts)
            labels_to_plot.append("Without FTS")
        
        if with_fts:
            data_to_plot.append(with_fts)
            labels_to_plot.append("With FTS")
        
        bp = ax.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True,
                       showmeans=True, meanline=True,
                       boxprops=dict(facecolor='lightcoral', alpha=0.7, linewidth=1.5),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5),
                       medianprops=dict(linewidth=2, color='darkred'),
                       meanprops=dict(linewidth=2, color='black', linestyle='--'))
        
        # Límites dinámicos
        chance_level = 1.0 / n_classes
        all_data = without_fts + with_fts
        data_min = np.nanmin(all_data)
        data_max = np.nanmax(all_data)
        y_min = max(0, min(chance_level * 0.9, data_min * 0.9))
        y_max = min(1, max(chance_level * 0.9, data_max * 1.1))
        ax.set_ylim(y_min, y_max)
        
        # Línea de probabilidad base
        ax.axhline(y=chance_level, color='red', linestyle=':', linewidth=2,
                  label=f'Chance level ({chance_level:.3f})', alpha=0.7)
        
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_title(f"{subset_name}", fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
    
    fig.suptitle(f"Accuracy by FT Surrogate Usage - {subject_name}", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def process_subject(subject_dir: Path, data_dir: Path, output_root: Path):
    """Procesa un sujeto individual y genera todas las visualizaciones."""
    subject_name = subject_dir.name
    print(f"\n[Visualizer] Processing subject: {subject_name}")
    
    # Buscar archivo .npz del sujeto
    subj_npz_path = data_dir / f"{subject_name}.npz"
    if not subj_npz_path.exists():
        matches = list(data_dir.glob(f"{subject_name.split('_')[0]}*.npz"))
        subj_npz_path = matches[0] if matches else None
    
    if subj_npz_path is None or not subj_npz_path.exists():
        print(f"  Warning: Could not find data file for {subject_name}")
        subj_npz_path = data_dir / "dummy.npz"  # Placeholder
    
    # Crear directorio de salida
    subject_out = output_root / subject_name
    subject_out.mkdir(parents=True, exist_ok=True)
    
    # Recopilar datos
    data_vocales = collect_fold_data(subject_dir, "vocales", subj_npz_path)
    data_comandos = collect_fold_data(subject_dir, "comandos", subj_npz_path)
    
    if data_vocales is None and data_comandos is None:
        print(f"  No data found for {subject_name}, skipping...")
        return None
    
    # Generar visualizaciones
    plot_learning_curves(data_vocales, data_comandos, 
                        subject_out / "learning_curves.png", subject_name)
    
    plot_confusion_matrices(data_vocales, data_comandos,
                           subject_out / "confusion_matrices.png", subject_name)
    
    plot_metrics_boxplots(data_vocales, data_comandos,
                         subject_out / "metrics_boxplots.png", subject_name)
    
    plot_band_accuracy_boxplots(data_vocales, data_comandos,
                               subject_out / "band_accuracy_boxplots.png", subject_name)
    
    plot_ftsurrogate_accuracy_boxplots(data_vocales, data_comandos,
                                      subject_out / "ftsurrogate_accuracy_boxplots.png", subject_name)
    
    # Guardar resumen JSON
    summary = {}
    for subset_name, data in [("vocales", data_vocales), ("comandos", data_comandos)]:
        if data is None:
            continue
        
        metrics_arr = np.array([[m["precision"], m["recall"], m["f1"]] for m in data["metrics"]]) if data["metrics"] else np.array([])
        
        summary[subset_name] = {
            "n_folds": len(data["metrics"]),
            "precision_mean": float(np.nanmean(metrics_arr[:, 0])) if metrics_arr.size else None,
            "precision_std": float(np.nanstd(metrics_arr[:, 0])) if metrics_arr.size else None,
            "recall_mean": float(np.nanmean(metrics_arr[:, 1])) if metrics_arr.size else None,
            "recall_std": float(np.nanstd(metrics_arr[:, 1])) if metrics_arr.size else None,
            "f1_mean": float(np.nanmean(metrics_arr[:, 2])) if metrics_arr.size else None,
            "f1_std": float(np.nanstd(metrics_arr[:, 2])) if metrics_arr.size else None,
        }
    
    save_json_safe(subject_out / "summary.json", summary)
    print(f"  Saved: {subject_out / 'summary.json'}")
    
    return summary


def plot_global_metrics_boxplots(all_subjects_data: Dict, output_path: Path):
    """Genera boxplots globales de P/R/F1 entre sujetos."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    all_data = []
    labels = []
    positions = []
    colors = []
    
    color_map = {"Precision": "#1f77b4", "Recall": "#ff7f0e", "F1-Score": "#2ca02c"}
    
    pos = 0
    for subset_name in SUBSETS:
        # Recopilar datos de todos los sujetos para este subset
        prec_list = []
        rec_list = []
        f1_list = []
        
        for subj_name, subj_summary in all_subjects_data.items():
            if subj_summary is None:
                continue
            subset_data = subj_summary.get(subset_name)
            if subset_data is None:
                continue
            
            if subset_data["precision_mean"] is not None:
                prec_list.append(subset_data["precision_mean"])
            if subset_data["recall_mean"] is not None:
                rec_list.append(subset_data["recall_mean"])
            if subset_data["f1_mean"] is not None:
                f1_list.append(subset_data["f1_mean"])
        
        if not prec_list and not rec_list and not f1_list:
            pos += 4
            continue
        
        for metric_data, metric_name in [(prec_list, "Precision"), (rec_list, "Recall"), (f1_list, "F1-Score")]:
            if metric_data:
                all_data.append(metric_data)
                labels.append(f"{subset_name.capitalize()}\n{metric_name}")
                positions.append(pos)
                colors.append(color_map[metric_name])
            pos += 1
        
        pos += 1  # Espacio entre subsets
    
    if not all_data:
        ax.text(0.5, 0.5, "No data available", ha='center', va='center')
        ax.axis('off')
    else:
        bp = ax.boxplot(all_data, positions=positions, widths=0.6, patch_artist=True,
                       showmeans=True, meanline=True,
                       boxprops=dict(linewidth=1.5),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5),
                       medianprops=dict(linewidth=2, color='red'),
                       meanprops=dict(linewidth=2, color='black', linestyle='--'))
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Score", fontsize=11)
        # Límites verticales dinámicos: 0.9 * min, 1.1 * max
        data_concat = np.concatenate([np.array(d) for d in all_data])
        min_val = float(np.min(data_concat))
        max_val = float(np.max(data_concat))
        ax.set_ylim(0.9 * min_val, 1.1 * max_val)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_title("Global Metrics Distribution (Across Subjects)", fontsize=14, fontweight='bold')
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_global_confusion_matrices(all_subjects_data_raw: Dict, output_path: Path):
    """Genera matrices de confusión globales promediadas entre sujetos."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, subset_name, class_names in zip(
        axes,
        SUBSETS,
        [VOCAL_CLASS_NAMES, COMANDO_CLASS_NAMES]
    ):
        cm_list = []
        
        for subj_name, subj_data in all_subjects_data_raw.items():
            subset_data = subj_data.get(subset_name)
            if subset_data is None or not subset_data["confusion_matrices"]:
                continue
            
            # Promediar matrices de este sujeto
            cm_stack = np.stack(subset_data["confusion_matrices"], axis=0)
            cm_mean_subj = np.nanmean(cm_stack, axis=0)
            cm_list.append(cm_mean_subj)
        
        if not cm_list:
            ax.text(0.5, 0.5, f"No data for {subset_name}", ha='center', va='center')
            ax.axis('off')
            continue
        
        cm_global_stack = np.stack(cm_list, axis=0)
        cm_mean = np.nanmean(cm_global_stack, axis=0)
        cm_std = np.nanstd(cm_global_stack, axis=0)
        
        n = cm_mean.shape[0]
        annot = np.empty((n, n), dtype=object)
        
        for r in range(n):
            for c in range(n):
                annot[r, c] = f"{cm_mean[r, c]:.1f}\n±{cm_std[r, c]:.1f}"
        
        sns.heatmap(cm_mean, annot=annot, fmt="", cmap="cividis", ax=ax,
                   cbar_kws={'label': '% (row-normalized)'},
                   linewidths=0.5, linecolor='gray',
                   annot_kws={"size": 9})
        
        ax.set_title(f"{subset_name.capitalize()} (mean ± std %)", fontsize=12, fontweight='bold')
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True", fontsize=10)
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(class_names, rotation=0, fontsize=9)
    
    fig.suptitle("Global Confusion Matrices (Across Subjects)", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_global_band_accuracy_boxplots(all_subjects_data_raw: Dict, output_path: Path):
    """Genera boxplots globales de accuracy por banda entre sujetos."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    band_freq_labels = ["delta\n0-4 Hz", "theta\n4-8 Hz", "alpha\n8-12 Hz", "beta\n12-32 Hz", "gamma\n32-64 Hz"]
    
    for ax, subset_name, n_classes in zip(axes, SUBSETS, [5, 6]):
        band_data_per_subject = []
        
        for subj_name, subj_data in all_subjects_data_raw.items():
            subset_data = subj_data.get(subset_name)
            if subset_data is None or not subset_data["band_accuracies"]:
                continue
            
            # Promediar bandas de este sujeto a través de folds
            band_arr = np.vstack(subset_data["band_accuracies"])
            band_mean_subj = np.nanmean(band_arr, axis=0)
            band_data_per_subject.append(band_mean_subj)
        
        if not band_data_per_subject:
            ax.text(0.5, 0.5, f"No data for {subset_name}", ha='center', va='center')
            ax.axis('off')
            continue
        
        band_matrix = np.vstack(band_data_per_subject)  # (n_subjects, n_bands)
        
        bp = ax.boxplot([band_matrix[:, i] for i in range(len(BAND_LABELS))],
                       labels=band_freq_labels, patch_artist=True,
                       showmeans=True, meanline=True,
                       boxprops=dict(facecolor='skyblue', alpha=0.7, linewidth=1.5),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5),
                       medianprops=dict(linewidth=2, color='red'),
                       meanprops=dict(linewidth=2, color='black', linestyle='--'))
        
        # Límites dinámicos
        chance_level = 1.0 / n_classes
        data_min = np.nanmin(band_matrix)
        data_max = np.nanmax(band_matrix)
        y_min = max(0, min(chance_level * 0.9, data_min * 0.9))
        y_max = min(1, max(chance_level * 0.9, data_max * 1.1))
        ax.set_ylim(y_min, y_max)
        
        # Línea de probabilidad base
        ax.axhline(y=chance_level, color='red', linestyle=':', linewidth=2,
                  label=f'Chance level ({chance_level:.3f})', alpha=0.7)
        
        ax.set_xlabel("Frequency Band", fontsize=10)
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_title(f"{subset_name.capitalize()}", fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=0, labelsize=8)
        ax.legend(loc='best', fontsize=8)
    
    fig.suptitle("Global Accuracy by Frequency Band (Across Subjects)", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_global_ftsurrogate_accuracy_boxplots(all_subjects_data_raw: Dict, output_path: Path):
    """Genera boxplots globales de accuracy por FT Surrogate entre sujetos."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax, subset_name, n_classes in zip(axes, SUBSETS, [5, 6]):
        with_fts_per_subject = []
        without_fts_per_subject = []
        
        for subj_name, subj_data in all_subjects_data_raw.items():
            subset_data = subj_data.get(subset_name)
            if subset_data is None or not subset_data["ftsurrogate_accuracies"]:
                continue
            
            # Promediar FTS accuracy de este sujeto
            with_fts = [d["with_fts"] for d in subset_data["ftsurrogate_accuracies"] if not np.isnan(d["with_fts"])]
            without_fts = [d["without_fts"] for d in subset_data["ftsurrogate_accuracies"] if not np.isnan(d["without_fts"])]
            
            if with_fts:
                with_fts_per_subject.append(np.nanmean(with_fts))
            if without_fts:
                without_fts_per_subject.append(np.nanmean(without_fts))
        
        if not with_fts_per_subject and not without_fts_per_subject:
            ax.text(0.5, 0.5, f"No data for {subset_name}", ha='center', va='center')
            ax.axis('off')
            continue
        
        data_to_plot = []
        labels_to_plot = []
        
        if without_fts_per_subject:
            data_to_plot.append(without_fts_per_subject)
            labels_to_plot.append("Without FTS")
        
        if with_fts_per_subject:
            data_to_plot.append(with_fts_per_subject)
            labels_to_plot.append("With FTS")
        
        bp = ax.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True,
                       showmeans=True, meanline=True,
                       boxprops=dict(facecolor='lightcoral', alpha=0.7, linewidth=1.5),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5),
                       medianprops=dict(linewidth=2, color='darkred'),
                       meanprops=dict(linewidth=2, color='black', linestyle='--'))
        
        # Límites dinámicos
        chance_level = 1.0 / n_classes
        all_data = without_fts_per_subject + with_fts_per_subject
        data_min = np.nanmin(all_data)
        data_max = np.nanmax(all_data)
        y_min = max(0, min(chance_level * 0.9, data_min * 0.9))
        y_max = min(1, max(chance_level * 0.9, data_max * 1.1))
        ax.set_ylim(y_min, y_max)
        
        # Línea de probabilidad base
        ax.axhline(y=chance_level, color='red', linestyle=':', linewidth=2,
                  label=f'Chance level ({chance_level:.3f})', alpha=0.7)
        
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_title(f"{subset_name.capitalize()}", fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
    
    fig.suptitle("Global Accuracy by FT Surrogate Usage (Across Subjects)", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ==================== MAIN ====================

if __name__ == "__main__":
    # Encontrar experimento
    EXP_ROOT = find_latest_experiment(EXPERIMENTS_ROOT, EXPERIMENT_NAME_PREFIX)
    print(f"[Visualizer] Using experiment root: {EXP_ROOT}")
    
    OUTPUT_ROOT = EXP_ROOT / OUTPUT_SUBDIR
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    # Descubrir sujetos
    subject_dirs = sorted([p for p in EXP_ROOT.iterdir() 
                          if p.is_dir() and p.name.upper().startswith("S")])
    print(f"[Visualizer] Found {len(subject_dirs)} subject directories")
    
    # Procesar cada sujeto
    all_subjects_summaries = {}
    all_subjects_data_raw = {}
    
    for subject_dir in subject_dirs:
        subject_name = subject_dir.name
        summary = process_subject(subject_dir, DATA_DIR, OUTPUT_ROOT)
        
        if summary is not None:
            all_subjects_summaries[subject_name] = summary
            
            # Guardar datos raw para análisis global
            subj_npz_path = DATA_DIR / f"{subject_name}.npz"
            if not subj_npz_path.exists():
                matches = list(DATA_DIR.glob(f"{subject_name.split('_')[0]}*.npz"))
                subj_npz_path = matches[0] if matches else DATA_DIR / "dummy.npz"
            
            all_subjects_data_raw[subject_name] = {
                "vocales": collect_fold_data(subject_dir, "vocales", subj_npz_path),
                "comandos": collect_fold_data(subject_dir, "comandos", subj_npz_path)
            }
    
    # Generar visualizaciones globales
    print("\n[Visualizer] Generating global visualizations...")
    global_out = OUTPUT_ROOT / "global"
    global_out.mkdir(parents=True, exist_ok=True)
    
    plot_global_metrics_boxplots(all_subjects_summaries, 
                                 global_out / "metrics_boxplots_global.png")
    
    plot_global_confusion_matrices(all_subjects_data_raw,
                                   global_out / "confusion_matrices_global.png")
    
    plot_global_band_accuracy_boxplots(all_subjects_data_raw,
                                       global_out / "band_accuracy_boxplots_global.png")
    
    plot_global_ftsurrogate_accuracy_boxplots(all_subjects_data_raw,
                                             global_out / "ftsurrogate_accuracy_boxplots_global.png")
    
    # Guardar resumen global
    global_summary = {
        "experiment_root": str(EXP_ROOT),
        "n_subjects": len(all_subjects_summaries),
        "subjects": list(all_subjects_summaries.keys()),
        "per_subject_summaries": all_subjects_summaries
    }
    
    save_json_safe(global_out / "global_summary.json", global_summary)
    print(f"  Saved: {global_out / 'global_summary.json'}")
    
    print(f"\n[Visualizer] Done! Results saved to: {OUTPUT_ROOT}")