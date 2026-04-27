# src/visualize_results_online_augmentation.py
"""
Script de visualización y resumen de resultados para experimentos con augmentación online.

ADAPTACIÓN:
- Eliminadas las visualizaciones de FTS, Bandas y Ventanas Temporales ya que
  en la arquitectura online rigurosa, el Test Set es inmutable y limpio.
- Se enfoca en métricas de rendimiento core: Curvas de aprendizaje, Matrices de Confusión y Boxplots de Métricas.

Genera dos tipos de análisis:
1. INTRASUJETO: Learning curves, Matrices de confusión, Boxplots P/R/F1.
2. INTERSUJETO: Boxplots P/R/F1 (global), Matrices de confusión (global).
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple

# ---------------- CONFIG ----------------
EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1] / "experiments"
EXPERIMENT_NAME_PREFIX = "1024_clean_EEGNet_20260427-055419_CatoranoBrothers"
OUTPUT_SUBDIR = "visualization_results"

VOCAL_CLASS_NAMES = ['A', 'E', 'I', 'O', 'U']
COMANDO_CLASS_NAMES = ['Arriba', 'Abajo', 'Izquierda', 'Derecha', 'Adelante', 'Atras']
SUBSETS = ["vocales", "comandos"]
METRICS_NAMES = ["Precision", "Recall", "F1-Score"]

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 200
COLORS = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51']
# ----------------------------------------


def find_latest_experiment(root: Path, prefix: str) -> Path:
    candidates = [p for p in root.iterdir() if p.is_dir() and prefix in p.name]
    if not candidates:
        raise FileNotFoundError(f"No experiment folders containing '{prefix}' found under {root}")
    return sorted(candidates, key=lambda p: p.name)[-1]


def load_json_safe(path: Path):
    if not path.exists():
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
    
    lengths = [len(l) for l in list_of_lists if l is not None]
    if not lengths:
        return np.array([]), np.array([])
        
    max_len = max(lengths)
    
    arr = np.full((len(list_of_lists), max_len), np.nan, dtype=float)
    for i, l in enumerate(list_of_lists):
        if l is not None and len(l) > 0:
            arr[i, :len(l)] = np.array(l, dtype=float)
    
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    
    return mean, std


def collect_fold_data(subject_dir: Path, subset: str):
    subset_dir = subject_dir / subset
    if not subset_dir.exists():
        return None
    
    metrics_list = []
    cm_list = []
    train_loss_list = []
    val_loss_list = []
    val_acc_list = []
    
    seed_dirs = sorted([p for p in subset_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")])
    if not seed_dirs:
        seed_dirs = [subset_dir]
    
    for seed_dir in seed_dirs:
        fold_dirs = sorted([p for p in seed_dir.iterdir() if p.is_dir() and p.name.startswith("fold_")])
        
        for fold_dir in fold_dirs:
            meta = load_json_safe(fold_dir / "metadata.json")
            if meta is None or meta.get("status") != "success":
                continue
            
            # 1. Métricas
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
                    with np.errstate(divide='ignore', invalid='ignore'):
                        row_sums = cm.sum(axis=1, keepdims=True)
                        cm_pct = np.divide(cm, row_sums, where=(row_sums != 0)) * 100.0
                        cm_pct = np.nan_to_num(cm_pct, nan=0.0)
                    cm_list.append(cm_pct)
                except Exception as e:
                    print(f"  Warning: Error loading confusion matrix: {e}")
            
            # 3. Learning curves
            train_metrics = load_json_safe(fold_dir / "train_metrics.json")
            if train_metrics:
                tl = train_metrics.get("train_losses")
                vl = train_metrics.get("val_losses")
                va = train_metrics.get("val_accs")
                
                if tl: train_loss_list.append(tl)
                if vl: val_loss_list.append(vl)
                if va: val_acc_list.append(va)
    
    return {
        "metrics": metrics_list,
        "confusion_matrices": cm_list,
        "train_losses": train_loss_list,
        "val_losses": val_loss_list,
        "val_accs": val_acc_list,
    }


def apply_custom_boxplot(ax, data, positions, labels, colors, widths=0.2):
    for i, (d, pos, col) in enumerate(zip(data, positions, colors)):
        if len(d) == 0: continue
        
        ax.boxplot(d, positions=[pos], widths=widths, 
                   showfliers=False, manage_ticks=False,
                   patch_artist=True,
                   medianprops=dict(color="orange", linewidth=1.5),
                   whiskerprops=dict(color=col, alpha=0.7),
                   capprops=dict(color=col, alpha=0.7),
                   boxprops=dict(facecolor=col, color=col, alpha=0.5))
        
        jitter = np.random.normal(pos, 0.02, size=len(d))
        ax.scatter(jitter, d, color=col, alpha=0.8, s=50, edgecolors='white', linewidths=0.5)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_learning_curves(data_vocales, data_comandos, output_path: Path, subject_name: str):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    for ax, data, subset_name in zip(axes, [data_vocales, data_comandos], ["Vocales", "Comandos"]):
        if data is None:
            ax.text(0.5, 0.5, f"No data for {subset_name}", ha='center', va='center')
            ax.axis('off')
            continue
        
        tl_mean, tl_std = pad_and_aggregate_series(data["train_losses"])
        vl_mean, vl_std = pad_and_aggregate_series(data["val_losses"])
        va_mean, va_std = pad_and_aggregate_series(data["val_accs"])
        
        if tl_mean.size == 0:
            ax.text(0.5, 0.5, f"No learning curves for {subset_name}", ha='center', va='center')
            ax.axis('off')
            continue
        
        epochs = np.arange(1, max(tl_mean.size, vl_mean.size, va_mean.size) + 1)
        
        color_train = "tab:blue"
        color_val = "tab:orange"
        
        ax.plot(epochs[:tl_mean.size], tl_mean, color=color_train, label="Train Loss", linewidth=2)
        ax.fill_between(epochs[:tl_mean.size], tl_mean - tl_std, tl_mean + tl_std, alpha=0.2, color=color_train)
        
        if vl_mean.size > 0 and not np.all(np.isnan(vl_mean)):
            ax.plot(epochs[:vl_mean.size], vl_mean, color=color_val, label="Val Loss", linewidth=2)
            ax.fill_between(epochs[:vl_mean.size], vl_mean - vl_std, vl_mean + vl_std, alpha=0.2, color=color_val)
        
        ax.set_ylabel("Loss", fontsize=11)
        ax.grid(True, alpha=0.3)
        
        ax2 = ax.twinx()
        color_acc = "tab:green"
        
        if va_mean.size > 0 and not np.all(np.isnan(va_mean)):
            ax2.plot(epochs[:va_mean.size], va_mean, color=color_acc, linestyle='--', label="Val Accuracy", linewidth=2)
            ax2.fill_between(epochs[:va_mean.size], va_mean - va_std, va_mean + va_std, alpha=0.15, color=color_acc)
        
        ax2.set_ylabel("Val Accuracy", fontsize=11)
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
        ax.set_title(f"Learning Curves - {subset_name}", fontsize=12, fontweight='bold')
    
    axes[-1].set_xlabel("Epoch", fontsize=11)
    fig.suptitle(f"Learning Curves - {subject_name}", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_confusion_matrices(data_vocales, data_comandos, output_path: Path, subject_name: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, data, subset_name, class_names in zip(
        axes, [data_vocales, data_comandos], ["Vocales", "Comandos"], [VOCAL_CLASS_NAMES, COMANDO_CLASS_NAMES]
    ):
        if data is None or not data["confusion_matrices"]:
            ax.text(0.5, 0.5, f"No data for {subset_name}", ha='center', va='center')
            ax.axis('off')
            continue
        
        cm_stack = np.stack(data["confusion_matrices"], axis=0)
        cm_mean = np.nanmean(cm_stack, axis=0)
        cm_std = np.nanstd(cm_stack, axis=0)
        
        n = cm_mean.shape[0]
        # Validar si las clases coinciden (Target IDX podría cambiar esto)
        if n != len(class_names):
            class_names = [f"Class {i}" for i in range(n)]

        annot = np.empty((n, n), dtype=object)
        for r in range(n):
            for c in range(n):
                annot[r, c] = f"{cm_mean[r, c]:.1f}\n±{cm_std[r, c]:.1f}"
        
        sns.heatmap(cm_mean, annot=annot, fmt="", cmap="cividis", ax=ax,
                   cbar_kws={'label': '% (row-normalized)'}, 
                   linewidths=0.5, linecolor='gray', annot_kws={"size": 9})
        
        ax.set_title(f"{subset_name} (mean ± std %)", fontsize=12, fontweight='bold')
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True", fontsize=10)
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(class_names, rotation=0, fontsize=9)
    
    fig.suptitle(f"Confusion Matrices - {subject_name}", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_metrics_boxplots(data_vocales, data_comandos, output_path: Path, subject_name: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    all_data, labels, positions, plot_colors = [], [], [], []
    pos = 0
    metric_colors = COLORS[:3] 
    
    for subset_name, data, n_classes in [("Vocales", data_vocales, 5), ("Comandos", data_comandos, 6)]:
        if data is None or not data["metrics"]:
            pos += 4
            continue
        
        metrics_arr = np.array([[m["precision"], m["recall"], m["f1"]] for m in data["metrics"]])
        
        for i, metric_name in enumerate(METRICS_NAMES):
            all_data.append(metrics_arr[:, i])
            labels.append(f"{subset_name}\n{metric_name}")
            positions.append(pos)
            plot_colors.append(metric_colors[i])
            pos += 1
        pos += 1
    
    if not all_data:
        ax.text(0.5, 0.5, "No data available", ha='center', va='center')
    else:
        apply_custom_boxplot(ax, all_data, positions, labels, plot_colors)
        ax.set_ylabel("Score")
        
        chance_level_vocales = 1.0 / 5  # 0.20
        chance_level_comandos = 1.0 / 6  # ~0.167
        
        # Agregar línea de chance para cada subset
        for subset_name in ["Vocales", "Comandos"]:
            # Encontrar la posición del primer y último boxplot de ese subset
            subset_positions = [p for p, l in zip(positions, labels) if l.startswith(subset_name)]
            if not subset_positions:
                continue
            
            chance = chance_level_vocales if subset_name == "Vocales" else chance_level_comandos
            ax.axhline(y=chance, color='gray', linestyle=':', linewidth=2, 
                      label=f'{subset_name} Chance (~{chance:.3f})')

        min_val = np.nanmin([np.nanmin(d) for d in all_data if d.size > 0])
        max_val = np.nanmax([np.nanmax(d) for d in all_data if d.size > 0])
        chance_min = min(chance_level_vocales, chance_level_comandos)  # 0.167
        ax.set_ylim(max(0, min(min_val, chance_min) - 0.05), min(1.0, max(max_val, 0.25) + 0.05))
        ax.set_title(f"Metrics Distribution - {subject_name}", fontweight='bold')
        ax.legend()
    
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def plot_global_confusion_matrices(all_subjects_data_raw: Dict, output_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, subset_name, class_names in zip(axes, SUBSETS, [VOCAL_CLASS_NAMES, COMANDO_CLASS_NAMES]):
        cm_list = []
        for subj_name, subj_data in all_subjects_data_raw.items():
            if subj_data is None: continue
            subset_data = subj_data.get(subset_name)
            if subset_data is None or not subset_data["confusion_matrices"]: continue
            
            cm_stack = np.stack(subset_data["confusion_matrices"], axis=0)
            cm_mean_subj = np.nanmean(cm_stack, axis=0)
            cm_list.append(cm_mean_subj)
        
        if not cm_list:
            ax.axis('off')
            continue
        
        cm_global_stack = np.stack(cm_list, axis=0)
        cm_mean = np.nanmean(cm_global_stack, axis=0)
        cm_std = np.nanstd(cm_global_stack, axis=0)
        
        n = cm_mean.shape[0]
        if n != len(class_names): class_names = [f"C{i}" for i in range(n)]

        annot = np.empty((n, n), dtype=object)
        for r in range(n):
            for c in range(n):
                annot[r, c] = f"{cm_mean[r, c]:.1f}\n±{cm_std[r, c]:.1f}"
        
        sns.heatmap(cm_mean, annot=annot, fmt="", cmap="cividis", ax=ax,
                   cbar_kws={'label': '% (row-normalized)'}, linewidths=0.5, linecolor='gray')
        
        ax.set_title(f"{subset_name.capitalize()} (mean ± std %)", fontweight='bold')
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names, rotation=0)
    
    fig.suptitle("Global Confusion Matrices (Across Subjects)", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_global_metrics_boxplots(all_subjects_data: Dict, output_path: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    all_data, labels, positions, plot_colors = [], [], [], []
    pos = 0

    for subset_name in SUBSETS:
        prec_list, rec_list, f1_list = [], [], []
        for subj_name, subj_data in all_subjects_data.items():
            s_sum = subj_data["summary"].get(subset_name)
            if s_sum:
                if s_sum["precision_mean"]: prec_list.append(s_sum["precision_mean"])
                if s_sum["recall_mean"]: rec_list.append(s_sum["recall_mean"])
                if s_sum["f1_mean"]: f1_list.append(s_sum["f1_mean"])
        
        for i, (m_data, m_name) in enumerate(zip([prec_list, rec_list, f1_list], METRICS_NAMES)):
            if m_data:
                all_data.append(m_data)
                labels.append(f"{subset_name}\n{m_name}")
                positions.append(pos)
                plot_colors.append(COLORS[i])
                pos += 1
        pos += 1

    if all_data:
        apply_custom_boxplot(ax, all_data, positions, labels, plot_colors)
        ax.set_ylabel("Score")
        
        # Agregar línea de chance separada para cada subset
        for subset_name in SUBSETS:
            subset_positions = [p for p, l in zip(positions, labels) if l.startswith(subset_name)]
            if not subset_positions:
                continue
            
            chance = 1.0 / 5 if subset_name == "vocales" else 1.0 / 6
            ax.axhline(y=chance, color='gray', linestyle=':', linewidth=2, 
                      label=f'{subset_name.capitalize()} Chance (~{chance:.3f})')
        
        min_val = np.nanmin([np.nanmin(d) for d in all_data if len(d) > 0])
        max_val = np.nanmax([np.nanmax(d) for d in all_data if len(d) > 0])
        chance_min = 1.0 / 6  # Usar el menor chance para el ylim
        ax.set_ylim(max(0, min(min_val, chance_min) - 0.05), min(1.0, max(max_val, 0.25) + 0.05))
        ax.set_title("Global Metrics Distribution (Across Subjects)", fontweight='bold')
        ax.legend()
    
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def plot_global_precision_by_subject(all_subjects_data: Dict, output_path: Path):
    """
    Genera boxplots de precisión por cada sujeto, con promedio entre sujetos a la derecha.
    Dos subplots: Vocales (arriba) y Comandos (abajo).
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
    
    # Extraer lista de sujetos ordenada
    subject_names = sorted([k for k in all_subjects_data.keys()])
    n_subjects = len(subject_names)
    
    for ax, subset_name in zip(axes, SUBSETS):
        # Recolectar precisión de cada pliegue para cada sujeto
        all_precision_per_subject = []  # Lista de listas: [ [prec_fold1, fold2, ...], ... ]
        
        for subj_name in subject_names:
            subj_data = all_subjects_data.get(subj_name)
            if subj_data is None:
                all_precision_per_subject.append([])
                continue
            
            subset_data = subj_data.get(subset_name)
            if subset_data is None or not subset_data.get("metrics"):
                all_precision_per_subject.append([])
                continue
            
            # Recolectar precisiones de todos los folds/seeds
            precisions = [m["precision"] for m in subset_data["metrics"] if "precision" in m]
            all_precision_per_subject.append(precisions)
        
        # Crear posiciones para boxplots: 15 sujetos + 1 para promedio
        positions = list(range(n_subjects))  # 0-14
        avg_position = n_subjects + 0.5  # Separado a la derecha
        
        # Calcular promedio por sujeto (media de precisiones de todos los folds)
        subject_means = []
        for prec_list in all_precision_per_subject:
            if prec_list:
                subject_means.append(np.mean(prec_list))
            else:
                subject_means.append(np.nan)
        
        # Preparar datos para boxplots
        box_data = []
        box_positions = []
        jitter_data = []
        jitter_positions = []
        
        # Boxplots individuales por sujeto
        for i, prec_list in enumerate(all_precision_per_subject):
            if prec_list:
                box_data.append(prec_list)
                box_positions.append(positions[i])
                # Puntos con jitter
                jitter_data.extend(prec_list)
                jitter_positions.extend([positions[i]] * len(prec_list))
        
        # Boxplot del promedio entre sujetos
        valid_means = [m for m in subject_means if not np.isnan(m)]
        if valid_means:
            box_data.append(valid_means)
            box_positions.append(avg_position)
            jitter_data.extend(valid_means)
            jitter_positions.extend([avg_position] * len(valid_means))
        
        # Graficar
        if box_data:
            # Boxplots
            bp = ax.boxplot(box_data, positions=box_positions, widths=0.3,
                           showfliers=False, manage_ticks=False,
                           patch_artist=True,
                           medianprops=dict(color="orange", linewidth=1.5),
                           whiskerprops=dict(color="#264653", alpha=0.7),
                           capprops=dict(color="#264653", alpha=0.7),
                           boxprops=dict(facecolor="#2a9d8f", color="#264653", alpha=0.5))
            
            # Scatter points con jitter
            jitter = np.random.normal(0, 0.04, size=len(jitter_data))
            ax.scatter(jitter_positions, jitter_data, color="#264653", alpha=0.6, s=30, 
                      edgecolors='white', linewidths=0.5, zorder=3)
        
        # Configurar ejes
        all_x = list(range(n_subjects)) + [avg_position]
        subject_labels = subject_names + ["AVG"]
        ax.set_xticks(all_x)
        ax.set_xticklabels(subject_labels, rotation=45, ha='right', fontsize=8)
        
        # Línea de chance
        chance = 1.0 / 5 if subset_name == "vocales" else 1.0 / 6
        ax.axhline(y=chance, color='red', linestyle='--', linewidth=2, 
                  label=f'Chance (~{chance:.3f})')
        
        ax.set_ylabel("Precision", fontsize=11)
        ax.set_title(f"{subset_name.capitalize()} - Precision by Subject", fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Ajustar ylim
        y_min = max(0, min(chance - 0.05, np.nanmin(jitter_data) if jitter_data else 0))
        y_max = min(1.0, max(chance + 0.15, np.nanmax(jitter_data) if jitter_data else 0.2))
        ax.set_ylim(y_min, y_max)
    
    axes[-1].set_xlabel("Subject", fontsize=11)
    fig.suptitle("Precision by Subject (with AVG)", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def process_subject(subject_dir: Path, output_root: Path):
    subject_name = subject_dir.name
    print(f"\n[Visualizer] Processing subject: {subject_name}")
    
    subject_out = output_root / subject_name
    subject_out.mkdir(parents=True, exist_ok=True)
    
    data_vocales = collect_fold_data(subject_dir, "vocales")
    data_comandos = collect_fold_data(subject_dir, "comandos")
    
    if data_vocales is None and data_comandos is None:
        print(f"  No data found for {subject_name}, skipping...")
        return None
    
    plot_learning_curves(data_vocales, data_comandos, subject_out / "learning_curves.png", subject_name)
    plot_confusion_matrices(data_vocales, data_comandos, subject_out / "confusion_matrices.png", subject_name)
    plot_metrics_boxplots(data_vocales, data_comandos, subject_out / "metrics_boxplots.png", subject_name)
    
    summary = {}
    for subset_name, data in [("vocales", data_vocales), ("comandos", data_comandos)]:
        if data is None or not data["metrics"]: continue
        
        metrics_arr = np.array([[m["precision"], m["recall"], m["f1"]] for m in data["metrics"]])
        
        summary[subset_name] = {
            "n_folds": len(data["metrics"]),
            "precision_mean": float(np.nanmean(metrics_arr[:, 0])),
            "precision_std": float(np.nanstd(metrics_arr[:, 0])),
            "recall_mean": float(np.nanmean(metrics_arr[:, 1])),
            "recall_std": float(np.nanstd(metrics_arr[:, 1])),
            "f1_mean": float(np.nanmean(metrics_arr[:, 2])),
            "f1_std": float(np.nanstd(metrics_arr[:, 2])),
        }
    
    save_json_safe(subject_out / "summary.json", summary)
    return {"vocales": data_vocales, "comandos": data_comandos, "summary": summary}

# ==================== MAIN ====================
if __name__ == "__main__":
    EXP_ROOT = find_latest_experiment(EXPERIMENTS_ROOT, EXPERIMENT_NAME_PREFIX)
    print(f"[Visualizer] Using experiment root: {EXP_ROOT}")
    
    OUTPUT_ROOT = EXP_ROOT / OUTPUT_SUBDIR
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    subject_dirs = sorted([p for p in EXP_ROOT.iterdir() if p.is_dir() and p.name.upper().startswith("S")])
    
    all_subjects_data = {}
    for subject_dir in subject_dirs:
        data = process_subject(subject_dir, OUTPUT_ROOT)
        if data is not None:
            all_subjects_data[subject_dir.name] = data
    
    print("\n[Visualizer] Generating global visualizations...")
    global_out = OUTPUT_ROOT / "global"
    global_out.mkdir(parents=True, exist_ok=True)
    
    plot_global_metrics_boxplots(all_subjects_data, global_out / "metrics_boxplots_global.png")
    plot_global_confusion_matrices(all_subjects_data, global_out / "confusion_matrices_global.png")
    plot_global_precision_by_subject(all_subjects_data, global_out / "precision_by_subject_global.png")
    
    global_summary = {
        "experiment_root": str(EXP_ROOT),
        "n_subjects": len(all_subjects_data),
        "subjects": list(all_subjects_data.keys())
    }
    save_json_safe(global_out / "global_summary.json", global_summary)
    
    print(f"\n[Visualizer] Done! Results saved to: {OUTPUT_ROOT}")