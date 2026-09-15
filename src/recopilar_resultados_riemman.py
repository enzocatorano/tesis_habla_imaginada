# src/recopilar_resultados_riemman.py
"""
Script de visualización y resumen de resultados de experimentos ML sobre el
espacio tangente Riemanniano (ML_TS_Riemman_experiment.py).

Genera dos tipos de análisis:
1. INTRASUJETO: Por cada sujeto individual
   - Boxplots de P/R/F1 por clasificador (un grupo por clasificador)
   - Matrices de confusión (media ± std %) por clasificador
   - Resumen JSON por clasificador y subset

2. INTERSUJETO: Agregado entre todos los sujetos
   - Boxplots de P/R/F1 globales por clasificador
   - Matrices de confusión globales por clasificador
   - Precisión por sujeto (cajas por clasificador dentro de cada sujeto,
     promedio general por clasificador y promedio absoluto)

Admite los modos definidos en experiment_config.json:
  - "binario"   -> subset "vocal_vs_comando" (2 clases, chance 0.5)
  - "separado"  -> subsets "vocales" (5) y "comandos" (6)
  - "unificado" -> subset "estimulo_unificado" (11 clases)
Si no existe config, asume formato binario (default de ML_TS_Riemman).

Los clasificadores se auto-detectan desde las carpetas reales dentro de cada fold.

Estructura de salida:
  experiments/<EXPERIMENT>/visualization_results/
    ├── <subject>/
    │   ├── metrics_boxplots.png
    │   ├── confusion_matrices.png
    │   └── summary.json
    └── global/
        ├── metrics_boxplots_global.png
        ├── confusion_matrices_global.png
        ├── precision_by_subject.png
        └── global_summary.json
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict

####################################################################################################################################
# CONFIG
####################################################################################################################################
EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1] / "experiments"
EXPERIMENT_NAME_PREFIX = "ML_TS_Riemman_espectroespacial_separado_20260915-083442_CatoranoBrothers"
OUTPUT_SUBDIR = "visualization_results"

METRICS_NAMES = ["Precision", "Recall", "F1-Score"]
METRIC_COLORS = {"Precision": "#1f77b4", "Recall": "#2ca02c", "F1-Score": "#d62728"}

CLASSIFIER_PALETTE_RGB = ["#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3"]

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 200

####################################################################################################################################
# TARGET CONFIG (según mode del experimento)
####################################################################################################################################
MODE_CONFIG = {
    "binario": {
        "subsets": ["vocal_vs_comando"],
        "subset_configs": [{
            "name": "vocal_vs_comando", "display": "Vocal vs Comando", "n_classes": 2,
            "class_names": ["Vocal", "Comando"], "chance": 0.5
        }]
    },
    "separado": {
        "subsets": ["vocales", "comandos"],
        "subset_configs": [
            {"name": "vocales", "display": "Vocales", "n_classes": 5,
             "class_names": ['A', 'E', 'I', 'O', 'U'], "chance": 1.0 / 5},
            {"name": "comandos", "display": "Comandos", "n_classes": 6,
             "class_names": ['Arriba', 'Abajo', 'Izquierda', 'Derecha', 'Adelante', 'Atras'],
             "chance": 1.0 / 6}
        ]
    },
    "unificado": {
        "subsets": ["estimulo_unificado"],
        "subset_configs": [{
            "name": "estimulo_unificado", "display": "Estimulo (1-11)", "n_classes": 11,
            "class_names": ['A', 'E', 'I', 'O', 'U', 'Arriba', 'Abajo',
                            'Izquierda', 'Derecha', 'Adelante', 'Atras'],
            "chance": 1.0 / 11
        }]
    }
}


####################################################################################################################################
# Helper funcs
####################################################################################################################################
def find_latest_experiment(root: Path, prefix: str) -> Path:
    candidates = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not candidates:
        raise FileNotFoundError(f"No experiment folders starting with '{prefix}' found under {root}")
    return sorted(candidates, key=lambda p: p.name)[-1]


def detect_experiment_config(exp_root: Path) -> dict:
    """Detecta el modo del experimento según experiment_config.json y devuelve la config de subsets."""
    config_path = exp_root / "experiment_config.json"
    if not config_path.exists():
        print("[Visualizer] No experiment_config.json found. Asumiendo modo binario.")
        return MODE_CONFIG["binario"]

    with open(config_path, encoding="utf8") as fh:
        config = json.load(fh)

    mode = config.get("mode", "binario")
    if mode in MODE_CONFIG:
        return MODE_CONFIG[mode]
    print(f"[Visualizer] mode '{mode}' no reconocido. Asumiendo modo binario.")
    return MODE_CONFIG["binario"]


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


def discover_classifiers(subset_dir: Path) -> List[str]:
    """
    Detecta los clasificadores desde las carpetas reales dentro de un fold.
    Busca la jerarquía seed_i/fold_i y lista los subdirectorios del primer fold con metadata.json.
    """
    seed_dirs = sorted([p for p in subset_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")])
    anchors = seed_dirs or [subset_dir]
    for anchor in anchors:
        fold_dirs = sorted([p for p in anchor.iterdir() if p.is_dir() and p.name.startswith("fold_")])
        if fold_dirs:
            return sorted([p.name for p in fold_dirs[0].iterdir() if p.is_dir()])
    return []


def classifier_color(name: str, n_total: int) -> str:
    """Color consistente por clasificador usando la paleta RGB circular."""
    try:
        idx = sorted_n_classifiers.index(name)
    except (ValueError, AttributeError):
        idx = 0
    n_color = len(CLASSIFIER_PALETTE_RGB)
    return CLASSIFIER_PALETTE_RGB[idx % n_color]


def _compute_ylimits(values, chances):
    """Límites Y para boxplots según fórmula pedida:
    inferior = max(0, min(0.9*min_val, 0.9*chance))
    superior = min(1.0, max(1.1*max_val, 1.1*chance))"""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    data_min = float(np.min(values)) if values.size else 0.0
    data_max = float(np.max(values)) if values.size else 1.0
    chance_min = float(np.min(chances))
    chance_max = float(np.max(chances))
    y_min = max(0.0, min(0.9 * data_min, 0.9 * chance_min))
    y_max = min(1.0, max(1.1 * data_max, 1.1 * chance_max))
    return y_min, y_max


def _overlay_points(ax, positions, data_groups, alpha=0.5, color="#333333", s=15, width=0.2):
    """Dibuja todos los puntos de cada grupo (jitter horizontal, semitransparentes)."""
    rng = np.random.default_rng(42)
    for pos, data in zip(positions, data_groups):
        data = np.asarray(data, dtype=float)
        data = data[~np.isnan(data)]
        if data.size == 0:
            continue
        jitter = rng.uniform(-width, width, size=data.size)
        ax.scatter(np.full_like(data, pos) + jitter, data, color=color, alpha=alpha,
                   s=s, edgecolors='white', linewidths=0.4, zorder=3)


def _draw_chance_hline(ax, y, label=None, x0=None, x1=None):
    """Línea punteada de probabilidad base (a ancho completo o en un tramo x0-x1)."""
    line_kwargs = dict(color='red', linestyle='--', linewidth=1.6, alpha=0.7)
    if label:
        line_kwargs['label'] = label
    if x0 is not None and x1 is not None:
        ax.plot([x0, x1], [y, y], **line_kwargs)
    else:
        ax.axhline(y=y, **line_kwargs)


####################################################################################################################################
# RECOLECCIÓN
####################################################################################################################################
def collect_fold_data(subject_dir: Path, subset_config: dict, classifiers: List[str]) -> dict:
    """
    Recopila datos de todos los seeds/folds para un sujeto y subset.

    Returns:
        dict {clasificador: {
            "metrics": lista de dicts {precision, recall, f1},
            "confusion_matrices": lista de matrices (normalizadas por fila, %),
            "test_accuracies": lista de float,
            "val_accuracies": lista de float
        }}
    """
    subset_dir = subject_dir / subset_config["name"]
    if not subset_dir.exists():
        return None

    result = {clf: {"metrics": [], "confusion_matrices": [], "test_accuracies": [], "val_accuracies": []}
              for clf in classifiers}

    seed_dirs = sorted([p for p in subset_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")])
    if not seed_dirs:
        seed_dirs = [subset_dir]

    for seed_dir in seed_dirs:
        fold_dirs = sorted([p for p in seed_dir.iterdir() if p.is_dir() and p.name.startswith("fold_")])
        for fold_dir in fold_dirs:
            for clf in classifiers:
                clf_dir = fold_dir / clf
                if not clf_dir.exists():
                    continue

                meta = load_json_safe(clf_dir / "metadata.json")
                if meta is None or meta.get("status") != "success":
                    continue

                cr = load_json_safe(clf_dir / "classification_report.json")
                if cr:
                    macro = cr.get("macro avg") or cr.get("macro_avg") or cr.get("macro-average")
                    if macro:
                        result[clf]["metrics"].append({
                            "precision": float(macro.get("precision", np.nan)),
                            "recall": float(macro.get("recall", np.nan)),
                            "f1": float(macro.get("f1-score", macro.get("f1_score", np.nan)))
                        })

                cm_path = clf_dir / "confusion_matrix.npy"
                if cm_path.exists():
                    try:
                        cm = np.load(cm_path).astype(np.float64)
                        row_sums = cm.sum(axis=1, keepdims=True)
                        cm_pct = np.full_like(cm, 0.0, dtype=np.float64)
                        np.divide(cm, row_sums, out=cm_pct, where=(row_sums != 0))
                        cm_pct = cm_pct * 100.0
                        result[clf]["confusion_matrices"].append(cm_pct)
                    except Exception as e:
                        print(f"  Warning: Error loading confusion matrix: {e}")

                if meta.get("test_accuracy") is not None:
                    result[clf]["test_accuracies"].append(float(meta["test_accuracy"]))
                if meta.get("val_accuracy") is not None:
                    result[clf]["val_accuracies"].append(float(meta["val_accuracy"]))

    # Eliminar clasificadores sin datos
    result = {clf: data for clf, data in result.items() if data["metrics"] or data["confusion_matrices"]}
    return result if result else None


####################################################################################################################################
# PLOTS INTRASUJETO
####################################################################################################################################
def plot_metrics_boxplots(subset_data_list, subset_configs, classifiers, output_path: Path, subject_name: str):
    """Boxplots de P/R/F1 por clasificador (una fila por subset). P azul, R verde, F1 rojo."""
    n_subsets = len(subset_configs)
    fig, axes = plt.subplots(n_subsets, 1, figsize=(12, 5 * n_subsets))
    if n_subsets == 1:
        axes = [axes]

    for ax, data, cfg in zip(axes, subset_data_list, subset_configs):
        subset_name = cfg["display"]
        chance_level = cfg["chance"]
        if data is None:
            ax.text(0.5, 0.5, f"No data for {subset_name}", ha='center', va='center')
            ax.axis('off')
            continue

        all_data = []
        labels = []
        positions = []
        colors = []
        pos = 1
        for clf in classifiers:
            clf_data = data.get(clf)
            if clf_data is None or not clf_data["metrics"]:
                pos += 4
                continue
            metrics_arr = np.array([[m["precision"], m["recall"], m["f1"]] for m in clf_data["metrics"]])
            for i, metric_name in enumerate(METRICS_NAMES):
                all_data.append(metrics_arr[:, i])
                labels.append(f"{clf}\n{metric_name}")
                positions.append(pos)
                colors.append(METRIC_COLORS[metric_name])
                pos += 1
            pos += 2  # Separación entre clasificadores

        if not all_data:
            ax.text(0.5, 0.5, f"No metrics for {subset_name}", ha='center', va='center')
            ax.axis('off')
            continue

        bp = ax.boxplot(all_data, positions=positions, widths=0.6, patch_artist=True,
                        showmeans=True, meanline=True, showfliers=False,
                        boxprops=dict(linewidth=1.5), whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5),
                        medianprops=dict(linewidth=2, color='black'),
                        meanprops=dict(linewidth=2, color='white', linestyle='--'))

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        _overlay_points(ax, positions, all_data, s=20, width=0.14)

        _draw_chance_hline(ax, chance_level, label=f'Chance level ({chance_level:.3f})')
        data_concat = np.concatenate([np.asarray(d, dtype=float) for d in all_data])
        y_min, y_max = _compute_ylimits(data_concat, [chance_level])
        ax.set_ylim(y_min, y_max)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Score", fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_title(subset_name, fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)

    fig.suptitle(f"Distribucion de metricas - {subject_name}", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_confusion_matrices(subset_data_list, subset_configs, classifiers, output_path: Path, subject_name: str):
    """Matrices de confusión: filas = subsets, columnas = clasificadores."""
    n_subsets = len(subset_configs)
    n_clf = len(classifiers)
    fig, axes = plt.subplots(n_subsets, n_clf, figsize=(5.5 * n_clf, 5 * n_subsets))
    if n_subsets == 1 and n_clf == 1:
        axes = [[axes]]
    elif n_subsets == 1:
        axes = [axes]
    elif n_clf == 1:
        axes = [[ax] for ax in axes]

    for r, (ax_row, data, cfg) in enumerate(zip(axes, subset_data_list, subset_configs)):
        class_names = cfg["class_names"]
        for c, (ax, clf) in enumerate(zip(ax_row, classifiers)):
            if data is None or data.get(clf) is None or not data[clf]["confusion_matrices"]:
                ax.text(0.5, 0.5, f"No data\n{clf}", ha='center', va='center')
                ax.axis('off')
                continue

            cm_stack = np.stack(data[clf]["confusion_matrices"], axis=0)
            cm_mean = np.nanmean(cm_stack, axis=0)
            cm_std = np.nanstd(cm_stack, axis=0)

            n = cm_mean.shape[0]
            annot = np.empty((n, n), dtype=object)
            for i in range(n):
                for j in range(n):
                    annot[i, j] = f"{cm_mean[i, j]:.1f}\n±{cm_std[i, j]:.1f}"

            sns.heatmap(cm_mean, annot=annot, fmt="", cmap="cividis", ax=ax,
                        vmin=0, vmax=100,
                        cbar_kws={'label': '% (normalizado por fila)'},
                        linewidths=0.5, linecolor='gray', annot_kws={"size": 9})

            ax.set_title(f"{clf}", fontsize=12, fontweight='bold')
            if c == 0:
                ax.set_ylabel(cfg["display"] + "\nVerdadera", fontsize=10)
            ax.set_xlabel("Predicha", fontsize=10)
            ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
            ax.set_yticklabels(class_names, rotation=0, fontsize=9)

    fig.suptitle(f"Matrices de confusion - {subject_name}", fontsize=15, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def process_subject(subject_dir: Path, output_root: Path, subset_configs: List[dict], classifiers: List[str]) -> dict:
    """Procesa un sujeto individual y genera todas las visualizaciones."""
    subject_name = subject_dir.name
    print(f"\n[Visualizer] Processing subject: {subject_name}")

    subject_out = output_root / subject_name
    subject_out.mkdir(parents=True, exist_ok=True)

    subset_data_list = [collect_fold_data(subject_dir, cfg, classifiers) for cfg in subset_configs]

    if all(data is None for data in subset_data_list):
        print(f"  No data found for {subject_name}, skipping...")
        return None

    plot_metrics_boxplots(subset_data_list, subset_configs, classifiers,
                          subject_out / "metrics_boxplots.png", subject_name)

    plot_confusion_matrices(subset_data_list, subset_configs, classifiers,
                            subject_out / "confusion_matrices.png", subject_name)

    summary = {}
    for cfg, data in zip(subset_configs, subset_data_list):
        if data is None:
            continue
        summary[cfg["name"]] = {}
        for clf in classifiers:
            clf_data = data.get(clf)
            if clf_data is None or not clf_data["metrics"]:
                continue
            metrics_arr = np.array([[m["precision"], m["recall"], m["f1"]] for m in clf_data["metrics"]])
            summary[cfg["name"]][clf] = {
                "n_folds": len(clf_data["metrics"]),
                "precision_mean": float(np.nanmean(metrics_arr[:, 0])),
                "precision_std": float(np.nanstd(metrics_arr[:, 0])),
                "recall_mean": float(np.nanmean(metrics_arr[:, 1])),
                "recall_std": float(np.nanstd(metrics_arr[:, 1])),
                "f1_mean": float(np.nanmean(metrics_arr[:, 2])),
                "f1_std": float(np.nanstd(metrics_arr[:, 2])),
                "test_accuracy_mean": float(np.nanmean(clf_data["test_accuracies"])) if clf_data["test_accuracies"] else None,
                "val_accuracy_mean": float(np.nanmean(clf_data["val_accuracies"])) if clf_data["val_accuracies"] else None,
            }

    save_json_safe(subject_out / "summary.json", summary)
    print(f"  Saved: {subject_out / 'summary.json'}")

    return summary


####################################################################################################################################
# PLOTS GLOBALES
####################################################################################################################################
def plot_global_metrics_boxplots(all_subjects_data: Dict, output_path: Path, subset_configs: List[dict], classifiers: List[str]):
    """Boxplots globales de P/R/F1 por clasificador (todas las folds de todos los sujetos)."""
    n_subsets = len(subset_configs)
    fig, axes = plt.subplots(n_subsets, 1, figsize=(12, 5 * n_subsets))
    if n_subsets == 1:
        axes = [axes]

    for ax, cfg in zip(axes, subset_configs):
        subset_name = cfg["display"]
        chance_level = cfg["chance"]

        all_data = []
        labels = []
        positions = []
        colors = []
        pos = 1
        for clf in classifiers:
            clf_metrics = []
            for subj_name, subj_data in all_subjects_data.items():
                subset_data = subj_data.get(cfg["name"])
                if subset_data is None or subset_data.get(clf) is None:
                    continue
                clf_metrics.extend([m for m in subset_data[clf]["metrics"] if m])

            if not clf_metrics:
                pos += 4
                continue

            metrics_arr = np.array([[m["precision"], m["recall"], m["f1"]] for m in clf_metrics])
            for i, metric_name in enumerate(METRICS_NAMES):
                all_data.append(metrics_arr[:, i])
                labels.append(f"{clf}\n{metric_name}")
                positions.append(pos)
                colors.append(METRIC_COLORS[metric_name])
                pos += 1
            pos += 2

        if not all_data:
            ax.text(0.5, 0.5, f"No data for {subset_name}", ha='center', va='center')
            ax.axis('off')
            continue

        bp = ax.boxplot(all_data, positions=positions, widths=0.6, patch_artist=True,
                        showmeans=True, meanline=True, showfliers=False,
                        boxprops=dict(linewidth=1.5), whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5),
                        medianprops=dict(linewidth=2, color='black'),
                        meanprops=dict(linewidth=2, color='white', linestyle='--'))

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        _overlay_points(ax, positions, all_data, s=20, width=0.14)

        _draw_chance_hline(ax, chance_level, label=f'Chance level ({chance_level:.3f})')
        data_concat = np.concatenate([np.asarray(d, dtype=float) for d in all_data])
        y_min, y_max = _compute_ylimits(data_concat, [chance_level])
        ax.set_ylim(y_min, y_max)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Score", fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_title(subset_name, fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)

    fig.suptitle("Global Metrics Distribution (Across Subjects)", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_global_confusion_matrices(all_subjects_data_raw: Dict, output_path: Path, subset_configs: List[dict], classifiers: List[str]):
    """Matrices de confusión globales: filas = subsets, columnas = clasificadores (promedio entre sujetos)."""
    n_subsets = len(subset_configs)
    n_clf = len(classifiers)
    fig, axes = plt.subplots(n_subsets, n_clf, figsize=(5.5 * n_clf, 5 * n_subsets))
    if n_subsets == 1 and n_clf == 1:
        axes = [[axes]]
    elif n_subsets == 1:
        axes = [axes]
    elif n_clf == 1:
        axes = [[ax] for ax in axes]

    for r, (ax_row, cfg) in enumerate(zip(axes, subset_configs)):
        class_names = cfg["class_names"]
        for c, (ax, clf) in enumerate(zip(ax_row, classifiers)):
            cm_list = []
            for subj_name, subj_data in all_subjects_data_raw.items():
                subset_data = subj_data.get(cfg["name"])
                if subset_data is None or subset_data.get(clf) is None:
                    continue
                cm_stack = np.stack(subset_data[clf]["confusion_matrices"], axis=0)
                if cm_stack.size:
                    cm_list.append(np.nanmean(cm_stack, axis=0))

            if not cm_list:
                ax.text(0.5, 0.5, f"No data\n{clf}", ha='center', va='center')
                ax.axis('off')
                continue

            cm_global_stack = np.stack(cm_list, axis=0)
            cm_mean = np.nanmean(cm_global_stack, axis=0)
            cm_std = np.nanstd(cm_global_stack, axis=0)

            n = cm_mean.shape[0]
            annot = np.empty((n, n), dtype=object)
            for i in range(n):
                for j in range(n):
                    annot[i, j] = f"{cm_mean[i, j]:.1f}\n±{cm_std[i, j]:.1f}"

            sns.heatmap(cm_mean, annot=annot, fmt="", cmap="cividis", ax=ax,
                        vmin=0, vmax=100,
                        cbar_kws={'label': '% (row-normalized)'},
                        linewidths=0.5, linecolor='gray', annot_kws={"size": 9})

            ax.set_title(f"{clf}", fontsize=12, fontweight='bold')
            if c == 0:
                ax.set_ylabel(cfg["display"] + "\nTrue", fontsize=10)
            ax.set_xlabel("Predicted", fontsize=10)
            ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
            ax.set_yticklabels(class_names, rotation=0, fontsize=9)

    fig.suptitle("Global Confusion Matrices (Across Subjects)", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_precision_by_subject(all_subjects_data_raw: Dict, output_path: Path, subset_configs: List[dict], classifiers: List[str]):
    """
    Precisión por sujeto (una fila por subset).
    Eje X: cada sujeto con cajas por clasificador (coloreadas), luego bloque
    'Promedio por clasificador' y 'Promedio absoluto'.
    """
    n_subsets = len(subset_configs)
    fig, axes = plt.subplots(n_subsets, 1, figsize=(16, 5.5 * n_subsets))
    if n_subsets == 1:
        axes = [axes]

    subject_names = sorted(all_subjects_data_raw.keys())
    n_subjects = len(subject_names)

    for ax, cfg in zip(axes, subset_configs):
        subset_name = cfg["display"]
        chance_level = cfg["chance"]

        # prec_by_subject[clf] = lista con el valor de precisión por fold de cada sujeto (todos juntos)
        prec_list_by_subject = {clf: [] for clf in classifiers}

        # Para colorear las cajas: guardamos (pos, color) y los datos
        box_data = []
        box_positions = []
        box_colors = []

        width = 0.4
        gap_clf = 0.5
        gap_subject = 3.0
        pos = 1.0

        # Coordenada central de cada sujeto
        subject_offsets = []
        # Columnas dentro de cada sujeto
        clf_deltas = {clf: (i - (len(classifiers) - 1) / 2.0) * gap_clf for i, clf in enumerate(classifiers)}

        for subj_idx, subj_name in enumerate(subject_names):
            subj_data = all_subjects_data_raw.get(subj_name)
            for clf in classifiers:
                precs = []
                if subj_data is not None:
                    subset_data = subj_data.get(cfg["name"])
                    if subset_data is not None and subset_data.get(clf) is not None:
                        precs = [m["precision"] for m in subset_data[clf]["metrics"] if m]
                if precs:
                    clf_pos = pos + clf_deltas[clf]
                    box_data.append(precs)
                    box_positions.append(clf_pos)
                    box_colors.append(classifier_color(clf, len(classifiers)))
                    prec_list_by_subject[clf].extend(precs)
            subject_offsets.append((pos, subj_name))
            pos += gap_subject

        # Bloque "Promedio por clasificador"
        avg_clf_positions = []
        avg_block_start = pos + 1.0
        for i, clf in enumerate(classifiers):
            all_prec = prec_list_by_subject[clf]
            if all_prec:
                clf_pos = avg_block_start + i * gap_clf
                box_data.append(all_prec)
                box_positions.append(clf_pos)
                box_colors.append(classifier_color(clf, len(classifiers)))
                avg_clf_positions.append(clf_pos)

        # Bloque "Promedio absoluto"
        all_absolute = []
        for clf in classifiers:
            all_absolute.extend(prec_list_by_subject[clf])
        if all_absolute:
            abs_pos = avg_block_start + len(classifiers) * gap_clf + 1.0
            box_data.append(all_absolute)
            box_positions.append(abs_pos)
            box_colors.append("#e0e0e0")

        if box_data:
            bp = ax.boxplot(box_data, positions=box_positions, widths=width, patch_artist=True,
                            showmeans=True, meanline=True, showfliers=False,
                            medianprops=dict(linewidth=1.6, color='#264653'),
                            whiskerprops=dict(color='#264653', alpha=0.7),
                            capprops=dict(color='#264653', alpha=0.7))
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.85)

            _overlay_points(ax, box_positions, box_data, alpha=0.25, s=18, width=width * 0.25)

        # Tick labels: sujetos + bloques
        subject_ticks = [subj_name for (p, subj_name) in subject_offsets]
        subject_tick_pos = [p for (p, subj_name) in subject_offsets]
        extra_ticks = []
        extra_labels = []
        if avg_clf_positions:
            extra_ticks.append(np.mean(avg_clf_positions))
            extra_labels.append("Promedio por\nclasificador")
        if all_absolute:
            extra_ticks.append(abs_pos)
            extra_labels.append("Promedio\nabsoluto")

        all_ticks = subject_tick_pos + extra_ticks
        all_labels = subject_ticks + extra_labels
        ax.set_xticks(all_ticks)
        ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=9)

        _draw_chance_hline(ax, chance_level, label=f'Chance level ({chance_level:.3f})')
        y_min, y_max = _compute_ylimits(all_absolute, [chance_level])
        ax.set_ylim(y_min, y_max)

        ax.set_ylabel("Precision", fontsize=11)
        ax.set_title(f"{subset_name} - Precision por sujeto y clasificador", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Leyenda de clasificadores (patches manuales)
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=classifier_color(clf, len(classifiers)), label=clf) for clf in classifiers]
        handles.append(Patch(facecolor="#e0e0e0", label="Promedio absoluto"))
        ax.legend(handles=handles, loc='best', fontsize=8)

    fig.suptitle("Precision por sujeto y clasificador", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


####################################################################################################################################
# MAIN
####################################################################################################################################
if __name__ == "__main__":
    global sorted_n_classifiers
    sorted_n_classifiers = []

    EXP_ROOT = find_latest_experiment(EXPERIMENTS_ROOT, EXPERIMENT_NAME_PREFIX)
    exp_config = detect_experiment_config(EXP_ROOT)
    subset_configs = exp_config["subset_configs"]
    print(f"[Visualizer] Using experiment root: {EXP_ROOT}")
    print(f"[Visualizer] Mode detectado, subsets: {[c['name'] for c in subset_configs]}")

    OUTPUT_ROOT = EXP_ROOT / OUTPUT_SUBDIR
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    subject_dirs = sorted([p for p in EXP_ROOT.iterdir() if p.is_dir() and p.name.upper().startswith("S")])
    print(f"[Visualizer] Found {len(subject_dirs)} subject directories")

    # Detectar clasificadores desde el primer sujeto disponible
    classifiers = []
    for subj_dir in subject_dirs:
        for cfg in subset_configs:
            classifiers = discover_classifiers(subj_dir / cfg["name"])
            if classifiers:
                break
        if classifiers:
            break

    if not classifiers:
        raise FileNotFoundError("No se pudieron detectar clasificadores en los folds del experimento")
    sorted_n_classifiers = classifiers
    print(f"[Visualizer] Clasificadores detectados: {classifiers}")

    all_subjects_summaries = {}
    all_subjects_data_raw = {}

    for subject_dir in subject_dirs:
        subject_name = subject_dir.name
        summary = process_subject(subject_dir, OUTPUT_ROOT, subset_configs, classifiers)

        if summary is not None:
            all_subjects_summaries[subject_name] = summary
            all_subjects_data_raw[subject_name] = {
                cfg["name"]: collect_fold_data(subject_dir, cfg, classifiers)
                for cfg in subset_configs
            }

    print("\n[Visualizer] Generating global visualizations...")
    global_out = OUTPUT_ROOT / "global"
    global_out.mkdir(parents=True, exist_ok=True)

    plot_global_metrics_boxplots(all_subjects_data_raw,
                                 global_out / "metrics_boxplots_global.png", subset_configs, classifiers)

    plot_global_confusion_matrices(all_subjects_data_raw,
                                   global_out / "confusion_matrices_global.png", subset_configs, classifiers)

    plot_precision_by_subject(all_subjects_data_raw,
                              global_out / "precision_by_subject.png", subset_configs, classifiers)

    global_summary = {
        "experiment_root": str(EXP_ROOT),
        "mode": exp_config,
        "classifiers": classifiers,
        "n_subjects": len(all_subjects_summaries),
        "subjects": list(all_subjects_summaries.keys()),
        "per_subject_summaries": all_subjects_summaries
    }

    save_json_safe(global_out / "global_summary.json", global_summary)
    print(f"  Saved: {global_out / 'global_summary.json'}")

    print(f"\n[Visualizer] Done! Results saved to: {OUTPUT_ROOT}")