# src/visualize_results_online_augmentation.py
"""
Script de visualización y resumen de resultados para experimentos con augmentación online.

Diferencias vs versión anterior:
- No usa archivos .npz de datos augmentados (ahora la augmentación es online)
- Lee augmentation_metadata.json para obtener parámetros de augmentación
- Extrae info de ventana temporal, banda y FTS desde test_preds.npz y Y_test guardado
- No necesita mapear índices globales al dataset augmentado

Genera dos tipos de análisis:
1. INTRASUJETO: Por cada sujeto individual
   - Learning curves (train/val loss + val acc)
   - Matrices de confusión (mean ± std)
   - Boxplots de P/R/F1
   - Boxplots por banda de frecuencia
   - Boxplots por uso de FT Surrogate
   - Boxplots por ventana temporal

2. INTERSUJETO: Agregado entre todos los sujetos
   - Boxplots de P/R/F1 entre sujetos
   - Matrices de confusión globales
   - Boxplots por banda (global)
   - Boxplots por FT Surrogate (global)
   - Boxplots por ventana temporal (global)
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple

# ---------------- CONFIG ----------------
EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1] / "experiments"
EXPERIMENT_NAME_PREFIX = "S01_iSpeechCNN_piloto_20260325-082107_CatoranoBrothers"
OUTPUT_SUBDIR = "visualization_results"

BAND_LABELS = ["delta", "theta", "alpha", "beta", "gamma"]
BAND_FREQ_LABELS = ["delta\n0-4 Hz", "theta\n4-8 Hz", "alpha\n8-12 Hz", "beta\n12-32 Hz", "gamma\n32-64 Hz"]
VOCAL_CLASS_NAMES = ['A', 'E', 'I', 'O', 'U']
COMANDO_CLASS_NAMES = ['Arriba', 'Abajo', 'Izquierda', 'Derecha', 'Adelante', 'Atras']
SUBSETS = ["vocales", "comandos"]
METRICS_NAMES = ["Precision", "Recall", "F1-Score"]

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 200
# Paleta de colores consistente con tus otros scripts
COLORS = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51']
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
    """Agrega series de diferentes longitudes rellenando con NaN."""
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


def collect_fold_data(subject_dir: Path, subset: str):
    """
    Recopila datos de todos los folds/seeds para un sujeto y subset específico.
    
    ADAPTADO para augmentación online: lee etiquetas desde test_preds.npz
    
    Returns:
        dict con:
            - metrics: lista de dicts {precision, recall, f1}
            - confusion_matrices: lista de matrices (normalizadas por fila)
            - band_accuracies: lista de arrays (n_bands,)
            - ftsurrogate_accuracies: lista de dicts {with_fts, without_fts}
            - window_accuracies: lista de arrays (n_windows,)
            - train_losses, val_losses, val_accs: listas de listas
            - augmentation_params: parámetros de augmentación
    """
    subset_dir = subject_dir / subset
    if not subset_dir.exists():
        return None
    
    metrics_list = []
    cm_list = []
    band_acc_list = []
    fts_acc_list = []
    window_acc_list = []
    train_loss_list = []
    val_loss_list = []
    val_acc_list = []
    augmentation_params_list = []
    
    # Iterar por seeds
    seed_dirs = sorted([p for p in subset_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")])
    if not seed_dirs:
        seed_dirs = [subset_dir]
    
    for seed_dir in seed_dirs:
        fold_dirs = sorted([p for p in seed_dir.iterdir() if p.is_dir() and p.name.startswith("fold_")])
        
        for fold_dir in fold_dirs:
            # Cargar metadata
            meta = load_json_safe(fold_dir / "metadata.json")
            if meta is None or meta.get("status") != "success":
                continue
            
            # Cargar augmentation metadata
            aug_meta = load_json_safe(fold_dir / "augmentation_metadata.json")
            if aug_meta:
                augmentation_params_list.append(aug_meta)
            
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
                    with np.errstate(divide='ignore', invalid='ignore'):
                        row_sums = cm.sum(axis=1, keepdims=True)
                        cm_pct = np.divide(cm, row_sums, where=(row_sums != 0)) * 100.0
                        cm_pct = np.nan_to_num(cm_pct, nan=0.0)
                    cm_list.append(cm_pct)
                except Exception as e:
                    print(f"  Warning: Error loading confusion matrix: {e}")
            
            # 3-5. Análisis por banda, FTS y ventana temporal
            # NUEVO: Cargar Y_test desde archivo separado (debe guardarse en el script de experimentos)
            y_test_path = fold_dir / "y_test.npy"
            preds_path = fold_dir / "test_preds.npz"
            
            if y_test_path.exists() and preds_path.exists():
                try:
                    # Cargar etiquetas completas del test set (shape: n_test, 6)
                    Y_test = np.load(y_test_path, allow_pickle=True)
                    
                    # Cargar predicciones
                    d = np.load(preds_path, allow_pickle=True)
                    y_true = d["y_true"]
                    y_pred = d["y_pred"]
                    
                    # Extraer etiquetas auxiliares
                    # Y_test columnas: [modalidad, estímulo, artefacto, ventana, banda, fts]
                    ventana_labels = Y_test[:, 3].astype(int)    # columna 3
                    banda_labels = Y_test[:, 4].astype(int)      # columna 4
                    fts_labels = Y_test[:, 5].astype(int)        # columna 5
                    
                    # 3. Band accuracy
                    band_accs = []
                    for b in range(1, 6):  # bandas 1-5
                        # Incluir solo los que fueron augmentados con esta banda
                        mask = (banda_labels == b)
                        if mask.sum() == 0:
                            band_accs.append(np.nan)
                        else:
                            band_accs.append(float(np.mean(y_true[mask] == y_pred[mask])))
                    
                    if any(not np.isnan(x) for x in band_accs):
                        band_acc_list.append(np.array(band_accs))
                    
                    # 4. FTS accuracy
                    mask_with = (fts_labels == 1)
                    mask_without = (fts_labels == 0)
                    
                    acc_with = float(np.mean(y_true[mask_with] == y_pred[mask_with])) if mask_with.sum() > 0 else np.nan
                    acc_without = float(np.mean(y_true[mask_without] == y_pred[mask_without])) if mask_without.sum() > 0 else np.nan
                    
                    if not np.isnan(acc_with) or not np.isnan(acc_without):
                        fts_acc_list.append({"with_fts": acc_with, "without_fts": acc_without})
                    
                    # 5. Window accuracy
                    n_windows = int(ventana_labels.max() + 1) if len(ventana_labels) > 0 else 6
                    window_accs = []
                    for w in range(n_windows):
                        mask = (ventana_labels == w)
                        if mask.sum() == 0:
                            window_accs.append(np.nan)
                        else:
                            window_accs.append(float(np.mean(y_true[mask] == y_pred[mask])))
                    
                    if any(not np.isnan(x) for x in window_accs):
                        window_acc_list.append(np.array(window_accs))
                    
                except Exception as e:
                    print(f"  Warning: Error processing test labels: {e}")
            
            # 6. Learning curves
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
        "window_accuracies": window_acc_list,
        "train_losses": train_loss_list,
        "val_losses": val_loss_list,
        "val_accs": val_acc_list,
        "augmentation_params": augmentation_params_list
    }


def apply_custom_boxplot(ax, data, positions, labels, colors, widths=0.2):
    """
    Función de utilidad para aplicar el estilo de boxplot con jitter 
    visto en el script de distribución de clases.
    """
    for i, (d, pos, col) in enumerate(zip(data, positions, colors)):
        if len(d) == 0: continue
        
        # Boxplot con patch_artist y alpha 0.5
        bp = ax.boxplot(d, positions=[pos], widths=widths, 
                        showfliers=False, manage_ticks=False,
                        patch_artist=True,
                        medianprops=dict(color="orange", linewidth=1.5),
                        whiskerprops=dict(color=col, alpha=0.7),
                        capprops=dict(color=col, alpha=0.7),
                        boxprops=dict(facecolor=col, color=col, alpha=0.5))
        
        # Jitter con puntos grandes (s=50)
        jitter = np.random.normal(pos, 0.02, size=len(d))
        ax.scatter(jitter, d, color=col, alpha=0.8, s=50, edgecolors='white', linewidths=0.5)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


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
        
        ax2 = ax.twinx()
        color_acc = "tab:green"
        
        if va_mean.size > 0:
            ax2.plot(epochs[:va_mean.size], va_mean, color=color_acc, linestyle='--', 
                    label="Val Accuracy", linewidth=2)
            ax2.fill_between(epochs[:va_mean.size], va_mean - va_std, va_mean + va_std, 
                            alpha=0.15, color=color_acc)
        
        ax2.set_ylabel("Val Accuracy", fontsize=11)
        ax2.tick_params(axis='y')
        
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
    """Genera figura de matrices de confusión."""
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
                   cbar_kws={'label': '% (row-normalized)'}, 
                   linewidths=0.5, linecolor='gray',
                   annot_kws={"size": 9})
        
        ax.set_title(f"{subset_name} (mean ± std %)", fontsize=12, fontweight='bold')
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True", fontsize=10)
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(class_names, rotation=0, fontsize=9)
    
    fig.suptitle(f"Confusion Matrices - {subject_name}", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_metrics_boxplots(data_vocales, data_comandos, output_path: Path, subject_name: str):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    all_data = []
    labels = []
    positions = []
    plot_colors = []
    
    pos = 0
    # Usamos los primeros 3 colores para P, R, F1
    metric_colors = COLORS[:3] 
    
    for subset_name, data in [("Vocales", data_vocales), ("Comandos", data_comandos)]:
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

        # limites verticales
        # nivel de chance
        if subset_name == "vocales":
            chance_level = 1.0 / 5
        else: # comandos
            chance_level = 1.0 / 6
        # para el limite maximo va el mas chico entre
        #   el mas grande entre
        #     el valor mas grande
        #     el nivel de chance
        #   el 1
        max_val_data = np.nanmax([np.nanmax(d) for d in all_data if d.size > 0]) if all_data else -np.inf
        upper_limit = min(1.0, max(max_val_data, chance_level) + 0.05)
        # para el limite minimo va el mas grande entre
        #   el mas chico entre
        #     el valor mas chico
        #     el nivel de chance
        #   el 0
        min_val_data = np.nanmin([np.nanmin(d) for d in all_data if d.size > 0]) if all_data else np.inf
        lower_limit = max(0.0, min(min_val_data, chance_level) - 0.05)
        ax.set_ylim(lower_limit, upper_limit)
        
        ax.set_title(f"Metrics Distribution - {subject_name}", fontweight='bold')
    
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_band_accuracy_boxplots(data_vocales, data_comandos, output_path: Path, subject_name: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, data, subset_name, n_classes in zip(axes, [data_vocales, data_comandos], ["Vocales", "Comandos"], [5, 6]):
        if data is None or not data["band_accuracies"]:
            ax.axis('off')
            continue
        
        band_arr = np.vstack(data["band_accuracies"])
        band_data = [band_arr[:, i] for i in range(len(BAND_LABELS))]
        positions = np.arange(len(BAND_LABELS))
        
        # Ciclo de colores para las bandas
        apply_custom_boxplot(ax, band_data, positions, BAND_FREQ_LABELS, COLORS, widths=0.3)
        
        chance_level = 1.0 / n_classes
        ax.axhline(y=chance_level, color='gray', linestyle=':', linewidth=2, label=f'Chance ({chance_level:.2f})')
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{subset_name}", fontweight='bold')

        # limites verticales
        min_val = min(np.nanmin(band_data) if band_data else np.inf, chance_level)
        max_val = max(np.nanmax(band_data) if band_data else -np.inf, chance_level)
        # Ajustar los límites para que sean un poco más amplios que los datos
        lower_bound = max(0.0, min_val - 0.05)
        upper_bound = min(1.0, max_val + 0.05)
        ax.set_ylim(lower_bound, upper_bound)

        ax.legend()
    
    fig.suptitle(f"Accuracy by Frequency Band - {subject_name}", fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_ftsurrogate_accuracy_boxplots(data_vocales, data_comandos, output_path: Path, subject_name: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    for ax, data, subset_name, n_classes in zip(axes, [data_vocales, data_comandos], ["Vocales", "Comandos"], [5, 6]):
        if data is None or not data["ftsurrogate_accuracies"]:
            ax.axis('off')
            continue
            
        with_fts = [d["with_fts"] for d in data["ftsurrogate_accuracies"] if not np.isnan(d["with_fts"])]
        without_fts = [d["without_fts"] for d in data["ftsurrogate_accuracies"] if not np.isnan(d["without_fts"])]
        
        data_plot = [without_fts, with_fts]
        labels = ["Without FTS", "With FTS"]
        # Usamos colores 4 y 5 de la paleta
        apply_custom_boxplot(ax, data_plot, [0, 1], labels, [COLORS[0], COLORS[4]], widths=0.4)
        
        chance_level = 1.0 / n_classes
        ax.axhline(y=chance_level, color='gray', linestyle=':', linewidth=2)
        ax.set_ylabel("Accuracy")

        # limites verticales
        min_val = min(np.nanmin(without_fts) if without_fts else np.inf, 
                      np.nanmin(with_fts) if with_fts else np.inf, 
                      chance_level)
        max_val = max(np.nanmax(without_fts) if without_fts else -np.inf, 
                      np.nanmax(with_fts) if with_fts else -np.inf, 
                      chance_level)
        # Ajustar los límites para que sean un poco más amplios que los datos
        # y siempre incluyan 0 y 1 si es posible, o al menos el nivel de azar.
        lower_bound = max(0.0, min_val - 0.05)
        upper_bound = min(1.0, max_val + 0.05)
        ax.set_ylim(lower_bound, upper_bound)

        ax.set_title(f"{subset_name}", fontweight='bold')
    
    fig.suptitle(f"Accuracy by FT Surrogate Usage - {subject_name}", fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_temporal_window_boxplots(data_vocales, data_comandos, output_path: Path, subject_name: str):
    """Genera boxplots de accuracy por ventana temporal para un sujeto."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, data, subset_name, n_classes in zip(axes, [data_vocales, data_comandos], ["Vocales", "Comandos"], [5, 6]):
        if data is None or not data["window_accuracies"]:
            ax.axis('off')
            continue
        
        # Convertimos la lista de arrays en una matriz (n_folds, n_windows)
        window_arr = np.vstack(data["window_accuracies"])
        n_windows = window_arr.shape[1]
        
        # Preparación de datos para el boxplot
        window_data = [window_arr[:, i] for i in range(n_windows)]
        positions = np.arange(n_windows)
        window_labels = [f"Win {i}\n({i*0.5:.1f}-{i*0.5+1.5:.1f}s)" for i in range(n_windows)]
        
        # Ciclo de colores: usamos la paleta COLORS y reiniciamos si hay más de 5 ventanas
        plot_colors = [COLORS[i % len(COLORS)] for i in range(n_windows)]
        
        # Aplicamos el estilo personalizado
        apply_custom_boxplot(ax, window_data, positions, window_labels, plot_colors, widths=0.3)
        
        # Línea de azar (chance level)
        chance_level = 1.0 / n_classes
        ax.axhline(y=chance_level, color='gray', linestyle=':', linewidth=2, 
                   label=f'Chance ({chance_level:.2f})', alpha=0.7)
        
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.set_title(f"{subset_name}", fontsize=12, fontweight='bold')
        
        # limites verticales
        min_val = min(np.nanmin(window_data) if window_data else np.inf, chance_level)
        max_val = max(np.nanmax(window_data) if window_data else -np.inf, chance_level)
        lower_bound = max(0.0, min_val - 0.05)
        upper_bound = min(1.0, max_val + 0.05)
        ax.set_ylim(lower_bound, upper_bound)

        ax.tick_params(axis='x', rotation=30, labelsize=8)
        ax.legend(loc='best', fontsize=8)
    
    fig.suptitle(f"Accuracy by Temporal Window - {subject_name}", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def process_subject(subject_dir: Path, output_root: Path):
    """Procesa un sujeto individual y genera todas las visualizaciones."""
    subject_name = subject_dir.name
    print(f"\n[Visualizer] Processing subject: {subject_name}")
    
    subject_out = output_root / subject_name
    subject_out.mkdir(parents=True, exist_ok=True)
    
    # Recopilar datos
    data_vocales = collect_fold_data(subject_dir, "vocales")
    data_comandos = collect_fold_data(subject_dir, "comandos")
    
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
    
    plot_temporal_window_boxplots(data_vocales, data_comandos,
                                 subject_out / "temporal_window_boxplots.png", subject_name)
    
    # Guardar resumen JSON
    summary = {}
    for subset_name, data in [("vocales", data_vocales), ("comandos", data_comandos)]:
        if data is None:
            continue
        
        metrics_arr = np.array([[m["precision"], m["recall"], m["f1"]] for m in data["metrics"]]) if data["metrics"] else np.array([])
        
        subset_summary = {
            "n_folds": len(data["metrics"]),
            "precision_mean": float(np.nanmean(metrics_arr[:, 0])) if metrics_arr.size else None,
            "precision_std": float(np.nanstd(metrics_arr[:, 0])) if metrics_arr.size else None,
            "recall_mean": float(np.nanmean(metrics_arr[:, 1])) if metrics_arr.size else None,
            "recall_std": float(np.nanstd(metrics_arr[:, 1])) if metrics_arr.size else None,
            "f1_mean": float(np.nanmean(metrics_arr[:, 2])) if metrics_arr.size else None,
            "f1_std": float(np.nanstd(metrics_arr[:, 2])) if metrics_arr.size else None,
        }
        
        # Agregar info de augmentación si está disponible
        if data["augmentation_params"]:
            subset_summary["augmentation_summary"] = data["augmentation_params"][0]  # Usar primero como referencia
        
        summary[subset_name] = subset_summary
    
    save_json_safe(subject_out / "summary.json", summary)
    print(f"  Saved: {subject_out / 'summary.json'}")
    
    return {
        "vocales": data_vocales,
        "comandos": data_comandos,
        "summary": summary
    }


def plot_global_metrics_boxplots(all_subjects_data: Dict, output_path: Path):
    fig, ax = plt.subplots(figsize=(12, 6))
    all_data, labels, positions, plot_colors = [], [], [], []
    pos = 0
    metric_colors = COLORS[:3]

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
                plot_colors.append(metric_colors[i])
                pos += 1
        pos += 1

    if all_data:
        apply_custom_boxplot(ax, all_data, positions, labels, plot_colors)
        
        # limites verticales
        ax.set_ylabel("Score")
        
        # Calcular el nivel de azar para vocales y comandos
        if subset_name == "vocales":
            chance_level = 1.0 / 5
        else: 
            chance_level = 1.0 / 6
        # Determinar los límites del eje y
        min_val_data = np.nanmin([np.nanmin(d) for d in all_data if len(d) > 0]) if all_data else np.inf
        max_val_data = np.nanmax([np.nanmax(d) for d in all_data if len(d) > 0]) if all_data else -np.inf
        # Asegurarse de que los límites incluyan los niveles de azar y los datos
        lower_limit = max(0.0, min(min_val_data, chance_level) - 0.05)
        upper_limit = min(1.0, max(max_val_data, chance_level) + 0.05)
        ax.set_ylim(lower_limit, upper_limit)

        ax.set_title("Global Metrics Distribution (Across Subjects)", fontweight='bold')
    
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_global_confusion_matrices(all_subjects_data_raw: Dict, output_path: Path):
    """Genera matrices de confusión globales."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, subset_name, class_names in zip(axes, SUBSETS, [VOCAL_CLASS_NAMES, COMANDO_CLASS_NAMES]):
        cm_list = []
        
        for subj_name, subj_data in all_subjects_data_raw.items():
            if subj_data is None:
                continue
            subset_data = subj_data.get(subset_name)
            if subset_data is None or not subset_data["confusion_matrices"]:
                continue
            
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
    """Genera boxplots globales de accuracy por banda (promedio por sujeto)."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for ax, subset_name, n_classes in zip(axes, SUBSETS, [5, 6]):
        band_data_per_subject = []
        
        for subj_name, subj_data in all_subjects_data_raw.items():
            if subj_data is None: continue
            subset_data = subj_data.get(subset_name)
            if subset_data is None or not subset_data["band_accuracies"]:
                continue
            
            # Promediamos los folds para obtener un único valor por banda para este sujeto
            band_arr = np.vstack(subset_data["band_accuracies"])
            band_mean_subj = np.nanmean(band_arr, axis=0)
            band_data_per_subject.append(band_mean_subj)
        
        if not band_data_per_subject:
            ax.axis('off')
            continue
        
        band_matrix = np.vstack(band_data_per_subject) # Shape: (n_subjects, 5)
        
        # Datos para aplicar el estilo
        plot_data = [band_matrix[:, i] for i in range(len(BAND_LABELS))]
        positions = np.arange(len(BAND_LABELS))
        
        # Aplicamos el estilo con jitter (cada punto es un sujeto)
        apply_custom_boxplot(ax, plot_data, positions, BAND_FREQ_LABELS, COLORS, widths=0.3)
        
        # Nivel de azar
        chance_level = 1.0 / n_classes
        ax.axhline(y=chance_level, color='gray', linestyle=':', linewidth=2, 
                   label=f'Chance ({chance_level:.2f})', alpha=0.7)
        
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.set_title(f"{subset_name.capitalize()}", fontsize=12, fontweight='bold')
        
        # limites verticales
        min_val = min(np.nanmin(plot_data) if plot_data else np.inf, chance_level)
        max_val = max(np.nanmax(plot_data) if plot_data else -np.inf, chance_level)
        lower_bound = max(0.0, min_val - 0.05)
        upper_bound = min(1.0, max_val + 0.05)
        ax.set_ylim(lower_bound, upper_bound)

        ax.legend(loc='best', fontsize=8)
    
    fig.suptitle("Global Accuracy by Frequency Band (Across Subjects)", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_global_ftsurrogate_accuracy_boxplots(all_subjects_data_raw: Dict, output_path: Path):
    """Genera boxplots globales de accuracy con y sin FT Surrogate."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    for ax, subset_name, n_classes in zip(axes, SUBSETS, [5, 6]):
        with_fts_per_subject = []
        without_fts_per_subject = []
        
        for subj_name, subj_data in all_subjects_data_raw.items():
            if subj_data is None: continue
            subset_data = subj_data.get(subset_name)
            if subset_data is None or not subset_data["ftsurrogate_accuracies"]:
                continue
            
            # Extraemos la media de los folds para este sujeto
            with_fts = [d["with_fts"] for d in subset_data["ftsurrogate_accuracies"] if not np.isnan(d["with_fts"])]
            without_fts = [d["without_fts"] for d in subset_data["ftsurrogate_accuracies"] if not np.isnan(d["without_fts"])]
            
            if with_fts:
                with_fts_per_subject.append(np.nanmean(with_fts))
            if without_fts:
                without_fts_per_subject.append(np.nanmean(without_fts))
        
        if not with_fts_per_subject and not without_fts_per_subject:
            ax.axis('off')
            continue
        
        # Graficamos la comparativa
        plot_data = [without_fts_per_subject, with_fts_per_subject]
        labels = ["Without FTS", "With FTS"]
        
        # Usamos los colores de los extremos para el contraste visual
        apply_custom_boxplot(ax, plot_data, [0, 1], labels, [COLORS[0], COLORS[4]], widths=0.4)
        
        chance_level = 1.0 / n_classes
        ax.axhline(y=chance_level, color='gray', linestyle=':', linewidth=2)
        
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.set_title(f"{subset_name.capitalize()}", fontsize=12, fontweight='bold')

        # limites verticales
        min_val = min(np.nanmin(with_fts_per_subject) if with_fts_per_subject else np.inf, 
                      np.nanmin(without_fts_per_subject) if without_fts_per_subject else np.inf, 
                      chance_level)
        max_val = max(np.nanmax(with_fts_per_subject) if with_fts_per_subject else -np.inf, 
                      np.nanmax(without_fts_per_subject) if without_fts_per_subject else -np.inf, 
                      chance_level)
        lower_bound = max(0.0, min_val - 0.05)
        upper_bound = min(1.0, max_val + 0.05)
        ax.set_ylim(lower_bound, upper_bound)
    
    fig.suptitle("Global Accuracy by FT Surrogate Usage (Across Subjects)", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_global_temporal_window_boxplots(all_subjects_data_raw: Dict, output_path: Path):
    """Genera boxplots globales de accuracy por ventana temporal (promedio entre sujetos)."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for ax, subset_name, n_classes in zip(axes, SUBSETS, [5, 6]):
        window_data_per_subject = []
        
        for subj_name, subj_data in all_subjects_data_raw.items():
            if subj_data is None: continue
            subset_data = subj_data.get(subset_name)
            if subset_data is None or not subset_data["window_accuracies"]:
                continue
            
            # Promediamos los folds de cada sujeto para tener un punto por sujeto en el global
            window_arr = np.vstack(subset_data["window_accuracies"])
            window_mean_subj = np.nanmean(window_arr, axis=0)
            window_data_per_subject.append(window_mean_subj)
        
        if not window_data_per_subject:
            ax.axis('off')
            continue
        
        window_matrix = np.vstack(window_data_per_subject)
        n_windows = window_matrix.shape[1]
        
        # Datos para graficar
        plot_data = [window_matrix[:, i] for i in range(n_windows)]
        positions = np.arange(n_windows)
        window_labels = [f"Win {i}\n({i*0.5:.1f}-{i*0.5+1.5:.1f}s)" for i in range(n_windows)]
        plot_colors = [COLORS[i % len(COLORS)] for i in range(n_windows)]
        
        # Aplicamos el estilo (cada punto jitter es un sujeto diferente)
        apply_custom_boxplot(ax, plot_data, positions, window_labels, plot_colors, widths=0.3)
        
        chance_level = 1.0 / n_classes
        ax.axhline(y=chance_level, color='gray', linestyle=':', linewidth=2,
                   label=f'Chance ({chance_level:.2f})', alpha=0.7)
        
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.set_title(f"{subset_name.capitalize()}", fontsize=12, fontweight='bold')
        
        # limites verticales
        min_val = min(np.nanmin(plot_data) if plot_data else np.inf, chance_level)
        max_val = max(np.nanmax(plot_data) if plot_data else -np.inf, chance_level)
        lower_bound = max(0.0, min_val - 0.05)
        upper_bound = min(1.0, max_val + 0.05)
        ax.set_ylim(lower_bound, upper_bound)

        ax.tick_params(axis='x', rotation=30, labelsize=8)
        ax.legend(loc='best', fontsize=8)
    
    fig.suptitle("Global Accuracy by Temporal Window (Across Subjects)", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ==================== MAIN ====================

if __name__ == "__main__":
    EXP_ROOT = find_latest_experiment(EXPERIMENTS_ROOT, EXPERIMENT_NAME_PREFIX)
    print(f"[Visualizer] Using experiment root: {EXP_ROOT}")
    
    OUTPUT_ROOT = EXP_ROOT / OUTPUT_SUBDIR
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    subject_dirs = sorted([p for p in EXP_ROOT.iterdir() 
                          if p.is_dir() and p.name.upper().startswith("S")])
    print(f"[Visualizer] Found {len(subject_dirs)} subject directories")
    
    all_subjects_data = {}
    
    for subject_dir in subject_dirs:
        subject_name = subject_dir.name
        data = process_subject(subject_dir, OUTPUT_ROOT)
        
        if data is not None:
            all_subjects_data[subject_name] = data
    
    print("\n[Visualizer] Generating global visualizations...")
    global_out = OUTPUT_ROOT / "global"
    global_out.mkdir(parents=True, exist_ok=True)
    
    plot_global_metrics_boxplots(all_subjects_data, 
                                 global_out / "metrics_boxplots_global.png")
    
    plot_global_confusion_matrices(all_subjects_data,
                                   global_out / "confusion_matrices_global.png")
    
    plot_global_band_accuracy_boxplots(all_subjects_data,
                                       global_out / "band_accuracy_boxplots_global.png")
    
    plot_global_ftsurrogate_accuracy_boxplots(all_subjects_data,
                                             global_out / "ftsurrogate_accuracy_boxplots_global.png")
    
    plot_global_temporal_window_boxplots(all_subjects_data,
                                        global_out / "temporal_window_boxplots_global.png")
    
    global_summary = {
        "experiment_root": str(EXP_ROOT),
        "n_subjects": len(all_subjects_data),
        "subjects": list(all_subjects_data.keys())
    }
    
    save_json_safe(global_out / "global_summary.json", global_summary)
    print(f"  Saved: {global_out / 'global_summary.json'}")
    
    print(f"\n[Visualizer] Done! Results saved to: {OUTPUT_ROOT}")