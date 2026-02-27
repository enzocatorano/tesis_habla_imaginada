"""
Análisis de precisión por ventana temporal.

Extrae la ventana temporal de origen (0-5) de cada trial augmentado
y analiza la precisión del modelo en función de la posición temporal.

Genera:
- Por sujeto: boxplots de accuracy por ventana (vocales y comandos)
- Global: boxplots agregados entre sujetos
- JSONs con estadísticas detalladas
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# ---------------- CONFIG ----------------
EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1] / "experiments"
EXPERIMENT_NAME_PREFIX = "EEGNet_full_baseline"
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "preproc_aug_segm_gnperband_fts"
OUTPUT_SUBDIR = "temporal_window_analysis"

SUBSETS = ["vocales", "comandos"]
N_WINDOWS = 6  # Número de ventanas temporales por trial original
AUGMENTATION_FACTOR_PER_WINDOW = 20  # 5 bandas × 4 versiones FTS

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 200
# ----------------------------------------


def find_latest_experiment(root: Path, prefix: str) -> Path:
    """Encuentra el experimento más reciente."""
    candidates = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not candidates:
        raise FileNotFoundError(f"No experiment folders starting with '{prefix}' found")
    return sorted(candidates, key=lambda p: p.name)[-1]


def load_json_safe(path: Path):
    """Carga JSON de forma segura."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf8"))
    except Exception:
        return None


def save_json_safe(path: Path, obj):
    """Guarda JSON de forma segura."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf8") as fh:
        json.dump(obj, fh, indent=2, default=lambda o: (o.tolist() if isinstance(o, np.ndarray) else str(o)))


def infer_temporal_window(global_idx: int) -> int:
    """
    Infiere la ventana temporal (0-5) a partir del índice global en el dataset.
    
    Estructura del dataset augmentado:
    - Cada trial original genera 120 trials augmentados
    - Primeros 20: ventana 0 (5 bandas × 4 FTS)
    - Siguientes 20: ventana 1
    - ...
    - Últimos 20: ventana 5
    
    Args:
        global_idx: índice en el dataset augmentado
    
    Returns:
        ventana temporal (0-5)
    """
    # Identificar qué grupo de 20 dentro del bloque de 120
    position_in_trial = (global_idx // AUGMENTATION_FACTOR_PER_WINDOW) % N_WINDOWS
    return int(position_in_trial)


def collect_window_accuracies(subject_dir: Path, subset: str) -> Dict:
    """
    Recopila accuracies por ventana temporal para un sujeto y subset.
    
    Returns:
        dict con:
            - window_accuracies: lista de arrays (n_folds, n_windows)
            - raw_predictions: lista de dicts con predicciones por fold
    """
    subset_dir = subject_dir / subset
    if not subset_dir.exists():
        return None
    
    window_acc_per_fold = []
    raw_predictions_per_fold = []
    
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
            
            test_idx_global = meta.get("test_idx_global")
            if test_idx_global is None:
                continue
            
            # Cargar predicciones
            preds_path = fold_dir / "test_preds.npz"
            if not preds_path.exists():
                continue
            
            try:
                d = np.load(preds_path, allow_pickle=True)
                y_true = d["y_true"]
                y_pred = d["y_pred"]
                
                # Inferir ventana temporal para cada trial del test set
                windows = np.array([infer_temporal_window(idx) for idx in test_idx_global])
                
                # Calcular accuracy por ventana
                window_accs = []
                for w in range(N_WINDOWS):
                    mask = (windows == w)
                    if mask.sum() == 0:
                        window_accs.append(np.nan)
                    else:
                        acc = float(np.mean(y_true[mask] == y_pred[mask]))
                        window_accs.append(acc)
                
                window_acc_per_fold.append(np.array(window_accs))
                
                # Guardar datos raw para análisis adicional
                raw_predictions_per_fold.append({
                    "windows": windows,
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "fold_path": str(fold_dir)
                })
                
            except Exception as e:
                print(f"  Warning: Error processing fold {fold_dir}: {e}")
                continue
    
    if not window_acc_per_fold:
        return None
    
    return {
        "window_accuracies": window_acc_per_fold,
        "raw_predictions": raw_predictions_per_fold
    }


def plot_temporal_window_boxplots(data_vocales, data_comandos, output_path: Path, subject_name: str):
    """Genera boxplots de accuracy por ventana temporal."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    window_labels = [f"Win {i}\n({i*0.5:.1f}-{i*0.5+1.5:.1f}s)" for i in range(N_WINDOWS)]
    
    for ax, data, subset_name, n_classes in zip(
        axes, 
        [data_vocales, data_comandos], 
        ["Vocales", "Comandos"],
        [5, 6]
    ):
        if data is None or not data["window_accuracies"]:
            ax.text(0.5, 0.5, f"No data for {subset_name}", ha='center', va='center')
            ax.axis('off')
            continue
        
        # Apilar accuracies: filas = folds, columnas = ventanas
        window_arr = np.vstack(data["window_accuracies"])  # shape: (n_folds, n_windows)
        
        # Crear boxplot
        bp = ax.boxplot([window_arr[:, i] for i in range(N_WINDOWS)],
                       labels=window_labels,
                       patch_artist=True,
                       showmeans=True,
                       meanline=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7, linewidth=1.5),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5),
                       medianprops=dict(linewidth=2, color='darkblue'),
                       meanprops=dict(linewidth=2, color='red', linestyle='--'))
        
        # Límites dinámicos
        chance_level = 1.0 / n_classes
        data_min = np.nanmin(window_arr)
        data_max = np.nanmax(window_arr)
        y_min = max(0, min(chance_level, data_min) * 0.9)
        y_max = min(1, max(chance_level, data_max) * 1.1)
        ax.set_ylim(y_min, y_max)
        
        # Línea de chance level
        ax.axhline(y=chance_level, color='red', linestyle=':', linewidth=2,
                  label=f'Chance level ({chance_level:.3f})', alpha=0.7)
        
        ax.set_xlabel("Temporal Window", fontsize=10)
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_title(f"{subset_name}", fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=30, labelsize=8)
        ax.legend(loc='best', fontsize=8)
    
    fig.suptitle(f"Accuracy by Temporal Window - {subject_name}", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_global_temporal_window_boxplots(all_subjects_data: Dict, output_path: Path):
    """Genera boxplots globales de accuracy por ventana temporal entre sujetos."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    window_labels = [f"Win {i}\n({i*0.5:.1f}-{i*0.5+1.5:.1f}s)" for i in range(N_WINDOWS)]
    
    for ax, subset_name, n_classes in zip(axes, SUBSETS, [5, 6]):
        window_data_per_subject = []
        
        for subj_name, subj_data in all_subjects_data.items():
            subset_data = subj_data.get(subset_name)
            if subset_data is None or not subset_data["window_accuracies"]:
                continue
            
            # Promediar ventanas de este sujeto a través de folds
            window_arr = np.vstack(subset_data["window_accuracies"])
            window_mean_subj = np.nanmean(window_arr, axis=0)  # (n_windows,)
            window_data_per_subject.append(window_mean_subj)
        
        if not window_data_per_subject:
            ax.text(0.5, 0.5, f"No data for {subset_name}", ha='center', va='center')
            ax.axis('off')
            continue
        
        # Apilar: filas = sujetos, columnas = ventanas
        window_matrix = np.vstack(window_data_per_subject)  # (n_subjects, n_windows)
        
        bp = ax.boxplot([window_matrix[:, i] for i in range(N_WINDOWS)],
                       labels=window_labels,
                       patch_artist=True,
                       showmeans=True,
                       meanline=True,
                       boxprops=dict(facecolor='lightgreen', alpha=0.7, linewidth=1.5),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5),
                       medianprops=dict(linewidth=2, color='darkgreen'),
                       meanprops=dict(linewidth=2, color='red', linestyle='--'))
        
        # Límites dinámicos
        chance_level = 1.0 / n_classes
        data_min = np.nanmin(window_matrix)
        data_max = np.nanmax(window_matrix)
        y_min = max(0, min(chance_level, data_min) * 0.9)
        y_max = min(1, max(chance_level, data_max) * 1.1)
        ax.set_ylim(y_min, y_max)
        
        # Línea de chance level
        ax.axhline(y=chance_level, color='gray', linestyle=':', linewidth=2,
                  label=f'Chance level ({chance_level:.3f})', alpha=0.7)
        
        ax.set_xlabel("Temporal Window", fontsize=10)
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_title(f"{subset_name.capitalize()}", fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=30, labelsize=8)
        ax.legend(loc='best', fontsize=8)
    
    fig.suptitle("Global Accuracy by Temporal Window (Across Subjects)", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def process_subject(subject_dir: Path, output_root: Path):
    """Procesa un sujeto y genera visualizaciones de ventanas temporales."""
    subject_name = subject_dir.name
    print(f"\n[Temporal Window Analysis] Processing subject: {subject_name}")
    
    subject_out = output_root / subject_name
    subject_out.mkdir(parents=True, exist_ok=True)
    
    # Recopilar datos
    data_vocales = collect_window_accuracies(subject_dir, "vocales")
    data_comandos = collect_window_accuracies(subject_dir, "comandos")
    
    if data_vocales is None and data_comandos is None:
        print(f"  No data found for {subject_name}, skipping...")
        return None
    
    # Generar boxplots
    plot_temporal_window_boxplots(data_vocales, data_comandos,
                                  subject_out / "temporal_window_boxplots.png",
                                  subject_name)
    
    # Guardar resumen JSON
    summary = {}
    for subset_name, data in [("vocales", data_vocales), ("comandos", data_comandos)]:
        if data is None:
            continue
        
        window_arr = np.vstack(data["window_accuracies"])
        
        summary[subset_name] = {
            "n_folds": len(data["window_accuracies"]),
            "window_accuracies_mean": np.nanmean(window_arr, axis=0).tolist(),
            "window_accuracies_std": np.nanstd(window_arr, axis=0).tolist(),
            "window_accuracies_min": np.nanmin(window_arr, axis=0).tolist(),
            "window_accuracies_max": np.nanmax(window_arr, axis=0).tolist(),
        }
    
    save_json_safe(subject_out / "temporal_window_summary.json", summary)
    print(f"  Saved: {subject_out / 'temporal_window_summary.json'}")
    
    return {
        "vocales": data_vocales,
        "comandos": data_comandos
    }


# ==================== MAIN ====================

if __name__ == "__main__":
    # Encontrar experimento
    EXP_ROOT = find_latest_experiment(EXPERIMENTS_ROOT, EXPERIMENT_NAME_PREFIX)
    print(f"[Temporal Window Analysis] Using experiment root: {EXP_ROOT}")
    
    OUTPUT_ROOT = EXP_ROOT / OUTPUT_SUBDIR
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    # Descubrir sujetos
    subject_dirs = sorted([p for p in EXP_ROOT.iterdir() 
                          if p.is_dir() and p.name.upper().startswith("S")])
    print(f"[Temporal Window Analysis] Found {len(subject_dirs)} subject directories")
    
    # Procesar cada sujeto
    all_subjects_data = {}
    
    for subject_dir in subject_dirs:
        subject_name = subject_dir.name
        data = process_subject(subject_dir, OUTPUT_ROOT)
        
        if data is not None:
            all_subjects_data[subject_name] = data
    
    # Generar visualizaciones globales
    print("\n[Temporal Window Analysis] Generating global visualizations...")
    global_out = OUTPUT_ROOT / "global"
    global_out.mkdir(parents=True, exist_ok=True)
    
    plot_global_temporal_window_boxplots(all_subjects_data,
                                         global_out / "temporal_window_boxplots_global.png")
    
    # Guardar resumen global
    global_summary = {
        "experiment_root": str(EXP_ROOT),
        "n_subjects": len(all_subjects_data),
        "subjects": list(all_subjects_data.keys()),
    }
    
    # Agregar estadísticas por subset
    for subset_name in SUBSETS:
        window_means_per_subject = []
        
        for subj_name, subj_data in all_subjects_data.items():
            subset_data = subj_data.get(subset_name)
            if subset_data is None or not subset_data["window_accuracies"]:
                continue
            
            window_arr = np.vstack(subset_data["window_accuracies"])
            window_mean = np.nanmean(window_arr, axis=0)
            window_means_per_subject.append(window_mean)
        
        if window_means_per_subject:
            window_matrix = np.vstack(window_means_per_subject)
            global_summary[subset_name] = {
                "n_subjects": len(window_means_per_subject),
                "window_accuracies_mean_across_subjects": np.nanmean(window_matrix, axis=0).tolist(),
                "window_accuracies_std_across_subjects": np.nanstd(window_matrix, axis=0).tolist(),
            }
    
    save_json_safe(global_out / "temporal_window_global_summary.json", global_summary)
    print(f"  Saved: {global_out / 'temporal_window_global_summary.json'}")
    
    print(f"\n[Temporal Window Analysis] Done! Results saved to: {OUTPUT_ROOT}")