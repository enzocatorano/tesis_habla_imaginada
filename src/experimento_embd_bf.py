# experimento_embd_bf.py
"""
Lanzador de experimentos de Machine Learning Clásico.
Específico para el modelo ESMB_BR (Ensamble de Árboles LogitBoost).
"""

import os
import sys
import time
import json
import traceback
import random
import re
import platform
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

try:
    from models import ESMB_BR
except ImportError:
    raise ImportError("No pude importar ESMB_BR. Asegurate de que models.py esté en el path.")

# -----------------------------
# CONFIGURACIÓN (modificar aquí)
# -----------------------------
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "preprocesamiento_segun_bolanos_rufiner"
EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1] / "experiments"
EXPERIMENT_NAME = "S01_ESMB_BR_prueba"
SUFIJO_DATOS = '_preprocessed'

# Nombre de los arrays dentro del .npz
NOMBRE_ARRAY_DATOS, NOMBRE_ARRAY_ETIQUETAS = "x", "y"

# Control de experimento
MASTER_SEED = 42
N_SEEDS = 1
K_FOLDS = 5
VAL_FRAC = 0.1 # Fracción de entrenamiento que se reserva para validación

# Selección de Sujetos: None => todos; o lista de enteros p.ej. [1, 2]
SUBJECT = [1]

# -----------------------------
# CONFIGURACIÓN DE ETIQUETAS
# -----------------------------
# Opciones de TARGET_LABEL: "modalidad" (col 0), "estimulo" (col 1), "artefacto" (col 2)
TARGET_LABEL = "estimulo" 
# Si TARGET_LABEL es "estimulo", definimos qué subconjuntos probar (puedes dejar uno o ambos)
ESTIMULO_SUBSETS = ["vocales", "comandos"] 

# -----------------------------
# CONFIGURACIÓN DEL MODELO
# -----------------------------
# Parámetros editables del ensamble ESMB_BR
MODEL_KWARGS = dict(
    learning_cycles=11,
    learning_rate=0.12,
    max_depth=10
)

# -----------------------------
# FUNCIONES AUXILIARES
# -----------------------------
def now_timestamp():
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime())

def set_global_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def make_experiment_root(base: Path, name: str):
    ts = now_timestamp()
    host = platform.node().replace(" ", "_")
    exp_dir = base / f"{name}_{ts}_{host}"
    exp_dir.mkdir(parents=True, exist_ok=False)
    return exp_dir

def save_json(path: Path, obj):
    with open(path, "w", encoding="utf8") as fh:
        json.dump(obj, fh, indent=2, default=lambda o: (o.tolist() if isinstance(o, (np.ndarray,)) else str(o)))

def aplicar_normalizacion_zscore(X_train, X_val, X_test):
    """
    Normaliza por Z-Score (media 0, desviación 1).
    Los parámetros se aprenden ÚNICAMENTE de X_train para evitar data leakage.
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-8] = 1e-8 # Evitar división por cero
    
    X_t_norm = (X_train - mean) / std
    X_v_norm = (X_val - mean) / std if X_val.size else X_val
    X_test_norm = (X_test - mean) / std if X_test.size else X_test
    
    return X_t_norm, X_v_norm, X_test_norm, mean, std

def _extract_subject_number_from_filename(path: Path):
    m = re.match(r"^[sS](\d+)", path.name)
    return int(m.group(1)) if m else None

# -----------------------------
# PREPARATIVOS
# -----------------------------
if MASTER_SEED is not None:
    set_global_seed(MASTER_SEED)
    rng = np.random.default_rng(MASTER_SEED)
    SEEDS_LIST = rng.integers(low=0, high=2**31 - 1, size=N_SEEDS).tolist()
else:
    SEEDS_LIST = list(range(N_SEEDS))

EXPERIMENT_ROOT = make_experiment_root(EXPERIMENTS_ROOT, EXPERIMENT_NAME)
print(f"[Launcher] Experiment root: {EXPERIMENT_ROOT}")

# Guardar configuración base
config = {
    "experiment_name": EXPERIMENT_NAME,
    "target_label": TARGET_LABEL,
    "estimulo_subsets": ESTIMULO_SUBSETS,
    "n_seeds": N_SEEDS, "k_folds": K_FOLDS, "val_frac": VAL_FRAC,
    "subject_selection": SUBJECT, "model_kwargs": MODEL_KWARGS,
    "master_seed": MASTER_SEED, "seeds_list": SEEDS_LIST
}
save_json(EXPERIMENT_ROOT / "experiment_config.json", config)

# Búsqueda de archivos
all_files = sorted([p for p in DATA_DIR.iterdir() if p.is_file() and p.suffix == ".npz"])
subject_files = [p for p in all_files if SUFIJO_DATOS.upper() in p.name.upper()]

if SUBJECT is not None:
    wanted = [SUBJECT] if isinstance(SUBJECT, int) else [int(x) for x in SUBJECT]
    subject_files = [p for p in subject_files if _extract_subject_number_from_filename(p) in wanted]

print(f"[Launcher] Found {len(subject_files)} subject files. Selected SUBJECT={SUBJECT}")

# -----------------------------
# BUCLE PRINCIPAL
# -----------------------------
for subj_idx, subj_path in enumerate(subject_files, start=1):
    subj_name = subj_path.stem
    print(f"\n=== SUBJECT {subj_idx}/{len(subject_files)}: {subj_name} ===")

    # Cargar datos
    data = np.load(subj_path, allow_pickle=True)
    X_all = data[NOMBRE_ARRAY_DATOS]     # Asumimos (trials, features)
    Y_all = data[NOMBRE_ARRAY_ETIQUETAS] # (trials, 3) -> [modalidad, estímulo, artefacto]
    
    # Aplanar características si por error vinieran en 3D (trials, canales, bins)
    if X_all.ndim == 3:
        X_all = X_all.reshape(X_all.shape[0], -1)

    subj_out = EXPERIMENT_ROOT / subj_name
    subj_out.mkdir(parents=True, exist_ok=True)

    # Definir subconjuntos iterables basados en la configuración
    subsets_to_run = []
    if TARGET_LABEL == "estimulo":
        if "vocales" in ESTIMULO_SUBSETS:
            subsets_to_run.append(("vocales", 1, 5, 5)) # nombre, min, max, n_classes
        if "comandos" in ESTIMULO_SUBSETS:
            subsets_to_run.append(("comandos", 6, 11, 6))
    elif TARGET_LABEL == "modalidad":
        subsets_to_run.append(("modalidad_all", None, None, 2)) # asumiendo binario 0/1
    elif TARGET_LABEL == "artefacto":
        subsets_to_run.append(("artefacto_all", None, None, 2)) # asumiendo binario 0/1

    for subset_name, stim_min, stim_max, n_classes in subsets_to_run:
        print(f"\n--- subset: {subset_name} (classes={n_classes}) ---")
        
        # Filtrado de muestras según la etiqueta objetivo
        if TARGET_LABEL == "estimulo":
            estimulo_col = Y_all[:, 1].astype(int)
            mask = (estimulo_col >= stim_min) & (estimulo_col <= stim_max)
            y_target = estimulo_col[mask] - stim_min # Mapear al rango 0 a (n_classes-1)
        else:
            mask = np.ones(Y_all.shape[0], dtype=bool)
            y_col = 0 if TARGET_LABEL == "modalidad" else 2
            y_target = Y_all[:, y_col].astype(int)

        if mask.sum() == 0:
            print(f"[Launcher] No samples for {subset_name}. Skipping.")
            continue

        X_subset = X_all[mask]
        
        subset_out = subj_out / subset_name
        subset_out.mkdir(parents=True, exist_ok=True)

        for seed_i in range(N_SEEDS):
            seed_val = int(SEEDS_LIST[seed_i])
            print(f">>> seed {seed_i} (val={seed_val})")
            seed_out = subset_out / f"seed_{seed_i}"
            seed_out.mkdir(parents=True, exist_ok=True)

            set_global_seed(seed_val)

            skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=seed_val)
            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_subset, y_target), start=1):
                fold_out = seed_out / f"fold_{fold_idx}"
                fold_out.mkdir(parents=True, exist_ok=True)
                
                metadata = {"status": "started", "error": None}

                try:
                    X_train_full = X_subset[train_idx]
                    y_train_full = y_target[train_idx]
                    X_test = X_subset[test_idx]
                    y_test = y_target[test_idx]

                    # Separar validación si es requerido
                    if VAL_FRAC > 0.0:
                        X_train, X_val, y_train, y_val = train_test_split(
                            X_train_full, y_train_full, test_size=VAL_FRAC, 
                            stratify=y_train_full, random_state=seed_val
                        )
                    else:
                        X_train, y_train = X_train_full, y_train_full
                        X_val, y_val = np.empty((0, X_train.shape[1])), np.array([])

                    # Normalización Z-Score
                    X_train, X_val, X_test, mean, std = aplicar_normalizacion_zscore(X_train, X_val, X_test)

                    # Instanciar y entrenar modelo ESMB_BR
                    t0 = time.time()
                    model = ESMB_BR(n_classes=n_classes, semilla=seed_val, **MODEL_KWARGS)
                    model.fit(X_train, y_train)
                    t1 = time.time()

                    # Evaluación en validación y test
                    y_pred_val = model.predict(X_val) if X_val.size else np.array([])
                    y_pred_test = model.predict(X_test)
                    
                    # Calcular métricas (pasamos 'labels' para que reconozca que existen n_classes válidas, 
                    # y así maneje la predicción de descarte '-1' correctamente)
                    valid_labels = np.arange(n_classes)
                    acc_test = accuracy_score(y_test, y_pred_test)
                    cm_test = confusion_matrix(y_test, y_pred_test, labels=valid_labels)
                    report_test = classification_report(y_test, y_pred_test, labels=valid_labels, output_dict=True, zero_division=0)
                    
                    # Guardar resultados del fold
                    np.savez_compressed(fold_out / "test_preds.npz", y_true=y_test, y_pred=y_pred_test)
                    np.save(fold_out / "confusion_matrix.npy", cm_test)
                    save_json(fold_out / "classification_report.json", report_test)
                    
                    # Tasa de rechazo (cuántos clasificó como -1)
                    rejection_rate = float(np.sum(y_pred_test == -1) / len(y_pred_test))

                    metadata.update({
                        "status": "success",
                        "train_time_s": t1 - t0,
                        "test_accuracy": float(acc_test),
                        "rejection_rate": rejection_rate,
                        "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test)
                    })
                    
                    print(f"    Fold {fold_idx}/{K_FOLDS} | Acc: {acc_test:.4f} | Rechazos: {rejection_rate:.2%}")

                except Exception:
                    metadata["status"] = "error"
                    metadata["error"] = traceback.format_exc()
                    print(f"[Error] Fold {fold_idx}: {metadata['error']}")
                finally:
                    save_json(fold_out / "metadata.json", metadata)

# Recopilación final
print("\n[Launcher] Todos los experimentos terminados. Generando índice de resumen...")
all_meta = []
for f in EXPERIMENT_ROOT.rglob("metadata.json"):
    try:
        all_meta.append(json.loads(f.read_text(encoding="utf8")))
    except Exception:
        pass

save_json(EXPERIMENT_ROOT / "summary_runs.json", {"n_runs_indexed": len(all_meta), "runs": all_meta})
print(f"Resumen guardado en: {EXPERIMENT_ROOT / 'summary_runs.json'}")