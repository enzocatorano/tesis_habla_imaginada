"""
experiment_runner.py
====================
Lanzador de experimentos para la replicación de Bolaños y Rufiner.

Ejecuta experiments de clasificación con:
  - Sujetos configurable (lista o "full")
  - Subsets: vocales (1-5) o comandos (6-11)
  - Modalidad: imaginada, pronunciada, o ambas
  - Dos modos de clasificación: binary_per_class | multiclass
  - Múltiples semillas y K-folds
  - Manejo de rechazos configurable (solo para binary)

Uso:
  python experiment_runner.py
"""

import os
import sys
import json
import time
import traceback
import random
import platform
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from data_loader import load_subject_data, SUBSET_CONFIG
from model import ESMB_BR_Binary, ESMB_BR_Multiclass

########################################################################################
########################################################################################
# CABEZAL: Configuración del experimento
########################################################################################

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "preprocesamiento_segun_bolanos_rufiner"
DATA_SUFFIX = "_preprocessed"

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[2] / "experiments"
EXPERIMENT_NAME = "like_bolanos_rufiner"

SUBJECTS = [1] # puede ser full

MODALIDAD_FILTER = 1 # 1, 2 o None para todos
SUBSETS = ["vocales"]

# binary_per_class o multiclass
CLASSIFICATION_MODE = "binary_per_class"

N_SEEDS = 1
K_FOLDS = 5

MODEL_KWARGS = dict(
    learning_cycles=11,
    learning_rate=0.12,
    max_depth=10,
)

# strict o zero_as_error
DISCARD_MODE = "strict"

########################################################################################
########################################################################################
# Funciones auxiliares
########################################################################################

def now_timestamp():
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime())


def set_global_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def make_experiment_root(base: Path, name: str) -> Path:
    ts = now_timestamp()
    host = platform.node().replace(" ", "_")
    exp_dir = base / f"{name}_{ts}_{host}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=lambda o: (
            o.tolist() if isinstance(o, (np.ndarray,)) else str(o)
        ))


def normalize_zscore(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-8] = 1e-8
    X_train_n = (X_train - mean) / std
    X_test_n = (X_test - mean) / std
    return X_train_n, X_test_n, mean, std


def evaluate_predictions(y_true, y_pred, mode, n_classes, discard_mode):
    """
    Evalúa predicciones diferenciando el manejo de rechazos.

    Parameters
    ----------
    y_true : array — ground truth (0..n_classes-1)
    y_pred : array — predicciones
        Para modo binario: puede contener -1 (descartado)
        Para multiclass: solo 0..n_classes-1
    mode : "binary_per_class" | "multiclass"
    n_classes : int
    discard_mode : "strict" | "zero_as_error" | "all_as_error"
        strict:        descartados (-1) excluidos del accuracy
        zero_as_error: descartados cuentan como error
        all_as_error:  descartados cuentan como error

    Returns
    -------
    dict con:
        accuracy, rejection_rate, n_valid, n_discarded, n_total,
        n_correct, n_incorrect, per_class_accuracy
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_total = len(y_true)

    if mode == "binary_per_class":
        n_valid = np.sum(y_pred != -1)
        n_discarded = np.sum(y_pred == -1)
        rejection_rate = n_discarded / n_total

        if discard_mode == "strict":
            mask_valid = y_pred != -1
            y_t_valid = y_true[mask_valid]
            y_p_valid = y_pred[mask_valid]
            n_valid_eval = len(y_t_valid)
            n_correct = int(np.sum(y_t_valid == y_p_valid))
            n_incorrect = n_valid_eval - n_correct
            accuracy = n_correct / n_valid_eval if n_valid_eval > 0 else 0.0
        else:
            n_incorrect = int(np.sum(y_pred != y_true))
            n_correct = int(np.sum(y_pred == y_true))
            accuracy = n_correct / n_total if n_total > 0 else 0.0

        per_class_acc = {}
        for c in range(n_classes):
            mask_c = y_true == c
            if mask_c.sum() == 0:
                per_class_acc[str(c)] = None
            else:
                per_class_acc[str(c)] = float(
                    np.mean(y_pred[mask_c] == c)
                )

        return {
            "accuracy": float(accuracy),
            "rejection_rate": float(rejection_rate),
            "n_valid": int(n_valid),
            "n_discarded": int(n_discarded),
            "n_total": int(n_total),
            "n_correct": int(n_correct),
            "n_incorrect": int(n_incorrect),
            "per_class_accuracy": per_class_acc,
        }

    else:
        n_correct = int(np.sum(y_pred == y_true))
        n_incorrect = int(np.sum(y_pred != y_true))
        accuracy = n_correct / n_total if n_total > 0 else 0.0
        rejection_rate = 0.0
        per_class_acc = {}
        for c in range(n_classes):
            mask_c = y_true == c
            if mask_c.sum() == 0:
                per_class_acc[str(c)] = None
            else:
                per_class_acc[str(c)] = float(
                    np.mean(y_pred[mask_c] == c)
                )
        return {
            "accuracy": float(accuracy),
            "rejection_rate": 0.0,
            "n_valid": int(n_total),
            "n_discarded": 0,
            "n_total": int(n_total),
            "n_correct": int(n_correct),
            "n_incorrect": int(n_incorrect),
            "per_class_accuracy": per_class_acc,
        }


########################################################################################
########################################################################################
# Preparativos
########################################################################################

if SUBJECTS == "full":
    SUBJECTS_LIST = list(range(1, 16))
else:
    SUBJECTS_LIST = [int(s) for s in SUBJECTS]

if MASTER_SEED := 42:
    set_global_seed(MASTER_SEED)
    rng = np.random.default_rng(MASTER_SEED)
    SEEDS_LIST = rng.integers(low=0, high=2**31 - 1, size=N_SEEDS).tolist()
else:
    SEEDS_LIST = list(range(N_SEEDS))

EXPERIMENT_ROOT = make_experiment_root(EXPERIMENTS_ROOT, EXPERIMENT_NAME)
print(f"[Launcher] Raíz del experimento: {EXPERIMENT_ROOT}")

config = {
    "experiment_name": EXPERIMENT_NAME,
    "classification_mode": CLASSIFICATION_MODE,
    "modalidad_filter": MODALIDAD_FILTER,
    "subsets": SUBSETS,
    "subjects": SUBJECTS_LIST if SUBJECTS != "full" else "full",
    "n_seeds": N_SEEDS,
    "k_folds": K_FOLDS,
    "model_kwargs": MODEL_KWARGS,
    "discard_mode": DISCARD_MODE,
    "master_seed": MASTER_SEED,
    "seeds_list": SEEDS_LIST,
    "timestamp": now_timestamp(),
    "hostname": platform.node(),
}
save_json(EXPERIMENT_ROOT / "experiment_config.json", config)

details_path = DATA_DIR / "details.json"
if details_path.exists():
    details = json.loads(details_path.read_text(encoding="utf-8"))
else:
    details = {}
save_json(EXPERIMENT_ROOT / "details.json", details)

########################################################################################
########################################################################################
# Bucle principal
########################################################################################

for subj in SUBJECTS_LIST:
    print(f"\n{'='*60}")
    print(f"SUJETO S{subj:02d}")
    print(f"{'='*60}")

    for subset_name in SUBSETS:
        cfg = SUBSET_CONFIG[subset_name]
        n_classes = cfg["n_classes"]
        class_names = cfg["names"]
        stim_min = cfg["stim_min"]
        stim_max = cfg["stim_max"]

        print(f"\n--- Subset: {subset_name} ({n_classes} clases) ---")

        try:
            X, y, meta = load_subject_data(
                subject=subj,
                data_dir=DATA_DIR,
                modalidad_filter=MODALIDAD_FILTER,
                subset=subset_name,
                suffix=DATA_SUFFIX,
            )
        except FileNotFoundError as e:
            print(f"  [ERROR] {e}")
            continue
        except ValueError as e:
            print(f"  [ADVERTENCIA] {e}")
            continue

        print(f"  Trials: {X.shape[0]}, Features: {X.shape[1]}")
        print(f"  Distribución: {dict(zip(*np.unique(y, return_counts=True)))}")

        subset_dir = EXPERIMENT_ROOT / f"S{subj:02d}" / subset_name
        subset_dir.mkdir(parents=True, exist_ok=True)

        save_json(subset_dir / "data_info.json", {
            "n_trials": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_classes": n_classes,
            "class_names": class_names,
            "class_distribution": {
                class_names[i]: int(c) for i, c in
                zip(*np.unique(y, return_counts=True))
            },
        })

        for seed_i, seed_val in enumerate(SEEDS_LIST):
            print(f"\n  >> Semilla {seed_i} (valor={seed_val})")
            set_global_seed(seed_val)
            seed_dir = subset_dir / f"seed_{seed_i}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=seed_val)

            for fold_idx, (train_idx, test_idx) in enumerate(
                skf.split(X, y), start=1
            ):
                print(f"    Fold {fold_idx}/{K_FOLDS}...", end=" ", flush=True)
                fold_dir = seed_dir / f"fold_{fold_idx}"
                fold_dir.mkdir(parents=True, exist_ok=True)

                metadata = {
                    "status": "started",
                    "subject": f"S{subj:02d}",
                    "subset": subset_name,
                    "seed_i": seed_i,
                    "seed_val": seed_val,
                    "fold": fold_idx,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "n_classes": n_classes,
                    "class_names": class_names,
                    "classification_mode": CLASSIFICATION_MODE,
                    "discard_mode": DISCARD_MODE if CLASSIFICATION_MODE == "binary_per_class" else None,
                    "model_kwargs": MODEL_KWARGS,
                }

                try:
                    X_train = X[train_idx]
                    y_train = y[train_idx]
                    X_test = X[test_idx]
                    y_test = y[test_idx]

                    X_train_n, X_test_n, mean, std = normalize_zscore(X_train, X_test)

                    t0 = time.time()

                    if CLASSIFICATION_MODE == "binary_per_class":
                        model = ESMB_BR_Binary(
                            n_classes=n_classes,
                            semilla=seed_val,
                            **MODEL_KWARGS,
                        )
                    else:
                        model = ESMB_BR_Multiclass(
                            n_classes=n_classes,
                            semilla=seed_val,
                            **MODEL_KWARGS,
                        )

                    model.fit(X_train_n, y_train)
                    y_pred = model.predict(X_test_n)

                    t1 = time.time()

                    if CLASSIFICATION_MODE == "binary_per_class":
                        activations = model.predict_proba_activations(X_test_n)
                    else:
                        activations = None

                    metrics = evaluate_predictions(
                        y_test, y_pred,
                        mode=CLASSIFICATION_MODE,
                        n_classes=n_classes,
                        discard_mode=DISCARD_MODE,
                    )

                    valid_labels = list(range(n_classes))
                    cm = confusion_matrix(y_test, y_pred, labels=valid_labels)
                    cr = classification_report(
                        y_test, y_pred, labels=valid_labels,
                        output_dict=True, zero_division=0,
                    )

                    np.savez_compressed(
                        fold_dir / "test_preds.npz",
                        y_true=y_test,
                        y_pred=y_pred,
                        activations=activations,
                    )
                    np.save(fold_dir / "confusion_matrix.npy", cm)
                    save_json(fold_dir / "classification_report.json", cr)
                    save_json(fold_dir / "metrics.json", metrics)

                    metadata.update({
                        "status": "success",
                        "train_time_s": float(t1 - t0),
                        "accuracy": metrics["accuracy"],
                        "rejection_rate": metrics["rejection_rate"],
                        "n_valid": metrics["n_valid"],
                        "n_discarded": metrics["n_discarded"],
                    })

                    print(
                        f"Acc={metrics['accuracy']:.4f} "
                        f"Rech={metrics['rejection_rate']:.2%} "
                        f"t={t1-t0:.1f}s"
                    )

                except Exception as e:
                    metadata["status"] = "error"
                    metadata["error"] = traceback.format_exc()
                    print(f"ERROR: {e}")

                finally:
                    save_json(fold_dir / "metadata.json", metadata)

print(f"\n{'='*60}")
print("EXPERIMENTO COMPLETO")
print(f"{'='*60}")
print(f"Resultados en: {EXPERIMENT_ROOT}")
