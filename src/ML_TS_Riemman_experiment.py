#!/usr/bin/env python3
# ML_TS_Riemman_experiment.py
"""
Lanzador de experimentos de Machine Learning Clásico sobre el espacio tangente
Riemanniano de matrices de covarianza (espacial o espectro-espacial) de cada trial.

MODO_MATRIZ selecciona qué dataset y qué bloques de la matriz se usan:
  "espacial"          -> data/clean_preprocessed_corr       (trials, 6, 6)          la matriz completa
  "espectro_espacial" -> data/clean_preprocessed_corr_bandas (trials, 30, 30)        la matriz completa (5 bandas x 6 canales)
  "espacial_banda"    -> data/clean_preprocessed_corr_bandas extrae 5 bloques 6x6 de la diagonal (uno por banda)
  "frecuencial_canal" -> data/clean_preprocessed_corr_bandas extrae 6 bloques 5x5 de la diagonal (uno por canal)

En todos los casos: y = (trials, 3) [modalidad, estímulo, artefacto]

Pipeline por fold (sin fuga de datos):
  1. Partición de TRIALS en train / val / test (StratifiedKFold + train_test_split).
  2. Cálculo de la media Riemanniana SOLO con las matrices de entrenamiento.
  3. Proyección log-map de los tres conjuntos al espacio tangente (pyriemann.tangent_space).
  4. (Opcional) Z-score de los vectores de features con parámetros del entrenamiento.
  5. Entrenamiento de los clasificadores activados en el cabezal y exportación de
     métricas + matrices de confusión por clasificador.
"""

import os
import re
import json
import time
import random
import platform
import traceback
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from pyriemann.utils.mean import mean_riemann
from pyriemann.tangentspace import tangent_space

# -----------------------------
# CONFIGURACIÓN (modificar aquí)
# -----------------------------
EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1] / "experiments"
EXPERIMENT_NAME = "ML_TS_Riemman_espectroespacial_separado"

SUFIJO_DATOS = "_clean"
NOMBRE_ARRAY_DATOS, NOMBRE_ARRAY_ETIQUETAS = "x", "y"

# Modo de matriz (qué subconjunto de casillas se usa; ver docstring):
#   "espacial", "espectro_espacial", "espacial_banda", "frecuencial_canal"
MODO_MATRIZ = "espectro_espacial"

N_BANDAS = 5
N_CANALES = 6
BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]
CHANNEL_NAMES = ["F3", "F4", "C3", "C4", "P3", "P4"]

# Tarea a resolver:
#   "separado"  -> subsets "vocales" (1-5) y "comandos" (6-11) por separado
#   "unificado" -> 11 clases de estímulo en un solo problema
#   "binario"   -> vocal vs comando (estímulos 1-5 -> 0 ; 6-11 -> 1)
EXPERIMENT_MODE = "separado"

# Control de experimento
MASTER_SEED = 42
N_SEEDS = 3
K_FOLDS = 5
VAL_FRAC = 0.2  # Fracción del entrenamiento reservada para validación

# Selección de sujetos: None => todos; o lista de enteros p.ej. [5]
SUBJECT = None

# -----------------------------
# CLASIFICADORES (habilitar/deshabilitar)
# -----------------------------
USE_LOGREG_L1 = True
USE_SVM_LINEAL = True
USE_KNN = True
USE_RANDOM_FOREST = True
USE_GRADIENT_BOOSTING = True

LOGREG_L1_KWARGS = dict(penalty="l1", solver="liblinear", C=1.0, max_iter=2000, class_weight="balanced")
SVM_LINEAL_KWARGS = dict(C=1.0, class_weight="balanced", max_iter=10000, dual="auto")
KNN_KWARGS = dict(n_neighbors=5, weights="distance", metric="minkowski")
RANDOM_FOREST_KWARGS = dict(n_estimators=100, max_depth=5, min_samples_leaf=2, random_state=0, class_weight="balanced", n_jobs=-1)
GRADIENT_BOOSTING_KWARGS = dict(n_estimators=100, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=0)

# -----------------------------
# POST-PROCESAMIENTO DE FEATURES
# -----------------------------
ZSCORE_FEATURES = False  # True => media/std aprendidas solo de X_train

# -----------------------------
# Helper funcs
# -----------------------------
def now_timestamp():
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime())

def set_global_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def data_dir_for_mode(modo: str) -> Path:
    """Selecciona el dataset de matrices según el modo."""
    base = Path(__file__).resolve().parents[1] / "data"
    if modo == "espacial":
        return base / "clean_preprocessed_corr"
    if modo in ("espectro_espacial", "espacial_banda", "frecuencial_canal"):
        return base / "clean_preprocessed_corr_bandas"
    raise ValueError(f"MODO_MATRIZ desconocido: {modo}. Válidos: espacial, espectro_espacial, espacial_banda, frecuencial_canal")

def bloques_diagonal(modo: str, X_all: np.ndarray):
    """
    Devuelve lista de (nombre_bloque, X_bloque).

      espacial / espectro_espacial -> [("", X_all)]              (matriz completa)
      espacial_banda               -> 5 bloques 6x6 de la diagonal (una por banda)
      frecuencial_canal            -> 6 bloques 5x5 de la diagonal (una por canal)

    X_all: (trials, N, N) con N = 6 (espacial) o 30 (espectro-espacial).
    """
    if modo in ("espacial", "espectro_espacial"):
        return [("", X_all)]

    if modo == "espacial_banda":
        if X_all.shape[1] != N_BANDAS * N_CANALES:
            raise ValueError(f"espacial_banda espera matrices {N_BANDAS * N_CANALES}x{N_BANDAS * N_CANALES}, se encontró {X_all.shape}")
        bloques = []
        for b, nombre in enumerate(BAND_NAMES):
            f0 = b * N_CANALES
            bloques.append((f"{nombre}_band", X_all[:, f0:f0 + N_CANALES, f0:f0 + N_CANALES]))
        return bloques

    if modo == "frecuencial_canal":
        if X_all.shape[1] != N_BANDAS * N_CANALES:
            raise ValueError(f"frecuencial_canal espera matrices {N_BANDAS * N_CANALES}x{N_BANDAS * N_CANALES}, se encontró {X_all.shape}")
        bloques = []
        for c, nombre in enumerate(CHANNEL_NAMES):
            bloques.append((f"{nombre}_chan", X_all[:, c::N_CANALES, c::N_CANALES]))
        return bloques

    raise ValueError(f"MODO_MATRIZ desconocido: {modo}")

def make_experiment_root(base: Path, name: str):
    ts = now_timestamp()
    host = platform.node().replace(" ", "_")
    exp_dir = base / f"{name}_{ts}_{host}"
    exp_dir.mkdir(parents=True, exist_ok=False)
    return exp_dir

def save_json(path: Path, obj):
    with open(path, "w", encoding="utf8") as fh:
        json.dump(obj, fh, indent=2, default=lambda o: (o.tolist() if isinstance(o, (np.ndarray,)) else str(o)))

def discover_subject_files(data_dir: Path):
    files = sorted([p for p in data_dir.iterdir() if p.is_file() and p.suffix == ".npz"])
    files = [p for p in files if p.name.upper().startswith("S") and SUFIJO_DATOS.upper() in p.name.upper()]
    return files

def _extract_subject_number_from_filename(path: Path):
    m = re.match(r"^[sS](\d+)", path.name)
    return int(m.group(1)) if m else None

def aplicar_zscore_features(X_train, X_val, X_test):
    """
    Z-score de los vectores de features. Parámetros (media/std) aprendidos
    ÚNICAMENTE de X_train para evitar data leakage.
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-8] = 1e-8

    X_tr = (X_train - mean) / std
    X_va = (X_val - mean) / std if X_val.size else X_val
    X_te = (X_test - mean) / std if X_test.size else X_test
    return X_tr, X_va, X_te, mean, std

# -----------------------------
# Definición de la tarea
# -----------------------------
def define_tarea(mode: str, Y_all: np.ndarray):
    """
    Devuelve lista de (subset_name, mask, y_target, n_classes).
    y_target se construye a partir de Y_all[:, 1] (estímulo).
    """
    if mode in ("separado", "unificado"):
        if mode == "separado":
            subsets = [
                ("vocales", 1, 5, 5),
                ("comandos", 6, 11, 6),
            ]
        else:
            subsets = [("estimulo_unificado", 1, 11, 11)]
        targets = []
        for nombre, stim_min, stim_max, n_classes in subsets:
            mask = (Y_all[:, 1].astype(int) >= stim_min) & (Y_all[:, 1].astype(int) <= stim_max)
            y_t = (Y_all[:, 1].astype(int) - stim_min)
            targets.append((nombre, mask, y_t, n_classes))
    elif mode == "binario":
        mask = np.ones(Y_all.shape[0], dtype=bool)
        y_t = (Y_all[:, 1].astype(int) >= 6).astype(int)
        targets = [("vocal_vs_comando", mask, y_t, 2)]
    else:
        raise ValueError(f"EXPERIMENT_MODE desconocido: {mode}. Valores válidos: separado, unificado, binario")
    return targets

# -----------------------------
# Clasificadores activados
# -----------------------------
def construir_clasificadores():
    clf = []
    if USE_LOGREG_L1:
        clf.append(("reg_log_L1", LogisticRegression(**LOGREG_L1_KWARGS)))
    if USE_SVM_LINEAL:
        clf.append(("svm_lineal", LinearSVC(**SVM_LINEAL_KWARGS)))
    if USE_KNN:
        clf.append(("knn", KNeighborsClassifier(**KNN_KWARGS)))
    if USE_RANDOM_FOREST:
        clf.append(("random_forest", RandomForestClassifier(**RANDOM_FOREST_KWARGS)))
    if USE_GRADIENT_BOOSTING:
        clf.append(("gradient_boosting", GradientBoostingClassifier(**GRADIENT_BOOSTING_KWARGS)))
    return clf

# -----------------------------
# INICIO DE EJECUCIÓN
# -----------------------------
if __name__ == '__main__':

    set_global_seed(MASTER_SEED)
    rng = np.random.default_rng(MASTER_SEED)
    SEEDS_LIST = rng.integers(low=0, high=2**31 - 1, size=N_SEEDS).tolist()

    EXPERIMENT_ROOT = make_experiment_root(EXPERIMENTS_ROOT, EXPERIMENT_NAME)
    print(f"[Launcher] Experiment root: {EXPERIMENT_ROOT}")

    DATA_DIR = data_dir_for_mode(MODO_MATRIZ)

    config = {
        "experiment_name": EXPERIMENT_NAME,
        "mode": EXPERIMENT_MODE,
        "modo_matriz": MODO_MATRIZ,
        "data_dir": str(DATA_DIR),
        "bands": BAND_NAMES,
        "channels": CHANNEL_NAMES,
        "n_seeds": N_SEEDS, "k_folds": K_FOLDS, "val_frac": VAL_FRAC,
        "subject_selection": SUBJECT,
        "zscore_features": ZSCORE_FEATURES,
        "classifiers": {
            "reg_log_L1": {"enabled": USE_LOGREG_L1, **LOGREG_L1_KWARGS},
            "svm_lineal": {"enabled": USE_SVM_LINEAL, **SVM_LINEAL_KWARGS},
            "knn": {"enabled": USE_KNN, **KNN_KWARGS},
            "random_forest": {"enabled": USE_RANDOM_FOREST, **RANDOM_FOREST_KWARGS},
            "gradient_boosting": {"enabled": USE_GRADIENT_BOOSTING, **GRADIENT_BOOSTING_KWARGS},
        },
        "master_seed": MASTER_SEED, "seeds_list": SEEDS_LIST
    }
    save_json(EXPERIMENT_ROOT / "experiment_config.json", config)

    subject_files = discover_subject_files(DATA_DIR)

    if SUBJECT is not None:
        wanted = [SUBJECT] if isinstance(SUBJECT, int) else [int(x) for x in SUBJECT]
        subject_files = [p for p in subject_files if _extract_subject_number_from_filename(p) in wanted]

    print(f"[Launcher] Found {len(subject_files)} subject files. MODE={EXPERIMENT_MODE} | MODO_MATRIZ={MODO_MATRIZ}")
    if not subject_files:
        raise FileNotFoundError(f"No se encontraron archivos {SUFIJO_DATOS}.npz en {DATA_DIR}")

    # -----------------------------
    # BUCLE PRINCIPAL
    # -----------------------------
    for subj_idx, subj_path in enumerate(subject_files, start=1):
        subj_name = subj_path.stem
        print(f"\n=== SUBJECT {subj_idx}/{len(subject_files)}: {subj_name} ===")

        data = np.load(subj_path, allow_pickle=True)
        X_all = data[NOMBRE_ARRAY_DATOS]    # (trials, n_canales, n_canales)
        Y_all = data[NOMBRE_ARRAY_ETIQUETAS]  # (trials, 3)

        if X_all.ndim != 3 or X_all.shape[1] != X_all.shape[2]:
            raise ValueError(f"[{subj_name}] Esperaba matrices (trials, N, N), se encontraron {X_all.shape}")

        subj_out = EXPERIMENT_ROOT / subj_name
        subj_out.mkdir(parents=True, exist_ok=True)

        bloques = bloques_diagonal(MODO_MATRIZ, X_all)

        tareas = define_tarea(EXPERIMENT_MODE, Y_all)

        for subset_name, mask, y_target, n_classes_target in tareas:
            if mask.sum() == 0:
                print(f"    No se encontraron trials para {subset_name}, saltando.")
                continue

            y_full = y_target[mask]

            subset_out = subj_out / subset_name
            subset_out.mkdir(parents=True, exist_ok=True)

            for bloque_name, X_bloque in bloques:
                X_subset = X_bloque[mask]
                m_size = X_subset.shape[1]

                if bloque_name:
                    block_out = subset_out / bloque_name
                    block_out.mkdir(parents=True, exist_ok=True)
                    print(f"\n--- Bloque {bloque_name}: matrices {m_size}x{m_size} ---")
                else:
                    block_out = subset_out

                for seed_i in range(N_SEEDS):
                    seed_val = int(SEEDS_LIST[seed_i])
                    print(f"\n>>> seed {seed_i} (val={seed_val})")
                    seed_out = block_out / f"seed_{seed_i}"
                    seed_out.mkdir(parents=True, exist_ok=True)

                    set_global_seed(seed_val)

                    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=seed_val)
                    splits = list(skf.split(X_subset, y_full))

                    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
                        print(f"\n>>> Fold {fold_idx}/{len(splits)}")
                        fold_out = seed_out / f"fold_{fold_idx}"
                        fold_out.mkdir(parents=True, exist_ok=True)

                        # 1. División de TRIALS (la partición se hace ANTES de cualquier operación topológica)
                        X_train_full = X_subset[train_idx]
                        y_train_full = y_full[train_idx]
                        X_test = X_subset[test_idx]
                        y_test = y_full[test_idx]

                        if VAL_FRAC > 0.0:
                            idx_pool = np.arange(len(X_train_full))
                            idx_tr, idx_va = train_test_split(
                                idx_pool, test_size=VAL_FRAC,
                                stratify=y_train_full, random_state=seed_val + fold_idx, shuffle=True
                            )
                            X_train = X_train_full[idx_tr]
                            y_train = y_train_full[idx_tr]
                            X_val = X_train_full[idx_va]
                            y_val = y_train_full[idx_va]
                        else:
                            X_train = X_train_full
                            y_train = y_train_full
                            X_val = np.empty((0, m_size, m_size))
                            y_val = np.array([])

                        metadata = {
                            "subject": subj_name,
                            "subset": subset_name,
                            "modo_matriz": MODO_MATRIZ,
                            "bloque": bloque_name if bloque_name else None,
                            "matriz_size": m_size,
                            "seed_val": seed_val,
                            "fold_idx": fold_idx,
                            "mode": EXPERIMENT_MODE,
                            "n_classes": n_classes_target,
                            "normalization_ref": "media_riemann_entrenamiento",
                            "zscore_features": ZSCORE_FEATURES,
                            "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
                            "status": "started"
                        }

                        try:
                            # 2. Media Riemanniana SOLO del entrenamiento (evita fuga de datos)
                            ref = mean_riemann(X_train)

                            # 3. Log-map + vectorización al espacio tangente (pyriemann)
                            F_train = tangent_space(X_train, ref)
                            F_test = tangent_space(X_test, ref)
                            if X_val.size:
                                F_val = tangent_space(X_val, ref)
                            else:
                                F_val = np.empty((0, F_train.shape[1]))

                            metadata["n_features_ts"] = int(F_train.shape[1])

                            # 4. Z-score de features (parámetros del entrenamiento)
                            if ZSCORE_FEATURES:
                                F_train, F_val, F_test, feat_mean, feat_std = aplicar_zscore_features(F_train, F_val, F_test)
                                metadata["zscore"] = {"mean_min": float(feat_mean.min()), "mean_max": float(feat_mean.max()),
                                                      "std_min": float(feat_std.min()), "std_max": float(feat_std.max())}

                            # 5. Entrenar/exportar cada clasificador activado
                            for clf_name, clf in construir_clasificadores():
                                clf_out = fold_out / clf_name
                                clf_out.mkdir(parents=True, exist_ok=True)
                                clf_meta = dict(metadata)
                                clf_meta["classifier"] = clf_name

                                try:
                                    t0 = time.time()
                                    clf.fit(F_train, y_train)
                                    t1 = time.time()

                                    y_pred_val = clf.predict(F_val) if F_val.size else np.array([])
                                    y_pred_test = clf.predict(F_test)

                                    valid_labels = np.arange(n_classes_target)
                                    acc_test = accuracy_score(y_test, y_pred_test)
                                    acc_val = accuracy_score(y_val, y_pred_val) if y_pred_val.size else None
                                    cm_test = confusion_matrix(y_test, y_pred_test, labels=valid_labels)
                                    report_test = classification_report(y_test, y_pred_test, labels=valid_labels, output_dict=True, zero_division=0)

                                    np.savez_compressed(clf_out / "test_preds.npz",
                                                        y_true=y_test, y_pred=y_pred_test,
                                                        y_true_val=y_val, y_pred_val=y_pred_val)
                                    np.save(clf_out / "confusion_matrix.npy", cm_test)
                                    save_json(clf_out / "classification_report.json", report_test)

                                    clf_meta.update({
                                        "status": "success",
                                        "train_time_s": float(t1 - t0),
                                        "test_accuracy": float(acc_test),
                                        "val_accuracy": (None if acc_val is None else float(acc_val))
                                    })
                                    print(f"    [{clf_name}] Acc test={acc_test:.4f} | Acc val={('%.4f' % acc_val) if acc_val is not None else 'N/A'}")

                                except Exception:
                                    clf_meta["status"] = "error"
                                    clf_meta["error"] = traceback.format_exc()
                                    print(f"    [Error] {clf_name}: {clf_meta['error']}")

                                save_json(clf_out / "metadata.json", clf_meta)

                        except Exception:
                            metadata["status"] = "error"
                            metadata["error"] = traceback.format_exc()
                            save_json(fold_out / "metadata.json", metadata)
                            print(f"    [Error] Fold {fold_idx}: {metadata['error']}")

    # -----------------------------
    # ÍNDICE DE RESUMEN
    # -----------------------------
    print("\n[Launcher] Todos los experimentos terminados. Generando índice de resumen...")
    all_meta = []
    for f in EXPERIMENT_ROOT.rglob("metadata.json"):
        try:
            all_meta.append(json.loads(f.read_text(encoding="utf8")))
        except Exception:
            pass

    save_json(EXPERIMENT_ROOT / "summary_runs.json", {"n_runs_indexed": len(all_meta), "runs": all_meta})
    print(f"Resumen guardado en: {EXPERIMENT_ROOT / 'summary_runs.json'}")