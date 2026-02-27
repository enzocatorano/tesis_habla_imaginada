#!/usr/bin/env python3
# run_experiments_with_online_augment.py
"""
Lanzador de experimentos con augmentación online y control de seed maestro.
"""

import os
import sys
import time
import json
import traceback
import random
from pathlib import Path
from pprint import pprint
import platform
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# Import Entrenador, Evaluador, EEGNet, Augmentar (Augmentar está en trainer.py según lo que compartiste)
try:
    from trainer import Entrenador, Evaluador, EEGNet, Augmentar
except Exception:
    try:
        from models import EEGNet
        from trainer import Entrenador, Evaluador, Augmentar
    except Exception as e:
        raise ImportError("No pude importar Entrenador/Evaluador/EEGNet/Augmentar. "
                          "Asegurate de que trainer.py (y models.py si corresponde) estén en el path.") from e

# -----------------------------
# CONFIGURACIÓN (modificar aquí)
# -----------------------------
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "preprocessed"
EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1] / "experiments"
EXPERIMENT_NAME = "EEGNet_S01_baseline_onlineAug_superaug"
SUFIJO_DATOS = '_preprocessed'
# El nombre debe ser S{numero de sujeto con 2 digitos}_{SUFIJO_DATOS}.npz
# por ejemplo: S01_preprocessed.npz
#NOMBRE_ARRAY_DATOS, NOMBRE_ARRAY_ETIQUETAS = "data", "labels"
NOMBRE_ARRAY_DATOS, NOMBRE_ARRAY_ETIQUETAS = "x", "y"
N_CHANNELS = 6

# experiment control
MASTER_SEED = 17    # setea a None si no querés seed maestro
DETERMINISTIC = False  # True => intenta operaciones deterministas en PyTorch (puede afectar performance)

N_SEEDS = 3
K_FOLDS = 6
VAL_FRAC = 0.1

# training hyperparams
BATCH_SIZE = 64
EPOCHS = 200
LR = 1e-3
PATIENCE = 10
DROPOUT = 0.5
HIDDEN_UNITS = None

MAX_SUBJECTS = 1   # None para todos, o int para debug
DEVICE = None         # None => autodetect
NUM_WORKERS = 0
SHUFFLE_TRAIN = True
SAVE_TRAIN_INDEX = True
SAVE_BEST_MODEL = False

EXTRA_MODEL_KWARGS = dict(F1=8, D=2, kernel_length=64, separable_kernel_length=16)

# Augmentation defaults (puedes modificarlos)
AUGMENT_KWARGS = dict(window_duration=2.0,
                      window_shift=1.0,
                      fs=128,
                      band_noise_factor_train=3/3,
                      band_noise_factor_eval=1.0,
                      fts_factor_train=3/3,
                      fts_factor_eval=1.0,
                      n_fts_versions=1,
                      noise_magnitude_relative=0.025,
                      save_metadata=True,
                      save_indices=SAVE_TRAIN_INDEX)

# -----------------------------
# Helper funcs: seeds, I/O, normalización
# -----------------------------
def now_timestamp():
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime())

def set_global_seed(seed: int, deterministic: bool = True):
    """Fija semillas en Python/NumPy/Torch y configura opciones deterministas en PyTorch."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # determinismo en PyTorch (puede lanzar excepción o degradar performance)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

def make_experiment_root(base: Path, name: str):
    ts = now_timestamp()
    host = platform.node().replace(" ", "_")
    exp_dir = base / f"{name}_{ts}_{host}"
    exp_dir.mkdir(parents=True, exist_ok=False)
    return exp_dir

def discover_subject_files(data_dir: Path):
    files = sorted([p for p in data_dir.iterdir() if p.is_file() and p.suffix == ".npz"])
    files = [p for p in files if p.name.upper().startswith("S") and SUFIJO_DATOS.upper() in p.name.upper()]
    return files

def save_json(path: Path, obj):
    with open(path, "w", encoding="utf8") as fh:
        json.dump(obj, fh, indent=2, default=lambda o: (o.tolist() if isinstance(o, (np.ndarray,)) else str(o)))

def save_experiment_config(exp_root: Path, master_seed, seeds_list):
    config = {
        "experiment_name": EXPERIMENT_NAME,
        "data_dir": str(DATA_DIR),
        "experiments_root": str(EXPERIMENTS_ROOT),
        "n_seeds": N_SEEDS,
        "k_folds": K_FOLDS,
        "val_frac": VAL_FRAC,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "patience": PATIENCE,
        "dropout": DROPOUT,
        "hidden_units": HIDDEN_UNITS,
        "max_subjects": MAX_SUBJECTS,
        "device": DEVICE,
        "num_workers": NUM_WORKERS,
        "shuffle_train": SHUFFLE_TRAIN,
        "save_train_index": SAVE_TRAIN_INDEX,
        "save_best_model": SAVE_BEST_MODEL,
        "extra_model_kwargs": EXTRA_MODEL_KWARGS,
        "augmentation_defaults": AUGMENT_KWARGS,
        "master_seed": master_seed,
        "seeds_list": seeds_list,
        "timestamp": now_timestamp(),
        "hostname": platform.node(),
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__
    }
    save_json(exp_root / "experiment_config.json", config)
    print(f"[Launcher] Saved experiment config to: {exp_root / 'experiment_config.json'}")

def compute_zscore_params(X_train):
    flat = X_train.reshape(X_train.shape[0], -1)
    mean = float(flat.mean())
    std = float(flat.std())
    if std < 1e-8:
        std = 1e-8
    return mean, std

def apply_zscore(X, mean, std):
    return (X - mean) / std

# -----------------------------
# Subsets definition
# -----------------------------
SUBSETS = {
    "vocales": dict(stim_min=1, stim_max=5, n_classes=5),
    "comandos": dict(stim_min=6, stim_max=11, n_classes=6)
}

# -----------------------------
# Device
# -----------------------------
if DEVICE is None:
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device_str = DEVICE
device = torch.device(device_str)
print(f"[Launcher] Using device: {device}")

# -----------------------------
# Prepare master seed and seeds list
# -----------------------------
if MASTER_SEED is not None:
    set_global_seed(MASTER_SEED, deterministic=DETERMINISTIC)
    rng = np.random.default_rng(MASTER_SEED)
    SEEDS_LIST = rng.integers(low=0, high=2**31 - 1, size=N_SEEDS).tolist()
else:
    SEEDS_LIST = list(range(N_SEEDS))

print(f"[Launcher] MASTER_SEED={MASTER_SEED}, DETERMINISTIC={DETERMINISTIC}")
print(f"[Launcher] SEEDS_LIST (len={len(SEEDS_LIST)}): {SEEDS_LIST}")

# -----------------------------
# Create experiment directory
# -----------------------------
EXPERIMENT_ROOT = make_experiment_root(EXPERIMENTS_ROOT, EXPERIMENT_NAME)
print(f"[Launcher] Experiment root: {EXPERIMENT_ROOT}")

# Save config (including seeds_list)
save_experiment_config(EXPERIMENT_ROOT, MASTER_SEED, SEEDS_LIST)

# -----------------------------
# Discover subject files
# -----------------------------
subject_files = discover_subject_files(DATA_DIR)
if MAX_SUBJECTS is not None:
    subject_files = subject_files[:MAX_SUBJECTS]

if not subject_files:
    raise FileNotFoundError(f"No subject .npz files found in {DATA_DIR}")

print(f"[Launcher] Found {len(subject_files)} subject files. Running up to: {MAX_SUBJECTS or 'ALL'}")

# -----------------------------
# Main loops
# -----------------------------
for subj_idx, subj_path in enumerate(subject_files, start=1):
    subj_name = subj_path.stem
    print(f"\n=== SUBJECT {subj_idx}/{len(subject_files)}: {subj_name} ===")

    # load subject NPZ (esperamos arrays 'data' y 'labels' con 3 columnas)
    data = np.load(subj_path, allow_pickle=True)
    X_all = data[NOMBRE_ARRAY_DATOS]    # (trials, channels, time)
    Y_all = data[NOMBRE_ARRAY_ETIQUETAS]  # (trials, 3) -> [modalidad, estímulo, artefacto]
    if Y_all.ndim != 2 or Y_all.shape[1] < 2:
        raise ValueError(f"Labels for {subj_path} don't have at least 2 columns (modalidad, estímulo). Found shape {Y_all.shape}")

    estimulo = Y_all[:, 1].astype(int)  # 1..11 expected
    # aca revisamos si que la forma de los datos sea (trial, canales, samples)
    # si no es asi, la cambiamos a tal
    # lo sabemos revisando si el 6 (canales) esta en la posicion que debe estar
    if X_all.shape[1] != N_CHANNELS:
        if X_all.shape[2] == N_CHANNELS: # (trials, samples, channels)
            X_all = np.transpose(X_all, (0, 2, 1)) # (trials, channels, samples)
        else:
            raise ValueError(f"Data for {subj_path} has unexpected shape {X_all.shape}. Expected (trials, channels, samples) or (trials, samples, channels).")
            
    subj_out = EXPERIMENT_ROOT / subj_name
    subj_out.mkdir(parents=True, exist_ok=True)

    for subset_name, params in SUBSETS.items():
        stim_min = params['stim_min']
        stim_max = params['stim_max']
        n_classes = params['n_classes']

        print(f"\n--- subset: {subset_name} (classes={n_classes}) ---")
        mask = (estimulo >= stim_min) & (estimulo <= stim_max)
        if mask.sum() == 0:
            print(f"[Launcher] No samples for {subset_name} in {subj_name}. Skipping.")
            continue

        X_subset = X_all[mask]      # (N, C, T)
        Y_subset_full = Y_all[mask]  # (N, 3)
        global_indices = np.where(mask)[0]

        # create subset folder
        subset_out = subj_out / subset_name
        subset_out.mkdir(parents=True, exist_ok=True)

        for seed_i in range(N_SEEDS):
            # use reproducible seed_val from SEEDS_LIST
            seed_val = int(SEEDS_LIST[seed_i])
            print(f"\n>>> seed {seed_i} (seed_val={seed_val})")
            seed_out = subset_out / f"seed_{seed_i}"
            seed_out.mkdir(parents=True, exist_ok=True)

            # set seeds for this seed_val (reproducible per-seed)
            set_global_seed(seed_val, deterministic=DETERMINISTIC)

            skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=seed_val)
            splits = list(skf.split(X_subset, Y_subset_full[:, 1].astype(int)))
            for fold_idx, (train_idx_local, test_idx_local) in enumerate(splits, start=1):
                print(f"\n>>> Fold {fold_idx}/{len(splits)}")
                fold_out = seed_out / f"fold_{fold_idx}"
                fold_out.mkdir(parents=True, exist_ok=True)

                metadata = {
                    "subject_file": str(subj_path.name),
                    "subject": subj_name,
                    "subset": subset_name,
                    "seed_idx": seed_i,
                    "seed_val": seed_val,
                    "fold_idx": fold_idx,
                    "k_folds": K_FOLDS,
                    "n_classes": n_classes,
                    "normalization": {"mean": None, "std": None},
                    "hyperparams": {
                        "batch_size": BATCH_SIZE,
                        "epochs": EPOCHS,
                        "lr": LR,
                        "dropout": DROPOUT,
                        "hidden_units": HIDDEN_UNITS,
                        **EXTRA_MODEL_KWARGS
                    },
                    "device": str(device),
                    "status": "started",
                    "error": None,
                    "train_time_s": None,
                    "augmentation_metadata_path": None
                }

                if SAVE_TRAIN_INDEX:
                    metadata["train_idx_local"] = train_idx_local.tolist()
                    metadata["test_idx_local"] = test_idx_local.tolist()
                    metadata["train_idx_global"] = global_indices[train_idx_local].tolist()
                    metadata["test_idx_global"] = global_indices[test_idx_local].tolist()
                    metadata["val_idx_local"] = None
                    metadata["val_idx_global"] = None

                try:
                    # Build train/val/test arrays (splits are by trial)
                    X_train_all = X_subset[train_idx_local]
                    Y_train_all = Y_subset_full[train_idx_local]  # (n_train_trials, 3)
                    X_test = X_subset[test_idx_local]
                    Y_test = Y_subset_full[test_idx_local]

                    # split train->train/val (stratify by stimulus column)
                    if VAL_FRAC and VAL_FRAC > 0.0:
                        idx_pool = np.arange(len(X_train_all))
                        try:
                            idx_train_rel, idx_val_rel = train_test_split(
                                idx_pool,
                                test_size=VAL_FRAC,
                                stratify=Y_train_all[:, 1],
                                random_state=seed_val,
                                shuffle=True
                            )
                        except ValueError:
                            idx_train_rel, idx_val_rel = train_test_split(
                                idx_pool,
                                test_size=VAL_FRAC,
                                random_state=seed_val,
                                shuffle=True
                            )

                        X_train = X_train_all[idx_train_rel]
                        Y_train = Y_train_all[idx_train_rel]
                        X_val = X_train_all[idx_val_rel]
                        Y_val = Y_train_all[idx_val_rel]

                        if SAVE_TRAIN_INDEX:
                            val_idx_local = train_idx_local[idx_val_rel]
                            metadata["val_idx_local"] = val_idx_local.tolist()
                            metadata["val_idx_global"] = global_indices[val_idx_local].tolist()
                    else:
                        X_train = X_train_all
                        Y_train = Y_train_all
                        X_val = np.empty((0, *X_train.shape[1:]))
                        Y_val = np.empty((0, Y_train.shape[1] if Y_train.ndim>1 else 3))
                        if SAVE_TRAIN_INDEX:
                            metadata["val_idx_local"] = []
                            metadata["val_idx_global"] = []

                    # -------------------------
                    # Augmentation (online) - se llama con los trials originales de cada split
                    # -------------------------
                    augment_kwargs = dict(AUGMENT_KWARGS)  # copy defaults
                    augment_kwargs.update(dict(seed=seed_val,
                                               metadata_path=str(fold_out),
                                               original_train_indices=train_idx_local,
                                               original_val_indices=(val_idx_local if VAL_FRAC>0 else None),
                                               original_test_indices=test_idx_local))
                    X_train_aug, Y_train_aug, X_val_aug, Y_val_aug, X_test_aug, Y_test_aug = Augmentar(
                        X_train, Y_train,
                        X_val, Y_val,
                        X_test, Y_test,
                        **augment_kwargs
                    )

                    # guardo las columnas de etiquetas de augmentacion
                    np.save(fold_out / "y_test.npy", Y_test_aug)

                    # Save augmentation metadata path (Augmentar guarda augmentation_metadata.json en metadata_path)
                    aug_meta_path = fold_out / "augmentation_metadata.json"
                    if aug_meta_path.exists():
                        metadata["augmentation_metadata_path"] = str(aug_meta_path)
                        try:
                            with open(aug_meta_path, 'r', encoding='utf8') as fh:
                                aug_meta = json.load(fh)
                            metadata["augmentation_summary"] = {
                                "data_shapes": aug_meta.get("data_shapes"),
                                "augmentation_factors": aug_meta.get("augmentation_factors"),
                                "band_noise": aug_meta.get("band_noise"),
                                "fts": aug_meta.get("fts"),
                                "window_params": aug_meta.get("window_params")
                            }
                        except Exception:
                            pass

                    # -------------------------
                    # Normalización z-score SOBRE X_train_aug
                    # -------------------------
                    mean, std = compute_zscore_params(X_train_aug)
                    metadata["normalization"]["mean"] = mean
                    metadata["normalization"]["std"] = std

                    X_train_aug = apply_zscore(X_train_aug, mean, std)
                    if X_val_aug.size:
                        X_val_aug = apply_zscore(X_val_aug, mean, std)
                    if X_test_aug.size:
                        X_test_aug = apply_zscore(X_test_aug, mean, std)

                    # -------------------------
                    # Preparar labels: extraer columna estímulo y mapear a 0..n_classes-1
                    # -------------------------
                    def map_to_class_indices(Y_aug, subset_name):
                        if Y_aug.size == 0:
                            return np.array([], dtype=int)
                        stim = Y_aug[:, 1].astype(int)
                        if subset_name == "vocales":
                            mapped = (stim - 1).astype(int)  # 1..5 -> 0..4
                        else:
                            mapped = (stim - 6).astype(int)  # 6..11 -> 0..5
                        return mapped

                    y_train_mapped = map_to_class_indices(Y_train_aug, subset_name)
                    y_val_mapped = map_to_class_indices(Y_val_aug, subset_name)
                    y_test_mapped = map_to_class_indices(Y_test_aug, subset_name)

                    # -------------------------
                    # Convert to tensors and dataloaders
                    # -------------------------
                    X_train_t = torch.tensor(X_train_aug, dtype=torch.float32)
                    Y_train_t = torch.tensor(y_train_mapped, dtype=torch.long)
                    X_test_t = torch.tensor(X_test_aug, dtype=torch.float32)
                    Y_test_t = torch.tensor(y_test_mapped, dtype=torch.long)

                    train_ds = TensorDataset(X_train_t, Y_train_t)
                    test_ds = TensorDataset(X_test_t, Y_test_t)

                    # only create val dataset/loader if non-empty
                    if X_val_aug.size and y_val_mapped.size:
                        X_val_t = torch.tensor(X_val_aug, dtype=torch.float32)
                        Y_val_t = torch.tensor(y_val_mapped, dtype=torch.long)
                        val_ds = TensorDataset(X_val_t, Y_val_t)
                        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
                    else:
                        val_loader = None

                    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=SHUFFLE_TRAIN, num_workers=NUM_WORKERS)
                    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

                    metadata["n_train"] = int(len(train_ds))
                    metadata["n_val"] = int(len(val_loader.dataset)) if val_loader is not None else 0
                    metadata["n_test"] = int(len(test_ds))

                    # -------------------------
                    # Model, optimizer, trainer
                    # -------------------------
                    model = EEGNet(in_ch=X_train_aug.shape[1], T=X_train_aug.shape[2], n_classes=n_classes, semilla=seed_val, dropout_prob=DROPOUT, hidden_units=HIDDEN_UNITS, **EXTRA_MODEL_KWARGS)
                    optimizer = optim.Adam(model.parameters(), lr=LR)
                    loss_fn = nn.CrossEntropyLoss()

                    model_output_path = str(fold_out / "best_model.pth") if SAVE_BEST_MODEL else None

                    trainer = Entrenador(modelo=model, optimizador=optimizer, func_perdida=loss_fn,
                                         device=str(device), parada_temprana=PATIENCE, log_dir=str(fold_out),
                                         histogram_freq=0, save_model=SAVE_BEST_MODEL)

                    # Train
                    t0 = time.time()
                    metrics = trainer.ajustar(cargador_entrenamiento=train_loader,
                                             cargador_validacion=val_loader if val_loader is not None else None,
                                             epocas=EPOCHS,
                                             nombre_modelo_salida=model_output_path,
                                             early_stop_patience=PATIENCE)
                    t1 = time.time()
                    metadata["train_time_s"] = float(t1 - t0)
                    metadata["status"] = "trained"

                    save_json(fold_out / "train_metrics.json", metrics)

                    # Evaluate
                    evaluator = Evaluador(modelo=trainer.modelo, device=str(device), clases=None)
                    y_true_all, y_pred_all = evaluator.probar(test_loader)
                    y_true_all = np.array(y_true_all).astype(int)
                    y_pred_all = np.array(y_pred_all).astype(int)

                    cm = confusion_matrix(y_true_all, y_pred_all, labels=np.arange(n_classes))
                    report_dict = classification_report(y_true_all, y_pred_all, labels=np.arange(n_classes), output_dict=True, zero_division=0)
                    acc = float(accuracy_score(y_true_all, y_pred_all))

                    np.savez_compressed(fold_out / "test_preds.npz", y_true=y_true_all, y_pred=y_pred_all)
                    np.save(fold_out / "confusion_matrix.npy", cm)
                    save_json(fold_out / "classification_report.json", report_dict)

                    metadata["status"] = "success"
                    metadata["test_accuracy"] = acc
                    metadata["hyperparams"]["trainer_run_metrics"] = metrics

                    save_json(fold_out / "metadata.json", metadata)
                    print(f"[Launcher] Fold completed OK. acc={acc:.4f} fold_dir={fold_out}")

                except Exception:
                    metadata["status"] = "error"
                    tb = traceback.format_exc()
                    metadata["error"] = tb
                    save_json(fold_out / "metadata.json", metadata)
                    print(f"[Launcher] ERROR on subject={subj_name} subset={subset_name} seed={seed_i} fold={fold_idx}")
                    print(tb)

                finally:
                    # cleanup to avoid GPU memory leaks
                    try:
                        del trainer
                    except Exception:
                        pass
                    try:
                        del model
                    except Exception:
                        pass
                    torch.cuda.empty_cache()

# After all runs, build summary index
print("\nAll experiments finished. Root results at:", EXPERIMENT_ROOT)
all_meta = []
for subj_dir in EXPERIMENT_ROOT.iterdir():
    if not subj_dir.is_dir():
        continue
    for subset_dir in subj_dir.iterdir():
        if not subset_dir.is_dir():
            continue
        for seed_dir in subset_dir.iterdir():
            if not seed_dir.is_dir():
                continue
            for fold_dir in seed_dir.iterdir():
                meta_file = fold_dir / "metadata.json"
                if meta_file.exists():
                    try:
                        m = json.loads(meta_file.read_text(encoding="utf8"))
                        all_meta.append(m)
                    except Exception:
                        pass

summary_index_path = EXPERIMENT_ROOT / "summary_runs.json"
save_json(summary_index_path, {"n_runs_indexed": len(all_meta), "runs": all_meta})
print("Saved summary index:", summary_index_path)
