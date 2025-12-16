"""
Lanzador de experimentos exhaustivo:
 - Itera por sujetos (archivos S##_*.npz en ../data/processed_aug/)
 - Para cada sujeto ejecuta 2 subsets: 'vocales' y 'comandos'
 - Para cada subset ejecuta N_SEEDS; para cada seed hace StratifiedKFold(k=K_FOLDS, random_state=seed)
 - Dentro de cada fold separa train/val (VAL_FRAC) y NORMALIZA por z-score usando SOLO X_train
 - Entrena con EEGNet + Entrenador y evalúa con Evaluador
 - Guarda por fold: train_metrics.json, test_preds.npz, confusion_matrix.npy, classification_report.json, best_model.pth (si existe), metadata.json
 - Carpeta de salida: experiments/<EXPERIMENT_NAME>/<Sxx>/<subset>/seed_<s>/fold_<f>/
"""
import os
import sys
import time
import json
import traceback
from pathlib import Path
from pprint import pprint
import platform

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm

# Import user modules (Entrenador, Evaluador, EEGNet)
# Try trainer first, then models as fallback
try:
    from trainer import Entrenador, Evaluador, EEGNet
except Exception:
    try:
        from models import EEGNet
        from trainer import Entrenador, Evaluador
    except Exception:
        # Try other fallback names
        raise ImportError("No pude importar Entrenador/Evaluador/EEGNet. Asegurate de que trainer.py o models.py estén en src/")

# -----------------------------
# CONFIGURACIÓN (modificar aquí)
# -----------------------------
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "preproc_aug_segm_gnperband_fts"
EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1] / "experiments"
EXPERIMENT_NAME = "EEGNet_full_baseline_prueba_S01"  # se añadirá timestamp para no sobreescribir
N_SEEDS = 3
K_FOLDS = 6
VAL_FRAC = 0.1
BATCH_SIZE = 64
EPOCHS = 200
LR = 2e-3
PATIENCE = 20
DROPOUT = 0.5
HIDDEN_UNITS = 32     # None o int
MAX_SUBJECTS = 1      # None para todos, o int para debug (ej. 1)
DEVICE = None         # None => autodetect
NUM_WORKERS = 0       # dataloader workers (0 en Windows seguro)
SHUFFLE_TRAIN = True
SAVE_TRAIN_INDEX = True  # Si True, guarda índices de train/val/test en metadata.json
SAVE_BEST_MODEL = False   # Si True, guarda el mejor modelo (best_model.pth)
EXTRA_MODEL_KWARGS = dict(  # parámetros que pasamos a EEGNet por defecto
    F1=8, D=2, kernel_length=64, separable_kernel_length=16,
)

# -----------------------------
# Helper funcs
# -----------------------------
def now_timestamp():
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime())

def make_experiment_root(base: Path, name: str):
    ts = now_timestamp()
    host = platform.node().replace(" ", "_")
    exp_dir = base / f"{name}_{ts}_{host}"
    exp_dir.mkdir(parents=True, exist_ok=False)
    return exp_dir

def discover_subject_files(data_dir: Path):
    files = sorted([p for p in data_dir.iterdir() if p.is_file() and p.suffix in (".npz",)])
    # optionally filter only Sxx files (assume pattern S##_)
    files = [p for p in files if p.name.upper().startswith("S")]
    return files

def save_json(path: Path, obj):
    with open(path, "w", encoding="utf8") as fh:
        json.dump(obj, fh, indent=2, default=lambda o: (o.tolist() if isinstance(o, (np.ndarray,)) else str(o)))

def save_experiment_config(exp_root: Path):
    """Guarda la configuración completa del experimento en un archivo JSON."""
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
        "subsets": {
            "vocales": {"stim_min": 1, "stim_max": 5, "n_classes": 5},
            "comandos": {"stim_min": 6, "stim_max": 11, "n_classes": 6}
        },
        "timestamp": now_timestamp(),
        "hostname": platform.node(),
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__
    }
    save_json(exp_root / "experiment_config.json", config)
    print(f"[Launcher] Saved experiment config to: {exp_root / 'experiment_config.json'}")

# Normalization z-score block (global scalar mean/std computed on X_train)
def compute_zscore_params(X_train):
    """
    X_train: numpy array shape (n_trials, channels, time)
    returns (mean, std) scalars
    """
    # flatten across trials/channels/time
    flat = X_train.reshape(X_train.shape[0], -1)
    mean = float(flat.mean())
    std = float(flat.std())
    if std < 1e-8:
        std = 1e-8
    return mean, std

def apply_zscore(X, mean, std):
    return (X - mean) / std

# -----------------------------
# Subset definitions
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
# Create experiment directory
# -----------------------------
EXPERIMENT_ROOT = make_experiment_root(EXPERIMENTS_ROOT, EXPERIMENT_NAME)
print(f"[Launcher] Experiment root: {EXPERIMENT_ROOT}")

# Save experiment configuration
save_experiment_config(EXPERIMENT_ROOT)

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
    subj_name = subj_path.stem  # e.g. S01_EEG_augmented
    print(f"\n=== SUBJECT {subj_idx}/{len(subject_files)}: {subj_name} ===")

    # load subject NPZ
    data = np.load(subj_path, allow_pickle=True)
    X_all = data['data']    # shape (trials, channels, time)
    Y_all = data['labels']  # shape (trials, 4)
    estimulo = Y_all[:, 1].astype(int)  # 1..11
    # note: banda index at column 3
    banda_all = Y_all[:, 3].astype(int)

    # create subject folder
    subj_out = EXPERIMENT_ROOT / subj_name
    subj_out.mkdir(parents=True, exist_ok=True)

    # iterate subsets
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
        Y_subset_full = estimulo[mask]  # original label values
        global_indices = np.where(mask)[0]  # indices relative to subject npz

        # map to 0..n_classes-1
        if subset_name == "vocales":
            Y_subset = (Y_subset_full - 1).astype(int)
        else:
            Y_subset = (Y_subset_full - 6).astype(int)

        # create subset folder
        subset_out = subj_out / subset_name
        subset_out.mkdir(parents=True, exist_ok=True)

        # run seeds
        for seed_i in range(N_SEEDS):
            seed_val = int(seed_i)  # seed integer used for splitting and randomization
            print(f"\n>>> seed {seed_i} (seed_val={seed_val})")
            seed_out = subset_out / f"seed_{seed_i}"
            seed_out.mkdir(parents=True, exist_ok=True)

            # reproducibility for numpy and torch at seed level
            np.random.seed(seed_val)
            torch.manual_seed(seed_val)
            torch.cuda.manual_seed_all(seed_val)

            # create StratifiedKFold for this seed
            skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=seed_val)
            splits = list(skf.split(X_subset, Y_subset))
            # iterate folds
            for fold_idx, (train_idx_local, test_idx_local) in enumerate(splits, start=1):
                print(f"\n>>> Fold {fold_idx}/{len(splits)}")
                fold_out = seed_out / f"fold_{fold_idx}"
                fold_out.mkdir(parents=True, exist_ok=True)

                # metadata skeleton
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
                }

                # Conditionally add index information
                if SAVE_TRAIN_INDEX:
                    metadata["train_idx_local"] = train_idx_local.tolist()
                    metadata["test_idx_local"] = test_idx_local.tolist()
                    metadata["train_idx_global"] = global_indices[train_idx_local].tolist()
                    metadata["test_idx_global"] = global_indices[test_idx_local].tolist()
                    metadata["val_idx_local"] = None
                    metadata["val_idx_global"] = None

                try:
                    # build train/val/test arrays
                    X_train_all = X_subset[train_idx_local]  # pool to split into train/val
                    Y_train_all = Y_subset[train_idx_local]
                    X_test = X_subset[test_idx_local]
                    Y_test = Y_subset[test_idx_local]

                    # split train->train/val (stratified) using indices to preserve mapping to global indices
                    if VAL_FRAC and VAL_FRAC > 0.0:
                        # idx_pool are indices into X_train_all
                        idx_pool = np.arange(len(X_train_all))
                        try:
                            idx_train_rel, idx_val_rel = train_test_split(
                                idx_pool,
                                test_size=VAL_FRAC,
                                stratify=Y_train_all,
                                random_state=seed_val,
                                shuffle=True
                            )
                        except ValueError as e:
                            # fallback: if stratify fails due to too few samples per class, do a non-stratified split
                            idx_train_rel, idx_val_rel = train_test_split(
                                idx_pool,
                                test_size=VAL_FRAC,
                                random_state=seed_val,
                                shuffle=True
                            )

                        # select arrays by relative indices
                        X_train = X_train_all[idx_train_rel]
                        Y_train = Y_train_all[idx_train_rel]
                        X_val = X_train_all[idx_val_rel]
                        Y_val = Y_train_all[idx_val_rel]

                        # compute and store val indices relative to the subset (local indices)
                        if SAVE_TRAIN_INDEX:
                            val_idx_local = train_idx_local[idx_val_rel]
                            metadata["val_idx_local"] = val_idx_local.tolist()
                            metadata["val_idx_global"] = global_indices[val_idx_local].tolist()

                    else:
                        # no validation split requested
                        X_train = X_train_all
                        Y_train = Y_train_all
                        X_val = np.empty((0, *X_train.shape[1:]))
                        Y_val = np.empty((0,))
                        if SAVE_TRAIN_INDEX:
                            metadata["val_idx_local"] = []
                            metadata["val_idx_global"] = []

                    # compute and apply z-score normalization using only X_train
                    mean, std = compute_zscore_params(X_train)
                    metadata["normalization"]["mean"] = mean
                    metadata["normalization"]["std"] = std

                    X_train = apply_zscore(X_train, mean, std)
                    if X_val.size:
                        X_val = apply_zscore(X_val, mean, std)
                    X_test = apply_zscore(X_test, mean, std)

                    # Convert to tensors and dataloaders
                    X_train_t = torch.tensor(X_train, dtype=torch.float32)
                    Y_train_t = torch.tensor(Y_train, dtype=torch.long)
                    X_val_t = torch.tensor(X_val, dtype=torch.float32)
                    Y_val_t = torch.tensor(Y_val, dtype=torch.long)
                    X_test_t = torch.tensor(X_test, dtype=torch.float32)
                    Y_test_t = torch.tensor(Y_test, dtype=torch.long)

                    train_ds = TensorDataset(X_train_t, Y_train_t)
                    val_ds = TensorDataset(X_val_t, Y_val_t)
                    test_ds = TensorDataset(X_test_t, Y_test_t)

                    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=SHUFFLE_TRAIN, num_workers=NUM_WORKERS)
                    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
                    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

                    metadata["n_train"] = int(len(train_ds))
                    metadata["n_val"] = int(len(val_ds))
                    metadata["n_test"] = int(len(test_ds))

                    # instantiate model, optimizer, loss, trainer
                    model = EEGNet(in_ch=X_train.shape[1], n_classes=n_classes, semilla=seed_val, dropout_prob=DROPOUT, hidden_units=HIDDEN_UNITS, **EXTRA_MODEL_KWARGS)
                    optimizer = optim.Adam(model.parameters(), lr=LR)
                    loss_fn = nn.CrossEntropyLoss()

                    # Determine model output path based on SAVE_BEST_MODEL
                    model_output_path = str(fold_out / "best_model.pth") if SAVE_BEST_MODEL else None

                    # create trainer with log_dir inside fold (trainer will create run_<timestamp>)
                    trainer = Entrenador(modelo=model, optimizador=optimizer, func_perdida=loss_fn,
                                         device=str(device), parada_temprana=PATIENCE, log_dir=str(fold_out), 
                                         histogram_freq=0, save_model=SAVE_BEST_MODEL)

                    # train and time
                    t0 = time.time()
                    metrics = trainer.ajustar(cargador_entrenamiento=train_loader,
                                             cargador_validacion=val_loader if len(val_ds) > 0 else None,
                                             epocas=EPOCHS,
                                             nombre_modelo_salida=model_output_path,
                                             early_stop_patience=PATIENCE)
                    t1 = time.time()
                    metadata["train_time_s"] = float(t1 - t0)
                    metadata["status"] = "trained"

                    # save returned train metrics as well
                    save_json(fold_out / "train_metrics.json", metrics)

                    # evaluate on test set
                    evaluator = Evaluador(modelo=trainer.modelo, device=str(device), clases=None)
                    y_true_all, y_pred_all = evaluator.probar(test_loader)
                    # ensure numpy arrays
                    y_true_all = np.array(y_true_all).astype(int)
                    y_pred_all = np.array(y_pred_all).astype(int)

                    # confusion matrix and classification report
                    cm = confusion_matrix(y_true_all, y_pred_all, labels=np.arange(n_classes))
                    report_dict = classification_report(y_true_all, y_pred_all, labels=np.arange(n_classes), output_dict=True, zero_division=0)
                    acc = float(accuracy_score(y_true_all, y_pred_all))

                    # save files
                    np.savez_compressed(fold_out / "test_preds.npz", y_true=y_true_all, y_pred=y_pred_all)
                    np.save(fold_out / "confusion_matrix.npy", cm)
                    save_json(fold_out / "classification_report.json", report_dict)

                    # finalize metadata
                    metadata["status"] = "success"
                    metadata["test_accuracy"] = acc
                    metadata["normalization"]["mean"] = mean
                    metadata["normalization"]["std"] = std
                    metadata["hyperparams"]["trainer_run_metrics"] = metrics

                    # save metadata
                    save_json(fold_out / "metadata.json", metadata)
                    print(f"[Launcher] Fold completed OK. acc={acc:.4f} fold_dir={fold_out}")

                except Exception as e:
                    # save error metadata and trace
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

# After all
print("\nAll experiments finished. Root results at:", EXPERIMENT_ROOT)
# Optionally write a summary index
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