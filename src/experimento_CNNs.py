#!/usr/bin/env python3
# experimento_CNNs.py
"""
Lanzador de experimentos con augmentación ONLINE VERDADERA y control de seed maestro.

Modificado para:
- Multiprocesamiento seguro en Windows (if __name__ == '__main__':)
- Augmentación estocástica "al vuelo" usando OnlineEEGDataset (evita colapso de RAM).
- Predicción multi-objetivo via TARGET_IDX (Modalidad, Estímulo, Artefacto).
- Normalización Z-score aislada por canal.
- Instanciación explícita de modelos (eliminando magia negra).
"""

import os
import sys
import time
import json
import traceback
import random
import re
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
from torch.utils.data import DataLoader
from tqdm import tqdm

# Imports directos y limpios. Asegúrate de tener dataset.py, trainer.py y models.py en el mismo directorio (o en el PYTHONPATH)
from dataset import OnlineEEGDataset
from trainer import Entrenador, Evaluador
from models import EEGNet, ShallowConvNet, DeepConvNet, iSpeechCNN

# -----------------------------
# CONFIGURACIÓN (modificar aquí)
# -----------------------------
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "synthetic_labels"
EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1] / "experiments"
EXPERIMENT_NAME = "S09_sintetico"
SUFIJO_DATOS = '_synthetic'
NOMBRE_ARRAY_DATOS, NOMBRE_ARRAY_ETIQUETAS = "x", "y"
N_CHANNELS = 6

# TARGET SELECTION (0: Modalidad, 1: Estímulo, 2: Artefacto)
TARGET_IDX = 1

# experiment control
MASTER_SEED = None    # setea a None si no querés seed maestro
DETERMINISTIC = False # True => intenta operaciones deterministas en PyTorch

N_SEEDS = 1
K_FOLDS = 5
VAL_FRAC = 0.1

# training hyperparams
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-4
PATIENCE = 20
DROPOUT = 0.1
HIDDEN_UNITS = None

SUBJECT = [9]

DEVICE = None         # None => autodetect
NUM_WORKERS = 0       # IMPORTANTE: Subir a 2, 4 u 8 para que el CPU procese el Online Dataset sin frenar la GPU
SHUFFLE_TRAIN = True
SAVE_TRAIN_INDEX = True
SAVE_BEST_MODEL = False

# -----------------------------
# MODEL selection block
# -----------------------------
MODEL_NAME = "iSpeechCNN"

if MODEL_NAME == "EEGNet":
    MODEL_CLASS = EEGNet
    MODEL_KWARGS = dict(
            F1=8, D=2, F2=None, kernel_length=64, separable_kernel_length=16,
            pool_time1=4, pool_time2=8, dropout_prob=DROPOUT, hidden_units=None,
            max_norm_spatial=1.0, max_norm_dense=0.25
            )
elif MODEL_NAME == "ShallowConvNet":
    MODEL_CLASS = ShallowConvNet
    MODEL_KWARGS = dict(
            n_filtros_temporales=40, longitud_kernel_temporal=25, pool_size=75,
            pool_stride=15, dropout=DROPOUT
            )
elif MODEL_NAME == "DeepConvNet":
    MODEL_CLASS = DeepConvNet
    MODEL_KWARGS = dict(dropout=DROPOUT)
elif MODEL_NAME == "iSpeechCNN":
    MODEL_CLASS = iSpeechCNN
    MODEL_KWARGS = dict(F1=20, dropout_iSpeech=DROPOUT)
else:
    raise ValueError(f"MODEL_NAME desconocido: {MODEL_NAME}")

# -----------------------------
# OPTIMIZER (Adam)
# -----------------------------
OPTIMIZER_KWARGS = dict(lr=LR,
                        betas=(0.9, 0.999),
                        eps=1e-8,
                        weight_decay=1e-3,
                        amsgrad=False)

# -----------------------------
# Augmentation defaults
# -----------------------------
AUGMENT_KWARGS = dict(
    window_duration=4.0,
    window_shift=4.0,
    fs=128,
    band_noise_factor_train=0.0, # Ajusta tu probabilidad real aquí (ej. 0.3)
    fts_factor_train=0.0,        # Ajusta tu probabilidad real aquí (ej. 0.3)
    noise_magnitude_relative=0.025
)

# -----------------------------
# Helper funcs
# -----------------------------
def now_timestamp():
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime())

def set_global_seed(seed: int, deterministic: bool = True):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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
        "target_idx": TARGET_IDX,
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
        "subject_selection": SUBJECT,
        "device": DEVICE,
        "num_workers": NUM_WORKERS,
        "shuffle_train": SHUFFLE_TRAIN,
        "save_train_index": SAVE_TRAIN_INDEX,
        "save_best_model": SAVE_BEST_MODEL,
        "augmentation_defaults": AUGMENT_KWARGS,
        "master_seed": master_seed,
        "seeds_list": seeds_list,
        "timestamp": now_timestamp(),
        "hostname": platform.node(),
        "model_name": MODEL_NAME,
        "model_kwargs": MODEL_KWARGS,
        "optimizer_kwargs": OPTIMIZER_KWARGS
    }
    save_json(exp_root / "experiment_config.json", config)

def compute_zscore_params(X_train):
    """Media y std aislada POR CANAL."""
    mean = np.mean(X_train, axis=(0, 2), keepdims=True)
    std = np.std(X_train, axis=(0, 2), keepdims=True)
    std[std < 1e-8] = 1e-8
    return mean, std

def apply_zscore(X, mean, std):
    return (X - mean) / std

def _extract_subject_number_from_filename(path: Path):
    m = re.match(r"^[sS](\d+)", path.name)
    if m: return int(m.group(1))
    return None

# -----------------------------
# Subsets definition
# -----------------------------
SUBSETS = {
    "vocales": dict(stim_min=1, stim_max=5, n_classes=5),
    "comandos": dict(stim_min=6, stim_max=11, n_classes=6)
}

# =====================================================================
# INICIO DE EJECUCIÓN (Protegido para multiprocesamiento en Windows)
# =====================================================================
if __name__ == '__main__':
    
    if DEVICE is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = DEVICE
    device = torch.device(device_str)
    print(f"[Launcher] Using device: {device} | TARGET_IDX: {TARGET_IDX}")

    if MASTER_SEED is not None:
        set_global_seed(MASTER_SEED, deterministic=DETERMINISTIC)
        rng = np.random.default_rng(MASTER_SEED)
        SEEDS_LIST = rng.integers(low=0, high=2**31 - 1, size=N_SEEDS).tolist()
    else:
        SEEDS_LIST = list(range(N_SEEDS))

    EXPERIMENT_ROOT = make_experiment_root(EXPERIMENTS_ROOT, EXPERIMENT_NAME)
    save_experiment_config(EXPERIMENT_ROOT, MASTER_SEED, SEEDS_LIST)

    subject_files = discover_subject_files(DATA_DIR)

    if SUBJECT is not None:
        wanted = [SUBJECT] if isinstance(SUBJECT, int) else [int(x) for x in SUBJECT]
        selected_files = [p for p in subject_files if _extract_subject_number_from_filename(p) in wanted]
        subject_files = selected_files

    if not subject_files:
        raise FileNotFoundError(f"No subject .npz files found in {DATA_DIR}")

    # -----------------------------
    # Bucle Principal
    # -----------------------------
    for subj_idx, subj_path in enumerate(subject_files, start=1):
        subj_name = subj_path.stem
        print(f"\n=== SUBJECT {subj_idx}/{len(subject_files)}: {subj_name} ===")

        data = np.load(subj_path, allow_pickle=True)
        X_all = data[NOMBRE_ARRAY_DATOS]    # (trials, channels, time)
        Y_all = data[NOMBRE_ARRAY_ETIQUETAS]  # (trials, 3) -> [modalidad, estímulo, artefacto]
        
        if X_all.shape[1] != N_CHANNELS:
            if X_all.shape[2] == N_CHANNELS: 
                X_all = np.transpose(X_all, (0, 2, 1)) 
            else:
                raise ValueError("Dimensiones incorrectas en los datos.")
                
        subj_out = EXPERIMENT_ROOT / subj_name
        subj_out.mkdir(parents=True, exist_ok=True)

        for subset_name, params in SUBSETS.items():
            stim_min = params['stim_min']
            stim_max = params['stim_max']
            n_classes_stim = params['n_classes']

            print(f"\n--- subset: {subset_name} ---")
            # Siempre filtramos las ventanas por el subset (Vocales o Comandos)
            mask = (Y_all[:, 1].astype(int) >= stim_min) & (Y_all[:, 1].astype(int) <= stim_max)
            if mask.sum() == 0:
                continue

            X_subset = X_all[mask]      # (N, C, T)
            Y_subset_full = Y_all[mask] # (N, 3)
            global_indices = np.where(mask)[0]
            
            # Ajuste Dinámico de n_classes según el TARGET_IDX
            if TARGET_IDX == 1:
                n_classes_target = n_classes_stim
            else:
                # Para modalidad o artefacto, contamos las clases únicas en ese subset
                n_classes_target = len(np.unique(Y_subset_full[:, TARGET_IDX]))

            subset_out = subj_out / subset_name
            subset_out.mkdir(parents=True, exist_ok=True)

            for seed_i in range(N_SEEDS):
                seed_val = int(SEEDS_LIST[seed_i])
                print(f"\n>>> seed {seed_i} (seed_val={seed_val})")
                seed_out = subset_out / f"seed_{seed_i}"
                seed_out.mkdir(parents=True, exist_ok=True)

                set_global_seed(seed_val, deterministic=DETERMINISTIC)

                # Estratificamos según el TARGET que vamos a predecir
                skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=seed_val)
                splits = list(skf.split(X_subset, Y_subset_full[:, TARGET_IDX].astype(int)))
                
                for fold_idx, (train_idx_local, test_idx_local) in enumerate(splits, start=1):
                    print(f"\n>>> Fold {fold_idx}/{len(splits)}")
                    fold_out = seed_out / f"fold_{fold_idx}"
                    fold_out.mkdir(parents=True, exist_ok=True)

                    metadata = {
                        "subject": subj_name,
                        "subset": subset_name,
                        "seed_val": seed_val,
                        "fold_idx": fold_idx,
                        "target_idx": TARGET_IDX,
                        "n_classes": n_classes_target,
                        "normalization": {"mean": None, "std": None},
                        "status": "started"
                    }

                    try:
                        # 1. Split estricto de Trials (Prevención de Data Leakage)
                        X_train_all = X_subset[train_idx_local]
                        Y_train_all = Y_subset_full[train_idx_local]
                        X_test_trials = X_subset[test_idx_local]
                        Y_test_trials = Y_subset_full[test_idx_local]

                        # Mapeo de Etiquetas a formato 0-indexed
                        def adjust_labels_for_loss(Y_labels, subset_name, target_idx):
                            Y_adj = Y_labels.copy()
                            if target_idx == 1:
                                # Estímulo
                                stim = Y_adj[:, 1].astype(int)
                                Y_adj[:, 1] = (stim - 1) if subset_name == "vocales" else (stim - 6)
                            else:
                                # Modalidad (0) o Artefacto (2) - Asegurar que empieza en 0
                                val = Y_adj[:, target_idx].astype(int)
                                Y_adj[:, target_idx] = val - np.min(val)
                            return Y_adj

                        Y_train_all = adjust_labels_for_loss(Y_train_all, subset_name, TARGET_IDX)
                        Y_test_trials = adjust_labels_for_loss(Y_test_trials, subset_name, TARGET_IDX)

                        if VAL_FRAC and VAL_FRAC > 0.0:
                            idx_pool = np.arange(len(X_train_all))
                            idx_train_rel, idx_val_rel = train_test_split(
                                idx_pool, test_size=VAL_FRAC, 
                                stratify=Y_train_all[:, TARGET_IDX], 
                                random_state=seed_val + fold_idx, shuffle=True
                            )
                            X_train_trials = X_train_all[idx_train_rel]
                            Y_train_trials = Y_train_all[idx_train_rel]
                            X_val_trials = X_train_all[idx_val_rel]
                            Y_val_trials = Y_train_all[idx_val_rel]
                        else:
                            X_train_trials = X_train_all
                            Y_train_trials = Y_train_all
                            X_val_trials, Y_val_trials = None, None

                        # 2. Normalización Z-Score Aislada por Canal
                        mean_val, std_val = compute_zscore_params(X_train_trials)
                        X_train_trials = apply_zscore(X_train_trials, mean_val, std_val)
                        if X_val_trials is not None:
                            X_val_trials = apply_zscore(X_val_trials, mean_val, std_val)
                        X_test_trials = apply_zscore(X_test_trials, mean_val, std_val)
                        
                        metadata["normalization"]["mean"] = mean_val.tolist()
                        metadata["normalization"]["std"] = std_val.tolist()

                        # 3. Datasets Online (Segmentación y Augmentación estocástica "al vuelo")
                        train_ds = OnlineEEGDataset(
                            X_train_trials, Y_train_trials, fs=AUGMENT_KWARGS['fs'],
                            window_duration=AUGMENT_KWARGS['window_duration'], 
                            window_shift=AUGMENT_KWARGS['window_shift'],
                            modo='train',
                            band_noise_factor=AUGMENT_KWARGS['band_noise_factor_train'],
                            fts_factor=AUGMENT_KWARGS['fts_factor_train'],
                            noise_magnitude_relative=AUGMENT_KWARGS['noise_magnitude_relative'],
                            seed=seed_val
                        )
                        
                        test_ds = OnlineEEGDataset(
                            X_test_trials, Y_test_trials, fs=AUGMENT_KWARGS['fs'],
                            window_duration=AUGMENT_KWARGS['window_duration'], 
                            window_shift=AUGMENT_KWARGS['window_shift'],
                            modo='test' 
                        )

                        if X_val_trials is not None:
                            val_ds = OnlineEEGDataset(
                                X_val_trials, Y_val_trials, fs=AUGMENT_KWARGS['fs'],
                                window_duration=AUGMENT_KWARGS['window_duration'], 
                                window_shift=AUGMENT_KWARGS['window_shift'],
                                modo='val'
                            )
                            val_loader = DataLoader(
                                val_ds, 
                                batch_size=BATCH_SIZE, 
                                shuffle=False, 
                                num_workers=NUM_WORKERS,
                                persistent_workers=True if NUM_WORKERS > 0 else False
                            )
                        else:
                            val_loader = None

                        train_loader = DataLoader(
                            train_ds, 
                            batch_size=BATCH_SIZE, 
                            shuffle=SHUFFLE_TRAIN, 
                            num_workers=NUM_WORKERS,
                            persistent_workers=True if NUM_WORKERS > 0 else False  # <--- EVITA QUE WINDOWS CIERRE LOS PROCESOS
                        )
                        test_loader = DataLoader(
                            test_ds, 
                            batch_size=BATCH_SIZE, 
                            shuffle=False, 
                            num_workers=NUM_WORKERS,
                            persistent_workers=True if NUM_WORKERS > 0 else False
                        )

                        # 4. Instanciación Explícita del Modelo
                        in_ch = int(X_train_trials.shape[1])
                        T = train_ds.duration_samples
                        
                        if MODEL_NAME == "EEGNet":
                            model = EEGNet(in_ch=in_ch, T=T, n_classes=n_classes_target, semilla=seed_val, **MODEL_KWARGS)
                        elif MODEL_NAME == "ShallowConvNet":
                            model = ShallowConvNet(n_canales=in_ch, n_samples=T, n_clases=n_classes_target, **MODEL_KWARGS)
                        elif MODEL_NAME == "DeepConvNet":
                            model = DeepConvNet(n_canales=in_ch, n_samples=T, n_clases=n_classes_target, **MODEL_KWARGS)
                        elif MODEL_NAME == "iSpeechCNN":
                            model = iSpeechCNN(n_channels=in_ch, n_timepoints=T, n_classes=n_classes_target, semilla=seed_val, **MODEL_KWARGS)
                        
                        model = model.to(device)
                        optimizer = optim.Adam(model.parameters(), **OPTIMIZER_KWARGS)
                        loss_fn = nn.CrossEntropyLoss()

                        model_output_path = str(fold_out / "best_model.pth") if SAVE_BEST_MODEL else None

                        # 5. Entrenador (Pasando el TARGET_IDX)
                        trainer = Entrenador(modelo=model, optimizador=optimizer, func_perdida=loss_fn,
                                             device=str(device), parada_temprana=PATIENCE, log_dir=str(fold_out),
                                             target_idx=TARGET_IDX, save_model=SAVE_BEST_MODEL)

                        t0 = time.time()
                        metrics = trainer.ajustar(cargador_entrenamiento=train_loader,
                                                 cargador_validacion=val_loader,
                                                 epocas=EPOCHS,
                                                 nombre_modelo_salida=model_output_path)
                        t1 = time.time()
                        metadata["train_time_s"] = float(t1 - t0)

                        save_json(fold_out / "train_metrics.json", metrics)

                        # 6. Evaluador
                        evaluator = Evaluador(modelo=trainer.modelo, device=str(device), target_idx=TARGET_IDX)
                        y_true_all, y_pred_all = evaluator.probar(test_loader)
                        
                        cm = confusion_matrix(y_true_all, y_pred_all, labels=np.arange(n_classes_target))
                        report_dict = classification_report(y_true_all, y_pred_all, labels=np.arange(n_classes_target), output_dict=True, zero_division=0)
                        acc = float(accuracy_score(y_true_all, y_pred_all))

                        np.savez_compressed(fold_out / "test_preds.npz", y_true=y_true_all, y_pred=y_pred_all)
                        np.save(fold_out / "confusion_matrix.npy", cm)
                        save_json(fold_out / "classification_report.json", report_dict)

                        metadata["status"] = "success"
                        metadata["test_accuracy"] = acc
                        save_json(fold_out / "metadata.json", metadata)
                        print(f"[Launcher] Fold OK. Acc={acc:.4f} Dir={fold_out}")

                    except Exception:
                        metadata["status"] = "error"
                        tb = traceback.format_exc()
                        metadata["error"] = tb
                        save_json(fold_out / "metadata.json", metadata)
                        print(f"[Launcher] ERROR: {tb}")

                    finally:
                        try: del trainer
                        except: pass
                        try: del model
                        except: pass
                        torch.cuda.empty_cache()

    # Indexador de resúmenes
    all_meta = []
    for subj_dir in EXPERIMENT_ROOT.iterdir():
        if not subj_dir.is_dir(): continue
        for subset_dir in subj_dir.iterdir():
            if not subset_dir.is_dir(): continue
            for seed_dir in subset_dir.iterdir():
                if not seed_dir.is_dir(): continue
                for fold_dir in seed_dir.iterdir():
                    meta_file = fold_dir / "metadata.json"
                    if meta_file.exists():
                        try:
                            m = json.loads(meta_file.read_text(encoding="utf8"))
                            all_meta.append(m)
                        except: pass

    summary_index_path = EXPERIMENT_ROOT / "summary_runs.json"
    save_json(summary_index_path, {"n_runs_indexed": len(all_meta), "runs": all_meta})
    print("\n[Launcher] Finalizado. Index guardado en:", summary_index_path)