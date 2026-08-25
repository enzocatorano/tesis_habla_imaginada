#!/usr/bin/env python3
# lanzador_TL.py
"""
Lanzador de experimentos TRANSFER LEARNING con arquitecturas CNN.

Entrena un modelo en N-1 sujetos (fuente) y evalúa zero-shot en el
sujeto restante (objetivo). Guarda el modelo pre-entrenado para
permitir fine-tuning posterior.

Basado en experimento_CNNs.py (OnlineEEGDataset, Entrenador, Evaluador).
"""

import os
import sys
import time
import json
import traceback
import random
import re
import argparse
from pathlib import Path
from pprint import pprint
import platform
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import OnlineEEGDataset
from trainer import Entrenador, Evaluador
from models import EEGNet, ShallowConvNet, DeepConvNet, iSpeechCNN

##############################
# CONFIGURACIÓN (modificar aquí)
##############################
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "clean_preprocessed"
EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1] / "experiments"
EXPERIMENT_NAME = "TL_DeepConvNet_clean"

# CLI args
parser = argparse.ArgumentParser(description='Transfer Learning EEG Experiment')
parser.add_argument('--trial-segment', type=float, nargs=2,
                    help='Segmento temporal (inicio fin) en segundos, ej: 0.0 0.5')
parser.add_argument('--experiment-name', type=str, default=None,
                    help='Nombre del experimento')
parser.add_argument('--window-duration', type=float, default=None,
                    help='Duración de la ventana en segundos')
parser.add_argument('--window-shift', type=float, default=None,
                    help='Desplazamiento de la ventana en segundos')
args = parser.parse_args()

if args.experiment_name:
    EXPERIMENT_NAME = args.experiment_name

SUFIJO_DATOS = '_clean'
NOMBRE_ARRAY_DATOS, NOMBRE_ARRAY_ETIQUETAS = "x", "y"
N_CHANNELS = 6
FS = 128

# Segmento temporal del trial (None = trial completo)
TRIAL_SEGMENT = tuple(args.trial_segment) if args.trial_segment else (0.5, 4.0)

# TARGET SELECTION (0: Modalidad, 1: Estímulo, 2: Artefacto)
TARGET_IDX = 1
UNIFIED_STIM = False

# Sujetos a usar como objetivo (held-out). None = todos los sujetos.
# Ejemplos:
#   HELD_OUT_SUBJECTS = None     → S01..S15 (todos)
#   HELD_OUT_SUBJECTS = [3]      → solo S03
#   HELD_OUT_SUBJECTS = [1,6,9]  → S01, S06, S09 (tres experimentos independientes)
HELD_OUT_SUBJECTS = None

# experiment control
MASTER_SEED = 17
DETERMINISTIC = False

N_SEEDS = 5
VAL_FRAC = 0.2

# training hyperparams
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
PATIENCE = 15
DROPOUT = 0.5
HIDDEN_UNITS = None

DEVICE = None
NUM_WORKERS = 0
SHUFFLE_TRAIN = True
SAVE_TRAIN_INDEX = True
SAVE_PRETRAINED = True

USE_CLASS_WEIGHT = True

##############################
# MODEL selection block
##############################
MODEL_NAME = "DeepConvNet"

if MODEL_NAME == "EEGNet":
    MODEL_CLASS = EEGNet
    MODEL_KWARGS = dict(
            F1=8, D=2, F2=None, kernel_length=FS//2, separable_kernel_length=16,
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
    MODEL_KWARGS = dict(dropout=DROPOUT, kernel_size_bloques=5)
elif MODEL_NAME == "iSpeechCNN":
    MODEL_CLASS = iSpeechCNN
    MODEL_KWARGS = dict(F1=20, dropout_iSpeech=DROPOUT)
else:
    raise ValueError(f"MODEL_NAME desconocido: {MODEL_NAME}")

##############################
# OPTIMIZER (Adam)
##############################
OPTIMIZER_KWARGS = dict(lr=LR,
                        betas=(0.9, 0.999),
                        eps=1e-8,
                        weight_decay=1e-3,
                        amsgrad=False)

##############################
# Augmentation defaults
##############################
AUGMENT_KWARGS = dict(
    window_duration=args.window_duration if args.window_duration else 1.0,
    window_shift=args.window_shift if args.window_shift else 0.5,
    fs=FS,
    band_noise_factor_train=0.0,
    fts_factor_train=0.0,
    noise_magnitude_relative=0.025
)

##############################
# Helper funcs
##############################
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

def _extract_subject_number_from_filename(path: Path):
    m = re.match(r"^[sS](\d+)", path.name)
    if m: return int(m.group(1))
    return None

def save_json(path: Path, obj):
    with open(path, "w", encoding="utf8") as fh:
        json.dump(obj, fh, indent=2, default=lambda o: (o.tolist() if isinstance(o, (np.ndarray,)) else str(o)))

def save_experiment_config(exp_root: Path, master_seed, seeds_list):
    config = {
        "experiment_name": EXPERIMENT_NAME,
        "target_idx": TARGET_IDX,
        "unified_stim": UNIFIED_STIM,
        "data_dir": str(DATA_DIR),
        "experiments_root": str(EXPERIMENTS_ROOT),
        "n_seeds": N_SEEDS,
        "val_frac": VAL_FRAC,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "patience": PATIENCE,
        "dropout": DROPOUT,
        "hidden_units": HIDDEN_UNITS,
        "device": DEVICE,
        "num_workers": NUM_WORKERS,
        "shuffle_train": SHUFFLE_TRAIN,
        "save_train_index": SAVE_TRAIN_INDEX,
        "save_pretrained": SAVE_PRETRAINED,
        "held_out_subjects": HELD_OUT_SUBJECTS,
        "augmentation_defaults": AUGMENT_KWARGS,
        "master_seed": master_seed,
        "seeds_list": seeds_list,
        "timestamp": now_timestamp(),
        "hostname": platform.node(),
        "model_name": MODEL_NAME,
        "model_kwargs": MODEL_KWARGS,
        "optimizer_kwargs": OPTIMIZER_KWARGS,
        "use_class_weight": USE_CLASS_WEIGHT
    }
    save_json(exp_root / "experiment_config.json", config)

def compute_zscore_params(X_train, start_idx=0, end_idx=None):
    X_focus = X_train[:, :, start_idx:end_idx]
    mean = np.mean(X_focus, axis=(0, 2), keepdims=True)
    std = np.std(X_focus, axis=(0, 2), keepdims=True)
    std[std < 1e-8] = 1e-8
    return mean, std

def apply_zscore(X, mean, std):
    return (X - mean) / std

def adjust_labels_for_loss(Y_labels, target_idx, stim_min=None):
    Y_adj = Y_labels.copy()
    if target_idx == 1:
        if stim_min is None:
            raise ValueError("stim_min es requerido para TARGET_IDX=1")
        stim = Y_adj[:, 1].astype(int)
        Y_adj[:, 1] = stim - stim_min
    else:
        val = Y_adj[:, target_idx].astype(int)
        Y_adj[:, target_idx] = val - np.min(val)
    return Y_adj

##############################
# Subsets definition
##############################
SUBSETS = {
    "vocales": dict(stim_min=1, stim_max=5, n_classes=5),
    "comandos": dict(stim_min=6, stim_max=11, n_classes=6)
}

##############################
# EJECUCIÓN PRINCIPAL
##############################
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

    # ------------------------------------------------------------------
    # 1. DESCUBRIR Y CARGAR TODOS LOS SUJETOS
    # ------------------------------------------------------------------
    subject_files = discover_subject_files(DATA_DIR)
    if not subject_files:
        raise FileNotFoundError(f"No subject .npz files found in {DATA_DIR}")

    print(f"[Launcher] Sujetos encontrados: {len(subject_files)}")
    all_subjects_data = {}
    for subj_path in subject_files:
        subj_name = subj_path.stem
        # Quitar sufijo de datos (ej: "S01_clean" → "S01")
        if subj_name.upper().endswith(SUFIJO_DATOS.upper()):
            subj_name = subj_name[:-len(SUFIJO_DATOS)]
        data = np.load(subj_path, allow_pickle=True)
        X = data[NOMBRE_ARRAY_DATOS]
        Y = data[NOMBRE_ARRAY_ETIQUETAS]

        if X.shape[1] != N_CHANNELS:
            if X.shape[2] == N_CHANNELS:
                X = np.transpose(X, (0, 2, 1))
            else:
                raise ValueError(f"Dimensiones incorrectas en {subj_name}: {X.shape}")

        all_subjects_data[subj_name] = (X, Y)
        print(f"  {subj_name}: {X.shape[0]} trials")

    all_subject_names = list(all_subjects_data.keys())
    print(f"[Launcher] Total sujetos cargados: {len(all_subject_names)}")

    # ------------------------------------------------------------------
    # 2. SUBSETS A PROCESAR
    # ------------------------------------------------------------------
    if TARGET_IDX == 1:
        if UNIFIED_STIM:
            subsets_to_process = [("estimulo_unificado", {
                "stim_min": 1, "stim_max": 11, "n_classes": 11
            })]
        else:
            subsets_to_process = list(SUBSETS.items())
    elif TARGET_IDX in (0, 2):
        target_names = {0: "modalidad", 2: "artefacto"}
        subsets_to_process = [(target_names[TARGET_IDX], {
            "stim_min": None, "stim_max": None, "n_classes": None
        })]
    else:
        raise ValueError(f"TARGET_IDX {TARGET_IDX} no soportado. Valores válidos: 0,1,2.")

    # ------------------------------------------------------------------
    # 3. BUCLE PRINCIPAL: leave-one-subject-out
    # ------------------------------------------------------------------

    if HELD_OUT_SUBJECTS is not None:
        target_names = [f"S{i:02d}" for i in HELD_OUT_SUBJECTS]
        missing = [n for n in target_names if n not in all_subjects_data]
        if missing:
            raise ValueError(f"Sujetos objetivo no encontrados: {missing}")
    else:
        target_names = all_subject_names

    print(f"[Launcher] Sujetos objetivo: {len(target_names)} ({', '.join(target_names)})")

    for heldout_name in target_names:
        print(f"\n{'='*70}")
        print(f"=== HELD-OUT: {heldout_name} ===")
        print(f"{'='*70}")

        for subset_name, params in subsets_to_process:
            stim_min = params.get('stim_min')
            stim_max = params.get('stim_max')
            n_classes_stim = params.get('n_classes')
            print(f"\n--- subset: {subset_name} ---")

            # ------------------------------------------------------------------
            # Separar fuente (N-1) y objetivo
            # ------------------------------------------------------------------
            source_X_list = []
            source_Y_list = []
            target_X = None
            target_Y = None

            for name in all_subject_names:
                X_subj, Y_subj = all_subjects_data[name]
                if TARGET_IDX == 1:
                    mask = (Y_subj[:, 1].astype(int) >= stim_min) & (Y_subj[:, 1].astype(int) <= stim_max)
                    X_f = X_subj[mask]
                    Y_f = Y_subj[mask]
                else:
                    X_f = X_subj
                    Y_f = Y_subj

                if name == heldout_name:
                    target_X = X_f
                    target_Y = Y_f
                else:
                    source_X_list.append(X_f)
                    source_Y_list.append(Y_f)

            if len(source_X_list) == 0:
                print(f"  [Warning] No hay sujetos fuente para {heldout_name}, saltando.")
                continue

            if target_X is None or target_X.shape[0] == 0:
                print(f"  [Warning] Sujeto objetivo {heldout_name} sin trials, saltando.")
                continue

            X_source = np.concatenate(source_X_list, axis=0)
            Y_source = np.concatenate(source_Y_list, axis=0)
            print(f"  Fuente: {X_source.shape[0]} trials | Objetivo: {target_X.shape[0]} trials")

            # n_classes
            if TARGET_IDX == 1:
                n_classes_target = n_classes_stim
            else:
                n_classes_target = len(np.unique(Y_source[:, TARGET_IDX]))

            # ------------------------------------------------------------------
            # Bucle de seeds
            # ------------------------------------------------------------------
            for seed_i in range(N_SEEDS):
                seed_val = int(SEEDS_LIST[seed_i])
                print(f"\n>>> seed {seed_i} (seed_val={seed_val})")

                heldout_out = EXPERIMENT_ROOT / f"{heldout_name}_heldout" / subset_name / f"seed_{seed_i}"
                heldout_out.mkdir(parents=True, exist_ok=True)

                metadata = {
                    "heldout_subject": heldout_name,
                    "subset": subset_name,
                    "seed_val": seed_val,
                    "target_idx": TARGET_IDX,
                    "n_classes": n_classes_target,
                    "source_subjects": [n for n in all_subject_names if n != heldout_name],
                    "n_source_trials": X_source.shape[0],
                    "n_target_trials": target_X.shape[0],
                    "normalization": {"mean": None, "std": None},
                    "status": "started"
                }

                try:
                    set_global_seed(seed_val, deterministic=DETERMINISTIC)

                    # ------------------------------------------------------------------
                    # Ajuste de etiquetas (0-indexed)
                    # ------------------------------------------------------------------
                    stim_min_adj = stim_min if TARGET_IDX == 1 else None
                    Y_source_adj = adjust_labels_for_loss(Y_source, TARGET_IDX, stim_min=stim_min_adj)
                    Y_target_adj = adjust_labels_for_loss(target_Y, TARGET_IDX, stim_min=stim_min_adj)

                    # ------------------------------------------------------------------
                    # Split fuente en train/val
                    # ------------------------------------------------------------------
                    if VAL_FRAC and VAL_FRAC > 0.0:
                        idx_pool = np.arange(len(X_source))
                        idx_train_rel, idx_val_rel = train_test_split(
                            idx_pool, test_size=VAL_FRAC,
                            stratify=Y_source_adj[:, TARGET_IDX],
                            random_state=seed_val, shuffle=True
                        )
                        X_train_trials = X_source[idx_train_rel]
                        Y_train_trials = Y_source_adj[idx_train_rel]
                        X_val_trials = X_source[idx_val_rel]
                        Y_val_trials = Y_source_adj[idx_val_rel]
                    else:
                        X_train_trials = X_source
                        Y_train_trials = Y_source_adj
                        X_val_trials = None
                        Y_val_trials = None

                    # ------------------------------------------------------------------
                    # Segmento temporal en muestras
                    # ------------------------------------------------------------------
                    z_start = 0 if TRIAL_SEGMENT[0] is None else int(TRIAL_SEGMENT[0] * FS)
                    z_end = X_train_trials.shape[2] if TRIAL_SEGMENT[1] is None else int(TRIAL_SEGMENT[1] * FS)

                    # ------------------------------------------------------------------
                    # Normalización Z-score por canal (sobre fuente train)
                    # ------------------------------------------------------------------
                    mean_val, std_val = compute_zscore_params(X_train_trials, start_idx=z_start, end_idx=z_end)

                    X_train_trials = apply_zscore(X_train_trials, mean_val, std_val)
                    if X_val_trials is not None:
                        X_val_trials = apply_zscore(X_val_trials, mean_val, std_val)
                    X_target_norm = apply_zscore(target_X.copy(), mean_val, std_val)
                    Y_target_norm = Y_target_adj

                    metadata["normalization"]["mean"] = mean_val.tolist()
                    metadata["normalization"]["std"] = std_val.tolist()

                    # ------------------------------------------------------------------
                    # Datasets ONLINE (con augmentación)
                    # ------------------------------------------------------------------
                    train_ds = OnlineEEGDataset(
                        X_train_trials, Y_train_trials, fs=AUGMENT_KWARGS['fs'],
                        window_duration=AUGMENT_KWARGS['window_duration'],
                        window_shift=AUGMENT_KWARGS['window_shift'],
                        trial_segment=TRIAL_SEGMENT,
                        modo='train',
                        band_noise_factor=AUGMENT_KWARGS['band_noise_factor_train'],
                        fts_factor=AUGMENT_KWARGS['fts_factor_train'],
                        noise_magnitude_relative=AUGMENT_KWARGS['noise_magnitude_relative'],
                        seed=seed_val
                    )

                    test_ds = OnlineEEGDataset(
                        X_target_norm, Y_target_norm, fs=AUGMENT_KWARGS['fs'],
                        window_duration=AUGMENT_KWARGS['window_duration'],
                        window_shift=AUGMENT_KWARGS['window_shift'],
                        trial_segment=TRIAL_SEGMENT,
                        modo='test'
                    )

                    if X_val_trials is not None:
                        val_ds = OnlineEEGDataset(
                            X_val_trials, Y_val_trials, fs=AUGMENT_KWARGS['fs'],
                            window_duration=AUGMENT_KWARGS['window_duration'],
                            window_shift=AUGMENT_KWARGS['window_shift'],
                            trial_segment=TRIAL_SEGMENT,
                            modo='val'
                        )
                        val_loader = DataLoader(
                            val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS,
                            persistent_workers=True if NUM_WORKERS > 0 else False
                        )
                    else:
                        val_loader = None

                    train_loader = DataLoader(
                        train_ds, batch_size=BATCH_SIZE, shuffle=SHUFFLE_TRAIN,
                        num_workers=NUM_WORKERS,
                        persistent_workers=True if NUM_WORKERS > 0 else False
                    )
                    test_loader = DataLoader(
                        test_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS,
                        persistent_workers=True if NUM_WORKERS > 0 else False
                    )

                    # ------------------------------------------------------------------
                    # Instanciar modelo
                    # ------------------------------------------------------------------
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

                    if USE_CLASS_WEIGHT:
                        y_train = Y_train_trials[:, TARGET_IDX].astype(int)
                        unique_classes = np.unique(y_train)
                        class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_train)
                        full_weights = np.ones(n_classes_target, dtype=np.float32)
                        for cls, w in zip(unique_classes, class_weights):
                            full_weights[cls] = w
                        weight_tensor = torch.tensor(full_weights, dtype=torch.float32, device=device)
                        loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
                    else:
                        loss_fn = nn.CrossEntropyLoss()

                    # ------------------------------------------------------------------
                    # Entrenamiento (fase fuente)
                    # ------------------------------------------------------------------
                    model_output_path = str(heldout_out / "pretrained_model.pth") if SAVE_PRETRAINED else None

                    trainer = Entrenador(modelo=model, optimizador=optimizer, func_perdida=loss_fn,
                                         device=str(device), parada_temprana=PATIENCE,
                                         log_dir=str(heldout_out), target_idx=TARGET_IDX,
                                         save_model=SAVE_PRETRAINED)

                    t0 = time.time()
                    metrics = trainer.ajustar(cargador_entrenamiento=train_loader,
                                              cargador_validacion=val_loader,
                                              epocas=EPOCHS,
                                              nombre_modelo_salida=model_output_path)
                    t1 = time.time()
                    metadata["train_time_s"] = float(t1 - t0)

                    save_json(heldout_out / "train_metrics.json", metrics)

                    # ------------------------------------------------------------------
                    # Evaluación zero-shot sobre sujeto objetivo
                    # ------------------------------------------------------------------
                    evaluator = Evaluador(modelo=trainer.modelo, device=str(device), target_idx=TARGET_IDX)
                    y_true_all, y_pred_all = evaluator.probar(test_loader)

                    cm = confusion_matrix(y_true_all, y_pred_all, labels=np.arange(n_classes_target))
                    report_dict = classification_report(y_true_all, y_pred_all,
                                                        labels=np.arange(n_classes_target),
                                                        output_dict=True, zero_division=0)
                    acc = float(accuracy_score(y_true_all, y_pred_all))

                    np.savez_compressed(heldout_out / "test_preds.npz", y_true=y_true_all, y_pred=y_pred_all)
                    np.save(heldout_out / "confusion_matrix.npy", cm)
                    save_json(heldout_out / "classification_report.json", report_dict)

                    metadata["status"] = "success"
                    metadata["test_accuracy"] = acc
                    save_json(heldout_out / "metadata.json", metadata)
                    print(f"[Launcher] Zero-shot OK. Acc={acc:.4f} | {heldout_out}")

                except Exception:
                    metadata["status"] = "error"
                    tb = traceback.format_exc()
                    metadata["error"] = tb
                    save_json(heldout_out / "metadata.json", metadata)
                    print(f"[Launcher] ERROR: {tb}")

                finally:
                    try: del trainer
                    except: pass
                    try: del model
                    except: pass
                    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Resumen final
    # ------------------------------------------------------------------
    all_meta = []
    for subj_dir in EXPERIMENT_ROOT.iterdir():
        if not subj_dir.is_dir(): continue
        for subset_dir in subj_dir.iterdir():
            if not subset_dir.is_dir(): continue
            for seed_dir in subset_dir.iterdir():
                if not seed_dir.is_dir(): continue
                meta_file = seed_dir / "metadata.json"
                if meta_file.exists():
                    try:
                        m = json.loads(meta_file.read_text(encoding="utf8"))
                        all_meta.append(m)
                    except: pass

    summary_index_path = EXPERIMENT_ROOT / "summary_runs.json"
    save_json(summary_index_path, {"n_runs_indexed": len(all_meta), "runs": all_meta})
    print(f"\n[Launcher] Finalizado. Index guardado en: {summary_index_path}")
