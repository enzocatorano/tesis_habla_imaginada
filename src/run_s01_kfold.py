# src/run_s01_kfold.py
import os
import json
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import shuffle as sk_shuffle
from torch.utils.data import DataLoader, TensorDataset

# importa tus clases desde src/trainer.py (están en el mismo directorio src)
from models import EEGNet
from trainer import Entrenador, Evaluador
import torch.nn as nn
import torch.optim as optim

# --------------------------
# Config
# --------------------------
DATA_PATH = os.path.join('..', 'data', 'processed_aug', 'S01_EEG_augmented.npz')
RESULTS_ROOT = os.path.join('..', 'experiments', 'S01_subject1_kfold')
K = 6                       # número de folds (k-fold)
RANDOM_SEED = 17
BATCH_SIZE = 64
EPOCHS = 200
LR = 1e-3
PATIENCE = 10               # early stopping (validación)
VAL_FRAC = 0.1              # fracción de validación dentro del train por fold
DEVICE = None               # None -> entrenador auto-detecta (cuda si disponible)
SHUFFLE_TRAIN = True

# reproducibilidad
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# util para serializar numpy scalars en json
def to_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")

# crear carpeta resultados
Path(RESULTS_ROOT).mkdir(parents=True, exist_ok=True)

# --------------------------
# Cargar datos
# --------------------------
print("Cargando:", DATA_PATH)
npz = np.load(DATA_PATH, allow_pickle=True)
x_all = npz['data']       # shape (trials, channels, time)
y_all = npz['labels']     # shape (trials, 4)

print("Shapes:", x_all.shape, y_all.shape)

# extraer columna de estimulo (índice 1 según lo indicaste)
estimulo = y_all[:, 1].astype(int)  # valores 1..11

# definir masks: vocales 1..5, comandos 6..11
mask_vocal = (estimulo >= 1) & (estimulo <= 5)
mask_comando = (estimulo >= 6) & (estimulo <= 11)

subsets = [
    ('vocales', mask_vocal, 5, 'estimulo_vocal'),      # 5 clases
    ('comandos', mask_comando, 6, 'estimulo_comando'),  # 6 clases
]

# --------------------------
# Función que corre k-folds para un subset
# --------------------------
def run_kfold_for_subset(name, mask, n_classes, eval_clases_str):
    print(f"\n=== Ejecutando subset: {name} | clases={n_classes} ===")
    X = x_all[mask]    # shape (n_trials_sub, C, T)
    Y_full = estimulo[mask]  # valores originales 1..11

    # mapear labels a 0..n_classes-1
    if name == 'vocales':
        # 1..5 -> 0..4
        Y = (Y_full - 1).astype(int)
    else:
        # 6..11 -> 0..5
        Y = (Y_full - 6).astype(int)

    # quick info
    n_trials, C, T = X.shape
    print(f"Trials: {n_trials}, Channels: {C}, Time: {T}")
    uniq, counts = np.unique(Y, return_counts=True)
    print("Clase counts:", dict(zip(uniq.tolist(), counts.tolist())))

    # crear carpeta resultados para este subset
    subset_root = Path(RESULTS_ROOT) / name
    subset_root.mkdir(parents=True, exist_ok=True)

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=RANDOM_SEED)

    fold_results = []  # list of dicts with fold metrics

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, Y), start=1):
        print(f"\n--- Fold {fold_idx}/{K} ---")
        # split train -> train/val (manteniendo estratificación)
        X_train_all = X[train_idx]
        Y_train_all = Y[train_idx]
        X_test = X[test_idx]
        Y_test = Y[test_idx]

        X_train, X_val, Y_train, Y_val = train_test_split(
            X_train_all, Y_train_all, test_size=VAL_FRAC, stratify=Y_train_all, random_state=RANDOM_SEED
        )

        # ------------------------------------------------------------
        # NORMALIZACIÓN Z-SCORE (solo usando el set de entrenamiento)
        # ------------------------------------------------------------
        # X_train, X_val tienen forma (trials, canales, tiempo)
        # 1) Aplanamos canales y tiempo para calcular media y desvío globales
        train_flat = X_train.reshape(X_train.shape[0], -1)
        # 2) Estadísticos SOLO del set de entrenamiento
        mean = train_flat.mean()
        std = train_flat.std()
        # Evita división por cero
        if std < 1e-8:
            std = 1e-8
        # 3) Aplicamos normalización global (misma media/std a todos los canales)
        X_train_norm = (X_train - mean) / std
        X_val_norm = (X_val - mean) / std
        X_test_norm = (X_test - mean) / std

        # convertir a tensores
        X_train_t = torch.tensor(X_train_norm, dtype=torch.float32)
        Y_train_t = torch.tensor(Y_train, dtype=torch.long)
        X_val_t = torch.tensor(X_val_norm, dtype=torch.float32)
        Y_val_t = torch.tensor(Y_val, dtype=torch.long)
        X_test_t = torch.tensor(X_test_norm, dtype=torch.float32)
        Y_test_t = torch.tensor(Y_test, dtype=torch.long)

        # dataloaders
        train_ds = TensorDataset(X_train_t, Y_train_t)
        val_ds = TensorDataset(X_val_t, Y_val_t)
        test_ds = TensorDataset(X_test_t, Y_test_t)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=SHUFFLE_TRAIN)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        # crear carpeta de fold
        fold_dir = subset_root / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # instanciar modelo (usa semilla interna para reproducibilidad)
        model = EEGNet(in_ch=6, n_classes=n_classes, semilla=RANDOM_SEED)
        model_name_path = str(fold_dir / "best_model.pth")

        # optimizador y loss
        optimizer = optim.Adam(model.parameters(), lr=LR)
        loss_fn = nn.CrossEntropyLoss()

        # Entrenador
        trainer = Entrenador(
            modelo=model,
            optimizador=optimizer,
            func_perdida=loss_fn,
            device=DEVICE,
            parada_temprana=PATIENCE,
            log_dir=str(fold_dir),    # los logs de TensorBoard quedarán en fold_dir/run_<ts>/*
            histogram_freq=1
        )

        # entrenar
        print("Entrenando...")
        metrics = trainer.ajustar(
            cargador_entrenamiento=train_loader,
            cargador_validacion=val_loader,
            epocas=EPOCHS,
            nombre_modelo_salida=model_name_path,
            early_stop_patience=PATIENCE
        )
        pprint(metrics)

        # guardar metrics por fold
        with open(fold_dir / "train_metrics.json", "w", encoding="utf8") as fh:
            json.dump(metrics, fh, indent=2, default=to_serializable)

        # Evaluación en test set
        evaluator = Evaluador(modelo=trainer.modelo, device=str(trainer.device), clases=eval_clases_str)
        print("Evaluando en test set...")
        cm = evaluator.matriz_confusion(test_loader, plot=False, titulo=f"{name} Fold {fold_idx} CM")
        report, acc = evaluator.reporte(test_loader, retornar_metricas=True)

        # guardar report / cm / preds
        np.savez_compressed(
            fold_dir / "test_preds.npz",
            y_true=Y_test,
            # recomputamos pred para guardarlas de forma coherente:
            y_pred=np.array(evaluator.probar(test_loader)[1])
        )

        # guardar confusion matrix
        np.save(fold_dir / "confusion_matrix.npy", cm)

        # guardar classification report (as json)
        report_path = fold_dir / "classification_report.json"
        with open(report_path, "w", encoding="utf8") as fh:
            json.dump(report, fh, indent=2, default=to_serializable)

        # resumen fold
        fold_summary = {
            "fold": fold_idx,
            "metrics_epochs": metrics,
            "test_acc": float(acc),
            "confusion_matrix": cm.tolist(),
            "n_train": int(len(train_idx) - int(len(train_idx) * VAL_FRAC)),
            "n_val": int(len(train_idx) * VAL_FRAC),
            "n_test": int(len(test_idx))
        }
        fold_results.append(fold_summary)

        # liberar CUDA memoria si aplica
        del trainer
        del model
        torch.cuda.empty_cache()

    # guardar resumen de todos los folds
    summary_path = subset_root / "kfold_summary.json"
    with open(summary_path, "w", encoding="utf8") as fh:
        json.dump(fold_results, fh, indent=2, default=to_serializable)
    print(f"Resultados guardados en: {subset_root}")

    return fold_results

# --------------------------
# Ejecutar para ambos subsets
# --------------------------
all_results = {}
for name, mask, n_classes, eval_clases_str in subsets:
    # si no hay muestras, saltar
    if mask.sum() == 0:
        print(f"No hay samples para {name}, saltando.")
        continue
    results = run_kfold_for_subset(name, mask, n_classes, eval_clases_str)
    all_results[name] = results

# guardar resumen global
global_path = Path(RESULTS_ROOT) / "all_results_summary.json"
with open(global_path, "w", encoding="utf8") as fh:
    json.dump(all_results, fh, indent=2, default=to_serializable)

print("FIN. Resultados globales guardados en:", global_path)