# src/train_mlp_subject1.py
"""
Entrenamiento MLP sobre features DWT (por sujeto, tareas separadas).
Script robusto en resolución de rutas respecto al config.yml y creación de carpetas.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import os
import json
import shutil
from sklearn.model_selection import train_test_split
from models import MLP
from trainer import Entrenador, Evaluador
import joblib
import datetime
import yaml

def load_subject_data(feat_dir: Path, subj_stem="S01_EEG"):
    base = feat_dir / "per_subject"
    feat_file = base / f"{subj_stem}_features.npy"
    labels_file = base / f"{subj_stem}_labels.npy"
    task_file = base / f"{subj_stem}_task.npy"
    if not feat_file.exists():
        raise FileNotFoundError(f"No encontré features en {feat_file}")
    if not labels_file.exists() or not task_file.exists():
        raise FileNotFoundError(f"Faltan labels or task en {base}. Esperaba {labels_file} y {task_file}")
    feat3 = np.load(feat_file)
    labels = np.load(labels_file)
    task = np.load(task_file)
    return feat3, labels, task

def prepare_task_dataset(feat3, labels, task, desired_task):
    mask = (task == desired_task)
    X = feat3[mask]
    Y_info = labels[mask]
    stim = Y_info[:,1].astype(int)
    if desired_task == 0:
        # vowels: stimuli 1..5 -> map to 0..4
        y = np.array([int(s)-1 for s in stim], dtype=np.int64)
    else:
        # commands: stimuli 6..11 -> map to 0..5
        y = np.array([int(s)-6 for s in stim], dtype=np.int64)
    return X, y

def normalize_by_statistic(X_train, X_val, X_test):
    # X shapes: (N, n_ch, L+1, n_stats)
    n_stats = X_train.shape[-1]
    means = np.zeros(n_stats, dtype=np.float32)
    stds = np.ones(n_stats, dtype=np.float32)
    for s in range(n_stats):
        vals = X_train[..., s].ravel()
        means[s] = float(np.mean(vals))
        stds[s] = float(np.std(vals)) if np.std(vals) > 0 else 1.0
    X_train_n = (X_train - means.reshape(1,1,1,-1)) / stds.reshape(1,1,1,-1)
    X_val_n = (X_val - means.reshape(1,1,1,-1)) / stds.reshape(1,1,1,-1)
    X_test_n= (X_test - means.reshape(1,1,1,-1)) / stds.reshape(1,1,1,-1)
    scaler = {"means": means.tolist(), "stds": stds.tolist()}
    return X_train_n.astype(np.float32), X_val_n.astype(np.float32), X_test_n.astype(np.float32), scaler

def flatten_for_mlp(X):
    N = X.shape[0]
    return X.reshape(N, -1).astype(np.float32)

def run_single_seed(X, y, seed, run_dir: Path, model_cfg: dict, training_cfg: dict, logging_cfg: dict, paths_cfg: dict,
                    logs_dir: Path, models_root: Path, runs_root: Path):
    # splits
    test_size = float(training_cfg.get("test_size", 0.3))
    val_frac = float(training_cfg.get("val_fraction_of_temp", 0.5))
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_frac, random_state=seed, stratify=y_temp)

    # normalize according to train statistics (per-statistic z-score)
    X_train_n, X_val_n, X_test_n, scaler = normalize_by_statistic(X_train, X_val, X_test)

    # store scaler in run_dir/features
    (run_dir / "features").mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, run_dir / "features" / "scaler.joblib")

    # flatten for MLP
    Xtr_flat = flatten_for_mlp(X_train_n)
    Xv_flat = flatten_for_mlp(X_val_n)
    Xte_flat= flatten_for_mlp(X_test_n)

    # dataloaders
    batch_size = int(model_cfg.get("batch_size", 64))
    train_ds = TensorDataset(torch.from_numpy(Xtr_flat), torch.from_numpy(y_train).long())
    val_ds   = TensorDataset(torch.from_numpy(Xv_flat), torch.from_numpy(y_val).long())
    test_ds  = TensorDataset(torch.from_numpy(Xte_flat), torch.from_numpy(y_test).long())
    tr_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader= DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader=DataLoader(test_ds, batch_size=1, shuffle=False)

    # model architecture
    in_dim = Xtr_flat.shape[1]
    out_dim = int(np.max(y)+1)
    arch = list(model_cfg.get("arq", [in_dim, 256, out_dim]))
    arch[0] = in_dim
    arch[-1]= out_dim
    modelo = MLP(arq=arch, func_act=model_cfg.get("act","relu"), usar_batch_norm=model_cfg.get("batchnorm", True),
                 dropout=model_cfg.get("dropout",0.0), semilla=seed)

    # loss & optim
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(modelo.parameters(), lr=float(model_cfg.get("lr", 1e-3)), weight_decay=float(model_cfg.get("weight_decay", 0.0)))

    # Entrenador: usar logs_dir (base) para que cree run-specific dentro de él
    histogram_freq = int(logging_cfg.get("save_histograms_every_n_epochs", 0))
    trainer = Entrenador(modelo=modelo,
                         optimizador=optim,
                         func_perdida=loss_fn,
                         device=model_cfg.get("device", None),
                         parada_temprana=int(model_cfg.get("early_stop", 0)) if model_cfg.get("early_stop", None) is not None else None,
                         log_dir=str(logs_dir),
                         histogram_freq=histogram_freq)

    # model output path: models_root / subj / task / model_seed{seed}.pt
    subj = paths_cfg.get("subj")
    taskname = paths_cfg.get("taskname")
    model_out_dir = models_root / subj / taskname
    model_out_dir.mkdir(parents=True, exist_ok=True)
    model_out_path = model_out_dir / f"model_seed{seed}.pt"

    # run-specific storage under runs_root / taskname / run_seed{seed}
    run_models_dir = runs_root / taskname / f"run_seed{seed}"
    run_models_dir.mkdir(parents=True, exist_ok=True)
    (run_models_dir / "results").mkdir(parents=True, exist_ok=True)

    # train
    metrics_epochs = trainer.ajustar(tr_loader,
                                    cargador_validacion=val_loader,
                                    epocas=int(model_cfg.get("epochs", 50)),
                                    nombre_modelo_salida=str(model_out_path),
                                    early_stop_patience=model_cfg.get("early_stop", None))

    # copy trainer's metrics file (if any) into run results for centralization
    try:
        tb_metrics_src = Path(trainer.run_specific_log_dir) / "metrics_epochs.json"
        if tb_metrics_src.exists():
            shutil.copy(str(tb_metrics_src), str(run_models_dir / "results" / "metrics_epochs_tb.json"))
    except Exception as e:
        print(f"Advertencia: no pude copiar metrics from tensorboard run dir: {e}")

    # evaluate final (trainer may have loaded best model)
    evaluator = Evaluador(modelo, device=model_cfg.get("device", None), clases=model_cfg.get("clases", None))
    rep, acc = evaluator.reporte(test_loader, retornar_metricas=True)

    # save test results and run meta info
    meta_run = {
        "seed": int(seed),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "acc_test": float(acc),
        "report": rep
    }
    with open(run_models_dir / "results" / "metrics.json", "w", encoding="utf8") as fh:
        json.dump(meta_run, fh, indent=2)

    with open(run_models_dir / "results" / "split_and_scaler_meta.json", "w", encoding="utf8") as fh:
        json.dump({"scaler":scaler, "n_train": int(Xtr_flat.shape[0]), "n_val":int(Xv_flat.shape[0]), "n_test":int(Xte_flat.shape[0])}, fh, indent=2)

    return float(acc), rep

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="experiments/exp_mlp_s01/config.yml", help="config YAML del experimento")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"No encontré config en {cfg_path}")
    cfg = yaml.safe_load(open(cfg_path, 'r', encoding='utf8'))

    # --- resolución de rutas robusta respecto al config.yml ---
    cfg_dir = cfg_path.parent.resolve()

    # feat_dir: resuelto respecto al cfg_dir (donde está config.yml)
    feat_dir_raw = cfg.get("feat_dir")
    if feat_dir_raw is None:
        raise ValueError("Debe especificar 'feat_dir' en el config.yml")
    feat_dir = Path(feat_dir_raw) if Path(feat_dir_raw).is_absolute() else (cfg_dir / Path(feat_dir_raw))
    feat_dir = feat_dir.resolve()

    # out_base: por defecto la carpeta que contiene el config
    out_base_raw = cfg.get("out_base", ".")
    out_base = Path(out_base_raw) if Path(out_base_raw).is_absolute() else (cfg_dir / Path(out_base_raw))
    out_base = out_base.resolve()

    # logging / paths (relativos a out_base)
    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})
    logging_cfg = cfg.get("logging", {})
    paths_cfg = cfg.get("paths", {}) or {}

    def resolve_rel_to_out(value, default):
        if value is None:
            value = default
        p = Path(value)
        return p.resolve() if p.is_absolute() else (out_base / p).resolve()

    logs_dir = resolve_rel_to_out(logging_cfg.get("logs_base", "./logs"), "./logs")
    models_root = resolve_rel_to_out(paths_cfg.get("models_root", "./models"), "./models")
    runs_root = resolve_rel_to_out(paths_cfg.get("runs_root", "./runs"), "./runs")
    aggregated_dir = resolve_rel_to_out((cfg.get("outputs", {}) or {}).get("aggregated_dir", "./aggregated"), "./aggregated")

    # ensure directories exist
    for p in [out_base, logs_dir, models_root, runs_root, aggregated_dir]:
        if p is not None:
            Path(p).mkdir(parents=True, exist_ok=True)

    # print resolved paths for debug/tracing
    print("="*60)
    print("CONFIG FILE:", str(cfg_path))
    print("CFG DIR     :", str(cfg_dir))
    print("FEAT DIR    :", str(feat_dir))
    print("OUT BASE    :", str(out_base))
    print("LOGS DIR    :", str(logs_dir))
    print("MODELS ROOT :", str(models_root))
    print("RUNS ROOT   :", str(runs_root))
    print("AGGREGATED  :", str(aggregated_dir))
    print("="*60)

    # add subj to paths_cfg for downstream usage
    subj = cfg.get("subj")
    paths_cfg["subj"] = subj

    # load data
    X3, labels, task = load_subject_data(feat_dir, subj)

    for tval, tname in [(0,"vowels"), (1,"commands")]:
        print(f"=== Tarea: {tname} ===")
        X_task, y_task = prepare_task_dataset(X3, labels, task, tval)
        if X_task.shape[0] == 0:
            print(f"No hay ejemplos para task {tval}")
            continue

        # experiment directories for this task
        exp_task_dir = runs_root / tname
        exp_task_dir.mkdir(parents=True, exist_ok=True)

        # ensure models root subtree exists
        (models_root / subj).mkdir(parents=True, exist_ok=True)

        seeds = training_cfg.get("seeds", [42+i for i in range(10)])
        metrics_all = []
        for seed in seeds:
            run_dir = exp_task_dir / f"run_seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            # store copy of config for traceability
            try:
                shutil.copy(str(cfg_path), str(run_dir / "config_used.yml"))
            except Exception:
                pass

            # pass task name into paths_cfg
            paths_cfg["taskname"] = tname

            acc, rep = run_single_seed(X_task, y_task, seed, run_dir, model_cfg, training_cfg, logging_cfg, paths_cfg,
                                       logs_dir=logs_dir, models_root=models_root, runs_root=runs_root)
            metrics_all.append({"seed":seed, "acc":acc, "report":rep})
            print(f"Seed {seed} -> acc test: {acc:.4f}")

        # save summary (ensure out_base exists)
        summary = {"task": tname, "n_runs": len(metrics_all), "acc_mean": float(np.mean([m["acc"] for m in metrics_all])), "acc_std": float(np.std([m["acc"] for m in metrics_all])), "runs": metrics_all}
        (out_base).mkdir(parents=True, exist_ok=True)
        with open(out_base / f"{tname}_summary.json", "w", encoding="utf8") as fh:
            json.dump(summary, fh, indent=2)
        print("Summary saved to", out_base / f"{tname}_summary.json")

if __name__ == "__main__":
    main()
