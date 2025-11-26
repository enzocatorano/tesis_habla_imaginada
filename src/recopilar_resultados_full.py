# src/recopilar_resultados_full.py
"""
Recopila resultados de un experimento (estructura creada por run_full_experperiments.py)
y genera:

Por sujeto (en experiments/.../results_summary/<SUJETO>/):
  - summary_vocales.json, summary_comandos.json
  - summary_results.json (ambos subsets juntos)
  - band_accuracy_boxplots.png (vocales y comandos a la vez)
  - metrics_bar_vocales_comandos.png (barras P/R/F1 con error)
  - confusion_meanstd_vocales_comandos.png (matrices intrasujeto vocales+comandos)
  - learning_curves_vocales_comandos.png (learning curves por sujeto, dual-axis, vocales+comandos)

Global (en experiments/.../results_summary/global/):
  - global_summary_vocales.json, global_summary_comandos.json
  - global_metrics_per_subject_vocales_comandos.png
  - global_confusion_meanstd_vocales_comandos.png
  - global_band_accuracy_vocales_comandos.png
  - global_learning_curves_vocales_comandos.png
  - global_summary.json (todo junto)
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- CONFIG ----------------
EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1] / "experiments"
EXPERIMENT_NAME_PREFIX = "EEGNet_full_baseline"  # prefix to select experiment dir
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed_aug"
OUTPUT_SUBDIR = "results_summary"
BAND_LABELS = ["delta 0-4", "theta 4-8", "alpha 8-12", "beta 12-32", "gamma 32-64"]
VOCAL_CLASS_NAMES = ['A','E','I','O','U']
COMANDO_CLASS_NAMES = ['Arriba','Abajo','Izquierda','Derecha','Adelante','Atras']
SUBSETS = ["vocales", "comandos"]
sns.set(style="whitegrid")
# ----------------------------------------

def find_latest_experiment(root: Path, prefix: str) -> Path:
    candidates = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not candidates:
        raise FileNotFoundError(f"No experiment folders starting with '{prefix}' found under {root}")
    candidates_sorted = sorted(candidates, key=lambda p: p.name)
    return candidates_sorted[-1]

def load_json_safe(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf8"))
    except Exception:
        return None

def save_json_safe(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf8") as fh:
        json.dump(obj, fh, indent=2, default=lambda o: (o.tolist() if isinstance(o, np.ndarray) else o))

def pad_and_agg_series(list_of_lists):
    if not list_of_lists:
        return np.array([]), np.array([])
    lengths = [len(l) for l in list_of_lists]
    max_len = max(lengths)
    arr = np.full((len(list_of_lists), max_len), np.nan, dtype=float)
    for i, l in enumerate(list_of_lists):
        if len(l) > 0:
            arr[i, :len(l)] = np.array(l, dtype=float)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    return mean, std

def safe_mean_std(arr, axis=0):
    arr = np.array(arr, dtype=float)
    if arr.size == 0:
        return np.array([]), np.array([])
    mean = np.nanmean(arr, axis=axis)
    std = np.nanstd(arr, axis=axis)
    return mean, std

# ---------------- Find experiment ----------------
EXP_ROOT = find_latest_experiment(EXPERIMENTS_ROOT, EXPERIMENT_NAME_PREFIX)
print("[Collector] Using experiment root:", EXP_ROOT)

RESULTS_ROOT = EXP_ROOT / OUTPUT_SUBDIR
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

# ---------------- Discover subject dirs ----------------
subject_dirs = [p for p in EXP_ROOT.iterdir() if p.is_dir() and p.name.upper().startswith("S")]
subject_dirs = sorted(subject_dirs)
print(f"[Collector] Found {len(subject_dirs)} subject dirs")

# global container
global_per_subject = {}

for subj_dir in subject_dirs:
    subj_name = subj_dir.name
    print(f"\n[Collector] Processing subject: {subj_name}")
    subj_npz_path = DATA_DIR / f"{subj_name}.npz"
    if not subj_npz_path.exists():
        matches = list(DATA_DIR.glob(f"{subj_name.split('_')[0]}*.npz"))
        subj_npz_path = matches[0] if matches else None
    if subj_npz_path is None or not subj_npz_path.exists():
        print(f"  Warning: subject data .npz not found for {subj_name}. Band-accuracy por run no disponible.")
        subj_data = None
        banda_all = None
    else:
        subj_data = np.load(subj_npz_path, allow_pickle=True)
        Y_all = subj_data['labels']
        banda_all = Y_all[:, 3].astype(int)

    subject_out = RESULTS_ROOT / subj_name
    subject_out.mkdir(parents=True, exist_ok=True)

    subj_summary = {}

    # --- Recolección por subset ---
    for subset in SUBSETS:
        subset_dir = subj_dir / subset
        if not subset_dir.exists():
            print(f"  subset dir missing: {subset_dir} -> skipping subset")
            continue

        run_metrics = []      # dicts {"precision","recall","f1"}
        run_confusions = []   # cm% per run
        run_band_accs = []    # (n_bands,)
        run_train_losses = [] # listas
        run_val_losses = []
        run_val_accs = []

        seeds = sorted([p for p in subset_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")]) or [subset_dir]

        for seed_dir in seeds:
            fold_dirs = sorted([p for p in seed_dir.iterdir() if p.is_dir() and p.name.startswith("fold_")])
            for fold_dir in fold_dirs:
                meta = load_json_safe(fold_dir / "metadata.json")
                if meta is None or meta.get("status") != "success":
                    continue

                cr = load_json_safe(fold_dir / "classification_report.json")
                if cr is None:
                    continue

                macro = cr.get("macro avg") or cr.get("macro_avg") or cr.get("macro-average")
                if macro is None:
                    # buscar alguna key con "macro"
                    candidates = [k for k in cr.keys() if "macro" in k]
                    macro = cr[candidates[0]] if candidates else None
                if macro is None:
                    continue

                prec = float(macro.get("precision", float('nan')))
                rec = float(macro.get("recall", float('nan')))
                f1  = float(macro.get("f1-score", macro.get("f1_score", float('nan'))))

                # confusion matrix
                cm_path = fold_dir / "confusion_matrix.npy"
                cm_pct = None
                if cm_path.exists():
                    try:
                        cm = np.load(cm_path)
                        with np.errstate(divide='ignore', invalid='ignore'):
                            row_sums = cm.sum(axis=1, keepdims=True)
                            cm_pct = np.divide(cm, row_sums, where=(row_sums != 0)) * 100.0
                            cm_pct = np.nan_to_num(cm_pct, nan=0.0)
                    except Exception:
                        cm_pct = None

                # band acc por run
                band_vec = None
                if subj_data is not None:
                    test_idx_global = meta.get("test_idx_global", None)
                    preds_path = fold_dir / "test_preds.npz"
                    if test_idx_global is not None and preds_path.exists():
                        try:
                            d = np.load(preds_path, allow_pickle=True)
                            y_true = d["y_true"]
                            y_pred = d["y_pred"]
                            idxs = np.array(test_idx_global, dtype=int)
                            bands_for_test = banda_all[idxs]
                            n_bands = len(BAND_LABELS)
                            band_accs = []
                            for b in range(n_bands):
                                mask_b = (bands_for_test == b)
                                if mask_b.sum() == 0:
                                    band_accs.append(np.nan)
                                else:
                                    band_accs.append(float(np.mean(y_true[mask_b] == y_pred[mask_b])))
                            band_vec = np.array(band_accs, dtype=float)
                        except Exception:
                            band_vec = None

                train_metrics = load_json_safe(fold_dir / "train_metrics.json")
                if train_metrics:
                    tl = train_metrics.get("train_losses", None)
                    vl = train_metrics.get("val_losses", None)
                    va = train_metrics.get("val_accs", None)
                else:
                    tl = vl = va = None

                run_metrics.append({"precision": prec, "recall": rec, "f1": f1})
                if cm_pct is not None:
                    run_confusions.append(cm_pct)
                if band_vec is not None:
                    run_band_accs.append(band_vec)
                if tl: run_train_losses.append(tl)
                if vl: run_val_losses.append(vl)
                if va: run_val_accs.append(va)

        # Agregados subset-sujeto
        if run_metrics:
            rm_arr = np.array([[r["precision"], r["recall"], r["f1"]] for r in run_metrics])
        else:
            rm_arr = np.zeros((0, 3))

        run_band_accs_arr = np.vstack(run_band_accs) if run_band_accs else np.zeros((0, len(BAND_LABELS)))
        run_conf_arr = np.stack(run_confusions) if run_confusions else np.zeros((0, 1, 1))

        # métricas
        prec_mean = float(np.nanmean(rm_arr[:, 0])) if rm_arr.size else None
        prec_std  = float(np.nanstd(rm_arr[:, 0])) if rm_arr.size else None
        rec_mean  = float(np.nanmean(rm_arr[:, 1])) if rm_arr.size else None
        rec_std   = float(np.nanstd(rm_arr[:, 1])) if rm_arr.size else None
        f1_mean   = float(np.nanmean(rm_arr[:, 2])) if rm_arr.size else None
        f1_std    = float(np.nanstd(rm_arr[:, 2])) if rm_arr.size else None

        if run_conf_arr.size:
            cm_mean = np.nanmean(run_conf_arr, axis=0)
            cm_std  = np.nanstd(run_conf_arr, axis=0)
        else:
            cm_mean = np.array([])
            cm_std  = np.array([])

        if run_band_accs_arr.size:
            band_mean = np.nanmean(run_band_accs_arr, axis=0).tolist()
            band_std  = np.nanstd(run_band_accs_arr, axis=0).tolist()
        else:
            band_mean = []
            band_std = []

        train_loss_mean, train_loss_std = pad_and_agg_series(run_train_losses)
        val_loss_mean, val_loss_std     = pad_and_agg_series(run_val_losses)
        val_acc_mean, val_acc_std       = pad_and_agg_series(run_val_accs)

        subset_summary = {
            "n_runs": int(len(run_metrics)),
            "run_metrics": [{"precision": float(r["precision"]), "recall": float(r["recall"]), "f1": float(r["f1"])} for r in run_metrics],
            "precision_mean": prec_mean,
            "precision_std": prec_std,
            "recall_mean": rec_mean,
            "recall_std": rec_std,
            "f1_mean": f1_mean,
            "f1_std": f1_std,
            "confusion_mean_pct": cm_mean.tolist() if cm_mean.size else [],
            "confusion_std_pct": cm_std.tolist() if cm_std.size else [],
            "band_accuracy_mean": band_mean,
            "band_accuracy_std": band_std,
            "train_loss_mean": train_loss_mean.tolist(),
            "train_loss_std": train_loss_std.tolist(),
            "val_loss_mean": val_loss_mean.tolist(),
            "val_loss_std": val_loss_std.tolist(),
            "val_acc_mean": val_acc_mean.tolist(),
            "val_acc_std": val_acc_std.tolist(),
        }

        subj_summary[subset] = subset_summary
        save_json_safe(subject_out / f"summary_{subset}.json", subset_summary)
        print(f"  Saved {subject_out / f'summary_{subset}.json'}")

    # ---------- Plots intrasujeto ----------

    # 1) Band accuracy boxplots (vocales y comandos en una sola figura)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, subset, title in zip(axes, SUBSETS, ["Vocales", "Comandos"]):
        ss = subj_summary.get(subset, None)
        if ss is None:
            ax.axis("off")
            continue

        # reconstruir band_accs por run (otra vez) para boxplot
        subset_dir = subj_dir / subset
        if not subset_dir.exists() or subj_data is None:
            ax.text(0.5, 0.5, "No data", ha='center')
            ax.axis("off")
            continue

        band_runs = []
        seeds = sorted([p for p in subset_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")]) or [subset_dir]
        for seed_dir in seeds:
            fold_dirs = sorted([p for p in seed_dir.iterdir() if p.is_dir() and p.name.startswith("fold_")])
            for fold_dir in fold_dirs:
                meta = load_json_safe(fold_dir / "metadata.json")
                if meta is None or meta.get("status") != "success":
                    continue
                preds_path = fold_dir / "test_preds.npz"
                if not preds_path.exists():
                    continue
                try:
                    d = np.load(preds_path, allow_pickle=True)
                    y_true = d["y_true"]
                    y_pred = d["y_pred"]
                    test_idx_g = meta.get("test_idx_global", None)
                    if test_idx_g is None:
                        continue
                    idxs = np.array(test_idx_g, dtype=int)
                    bands_for_test = banda_all[idxs]
                    n_bands = len(BAND_LABELS)
                    band_accs = []
                    for b in range(n_bands):
                        maskb = (bands_for_test == b)
                        if maskb.sum() == 0:
                            band_accs.append(np.nan)
                        else:
                            band_accs.append(float(np.mean(y_true[maskb] == y_pred[maskb])))
                    band_runs.append(band_accs)
                except Exception:
                    continue
        if band_runs:
            band_runs_arr = np.vstack(band_runs)  # (n_runs, n_bands)
            sns.boxplot(data=band_runs_arr, ax=ax)
            ax.set_xticks(np.arange(len(BAND_LABELS)))
            ax.set_xticklabels(BAND_LABELS, rotation=30, ha='right')
            ax.set_ylim(max(0.0, np.min(band_runs) - 0.1), min(1.0, np.max(band_runs) + 0.1))
            ax.set_title(f"Band accuracies - {title}")
            ax.set_ylabel("Accuracy")
        else:
            ax.text(0.5, 0.5, "No band data", ha='center')
            ax.axis("off")

    fig.suptitle(f"Band accuracies - subject {subj_name}")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fname_band = subject_out / "band_accuracy_boxplots.png"
    fig.savefig(fname_band, dpi=200)
    plt.close(fig)
    print(f"  Saved band boxplots -> {fname_band}")

    # 2) Metrics bar chart (vocales/comandos, precision/recall/f1)
    metrics_labels = ["precision", "recall", "f1"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    group_means = []
    group_stds = []
    for subset in SUBSETS:
        ss = subj_summary.get(subset, None)
        if ss is None:
            group_means.append([np.nan, np.nan, np.nan])
            group_stds.append([0.0, 0.0, 0.0])
        else:
            group_means.append([
                ss["precision_mean"] if ss["precision_mean"] is not None else float('nan'),
                ss["recall_mean"] if ss["recall_mean"] is not None else float('nan'),
                ss["f1_mean"] if ss["f1_mean"] is not None else float('nan')
            ])
            group_stds.append([
                ss["precision_std"] if ss["precision_std"] is not None else 0.0,
                ss["recall_std"] if ss["recall_std"] is not None else 0.0,
                ss["f1_std"] if ss["f1_std"] is not None else 0.0
            ])
    group_means = np.array(group_means)
    group_stds = np.array(group_stds)

    fig, ax = plt.subplots(figsize=(8,5))
    n_groups = len(SUBSETS)
    n_metrics = len(metrics_labels)
    ind = np.arange(n_groups)
    width = 0.2
    offsets = np.linspace(-width, width, n_metrics)
    for m in range(n_metrics):
        vals = group_means[:, m]
        errs = group_stds[:, m]
        ax.bar(ind + offsets[m], vals, width=width, yerr=errs, label=metrics_labels[m].capitalize(),
               color=colors[m], capsize=5)
    ax.set_xticks(ind)
    ax.set_xticklabels([s.capitalize() for s in SUBSETS])
    ax.set_ylim(max(0.0, np.min(group_means) - 0.1), min(1.0, np.max(group_means) + 0.1))
    ax.set_ylabel("Score")
    ax.set_title(f"Precision / Recall / F1 (mean ± std) - subject {subj_name}")
    ax.legend()
    fig.tight_layout()
    fname_bar = subject_out / "metrics_bar_vocales_comandos.png"
    fig.savefig(fname_bar, dpi=200)
    plt.close(fig)
    print(f"  Saved metrics bar chart -> {fname_bar}")

    # 3) Confusion matrices intrasujeto (vocales/comandos en una figura)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, subset in zip(axes, SUBSETS):
        ss = subj_summary.get(subset, None)
        if ss is None or not ss["confusion_mean_pct"]:
            ax.axis("off")
            continue
        cm_mean = np.array(ss["confusion_mean_pct"])
        cm_std  = np.array(ss["confusion_std_pct"])
        n = cm_mean.shape[0]
        annot = np.empty((n, n), dtype=object)
        for r in range(n):
            for c in range(n):
                annot[r, c] = f"{cm_mean[r,c]:.1f}\n±{cm_std[r,c]:.1f}"
        sns.heatmap(cm_mean, annot=annot, fmt="", cmap="cividis", ax=ax,
                    cbar_kws={'label':'% (row-normalized mean)'},
                    linewidths=0.5, linecolor='gray', annot_kws={"size":10})
        ax.set_title(f"{subset.capitalize()} (mean ± std %)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        if subset == "vocales":
            ax.set_xticklabels(VOCAL_CLASS_NAMES, rotation=45, ha='right')
            ax.set_yticklabels(VOCAL_CLASS_NAMES, rotation=0)
        else:
            ax.set_xticklabels(COMANDO_CLASS_NAMES, rotation=45, ha='right')
            ax.set_yticklabels(COMANDO_CLASS_NAMES, rotation=0)
    fig.suptitle(f"Confusion matrices (mean ± std) - subject {subj_name}")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fname_cm = subject_out / "confusion_meanstd_vocales_comandos.png"
    fig.savefig(fname_cm, dpi=200)
    plt.close(fig)
    print(f"  Saved subject confusion matrices -> {fname_cm}")

    # 4) Learning curves intrasujeto (vocales + comandos)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax_loss, subset in zip(axes, SUBSETS):
        ss = subj_summary.get(subset, None)
        if ss is None:
            ax_loss.axis("off")
            continue
        tl_mean = np.array(ss["train_loss_mean"], dtype=float)
        tl_std  = np.array(ss["train_loss_std"], dtype=float)
        vl_mean = np.array(ss["val_loss_mean"], dtype=float)
        vl_std  = np.array(ss["val_loss_std"], dtype=float)
        va_mean = np.array(ss["val_acc_mean"], dtype=float)
        va_std  = np.array(ss["val_acc_std"], dtype=float)

        if tl_mean.size == 0 and vl_mean.size == 0 and va_mean.size == 0:
            ax_loss.axis("off")
            continue

        epochs = np.arange(1, max(tl_mean.size, vl_mean.size, va_mean.size) + 1)

        # eje de pérdidas
        color_train = "tab:blue"
        color_val = "tab:orange"
        if tl_mean.size:
            ax_loss.plot(epochs[:tl_mean.size], tl_mean, color=color_train, label="Train Loss")
            ax_loss.fill_between(epochs[:tl_mean.size], tl_mean - tl_std, tl_mean + tl_std, alpha=0.2, color=color_train)
        if vl_mean.size:
            ax_loss.plot(epochs[:vl_mean.size], vl_mean, color=color_val, label="Val Loss")
            ax_loss.fill_between(epochs[:vl_mean.size], vl_mean - vl_std, vl_mean + vl_std, alpha=0.2, color=color_val)
        ax_loss.set_ylabel("Loss")
        ax_loss.grid(True)

        # eje de accuracy
        ax_acc = ax_loss.twinx()
        color_acc = "tab:green"
        if va_mean.size:
            ax_acc.plot(epochs[:va_mean.size], va_mean, color=color_acc, linestyle='--', label="Val Acc")
            ax_acc.fill_between(epochs[:va_mean.size], va_mean - va_std, va_mean + va_std, alpha=0.15, color=color_acc)
        ax_acc.set_ylabel("Val Accuracy")
        # combinar leyendas
        lines_loss, labels_loss = ax_loss.get_legend_handles_labels()
        lines_acc, labels_acc = ax_acc.get_legend_handles_labels()
        ax_loss.legend(lines_loss + lines_acc, labels_loss + labels_acc, loc='upper right')
        ax_loss.set_title(f"Learning curves (averaged) - {subset}, subject {subj_name}")

    axes[-1].set_xlabel("Epoch")
    fig.tight_layout()
    fname_lc = subject_out / "learning_curves_vocales_comandos.png"
    fig.savefig(fname_lc, dpi=200)
    plt.close(fig)
    print(f"  Saved subject learning curves -> {fname_lc}")

    # Guardar resumen sujeto-completo
    subj_summary_path = subject_out / "summary_results.json"
    save_json_safe(subj_summary_path, subj_summary)
    print(f"  Saved subject summary JSON -> {subj_summary_path}")

    # guardar datos para global
    global_per_subject[subj_name] = {}
    for subset in SUBSETS:
        ss = subj_summary.get(subset, None)
        if ss is None:
            global_per_subject[subj_name][subset] = None
        else:
            global_per_subject[subj_name][subset] = {
                "precision_mean": ss["precision_mean"],
                "precision_std": ss["precision_std"],
                "recall_mean": ss["recall_mean"],
                "recall_std": ss["recall_std"],
                "f1_mean": ss["f1_mean"],
                "f1_std": ss["f1_std"],
                "band_mean": ss["band_accuracy_mean"],
                "band_std": ss["band_accuracy_std"],
                "confusion_mean_pct": ss["confusion_mean_pct"],
                "confusion_std_pct": ss["confusion_std_pct"],
                "train_loss_mean": ss["train_loss_mean"],
                "train_loss_std": ss["train_loss_std"],
                "val_loss_mean": ss["val_loss_mean"],
                "val_loss_std": ss["val_loss_std"],
                "val_acc_mean": ss["val_acc_mean"],
                "val_acc_std": ss["val_acc_std"],
            }

# ---------------- Global aggregation ----------------
print("\n[Collector] Computing global summaries across subjects...")

global_out = RESULTS_ROOT / "global"
global_out.mkdir(parents=True, exist_ok=True)

# 1) Global summaries por subset
for subset in SUBSETS:
    subj_names = []
    p_means = []
    p_stds  = []
    r_means = []
    r_stds  = []
    f_means = []
    f_stds  = []
    band_means_list = []
    conf_means_list = []
    tl_means_subject = []
    vl_means_subject = []
    va_means_subject = []

    for subj_name, info in global_per_subject.items():
        entry = info.get(subset)
        if entry is None:
            continue
        subj_names.append(subj_name)
        p_means.append(entry["precision_mean"])
        p_stds.append(entry["precision_std"])
        r_means.append(entry["recall_mean"])
        r_stds.append(entry["recall_std"])
        f_means.append(entry["f1_mean"])
        f_stds.append(entry["f1_std"])
        band_means_list.append(entry["band_mean"] if entry["band_mean"] else [float('nan')] * len(BAND_LABELS))
        if entry["confusion_mean_pct"]:
            conf_means_list.append(np.array(entry["confusion_mean_pct"]))
        # curves
        tl_means_subject.append(entry["train_loss_mean"])
        vl_means_subject.append(entry["val_loss_mean"])
        va_means_subject.append(entry["val_acc_mean"])

    global_summary = {
        "subjects": subj_names,
        "precision_means_per_subject": p_means,
        "precision_stds_per_subject": p_stds,
        "recall_means_per_subject": r_means,
        "recall_stds_per_subject": r_stds,
        "f1_means_per_subject": f_means,
        "f1_stds_per_subject": f_stds,
        "band_means_per_subject": band_means_list,
        "confusion_means_per_subject": [cm.tolist() for cm in conf_means_list],
        "train_loss_means_per_subject": tl_means_subject,
        "val_loss_means_per_subject": vl_means_subject,
        "val_acc_means_per_subject": va_means_subject,
    }
    save_json_safe(global_out / f"global_summary_{subset}.json", global_summary)
    print(f"  Saved global summary json for subset {subset}")

# 2) Figura global metrics per subject (vocales arriba, comandos abajo)
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
for ax, subset in zip(axes, SUBSETS):
    g = load_json_safe(global_out / f"global_summary_{subset}.json")
    if g is None or not g["subjects"]:
        ax.text(0.5, 0.5, f"No data for {subset}", ha='center')
        ax.axis("off")
        continue
    names = g["subjects"]
    x = np.arange(len(names))
    width = 0.2
    p_means = np.array(g["precision_means_per_subject"], dtype=float)
    p_stds  = np.array(g["precision_stds_per_subject"], dtype=float)
    r_means = np.array(g["recall_means_per_subject"], dtype=float)
    r_stds  = np.array(g["recall_stds_per_subject"], dtype=float)
    f_means = np.array(g["f1_means_per_subject"], dtype=float)
    f_stds  = np.array(g["f1_stds_per_subject"], dtype=float)
    offsets = [-width, 0, width]
    ax.bar(x + offsets[0], p_means, width=width, yerr=p_stds, label="Precision", color="#1f77b4", capsize=4)
    ax.bar(x + offsets[1], r_means, width=width, yerr=r_stds, label="Recall", color="#ff7f0e", capsize=4)
    ax.bar(x + offsets[2], f_means, width=width, yerr=f_stds, label="F1", color="#2ca02c", capsize=4)
    ax.set_ylabel("Score")
    limite_inferior = np.nanmin(np.array([p_means, r_means, f_means])) - 0.1
    limite_superior = np.nanmax(np.array([p_means, r_means, f_means])) + 0.1
    ax.set_ylim(max(0, limite_inferior), min(1.0, limite_superior))
    ax.set_title(f"{subset.capitalize()} - per-subject mean ± std")
    ax.legend()
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')

global_fig_path = global_out / "global_metrics_per_subject_vocales_comandos.png"
fig.tight_layout()
fig.savefig(global_fig_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"[Collector] Saved global per-subject metrics figure -> {global_fig_path}")

# 3) Global confusion matrices (vocales/comandos)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, subset in zip(axes, SUBSETS):
    g = load_json_safe(global_out / f"global_summary_{subset}.json")
    conf_list = g.get("confusion_means_per_subject") if g else None
    if not conf_list:
        ax.axis("off")
        continue
    conf_arr = np.stack([np.array(cm) for cm in conf_list], axis=0)
    cm_mean = np.nanmean(conf_arr, axis=0)
    cm_std  = np.nanstd(conf_arr, axis=0)
    n = cm_mean.shape[0]
    annot = np.empty((n, n), dtype=object)
    for r in range(n):
        for c in range(n):
            annot[r, c] = f"{cm_mean[r,c]:.1f}\n±{cm_std[r,c]:.1f}"
    sns.heatmap(cm_mean, annot=annot, fmt="", cmap="cividis", ax=ax,
                cbar_kws={'label': '% (row-normalized mean)'},
                linewidths=0.5, linecolor='gray', annot_kws={"size":10})
    ax.set_title(f"Global confusion (mean ± std) - {subset}")
    if subset == "vocales":
        ax.set_xticklabels(VOCAL_CLASS_NAMES, rotation=45, ha='right')
        ax.set_yticklabels(VOCAL_CLASS_NAMES, rotation=0)
    else:
        ax.set_xticklabels(COMANDO_CLASS_NAMES, rotation=45, ha='right')
        ax.set_yticklabels(COMANDO_CLASS_NAMES, rotation=0)

fig.suptitle("Global confusion matrices (mean ± std across subjects)")
fig.tight_layout(rect=[0, 0, 1, 0.96])
global_cm_path = global_out / "global_confusion_meanstd_vocales_comandos.png"
fig.savefig(global_cm_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"[Collector] Saved global confusion matrices -> {global_cm_path}")

# 4) Global band accuracy: figura con 2 subplots (vocales arriba, comandos abajo), BOXplots entre sujetos
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
x = np.arange(len(BAND_LABELS))

for ax, subset in zip(axes, SUBSETS):
    g = load_json_safe(global_out / f"global_summary_{subset}.json")
    if g is None or not g["subjects"]:
        ax.axis("off")
        continue

    band_means_per_subject = g["band_means_per_subject"]
    # band_means_per_subject es lista de listas: [ [banda0, banda1, ...], ... ] por sujeto
    if not band_means_per_subject:
        ax.axis("off")
        continue

    band_arr = np.array(band_means_per_subject, dtype=float)  # shape: (n_subjects, n_bands)

    if band_arr.size == 0:
        ax.axis("off")
        continue

    # Boxplot: cada columna = una banda, las filas = sujetos
    # sns.boxplot espera shape (n_samples, n_features) si se le pasa como "data"
    sns.boxplot(data=band_arr, ax=ax)
    ax.set_ylim(max(0.0, np.nanmin(band_arr) - 0.1), min(1.0, np.nanmax(band_arr) + 0.1))
    ax.set_xticks(np.arange(len(BAND_LABELS)))
    ax.set_xticklabels(BAND_LABELS, rotation=30, ha='right')
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Global band accuracy (boxplot across subjects) - {subset}")

axes[-1].set_xlabel("Banda")
fig.tight_layout()
global_band_path = global_out / "global_band_accuracy_vocales_comandos.png"
fig.savefig(global_band_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"[Collector] Saved global band accuracy boxplots -> {global_band_path}")

# 5) Global learning curves (vocales + comandos)
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
for ax_loss, subset in zip(axes, SUBSETS):
    g = load_json_safe(global_out / f"global_summary_{subset}.json")
    if g is None or not g["subjects"]:
        ax_loss.axis("off")
        continue

    # cada sujeto tiene una curva media; agregamos esas curvas entre sujetos
    tl_subject = g["train_loss_means_per_subject"]
    vl_subject = g["val_loss_means_per_subject"]
    va_subject = g["val_acc_means_per_subject"]
    tl_mean, tl_std = pad_and_agg_series(tl_subject)
    vl_mean, vl_std = pad_and_agg_series(vl_subject)
    va_mean, va_std = pad_and_agg_series(va_subject)

    if tl_mean.size == 0 and vl_mean.size == 0 and va_mean.size == 0:
        ax_loss.axis("off")
        continue

    epochs = np.arange(1, max(tl_mean.size, vl_mean.size, va_mean.size) + 1)
    color_train = "tab:blue"
    color_val = "tab:orange"
    if tl_mean.size:
        ax_loss.plot(epochs[:tl_mean.size], tl_mean, color=color_train, label="Train Loss")
        ax_loss.fill_between(epochs[:tl_mean.size], tl_mean - tl_std, tl_mean + tl_std, alpha=0.2, color=color_train)
    if vl_mean.size:
        ax_loss.plot(epochs[:vl_mean.size], vl_mean, color=color_val, label="Val Loss")
        ax_loss.fill_between(epochs[:vl_mean.size], vl_mean - vl_std, vl_mean + vl_std, alpha=0.2, color=color_val)
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True)

    ax_acc = ax_loss.twinx()
    color_acc = "tab:green"
    if va_mean.size:
        ax_acc.plot(epochs[:va_mean.size], va_mean, color=color_acc, linestyle='--', label="Val Acc")
        ax_acc.fill_between(epochs[:va_mean.size], va_mean - va_std, va_mean + va_std, alpha=0.15, color=color_acc)
    ax_acc.set_ylabel("Val Accuracy")
    lines_loss, labels_loss = ax_loss.get_legend_handles_labels()
    lines_acc, labels_acc = ax_acc.get_legend_handles_labels()
    ax_loss.legend(lines_loss + lines_acc, labels_loss + labels_acc, loc='upper right')
    ax_loss.set_title(f"Global learning curves (averaged across subjects) - {subset}")

axes[-1].set_xlabel("Epoch")
fig.tight_layout()
global_lc_path = global_out / "global_learning_curves_vocales_comandos.png"
fig.savefig(global_lc_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"[Collector] Saved global learning curves -> {global_lc_path}")

# 6) Resumen global final
final_global_summary = {
    "experiment_root": str(EXP_ROOT),
    "n_subjects_processed": len(global_per_subject),
    "per_subject_summary": global_per_subject
}
save_json_safe(global_out / "global_summary.json", final_global_summary)
print(f"[Collector] Saved final global summary -> {global_out / 'global_summary.json'}")

print("\n[Collector] Done. Results are under:", RESULTS_ROOT)
