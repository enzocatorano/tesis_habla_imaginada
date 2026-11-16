#!/usr/bin/env python3
"""
Heatmap de precision promedio por banda × sujeto para cada clasificador,
en experimentos con MODO_MATRIZ espacial_banda o frecuencial_canal.
Subplots apilados verticalmente. Columna extra "Promedio" a la derecha.
Todas las celdas muestran media ± desviación estándar.
"""
import json, glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ─── CONFIG ──────────────────────────────────────────────────────────
EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1] / "experiments"
EXPERIMENT_NAME = "ML_TS_Riemman_espacialbandas_20260916-065133_CatoranoBrothers"

BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]
CLASSIFIER_ORDER = ["reg_log_L1", "svm_lineal", "knn", "random_forest", "gradient_boosting"]
CLASSIFIER_DISPLAY = {
    "reg_log_L1": "Reg.Log.L1",
    "svm_lineal": "SVM lineal",
    "knn": "KNN",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
}

# ─── COLLECT ─────────────────────────────────────────────────────────
def collect_data(exp_root):
    clfs_set = set(CLASSIFIER_ORDER)
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for rep in glob.glob(exp_root + "/S*/vocal_vs_comando/*/seed_*/fold_*/*/classification_report.json", recursive=True):
        parts = [p for p in rep.split("/") if p]
        sxx = [p for p in parts if p.startswith("S") and p.endswith("_clean")][0]
        band = [p for p in parts if p.endswith("_band")]
        clf = [p for p in parts if p in clfs_set]
        if not band or not clf:
            continue
        r = json.load(open(rep))
        mavg = r.get("macro avg", {})
        data[clf[0]][band[0]][sxx].append(mavg.get("precision", np.nan))

    subjects = sorted({s for cd in data.values() for bd in cd.values() for s in bd})
    return data, subjects


# ─── PLOT ────────────────────────────────────────────────────────────
def plot_heatmaps(data, subjects, output_path):
    n_clf = len(CLASSIFIER_ORDER)
    n_bands = len(BAND_NAMES)
    n_subj = len(subjects)
    n_cols = n_subj + 1  # subjects + promedio

    fig, axes = plt.subplots(n_clf, 1, figsize=(0.7 * n_cols, 4.2 * n_clf))
    fig.suptitle("Precision promedio por banda y sujeto (± std entre folds)",
                 fontsize=16, fontweight="bold", y=0.995)

    vmin, vmax = 45.0, 100.0
    subject_labels = [s.replace("_clean", "") for s in subjects]

    for ax_idx, (ax, clf) in enumerate(zip(axes, CLASSIFIER_ORDER)):
        # ── Build means and stds matrices (bands × subjects) ──
        means = np.full((n_bands, n_subj), np.nan)
        stds = np.full((n_bands, n_subj), np.nan)
        for i, band in enumerate(BAND_NAMES):
            band_tag = f"{band}_band"
            for j, sxx in enumerate(subjects):
                vals = data.get(clf, {}).get(band_tag, {}).get(sxx, [])
                vals = [v for v in vals if not np.isnan(v)]
                if vals:
                    means[i, j] = np.mean(vals) * 100.0
                    stds[i, j] = np.std(vals, ddof=1) * 100.0 if len(vals) > 1 else 0.0

        # ── Promedio column (mean/std across subjects per band) ──
        prom_mean = np.nanmean(means, axis=1)     # (5,)
        prom_std = np.nanstd(means, axis=1, ddof=1)  # (5,)

        # ── Full matrix (bands × n_cols) for coloring ──
        full_mean = np.column_stack([means, prom_mean])   # (5, 16)
        full_std = np.column_stack([stds, prom_std])

        # ── Image ──
        im = ax.imshow(full_mean, cmap="cividis", vmin=vmin, vmax=vmax, aspect="equal")

        # ── White border around promedio column ──
        for i in range(n_bands):
            rect = Rectangle((n_subj - 0.5, i - 0.5), 1.0, 1.0,
                              linewidth=2.5, edgecolor="white", facecolor="none", zorder=5)
            ax.add_patch(rect)

        # ── Annotate all cells ──
        for i in range(n_bands):
            for j in range(n_cols):
                m, s = full_mean[i, j], full_std[i, j]
                if np.isnan(m):
                    continue
                text_color = "white" if m < 55 else "black"
                txt = f"{m:.1f}\n±{s:.1f}" if not np.isnan(s) else f"{m:.1f}"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=6.5, fontweight="bold", color=text_color)

        # ── Axes labels ──
        xtick_labels = subject_labels + ["Promedio"]
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n_bands))
        ax.set_yticklabels(BAND_NAMES, fontsize=9)
        ax.set_title(CLASSIFIER_DISPLAY.get(clf, clf), fontsize=12, fontweight="bold",
                     loc="left", pad=8)

    # ── Shared colorbar ──
    fig.subplots_adjust(right=0.91, hspace=0.35)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap="cividis", norm=plt.Normalize(vmin=vmin, vmax=vmax)),
                        cax=cbar_ax)
    cbar.set_label("Precision (%)", fontsize=10)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {output_path}")


# ─── MAIN ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    exp_root = str(EXPERIMENTS_ROOT / EXPERIMENT_NAME)
    data, subjects = collect_data(exp_root)
    print(f"Clasificadores: {len(data)} | Sujetos: {len(subjects)}")
    out = EXPERIMENTS_ROOT / EXPERIMENT_NAME / "precision_heatmap_bandas.png"
    plot_heatmaps(data, subjects, out)
