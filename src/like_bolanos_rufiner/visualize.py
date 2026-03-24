"""
visualize.py
============
Recopila resultados de experimentos y genera gráficos.

Busca experimentos bajo EXPERIMENTS_ROOT con el prefijo EXPERIMENT_NAME_PREFIX,
recopila todas las métricas, y genera:
  - Accuracy por sujeto (barplot con error bars)
  - Rejection rate por sujeto (binary mode)
  - Matriz de confusión global
  - Accuracy vs rejection scatter
  - Boxplots de accuracy por clase
  - Resumen JSON agregado

Uso:
  python visualize.py
"""

import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1] / "experiments"
EXPERIMENT_NAME_PREFIX = "like_bolanos_rufiner_ESMB"

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150

SUBSET_CONFIG = {
    "vocales":  {"n_classes": 5, "names": ["A", "E", "I", "O", "U"]},
    "comandos": {"n_classes": 6, "names": ["up", "down", "left", "right", "forward", "back"]},
}


def find_latest_experiment(root: Path, prefix: str) -> Path:
    candidates = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not candidates:
        raise FileNotFoundError(f"No se encontraron experimentos con prefijo '{prefix}'")
    return sorted(candidates, key=lambda p: p.name)[-1]


def load_json_safe(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=lambda o: (
            o.tolist() if isinstance(o, (np.ndarray,)) else str(o)
        ))


def collect_experiment_data(exp_root: Path):
    """
    Recopila todos los resultados del experimento.
    Retorna estructura anidada: {sujeto: {subset: {seed_i: {fold_i: metrics}}}}
    """
    subjects_data = {}

    subj_dirs = sorted(
        [p for p in exp_root.iterdir() if p.is_dir() and p.name.startswith("S")],
        key=lambda p: p.name,
    )

    for subj_dir in subj_dirs:
        subj_name = subj_dir.name
        subjects_data[subj_name] = {}

        for subset_dir in sorted(subj_dir.iterdir()):
            if not subset_dir.is_dir():
                continue
            subset_name = subset_dir.name
            subjects_data[subj_name][subset_name] = {}

            seed_dirs = sorted(
                [p for p in subset_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")],
                key=lambda p: int(p.name.split("_")[1]),
            )

            for seed_dir in seed_dirs:
                seed_i = int(seed_dir.name.split("_")[1])
                subjects_data[subj_name][subset_name][seed_i] = {}

                fold_dirs = sorted(
                    [p for p in seed_dir.iterdir() if p.is_dir() and p.name.startswith("fold_")],
                    key=lambda p: int(p.name.split("_")[1]),
                )

                for fold_dir in fold_dirs:
                    fold_i = int(fold_dir.name.split("_")[1])
                    meta = load_json_safe(fold_dir / "metadata.json")
                    metrics = load_json_safe(fold_dir / "metrics.json")

                    if meta is None or meta.get("status") != "success":
                        continue

                    subjects_data[subj_name][subset_name][seed_i][fold_i] = {
                        "accuracy": metrics.get("accuracy", np.nan),
                        "rejection_rate": metrics.get("rejection_rate", np.nan),
                        "n_valid": metrics.get("n_valid", 0),
                        "n_discarded": metrics.get("n_discarded", 0),
                        "n_total": metrics.get("n_total", 0),
                        "per_class_accuracy": metrics.get("per_class_accuracy", {}),
                        "cm": load_json_safe(fold_dir / "confusion_matrix.json"),
                    }

    return subjects_data


def aggregate_results(subjects_data):
    """Agrega resultados por sujeto, subset, seed."""
    results = {}

    for subj_name, subsets_data in subjects_data.items():
        results[subj_name] = {}

        for subset_name, seeds_data in subsets_data.items():
            cfg = SUBSET_CONFIG.get(subset_name, {"n_classes": 5, "names": []})
            n_classes = cfg["n_classes"]
            class_names = cfg["names"]

            all_accuracies = []
            all_rejection_rates = []
            all_per_class = {i: [] for i in range(n_classes)}
            all_cms = []

            for seed_i, folds_data in seeds_data.items():
                for fold_i, fold_data in folds_data.items():
                    all_accuracies.append(fold_data["accuracy"])
                    all_rejection_rates.append(fold_data["rejection_rate"])
                    all_cms.append(fold_data.get("cm"))

                    pca = fold_data["per_class_accuracy"]
                    for c in range(n_classes):
                        val = pca.get(str(c))
                        if val is not None:
                            all_per_class[c].append(val)

            results[subj_name][subset_name] = {
                "accuracy_mean": float(np.mean(all_accuracies)) if all_accuracies else np.nan,
                "accuracy_std": float(np.std(all_accuracies)) if all_accuracies else np.nan,
                "rejection_mean": float(np.mean(all_rejection_rates)) if all_rejection_rates else np.nan,
                "rejection_std": float(np.std(all_rejection_rates)) if all_rejection_rates else np.nan,
                "n_folds": len(all_accuracies),
                "per_class_mean": {
                    class_names[c]: float(np.mean(v)) if v else np.nan
                    for c, v in all_per_class.items()
                },
                "per_class_std": {
                    class_names[c]: float(np.std(v)) if v else np.nan
                    for c, v in all_per_class.items()
                },
                "all_accuracies": all_accuracies,
                "all_rejection_rates": all_rejection_rates,
            }

    return results


def plot_accuracy_por_sujeto(agg_results, output_path):
    """Barplot de accuracy por sujeto con error bars."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (subset_name, cfg) in zip(axes, SUBSET_CONFIG.items()):
        subjects = sorted(agg_results.keys())
        means = []
        stds = []
        labels = []

        for subj in subjects:
            sub_data = agg_results[subj].get(subset_name)
            if sub_data:
                means.append(sub_data["accuracy_mean"])
                stds.append(sub_data["accuracy_std"])
            else:
                means.append(np.nan)
                stds.append(np.nan)
            labels.append(subj)

        x = np.arange(len(subjects))
        bars = ax.bar(x, means, yerr=stds, capsize=3,
                      color="steelblue", alpha=0.8, edgecolor="black")

        chance = 1.0 / cfg["n_classes"]
        ax.axhline(chance, color="red", linestyle="--", linewidth=1.5,
                   label=f"Chance ({chance:.3f})")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{subset_name.capitalize()} ({cfg['n_classes']} clases)")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Accuracy por Sujeto (media ± std)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {output_path}")


def plot_rejection_rate(agg_results, output_path):
    """Barplot de rejection rate por sujeto (solo binary mode)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (subset_name, cfg) in zip(axes, SUBSET_CONFIG.items()):
        subjects = sorted(agg_results.keys())
        means = []
        stds = []

        for subj in subjects:
            sub_data = agg_results[subj].get(subset_name)
            if sub_data:
                means.append(sub_data["rejection_mean"])
                stds.append(sub_data["rejection_std"])
            else:
                means.append(np.nan)
                stds.append(np.nan)

        x = np.arange(len(subjects))
        ax.bar(x, means, yerr=stds, capsize=3,
               color="coral", alpha=0.8, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(subjects, fontsize=8, rotation=45, ha="right")
        ax.set_ylabel("Tasa de Rechazo")
        ax.set_title(f"{subset_name.capitalize()}")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Tasa de Rechazo por Sujeto (media ± std)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {output_path}")


def plot_accuracy_vs_rejection(agg_results, output_path):
    """Scatter plot accuracy vs rejection rate."""
    fig, ax = plt.subplots(figsize=(8, 6))

    markers = {"vocales": "o", "comandos": "s"}
    colors = plt.cm.tab10(np.linspace(0, 1, len(agg_results)))

    for subj_i, subj_name in enumerate(sorted(agg_results.keys())):
        for subset_name, cfg in SUBSET_CONFIG.items():
            sub_data = agg_results[subj_name].get(subset_name)
            if not sub_data:
                continue
            ax.scatter(
                sub_data["rejection_mean"],
                sub_data["accuracy_mean"],
                marker=markers[subset_name],
                color=colors[subj_i],
                s=80,
                alpha=0.8,
                label=f"{subj_name} ({subset_name})",
            )

    ax.set_xlabel("Rejection Rate")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Tasa de Rechazo")
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {output_path}")


def plot_per_class_accuracies(agg_results, output_path):
    """Boxplot de accuracy por clase individual."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (subset_name, cfg) in zip(axes, SUBSET_CONFIG.items()):
        class_names = cfg["names"]
        data_per_class = {cn: [] for cn in class_names}

        for subj_name in sorted(agg_results.keys()):
            sub_data = agg_results[subj_name].get(subset_name)
            if not sub_data:
                continue
            pcm = sub_data["per_class_mean"]
            for c, cn in enumerate(class_names):
                val = pcm.get(cn)
                if val is not None:
                    data_per_class[cn].append(val)

        valid_data = {cn: v for cn, v in data_per_class.items() if v}
        if valid_data:
            bp = ax.boxplot(
                list(valid_data.values()),
                labels=list(valid_data.keys()),
                patch_artist=True,
                showmeans=True,
                meanline=True,
                boxprops=dict(facecolor="lightgreen", alpha=0.7),
                medianprops=dict(linewidth=2, color="darkgreen"),
                meanprops=dict(linewidth=2, color="black", linestyle="--"),
            )
        else:
            ax.text(0.5, 0.5, "Sin datos", ha="center", va="center")
            ax.axis("off")

        chance = 1.0 / cfg["n_classes"]
        ax.axhline(chance, color="red", linestyle="--", linewidth=1.5,
                   label=f"Chance ({chance:.3f})")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{subset_name.capitalize()} — Accuracy por Clase")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Accuracy por Clase Individual (media entre sujetos)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {output_path}")


def plot_global_summary_table(agg_results, output_path):
    """Tabla resumen global de resultados."""
    lines = ["Resumen Global de Resultados", "=" * 60]

    for subset_name, cfg in SUBSET_CONFIG.items():
        lines.append(f"\n{subset_name.upper()} ({cfg['n_classes']} clases)")
        lines.append("-" * 40)
        lines.append(f"{'Sujeto':<10} {'Acc Mean':>10} {'Acc Std':>10} "
                     f"{'Rech Mean':>10} {'Rech Std':>10}")

        all_acc = []
        all_rej = []

        for subj in sorted(agg_results.keys()):
            sub_data = agg_results[subj].get(subset_name)
            if sub_data:
                lines.append(
                    f"{subj:<10} {sub_data['accuracy_mean']:>10.4f} "
                    f"{sub_data['accuracy_std']:>10.4f} "
                    f"{sub_data['rejection_mean']:>10.4f} "
                    f"{sub_data['rejection_std']:>10.4f}"
                )
                all_acc.append(sub_data["accuracy_mean"])
                all_rej.append(sub_data["rejection_mean"])

        if all_acc:
            lines.append("-" * 40)
            lines.append(
                f"{'GLOBAL':<10} {np.mean(all_acc):>10.4f} "
                f"{np.std(all_acc):>10.4f} "
                f"{np.mean(all_rej):>10.4f} "
                f"{np.std(all_rej):>10.4f}"
            )

    text = "\n".join(lines)
    output_path.write_text(text, encoding="utf-8")
    print(f"  Guardado: {output_path}")
    print(text)


def main():
    print("=" * 60)
    print("VISUALIZACIÓN DE RESULTADOS")
    print("=" * 60)

    exp_root = find_latest_experiment(EXPERIMENTS_ROOT, EXPERIMENT_NAME_PREFIX)
    print(f"Experimento: {exp_root.name}")

    vis_dir = exp_root / "visualization_results"
    vis_dir.mkdir(parents=True, exist_ok=True)

    subjects_data = collect_experiment_data(exp_root)
    agg_results = aggregate_results(subjects_data)

    plot_accuracy_por_sujeto(agg_results, vis_dir / "accuracy_por_sujeto.png")
    plot_rejection_rate(agg_results, vis_dir / "rejection_rate.png")
    plot_accuracy_vs_rejection(agg_results, vis_dir / "accuracy_vs_rejection.png")
    plot_per_class_accuracies(agg_results, vis_dir / "per_class_accuracies.png")
    plot_global_summary_table(agg_results, vis_dir / "summary_table.txt")

    save_json(vis_dir / "aggregated_results.json", agg_results)

    config = load_json_safe(exp_root / "experiment_config.json")
    save_json(vis_dir / "experiment_config_copy.json", config)

    print(f"\nGráficos guardados en: {vis_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
