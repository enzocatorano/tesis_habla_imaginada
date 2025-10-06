# notebooks/plot_aggregate_curves.py
# Ejecutar: python notebooks/plot_aggregate_curves.py --exp experiments/exp_mlp_s01 --task vowels
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import argparse

def load_metrics_from_run(results_json_path):
    """
    Carga metrics_epochs.json producidos por Entrenador o el file metrics_epochs_tb.json copiado.
    Espera dict con keys: train_losses (list), val_losses (list or None), val_accs (list or None)
    """
    if not results_json_path.exists():
        return None
    with open(results_json_path, "r", encoding="utf8") as fh:
        data = json.load(fh)
    # compatibilidad: diferentes nombres posibles
    # trainer.ajustar devuelve keys: train_losses, val_losses, val_accs
    # si el usuario guardó otro dict, intentamos encontrar los arrays
    train = data.get("train_losses", None)
    val = data.get("val_losses", None)
    acc = data.get("val_accs", None)
    # Also accept metrics_epochs_tb.json which may have same format
    if train is None:
        # try other keys heuristics
        keys = list(data.keys())
        for k in keys:
            v = data[k]
            if isinstance(v, list) and all(isinstance(x, (int, float, type(None))) for x in v):
                # pick first plausible list as train
                train = train or v
    return {"train": train, "val": val, "acc": acc}

def aggregate_runs(runs_root: Path, task: str):
    """
    Busca runs bajo runs_root/task/run_seed*/results/metrics_epochs.json
    Devuelve arrays (n_runs, max_epochs) con np.nan para faltantes.
    """
    run_results = list((runs_root / task).glob("run_seed*/results/metrics_epochs.json"))
    # fallback: quizás trainer escribió metrics_epochs_tb.json en run_results
    if len(run_results) == 0:
        run_results = list((runs_root / task).glob("run_seed*/results/metrics_epochs_tb.json"))
    print(f"Encontrados {len(run_results)} archivos de metrics en {runs_root/task}")
    all_train = []
    all_val = []
    all_acc = []
    max_epochs = 0
    for p in run_results:
        m = load_metrics_from_run(p)
        if m is None:
            print("No pude cargar", p)
            continue
        train = m["train"] or []
        val = m["val"] or []
        acc = m["acc"] or []
        max_epochs = max(max_epochs, len(train), len(val), len(acc))
        all_train.append(train)
        all_val.append(val)
        all_acc.append(acc)
    # pad with np.nan to rectangular arrays
    def pad_list_of_lists(list_of_lists, maxlen):
        arr = np.full((len(list_of_lists), maxlen), np.nan, dtype=np.float32)
        for i, lst in enumerate(list_of_lists):
            arr[i, :len(lst)] = lst
        return arr
    train_arr = pad_list_of_lists(all_train, max_epochs) if all_train else np.empty((0,0))
    val_arr = pad_list_of_lists(all_val, max_epochs) if all_val else np.empty((0,0))
    acc_arr = pad_list_of_lists(all_acc, max_epochs) if all_acc else np.empty((0,0))
    return train_arr, val_arr, acc_arr

def plot_mean_std(x, arr, label, ax=None):
    """
    arr shape (n_runs, n_epochs) with np.nan for missing.
    Plotea media ignorando NaN y banda +/- std.
    """
    if arr.size == 0:
        return
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    epochs = np.arange(1, len(mean) + 1)
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(epochs, mean, label=label)
    ax.fill_between(epochs, mean - std, mean + std, alpha=0.25)
    ax.set_xlabel("Epoca")
    ax.set_xlim(1, len(mean))
    return ax

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="experiments/exp_mlp_s01", help="carpeta del experimento")
    parser.add_argument("--task", type=str, default="vowels", help="vowels o commands")
    parser.add_argument("--save", action="store_true", help="guardar figura en carpeta aggregated")
    args = parser.parse_args()

    exp_root = Path(args.exp).resolve()
    runs_root = exp_root / "runs"
    aggregated_dir = exp_root / "aggregated"
    aggregated_dir.mkdir(parents=True, exist_ok=True)

    train_arr, val_arr, acc_arr = aggregate_runs(runs_root, args.task)

    # plot losses
    fig, ax = plt.subplots(1,1, figsize=(8,4))
    plot_mean_std(None, train_arr, label="Train loss", ax=ax)
    plot_mean_std(None, val_arr, label="Val loss", ax=ax)
    ax.set_title(f"Loss (mean ± std) — {args.task}")
    ax.legend()
    plt.tight_layout()
    if args.save:
        fig.savefig(aggregated_dir / f"{args.task}_loss_mean_std.png", dpi=200)
    plt.show()

    # plot val acc
    if acc_arr.size != 0:
        fig2, ax2 = plt.subplots(1,1, figsize=(8,4))
        plot_mean_std(None, acc_arr, label="Val accuracy", ax=ax2)
        # graficar linea de puntos con probabilidad base
        if args.task == "vowels":
            ax2.axhline(y=1/5, color='black', linestyle='--', label='Prob. base (1/5)')
        elif args.task == "commands":
            ax2.axhline(y=1/6, color='black', linestyle='--', label='Prob. base (1/6)')
        ax2.set_title(f"Val accuracy (mean ± std) — {args.task}")
        ax2.legend()
        plt.tight_layout()
        if args.save:
            fig2.savefig(aggregated_dir / f"{args.task}_valacc_mean_std.png", dpi=200)
        plt.show()
    else:
        print("No hay val_acc en los metrics; quizá no se grabaron o la configuracion no calcula accuracy por epoca.")

if __name__ == "__main__":
    main()
