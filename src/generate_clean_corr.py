#!/usr/bin/env python3
"""
Genera dataset donde cada trial es la matriz de covarianza espacial entre canales.

Fuente: data/clean_preprocessed/SXX_clean.npz (x: (trials, 6, 512) a 128 Hz)
Salida: data/clean_preprocessed_corr/SXX_clean.npz (x: (trials, 6, 6))

Por cada trial (C x T):
  1. Se resta la media por canal (centrar cada fila).
  2. Se multiplica la matriz centrada (C x T) por su transpuesta (T x C).
  3. Se divide todo por (T - 1).

Resultado: matriz (C x C) de covarianza espacial SIN normalizar por la
desviación estándar. Es simétrica, y sus valores NO están acotados (-inf a inf).

Las etiquetas (y) se conservan sin cambios.
"""

import numpy as np
import json
from pathlib import Path

INPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "clean_preprocessed"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "clean_preprocessed_corr"

N_CHANNELS = 6
SUBJ_IDS = [f"S{i:02d}" for i in range(1, 16)]


def trial_to_cov_matrix(trial: np.ndarray) -> np.ndarray:
    """
    Calcula la matriz de covarianza espacial (sin normalizar) para un trial.
    Input: (n_channels, n_timepoints) -> Output: (n_channels, n_channels).

    Pasos:
      1. Centrar por canal: Xc = trial - media(trial, axis=1)
      2. R = (Xc @ Xc.T) / (T - 1)
    Valores no acotados (-inf a inf).
    """
    Xc = trial - trial.mean(axis=1, keepdims=True)
    return (Xc @ Xc.T) / (trial.shape[1] - 1)


def process_subject(subj_id: str) -> dict:
    """Procesa un sujeto: convierte cada trial a matriz de covarianza espacial entre canales."""
    src_path = INPUT_DIR / f"{subj_id}_clean.npz"
    if not src_path.exists():
        raise FileNotFoundError(f"No encontrado: {src_path}")

    data = np.load(src_path, allow_pickle=True)
    X = data["x"]  # (trials, n_channels, n_timepoints)
    Y = data["y"]  # (trials, 3)

    n_trials, n_ch, n_tp = X.shape
    if n_ch != N_CHANNELS:
        raise ValueError(f"[{subj_id}] Esperaba {N_CHANNELS} canales, se encontraron {n_ch}")

    X_cov = np.empty((n_trials, n_ch, n_ch), dtype=float)
    for i in range(n_trials):
        X_cov[i] = trial_to_cov_matrix(X[i])

    out_path = OUTPUT_DIR / f"{subj_id}_clean.npz"
    np.savez_compressed(out_path, x=X_cov, y=Y)

    print(f"  {subj_id}: {n_trials} trials -> {X_cov.shape}")

    return {"id": subj_id, "trials": int(n_trials)}


def process_all_subjects():
    """Procesa todos los sujetos S01-S15 y genera process_log.json."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    subjects_data = []
    for subj_id in SUBJ_IDS:
        print(f"Procesando {subj_id}...")
        subjects_data.append(process_subject(subj_id))

    total_trials = sum(s["trials"] for s in subjects_data)

    log = {
        "pipeline": "Covarianza espacial por trial (centrada por canal, sin normalizar)",
        "description": "Cada trial EEG es reemplazado por su matriz de covarianza espacial entre canales. "
                       "Se resta la media por canal (Xc) y se computa R = (Xc @ Xc.T) / (T - 1). "
                       f"La matriz ({N_CHANNELS}x{N_CHANNELS}) es simétrica y sus valores NO están acotados (-inf a inf).",
        "source": "data/clean_preprocessed/",
        "fs": 128,
        "shape": f"(trials, {N_CHANNELS}, {N_CHANNELS})",
        "trials": {
            "total": total_trials
        },
        "labels": {
            "column_0_modalidad": {
                "description": "Modalidad del habla",
                "values": {
                    "1": "Imaginada (único valor en este dataset)"
                },
                "note": "Solo trials de habla imaginada - se eliminan los pronunciados"
            },
            "column_1_estimulo": {
                "description": "Estímulo",
                "values": {
                    "1": "A (vocal)",
                    "2": "E (vocal)",
                    "3": "I (vocal)",
                    "4": "O (vocal)",
                    "5": "U (vocal)",
                    "6": "Arriba (comando)",
                    "7": "Abajo (comando)",
                    "8": "Adelante (comando)",
                    "9": "Atras (comando)",
                    "10": "Derecha (comando)",
                    "11": "Izquierda (comando)"
                }
            },
            "column_2_artefacto": {
                "description": "Presencia de artefacto",
                "values": {
                    "1": "Limpio (sin parpadeo - único valor en este dataset)"
                },
                "note": "Solo trials limpios incluidos - se eliminan los de parpadeo"
            }
        },
        "channels": {
            "count": N_CHANNELS,
            "names": ["F3", "F4", "C3", "C4", "P3", "P4"],
            "layout": "Sistema internacional 10-20"
        },
        "time": {
            "duration_seconds": 4,
            "samples_per_trial": 512,
            "fs": 128,
            "note": "El trial completo se usa para estimar la covarianza (sin ventaneo)"
        },
        "covariance": {
            "type": "espacial (sin normalizar por desviación estándar)",
            "formula": "R = (Xc @ Xc.T) / (T - 1)",
            "centering": "se resta la media por canal (axis=1) antes del producto",
            "estimated_over": "tiempo (todas las muestras del trial)",
            "per_trial_matrix": f"({N_CHANNELS}x{N_CHANNELS}) simétrica",
            "diagonal": "varianza muestral por canal (suma de cuadrados / (T-1))",
            "range": "(-inf, inf) - valores no acotados",
            "note": "No se divide por la desviación estándar a diferencia de la correlación de Pearson"
        },
        "subjects": subjects_data,
        "subjects_processed": [f"{s['id']}_clean.npz" for s in subjects_data],
        "data_format": {
            "x": f"(trials, {N_CHANNELS}, {N_CHANNELS}) - matriz de covarianza espacial entre canales (sin normalizar)",
            "y": "(trials, 3) - etiquetas [modalidad, estimulo, artefacto] (sin cambios)"
        },
        "preprocessing_applied": "ninguno adicional: solo se sustituye cada trial (6, 512) por su matriz de covarianza espacial (6, 6)",
        "note": "Versión de data/clean_preprocessed donde cada trial es la matriz de covarianza espacial entre canales "
                "(centrada por canal, sin normalizar por std), útil como entrada de conectividad espacial para clasificación de habla imaginada"
    }

    log_path = OUTPUT_DIR / "process_log.json"
    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2, ensure_ascii=False)

    print(f"\n=== COMPLETADO ===")
    print(f"Directorio: {OUTPUT_DIR}")
    print(f"Total trials: {total_trials}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    process_all_subjects()