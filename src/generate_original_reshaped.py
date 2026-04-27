#!/usr/bin/env python3
"""
Reshape datos originales a formato (trial, canal, bins).

Mantiene todos los trials (sin filtrar), solo reordena de flatten a (6, 4096).
"""

import numpy as np
import json
from pathlib import Path

DATA_ORIGINAL = Path(__file__).resolve().parent.parent / "data" / "original"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "original_reshaped"

N_SIGNAL_COLS = 6 * 4096  # 24576
N_CHANNELS = 6
SAMPLES_PER_TRIAL = 4096


def reshape_subject(subject_id: int):
    """Carga datos originales y los reshapea."""
    file_path = DATA_ORIGINAL / f"S{subject_id:02d}_EEG.npz"
    
    if not file_path.exists():
        raise FileNotFoundError(f"No encontrado: {file_path}")
    
    data = np.load(file_path, allow_pickle=True)["data"]
    n_trials = data.shape[0]
    
    signal_flat = data[:, :N_SIGNAL_COLS]
    X = signal_flat.reshape(-1, N_CHANNELS, SAMPLES_PER_TRIAL)
    
    Y = data[:, N_SIGNAL_COLS:N_SIGNAL_COLS + 3]
    
    return X, Y, n_trials


def process_all_subjects():
    """Procesa todos los sujeto S01-S15."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    subjects_data = []
    
    for subj_id in range(1, 16):
        subj_name = f"S{subj_id:02d}"
        print(f"Procesando {subj_name}...")
        
        X, Y, n_trials = reshape_subject(subj_id)
        
        output_path = OUTPUT_DIR / f"S{subj_id:02d}_EEG.npz"
        np.savez_compressed(output_path, x=X, y=Y)
        
        subjects_data.append({
            "id": subj_name,
            "trials": int(n_trials),
            "file": output_path.name
        })
        
        print(f"  {subj_name}: {n_trials} trials, X shape={X.shape}, Y shape={Y.shape}")
    
    total_trials = sum(s["trials"] for s in subjects_data)
    
    log = {
        "pipeline": "Reshape only - no filtering",
        "description": "Original data reshaped from (trials, 24579) flat to (trials, 6, 4096). ALL trials preserved.",
        "source": "data/original/",
        "fs": 1024,
        "shape": f"(trials, {N_CHANNELS}, {SAMPLES_PER_TRIAL})",
        "trials": {
            "total": total_trials,
            "note": "All trials included - no filtering by modality or artifact"
        },
        "labels": {
            "column_0_modalidad": {
                "description": "Speech modality",
                "values": {
                    "1": "Imaginada",
                    "2": "Pronunciada"
                }
            },
            "column_1_estimulo": {
                "description": "Stimulus/cue",
                "values": {
                    "1": "A (vowel)",
                    "2": "E (vowel)",
                    "3": "I (vowel)",
                    "4": "O (vowel)",
                    "5": "U (vowel)",
                    "6": "Arriba (command)",
                    "7": "Abajo (command)",
                    "8": "Adelante (command)",
                    "9": "Atras (command)",
                    "10": "Derecha (command)",
                    "11": "Izquierda (command)"
                }
            },
            "column_2_artefacto": {
                "description": "Artifact presence",
                "values": {
                    "1": "Limpio (no blink)",
                    "2": "Blink (artifact)"
                }
            }
        },
        "channels": {
            "count": N_CHANNELS,
            "names": ["F3", "F4", "C3", "C4", "P3", "P4"],
            "layout": "International 10-20 system"
        },
        "time": {
            "duration_seconds": SAMPLES_PER_TRIAL / 1024,
            "samples": SAMPLES_PER_TRIAL,
            "fs": 1024
        },
        "subjects": subjects_data,
        "subjects_processed": [s["file"] for s in subjects_data],
        "data_format": {
            "x": f"(trials, {N_CHANNELS}, {SAMPLES_PER_TRIAL}) - EEG signals, raw",
            "y": "(trials, 3) - labels [modalidad, stim, artefact]"
        },
        "preprocessing_applied": "RESHAPE ONLY: from (trials, 24579) flat to (trials, 6, 4096). No filtering.",
        "note": "This is the original data reshaped, preserving all trials and all labels"
    }
    
    log_path = OUTPUT_DIR / "process_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== COMPLETADO ===")
    print(f"Directorio: {OUTPUT_DIR}")
    print(f"Total trials: {total_trials}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    process_all_subjects()