#!/usr/bin/env python3
"""
Genera dataset clean a 1024Hz (sin downsampling).

Extrae solo trials de habla imaginada (modalidad=1) sin artefactos de parpadeo (artifact=1)
desde los datos originales (1024Hz, 4096 samples/trial).
"""

import numpy as np
import json
from pathlib import Path

DATA_ORIGINAL = Path(__file__).resolve().parent.parent / "data" / "original"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "clean_1024hz"

N_SIGNAL_COLS = 6 * 4096  # 24576
N_CHANNELS = 6
SAMPLES_PER_TRIAL = 4096
FS = 1024


def load_and_filter_subject(subject_id: int) -> tuple:
    """Carga datos originales y filtra por modalidad=1 y artifact=1."""
    file_path = DATA_ORIGINAL / f"S{subject_id:02d}_EEG.npz"
    
    if not file_path.exists():
        raise FileNotFoundError(f"No encontrado: {file_path}")
    
    data = np.load(file_path, allow_pickle=True)["data"]
    n_trials = data.shape[0]
    
    signal_flat = data[:, :N_SIGNAL_COLS]
    modality = data[:, N_SIGNAL_COLS]
    stimulus = data[:, N_SIGNAL_COLS + 1]
    artifact = data[:, N_SIGNAL_COLS + 2]
    
    mask = (modality == 1) & (artifact == 1)
    n_filtered = mask.sum()
    
    X = signal_flat[mask].reshape(-1, N_CHANNELS, SAMPLES_PER_TRIAL)
    Y = np.column_stack([
        modality[mask],
        stimulus[mask],
        artifact[mask]
    ])
    
    return X, Y, n_trials, n_filtered


def process_all_subjects():
    """Procesa todos los sujeto S01-S15."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    subjects_data = []
    
    for subj_id in range(1, 16):
        subj_name = f"S{subj_id:02d}"
        print(f"Procesando {subj_name}...")
        
        X, Y, n_orig, n_clean = load_and_filter_subject(subj_id)
        
        output_path = OUTPUT_DIR / f"{subj_name}_clean_1024.npz"
        np.savez_compressed(output_path, x=X, y=Y)
        
        subjects_data.append({
            "id": subj_name,
            "trials_original": int(n_orig),
            "trials_clean": int(n_clean),
            "file": output_path.name
        })
        
        print(f"  {subj_name}: {n_orig} -> {n_clean} trials ({100*n_clean/n_orig:.1f}%)")
    
    total_orig = sum(s["trials_original"] for s in subjects_data)
    total_clean = sum(s["trials_clean"] for s in subjects_data)
    
    log = {
        "pipeline": "Filtered: only imagined speech + clean (no blink)",
        "description": "Dataset containing only trials with: (1) imagined speech (not pronounced) and (2) no blink artifacts. NO DOWNSAMPLING - keeps original 1024Hz",
        "source": "data/original/",
        "fs": FS,
        "shape": f"(trials, {N_CHANNELS}, {SAMPLES_PER_TRIAL})",
        "trials": {
            "total": total_clean,
            "note": f"Original: ~{total_orig} trials, Filtered: {total_clean} trials ({100*total_clean/total_orig:.1f}% retained)"
        },
        "labels": {
            "column_0_modalidad": {
                "description": "Speech modality",
                "values": {
                    "1": "Imaginada (only value in this dataset)"
                },
                "note": "Only imagined speech included - pronounced trials removed"
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
                    "1": "Limpio (no blink - only value in this dataset)"
                },
                "note": "Only clean trials included - blink trials removed"
            }
        },
        "channels": {
            "count": N_CHANNELS,
            "names": ["F3", "F4", "C3", "C4", "P3", "P4"],
            "layout": "International 10-20 system"
        },
        "time": {
            "duration_seconds": SAMPLES_PER_TRIAL / FS,
            "samples": SAMPLES_PER_TRIAL,
            "fs": FS
        },
        "subjects": subjects_data,
        "subjects_processed": [s["file"] for s in subjects_data],
        "data_format": {
            "x": f"(trials, {N_CHANNELS}, {SAMPLES_PER_TRIAL}) - EEG signals, raw (no normalization)",
            "y": "(trials, 3) - labels [modalidad, stim, artefact]"
        },
        "preprocessing_applied": "ONLY filtering: modality=1 & artifact=1. NO DOWNSAMPLING.",
        "note": "This dataset is a subset of original data at native 1024Hz, containing only imagined speech trials without blink artifacts for optimal model training"
    }
    
    log_path = OUTPUT_DIR / "process_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== COMPLETADO ===")
    print(f"Directorio: {OUTPUT_DIR}")
    print(f"Total trials: {total_clean}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    process_all_subjects()