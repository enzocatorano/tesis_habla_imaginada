#!/usr/bin/env python3
"""
preprocess_cooney_interactivo.py

Versión interactiva del preprocesamiento descrito en Cooney et al., Sensors 2020.

Características:
- Filtrado FIR 2-40 Hz
- ICA (FastICA) sobre señal concatenada (downsample por factor 3 para ICA)
- Plantilla de blink desde S01 (componente con mayor energía frontal)
- Para cada sujeto: muestra figura con series temporales de ICs + topografías + plantilla + suma F3+F4 (downsampled).
  Tras cerrar la figura, te pregunta por consola qué componentes eliminar (recomendadas: corr abs >= 0.8).
- Aplica la remoción indicada, reconstruye y remuestrea DIRECTO a 128 Hz (no vuelve a 1024Hz).
- Guarda .npz por sujeto con datos preprocesados y un JSON meta por sujeto y un JSON resumen global.

No solicita parámetros por consola (todo hardcoded).
"""

import os
import json
import numpy as np
from glob import glob
from scipy import signal
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt

# -------------------------
# PARÁMETROS (explícitos)
# -------------------------
INPUT_DIR  = "data/original"
OUTPUT_DIR = "data/preprocesamiento_segun_Cooney_interactivo"

ORIG_FS   = 1024.0
FINAL_FS  = 128.0
TRIAL_SEC = 4.0
N_CHANNELS = 6
CHANNEL_NAMES = ["F3", "F4", "C3", "C4", "P3", "P4"]

SAMPLES_PER_TRIAL = int(ORIG_FS * TRIAL_SEC)    # 4096
FLAT_EEG_LEN = SAMPLES_PER_TRIAL * N_CHANNELS   # 24576
LABELS_PER_TRIAL = 3
FLAT_ROW_LEN = FLAT_EEG_LEN + LABELS_PER_TRIAL  # 24579

BP_LOW = 2.0
BP_HIGH = 40.0
FIR_TAPS = 801

ICA_DOWNSAMPLE_FACTOR = 3
ICA_CORR_THRESHOLD = 0.8
ICA_RANDOM_STATE = 42
ICA_MAX_ITER = 2000

AUTO_REMOVE_DEFAULT = False  # no usado: cada sujeto se pregunta manualmente

# visual
POSICION = 1000
EXTENSION = 5000  # parte de la serie para mostrar (asegurá que <= n_samples_ds)

# -------------------------
# HELPERS
# -------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_main_array(npz_path):
    with np.load(npz_path, allow_pickle=True) as z:
        key = max(z.files, key=lambda k: z[k].size)
        return z[key]

def reshape_trials(flat):
    arr = np.asarray(flat)
    if arr.ndim == 1:
        if arr.size % FLAT_ROW_LEN != 0:
            raise ValueError(f"El archivo {npz_path} no tiene longitud compatible.")
        arr = arr.reshape(-1, FLAT_ROW_LEN)
    n_trials, rowlen = arr.shape
    if rowlen not in (FLAT_ROW_LEN, FLAT_EEG_LEN):
        raise ValueError(f"Formato inesperado: filas de longitud {rowlen}.")
    eeg_part = arr[:, :FLAT_EEG_LEN]
    labels = arr[:, FLAT_EEG_LEN:FLAT_ROW_LEN] if rowlen == FLAT_ROW_LEN else np.zeros((n_trials,0))
    eeg = eeg_part.reshape(n_trials, N_CHANNELS, SAMPLES_PER_TRIAL)
    eeg = np.transpose(eeg, (0, 2, 1))  # (n_trials, samples, ch)
    return eeg, labels

def design_fir_bandpass(low, high, fs, numtaps=FIR_TAPS):
    return signal.firwin(numtaps, [low, high], pass_zero=False, fs=fs)

def filter_trials(eeg_trials, low, high, fs):
    fir = design_fir_bandpass(low, high, fs)
    n_trials, n_samples, n_ch = eeg_trials.shape
    out = np.zeros_like(eeg_trials)
    for tr in range(n_trials):
        for ch in range(n_ch):
            out[tr, :, ch] = signal.filtfilt(fir, 1.0, eeg_trials[tr, :, ch])
    return out

def concat_trials_for_ica(eeg_trials):
    return eeg_trials.reshape(-1, N_CHANNELS)

def run_ica_on_subject(eeg_filt):
    concat = concat_trials_for_ica(eeg_filt)  # (n_trials*n_samples, n_ch)
    concat_ds = signal.resample_poly(concat, up=1, down=ICA_DOWNSAMPLE_FACTOR, axis=0)
    ica = FastICA(n_components=N_CHANNELS, random_state=ICA_RANDOM_STATE, max_iter=ICA_MAX_ITER)
    S_ds = ica.fit_transform(concat_ds)  # (n_samples_ds, n_comp)
    mixing = ica.mixing_  # (n_ch, n_comp)
    return {
        "S_ds": S_ds,
        "mixing": mixing,
        "concat_ds": concat_ds
    }

def normalize_rows(arr):
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    return arr / norms

def reconstruct_and_resample_to_final(S_ds_clean, mixing, n_trials):
    # reconstruir en domain downsampled
    recon_ds = S_ds_clean @ mixing.T  # shape (n_samples_ds, n_ch)
    expected_total = int(n_trials * FINAL_FS * TRIAL_SEC)  # n_trials * 512
    # usamos signal.resample para obtener exactamente expected_total muestras
    recon = signal.resample(recon_ds, expected_total, axis=0)
    recon_trials = recon.reshape(n_trials, int(FINAL_FS * TRIAL_SEC), N_CHANNELS)
    return recon_trials

# -------------------------
# MAIN
# -------------------------
def main():
    ensure_dir(OUTPUT_DIR)
    files = sorted(glob(os.path.join(INPUT_DIR, "S*_EEG.npz")))
    if len(files) == 0:
        raise FileNotFoundError(f"No se encontraron archivos {os.path.join(INPUT_DIR,'S*_EEG.npz')}")
    print(f"[+] Encontrados {len(files)} sujetos en {INPUT_DIR}. Salida: {OUTPUT_DIR}")

    # --- construir plantilla desde S01 (primer sujeto) ---
    s01_path = files[0]
    print(f"[+] Construyendo plantilla de parpadeo desde {os.path.basename(s01_path)}")
    arr_s01 = load_main_array(s01_path)
    eeg_s01, _ = reshape_trials(arr_s01)
    eeg_s01_filt = filter_trials(eeg_s01, BP_LOW, BP_HIGH, ORIG_FS)
    ica_s01 = run_ica_on_subject(eeg_s01_filt)
    S01_S = ica_s01["S_ds"]
    S01_mix = ica_s01["mixing"]
    topos_s01 = normalize_rows(S01_mix.T)  # (n_components, n_ch)
    frontal_idx = [0,1]  # F3, F4
    frontal_power = np.sum(np.abs(topos_s01[:, frontal_idx]), axis=1)
    template_idx = int(np.argmax(frontal_power))
    template_topo = topos_s01[template_idx]
    print(f"    -> plantilla: componente {template_idx} (mayor energia frontal).")

    # guardar plantilla
    ensure_dir(OUTPUT_DIR)
    with open(os.path.join(OUTPUT_DIR, "ica_blink_template.json"), "w") as f:
        json.dump({
            "template_subject": os.path.basename(s01_path),
            "template_component_idx": int(template_idx),
            "template_topo": template_topo.tolist()
        }, f, indent=2)

    summary = {}

    # --- procesar cada sujeto con interacción ---
    for subj_path in files:
        subj_name = os.path.basename(subj_path).replace(".npz","")
        print("\n" + "="*60)
        print(f"[+] Procesando {subj_name}")
        subj_outdir = os.path.join(OUTPUT_DIR, subj_name)
        ensure_dir(subj_outdir)

        arr = load_main_array(subj_path)
        eeg_raw, labels = reshape_trials(arr)
        n_trials = eeg_raw.shape[0]
        print(f"    - Trials: {n_trials}, forma raw {eeg_raw.shape}")

        #Filtro
        eeg_filt = filter_trials(eeg_raw, BP_LOW, BP_HIGH, ORIG_FS)

        # ICA
        ica_res = run_ica_on_subject(eeg_filt)
        S_ds = ica_res["S_ds"]
        mixing = ica_res["mixing"]
        concat_ds = ica_res["concat_ds"]
        topos = normalize_rows(mixing.T)  # (n_comp, n_ch)

        # correlaciones topográficas con la plantilla
        corrs = np.array([np.corrcoef(template_topo, topos[i])[0,1] for i in range(N_CHANNELS)])
        candidates = np.where(np.abs(corrs) >= ICA_CORR_THRESHOLD)[0].tolist()

        # preparar datos para plot
        S_plot = S_ds  # (n_samples_ds, n_comp)
        # asegurar rango para POSICION/EXTENSION no exceda
        n_ds_samples = S_plot.shape[0]
        pos = min(POSICION, max(0, n_ds_samples-1))
        ext = min(EXTENSION, n_ds_samples)
        if pos >= ext:
            pos = 0
            ext = min(4000, n_ds_samples)

        # dibujar figura (N_CHANNELS rows + 2, 2 cols)
        fig, axs = plt.subplots(N_CHANNELS+2, 2, figsize=(16, 9), gridspec_kw={'width_ratios':[5,1]})
        for i in range(N_CHANNELS):
            # serie temporal: mostrar S_ds (component i)
            axs[i,0].plot(S_plot[pos:ext, i])
            axs[i,0].set_title(f'Componente {i} - Serie Temporal')
            axs[i,0].set_ylabel('Amplitud')
            axs[i,0].set_xticks([])
            # topografía
            axs[i,1].bar(range(N_CHANNELS), topos[i])
            axs[i,1].set_title(f'Componente {i} - Topografía')
            axs[i,1].set_ylim(-1,1)
            axs[i,1].set_xticks(range(N_CHANNELS))
            axs[i,1].set_xticklabels(CHANNEL_NAMES, rotation=45)
            # resaltar candidatos
            if i in candidates:
                axs[i,0].patch.set_facecolor('red')
                axs[i,0].patch.set_alpha(0.15)
                axs[i,1].patch.set_facecolor('red')
                axs[i,1].patch.set_alpha(0.15)

        # fila N_CHANNELS: plantilla (serie temporal de la componente plantilla en S01, y topografía)
        # Tomamos la serie temporal de S01_S para la componente template_idx (puede ser más larga)
        template_series = S01_S[:, template_idx]
        axs[N_CHANNELS, 0].plot(template_series[pos:ext] if pos < template_series.size else template_series[:ext-pos])
        axs[N_CHANNELS, 0].set_title('S01 - Plantilla (serie temporal) - posible blink')
        axs[N_CHANNELS, 0].set_ylabel('Amplitud')
        axs[N_CHANNELS, 0].set_xticks([])
        axs[N_CHANNELS, 1].bar(range(N_CHANNELS), template_topo)
        axs[N_CHANNELS, 1].set_title('S01 - Plantilla (topografía normalizada)')
        axs[N_CHANNELS, 1].set_ylim(-1,1)
        axs[N_CHANNELS, 1].set_xticks(range(N_CHANNELS))
        axs[N_CHANNELS, 1].set_xticklabels(CHANNEL_NAMES, rotation=45)

        # fila N_CHANNELS+1: suma F3+F4 downsampled de concat_ds (si existe)
        frontal_sum = concat_ds[:, frontal_idx[0]] + concat_ds[:, frontal_idx[1]]
        axs[N_CHANNELS+1, 0].plot(frontal_sum[pos:ext])
        axs[N_CHANNELS+1, 0].set_title(f'{subj_name} - Suma F3+F4 (downsampled)')
        axs[N_CHANNELS+1, 0].set_ylabel('Amplitud')
        axs[N_CHANNELS+1, 0].set_xlabel('Muestras')
        axs[N_CHANNELS+1, 1].axis('off')

        plt.tight_layout()
        plt.show()  # bloqueante hasta cerrar la figura

        # Tras cerrar la figura, preguntar por consola
        print("\nLas correlaciones topográficas con la plantilla (por componente) son:")
        for i, c in enumerate(corrs):
            mark = "<-- candidato" if i in candidates else ""
            print(f"  Comp {i}: corr = {c:.3f} {mark}")
        print(f"\nCandidatas automáticas (|corr| >= {ICA_CORR_THRESHOLD}): {candidates}")

        user_in = input("\nQue componentes eliminamos? (ej: 0,3,5) - escribir 'all' para eliminar candidatas - Enter = ninguna\n> ").strip()
        to_remove = []
        if user_in.lower() in ("all","a"):
            to_remove = candidates.copy()
        elif user_in == "":
            to_remove = []
        else:
            # parsear lista
            try:
                parts = [p.strip() for p in user_in.split(",") if p.strip() != ""]
                to_remove = sorted(list({int(p) for p in parts}))
                # validar
                to_remove = [int(x) for x in to_remove if 0 <= int(x) < N_CHANNELS]
            except Exception as e:
                print("Entrada no válida. No se eliminará nada.")
                to_remove = []

        print(f"Componentes a eliminar para {subj_name}: {to_remove}")

        # aplicar remoción poniendo a cero las fuentes en S_ds
        S_ds_clean = S_ds.copy()
        if len(to_remove) > 0:
            S_ds_clean[:, to_remove] = 0.0

        # reconstruir y remuestrear directo a FINAL_FS
        eeg_final = reconstruct_and_resample_to_final(S_ds_clean, mixing, n_trials)

        # guardar .npz y meta
        out_npz = os.path.join(subj_outdir, subj_name + ".npz")
        np.savez_compressed(out_npz, data=eeg_final, labels=labels)

        meta = {
            "subject": subj_name,
            "n_trials": int(n_trials),
            "orig_fs": ORIG_FS,
            "ica_fs": float(ORIG_FS / ICA_DOWNSAMPLE_FACTOR),
            "final_fs": FINAL_FS,
            "bandpass": [BP_LOW, BP_HIGH],
            "ica_downsample_factor": ICA_DOWNSAMPLE_FACTOR,
            "ica_corr_threshold": ICA_CORR_THRESHOLD,
            "corrs": corrs.tolist(),
            "candidates_auto": candidates,
            "removed_by_user": to_remove
        }
        with open(os.path.join(subj_outdir, subj_name + "_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # guardar figura como PNG también para registro
        try:
            fig.savefig(os.path.join(subj_outdir, f"{subj_name}_ica_review.png"), dpi=150)
        except Exception:
            pass
        plt.close(fig)

        summary[subj_name] = meta
        print(f"[+] {subj_name} procesado y guardado en {subj_outdir}")

    # guardar resumen global
    with open(os.path.join(OUTPUT_DIR, "metadatos_preprocesamiento_interactivo.json"), "w") as f:
        json.dump({
            "fs_original": ORIG_FS,
            "fs_final": FINAL_FS,
            "bandpass": [BP_LOW, BP_HIGH],
            "ica_downsample_factor": ICA_DOWNSAMPLE_FACTOR,
            "ica_corr_threshold": ICA_CORR_THRESHOLD,
            "subjects": summary
        }, f, indent=2)

    print("\n✔ Preprocesamiento interactivo completado para todos los sujetos.")

if __name__ == "__main__":
    main()
