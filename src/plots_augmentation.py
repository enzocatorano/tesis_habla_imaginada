# Python code to generate example figures of augmentations.
# Saves figures into data/processed_aug/figures_examples (or creates synthetic demo if no npz found).
#
# Usage: save this as a script (por ejemplo: src/plot_augment_examples.py) and ejecútalo
# desde el root del proyecto: python src/plot_augment_examples.py
#
# Requisitos: numpy, scipy, matplotlib
# Colores: dodgerblue para señales en tiempo, orange para magnitudes, red para fases.

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack

# graficos oscuros
plt.style.use('dark_background')

# --- configuración robusta de rutas ---
BASE_DIR = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
NO_AUG_DIR = BASE_DIR / "data" / "preproc"
AUG_DIR = BASE_DIR / "data" / "preproc_aug_segm_gnperband_fts"
FIG_DIR = AUG_DIR / "figures_examples"
FIG_DIR.mkdir(parents=True, exist_ok=True)

FS = 128  # frecuencia de muestreo esperada para las señales resampleadas
COL_TIME = "dodgerblue"
COL_MAG = "orange"
COL_PHASE = "red"

# azares
sujeto = 1
trial = 0
canal = 0

###############################################################################################

def cargar_datos(sujeto = 1):
    # intenta cargar un archivo real
    try:
        filepath = NO_AUG_DIR / f"S{str(sujeto).zfill(2)}_EEG_augmented.npz"
        data = np.load(filepath)
        x_orig = data['data']
        y_orig = data['labels']
        print(f"Cargando datos de {filepath}")
        return x_orig, y_orig
    except FileNotFoundError:
        print(f"No se encontró {filepath}.")

def cargar_datos_aug(sujeto = 1):
    # intenta cargar un archivo real aumentado
    try:
        filepath_aug = AUG_DIR / f"S{str(sujeto).zfill(2)}_EEG_augmented.npz"
        data_aug = np.load(filepath_aug)
        x_aug = data_aug['data']
        y_aug = data_aug['labels']
        nombres_bandas = {'delta': (0.5, 4.0),
                          'theta': (4.0, 8.0),
                          'alpha': (8.0, 12.0),
                          'beta' : (12.0, 32.0),
                          'gamma': (32.0, 63.5)}
        print(f"Cargando datos de {filepath_aug}")
        return x_aug, y_aug, nombres_bandas
    except FileNotFoundError:
        print(f"No se encontró {filepath_aug}.")



def plot_segmentaciones(x_orig, y_orig, x_aug, y_aug, nombres_bandas):
    # un grafico de 7 filas, siendo la primera el trial original
    # y las siguientes 6 las augmentaciones
    tiempo_original = np.linspace(0, x_orig.shape[2] / FS, x_orig.shape[2])
    tiempo_segmentado = np.linspace(0, x_aug.shape[2] / FS, x_aug.shape[2])

    fig, ax = plt.subplots(7, 1, figsize=(10, 12))

    # Señal original
    ax[0].plot(tiempo_original, x_orig[trial, canal, :], color=COL_TIME)
    ax[0].set_title(f'Sujeto {sujeto}, Trial {trial}, Canal {canal} - Original')
    ax[0].set_ylabel('Amplitud (uV)')
    ax[0].set_xticks([])
    limites_eje_y = ax[0].get_ylim() # todos los graficos tendran el mismo limite

    # Augmentaciones (las primeras 6 ventanas del trial original)
    for i in range(6):
        ax[i+1].plot(tiempo_segmentado, x_aug[i, canal, :], color=COL_TIME)
        ax[i+1].set_title(f'Ventana {i} (Augmentada)')
        ax[i+1].set_ylabel('Amplitud (uV)')
        ax[i+1].set_xticks([])
        ax[i+1].set_ylim(limites_eje_y)

    ax[-1].set_xlabel('Tiempo (s)')
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"S{sujeto}_T{trial}_C{canal}_segmentaciones.png")
    plt.close()



def plot_matriz_ruido_bandas(x_aug, y_aug, nombres_bandas):
    # las filas son las bandas
    # las columnas son los canales
    # la primer fila tiene la señal original, sin ruido
    # recordar que los datos estan en grupos de 4 consecutivos
    # para una misma banda afectada (sin FTS + 3 FTS)
    fig, ax = plt.subplots(len(nombres_bandas) + 1, x_aug.shape[1], figsize=(15, 10))
    
    # Señal original (sin ruido)
    for ch in range(x_aug.shape[1]):
        ax[0, ch].plot(x_aug[trial, ch, :], color=COL_TIME)
        ax[0, ch].set_title(f'Canal {ch} - Original')
        ax[0, ch].set_xticks([])
        ax[0, ch].set_yticks([])
        aux = ax[0, ch].get_ylim() # todos los graficos tendran el mismo limite
        # multiplico ambos limites por 1.1
        limites_eje_y = (aux[0]*1.1, aux[1]*1.1)
        ax[0, ch].set_ylim(limites_eje_y[0], limites_eje_y[1])
    ax[0, 0].set_ylabel('Original')

    # Señales con ruido por banda
    for i, (bname, (low, high)) in enumerate(nombres_bandas.items()):
        for ch in range(x_aug.shape[1]):
            ax[i+1, ch].plot(x_aug[trial + 4 * i, ch, :], color=COL_TIME)
            ax[i+1, ch].set_xticks([])
            ax[i+1, ch].set_yticks([])
            ax[i+1, ch].set_ylim(limites_eje_y[0], limites_eje_y[1])
        ax[i+1, 0].set_ylabel(f'{bname.capitalize()} Noise')

    plt.tight_layout()
    plt.savefig(FIG_DIR / f"S{sujeto}_T{trial}_band_noise_matrix.png")
    plt.close()


def plot_fft_bandnoise(x_aug, y_aug, nombres_bandas):
    # una figura de 4 columnas y bandas + 1 renglones
    # el primer renglon es la señal original
    # los demas renglones son cada banda ensuciada
    # la primer columna es la señal en cuestion
    # la segunda su fft
    # la tercera es la señal - señal original
    # la cuarta es la fft de la rest
    tiempo = np.linspace(0, x_aug.shape[2] / FS, x_aug.shape[2])

    fig, ax = plt.subplots(len(nombres_bandas) + 1, 4, figsize=(16, 12))
    
    # Señal original (sin ruido)
    signal_orig = x_aug[trial, canal, :]
    fft_orig = fftpack.fft(signal_orig)
    freqs_orig = fftpack.fftfreq(len(signal_orig), 1/FS)
    
    ax[0, 0].plot(tiempo, signal_orig, color=COL_TIME)
    ax[0, 0].set_title('Original (Tiempo)')
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    aux = ax[0, 0].get_ylim() # todos los graficos tendran el mismo limite
    # multiplico ambos limites por 1.1
    limites_eje_y = (aux[0]*1.1, aux[1]*1.1)
    ax[0, 0].set_ylim(limites_eje_y[0], limites_eje_y[1])
    
    ax[0, 1].plot(freqs_orig, np.abs(fft_orig), color=COL_MAG)
    ax[0, 1].set_title('Original (FFT Mag)')
    ax[0, 1].set_xlim(0, FS/2)
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    aux = ax[0, 1].get_ylim() # todos los graficos tendran el mismo limite
    # multiplico ambos limites por 1.1
    limites_eje_y_fft = (aux[0]*1.1, aux[1]*1.1)
    ax[0, 1].set_ylim(limites_eje_y_fft[0], limites_eje_y_fft[1])

    # Las columnas 2 y 3 no tienen sentido para la señal original
    ax[0, 2].axis('off')
    ax[0, 3].axis('off')
    
    # Señales con ruido por banda
    for i, (bname, (low, high)) in enumerate(nombres_bandas.items()):
        # Tomamos la primera augmentación de cada banda (sin FT surrogate)
        signal_aug = x_aug[trial + 4 * i, canal, :]
        fft_aug = fftpack.fft(signal_aug)
        freqs_aug = fftpack.fftfreq(len(signal_aug), 1/FS)

        # Columna 0: Señal en tiempo
        ax[i+1, 0].plot(tiempo, signal_aug, color=COL_TIME)
        ax[i+1, 0].set_title(f'{bname.capitalize()} (Tiempo)')
        ax[i+1, 0].set_xticks([])
        ax[i+1, 0].set_yticks([])
        ax[i+1, 0].set_ylim(limites_eje_y[0], limites_eje_y[1])

        # Columna 1: FFT de la señal
        ax[i+1, 1].plot(freqs_aug, np.abs(fft_aug), color=COL_MAG)
        ax[i+1, 1].set_title(f'{bname.capitalize()} (FFT Mag)')
        ax[i+1, 1].set_xlim(0, FS/2)
        ax[i+1, 1].set_xticks([])
        ax[i+1, 1].set_yticks([])
        ax[i+1, 1].set_ylim(limites_eje_y_fft[0], limites_eje_y_fft[1])
        ax[i+1, 1].axvspan(low, high, color='gray', alpha=0.3) # Resaltar la banda

        # Columna 2: Diferencia en tiempo (ruido añadido)
        diff_signal = signal_aug - signal_orig
        ax[i+1, 2].plot(tiempo, diff_signal, color=COL_TIME)
        ax[i+1, 2].set_title(f'Ruido {bname.capitalize()} (Tiempo)')
        ax[i+1, 2].set_xticks([])
        ax[i+1, 2].set_yticks([])
        ax[i+1, 2].set_ylim(limites_eje_y[0], limites_eje_y[1])

        # Columna 3: FFT de la diferencia
        fft_diff = fftpack.fft(diff_signal)
        freqs_diff = fftpack.fftfreq(len(diff_signal), 1/FS)
        ax[i+1, 3].plot(freqs_diff, np.abs(fft_diff), color=COL_MAG)
        ax[i+1, 3].set_title(f'Ruido {bname.capitalize()} (FFT Mag)')
        ax[i+1, 3].set_xlim(0, FS/2)
        ax[i+1, 3].set_xticks([])
        ax[i+1, 3].set_yticks([])
        ax[i+1, 3].set_ylim(limites_eje_y_fft[0], limites_eje_y_fft[1])
        ax[i+1, 3].axvspan(low, high, color='gray', alpha=0.3) # Resaltar la banda

    ax[-1, 0].set_xlabel('Tiempo (s)')
    ax[-1, 1].set_xlabel('Frecuencia (Hz)')
    ax[-1, 2].set_xlabel('Tiempo (s)')
    ax[-1, 3].set_xlabel('Frecuencia (Hz)')
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"S{sujeto}_T{trial}_C{canal}_fft_bandnoise.png")
    plt.close()


def plot_ft_surrogates(x_aug, y_aug, nombres_bandas):
    # una figura de 3 columnas y 4 renglones
    # columna izquierda es señal temporal
    # columna del medio es magnitud fft
    # columna derecha es fase
    # primer fila es sin FTS
    # las otras 3 es con FTS
    tiempo = np.linspace(0, x_aug.shape[2] / FS, x_aug.shape[2])

    fig, ax = plt.subplots(4, 3, figsize=(12, 10))
    
    # Tomamos la primera augmentación de la primera banda (delta)
    # Los surrogates de FT son los 3 siguientes
    
    # Señal original (sin FT surrogate)
    signal_orig = x_aug[trial, canal, :]
    fft_orig = fftpack.fft(signal_orig)
    freqs = fftpack.fftfreq(len(signal_orig), 1/FS)
    phases_orig = np.angle(fft_orig)

    ax[0, 0].plot(tiempo, signal_orig, color=COL_TIME)
    ax[0, 0].set_title('Original (Tiempo)')
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    limites_eje_y = ax[0, 0].get_ylim()
    limites_eje_y = (limites_eje_y[0] * 1.1, limites_eje_y[1] * 1.1)
    ax[0, 0].set_ylim(limites_eje_y)
    
    ax[0, 1].plot(freqs, np.abs(fft_orig), color=COL_MAG)
    ax[0, 1].set_title('Original (FFT Mag)')
    ax[0, 1].set_xlim(0, FS/2)
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    limites_eje_y_fft = ax[0, 1].get_ylim()
    limites_eje_y_fft = (limites_eje_y_fft[0] * 1.1, limites_eje_y_fft[1] * 1.1)
    ax[0, 1].set_ylim(limites_eje_y_fft)

    ax[0, 2].plot(freqs, phases_orig, color=COL_PHASE)
    ax[0, 2].set_title('Original (FFT Fase)')
    ax[0, 2].set_xlim(0, FS/2)
    ax[0, 2].set_xticks([])
    ax[0, 2].set_yticks([])
    ax[0, 2].set_ylim(-np.pi, np.pi) # Fases van de -pi a pi
    
    # FT Surrogates
    for i in range(3):
        # Los surrogates están en los índices trial+1, trial+2, trial+3
        signal_fts = x_aug[trial + 1+ i, canal, :]
        fft_fts = fftpack.fft(signal_fts)
        phases_fts = np.angle(fft_fts)

        ax[i+1, 0].plot(tiempo, signal_fts, color=COL_TIME)
        ax[i+1, 0].set_title(f'FTS {i+1} (Tiempo)')
        ax[i+1, 0].set_xticks([])
        ax[i+1, 0].set_yticks([])
        ax[i+1, 0].set_ylim(limites_eje_y)

        ax[i+1, 1].plot(freqs, np.abs(fft_fts), color=COL_MAG)
        ax[i+1, 1].set_title(f'FTS {i+1} (FFT Mag)')
        ax[i+1, 1].set_xlim(0, FS/2)
        ax[i+1, 1].set_xticks([])
        ax[i+1, 1].set_yticks([])
        ax[i+1, 1].set_ylim(limites_eje_y_fft)

        ax[i+1, 2].plot(freqs, phases_fts, color=COL_PHASE)
        ax[i+1, 2].set_title(f'FTS {i+1} (FFT Fase)')
        ax[i+1, 2].set_xlim(0, FS/2)
        ax[i+1, 2].set_xticks([])
        ax[i+1, 2].set_yticks([])
        ax[i+1, 2].set_ylim(-np.pi, np.pi)

    ax[-1, 0].set_xlabel('Tiempo (s)')
    ax[-1, 1].set_xlabel('Frecuencia (Hz)')
    ax[-1, 2].set_xlabel('Frecuencia (Hz)')
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"S{sujeto}_T{trial}_C{canal}_ft_surrogates.png")
    plt.close()


if __name__ == "__main__":
    x_orig, y_orig = cargar_datos(sujeto)
    x_aug, y_aug, nombres_bandas = cargar_datos_aug(sujeto)
    if x_orig is not None and x_aug is not None:
        print("Generando gráficos de ejemplo...")
        plot_segmentaciones(x_orig, y_orig, x_aug, y_aug, nombres_bandas)
        plot_matriz_ruido_bandas(x_aug, y_aug, nombres_bandas)
        plot_fft_bandnoise(x_aug, y_aug, nombres_bandas)
        plot_ft_surrogates(x_aug, y_aug, nombres_bandas)
        print(f"Gráficos guardados en {FIG_DIR}")
    else:
        print("No se pudieron cargar los datos. Asegúrate de que los archivos .npz existan.")