"""
pruebas_datos_repositorio.py
===========================
Script para visualizar datos del repo de Liwicki.
Ejecutar: python src/like_liwicki/pruebas_datos_repositorio.py
"""

import numpy as np
from scipy.io import loadmat
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

SUJETO = 1
VOWEL = 'a'
VOWELS = ['a', 'e', 'i', 'o', 'u']

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / 'data' / 'liwicki' / 'vowels_nodownsampled'

CANALES = ['F3', 'F4', 'C3', 'C4', 'P3', 'P4']
FS = 1024  # Sin downsampling

def cargar_datos(sujeto, vowel):
    """Carga datos de un sujeto y vocal."""
    subj_str = f'S{sujeto:02d}'
    filepath = DATA_DIR / f'EEG_inner_vowels_{subj_str}.mat'
    data = loadmat(filepath)
    return data[vowel]

def calcular_fft(senal, fs):
    """Calcula FFT y retorna frecuencias y magnitud."""
    fft_vals = rfft(senal)
    freqs = rfftfreq(len(senal), 1/fs)
    magnitud = np.abs(fft_vals)
    return freqs, magnitud

def visualizar_trial(trial_1d, fs, canales, titulo=""):
    """Visualiza trial: señal temporal + FFT por canal."""
    n_timepoints = trial_1d.shape[0] // 6
    trial_2d = trial_1d.reshape(6, n_timepoints)  # (canales, muestras)
    n_canales = len(canales)
    
    fig, axes = plt.subplots(n_canales, 2, figsize=(14, 12))
    fig.suptitle(f'{titulo}\nVocal /{VOWEL}/ - Sujeto S{SUJETO:02d}', fontsize=14, fontweight='bold')
    
    t = np.arange(trial_2d.shape[1]) / fs
    
    for i, canal in enumerate(canales):
        senal = trial_2d[i]
        freqs, magnitud = calcular_fft(senal, fs)
        
        axes[i, 0].plot(t, senal, 'b-', linewidth=0.8)
        axes[i, 0].set_ylabel(canal)
        axes[i, 0].grid(True, alpha=0.3)
        if i == 0:
            axes[i, 0].set_title('Señal Temporal', fontweight='bold')
        if i == n_canales - 1:
            axes[i, 0].set_xlabel('Tiempo (s)')
        
        axes[i, 1].plot(freqs[1:], magnitud[1:], 'r-', linewidth=0.8)
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].axvspan(2, 40, alpha=0.15, color='green')
        if i == 0:
            axes[i, 1].set_title('FFT', fontweight='bold')
        if i == n_canales - 1:
            axes[i, 1].set_xlabel('Frecuencia (Hz)')
    
    plt.tight_layout()
    plt.show()

def visualizar_todos_vowels(data_dict, fs, canales, sujeto):
    """Visualiza un trial de cada vocal."""
    fig, axes = plt.subplots(5, 2, figsize=(14, 16))
    fig.suptitle(f'Sujeto S{sujeto:02d} - Todas las vocales', fontsize=14, fontweight='bold')
    
    for idx, vowel in enumerate(VOWELS):
        trial_1d = data_dict[vowel][0]
        n_timepoints = trial_1d.shape[0] // 6
        trial_2d = trial_1d.reshape(6, n_timepoints)
        t = np.arange(trial_2d.shape[1]) / fs
        
        senal = trial_2d[0]  # Solo canal F3
        freqs, magnitud = calcular_fft(senal, fs)
        
        axes[idx, 0].plot(t, senal, 'b-', linewidth=0.8)
        axes[idx, 0].set_ylabel(f'/{vowel}/')
        axes[idx, 0].grid(True, alpha=0.3)
        if idx == 0:
            axes[idx, 0].set_title('Temporal (F3)', fontweight='bold')
        
        axes[idx, 1].plot(freqs[1:], magnitud[1:], 'r-', linewidth=0.8)
        axes[idx, 1].grid(True, alpha=0.3)
        if idx == 0:
            axes[idx, 1].set_title('FFT (F3)', fontweight='bold')
    
    for ax in axes[-1, :]:
        ax.set_xlabel('Frecuencia (Hz)')
    
    plt.tight_layout()
    plt.show()

def estadisticas(data_dict, canales):
    """Imprime estadísticas de cada vocal."""
    print(f'{"Vocal":<8} | {"Trials":<8} | {"Shape":<15} | {"Mean":>10} | {"Std":>10}')
    print('-' * 60)
    for vowel in VOWELS:
        data = data_dict[vowel]
        print(f'{vowel:<8} | {data.shape[0]:<8} | {str(data.shape):<15} | {data.mean():>10.4f} | {data.std():>10.4f}')

if __name__ == '__main__':
    print('=' * 60)
    print(f'VISUALIZACIÓN DATOS LIWICKI - Sujeto S{SUJETO:02d}')
    print('=' * 60)
    
    subj_str = f'S{SUJETO:02d}'
    filepath = DATA_DIR / f'EEG_inner_vowels_{subj_str}.mat'
    print(f'Archivo: {filepath.name}')
    
    data = loadmat(filepath)
    data_dict = {v: data[v] for v in VOWELS}
    
    print('\n--- Keys en archivo .mat ---')
    print([k for k in data.keys() if not k.startswith('_')])
    
    print('\n--- Estadísticas por vocal ---')
    estadisticas(data_dict, CANALES)
    
    print(f'\n--- Visualizando vowel /{VOWEL}/ ---')
    trial = data_dict[VOWEL][0]
    print(f'Shape trial: {trial.shape}')
    print(f'Shape reshaped (6, 4096): {trial.reshape(6, 4096).shape}')
    
    visualizar_trial(trial, FS, CANALES, titulo='Datos Liwicki')
    
    print('\n--- Visualizando todas las vocales ---')
    visualizar_todos_vowels(data_dict, FS, CANALES, SUJETO)