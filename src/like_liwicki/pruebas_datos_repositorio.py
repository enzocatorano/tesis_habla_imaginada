"""
pruebas_datos_repositorio.py
===========================
Script para explorar los datos del repo de Liwicki y compararlos con tus datos originales.
Ejecutar: python src/like_liwicki/pruebas_datos_repositorio.py
"""

import numpy as np
from scipy.io import loadmat
from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
SUJETO = 1
VOWEL = 'a'

DATA_DIR_LIWICKI = Path(__name__).parent.parent.parent / 'data' / 'liwicki' / 'vowels_128'
DATA_DIR_ORIGINAL = Path(__name__).parent.parent.parent / 'data' / 'original'
VOWELS = ['a', 'e', 'i', 'o', 'u']

# ============================================================================
# 1. CARGAR DATOS DE LIWICKI
# ============================================================================
print('=' * 60)
print('1. DATOS DE LIWICKI')
print('=' * 60)

subj_str = f'S{SUJETO:02d}'
filename_liwicki = DATA_DIR_LIWICKI / f'EEG_inner_vowels_{subj_str}.mat'
print(f'Archivo: {filename_liwicki}')

data_liwicki = loadmat(filename_liwicki)
print(f'Keys: {[k for k in data_liwicki.keys() if not k.startswith("_")]}')

print('\n--- Shape de cada vowel ---')
for vowel in VOWELS:
    arr = data_liwicki[vowel]
    print(f'  {vowel}: {arr.shape}')

# ============================================================================
# 2. ESTADÍSTICAS DE DATOS LIWICKI
# ============================================================================
print('\n--- Estadísticas vowel /a/ ---')
a_data = data_liwicki['a']
print(f'  Shape: {a_data.shape}')
print(f'  dtype: {a_data.dtype}')
print(f'  min: {a_data.min():.4f}')
print(f'  max: {a_data.max():.4f}')
print(f'  mean: {a_data.mean():.4f}')
print(f'  std: {a_data.std():.4f}')

# ============================================================================
# 3. DISPOSICIÓN DE CANALES
# ============================================================================
print('\n--- Disposición de canales ---')
# 3072 = 6 canales × 512 muestras
trial_liwicki = data_liwicki[VOWEL][0]
# grafico todo el trial concatenado
plt.figure(figsize=(15, 4))
plt.plot(trial_liwicki)
plt.tight_layout()
plt.show()

trial_2d = trial_liwicki.reshape(6, 512)  # (canales, muestras)
print(f'Trial como (6, 512): {trial_2d.shape}')
print(f'Canal 0 - primeras 5 muestras: {trial_2d[0, :5]}')
print(f'Canal 1 - primeras 5 muestras: {trial_2d[1, :5]}')

plt.figure(figsize=(8, 9))
for i in range(6):
    plt.subplot(6, 1, i+1)
    plt.plot(trial_2d[i], label=f'Canal {i+1}')
plt.tight_layout()
plt.legend()
plt.show()

# probemos hacer un reshape pensando que los datos originales estan
# codificados de forma alternada en los canales, es decir
# f3_0, f4_0, c3_0, c4_0, p3_0, p4_0, f3_1, ... , p4_511
# el canal f3 se construye saltando de a 6 muestras, partiendo desde la primera
trial_2d_alt = np.zeros((6, 512))
for i in range(6):
    trial_2d_alt[i] = trial_liwicki[i::6]
print(f'Trial como (6, 512): {trial_2d_alt.shape}')
print(f'Canal 0 - primeras 5 muestras: {trial_2d_alt[0, :5]}')
print(f'Canal 1 - primeras 5 muestras: {trial_2d_alt[1, :5]}')

plt.figure(figsize=(12, 9))
for i in range(6):
    plt.subplot(6, 2, 2*i+1)
    # en el tiempo
    plt.plot(trial_2d_alt[i], label=f'Canal {i+1}')
    # fft
    plt.subplot(6, 2, 2*i+2)
    fft_vals = np.fft.fft(trial_2d_alt[i])
    freqs = np.fft.fftfreq(len(trial_2d_alt[i]), 1/128)
    magnitud = np.abs(fft_vals)
    plt.plot(freqs, magnitud)
plt.tight_layout()
plt.legend()
plt.show()

# ============================================================================
# 4. CARGAR TUS DATOS ORIGINALES
# ============================================================================
print('\n' + '=' * 60)
print('4. DATOS ORIGINALES')
print('=' * 60)

filename_original = DATA_DIR_ORIGINAL / f'S{SUJETO:02d}_EEG.npz'
tus_raw = np.load(filename_original, allow_pickle=True)
tus_raw = tus_raw[tus_raw.files[0]]

print(f'Shape: {tus_raw.shape}')
print(f'  Trials: {tus_raw.shape[0]}')
print(f'  Columns: {tus_raw.shape[1]}')

# Labels: [modalidad, estímulo, artefacto]
labels = tus_raw[:, -3:]
print(f'\n--- Labels ---')
print(f'  Modalidad: 1=imaginada, 2=pronunciada')
print(f'  Estímulo: 1-5 (vocales), 6-11 (comandos)')
print(f'  Mapeo vocales: 1=a, 2=e, 3=i, 4=o, 5=u')

# ============================================================================
# 5. ENCONTRAR TRIAL DE VOCAL 'a' EN TUS DATOS
# ============================================================================
print('\n--- Buscando vocal "a" imaginada ---')
mask = (labels[:, 0] == 1) & (labels[:, 1] == 1)  # imaginada + 'a'
indices_a = np.where(mask)[0]
print(f'Trials de "a" imaginada: {len(indices_a)}')
print(f'Primer trial con índice: {indices_a[0]}')

# Extraer primer trial de 'a'
primer_idx = indices_a[0]
senales = tus_raw[primer_idx, :-3]  # Sin labels
trial_original_2d = senales.reshape(6, 4096)  # (canales, muestras)
print(f'Trial original como (6, 4096): {trial_original_2d.shape}')

# ============================================================================
# 6. COMPARACIÓN VISUAL: ORIGINAL vs LIWICKI
# ============================================================================
print('\n' + '=' * 60)
print('6. COMPARACIÓN VISUAL')
print('=' * 60)

canales = ['F3', 'F4', 'C3', 'C4', 'P3', 'P4']

# Downsample original para comparar (4096 → 512)
trial_orig_ds = trial_original_2d[:, ::8]  # 1024/8 = 128 Hz

fig, axes = plt.subplots(6, 2, figsize=(14, 12))
fig.suptitle(f'Vocal /{VOWEL}/ - Sujeto S{SUJETO:02d}\nIzq: Original | Der: Liwicki', 
             fontsize=14, fontweight='bold')

t_orig = np.arange(trial_orig_ds.shape[1]) / 128  # Tiempo en segundos
t_liwi = np.arange(trial_2d.shape[1]) / 128

for i in range(6):
    # Columna izquierda: Original
    axes[i, 0].plot(t_orig, trial_orig_ds[i], 'b-', linewidth=0.8)
    axes[i, 0].set_ylabel(canales[i])
    if i == 0:
        axes[i, 0].set_title('ORIGINAL (1024 Hz → 128 Hz)', fontweight='bold', color='blue')
    if i == 5:
        axes[i, 0].set_xlabel('Tiempo (s)')
    axes[i, 0].grid(True, alpha=0.3)
    axes[i, 0].set_ylim([-50, 50])
    
    # Columna derecha: Liwicki
    axes[i, 1].plot(t_liwi, trial_2d[i], 'r-', linewidth=0.8)
    if i == 0:
        axes[i, 1].set_title('LIWICKI (128 Hz + filtrado 2-40Hz + ICA)', fontweight='bold', color='red')
    if i == 5:
        axes[i, 1].set_xlabel('Tiempo (s)')
    axes[i, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# 7. ESTADÍSTICAS COMPARATIVAS
# ============================================================================
print('\n--- Comparación de estadísticas ---')
print(f'{"Canal":<8} | {"Original mean":>12} | {"Liwicki mean":>12} | {"Original std":>10} | {"Liwicki std":>10}')
print('-' * 60)
for i, canal in enumerate(canales):
    orig_mean = trial_original_2d[i].mean()
    liwi_mean = trial_2d[i].mean()
    orig_std = trial_original_2d[i].std()
    liwi_std = trial_2d[i].std()
    print(f'{canal:<8} | {orig_mean:>12.4f} | {liwi_mean:>12.4f} | {orig_std:>10.4f} | {liwi_std:>10.4f}')

print('\n--- Observaciones ---')
print('• Datos Liwicki: normalizados (std ≈ 1.0)')
print('• Datos Originales: NO normalizados (std variable)')
print('• Liwicki aplica: filtrado 2-40 Hz + downsampling + ICA')

# ============================================================================
# 8. SEÑAL TEMPORAL Y FFT POR CANAL (LIWICKI)
# ============================================================================
print('\n' + '=' * 60)
print('8. SEÑAL TEMPORAL Y FFT POR CANAL (LIWICKI)')
print('=' * 60)

from scipy.fft import rfft, rfftfreq

fs = 128  # Frecuencia de muestreo de Liwicki

# Calcular FFT
def calcular_fft(senal, fs):
    fft_vals = rfft(senal)
    freqs = rfftfreq(len(senal), 1/fs)
    magnitud = np.abs(fft_vals)
    return freqs, magnitud

# Figura: 6 filas x 2 columnas
# Izquierda: señal temporal | Derecha: FFT
fig, axes = plt.subplots(6, 2, figsize=(14, 14))
fig.suptitle(f'Vocal /{VOWEL}/ - Sujeto S{SUJETO:02d} (Liwicki)\nIzq: Temporal | Der: FFT', 
             fontsize=14, fontweight='bold')

t = np.arange(trial_2d.shape[1]) / fs  # Tiempo en segundos

for i, canal in enumerate(canales):
    senal = trial_2d[i]
    freqs, magnitud = calcular_fft(senal, fs)
    
    # Columna izquierda: Señal temporal
    axes[i, 0].plot(t, senal, 'b-', linewidth=0.8)
    axes[i, 0].set_ylabel(canales[i])
    if i == 0:
        axes[i, 0].set_title('Señal Temporal', fontweight='bold')
    if i == 5:
        axes[i, 0].set_xlabel('Tiempo (s)')
    axes[i, 0].grid(True, alpha=0.3)
    
    # Columna derecha: FFT
    axes[i, 1].plot(freqs, magnitud, 'r-', linewidth=0.8)
    if i == 0:
        axes[i, 1].set_title('FFT', fontweight='bold')
    if i == 5:
        axes[i, 1].set_xlabel('Frecuencia (Hz)')
    axes[i, 1].grid(True, alpha=0.3)
    axes[i, 1].axvspan(2, 40, alpha=0.15, color='green')  # Banda filtrada

plt.tight_layout()
plt.show()

# ============================================================================
# 9. APLICAR PIPELINE DE LIWICKI A TUS DATOS ORIGINALES
# ============================================================================
print('\n' + '=' * 60)
print('9. PIPELINE DE PREPROCESAMIENTO COMO LIWICKI')
print('=' * 60)

from scipy.signal import butter, filtfilt
from mne.preprocessing import ICA
from mne.io import RawArray
import mne

# Cargar datos originales
filename_original = DATA_DIR_ORIGINAL / f'S{SUJETO:02d}_EEG.npz'
tus_raw = np.load(filename_original, allow_pickle=True)
tus_raw = tus_raw[tus_raw.files[0]]

# Parámetros
FS_ORIG = 1024
FS_TARGET = 128
N_CANALES = 6
N_MUESTRAS_ORIG = 4096
N_MUESTRAS_TARGET = 512

# Separar señales y labels
senales = tus_raw[:, :-3]  # (688, 24576)
labels = tus_raw[:, -3:]   # (688, 3)

# Filtrar por vocal 'a' imaginada
mask = (labels[:, 0] == 1) & (labels[:, 1] == 1)  # imaginada + 'a'
indices_a = np.where(mask)[0]
print(f'Trials de "a" imaginada: {len(indices_a)}')

# Reshape a (trials, canales, muestras)
senales_reshaped = senales.reshape(-1, N_CANALES, N_MUESTRAS_ORIG)
senales_a = senales_reshaped[indices_a]  # (n_trials, 6, 4096)
print(f'Shape vocal /a/ imaginada: {senales_a.shape}')

# 9.1 FILTRADO 2-40 Hz
print('\n--- 9.1 Filtrado 2-40 Hz ---')
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def filtrar_banda(data, lowcut, highcut, fs):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return filtfilt(b, a, data, axis=-1)

senales_filtradas = filtrar_banda(senales_a, 2, 40, FS_ORIG)
print(f'Filtrado: {senales_filtradas.shape}')

# 9.2 DOWNSAMPLING a 128 Hz
print('\n--- 9.2 Downsampling 128 Hz ---')
from scipy.signal import resample_poly
factor_down = FS_ORIG // FS_TARGET  # 8
senales_ds = resample_poly(senales_filtradas, 1, factor_down, axis=-1)
print(f'Downsampled: {senales_ds.shape}')

# 9.3 APLICAR ICA PARA REMOCIÓN DE ARTEFACTOS
print('\n--- 9.3 ICA para remoción de artefactos ---')

# Concatenar todos los trials para ICA
senales_concat = senales_ds.transpose(1, 0, 2).reshape(N_CANALES, -1)
print(f'Concatenado para ICA: {senales_concat.shape}')

# Crear objeto Raw de MNE
ch_names = ['F3', 'F4', 'C3', 'C4', 'P3', 'P4']
info = mne.create_info(ch_names=ch_names, sfreq=FS_TARGET, ch_types=['eeg'] * N_CANALES)
raw = mne.io.RawArray(senales_concat, info)

# High-pass 1Hz para ICA
raw_hp = raw.copy().filter(l_freq=1.0, h_freq=None, method='fir')
raw_hp.set_montage('standard_1005')

# Fit ICA
ica = ICA(n_components=N_CANALES, method='fastica', max_iter=1000, random_state=42)
ica.fit(raw_hp)

# Detectar componentes EOG (parpadeo)
print('Detectando componentes EOG...')
eog_inds, eog_scores = ica.find_bads_eog(raw_hp, ch_name='F3', threshold=3.0)
print(f'Componentes EOG detectados: {eog_inds}')

# Excluir y aplicar
ica.exclude = eog_inds
raw_corrected = ica.apply(raw_hp.copy())

# Obtener datos corregidos
datos_corrected = raw_corrected.get_data()
print(f'Datos corregidos: {datos_corrected.shape}')

# 9.4 RESHAPE A FORMATO LIWICKI (trials, 3072)
print('\n--- 9.4 Reshape a formato Liwicki ---')
# Reorganizar: (canales, n_trials, muestras) -> (n_trials, canales, muestras) -> (n_trials, 3072)
n_trials = senales_ds.shape[0]
datos_reshaped = datos_corrected.reshape(N_CANALES, n_trials, N_MUESTRAS_TARGET)
datos_reshaped = datos_reshaped.transpose(1, 0, 2)  # (trials, canales, muestras)
datos_final = datos_reshaped.reshape(n_trials, -1)  # (trials, 3072)
print(f'Final: {datos_final.shape}')

# 9.5 COMPARACIÓN VISUAL
print('\n--- 9.5 Comparación visual ---')

# Elegir el primer trial para comparar
trial_nuestro = datos_final[0]  # (3072,)
trial_liwicki = data_liwicki[VOWEL][0]  # (3072,)

trial_nuestro_2d = trial_nuestro.reshape(6, 512)
trial_liwicki_2d = trial_liwicki.reshape(6, 512)

fig, axes = plt.subplots(6, 2, figsize=(14, 14))
fig.suptitle(f'Vocal /{VOWEL}/ - Sujeto S{SUJETO:02d}\nIzq: Nuestro Pipeline | Der: Liwicki', 
             fontsize=14, fontweight='bold')

t = np.arange(512) / FS_TARGET

for i in range(6):
    # Columna izquierda: Nuestro pipeline
    axes[i, 0].plot(t, trial_nuestro_2d[i], 'b-', linewidth=0.8)
    axes[i, 0].set_ylabel(canales[i])
    if i == 0:
        axes[i, 0].set_title('NUESTRO (filtrado 2-40Hz + downsampling + ICA)', fontweight='bold', color='blue')
    if i == 5:
        axes[i, 0].set_xlabel('Tiempo (s)')
    axes[i, 0].grid(True, alpha=0.3)
    
    # Columna derecha: Liwicki
    axes[i, 1].plot(t, trial_liwicki_2d[i], 'r-', linewidth=0.8)
    if i == 0:
        axes[i, 1].set_title('LIWICKI', fontweight='bold', color='red')
    if i == 5:
        axes[i, 1].set_xlabel('Tiempo (s)')
    axes[i, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 9.6 ESTADÍSTICAS COMPARATIVAS
print('\n--- Comparación de estadísticas ---')
print(f'{"Canal":<8} | {"Nuestro mean":>12} | {"Liwicki mean":>12} | {"Nuestro std":>10} | {"Liwicki std":>10}')
print('-' * 60)
for i, canal in enumerate(canales):
    nuestro_mean = trial_nuestro_2d[i].mean()
    liwi_mean = trial_liwicki_2d[i].mean()
    nuestro_std = trial_nuestro_2d[i].std()
    liwi_std = trial_liwicki_2d[i].std()
    print(f'{canal:<8} | {nuestro_mean:>12.4f} | {liwi_mean:>12.4f} | {nuestro_std:>10.4f} | {liwi_std:>10.4f}')

print('\n--- Resumen ---')
print('Nuestro pipeline:')
print('  1. Filtrado bandpass 2-40 Hz')
print('  2. Downsampling 1024→128 Hz')
print('  3. High-pass 1Hz para ICA')
print('  4. ICA fit con FastICA')
print('  5. Detección automática EOG con find_bads_eog')
print('  6. Remoción de componentes EOG')
print('  7. Reconstrucción de señal')
