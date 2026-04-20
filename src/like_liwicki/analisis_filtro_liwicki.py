"""
analisis_filtro_liwicki.py
========================
Analiza el filtro FIR de Liwicki y su efecto en datos EEG.
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy import signal as np_signal

BASE_DIR = Path(__file__).parent.parent.parent
DATA_ORIG = BASE_DIR / 'data' / 'original'

# =============================================================================
# 1. FILTRO FIR DE LIWICKI (mismos parámetros)
# =============================================================================
def disenar_filtros_liwicki(fs=1024):
    """Diseña filtros FIR con parámetros de Liwicki."""
    nyq = fs / 2
    
    # Parámetros de Liwicki
    n1 = 372   # orden lowpass
    Wn1 = 2    # lowcut 2Hz
    n2 = 1204  # orden highpass
    Wn2 = 40   # highcut 40Hz
    
    # Lowpass @ 2Hz (usa ventana de Hamming)
    b_lp = np_signal.firwin(n1 + 1, Wn1 / nyq, window='hamming')
    
    # Highpass @ 40Hz
    # firwin no tiene highpass directo, usamos firwin2
    b_hp = np_signal.firwin2(n2 + 1, [0, Wn2/nyq-0.01, Wn2/nyq, 1], 
                          [0, 0, 1, 1], window='hamming')
    
    return b_lp, b_hp, Wn1, Wn2

def respuesta_filtro(b, fs=1024, n_points=4096):
    """Calcula respuesta en frecuencia."""
    w = np.fft.rfftfreq(n_points, 1/fs)
    H = np.fft.rfft(b, n_points)
    return w, np.abs(H)

def aplicar_filtro(senal, b):
    """Aplica filtro FIR."""
    from scipy.signal import lfilter
    return lfilter(b, 1, senal)

# =============================================================================
# 2. SEÑAL FICTICIA DE ESPECTRO UNIFORME
# =============================================================================
def senal_ficticia_espectro_uniforme(n=4096):
    """Crea señal con espectro uniforme (ruido blanco)."""
    return np.random.randn(n)

# =============================================================================
# 3. MAIN
# =============================================================================
if __name__ == '__main__':
    fs = 1024
    n_muestras = 4096
    
    print('=' * 60)
    print('ANÁLISIS FILTRO LIWICKI')
    print('=' * 60)
    
    # -------------------------------------------------------------------------
    # 3.1 Crear señal ficticia
    # -------------------------------------------------------------------------
    print('\n--- Señal ficticia (espectro uniforme) ---')
    senal_fict = senal_ficticia_espectro_uniforme(n_muestras)
    freqs_fict = rfftfreq(n_muestras, 1/fs)
    fft_fict = np.abs(rfft(senal_fict))
    
    # -------------------------------------------------------------------------
    # 3.2 Diseñar filtros
    # -------------------------------------------------------------------------
    print('\n--- Diseñando filtros FIR ---')
    b_lp, b_hp, Wn1, Wn2 = disenar_filtros_liwicki(fs)
    print(f'Lowpass: orden {len(b_lp)}, corte {Wn1} Hz')
    print(f'Highpass: orden {len(b_hp)}, corte {Wn2} Hz')
    
    # -------------------------------------------------------------------------
    # 3.3 Respuesta del filtro
    # -------------------------------------------------------------------------
    print('\n--- Respuesta del filtro ---')
    w_filt, H_lp = respuesta_filtro(b_lp, fs, n_muestras)
    _, H_hp = respuesta_filtro(b_hp, fs, n_muestras)
    
    # Cascada
    H_cascada = H_lp * H_hp
    H_norm = H_cascada / H_cascada.max()  # Escala 0-1
    
    # Graficar respuesta del filtro
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(w_filt, H_norm, 'b-', linewidth=2, label='Filtro cascada (LP 2Hz + HP 40Hz)')
    ax1.axvspan(2, 40, alpha=0.15, color='green', label='Banda 2-40 Hz')
    ax1.axhline(0.5, color='r', linestyle='--', alpha=0.5, label='-3dB')
    ax1.set_xlabel('Frecuencia (Hz)')
    ax1.set_ylabel('Magnitud normalizada (0-1)')
    ax1.set_title('Respuesta del filtro FIR de Liwicki (escala 0-1)')
    ax1.set_xlim([0, 64])
    ax1.set_ylim([0, 1.1])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('data/liwicki/filtro_respuesta.png', dpi=150)
    print('Guardado: data/liwicki/filtro_respuesta.png')
    plt.close()
    
    # -------------------------------------------------------------------------
    # 3.4 Aplicar filtro a señal ficticia
    # -------------------------------------------------------------------------
    print('\n--- Aplicando filtro a señal ficticia ---')
    senal_filt = aplicar_filtro(aplicar_filtro(senal_fict, b_lp), b_hp)
    fft_filt = np.abs(rfft(senal_filt))
    
    # Comparar FFT antes/después
    fig2, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # FFT antes
    axes[0].plot(freqs_fict[1:], fft_fict[1:], 'b-', linewidth=1)
    axes[0].set_title('FFT ANTES de filtrar (espectro uniforme)')
    axes[0].set_xlabel('Frecuencia (Hz)')
    axes[0].set_ylabel('Magnitud')
    axes[0].set_xlim([0, 64])
    axes[0].grid(True, alpha=0.3)
    
    # FFT después
    axes[1].plot(freqs_fict[1:], fft_filt[1:], 'r-', linewidth=1)
    axes[1].plot(w_filt, H_norm * fft_fict.max(), 'b--', linewidth=1, alpha=0.5, label='Filtro')
    axes[1].set_title('FFT DESPUÉS de filtrar')
    axes[1].set_xlabel('Frecuencia (Hz)')
    axes[1].set_ylabel('Magnitud')
    axes[1].set_xlim([0, 64])
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/liwicki/señal_ficticia_fft.png', dpi=150)
    print('Guardado: data/liwicki/señal_ficticia_fft.png')
    plt.close()
    
    # -------------------------------------------------------------------------
    # 3.5 Cargar datos originales S01
    # -------------------------------------------------------------------------
    print('\n--- Cargando datos originales S01 ---')
    data = np.load(DATA_ORIG / 'S01_EEG.npz', allow_pickle=True)
    data = data[data.files[0]]
    print(f'Shape: {data.shape}')
    
    # Extraer primer trial (vocal 'a' imaginada)
    labels = data[:, -3:]
    mask = (labels[:, 0] == 1) & (labels[:, 1] == 1)
    idx = np.where(mask)[0][0]
    senal_original = data[idx, :-3].reshape(6, 4096)
    print(f'Trial shape: {senal_original.shape}')
    
    # -------------------------------------------------------------------------
    # 3.6 Aplicar filtro a datos reales
    # -------------------------------------------------------------------------
    print('\n--- Aplicando filtro a datos reales ---')
    # Aplicar a cada canal
    senal_filtrada = np.zeros_like(senal_original)
    for i in range(6):
        s = senal_original[i]
        s_lp = aplicar_filtro(s, b_lp)
        senal_filtrada[i] = aplicar_filtro(s_lp, b_hp)
    
    # -------------------------------------------------------------------------
    # 3.7 Comparar temporal y FFT
    # -------------------------------------------------------------------------
    canales = ['F3', 'F4', 'C3', 'C4', 'P3', 'P4']
    t = np.arange(4096) / fs
    
    fig3, axes = plt.subplots(6, 2, figsize=(14, 14))
    fig3.suptitle('Datos originales S01 - Antes vs Después del filtro Liwicki', fontsize=14)
    
    for i, ch in enumerate(canales):
        # Temporal antes
        axes[i, 0].plot(t, senal_original[i], 'b-', linewidth=0.8)
        axes[i, 0].set_ylabel(ch)
        axes[i, 0].grid(True, alpha=0.3)
        if i == 0:
            axes[i, 0].set_title('ANTES', fontweight='bold')
        if i == 5:
            axes[i, 0].set_xlabel('Tiempo (s)')
        
        # FFT antes
        freqs = rfftfreq(4096, 1/fs)
        fft_antes = np.abs(rfft(senal_original[i]))
        
        # Temporal después
        axes[i, 1].plot(t, senal_filtrada[i], 'r-', linewidth=0.8)
        axes[i, 1].grid(True, alpha=0.3)
        if i == 0:
            axes[i, 1].set_title('DESPUÉS', fontweight='bold')
        if i == 5:
            axes[i, 1].set_xlabel('Tiempo (s)')
    
    plt.tight_layout()
    plt.savefig('data/liwicki/datos_reales_temporal.png', dpi=150)
    print('Guardado: data/liwicki/datos_reales_temporal.png')
    plt.close()
    
    # FFT comparacion
    fig4, axes = plt.subplots(6, 2, figsize=(14, 14))
    fig4.suptitle('FFT - Antes vs Después del filtro Liwicki', fontsize=14)
    
    for i, ch in enumerate(canales):
        freqs = rfftfreq(4096, 1/fs)
        fft_antes = np.abs(rfft(senal_original[i]))
        fft_despues = np.abs(rfft(senal_filtrada[i]))
        
        axes[i, 0].plot(freqs[1:], fft_antes[1:], 'b-', linewidth=0.8)
        axes[i, 0].set_ylabel(ch)
        axes[i, 0].set_xlim([0, 64])
        axes[i, 0].grid(True, alpha=0.3)
        if i == 0:
            axes[i, 0].set_title('FFT ANTES', fontweight='bold')
        
        axes[i, 1].plot(freqs[1:], fft_despues[1:], 'r-', linewidth=0.8)
        axes[i, 1].set_xlim([0, 64])
        axes[i, 1].grid(True, alpha=0.3)
        if i == 0:
            axes[i, 1].set_title('FFT DESPUÉS', fontweight='bold')
        if i == 5:
            axes[i, 0].set_xlabel('Frecuencia (Hz)')
            axes[i, 1].set_xlabel('Frecuencia (Hz)')
    
    plt.tight_layout()
    plt.savefig('data/liwicki/datos_reales_fft.png', dpi=150)
    print('Guardado: data/liwicki/datos_reales_fft.png')
    plt.close()
    
    print('\n--- Comparación de potencia por banda ---')
    for i, ch in enumerate(canales):
        fft_antes = np.abs(rfft(senal_original[i]))**2
        fft_despues = np.abs(rfft(senal_filtrada[i]))**2
        freqs = rfftfreq(4096, 1/fs)
        
        mask_2_10 = (freqs >= 2) & (freqs <= 10)
        mask_10_20 = (freqs >= 10) & (freqs <= 20)
        mask_20_40 = (freqs >= 20) & (freqs <= 40)
        
        pot_antes_2_10 = np.sum(fft_antes[mask_2_10])
        pot_antes_10_20 = np.sum(fft_antes[mask_10_20])
        pot_antes_20_40 = np.sum(fft_antes[mask_20_40])
        pot_total_antes = np.sum(fft_antes[1:])
        
        pot_desp_2_10 = np.sum(fft_despues[mask_2_10])
        pot_desp_10_20 = np.sum(fft_despues[mask_10_20])
        pot_desp_20_40 = np.sum(fft_despues[mask_20_40])
        pot_total_desp = np.sum(fft_despues[1:])
        
        print(f'{ch}:')
        print(f'  2-10 Hz:  {pot_antes_2_10/pot_total_antes*100:.1f}% -> {pot_desp_2_10/pot_total_desp*100:.1f}%')
        print(f'  10-20 Hz: {pot_antes_10_20/pot_total_antes*100:.1f}% -> {pot_desp_10_20/pot_total_desp*100:.1f}%')
        print(f'  20-40 Hz: {pot_antes_20_40/pot_total_antes*100:.1f}% -> {pot_desp_20_40/pot_total_desp*100:.1f}%')
    
    print('\n¡Listo!')