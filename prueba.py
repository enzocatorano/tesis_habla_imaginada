import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pywt
import seaborn as sns
from sklearn.manifold import TSNE
from scipy.signal import resample_poly
from scipy import stats


# quiero cargar este archivo
ruta_archivo = 'data\processed\S01_EEG.npz'

# Cargar el archivo .npz
data = np.load(ruta_archivo)
eeg_data = data['data']

# Asumiendo que las últimas 3 columnas son etiquetas y el resto son datos EEG
# y que los datos EEG están en formato (n_trials, n_channels * n_samples)
labels = eeg_data[:, -3:]
eeg_signals_flat = eeg_data[:, :-3]

# reestructuro la señal para que quede en formato (trials, canales, samples)
signal = eeg_signals_flat.reshape(eeg_signals_flat.shape[0], 6, -1)

# a cada canal lo submuestreo de 1024 a 128 Hz
# usando resample_poly
resampled_signal = np.zeros((signal.shape[0], signal.shape[1], 128*4))
for i in range(signal.shape[0]):
    for j in range(signal.shape[1]):
        resampled_signal[i, j, :] = resample_poly(signal[i, j, :], 128, 1024)
signal = resampled_signal

# ahora generamos 2 pares de array signal-label para comandos y vocales
vowels_mask = (labels[:, 1] >= 1) & (labels[:, 1] <= 5)
commands_mask = (labels[:, 1] >= 6) & (labels[:, 1] <= 11)

signal_vowels = signal[vowels_mask]
labels_vowels = labels[vowels_mask]

signal_commands = signal[commands_mask]
labels_commands = labels[commands_mask]

# canales y colores
nombres_canales = ['F3', 'F4', 'C3', 'C4', 'P3', 'P4']

posible_modalidad = ['Imaginada', 'Pronunciada']
posible_vocal = ['A', 'E', 'I', 'O', 'U']
posible_comando = ['Arrriba', 'Abajo', 'Izquierda', 'Derecha', 'Adelante', 'Atras']
posible_artefacto = ['Limpio', 'Parpadeo']

predecir_estimulo = True

# si predecir_estimulo = True hacer 2 figuras de tSNE, si no 1
# para cada canal por separado
# la primera tiene etiquetado por color la modalidad y el artefacto de cada trial
# la segunda tiene vocales y comandos para cada subconjunto de trials

# parametros graficos
paleta = 'jet'
ancho = 18
alto = 10
tam_puntos = 10
alpha = 0.75

# parametros tSNE
n_components = 2
perplexity = 100
max_iter = 2000
semilla = 17

if predecir_estimulo:
    fig, ax = plt.subplots(4, 6, figsize=(ancho, alto))
    for i in range(6): # modalidad y artefactos sobre todos los trials
        canal_actual = signal[:, i, :].reshape(signal.shape[0], -1)
        tsne = TSNE(n_components = n_components,
                    perplexity = perplexity,
                    random_state = semilla,
                    max_iter = max_iter)
        tsne_resultado = tsne.fit_transform(canal_actual)
        # fila 0, modalidad
        ax[0, i].scatter(tsne_resultado[:, 0], tsne_resultado[:, 1], c=labels[:, 0], cmap = paleta, alpha = alpha, s = tam_puntos)
        # fila 1, artefacto
        ax[1, i].scatter(tsne_resultado[:, 0], tsne_resultado[:, 1], c=labels[:, 2], cmap = paleta, alpha = alpha, s = tam_puntos)
    for i in range(6): # vocales y comandos por separado
        canal_actual = signal_vowels[:, i, :].reshape(signal_vowels.shape[0], -1)
        tsne = TSNE(n_components = n_components,
                    perplexity = perplexity,
                    random_state = semilla,
                    max_iter = max_iter)
        tsne_resultado = tsne.fit_transform(canal_actual)
        # fila 2, vocales
        ax[2, i].scatter(tsne_resultado[:, 0], tsne_resultado[:, 1], c=labels_vowels[:, 1], cmap = paleta, alpha = alpha, s = tam_puntos)

        canal_actual = signal_commands[:, i, :].reshape(signal_commands.shape[0], -1)
        tsne = TSNE(n_components = n_components,
                    perplexity = perplexity,
                    random_state = semilla,
                    max_iter = max_iter)
        tsne_resultado = tsne.fit_transform(canal_actual)
        # fila 3, comandos
        ax[3, i].scatter(tsne_resultado[:, 0], tsne_resultado[:, 1], c=labels_commands[:, 1], cmap = paleta, alpha = alpha, s = tam_puntos)

    for i in range(6):
        ax[0, i].set_title(f'Modalidad - {nombres_canales[i]}')
        ax[1, i].set_title(f'Artefacto - {nombres_canales[i]}')
        ax[2, i].set_title(f'Vocales - {nombres_canales[i]}')
        ax[3, i].set_title(f'Comandos - {nombres_canales[i]}')

    plt.tight_layout()
    plt.show()

else:
    fig, ax = plt.subplots(2, 6, figsize=(ancho, alto))
    for i in range(6):
        canal_actual = signal[:, i, :].reshape(signal.shape[0], -1)
        tsne = TSNE(n_components = n_components,
                    perplexity = perplexity,
                    random_state = semilla,
                    max_iter = max_iter)
        tsne_resultado = tsne.fit_transform(canal_actual)
        # fila 0
        ax[0, i].scatter(tsne_resultado[:, 0], tsne_resultado[:, 1], c=labels[:, 0], cmap = paleta, alpha = alpha, s = tam_puntos)
        # fila 1
        ax[1, i].scatter(tsne_resultado[:, 0], tsne_resultado[:, 1], c=labels[:, 2], cmap = paleta, alpha = alpha, s = tam_puntos)

    for i in range(6):
        ax[0, i].set_title(f'Modalidad - {nombres_canales[i]}')
        ax[1, i].set_title(f'Artefacto - {nombres_canales[i]}')

    plt.tight_layout()
    plt.show()

#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
# descomponemos por DWT cada banda

ondita = 'db4'
niveles = 5

coeffs = []
for i in range(signal.shape[0]): # para cada trial
    trial_coeffs = []
    for j in range(signal.shape[1]): # para cada canal
        coeffs_ch = pywt.wavedec(signal[i, j, :], ondita, level=niveles)
        trial_coeffs.append(coeffs_ch)
    coeffs.append(trial_coeffs)

# coeffs ahora es una lista de listas de tuplas de arrays
# coeffs[trial_idx][channel_idx][level_idx]
# donde los vectores de los diferentes niveles tienen longitudes diferentes

# quiero reordenar los datos
# necesito que queden expresados (niveles + 1) arrays
# cada array tendra dimension (trials, canales, longitud descomposicion)
# para cada nivel de descomposicion
# la longitud de los coeficientes es diferente
# por lo tanto, no puedo apilar directamente los coeficientes en un solo array
# en su lugar, voy a procesar cada nivel por separado
# para cada nivel de descomposicion, extraigo los coeficientes
# y los almaceno en una lista de arrays
coeffs_por_nivel = [[] for _ in range(niveles + 1)]
for trial_idx in range(signal.shape[0]):
    for channel_idx in range(signal.shape[1]):
        for level_idx in range(niveles + 1):
            coeffs_por_nivel[level_idx].append(coeffs[trial_idx][channel_idx][level_idx])

# ahora, para cada nivel, convierto la lista de arrays en un solo array numpy
# la forma sera (n_trials * n_channels, longitud_coeficientes_nivel)
# esto es para poder aplicar t-SNE a los coeficientes de cada nivel
# y luego mapear de vuelta a los trials y canales originales
for level_idx in range(niveles + 1):
    coeffs_por_nivel[level_idx] = np.array(coeffs_por_nivel[level_idx])
    # la forma actual es (n_trials * n_channels, longitud_coeficientes_nivel)
    # la quiero (n_trials, n_channels, longitud_coeficientes_nivel)
    coeffs_por_nivel[level_idx] = coeffs_por_nivel[level_idx].reshape(signal.shape[0], signal.shape[1], -1)

# quiero graficar con tSNE como se ven las distintas 3 etiquetas
# modalidad, estimulo (separando vocales de comandos) y artefactos
# para cada canal y cada nivel de descomposicion

# nombres de las descomposiciones
nombres_niveles = [f'cA{niveles} (<{128/(2**(niveles + 1))})']
aux = []
for i in range(niveles):
    aux.append(f'cD{i + 1} ({128/(2**(i + 2))} - {128/(2**(i + 1))})')
aux = aux[::-1]
nombres_niveles += aux
print(nombres_niveles)

# parametros graficos
paleta = 'jet'
ancho = 20
alto = 9
tam_puntos = 10
alpha = 0.85
mask_modalidad = (labels[:, 0] == 1) # mascara de etiqueta de modalidad para los trials
mask_artefacto = (labels[:, 2] == 1) # mascara de etiqueta de artefacto para los trials
color = ['r', 'b']

# parametros tSNE
n_components = 2
perplexity = 50
max_iter = 1000
semilla = 17

for tipo in range(2): # modalidad y artefactos (usan todos los datos), vocales y comandos (enmascaran partes diferentes)
    if tipo == 0: # modalidad y artefactos
        fig, ax = plt.subplots(6, 6, figsize=(ancho, alto)) # modalidades
        for canal in range(6): # para cada canal
            for nivel in range(niveles + 1): # para cada nivel de descomposicion
                signal_actual = coeffs_por_nivel[nivel][:, canal, :].reshape(signal.shape[0], -1)
                tsne = TSNE(n_components = n_components,
                            perplexity = perplexity,
                            random_state = semilla,
                            max_iter = max_iter)
                tsne_resultado = tsne.fit_transform(signal_actual)

                ax[canal, nivel].scatter(tsne_resultado[mask_modalidad, 0], tsne_resultado[mask_modalidad, 1], c=color[0], label=posible_modalidad[0], alpha=alpha, s=tam_puntos)
                ax[canal, nivel].scatter(tsne_resultado[~mask_modalidad, 0], tsne_resultado[~mask_modalidad, 1], c=color[1], label=posible_modalidad[1], alpha=alpha, s=tam_puntos)

                ax[canal, nivel].set_xticks([])
                ax[canal, nivel].set_yticks([])
        fig.suptitle('t-SNE de Coeficientes DWT por Modalidad', fontsize=16)
        # el recuadro de leyenda lo quiero debajo de toda la figura, que no se cruze con nada
        handles, labels_legend = ax[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels_legend, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02))
        for i in range(6):
            ax[i, 0].set_ylabel(f'Canal {nombres_canales[i]}')
        for i in range(niveles + 1):
            ax[0, i].set_title(f'Nivel {nombres_niveles[i]}')
        plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots(6, 6, figsize=(ancho, alto)) # artefactos
        for canal in range(6): # para cada canal
            for nivel in range(niveles + 1): # para cada nivel de descomposicion
                signal_actual = coeffs_por_nivel[nivel][:, canal, :].reshape(signal.shape[0], -1)
                tsne = TSNE(n_components = n_components,
                            perplexity = perplexity,
                            random_state = semilla,
                            max_iter = max_iter)
                tsne_resultado = tsne.fit_transform(signal_actual)

                ax[canal, nivel].scatter(tsne_resultado[mask_artefacto, 0], tsne_resultado[mask_artefacto, 1], c=color[0], label=posible_artefacto[0], alpha=alpha, s=tam_puntos)
                ax[canal, nivel].scatter(tsne_resultado[~mask_artefacto, 0], tsne_resultado[~mask_artefacto, 1], c=color[1], label=posible_artefacto[1], alpha=alpha, s=tam_puntos)
                
                ax[canal, nivel].set_xticks([])
                ax[canal, nivel].set_yticks([])
        fig.suptitle('t-SNE de Coeficientes DWT por Artefacto', fontsize=16)
        handles, labels_legend = ax[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels_legend, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02))
        for i in range(6):
            ax[i, 0].set_ylabel(f'Canal {nombres_canales[i]}')
        for i in range(niveles + 1):
            ax[0, i].set_title(f'Nivel {nombres_niveles[i]}')
        plt.tight_layout()
        plt.show()

    else: # vocales y comandos (los hacemos en 2 figuras separadas)
        fig, ax = plt.subplots(6, 6, figsize=(ancho, alto))
        for canal in range(6): # para cada canal
            for nivel in range(niveles + 1): # para cada nivel de descomposicion
                signal_actual = coeffs_por_nivel[nivel][vowels_mask, canal, :].reshape(signal_vowels.shape[0], -1)
                tsne = TSNE(n_components = n_components,
                            perplexity = perplexity,
                            random_state = semilla,
                            max_iter = max_iter)
                tsne_resultado = tsne.fit_transform(signal_actual)
                ax[canal, nivel].scatter(tsne_resultado[:, 0], tsne_resultado[:, 1], c=labels_vowels[:, 1], cmap = paleta, alpha = alpha, s = tam_puntos)
                ax[canal, nivel].set_xticks([])
                ax[canal, nivel].set_yticks([])
        fig.suptitle('t-SNE de Coeficientes DWT para Vocales', fontsize=16)
        for i in range(6):
            ax[i, 0].set_ylabel(f'Canal {nombres_canales[i]}')
        for i in range(niveles + 1):
            ax[0, i].set_title(f'Nivel {nombres_niveles[i]}')
        plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots(6, 6, figsize=(ancho, alto))
        for canal in range(6): # para cada canal
            for nivel in range(niveles + 1): # para cada nivel de descomposicion
                signal_actual = coeffs_por_nivel[nivel][commands_mask, canal, :].reshape(signal_commands.shape[0], -1)
                tsne = TSNE(n_components = n_components,
                            perplexity = perplexity,
                            random_state = semilla,
                            max_iter = max_iter)
                tsne_resultado = tsne.fit_transform(signal_actual)
                ax[canal, nivel].scatter(tsne_resultado[:, 0], tsne_resultado[:, 1], c=labels_commands[:, 1], cmap = paleta, alpha = alpha, s = tam_puntos)
                ax[canal, nivel].set_xticks([])
                ax[canal, nivel].set_yticks([])
        fig.suptitle('t-SNE de Coeficientes DWT para Comandos', fontsize=16)
        for i in range(6):
            ax[i, 0].set_ylabel(f'Canal {nombres_canales[i]}')
        for i in range(niveles + 1):
            ax[0, i].set_title(f'Nivel {nombres_niveles[i]}')
        plt.tight_layout()
        plt.show()

#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
# calculamos features por banda y descomposicion
# media, media absoluta, varianza, desviacion estandar, skewness y kurtosis

def calcular_estadisticos (vector):
    media = np.mean(vector)
    media_abs = np.mean(np.abs(vector))
    varianza = np.var(vector)
    desviacion_estandar = np.std(vector)
    skewness = stats.skew(vector)
    kurtosis = stats.kurtosis(vector)

    vector_estadisticos = [media, media_abs, varianza, desviacion_estandar, skewness, kurtosis]
    return vector_estadisticos

# quiero componer todos los datos en un unico vector con forma
# (trials, canales, descomposiciones, estadisticos)
estadisticos_dwt = np.zeros((signal.shape[0], signal.shape[1], niveles + 1, 6))

for trial_idx in range(signal.shape[0]):
    for channel_idx in range(signal.shape[1]):
        for level_idx in range(niveles + 1):
            vector_coeficientes = coeffs_por_nivel[level_idx][trial_idx, channel_idx, :]
            estadisticos_dwt[trial_idx, channel_idx, level_idx, :] = calcular_estadisticos(vector_coeficientes)

# ahora quiero graficar con tSNE como se ven las distintas 3 etiquetas
# modalidad, estimulo (separando vocales de comandos) y artefactos
# para cada canal y cada banda

# nombres de los estadisticos
nombres_estadisticos = ['Media', 'Media Absoluta', 'Varianza', 'Desviacion Estandar', 'Skewness', 'Kurtosis']

# parametros graficos
paleta = 'gist_rainbow'
ancho = 18
alto = 9
tam_puntos = 10
alpha = [0.35, 0.85]

# parametros tSNE
n_components = 2
perplexity = 30
max_iter = 2000
semilla = 17

for tipo in range(2): # modalidad y artefactos (usan todos los datos), vocales y comandos (enmascaran partes diferentes)
    if tipo == 0: # modalidad y artefactos
        fig, ax = plt.subplots(6, niveles + 1, figsize=(ancho, alto)) # modalidades
        for canal in range(6): # para cada canal
            for nivel in range(niveles + 1): # para cada nivel de descomposicion
                # Aplanar los estadísticos para el t-SNE
                stats_actual = estadisticos_dwt[:, canal, nivel, :].reshape(signal.shape[0], -1)
                tsne = TSNE(n_components = n_components,
                            perplexity = perplexity,
                            random_state = semilla,
                            max_iter = max_iter)
                tsne_resultado = tsne.fit_transform(stats_actual)

                ax[canal, nivel].scatter(tsne_resultado[mask_modalidad, 0], tsne_resultado[mask_modalidad, 1], c=color[0], label=posible_modalidad[0], alpha=alpha[tipo], s=tam_puntos)
                ax[canal, nivel].scatter(tsne_resultado[~mask_modalidad, 0], tsne_resultado[~mask_modalidad, 1], c=color[1], label=posible_modalidad[1], alpha=alpha[tipo], s=tam_puntos)

                ax[canal, nivel].set_xticks([])
                ax[canal, nivel].set_yticks([])
        fig.suptitle('t-SNE de Estadísticos DWT por Modalidad', fontsize=16)
        handles, labels_legend = ax[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels_legend, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02))
        for i in range(6):
            ax[i, 0].set_ylabel(f'Canal {nombres_canales[i]}')
        for i in range(niveles + 1):
            ax[0, i].set_title(f'Nivel {nombres_niveles[i]}')
        plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots(6, niveles + 1, figsize=(ancho, alto)) # artefactos
        for canal in range(6): # para cada canal
            for nivel in range(niveles + 1): # para cada nivel de descomposicion
                stats_actual = estadisticos_dwt[:, canal, nivel, :].reshape(signal.shape[0], -1)
                tsne = TSNE(n_components = n_components,
                            perplexity = perplexity,
                            random_state = semilla,
                            max_iter = max_iter)
                tsne_resultado = tsne.fit_transform(stats_actual)

                ax[canal, nivel].scatter(tsne_resultado[mask_artefacto, 0], tsne_resultado[mask_artefacto, 1], c=color[0], label=posible_artefacto[0], alpha=alpha[tipo], s=tam_puntos)
                ax[canal, nivel].scatter(tsne_resultado[~mask_artefacto, 0], tsne_resultado[~mask_artefacto, 1], c=color[1], label=posible_artefacto[1], alpha=alpha[tipo], s=tam_puntos)
                
                ax[canal, nivel].set_xticks([])
                ax[canal, nivel].set_yticks([])
        fig.suptitle('t-SNE de Estadísticos DWT por Artefacto', fontsize=16)
        handles, labels_legend = ax[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels_legend, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02))
        for i in range(6):
            ax[i, 0].set_ylabel(f'Canal {nombres_canales[i]}')
        for i in range(niveles + 1):
            ax[0, i].set_title(f'Nivel {nombres_niveles[i]}')
        plt.tight_layout()
        plt.show()

    else: # vocales y comandos (los hacemos en 2 figuras separadas)
        fig, ax = plt.subplots(6, niveles + 1, figsize=(ancho, alto))
        for canal in range(6): # para cada canal
            for nivel in range(niveles + 1): # para cada nivel de descomposicion
                stats_actual = estadisticos_dwt[vowels_mask, canal, nivel, :].reshape(signal_vowels.shape[0], -1)
                tsne = TSNE(n_components = n_components,
                            perplexity = perplexity,
                            random_state = semilla,
                            max_iter = max_iter)
                tsne_resultado = tsne.fit_transform(stats_actual)
                ax[canal, nivel].scatter(tsne_resultado[:, 0], tsne_resultado[:, 1], c=labels_vowels[:, 1], cmap = paleta, alpha = alpha[tipo], s = tam_puntos)
                ax[canal, nivel].set_xticks([])
                ax[canal, nivel].set_yticks([])
        fig.suptitle('t-SNE de Estadísticos DWT para Vocales', fontsize=16)
        for i in range(6):
            ax[i, 0].set_ylabel(f'Canal {nombres_canales[i]}')
        for i in range(niveles + 1):
            ax[0, i].set_title(f'Nivel {nombres_niveles[i]}')
        plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots(6, niveles + 1, figsize=(ancho, alto))
        for canal in range(6): # para cada canal
            for nivel in range(niveles + 1): # para cada nivel de descomposicion
                stats_actual = estadisticos_dwt[commands_mask, canal, nivel, :].reshape(signal_commands.shape[0], -1)
                tsne = TSNE(n_components = n_components,
                            perplexity = perplexity,
                            random_state = semilla,
                            max_iter = max_iter)
                tsne_resultado = tsne.fit_transform(stats_actual)
                ax[canal, nivel].scatter(tsne_resultado[:, 0], tsne_resultado[:, 1], c=labels_commands[:, 1], cmap = paleta, alpha = alpha[tipo], s = tam_puntos)
                ax[canal, nivel].set_xticks([])
                ax[canal, nivel].set_yticks([])
        fig.suptitle('t-SNE de Estadísticos DWT para Comandos', fontsize=16)
        for i in range(6):
            ax[i, 0].set_ylabel(f'Canal {nombres_canales[i]}')
        for i in range(niveles + 1):
            ax[0, i].set_title(f'Nivel {nombres_niveles[i]}')
        plt.tight_layout()
        plt.show()

# normalizando por estadistico
estadisticos_normalizados_dwt = np.zeros(estadisticos_dwt.shape)
# con z-score
for estadistico in range(6):
    media = np.mean(estadisticos_dwt[:, :, :, estadistico])
    desviacion = np.std(estadisticos_dwt[:, :, :, estadistico])
    estadisticos_normalizados_dwt[:, :, :, estadistico] = (estadisticos_dwt[:, :, :, estadistico] - media) / desviacion

# y si normalizo por estadistico, pero dentro del grupo de las bandas
# es decir, todos los trials y canales 

# parametros graficos
paleta = 'gist_rainbow'
ancho = 18
alto = 9
tam_puntos = 10
alpha = [0.5, 0.85]

# parametros tSNE
n_components = 2
perplexity = 50
max_iter = 500
semilla = 17

for tipo in range(2): # modalidad y artefactos (usan todos los datos), vocales y comandos (enmascaran partes diferentes)
    if tipo == 0: # modalidad y artefactos
        fig, ax = plt.subplots(6, niveles + 1, figsize=(ancho, alto)) # modalidades
        for canal in range(6): # para cada canal
            for nivel in range(niveles + 1): # para cada nivel de descomposicion
                # Aplanar los estadísticos para el t-SNE
                stats_actual = estadisticos_normalizados_dwt[:, canal, nivel, :].reshape(signal.shape[0], -1)
                tsne = TSNE(n_components = n_components,
                            perplexity = perplexity,
                            random_state = semilla,
                            max_iter = max_iter)
                tsne_resultado = tsne.fit_transform(stats_actual)

                ax[canal, nivel].scatter(tsne_resultado[mask_modalidad, 0], tsne_resultado[mask_modalidad, 1], c=color[0], label=posible_modalidad[0], alpha=alpha[tipo], s=tam_puntos)
                ax[canal, nivel].scatter(tsne_resultado[~mask_modalidad, 0], tsne_resultado[~mask_modalidad, 1], c=color[1], label=posible_modalidad[1], alpha=alpha[tipo], s=tam_puntos)

                ax[canal, nivel].set_xticks([])
                ax[canal, nivel].set_yticks([])
        fig.suptitle('t-SNE de Estadísticos DWT por Modalidad', fontsize=16)
        handles, labels_legend = ax[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels_legend, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02))
        for i in range(6):
            ax[i, 0].set_ylabel(f'Canal {nombres_canales[i]}')
        for i in range(niveles + 1):
            ax[0, i].set_title(f'Nivel {nombres_niveles[i]}')
        plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots(6, niveles + 1, figsize=(ancho, alto)) # artefactos
        for canal in range(6): # para cada canal
            for nivel in range(niveles + 1): # para cada nivel de descomposicion
                stats_actual = estadisticos_normalizados_dwt[:, canal, nivel, :].reshape(signal.shape[0], -1)
                tsne = TSNE(n_components = n_components,
                            perplexity = perplexity,
                            random_state = semilla,
                            max_iter = max_iter)
                tsne_resultado = tsne.fit_transform(stats_actual)

                ax[canal, nivel].scatter(tsne_resultado[mask_artefacto, 0], tsne_resultado[mask_artefacto, 1], c=color[0], label=posible_artefacto[0], alpha=alpha[tipo], s=tam_puntos)
                ax[canal, nivel].scatter(tsne_resultado[~mask_artefacto, 0], tsne_resultado[~mask_artefacto, 1], c=color[1], label=posible_artefacto[1], alpha=alpha[tipo], s=tam_puntos)
                
                ax[canal, nivel].set_xticks([])
                ax[canal, nivel].set_yticks([])
        fig.suptitle('t-SNE de Estadísticos DWT por Artefacto', fontsize=16)
        handles, labels_legend = ax[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels_legend, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02))
        for i in range(6):
            ax[i, 0].set_ylabel(f'Canal {nombres_canales[i]}')
        for i in range(niveles + 1):
            ax[0, i].set_title(f'Nivel {nombres_niveles[i]}')
        plt.tight_layout()
        plt.show()

    else: # vocales y comandos (los hacemos en 2 figuras separadas)
        fig, ax = plt.subplots(6, niveles + 1, figsize=(ancho, alto))
        for canal in range(6): # para cada canal
            for nivel in range(niveles + 1): # para cada nivel de descomposicion
                stats_actual = estadisticos_normalizados_dwt[vowels_mask, canal, nivel, :].reshape(signal_vowels.shape[0], -1)
                tsne = TSNE(n_components = n_components,
                            perplexity = perplexity,
                            random_state = semilla,
                            max_iter = max_iter)
                tsne_resultado = tsne.fit_transform(stats_actual)
                ax[canal, nivel].scatter(tsne_resultado[:, 0], tsne_resultado[:, 1], c=labels_vowels[:, 1], cmap = paleta, alpha = alpha[tipo], s = tam_puntos)
                ax[canal, nivel].set_xticks([])
                ax[canal, nivel].set_yticks([])
        fig.suptitle('t-SNE de Estadísticos DWT para Vocales', fontsize=16)
        for i in range(6):
            ax[i, 0].set_ylabel(f'Canal {nombres_canales[i]}')
        for i in range(niveles + 1):
            ax[0, i].set_title(f'Nivel {nombres_niveles[i]}')
        plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots(6, niveles + 1, figsize=(ancho, alto))
        for canal in range(6): # para cada canal
            for nivel in range(niveles + 1): # para cada nivel de descomposicion
                stats_actual = estadisticos_normalizados_dwt[commands_mask, canal, nivel, :].reshape(signal_commands.shape[0], -1)
                tsne = TSNE(n_components = n_components,
                            perplexity = perplexity,
                            random_state = semilla,
                            max_iter = max_iter)
                tsne_resultado = tsne.fit_transform(stats_actual)
                ax[canal, nivel].scatter(tsne_resultado[:, 0], tsne_resultado[:, 1], c=labels_commands[:, 1], cmap = paleta, alpha = alpha[tipo], s = tam_puntos)
                ax[canal, nivel].set_xticks([])
                ax[canal, nivel].set_yticks([])
        fig.suptitle('t-SNE de Estadísticos DWT para Comandos', fontsize=16)
        for i in range(6):
            ax[i, 0].set_ylabel(f'Canal {nombres_canales[i]}')
        for i in range(niveles + 1):
            ax[0, i].set_title(f'Nivel {nombres_niveles[i]}')
        plt.tight_layout()
        plt.show()


