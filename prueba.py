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

#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
# prueba de actividad por banda y el impacto del bloque espectral
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

duracion = 4
fs = 256
t = np.linspace(0, duracion, duracion * fs, endpoint=False)

# funciones de filtro
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# filtro las bandas
delta = butter_bandpass_filter(np.random.randn(len(t)), 0.5, 4, fs, order=4)
theta = butter_bandpass_filter(np.random.randn(len(t)), 4, 8, fs, order=6)*0.6
alpha = butter_bandpass_filter(np.random.randn(len(t)), 8, 12, fs, order=6)*0.6**2
beta = butter_bandpass_filter(np.random.randn(len(t)), 12, 32, fs, order=6)*0.6**3
gamma = butter_bandpass_filter(np.random.randn(len(t)), 32, 64, fs, order=6)*0.6**4

bandas = [delta, theta, alpha, beta, gamma]

s = delta + theta + alpha + beta + gamma

# graficar la señal original y cada banda debajo, todo a la izquierda
# a la derecha cada fft
fig, ax = plt.subplots(6, 2, figsize=(16, 8))
ax[0, 0].plot(t, s)
ax[0, 0].set_title('Señal Original')
eje_freq = np.fft.fftfreq(len(s), 1/fs)
ax[0, 1].plot(eje_freq,np.abs(np.fft.fft(s)))
ax[0, 1].set_title('FFT Señal Original')
maximo_freq = np.max(np.abs(np.fft.fft(s)))
ax[0, 1].set_ylim(0, 1.1*maximo_freq)
#ax[0, 1].set_xlim(0, fs//2)
for i in range(5):
    ax[i + 1, 0].plot(t, bandas[i])
    ax[i + 1, 0].set_title(f'Banda {i + 1}')
    ax[i + 1, 1].plot(eje_freq, np.abs(np.fft.fft(bandas[i])))
    ax[i + 1, 1].set_title(f'FFT Banda {i + 1}')
    ax[i + 1, 1].set_ylim(0, 1.1*maximo_freq)
    #ax[i + 1, 1].set_xlim(0, fs//2)
plt.tight_layout()
plt.show()

# ahora quiero hacer una convolucion del tensor (bandas x tiempo)
# con un kernel que sea de dimension (bandas x 1)
# de esta forma (1, 0, 0, 1, 0) es decir que ve activaciones en delta y beta simultaneamente
kernel = np.array([1, 0, 0, 1, 0])

# convierto la lista de bandas en un array numpy
bandas_array = np.array(bandas) # (5, 1024)

# realizo la convolucion
# el resultado es un vector de (n_muestras)
convolucion = np.dot(kernel, bandas_array)

# graficar la convolucion
plt.figure(figsize=(17, 6))
plt.subplot(3,1,1)
plt.plot(t,delta)
plt.title('Delta')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.subplot(3,1,2)
plt.plot(t,beta)
plt.title('Beta')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.subplot(3,1,3)
plt.plot(t, convolucion)
plt.title('Convolución de Bandas con Kernel [1, 0, 0, 1, 0]')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.tight_layout()
plt.show()

# le quiero aplicar lReLu a convolucion
convolucion_lrelu = np.maximum(0 * convolucion, convolucion)
eje_freq = np.fft.fftfreq(len(convolucion), 1/fs)

# grafico la convolucion con y sin lrelu y a la derecha la fft de ambas
plt.figure(figsize=(16, 6))
plt.subplot(2,2,1)
plt.plot(t, convolucion)
plt.title('Convolución de Bandas con Kernel [1, 0, 0, 1, 0]')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.subplot(2,2,2)
plt.plot(t, convolucion_lrelu)
plt.title('Convolución de Bandas con Kernel [1, 0, 0, 1, 0] con lReLu')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.subplot(2,2,3)
plt.plot(eje_freq, np.abs(np.fft.fft(convolucion)))
plt.title('FFT Convolución de Bandas con Kernel [1, 0, 0, 1, 0]')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud')
plt.xlim(-1, fs//2)
plt.ylim(0, 200)
# marco lineas verticales en los bordes de banda
plt.axvline(x=4, color='b', linestyle='--', linewidth=0.8)
plt.axvline(x=8, color='b', linestyle='--', linewidth=0.8)
plt.axvline(x=12, color='b', linestyle='--', linewidth=0.8)
plt.axvline(x=32, color='b', linestyle='--', linewidth=0.8)
plt.subplot(2,2,4)
plt.plot(eje_freq, np.abs(np.fft.fft(convolucion_lrelu)))
plt.title('FFT Convolución de Bandas con Kernel [1, 0, 0, 1, 0] con lReLu')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud')
plt.xlim(-1, fs//2)
plt.ylim(0, 2*maximo_freq)
# marco lineas verticales en los bordes de banda
plt.axvline(x=4, color='b', linestyle='--', linewidth=0.8)
plt.axvline(x=8, color='b', linestyle='--', linewidth=0.8)
plt.axvline(x=12, color='b', linestyle='--', linewidth=0.8)
plt.axvline(x=32, color='b', linestyle='--', linewidth=0.8)
plt.tight_layout()
plt.show()

# calculo la potencia de banda normalizada para la señal con y sin lrelu
def band_power(data, fs, band, window_size=None):
    """
    Calcula la potencia de banda de una señal.
    data: señal de entrada
    fs: frecuencia de muestreo
    band: tupla (lowcut, highcut) para la banda de frecuencia
    window_size: tamaño de la ventana para el cálculo de la potencia (si es None, usa toda la señal)
    """
    if window_size is None:
        window_size = len(data)
    
    # Aplicar filtro de banda
    b, a = butter_bandpass(band[0], band[1], fs, order=4)
    filtered_data = lfilter(b, a, data)
    
    # Calcular potencia
    power = np.sum(filtered_data**2) / window_size
    return power

# Bandas de frecuencia para el cálculo de potencia
bands_to_analyze = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 12),
    'Beta': (12, 32),
    'Gamma': (32, 64)
}

# printeo las potencias de bandas normalizadas de ambas señales
print("Potencias de banda para la convolución original:")
total_power_conv = sum(band_power(convolucion, fs, band) for band in bands_to_analyze.values())
for name, band in bands_to_analyze.items():
    power = band_power(convolucion, fs, band)
    normalized_power = power / total_power_conv if total_power_conv > 0 else 0
    print(f"  {name}: {normalized_power:.4f}")

print("\nPotencias de banda para la convolución con LReLU:")
total_power_lrelu = sum(band_power(convolucion_lrelu, fs, band) for band in bands_to_analyze.values())
for name, band in bands_to_analyze.items():
    power = band_power(convolucion_lrelu, fs, band)
    normalized_power = power / total_power_lrelu if total_power_lrelu > 0 else 0
    print(f"  {name}: {normalized_power:.4f}")

#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
# aumentacion de datos

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample_poly, butter, lfilter, sosfiltfilt

# quiero cargar este archivo
ruta_archivo = 'data\processed\S01_EEG.npz'
# Cargar el archivo .npz
data = np.load(ruta_archivo)
eeg_data = data['data']
# datos y etiquetas
labels = eeg_data[:, -3:]
eeg_signals_flat = eeg_data[:, :-3]
# reestructuro la señal para que quede en formato (trials, canales, samples)
signal = eeg_signals_flat.reshape(eeg_signals_flat.shape[0], 6, -1)
# submuestro 1024 -> 128 Hz
resampled_signal = np.zeros((signal.shape[0], signal.shape[1], 128*4))
for i in range(signal.shape[0]):
    for j in range(signal.shape[1]):
        resampled_signal[i, j, :] = resample_poly(signal[i, j, :], 128, 1024)
signal = resampled_signal

# Parámetros (los mismos que antes)
fs = 128
window_duration_s = 1.5
step_duration_s = 0.5
window_len = int(window_duration_s * fs)
step_len = int(step_duration_s * fs)

n_trials, n_channels, n_samples = signal.shape

# 1. Crear una "vista" de todas las ventanas posibles a lo largo del eje de muestras (axis=-1)
#    La forma resultante es (trials, channels, num_possible_windows, window_len)
all_possible_windows = np.lib.stride_tricks.sliding_window_view(signal, window_shape=window_len, axis=-1)

# 2. Seleccionar las ventanas según el paso (step_len)
#    Hacemos un slicing en el eje de las ventanas (el tercero)
windowed_view = all_possible_windows[:, :, ::step_len, :]

# 3. Reordenar los ejes para agrupar las ventanas por trial
#    Cambiamos de (trials, channels, num_windows, window_len) a (trials, num_windows, channels, window_len)
windowed_view_transposed = np.transpose(windowed_view, (0, 2, 1, 3))

# 4. Colapsar los trials y las ventanas en una sola dimensión
#    Esto crea el array final con forma (total_windows, channels, window_len)
n_windows_per_trial = windowed_view_transposed.shape[1]
windowed_signal_np = windowed_view_transposed.reshape(-1, n_channels, window_len)

# 5. Actualizar las etiquetas para que coincidan con el nuevo número de muestras (ventanas)
#    Usamos np.repeat para duplicar cada etiqueta 'n_windows_per_trial' veces.
windowed_labels_np = np.repeat(labels, n_windows_per_trial, axis=0)


print("\n--- Resultados con NumPy (eficiente) ---")
print(f"Forma de la señal ventaneada: {windowed_signal_np.shape}")
print(f"Forma de las nuevas etiquetas: {windowed_labels_np.shape}")

# grafico en una sola figura las 6 particiones de un mismo trial original
# en la primer fila grafico la señal original
tiempo_original = np.linspace(0, signal.shape[2] / fs, signal.shape[2])
tiempo_segmentado = np.linspace(0, windowed_signal_np.shape[2] / fs, windowed_signal_np.shape[2])

plt.style.use('dark_background')
plt.figure(figsize=(6, 9))
plt.subplot(n_windows_per_trial + 1, 1, 1)
plt.plot(tiempo_original, signal[0, 0, :])
plt.title('Señal Original')
plt.ylim(-120, 120)
plt.xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
for i in range(1, n_windows_per_trial + 1):
    print(i)
    plt.subplot(n_windows_per_trial + 1, 1, i+1)
    plt.plot(tiempo_segmentado + 0.5*(i-1), windowed_signal_np[i-1, 0, :]) # Graficar el primer canal de cada ventana
    plt.title(f'Ventana {i-1} del primer trial')
    plt.ylim(-120, 120)
    plt.xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
plt.tight_layout()
plt.show()

# contaminacion de bandas de eeg
# tomo un trial de los nuevos, los resultantes de la segmentacion
ejemplo = windowed_signal_np[0, :, :]

def band_defs():
    return {
        'delta': (0.5, 4.0),
        'theta': (4.0, 8.0),
        'alpha': (8.0, 12.0),
        'beta' : (12.0, 32.0),
        'gamma': (32.0, 63.5)
    }

def bandpass_sosfiltfilt(x, low, high, fs, order=4, axis=1):
    """
    Filtrado bandpass usando Butterworth (SOS) + sosfiltfilt con reducción de orden automática
    si filtfilt falla por padlen. x tiene shape (n_channels, n_samples) y el filtrado se aplica
    sobre axis (por defecto axis=1 = tiempo).
    """
    cur_order = int(order)
    last_exception = None
    while cur_order >= 1:
        try:
            sos = butter(cur_order, [low, high], btype='band', fs=fs, output='sos')
            # sosfiltfilt aplica filtrado forward-backward (zero-phase)
            y = sosfiltfilt(sos, x, axis=axis)
            return y
        except Exception as e:
            # si falla por padlen u otra razón numérica, reducimos el orden y probamos de nuevo
            last_exception = e
            cur_order -= 1
    # Si llegamos acá, no pudimos filtrar con sosfiltfilt con órdenes <= original
    raise RuntimeError(f"No se pudo aplicar sosfiltfilt (intentadas órdenes hasta 1). Error final: {last_exception}")

def generate_band_limited_noise_butter(n_channels, n_samples, low, high, fs, order=4, seed=None):
    """
    Genera ruido blanco y lo filtra con el mismo bandpass butterworth (sosfiltfilt).
    """
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, 1, size=(n_channels, n_samples)).astype(float)
    # filtrar ruido en banda
    noise_filt = bandpass_sosfiltfilt(noise, low, high, fs, order=order, axis=1)
    return noise_filt

def compute_energy(x):
    # energía por canal
    return np.sum(x.astype(float)**2, axis=1)

def augment_band_noise_butter(trial, fs=128, factor=0.5, order=4, seed=None, per_channel_scale=True):
    """
    trial: np.array (n_channels, n_samples)
    factor: energía objetivo del ruido = energy(banda_clean) * factor
    order: orden inicial del Butterworth (int). El código intentará reducir si filtfilt falla.
    per_channel_scale: si True escala ruido por canal; si False usa escala global.
    Devuelve dict {band_name: info_dict} con info_dict['augmented'] la señal aumentada.
    """
    assert isinstance(trial, np.ndarray), "trial debe ser numpy array"
    assert trial.ndim == 2, "trial debe tener shape (n_channels, n_samples)"
    bands = band_defs()
    augmented = {}
    n_channels, n_samples = trial.shape
    for i, (bname, (low, high)) in enumerate(bands.items()):
        # 1) extraer componente limpia con butterworth + sosfiltfilt
        band_clean = bandpass_sosfiltfilt(trial, low, high, fs, order=order, axis=1)
        band_energy = compute_energy(band_clean)
        target_energy = band_energy * float(factor)
        
        # 2) generar ruido banda-limitado con el mismo filtro
        noise = generate_band_limited_noise_butter(n_channels, n_samples, low, high, fs, order=order, seed=(None if seed is None else seed + i))
        noise_energy = compute_energy(noise)
        
        # 3) escalar ruido para cumplir target_energy
        eps = 1e-12
        if per_channel_scale:
            scales = np.sqrt((target_energy + eps) / (noise_energy + eps))
            scales = np.clip(scales, 0.0, 1e6)
            noise_scaled = noise * scales[:, None]
        else:
            total_target = np.sum(target_energy)
            total_noise = np.sum(noise_energy) + eps
            global_scale = np.sqrt(total_target / total_noise)
            noise_scaled = noise * global_scale
        
        # 4) sumar ruido al trial original
        augmented_trial = trial.astype(float) + noise_scaled
        
        info = {
            'augmented': augmented_trial,
            'band_clean_energy_per_channel': band_energy,
            'noise_energy_before_scaling_per_channel': noise_energy,
            'noise_energy_after_scaling_per_channel': compute_energy(noise_scaled),
            'factor': factor,
            'band': (low, high),
            'butter_order_used': order
        }
        augmented[bname] = info
    return augmented

aug = augment_band_noise_butter(ejemplo, fs=128, factor=0.5, order=4, seed=123, per_channel_scale=True)
# Mostrar resumen de energías para comprobación
for band, info in aug.items():
    e_clean = info['band_clean_energy_per_channel']
    e_noise_before = info['noise_energy_before_scaling_per_channel']
    e_noise_after = info['noise_energy_after_scaling_per_channel']
    print(f"\nBanda: {band} {info['band']} (order {info['butter_order_used']})")
    for ch in range(e_clean.size):
        print(f" ch{ch:02d}: cleanE={e_clean[ch]:8.4f} | noise_before={e_noise_before[ch]:8.4f} | noise_after={e_noise_after[ch]:8.4f}")

# grafico una figura (5 + 1 aumentaciones, canales)
# la primer fila tiene los 6 canales originales
# las siguientes filas son cada canal con una banda afectada
fig, ax = plt.subplots(len(aug) + 1, 6, figsize=(16, 8))
# Graficar la señal original en la primera fila
for ch in range(6):
    print(ch)
    ax[0, ch].plot(tiempo_segmentado, ejemplo[ch, :])
    ax[0, ch].set_title(f'Original - Canal {ch}')
    ax[0, ch].set_xticks([])
    ax[0, ch].set_yticks([])
    ax[0, ch].set_ylim(-120, 120)
# Graficar las señales aumentadas en las filas siguientes
for i, (band_name, info) in enumerate(aug.items()):
    augmented_signal = info['augmented']
    for ch in range(6):
        ax[i + 1, ch].plot(tiempo_segmentado, augmented_signal[ch, :])
        ax[i + 1, ch].set_title(f'Aumentado {band_name} - Canal {ch}')
        ax[i + 1, ch].set_xticks([])
        ax[i + 1, ch].set_yticks([])
        ax[i + 1, ch].set_ylim(-120, 120)
for i in range(6):
    ax[5, i].set_xticks([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
plt.tight_layout()
plt.show()

# ahora grafico un canal al azar
# en la columna 0 grafico la señal original y debajo las 5 aumentaciones
# en la columna 1 la resta entre la señal modificada y la original
# en la columna 2 la fft de cada grafico
# en la columna 3 la fft de la resta
canal = np.random.randint(0, 6)
fig, ax = plt.subplots(6, 4, figsize=(12, 8))
# Graficar la señal
ax[0, 0].plot(tiempo_segmentado, ejemplo[canal, :])
ax[0, 0].set_title(f'Original - Canal {canal}')
ax[0, 0].set_xticks([])
ax[0, 0].set_ylim(-120, 120)
# Graficar la señal resta
ax[0, 1].plot(tiempo_segmentado, ejemplo[canal, :] - ejemplo[canal, :])
ax[0, 1].set_title(f'Resta Original - Canal {canal}')
ax[0, 1].set_xticks([])
ax[0, 1].set_ylim(-120, 120)
# Graficar la FFT de la señal original
eje_freq_segmentado = np.fft.fftfreq(len(ejemplo[canal, :]), 1/fs)
ax[0, 2].plot(eje_freq_segmentado, np.abs(np.fft.fft(ejemplo[canal, :])))
ax[0, 2].set_title(f'FFT Original - Canal {canal}')
ax[0, 2].set_xlim(0, fs//2)
ax[0, 2].set_ylim(0.8*np.min(np.abs(np.fft.fft(ejemplo[canal, :]))), 1.2 * np.max(np.abs(np.fft.fft(ejemplo[canal, :]))))
ax[0, 2].set_yscale('log')
# Graficar la fft de la resta
ax[0, 3].plot(eje_freq_segmentado, np.abs(np.fft.fft(ejemplo[canal, :] - ejemplo[canal, :])))
ax[0, 3].set_title(f'FFT Resta Original - Canal {canal}')
ax[0, 3].set_xlim(0, fs//2)
ax[0, 3].set_ylim(0.8*np.min(np.abs(np.fft.fft(ejemplo[canal, :]))), 1.2 * np.max(np.abs(np.fft.fft(ejemplo[canal, :]))))
ax[0, 3].set_yscale('log')
# Graficar las señales aumentadas y sus FFTs
for i, (band_name, info) in enumerate(aug.items()):
    augmented_signal = info['augmented']
    ax[i + 1, 0].plot(tiempo_segmentado, augmented_signal[canal, :])
    ax[i + 1, 0].set_title(f'Aumentado {band_name} - Canal {canal}')
    ax[i + 1, 0].set_xticks([])
    ax[i + 1, 0].set_ylim(-120, 120)
    
    ax[i + 1, 1].plot(tiempo_segmentado, augmented_signal[canal, :] - ejemplo[canal, :])
    ax[i + 1, 1].set_title(f'Resta {band_name} - Canal {canal}')
    ax[i + 1, 1].set_xticks([])
    ax[i + 1, 1].set_ylim(-120, 120)

    ax[i + 1, 2].plot(eje_freq_segmentado, np.abs(np.fft.fft(augmented_signal[canal, :])))
    ax[i + 1, 2].set_title(f'FFT Aumentado {band_name} - Canal {canal}')
    ax[i + 1, 2].set_xlim(0, fs//2)
    ax[i + 1, 2].set_ylim(0.8*np.min(np.abs(np.fft.fft(ejemplo[canal, :]))), 1.2 * np.max(np.abs(np.fft.fft(ejemplo[canal, :]))))
    ax[i + 1, 2].set_yscale('log')

    ax[i + 1, 3].plot(eje_freq_segmentado, np.abs(np.fft.fft(augmented_signal[canal, :] - ejemplo[canal, :])))
    ax[i + 1, 3].set_title(f'FFT Resta {band_name} - Canal {canal}')
    ax[i + 1, 3].set_xlim(0, fs//2)
    ax[i + 1, 3].set_ylim(0.8*np.min(np.abs(np.fft.fft(ejemplo[canal, :]))), 1.2 * np.max(np.abs(np.fft.fft(ejemplo[canal, :]))))
    ax[i + 1, 3].set_yscale('log')
plt.tight_layout()
plt.show()

# pruebo lo mismo pero cargando de los datos augmentados creados
ruta = 'data\preproc_aug_segm_gnperband_fts\S01_EEG_augmented.npz'
data = np.load(ruta)
x = data['data']
y = data['labels']
i = 0
tiempo_segmentado = np.linspace(0, x.shape[2] / 128, x.shape[2])
eje_freq_segmentado = np.fft.fftfreq(len(x[i, ch, :]), 1/fs)
# tomo uno al azar y lo grafico junto a su fft y un cuadro de texto con las etiquetas
#i = np.random.randint(0, x.shape[0])
plt.style.use('dark_background')
fig, ax = plt.subplots(6, 2, figsize=(10, 8))
for ch in range(6):
    ax[ch, 0].plot(tiempo_segmentado, x[i, ch, :])
    ax[ch, 0].set_title(f'Canal {ch}')
    ax[ch, 0].set_xticks([])
    ax[ch, 0].set_ylim(-120, 120)

    ax[ch, 1].plot(eje_freq_segmentado, np.abs(np.fft.fft(x[i, ch, :])))
    ax[ch, 1].set_title(f'FFT Canal {ch}')
    ax[ch, 1].set_xlim(0, fs//2)
    ax[ch, 1].set_ylim(0.8*np.min(np.abs(np.fft.fft(x[i, ch, :]))), 1.2 * np.max(np.abs(np.fft.fft(x[i, ch, :]))))
    ax[ch, 1].set_yscale('log')
# Añadir cuadro de texto con las etiquetas
text_labels = (f"Modalidad: {int(y[i, 0])} - Estímulo: {int(y[i, 1])} - Artefacto: {int(y[i, 2])} - Banda: {int(y[i, 3])} - FTS: {int(y[i, 4])}")
fig.text(0.5, 0.01, text_labels, ha='center', va='bottom', fontsize=12,
         bbox=dict(boxstyle="round,pad=0.3", fc="black", lw=1))
plt.tight_layout(rect=[0, 0.03, 1, 1]) # Ajustar layout para dejar espacio para el texto
plt.show()

# quiero ver las primeras etiquetas
for i in range(120):
    print(y[i])

############################################################################################################
############################################################################################################
############################################################################################################
# quiero ver todas las carpetas de datos que tengo
import os
import numpy as np

rutas = ['data\preproc',
         'data\preproc_aug_segm_gnperband_fts',
         'data\processed']

# veo los datos
for ruta in rutas:
    print(f"Archivos en {ruta}:")
    for root, dirs, files in os.walk(ruta):
        for file in files:
            if file.endswith('.npz'):
                file_path = os.path.join(root, file)
                try:
                    with np.load(file_path) as data:
                        keys = list(data.keys())
                        print(f"  {file}: {keys}")
                except Exception as e:
                    print(f"  Error al cargar {file}: {e}")
        
# cargo el sujeto 1 de cada ruta
s01_preproc = np.load('data/preproc/S01_EEG_augmented.npz')
s01_preproc_aug_segm_gnperband_fts = np.load('data/preproc_aug_segm_gnperband_fts/S01_EEG_augmented.npz')
s01_processed = np.load('data/processed/S01_EEG.npz')

print("\nContenido de S01_preproc.npz:")
for key in s01_preproc.keys():
    print(f"  {key}: {s01_preproc[key].shape}")

print("\nContenido de S01_preproc_aug_segm_gnperband_fts.npz:")
for key in s01_preproc_aug_segm_gnperband_fts.keys():
    print(f"  {key}: {s01_preproc_aug_segm_gnperband_fts[key].shape}")

print("\nContenido de S01_processed.npz:")
for key in s01_processed.keys():
    print(f"  {key}: {s01_processed[key].shape}")

############################################################################################################
############################################################################################################
############################################################################################################
# preprocesamiento segun Cooney

import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import FastICA
from math import gcd

# Parámetros (ajustables)
ORIG_SFREQ = 1024.0
FINAL_SFREQ = 128.0
TRIAL_SECONDS = 4.0
N_CHANNELS = 6
SAMPLES_PER_TRIAL = int(ORIG_SFREQ * TRIAL_SECONDS)  # 4096
FLAT_EEG_LEN = SAMPLES_PER_TRIAL * N_CHANNELS  # 24576
LABELS_PER_TRIAL = 3
FLAT_ROW_LEN = FLAT_EEG_LEN + LABELS_PER_TRIAL  # 24579
CHANNEL_NAMES = ["F3","F4","C3","C4","P3","P4"]
ICA_DOWNSAMPLE_FACTOR = 3  # replicando el paper
ICA_CORR_THRESHOLD = 0.8

plt.rcParams['figure.figsize'] = (10,4)

########################################################################
def find_main_array_in_npz(npz_path):
    with np.load(npz_path, allow_pickle=True) as z:
        keys = list(z.keys())
        if len(keys) == 0:
            raise ValueError("npz vacío")
        sizes = {k: np.size(z[k]) for k in keys}
        main_key = max(sizes, key=sizes.get)
        return z[main_key]

def reshape_flat_trials(flat_matrix):
    arr = np.asarray(flat_matrix)
    if arr.ndim == 1:
        L = arr.size
        if L % FLAT_ROW_LEN != 0:
            raise ValueError(f"Longitud {L} no divisible por {FLAT_ROW_LEN}.")
        n_trials = L // FLAT_ROW_LEN
        arr = arr.reshape(n_trials, FLAT_ROW_LEN)
    if arr.ndim == 2:
        n_trials, rowlen = arr.shape
        if rowlen == FLAT_ROW_LEN:
            eeg_part = arr[:, :FLAT_EEG_LEN]
            labels = arr[:, FLAT_EEG_LEN:FLAT_ROW_LEN]
        elif rowlen == FLAT_EEG_LEN:
            eeg_part = arr
            labels = None
        else:
            raise ValueError(f"fila len {rowlen} inesperada")
        eeg = eeg_part.reshape(n_trials, N_CHANNELS, SAMPLES_PER_TRIAL)
        eeg = np.transpose(eeg, (0, 2, 1))  # -> (n_trials, samples, channels)
        return eeg, labels
    raise ValueError("Formato no soportado")

def design_fir_bandpass(low, high, fs, numtaps=801):
    return signal.firwin(numtaps, [low, high], pass_zero=False, fs=fs)

def apply_fir_forward_backward(data_1d, fir):
    return signal.filtfilt(fir, 1.0, data_1d)

def filter_trials(eeg_trials, low, high, sfreq, numtaps=801):
    fir = design_fir_bandpass(low, high, sfreq, numtaps=numtaps)
    n_trials, n_samples, n_ch = eeg_trials.shape
    out = np.zeros_like(eeg_trials)
    for tr in range(n_trials):
        for ch in range(n_ch):
            out[tr, :, ch] = apply_fir_forward_backward(eeg_trials[tr, :, ch], fir)
    return out

def resample_trials(eeg_trials, orig_fs, new_fs):
    # resample_poly con factores reducidos
    up_fact = int(new_fs * 1000)
    down_fact = int(orig_fs * 1000)
    g = gcd(up_fact, down_fact)
    up_fact //= g
    down_fact //= g
    n_trials, n_samples, n_ch = eeg_trials.shape
    new_n = int(np.round(n_samples * new_fs / orig_fs))
    out = np.zeros((n_trials, new_n, n_ch), dtype=np.float32)
    for tr in range(n_trials):
        for ch in range(n_ch):
            out[tr, :, ch] = signal.resample_poly(eeg_trials[tr, :, ch], up_fact, down_fact)
    return out

def concat_trials_for_ica(eeg_trials):
    # devuelve array (n_samples_total, n_ch)
    n_trials, n_samples, n_ch = eeg_trials.shape
    concat = eeg_trials.reshape(n_trials * n_samples, n_ch)
    return concat

########################################################################
# Ruta al archivo real (ajustá)
ruta_referencia = r".\data\original\S01_EEG.npz"  # cambia por tu ruta local (u "data/original/S01_EEG.npz")
ruta_objetivo = r'.\data\original\S02_EEG.npz'
# referencia
arr_referencia = find_main_array_in_npz(ruta_referencia)
eeg_referencia, labels_referencia = reshape_flat_trials(arr_referencia)
# objetivo
arr_objetivo = find_main_array_in_npz(ruta_objetivo)
eeg_objetivo, labels_objetivo = reshape_flat_trials(arr_objetivo)

print("Forma EEG:", eeg_referencia.shape, eeg_objetivo.shape)  # (n_trials, samples, channels)

########################################################################
eeg_filt_referencia = filter_trials(eeg_referencia, 2.0, 40.0, ORIG_SFREQ, numtaps=801)
print("Filtrado OK — forma:", eeg_filt_referencia.shape)
eeg_filt_objetivo = filter_trials(eeg_objetivo, 2.0, 40.0, ORIG_SFREQ, numtaps=801)
print("Filtrado OK — forma:", eeg_filt_objetivo.shape)
# visualizá un trozo de un trial para comparar antes/despues (primer canal)
trial_idx = 1
ch = 0
plt.subplot(2,2,1)
plt.plot(eeg_objetivo[trial_idx, :2000, ch])
plt.title("Raw (canal {}) trial {}".format(ch,trial_idx))
# fft
freq_orig, Pxx_orig = signal.welch(eeg_objetivo[trial_idx, :, ch], fs=ORIG_SFREQ, nperseg=SAMPLES_PER_TRIAL)
plt.subplot(2,2,2)
plt.semilogy(freq_orig, Pxx_orig)
plt.title("PSD Raw")
plt.xlim([0,60])
freq_filt, Pxx_filt = signal.welch(eeg_filt_objetivo[trial_idx, :, ch], fs=ORIG_SFREQ, nperseg=SAMPLES_PER_TRIAL)
plt.subplot(2,2,3)
plt.plot(eeg_filt_objetivo[trial_idx, :2000, ch])
plt.title("Filtered 2-40 Hz")
plt.subplot(2,2,4)
plt.semilogy(freq_filt, Pxx_filt)
plt.title("PSD Filtered 2-40 Hz")
plt.xlim([0,60])
plt.tight_layout()
plt.show()

########################################################################
concat_referencia = concat_trials_for_ica(eeg_filt_referencia)  # (n_trials*n_samples, n_ch)
ica_ds_referencia = signal.resample_poly(concat_referencia, up=1, down=ICA_DOWNSAMPLE_FACTOR, axis=0)
ica_fs_referencia = ORIG_SFREQ / ICA_DOWNSAMPLE_FACTOR
print("Concat:", concat_referencia.shape, " -> downsampled:", ica_ds_referencia.shape, "fs ICA=", ica_fs_referencia)

concat_objetivo = concat_trials_for_ica(eeg_filt_objetivo)  # (n_trials*n_samples, n_ch)
ica_ds_objetivo = signal.resample_poly(concat_objetivo, up=1, down=ICA_DOWNSAMPLE_FACTOR, axis=0)
ica_fs_objetivo = ORIG_SFREQ / ICA_DOWNSAMPLE_FACTOR
print('Concat:', concat_objetivo.shape, '-> downsampled:', ica_ds_objetivo.shape, 'fs ICA=', ica_fs_objetivo)

ica_referencia = FastICA(n_components=N_CHANNELS, random_state=17, max_iter=2000)
S_referencia_ = ica_referencia.fit_transform(ica_ds_referencia)  # shape (n_samples_ds, n_components)
mixing_referencia_ = ica_referencia.mixing_  # shape (n_ch, n_comp)
print("ICA OK. mixing shape:", mixing_referencia_.shape)
ica_objetivo = FastICA(n_components=N_CHANNELS, random_state=17, max_iter=2000)
S_objetivo_ = ica_objetivo.fit_transform(ica_ds_objetivo)
mixing_objetivo_ = ica_objetivo.mixing_
print("ICA OK. mixing shape:", mixing_objetivo_.shape)

# topographies (n_components, n_ch)
topos_referencia = mixing_referencia_.T.copy()
topos_objetivo = mixing_objetivo_.T.copy()

# normalizar por norma L2 para visualización comparativa
topos_norm_referencia = topos_referencia / (np.linalg.norm(topos_referencia, axis=1, keepdims=True) + 1e-12)
topos_norm_objetivo = topos_objetivo / (np.linalg.norm(topos_objetivo, axis=1, keepdims=True) + 1e-12)

########################################################################
# Identificación de componente de parpadeo en el sujeto de referencia (S01)
frontal_idx = [0, 1]  # F3, F4 en CHANNEL_NAMES
frontal_power_referencia = np.sum(np.abs(topos_referencia[:, frontal_idx]), axis=1)
template_idx_referencia = int(np.argmax(frontal_power_referencia))
template_topo_referencia = topos_norm_referencia[template_idx_referencia]

print(f"S01: Componente plantilla seleccionada (blink): {template_idx_referencia}")
plt.figure(figsize=(6, 4))
plt.bar(range(N_CHANNELS), template_topo_referencia)
plt.xticks(range(N_CHANNELS), CHANNEL_NAMES, rotation=45)
plt.title("S01 - Plantilla (topografía normalizada) - posible blink")
plt.tight_layout()
plt.show()

########################################################################
# Correlacionar la plantilla de S01 con las topografías de S02
corrs_objetivo = np.array([np.corrcoef(template_topo_referencia, topos_norm_objetivo[i])[0, 1] for i in range(N_CHANNELS)])

print(f"Correlaciones de las componentes de S02 con la plantilla de S01 (blink):")
for i, c in enumerate(corrs_objetivo):
    print(f"  Comp {i}: corr = {c:.3f}")
blink_candidates_objetivo = np.where(np.abs(corrs_objetivo) >= ICA_CORR_THRESHOLD)[0].tolist()
print("S02: Candidatas (|corr| >= {:.2f}):".format(ICA_CORR_THRESHOLD), blink_candidates_objetivo)

# quiero graficar en una figura con 2 columnas
# en la izquierda necesito graficar una fraccion de las series temporales de cada componente
# en la derecha quiero las barras de cada componente
# quiero que las series temporales usen en 75% del ancho de la figura
posicion = 1000
extension = 5000
fig, axs = plt.subplots(N_CHANNELS+2, 2, figsize=(16, 9), gridspec_kw={'width_ratios': [5, 1]})
for i in range(N_CHANNELS):
    # Columna izquierda: Series temporales de las componentes
    axs[i, 0].plot(S_objetivo_[posicion:extension, i])  # Mostrar una porción de la serie temporal
    axs[i, 0].set_title(f'Componente {i} - Serie Temporal')
    axs[i, 0].set_ylabel('Amplitud')
    axs[i, 0].set_xticks([])
    # Columna derecha: Topografía de la componente (barras)
    axs[i, 1].bar(range(N_CHANNELS), topos_norm_objetivo[i])
    axs[i, 1].set_title(f'Componente {i} - Topografía')
    axs[i, 1].set_xticks(range(N_CHANNELS), CHANNEL_NAMES, rotation=45)
    axs[i, 1].set_ylim(-1, 1) # Normalizado
    axs[i, 1].set_xticks([])
    # Resaltar componentes candidatas a blink
    if i in blink_candidates_objetivo:
        axs[i, 0].patch.set_facecolor('red')
        axs[i, 0].patch.set_alpha(0.2)
        axs[i, 1].patch.set_facecolor('red')
        axs[i, 1].patch.set_alpha(0.2)
# fila N+1 a la izquierda pongo la serie temporal de plantilla
axs[N_CHANNELS, 0].plot(S_referencia_[posicion:extension, 2])
axs[N_CHANNELS, 0].set_title('S01 - Plantilla (topografía normalizada) - posible blink')
axs[N_CHANNELS, 0].set_ylabel('Amplitud')
axs[N_CHANNELS, 1].bar(range(N_CHANNELS), template_topo_referencia)
axs[N_CHANNELS, 1].set_title('S01 - Plantilla (topografía normalizada) - posible blink')
axs[N_CHANNELS, 1].set_xticks(range(N_CHANNELS), CHANNEL_NAMES, rotation=45)
axs[N_CHANNELS, 1].set_ylim(-1, 1)
# fila N+2 a la izquierda ponga la serie temporal original concatenada submuestreada
# suma de canales F3 + F4 dividido 2
axs[N_CHANNELS + 1, 0].plot(ica_ds_objetivo[posicion:extension, frontal_idx[0]] + ica_ds_objetivo[posicion:extension, frontal_idx[1]])
axs[N_CHANNELS + 1, 0].set_title('S02 - Suma F3+F4 (downsampled)')
axs[N_CHANNELS + 1, 0].set_ylabel('Amplitud')
axs[N_CHANNELS + 1, 0].set_xlabel('Muestras')
plt.tight_layout()
plt.show()

########################################################################
# Remoción de componentes en S02
S_ds_clean_objetivo = S_objetivo_.copy()
S_ds_clean_objetivo[:, blink_candidates_objetivo] = 0.0
recon_ds_objetivo = S_ds_clean_objetivo @ mixing_objetivo_.T  # (n_samples_ds, n_ch)

# volver a fs original por upsampling
recon_objetivo = signal.resample_poly(recon_ds_objetivo, up=ICA_DOWNSAMPLE_FACTOR, down=1, axis=0)  # (n_samples_orig, n_ch)

# reconstrucción completa (toda la señal concatenada) -> pasar a trials
n_trials_objetivo = eeg_filt_objetivo.shape[0]
recon_trials_objetivo = recon_objetivo[:FLAT_EEG_LEN].reshape(n_trials_objetivo, SAMPLES_PER_TRIAL, N_CHANNELS)

# comparar en trial ejemplo
tr = 1
ch = 0
t = np.arange(SAMPLES_PER_TRIAL)/ORIG_SFREQ
plt.figure(figsize=(12,6))
plt.subplot(2,1,1)
plt.plot(t[:2000], eeg_filt_objetivo[tr,:2000,ch], label="original (filtrado)")
plt.plot(t[:2000], recon_trials_objetivo[tr,:2000,ch], label="recon sin comps candidatas", alpha=0.8)
plt.legend(); plt.title(f"Trial {tr} Canal {CHANNEL_NAMES[ch]} — muestra parcial")
plt.subplot(2,1,2)
# PSD comparison (Welch)
f_orig, Pxx_orig = signal.welch(eeg_filt_objetivo[tr,:,ch], fs=ORIG_SFREQ, nperseg=4096)
f_rec, Pxx_rec = signal.welch(recon_trials_objetivo[tr,:,ch], fs=ORIG_SFREQ, nperseg=4096)
plt.semilogy(f_orig, Pxx_orig, label='original')
plt.semilogy(f_rec, Pxx_rec, label='recon sin comps')
plt.xlim([0,60]); plt.legend(); plt.title("PSD (Welch) comparativa")
plt.tight_layout()
plt.show()

########################################################################################################

# prueba del modelo ESMB_BR
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier

class ESMB_BR():
    def __init__(self, n_classes=5, learning_cycles=11, learning_rate=0.12, max_depth=3, semilla=42):
        """
        Inicializa el ensamble de clasificadores paralelos.
        
        Parámetros por defecto basados en el paper:
        - n_classes: 5 (para las vocales /a/, /e/, /i/, /o/, /u/)
        - learning_cycles: 11 ciclos de aprendizaje
        - learning_rate: 0.12
        """
        self.n_classes = n_classes
        self.learning_cycles = learning_cycles
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = semilla
        
        # Lista que contendrá los N clasificadores binarios independientes
        self.modules = []
        
        for i in range(self.n_classes):
            # Se instancia el modelo usando pérdida logística ('log_loss') 
            # para emular el método LogitBoost especificado por los autores.
            model = GradientBoostingClassifier(
                loss='log_loss', 
                n_estimators=self.learning_cycles,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state + i # Semillas distintas para cada módulo
            )
            self.modules.append(model)

    def fit(self, X, y):
        """
        Entrena los N clasificadores binarios de forma independiente.
        
        X: array-like de forma (n_muestras, n_caracteristicas)
        y: array-like de forma (n_muestras,) con etiquetas enteras (0 a n_classes-1)
        """
        # Se asegura de que y sea un array de numpy para facilitar el enmascarado
        y = np.array(y)
        
        for c in range(self.n_classes):
            # Crea un vector objetivo binario para la clase 'c': 
            # 1 si pertenece a la clase 'c', 0 en caso contrario
            y_binary = (y == c).astype(int)
            
            # Entrena el módulo correspondiente con este vector binario
            self.modules[c].fit(X, y_binary)
            
        return self

    def predict(self, X):
        """
        Realiza la predicción aplicando la estricta lógica combinadora "uno de cinco".
        Retorna la clase predicha, o -1 si el caso es descartado por ambigüedad.
        """
        n_samples = X.shape[0]
        # Matriz para almacenar las predicciones binarias de cada módulo (n_muestras, n_clases)
        predictions = np.zeros((n_samples, self.n_classes))
        
        # Recolectar las predicciones (0 o 1) de cada clasificador individual [cite: 258, 259]
        for c in range(self.n_classes):
            predictions[:, c] = self.modules[c].predict(X)
            
        final_predictions = np.full(n_samples, -1) # -1 representa una clase inválida/descartada
        
        # Contar cuántos clasificadores se activaron (dieron 1) por cada muestra
        activation_counts = np.sum(predictions, axis=1)
        
        # Lógica de "uno de cinco": solo es válido si exactamente un clasificador se activa [cite: 260, 328, 329]
        valid_mask = (activation_counts == 1)
        
        # Para las muestras válidas, asignamos el índice del clasificador que se activó
        final_predictions[valid_mask] = np.argmax(predictions[valid_mask], axis=1)
                
        return final_predictions


# Aseguramos que la clase ESMB_BR esté definida aquí arriba...

# 1. Generamos datos simulados (imitando la salida de wcm_preprocessing_dwt.py)
# Supongamos 200 trials, 36 features (6 canales x 6 modos WCM), y 5 clases (vocales)
np.random.seed(42)
X_wcm = np.random.rand(200, 36) 
y_etiquetas = np.random.randint(0, 5, 200)

# 2. Partición clásica en Train y Test
X_train, X_test, y_train, y_test = train_test_split(
    X_wcm, y_etiquetas, test_size=0.2, random_state=10
)

# 3. Instanciamos el modelo con los parámetros de Bolaños y Rufiner
modelo_br = ESMB_BR(
    n_classes=5, 
    learning_cycles=11, 
    learning_rate=0.12, 
    max_depth=3
)

# 4. Entrenamiento
print("Entrenando los 5 módulos paralelos...")
modelo_br.fit(X_train, y_train)

# 5. Inferencia / Predicción
print("Realizando predicciones...")
predicciones = modelo_br.predict(X_test)

# 6. Evaluación de la lógica estricta "uno de cinco"
print("\n--- Resultados de la Inferencia ---")
print(f"Etiquetas reales: {y_test[:15]}...")
print(f"Predicciones:     {predicciones[:15]}... (-1 significa descartado)")

# Calculamos estadísticas de los descartes
total_test = len(y_test)
descartados = np.sum(predicciones == -1)
validos_mask = predicciones != -1

print(f"\nTotal de ensayos en test: {total_test}")
print(f"Ensayos descartados por ambigüedad: {descartados} ({(descartados/total_test)*100:.1f}%)")

# Calculamos la precisión (Accuracy) SÓLO sobre los ensayos válidos
if np.sum(validos_mask) > 0:
    acc_valida = accuracy_score(y_test[validos_mask], predicciones[validos_mask])
    print(f"Precisión sobre ensayos válidos: {acc_valida*100:.2f}%")
else:
    print("Todos los ensayos fueron descartados por el modelo.")