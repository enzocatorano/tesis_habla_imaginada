# preprocesamiento segun Bolanos y Rufiner

import os
import numpy as np
import pywt
import json
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

########################################################################################
########################################################################################
# parametros de preprocesamiento

data_input_dir = 'data/preprocessed'
data_output_dir = 'data/preprocesamiento_segun_bolanos_rufiner'
sufijo_data_input = '_preprocessed.npz'

ondita = 'db4'

########################################################################################
########################################################################################
# funciones

# carga de datos
def load_data(subject, data_input_dir, sufijo_data_input):
    if subject < 10:
        fixed_subject = f'0{subject}'
    else:
        fixed_subject = subject
    data = np.load(os.path.join(data_input_dir, f'S{fixed_subject}{sufijo_data_input}'))
    x = data['x']
    y = data['y']
    return x, y

# descomposicion quasi-DWT
def quasi_dwt_decomposition(channel_data, wavelet_name='db4'):
    """
    Aplica una Quasi-DWT: Filtra usando la ondita Daubechies 4 y submuestrea 
    solo una vez para obtener vectores de igual longitud (N/2).
    
    Asume una longitud de entrada de 512 muestras. El uso de mode='periodization' 
    garantiza que la salida sea exactamente de 256 muestras, eliminando la 
    necesidad de padding manual.
    """
    # 1. Primer Nivel con DWT estándar (Submuestreo por 2)
    # cA: Aproximación (0-32 Hz), cD: Detalle (32-64 Hz)
    # mode='periodization' evita el agregado de muestras extra en los bordes.
    cA, cD = pywt.dwt(channel_data, wavelet_name, mode='periodization')
    
    # La banda Gamma ya está lista y submuestreada a N/2 (256 muestras)
    gamma = cD  # 32 - 64 Hz
    
    # 2. Descomposición de la rama cA (0-32 Hz) SIN submuestrear más.
    # Usamos SWT (Stationary Wavelet Transform) para mantener la longitud N/2.
    # Como la entrada tiene 256 muestras (múltiplo de 8), no requiere padding.
    coeffs = pywt.swt(cA, wavelet_name, level=3, start_level=0)
    
    # Extraer los coeficientes de la SWT (el orden devuelto es desde el nivel más profundo)
    cA3_swt, cD3_swt = coeffs[0]  # Nivel 3: 0-4 Hz (Delta) y 4-8 Hz (Theta)
    cA2_swt, cD2_swt = coeffs[1]  # Nivel 2: 0-8 Hz y 8-16 Hz
    cA1_swt, cD1_swt = coeffs[2]  # Nivel 1: 0-16 Hz y 16-32 Hz (Beta)
    
    delta = cA3_swt
    theta = cD3_swt
    beta = cD1_swt
    
    # 3. Separar la banda de 8-16 Hz (cD2_swt) en Alpha (8-12 Hz) y Sigma (12-16 Hz)
    # Aplicamos un nivel más de SWT específicamente a este nodo
    coeffs_alpha_sigma = pywt.swt(cD2_swt, wavelet_name, level=1, start_level=0)
    alpha = coeffs_alpha_sigma[0][0]  # Aproximación (mitad inferior: 8-12 Hz)
    sigma = coeffs_alpha_sigma[0][1]  # Detalle (mitad superior: 12-16 Hz)
    
    # Matriz A (6 bandas x N/2 muestras)
    A = np.vstack([delta, theta, alpha, sigma, beta, gamma])
    return A

# autovalores del cuadrado
def autovalores_del_cuadrado(A):
    cuadrado = A @ A.T
    autovalores = np.linalg.eigvalsh(cuadrado)
    return autovalores

# aplanar trial
def aplanar_trial(trial):
    '''
    Toma un dato de forma (canales, bins)
    y devuelve uno de forma (canales x autovalores)
    utilizando la descomposicion quasi-DWT y autovalores del cuadrado.
    '''
    aux = np.array([])
    for i in range(trial.shape[0]):
        A = quasi_dwt_decomposition(trial[i])
        autovalores = autovalores_del_cuadrado(A)
        aux = np.append(aux, autovalores)
    return aux

# json con detalles
def save_details(data_output_dir):
    details = {}
    details['data_input_dir'] = data_input_dir
    details['data_output_dir'] = data_output_dir
    details['subjects_processed'] = 15
    details['ondita'] = ondita
    details['input_structure'] = '(trials, 6 channels, 4seg x 128Hz)'
    details['output_structure'] = '(trials, 6 channels x 6 autovalores)'
    details['description'] = 'Preprocesamiento segun Bolanos y Rufiner'
    with open(os.path.join(data_output_dir, 'details.json'), 'w') as f:
        json.dump(details, f)

########################################################################################
########################################################################################
# main
if __name__ == '__main__':
    for subject in range(1, 16): # recorre los 15 sujetos
        x, y = load_data(subject, data_input_dir, sufijo_data_input)
        x_new = np.zeros((x.shape[0], x.shape[1]*6))
        for trial in range(x.shape[0]):
            x_new[trial] = aplanar_trial(x[trial])
        if subject < 10:
            fixed_subject = f'0{subject}'
        else:
            fixed_subject = subject
        if not os.path.exists(data_output_dir):
            os.makedirs(data_output_dir)
        np.savez(os.path.join(data_output_dir, f'S{fixed_subject}_preprocessed.npz'), x=x_new, y=y)
        print(f'Subject {subject} processed.')
    save_details(data_output_dir)
    print('All subjects processed.')
    print('Check the output directory for the preprocessed data.')
