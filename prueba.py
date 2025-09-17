import numpy as np
from pathlib import Path

subj_file = Path("features/preproc_dwt_L5_db4/per_subject/S01_EEG_features.npy")
feat = np.load(subj_file)
print("Shape features:", feat.shape)  
# Esperado: (n_trials, n_channels, L+1, n_stats) → (n_trials, 6, 6, 6) si L=5 y n_stats=6

flat = np.load(subj_file.parent / "S01_EEG_features_flat.npy")
print("Shape flat:", flat.shape)
# Esperado: (n_trials, n_channels*(L+1)*n_stats) → (n_trials, 6*6*6=216)

print("Min, max, mean of features:", feat.min(), feat.max(), feat.mean())
print("Any NaN?", np.isnan(feat).any())

from scipy.signal import resample_poly

sig_orig = np.random.randn(1024)  # reemplaza con un canal real
sig_rs = resample_poly(sig_orig, 128, 1024)
print("Original length:", len(sig_orig), "Resampled length:", len(sig_rs))
# Debe ser aproximadamente len(sig_orig) * 128/1024 = 128

import pywt
coeffs = pywt.wavedec(sig_orig, 'db4', level=5)
print("DWT coeff lengths:", [len(c) for c in coeffs])

labels = np.load(subj_file.parent / "S01_EEG_labels.npy")
task = np.load(subj_file.parent / "S01_EEG_task.npy")

print("Labels shape:", labels.shape)
print("Task unique values:", np.unique(task))
# Debe mostrar [0,1] solo si tu mapping es correcto

import matplotlib.pyplot as plt
plt.figure()
plt.plot(feat[117,0,:,0], label="cA5")  # aproximación del nivel más alto
plt.title("Primer trial, canal 0, coeficiente cA5")
plt.legend()
plt.show()

import seaborn as sns
sns.histplot(flat[:,0], bins=50)  # distribución de la primera feature

X = np.load("features/preproc_dwt_L5_db4/X.npy")
y = np.load("features/preproc_dwt_L5_db4/y.npy")
task = np.load("features/preproc_dwt_L5_db4/task.npy")

print("X shape:", X.shape)
print("y shape:", y.shape)
print("task shape:", task.shape)
print("X min/max:", X.min(), X.max())