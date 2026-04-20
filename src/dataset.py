# src/dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt

class OnlineEEGDataset(Dataset):
    def __init__(self, 
                 X_trials, 
                 Y_trials, 
                 fs=128,
                 window_duration=1.5, 
                 window_shift=0.5,
                 modo='train',
                 band_noise_factor=0.0,
                 fts_factor=0.0,
                 noise_magnitude_relative=0.025,
                 seed=None):
        
        # 1. OPTIMIZACIÓN CRÍTICA: Convertir a Tensores de PyTorch INMEDIATAMENTE.
        # Esto habilita el uso de Memoria Compartida (Shared Memory) entre workers,
        # evitando que Windows haga copias masivas de memoria.
        self.X = torch.tensor(X_trials, dtype=torch.float32)
        self.Y = torch.tensor(Y_trials, dtype=torch.long)
        
        self.fs = fs
        self.modo = modo
        
        self.duration_samples = int(window_duration * fs)
        self.shift_samples = int(window_shift * fs)
        
        n_timepoints = self.X.shape[2]
        self.n_windows_per_trial = int((n_timepoints - self.duration_samples) / self.shift_samples) + 1
        
        if self.n_windows_per_trial <= 0:
             raise ValueError("La duración de la ventana es mayor que el tamaño del trial.")
        
        self.n_trials = self.X.shape[0]
        self.total_windows = self.n_trials * self.n_windows_per_trial
        
        self.band_noise_factor = float(band_noise_factor) if modo == 'train' else 0.0
        self.fts_factor = float(fts_factor) if modo == 'train' else 0.0
        self.noise_magnitude_relative = noise_magnitude_relative
        
        self.seed = seed
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        
        self.nyquist = fs / 2
        self.bands = {
            1: (0.5, 4), 2: (4, 8), 3: (8, 12), 4: (12, 32), 5: (32, 63.5)
        }
        self.filters = {}
        
        # Solo calculamos filtros si realmente hay probabilidad de usarlos
        if self.band_noise_factor > 0:
            for band_id, (low, high) in self.bands.items():
                low_norm = low / self.nyquist
                high_norm = min(high / self.nyquist, 0.99)
                if low_norm >= high_norm: high_norm = low_norm + 0.01
                b, a = butter(4, [low_norm, high_norm], btype='band')
                self.filters[band_id] = (b, a)

    def __len__(self):
        return self.total_windows

    def _apply_band_noise(self, window_np, rng_state):
        band_id = rng_state.integers(1, 6)
        b, a = self.filters[band_id]
        
        window_noisy = window_np.copy()
        signal_power = np.std(window_np)
        noise_std = signal_power * self.noise_magnitude_relative
        
        for ch in range(window_np.shape[0]):
            noise = rng_state.normal(0, noise_std, window_np.shape[1])
            noise_filtered = filtfilt(b, a, noise)
            window_noisy[ch] += noise_filtered
            
        return window_noisy, band_id

    def _apply_fts(self, window_np, rng_state):
        n_freqs = window_np.shape[1]
        random_phase_shifts = rng_state.uniform(0, 2*np.pi, n_freqs)
        
        window_fts = np.zeros_like(window_np)
        for ch in range(window_np.shape[0]):
            fft = np.fft.fft(window_np[ch])
            amplitudes = np.abs(fft)
            original_phases = np.angle(fft)
            
            new_phases = original_phases + random_phase_shifts
            fft_surrogate = amplitudes * np.exp(1j * new_phases)
            
            window_fts[ch] = np.fft.ifft(fft_surrogate).real
            
        return window_fts

    def __getitem__(self, idx):
        trial_idx = idx // self.n_windows_per_trial
        window_idx = idx % self.n_windows_per_trial
        
        start_sample = window_idx * self.shift_samples
        end_sample = start_sample + self.duration_samples
        
        # Extracción ultrarrápida usando slicing de tensores
        x_tensor = self.X[trial_idx, :, start_sample:end_sample].clone()
        base_label = self.Y[trial_idx].clone()
        
        aug_labels = torch.tensor([window_idx, 0, 0], dtype=torch.long)
        
        # FAST PATH: Si las probabilidades son 0 (o estamos en validación/test), devolvemos inmediatamente.
        if self.modo != 'train' or (self.band_noise_factor == 0.0 and self.fts_factor == 0.0):
            final_label = torch.cat([base_label, aug_labels])
            return x_tensor, final_label

        # SLOW PATH: Entramos a la matemática de NumPy solo si los dados mandan
        aplica_ruido = self.rng.random() < self.band_noise_factor
        aplica_fts = self.rng.random() < self.fts_factor
        
        if aplica_ruido or aplica_fts:
            window_np = x_tensor.numpy() # Convertimos a numpy temporalmente
            
            if aplica_ruido:
                window_np, band_id = self._apply_band_noise(window_np, self.rng)
                aug_labels[1] = band_id
                
            if aplica_fts:
                window_np = self._apply_fts(window_np, self.rng)
                aug_labels[2] = 1
                
            x_tensor = torch.tensor(window_np, dtype=torch.float32)
            
        final_label = torch.cat([base_label, aug_labels])
        return x_tensor, final_label