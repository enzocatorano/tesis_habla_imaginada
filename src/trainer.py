# src/trainer.py
"""
Entrenador y Evaluador mejorados.

Entrenador:
 - registro de scalars en TensorBoard (losses, acc)
 - opción histogram_freq: 0 = nunca, >0 = cada N epochs
 - collect epoch-wise arrays (train_losses, val_losses, val_accs)
 - escribe metrics_epochs.json en el run_specific_log_dir
 - devuelve un dict con las listas para que el script de entrenamiento las guarde donde quiera
 - save_model: controla si se guarda el mejor modelo durante entrenamiento

Evaluador:
 - evaluar, matriz de confusión, reporte y devolver métricas
"""
import os
import json
import copy
import datetime
from pathlib import Path
from tqdm import tqdm
from scipy.signal import butter, filtfilt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset, Subset

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class Entrenador:
    def __init__(self,
                 modelo: nn.Module,
                 optimizador: optim.Optimizer = None,
                 func_perdida: nn.Module = None,
                 device: str = None,
                 parada_temprana: int = None,
                 log_dir: str = 'runs',
                 histogram_freq: int = 0,
                 save_model: bool = True):
        """
        Args:
            modelo: nn.Module
            optimizador: torch optimizer (si None se crea Adam por defecto)
            func_perdida: loss function (obligatoria)
            device: 'cuda' or 'cpu' o None (auto detect)
            parada_temprana: patience en epocas (None para desactivar)
            log_dir: directorio raíz para TensorBoard (cada run tendrá un subdir timestamp)
            histogram_freq: 0 = nunca, >0 = cada N epochs loguea histogramas
            save_model: si True, guarda el mejor modelo durante entrenamiento; si False, no guarda
        """
        # device handling
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            if device == 'cuda' and not torch.cuda.is_available():
                print("Advertencia: CUDA no disponible, usando CPU.")
                self.device = 'cpu'
            else:
                self.device = device
        self.device = torch.device(self.device)
        print(f"[Entrenador] Usando dispositivo: {self.device}")

        if func_perdida is None:
            raise ValueError("Debe especificarse func_perdida (ej. nn.CrossEntropyLoss()).")

        if optimizador is None:
            optimizador = optim.Adam(modelo.parameters(), lr=1e-3)

        self.modelo = modelo.to(self.device)
        self.optimizador = optimizador
        self.func_perdida = func_perdida
        self.parada_temprana = parada_temprana
        self.save_model = save_model

        # logging
        self.base_log_dir = Path(log_dir)
        self.base_log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        # run-specific folder under base
        self.run_specific_log_dir = self.base_log_dir / f"run_{timestamp}"
        self.run_specific_log_dir.mkdir(parents=True, exist_ok=True)
        self.escritor = SummaryWriter(log_dir=str(self.run_specific_log_dir))

        # histogram frequency control
        self.histogram_freq = int(histogram_freq) if histogram_freq is not None else 0

    def _epoca_entrenamiento(self, cargador_entrenamiento: DataLoader, epoca: int):
        self.modelo.train()
        perdida_total = 0.0
        n_samples = 0
        for x, y in tqdm(cargador_entrenamiento, desc=f"Epoca {epoca} Entrenamiento", leave=False):
            x = x.to(self.device)
            y = y.to(self.device)
            self.optimizador.zero_grad()
            pred = self.modelo(x)
            loss = self.func_perdida(pred, y)
            loss.backward()
            self.optimizador.step()

            # aplicar max-norm si el modelo tiene ese método
            if hasattr(self.modelo, "apply_max_norm"):
                try:
                    self.modelo.apply_max_norm()
                except Exception as e:
                    # no fallar si la implementación del modelo es distinta
                    print(f"[Entrenador] Warning apply_max_norm fallo: {e}")

            bs = x.shape[0]
            perdida_total += float(loss.item()) * bs
            n_samples += bs
        perdida_promedio = perdida_total / n_samples if n_samples > 0 else 0.0
        self.escritor.add_scalar('Perdida/entrenamiento', perdida_promedio, epoca)
        return perdida_promedio

    def _epoca_validacion(self, cargador_validacion: DataLoader, epoca: int):
        self.modelo.eval()
        perdida_total = 0.0
        n_samples = 0
        correctas = 0
        with torch.no_grad():
            for x, y in tqdm(cargador_validacion, desc=f"Epoca {epoca} Validacion", leave=False):
                x = x.to(self.device)
                y = y.to(self.device)
                pred = self.modelo(x)
                loss = self.func_perdida(pred, y)
                bs = x.shape[0]
                perdida_total += float(loss.item()) * bs
                n_samples += bs
                # accuracy
                if pred.dim() > 1:
                    _, pred_idx = torch.max(pred, 1)
                    correctas += (pred_idx == y).sum().item()
        perdida_promedio = perdida_total / n_samples if n_samples > 0 else 0.0
        precision = correctas / n_samples if n_samples > 0 else 0.0
        self.escritor.add_scalar('Perdida/validacion', perdida_promedio, epoca)
        self.escritor.add_scalar('Precision/validacion', precision, epoca)
        return perdida_promedio, precision

    def _log_pesos_y_gradientes(self, epoca: int):
        # logs histograms of params & grads (costly) - call this only when needed
        for nombre, parametro in self.modelo.named_parameters():
            self.escritor.add_histogram(f'Parametros/{nombre}', parametro.data.detach().cpu(), epoca)
            if parametro.grad is not None:
                self.escritor.add_histogram(f'Gradientes/{nombre}', parametro.grad.detach().cpu(), epoca)

    def _revisar_formato_etiquetas(self, preds: torch.Tensor, etiquetas: torch.Tensor):
        # CrossEntropyLoss expects labels as LongTensor of shape [batch]
        if isinstance(self.func_perdida, nn.CrossEntropyLoss):
            if etiquetas.dim() > 1:
                raise ValueError("CrossEntropyLoss espera etiquetas como índices (vector 1D).")
            if etiquetas.dtype != torch.long:
                raise ValueError("CrossEntropyLoss espera etiquetas dtype=torch.long.")
        elif isinstance(self.func_perdida, nn.MSELoss):
            if etiquetas.shape != preds.shape:
                raise ValueError("MSELoss espera preds y etiquetas con misma shape.")

    def ajustar(self,
                cargador_entrenamiento: DataLoader,
                cargador_validacion: DataLoader = None,
                epocas: int = 100,
                nombre_modelo_salida: str = None,
                early_stop_patience: int = None):
        """
        Ejecuta el loop completo de entrenamiento y validación.
        Devuelve un dict con listas por época:
            {
              "train_losses": [...],
              "val_losses": [...],
              "val_accs": [...],
              "best_val_loss": float,
              "best_epoch": int,
              "n_epochs_run": int
            }
        Además guarda metrics_epochs.json dentro de run_specific_log_dir.
        
        El modelo solo se guarda si self.save_model es True Y se proporciona nombre_modelo_salida.
        """
        # quick format check with one batch
        x_batch, y_batch = next(iter(cargador_entrenamiento))
        self._revisar_formato_etiquetas(self.modelo(x_batch.to(self.device)), y_batch.to(self.device))

        train_losses = []
        val_losses = []
        val_accs = []
        best_val_loss = float('inf')
        best_epoch = -1
        epochs_no_improve = 0
        best_state = copy.deepcopy(self.modelo.state_dict())

        patience = early_stop_patience if early_stop_patience is not None else self.parada_temprana

        for ep in range(1, int(epocas) + 1):
            # entrenamiento
            train_loss = self._epoca_entrenamiento(cargador_entrenamiento, ep)
            train_losses.append(float(train_loss))

            # validación
            if cargador_validacion is not None:
                val_loss, val_acc = self._epoca_validacion(cargador_validacion, ep)
                val_losses.append(float(val_loss))
                val_accs.append(float(val_acc))

                # guardar mejor modelo según val_loss
                if val_loss < best_val_loss:
                    best_val_loss = float(val_loss)
                    best_epoch = ep
                    epochs_no_improve = 0
                    best_state = copy.deepcopy(self.modelo.state_dict())
                    
                    # Solo guardar si save_model está activado Y se proporcionó un nombre
                    if self.save_model and nombre_modelo_salida:
                        try:
                            torch.save(self.modelo.state_dict(), nombre_modelo_salida)
                            print(f"[Entrenador] Modelo guardado en {nombre_modelo_salida}")
                        except Exception as e:
                            print(f"[Entrenador] Warning: no se pudo guardar el modelo: {e}")
                else:
                    epochs_no_improve += 1

                # early stopping
                if patience is not None and epochs_no_improve >= int(patience):
                    print(f"[Entrenador] Parada temprana luego de epoca {ep} (patience={patience})")
                    break
            else:
                # si no hay validación, dejamos placeholders
                val_losses.append(None)
                val_accs.append(None)

            # log histograms condicionalmente
            if self.histogram_freq and (ep % self.histogram_freq == 0):
                self._log_pesos_y_gradientes(ep)

        # al finalizar: restaurar mejor modelo si hubo validación
        if cargador_validacion is not None and best_epoch >= 0:
            self.modelo.load_state_dict(best_state)

        # close writer
        self.escritor.close()

        # preparar dict serializable
        metrics = {
            "train_losses": [float(x) for x in train_losses],
            "val_losses": [float(x) if x is not None else None for x in val_losses],
            "val_accs": [float(x) if x is not None else None for x in val_accs],
            "best_val_loss": float(best_val_loss) if best_val_loss != float('inf') else None,
            "best_epoch": int(best_epoch) if best_epoch >= 0 else None,
            "n_epochs_run": len(train_losses),
            "model_saved": self.save_model and nombre_modelo_salida is not None
        }

        # escribir metrics_epochs.json en el run_specific_log_dir
        try:
            metrics_path = Path(self.run_specific_log_dir) / "metrics_epochs.json"
            with open(metrics_path, "w", encoding="utf8") as fh:
                json.dump(metrics, fh, indent=2)
            print(f"[Entrenador] Epoch metrics saved to {metrics_path}")
        except Exception as e:
            print(f"[Entrenador] Warning: no pude escribir metrics_epochs.json: {e}")

        return metrics

##########################################################################################################

class Evaluador:
    def __init__(self, modelo: nn.Module, device: str = None, clases: str = None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            if device == 'cuda' and not torch.cuda.is_available():
                print("Advertencia: CUDA no disponible, usando CPU.")
                self.device = 'cpu'
            else:
                self.device = device
        self.device = torch.device(self.device)
        self.modelo = modelo.to(self.device)
        self.clases = clases
        self.nombres_clases = None
        if self.clases:
            self._set_nombres_clases()

    def _set_nombres_clases(self):
        if self.clases == 'modalidad':
            self.nombres_clases = ['Imaginada', 'Pronunciada']
        elif self.clases == 'estimulo':
            self.nombres_clases = ['A','E','I','O','U','Arriba','Abajo','Adelante','Atras','Derecha','Izquierda']
        elif self.clases == 'artefacto':
            self.nombres_clases = ['Limpio','Parpadeo']
        elif self.clases == 'estimulo_binario':
            self.nombres_clases = ['Vocal','Comando']
        elif self.clases == 'estimulo_vocal':
            self.nombres_clases = ['A','E','I','O','U']
        elif self.clases == 'estimulo_comando':
            self.nombres_clases = ['Arriba','Abajo','Izquierda','Derecha','Adelante','Atras']

    def _inferir_clases(self, cargador):
        if self.clases:
            return
        dataset_obj = cargador.dataset
        while isinstance(dataset_obj, Subset):
            dataset_obj = dataset_obj.dataset
        if hasattr(dataset_obj, 'etiqueta'):
            self.clases = dataset_obj.etiqueta
            self._set_nombres_clases()

    def probar(self, dataloader: DataLoader):
        self.modelo.eval()
        preds_all = []
        y_all = []
        self._inferir_clases(dataloader)
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                out = self.modelo(x)
                if out.dim() > 1 and out.shape[1] > 1:
                    pred = torch.argmax(out, dim=1)
                else:
                    pred = (out > 0.5).long().squeeze()
                if y.dim() > 1:
                    true = torch.argmax(y, dim=1)
                else:
                    true = y
                preds_all.append(pred.cpu().numpy())
                y_all.append(true.cpu().numpy())
        y_all = np.concatenate(y_all)
        preds_all = np.concatenate(preds_all)
        return y_all, preds_all

    def matriz_confusion(self, dataloader, plot=True, titulo="Matriz de Confusión"):
        y_true, y_pred = self.probar(dataloader)
        cm = confusion_matrix(y_true, y_pred)
        if plot:
            plt.figure(figsize=(8,6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="cividis",
                        xticklabels=self.nombres_clases if self.nombres_clases else np.unique(y_true),
                        yticklabels=self.nombres_clases if self.nombres_clases else np.unique(y_true))
            plt.xlabel("Predicha"); plt.ylabel("Verdadera"); plt.title(titulo); plt.show()
        return cm

    def reporte(self, dataloader, retornar_metricas=False):
        y_true, y_pred = self.probar(dataloader)
        report = classification_report(y_true, y_pred, target_names=self.nombres_clases if self.nombres_clases else None, output_dict=retornar_metricas)
        print(classification_report(y_true, y_pred, target_names=self.nombres_clases if self.nombres_clases else None))
        if retornar_metricas:
            acc = accuracy_score(y_true, y_pred)
            return report, acc
        return None

############################################################################################################

def Augmentar(X_train, Y_train, X_val, Y_val, X_test, Y_test,
              window_duration=1.5, window_shift=0.5, fs=128,
              band_noise_factor_train=1/3, band_noise_factor_eval=1.0,
              fts_factor_train=1/3, fts_factor_eval=1.0,
              n_fts_versions=1, noise_magnitude_relative=0.025,
              seed=None, save_metadata=True, metadata_path=None,
              save_indices=True, original_train_indices=None,
              original_val_indices=None, original_test_indices=None):
    """
    Augmentación online de datos EEG con segmentación temporal y dos tipos de augmentación:
    ruido en bandas de frecuencia y Fourier Transform Surrogate (FTS).
    
    REPRODUCIBILIDAD GARANTIZADA: Si se usa la misma semilla, los resultados serán idénticos.
    
    Proceso completo:
    1. Segmentación temporal (sliding windows) - TODOS los sets
    2. Augmentación por ruido en bandas - según factores train/val/test
    3. Augmentación por FTS - según factores train/val/test
    4. Agregar etiquetas nuevas: ventana, banda afectada, FTS usado
    
    Args:
        X_train, X_val, X_test: arrays (n_trials, n_channels, n_timepoints)
        Y_train, Y_val, Y_test: arrays (n_trials, 3) con [modalidad, estímulo, artefacto]
        window_duration: duración de cada ventana en segundos (default 1.5s)
        window_shift: desplazamiento entre ventanas en segundos (default 0.5s)
        fs: frecuencia de muestreo en Hz (default 128)
        band_noise_factor_train: proporción [0-1] de datos train/val a augmentar con ruido
        band_noise_factor_eval: proporción [0-1] de datos test a augmentar con ruido
        fts_factor_train: proporción [0-1] de datos train/val a augmentar con FTS
        fts_factor_eval: proporción [0-1] de datos test a augmentar con FTS
        n_fts_versions: número de versiones FTS por trial seleccionado (default 3)
        noise_magnitude_relative: magnitud del ruido como fracción de la potencia de señal
        seed: semilla para reproducibilidad (None = no reproducible)
        save_metadata: si True, guarda JSON con detalles de augmentación
        metadata_path: ruta donde guardar metadata.json (si None, no guarda)
        save_indices: si True, incluye índices originales en metadata
        original_train_indices: índices originales de train (para metadata)
        original_val_indices: índices originales de val (para metadata)
        original_test_indices: índices originales de test (para metadata)
    
    Returns:
        X_train_aug, Y_train_aug: datos de entrenamiento augmentados
        X_val_aug, Y_val_aug: datos de validación augmentados
        X_test_aug, Y_test_aug: datos de test augmentados
        
        Y shape: (n_trials_aug, 6) donde las columnas son:
            [0] modalidad (0=imaginada, 1=pronunciada)
            [1] estímulo (1-11: A,E,I,O,U,arriba,abajo,izq,der,adelante,atrás)
            [2] artefacto (0=limpio, 1=parpadeo, 2=otro)
            [3] ventana temporal (0-N indicando posición de la ventana)
            [4] banda afectada:
                -1 = no seleccionado para augmentación de banda
                 0 = seleccionado pero no augmentado (original)
                 1-5 = banda ensuciada (delta, theta, alpha, beta, gamma)
            [5] FTS aplicado:
                -1 = no seleccionado para FTS
                 0 = seleccionado pero no augmentado (original)
                 1 = augmentado con FTS
    """
    
    # ============================================================
    # INICIALIZACIÓN DE GENERADOR ALEATORIO PARA REPRODUCIBILIDAD
    # ============================================================
    if seed is not None:
        rng = np.random.Generator(np.random.PCG64(seed))
    else:
        rng = np.random.default_rng()
    
    # Metadata container
    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "seed": seed,
        "reproducible": seed is not None
    }
    
    # ============================================================
    # PASO 1: SEGMENTACIÓN TEMPORAL (SLIDING WINDOWS)
    # ============================================================
    
    def segment_sliding_windows(X, Y, duration_samples, shift_samples):
        """
        Segmenta trials en ventanas deslizantes.
        
        Args:
            X: (n_trials, n_channels, n_timepoints)
            Y: (n_trials, n_labels)
            duration_samples: duración de ventana en muestras
            shift_samples: desplazamiento entre ventanas en muestras
        
        Returns:
            X_windows: (n_windows_total, n_channels, duration_samples)
            Y_windows: (n_windows_total, n_labels + 1) con etiqueta de ventana agregada
        """
        n_trials, n_channels, n_timepoints = X.shape
        
        # Calcular número de ventanas por trial
        n_windows_per_trial = int((n_timepoints - duration_samples) / shift_samples) + 1
        
        X_windows_list = []
        Y_windows_list = []
        
        for trial_idx in range(n_trials):
            trial_data = X[trial_idx]  # (n_channels, n_timepoints)
            trial_labels = Y[trial_idx]  # (n_labels,)
            
            # Extraer ventanas deslizantes
            for window_idx in range(n_windows_per_trial):
                start = window_idx * shift_samples
                end = start + duration_samples
                
                if end > n_timepoints:
                    break
                
                # Extraer ventana
                window = trial_data[:, start:end]  # (n_channels, duration_samples)
                X_windows_list.append(window)
                
                # Agregar etiqueta de ventana (columna 3)
                labels_with_window = np.append(trial_labels, window_idx)
                Y_windows_list.append(labels_with_window)
        
        X_windows = np.stack(X_windows_list, axis=0)
        Y_windows = np.stack(Y_windows_list, axis=0)
        
        return X_windows, Y_windows, n_windows_per_trial
    
    # Calcular parámetros de ventanas
    duration_samples = int(window_duration * fs)
    shift_samples = int(window_shift * fs)
    
    # Segmentar cada set
    X_train_seg, Y_train_seg, n_win_train = segment_sliding_windows(
        X_train, Y_train, duration_samples, shift_samples)
    X_val_seg, Y_val_seg, n_win_val = segment_sliding_windows(
        X_val, Y_val, duration_samples, shift_samples)
    X_test_seg, Y_test_seg, n_win_test = segment_sliding_windows(
        X_test, Y_test, duration_samples, shift_samples)
    
    # Metadata: window params
    metadata["window_params"] = {
        "duration_seconds": window_duration,
        "shift_seconds": window_shift,
        "duration_samples": duration_samples,
        "shift_samples": shift_samples,
        "n_windows_per_trial": n_win_train,
        "sampling_rate_hz": fs
    }
    
    # ============================================================
    # PASO 2: AUGMENTACIÓN POR RUIDO EN BANDAS
    # ============================================================
    
    # Definir bandas de frecuencia (Hz)
    bands = {
        1: (0.5, 4),      # delta
        2: (4, 8),      # theta
        3: (8, 12),     # alpha
        4: (12, 32),    # beta
        5: (32, 63.5),    # gamma
    }
    
    def add_band_noise(X, Y, factor, magnitude_rel, set_name, rng_state):
        """
        Augmenta con ruido gaussiano en bandas de frecuencia.
        
        Proceso:
        1. Selecciona aleatoriamente 'factor'% de los trials
        2. Divide los seleccionados en 5 grupos iguales
        3. A cada grupo le inyecta ruido en UNA banda específica
        4. El ruido es independiente por canal pero en la misma banda
        
        Args:
            X: (n_trials, n_channels, n_timepoints)
            Y: (n_trials, n_labels) debe tener 4 columnas
            factor: proporción [0-1] de trials a seleccionar
            magnitude_rel: magnitud del ruido relativa a potencia de señal
            set_name: nombre del set para metadata
            rng_state: generador aleatorio para reproducibilidad
        
        Returns:
            X_aug: trials originales + augmentados
            Y_aug: etiquetas con columna 4 agregada (banda afectada)
            stats: estadísticas para metadata
        """
        n_trials = X.shape[0]
        
        # Inicializar etiqueta de banda: -1 = no seleccionado
        band_labels = np.full(n_trials, -1, dtype=int)
        
        stats = {
            "applied": factor > 0,
            "factor": float(factor),
            "n_trials_total": int(n_trials),
            "n_selected": 0,
            "n_augmented": 0
        }
        
        if factor == 0:
            Y_with_band = np.column_stack([Y, band_labels])
            return X.copy(), Y_with_band, stats
        
        # Seleccionar aleatoriamente según factor
        n_to_augment = int(n_trials * factor)
        if n_to_augment == 0:
            Y_with_band = np.column_stack([Y, band_labels])
            return X.copy(), Y_with_band, stats
        
        selected_indices = rng_state.choice(n_trials, size=n_to_augment, replace=False)
        stats["n_selected"] = int(n_to_augment)
        
        # Marcar seleccionados como 0 (original sin augmentar aún)
        band_labels[selected_indices] = 0
        
        # Dividir seleccionados en 5 grupos para las 5 bandas
        n_per_band = n_to_augment // 5
        remainder = n_to_augment % 5
        
        # Mezclar índices seleccionados aleatoriamente
        shuffled_selected = rng_state.permutation(selected_indices)
        
        X_augmented_list = []
        Y_augmented_list = []
        
        # Diseñar filtros para cada banda (una sola vez)
        filters = {}
        nyquist = fs / 2
        for band_id, (low, high) in bands.items():
            low_norm = low / nyquist
            high_norm = min(high / nyquist, 0.99)
            
            if low_norm >= high_norm:
                high_norm = low_norm + 0.01
            
            b, a = butter(4, [low_norm, high_norm], btype='band')
            filters[band_id] = (b, a)
        
        # Procesar cada banda
        start_idx = 0
        for band_id in range(1, 6):
            # Distribuir remainder en las primeras bandas
            n_current_band = n_per_band + (1 if band_id <= remainder else 0)
            end_idx = start_idx + n_current_band
            
            if n_current_band == 0:
                continue
            
            band_indices = shuffled_selected[start_idx:end_idx]
            b, a = filters[band_id]
            
            # Aplicar ruido en esta banda
            for idx in band_indices:
                trial = X[idx].copy()  # (n_channels, n_timepoints)
                
                # Calcular potencia de la señal para normalizar ruido
                signal_power = np.std(trial)
                noise_std = signal_power * magnitude_rel
                
                trial_noisy = trial.copy()
                
                # Aplicar ruido independiente por canal pero en la misma banda
                for ch in range(trial.shape[0]):
                    # Generar ruido con el RNG (reproducible)
                    noise = rng_state.normal(0, noise_std, trial.shape[1])
                    
                    # Filtrar ruido en la banda específica
                    noise_filtered = filtfilt(b, a, noise)
                    
                    # Agregar a la señal
                    trial_noisy[ch] += noise_filtered
                
                X_augmented_list.append(trial_noisy)
                
                # Copiar etiquetas y marcar banda afectada
                labels_aug = Y[idx].copy()
                labels_aug = np.append(labels_aug, band_id)
                Y_augmented_list.append(labels_aug)
            
            start_idx = end_idx
        
        stats["n_augmented"] = len(X_augmented_list)
        
        # Combinar originales con augmentados
        Y_with_band = np.column_stack([Y, band_labels])
        
        if X_augmented_list:
            X_aug_array = np.stack(X_augmented_list, axis=0)
            Y_aug_array = np.stack(Y_augmented_list, axis=0)
            
            X_combined = np.vstack([X, X_aug_array])
            Y_combined = np.vstack([Y_with_band, Y_aug_array])
        else:
            X_combined = X.copy()
            Y_combined = Y_with_band
        
        return X_combined, Y_combined, stats
    
    # Aplicar augmentación de banda según factor
    X_train_band, Y_train_band, stats_band_train = add_band_noise(
        X_train_seg, Y_train_seg, band_noise_factor_train, 
        noise_magnitude_relative, "train", rng)
    
    X_val_band, Y_val_band, stats_band_val = add_band_noise(
        X_val_seg, Y_val_seg, band_noise_factor_train, 
        noise_magnitude_relative, "val", rng)
    
    X_test_band, Y_test_band, stats_band_test = add_band_noise(
        X_test_seg, Y_test_seg, band_noise_factor_eval, 
        noise_magnitude_relative, "test", rng)
    
    # Metadata: band noise
    metadata["band_noise"] = {
        "train_factor": float(band_noise_factor_train),
        "eval_factor": float(band_noise_factor_eval),
        "magnitude_relative": float(noise_magnitude_relative),
        "filter_type": "butterworth",
        "filter_order": 4,
        "filter_design": "scipy.signal.butter",
        "bands_hz": {
            "delta": list(bands[1]),
            "theta": list(bands[2]),
            "alpha": list(bands[3]),
            "beta": list(bands[4]),
            "gamma": list(bands[5])
        },
        "noise_characteristics": "independent_per_channel_same_band",
        "train": stats_band_train,
        "val": stats_band_val,
        "test": stats_band_test
    }
    
    # ============================================================
    # PASO 3: AUGMENTACIÓN POR FOURIER TRANSFORM SURROGATE (FTS)
    # ============================================================
    
    def add_fts_augmentation(X, Y, factor, n_versions, set_name, rng_state):
        """
        Augmenta con Fourier Transform Surrogate.
        
        CRÍTICO: Las mismas fases aleatorias se aplican a TODOS los canales del trial.
        
        Args:
            X: (n_trials, n_channels, n_timepoints)
            Y: (n_trials, n_labels) debe tener 5 columnas
            factor: proporción [0-1] de trials a seleccionar
            n_versions: número de versiones FTS por trial
            set_name: nombre del set para metadata
            rng_state: generador aleatorio
        
        Returns:
            X_aug: trials originales + FTS augmentados
            Y_aug: etiquetas con columna 5 agregada (FTS usado)
            stats: estadísticas para metadata
        """
        n_trials = X.shape[0]
        
        # Inicializar etiqueta FTS: -1 = no seleccionado
        fts_labels = np.full(n_trials, -1, dtype=int)
        
        stats = {
            "applied": factor > 0,
            "factor": float(factor),
            "n_versions": int(n_versions),
            "n_trials_total": int(n_trials),
            "n_selected": 0,
            "n_augmented": 0
        }
        
        if factor == 0:
            Y_with_fts = np.column_stack([Y, fts_labels])
            return X.copy(), Y_with_fts, stats
        
        # Seleccionar aleatoriamente
        n_to_augment = int(n_trials * factor)
        if n_to_augment == 0:
            Y_with_fts = np.column_stack([Y, fts_labels])
            return X.copy(), Y_with_fts, stats
        
        selected_indices = rng_state.choice(n_trials, size=n_to_augment, replace=False)
        stats["n_selected"] = int(n_to_augment)
        
        # Marcar seleccionados como 0
        fts_labels[selected_indices] = 0
        
        X_fts_list = []
        Y_fts_list = []
        
        # Generar versiones FTS
        for idx in selected_indices:
            trial = X[idx]  # (n_channels, n_timepoints)
            n_freqs = trial.shape[1]
            
            for version in range(n_versions):
                # CRÍTICO: Generar fases aleatorias UNA SOLA VEZ para este trial
                # Estas fases se aplicarán a TODOS los canales
                random_phase_shifts = rng_state.uniform(0, 2*np.pi, n_freqs)
                
                # Aplicar FTS con las MISMAS fases a cada canal
                trial_fts = np.zeros_like(trial)
                
                for ch in range(trial.shape[0]):
                    signal = trial[ch]
                    
                    # FFT
                    fft = np.fft.fft(signal)
                    amplitudes = np.abs(fft)
                    original_phases = np.angle(fft)
                    
                    # Sumar las fases aleatorias a las fases originales
                    # TODAS LOS CANALES USAN LAS MISMAS random_phase_shifts
                    new_phases = original_phases + random_phase_shifts
                    
                    # Reconstruir con nuevas fases
                    fft_surrogate = amplitudes * np.exp(1j * new_phases)
                    
                    # IFFT
                    signal_surrogate = np.fft.ifft(fft_surrogate).real
                    trial_fts[ch] = signal_surrogate
                
                X_fts_list.append(trial_fts)
                
                # Copiar etiquetas y marcar FTS=1
                labels_fts = Y[idx].copy()
                labels_fts = np.append(labels_fts, 1)
                Y_fts_list.append(labels_fts)
        
        stats["n_augmented"] = len(X_fts_list)
        
        # Combinar originales con FTS
        Y_with_fts = np.column_stack([Y, fts_labels])
        
        if X_fts_list:
            X_fts_array = np.stack(X_fts_list, axis=0)
            Y_fts_array = np.stack(Y_fts_list, axis=0)
            
            X_combined = np.vstack([X, X_fts_array])
            Y_combined = np.vstack([Y_with_fts, Y_fts_array])
        else:
            X_combined = X.copy()
            Y_combined = Y_with_fts
        
        return X_combined, Y_combined, stats
    
    # Aplicar FTS según factor
    X_train_aug, Y_train_aug, stats_fts_train = add_fts_augmentation(
        X_train_band, Y_train_band, fts_factor_train, n_fts_versions, "train", rng)
    
    X_val_aug, Y_val_aug, stats_fts_val = add_fts_augmentation(
        X_val_band, Y_val_band, fts_factor_train, n_fts_versions, "val", rng)
    
    X_test_aug, Y_test_aug, stats_fts_test = add_fts_augmentation(
        X_test_band, Y_test_band, fts_factor_eval, n_fts_versions, "test", rng)
    
    # Metadata: FTS
    metadata["fts"] = {
        "train_factor": float(fts_factor_train),
        "eval_factor": float(fts_factor_eval),
        "n_versions_per_trial": int(n_fts_versions),
        "phase_randomization": "uniform_0_2pi_added_to_original",
        "phase_application": "same_random_phases_across_all_channels",
        "train": stats_fts_train,
        "val": stats_fts_val,
        "test": stats_fts_test
    }
    
    # ============================================================
    # METADATA FINAL
    # ============================================================
    
    metadata["data_shapes"] = {
        "original_train": list(X_train.shape),
        "original_val": list(X_val.shape),
        "original_test": list(X_test.shape),
        "segmented_train": list(X_train_seg.shape),
        "segmented_val": list(X_val_seg.shape),
        "segmented_test": list(X_test_seg.shape),
        "final_train": list(X_train_aug.shape),
        "final_val": list(X_val_aug.shape),
        "final_test": list(X_test_aug.shape)
    }
    
    metadata["augmentation_factors"] = {
        "train": float(X_train_aug.shape[0] / X_train.shape[0]),
        "val": float(X_val_aug.shape[0] / X_val.shape[0]),
        "test": float(X_test_aug.shape[0] / X_test.shape[0])
    }
    
    # Agregar índices si se solicita
    if save_indices:
        metadata["indices"] = {
            "save_indices": True,
            "train_original_indices": original_train_indices.tolist() if original_train_indices is not None else None,
            "val_original_indices": original_val_indices.tolist() if original_val_indices is not None else None,
            "test_original_indices": original_test_indices.tolist() if original_test_indices is not None else None
        }
    else:
        metadata["indices"] = {"save_indices": False}
    
    # Guardar metadata si se solicita
    if save_metadata and metadata_path is not None:
        metadata_file = Path(metadata_path) / "augmentation_metadata.json"
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"[Augmentar] Metadata saved to: {metadata_file}")
    
    return X_train_aug, Y_train_aug, X_val_aug, Y_val_aug, X_test_aug, Y_test_aug