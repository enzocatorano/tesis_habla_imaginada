# src/models.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import List, Optional, Callable
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier

####################################################################################################################################

class MLP(nn.Module):
    """
    MLP configurable.
    arq: lista de enteros [in_dim, hidden1, ..., out_dim]
    func_act: 'relu','sigmoid','tanh','leakyrelu' o lista de strings por capa (len = len(arq)-1)
    usar_batch_norm: aplicar BatchNorm1d en capas ocultas
    dropout: float in [0,1] o None
    metodo_init_pesos: función que acepta tensor (ej. nn.init.xavier_uniform_)
    semilla: int
    activate_last: bool (si True aplica la activación también en la última capa)
    """
    def __init__(self,
                 arq: List[int],
                 func_act='relu',
                 usar_batch_norm: bool = True,
                 dropout: Optional[float] = None,
                 metodo_init_pesos: Optional[Callable] = None,
                 semilla: Optional[int] = None,
                 activate_last: bool = False):
        super().__init__()

        import torch
        # reproducibilidad básica
        if semilla is not None:
            torch.manual_seed(int(semilla))
            torch.cuda.manual_seed_all(int(semilla))
            # para determinismo (puede reducir throughput)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            self.semilla = int(semilla)

        # preparar lista de activaciones
        if isinstance(func_act, str):
            func_list = [func_act] * (len(arq) - 1)
        elif isinstance(func_act, list) and len(func_act) == len(arq) - 1:
            func_list = func_act
        else:
            raise ValueError('func_act debe ser str o lista de longitud len(arq)-1')

        # si activate_last == False, forzamos identidad en la ultima (no aplicar activación)
        if not activate_last:
            func_list = list(func_list)
            func_list[-1] = 'identity'

        self.func_act = self._dar_activaciones(func_list)

        # dropout validation
        if dropout is not None and (dropout < 0 or dropout > 1):
            raise ValueError('dropout debe estar entre 0 y 1.')
        self.use_batch_norm = bool(usar_batch_norm)
        self.dropout_rate = float(dropout) if dropout is not None else 0.0

        # construir bloques con nombres
        bloques = OrderedDict()
        n_layers = len(arq) - 1
        for i in range(n_layers):
            in_f, out_f = int(arq[i]), int(arq[i+1])
            bloques[f"linear{i}"] = nn.Linear(in_f, out_f)
            # batchnorm solo en capas ocultas
            if self.use_batch_norm and i < n_layers - 1:
                bloques[f"batchnorm{i}"] = nn.BatchNorm1d(out_f)
            # activación (puede ser nn.Identity())
            bloques[f"act{i}"] = self.func_act[i]
            # dropout solo en capas ocultas
            if (self.dropout_rate and self.dropout_rate > 0) and i < n_layers - 1:
                bloques[f"dropout{i}"] = nn.Dropout(self.dropout_rate)

        self.estructura_total = nn.Sequential(bloques)

        # inicializacion de pesos
        if metodo_init_pesos is not None:
            self._inicializar_pesos(metodo_init_pesos)

    def _dar_activaciones(self, func_list):
        acts = []
        for a in func_list:
            if a is None or a == 'identity':
                acts.append(nn.Identity())
                continue
            a_low = a.lower()
            if a_low == 'relu':
                acts.append(nn.ReLU())
            elif a_low == 'sigmoid':
                acts.append(nn.Sigmoid())
            elif a_low == 'tanh':
                acts.append(nn.Tanh())
            elif a_low == 'leakyrelu':
                acts.append(nn.LeakyReLU())
            else:
                raise ValueError("Activación debe ser 'relu','sigmoid','tanh','leakyrelu' o 'identity'.")
        return acts

    def _inicializar_pesos(self, metodo_init_pesos):
        for mod in self.estructura_total:
            if isinstance(mod, nn.Linear):
                metodo_init_pesos(mod.weight)
                if mod.bias is not None:
                    nn.init.normal_(mod.bias, mean=0.0, std=0.01)

    def forward(self, x):
        return self.estructura_total(x)

    def __str__(self):
        s = "MLP architecture:\n"
        for i, m in enumerate(self.estructura_total):
            s += f"[{i}] {m}\n"
        return s

####################################################################################################################################

class EEGNet(nn.Module):
    """
    EEGNet (PyTorch). Diseño inspirado en Lawhern et al.
    Input: (batch, C, T) where C = n_channels, T = n_timepoints.
    This implementation uses conv2d with input reshaped to (batch, 1, C, T).

    Key args (defaults approximate the original paper's EEGNet-8,2 for fs=128):
      - in_ch: number of EEG channels (C)
      - n_classes: number of output classes (e.g. 5 or 6)
      - F1: number of temporal filters (default 8)
      - D: depth multiplier (spatial filters per temporal filter) (default 2)
      - F2: number of pointwise filters after separable conv (default = F1 * D)
      - kernel_length: temporal kernel length for first temporal conv (L_t) default 64
      - separable_kernel_length: temporal kernel in separable block (L_s) default 16
      - pool_time1: pooling factor after first block (default 4)
      - pool_time2: pooling factor after separable block (default 8)
      - dropout_prob: dropout probability (default 0.5)
      - hidden_units: int or None. If int, add a Dense hidden layer between flatten and final FC.
      - max_norm_spatial: max-norm constraint for spatial conv filters (default 1.0)
      - max_norm_dense: max-norm for final dense layer weights (default 0.25)
      - semilla: int or None.
    """
    def __init__(self,
                 in_ch: int,
                 n_classes: int,
                 T: int = 512,
                 F1: int = 8,
                 D: int = 2,
                 F2: int | None = None,
                 kernel_length: int = 64,
                 separable_kernel_length: int = 16,
                 pool_time1: int = 4,
                 pool_time2: int = 8,
                 dropout_prob: float = 0.25,
                 hidden_units: int | None = None,
                 max_norm_spatial: float = 1.0,
                 max_norm_dense: float = 0.25,
                 semilla=None):
        super().__init__()

        if semilla is not None:
            torch.manual_seed(int(semilla))
            torch.cuda.manual_seed_all(int(semilla))
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if F2 is None:
            F2 = F1 * D

        self.in_ch = in_ch
        self.n_classes = n_classes
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.kernel_length = kernel_length
        self.separable_kernel_length = separable_kernel_length
        self.pool_time1 = pool_time1
        self.pool_time2 = pool_time2
        self.dropout_prob = dropout_prob
        self.hidden_units = hidden_units
        self.max_norm_spatial = max_norm_spatial
        self.max_norm_dense = max_norm_dense

        # ---------------------
        # Block 1: Temporal + Spatial
        # ---------------------
        self.conv_temporal = nn.Conv2d(1, self.F1, (1, self.kernel_length),
                                       padding=(0, self.kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(self.F1, eps=1e-3, momentum=0.1, affine=True)

        self.depthwise_spatial = nn.Conv2d(self.F1, self.F1 * self.D, (self.in_ch, 1),
                                           groups=self.F1, bias=False)
        self.bn_depth = nn.BatchNorm2d(self.F1 * self.D, eps=1e-3, momentum=0.1, affine=True)
        
        self.pool1 = nn.AvgPool2d((1, self.pool_time1))
        self.dropout1 = nn.Dropout(p=self.dropout_prob)
        self.elu = nn.ELU()

        # ---------------------
        # Block 2: Separable
        # ---------------------
        self.depthwise_time = nn.Conv2d(self.F1 * self.D, self.F1 * self.D, 
                                        (1, self.separable_kernel_length),
                                        padding=(0, self.separable_kernel_length // 2),
                                        groups=self.F1 * self.D, bias=False)
        self.pointwise = nn.Conv2d(self.F1 * self.D, self.F2, (1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(self.F2)
        
        self.pool2 = nn.AvgPool2d((1, self.pool_time2))
        self.dropout2 = nn.Dropout(p=self.dropout_prob)

        # ---------------------
        # Classifier: Cálculo de dimensión de entrada
        # ---------------------
        # T_out tras las dos operaciones de pooling (AvgPool2d)
        t_out = T // self.pool_time1
        t_out = t_out // self.pool_time2
        self.flattened_dim = self.F2 * t_out

        if self.hidden_units is None:
            self.final_fc = nn.Linear(self.flattened_dim, self.n_classes)
            self.hidden_fc = None
        else:
            self.hidden_fc = nn.Linear(self.flattened_dim, self.hidden_units)
            self.final_fc = nn.Linear(self.hidden_units, self.n_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        # Input: (batch, C, T) -> Reshape: (batch, 1, C, T)
        x = x.unsqueeze(1)

        # Block 1
        x = self.conv_temporal(x)
        x = self.bn1(x)
        x = self.depthwise_spatial(x)
        x = self.bn_depth(x)
        x = self.elu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Block 2
        x = self.depthwise_time(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Classifier
        if self.hidden_fc is not None:
            x = self.hidden_fc(x)
            x = F.elu(x)
        
        logits = self.final_fc(x)
        return logits

    def apply_max_norm(self):
        # Spatial conv max-norm
        w = self.depthwise_spatial.weight.data
        norm = w.view(w.size(0), -1).norm(2, dim=1, keepdim=True).view(w.size(0), 1, 1, 1)
        desired = torch.clamp(norm, max=self.max_norm_spatial)
        self.depthwise_spatial.weight.data *= (desired / (1e-8 + norm))

        # Final dense max-norm
        if self.final_fc is not None:
            w = self.final_fc.weight.data
            norm = w.norm(2, dim=1, keepdim=True)
            desired = torch.clamp(norm, max=self.max_norm_dense)
            self.final_fc.weight.data *= (desired / (1e-8 + norm))

    def summary(self, T_example: int):
        """
        Utility: print a brief summary for a given number of time samples T_example.
        Useful to inspect final flatten size.
        """
        device = next(self.parameters()).device
        x = torch.zeros((1, self.in_ch, T_example), device=device)
        with torch.no_grad():
            s = self.forward(x)
        print(f"Example forward with T={T_example} produced logits shape: {s.shape}")

################################################################################################

class ShallowConvNet(nn.Module):
    def __init__(
        self,
        n_canales: int,
        n_clases: int,
        n_samples: int = 512,
        n_filtros_temporales: int = 40,
        longitud_kernel_temporal: int = 25,
        pool_size: int = 75,
        pool_stride: int = 15,
        dropout: float = 0.5):

        super().__init__()

        self.temporal_block = nn.Sequential(
            nn.Conv2d(1, n_filtros_temporales, kernel_size=(1, longitud_kernel_temporal), bias=False),
            nn.Conv2d(n_filtros_temporales, n_filtros_temporales, kernel_size=(n_canales, 1), bias=False),
            nn.BatchNorm2d(n_filtros_temporales),
        )
        self.pooling_block = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, pool_size), stride=(1, pool_stride)),
            nn.Dropout(dropout)
        )
        # Calculamos la dimensión de salida para el clasificador
        out_dim = self._get_final_flattened_size(n_canales, n_samples)
        self.clasificador = nn.Linear(out_dim, n_clases)

    def _get_final_flattened_size(self, n_ch, n_s):
        with torch.no_grad():
            x = torch.zeros(1, 1, n_ch, n_s)
            x = self.temporal_block(x)
            # Simulación de la no linealidad square
            x = x**2
            x = self.pooling_block(x)
            return x.numel()

    def forward(self, x):
        if x.ndim == 3: x = x.unsqueeze(1) # (B, 1, C, T)
        x = self.temporal_block(x)
        x = x**2
        x = self.pooling_block(x)
        # Log-activation (estilo FBCSP)
        x = torch.log(torch.clamp(x, min=1e-6))
        x = x.view(x.size(0), -1)
        return self.clasificador(x)

###########################################################################################

class DeepConvNet(nn.Module):
    def __init__(self,
                 n_canales: int,
                 n_clases: int,
                 n_samples: int = 512,
                 dropout: float = 0.5):
         
        super().__init__()

        # Bloque 1: Temporal + Espacial (Siguiendo a Cooney 2020)
        self.bloque1 = nn.Sequential(
            nn.Conv2d(1, 15, kernel_size=(1, 10), bias=False), # Kernel temporal inicial
            nn.Conv2d(15, 15, kernel_size=(n_canales, 1), bias=False),
            nn.BatchNorm2d(15),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(dropout)
        )
        # Bloques Profundos (Cooney usa 3 bloques idénticos)
        self.bloque2 = self._make_block(15, 30, dropout)
        self.bloque3 = self._make_block(30, 60, dropout)
        self.bloque4 = self._make_block(60, 120, dropout)

        out_dim = self._get_final_flattened_size(n_canales, n_samples)
        self.clasificador = nn.Linear(out_dim, n_clases)

    def _make_block(self, in_f, out_f, drop):
        return nn.Sequential(
            nn.Conv2d(in_f, out_f, kernel_size=(1, 10), bias=False),
            nn.BatchNorm2d(out_f),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(drop)
        )

    def _get_final_flattened_size(self, n_ch, n_s):
        with torch.no_grad():
            x = torch.zeros(1, 1, n_ch, n_s)
            x = self.bloque1(x)
            x = self.bloque2(x)
            x = self.bloque3(x)
            x = self.bloque4(x)
            return x.numel()

    def forward(self, x):
        if x.ndim == 3: x = x.unsqueeze(1)
        x = self.bloque1(x)
        x = self.bloque2(x)
        x = self.bloque3(x)
        x = self.bloque4(x)
        x = x.view(x.size(0), -1)
        return self.clasificador(x)


############################################################################################################

class iSpeechCNN(nn.Module):
    """
    iSpeechCNN from LTU-Machine-Learning/Rethinking-Methods-Inner-Speech repo.
    Architecture: 6 convolutional layers with explicit temporal/spatial separation.
    Input: (batch, n_channels, n_timepoints)
    Output: logits for n_classes
    
    Based on the architecture described in their training scripts:
    - Layer 1: Temporal conv [1×5] → F1 filters
    - Layer 2: Spatial conv [n_channels×1] → F1 filters
    - Layer 3: Temporal conv [1×5] → 2×F1 filters
    - Layer 4: Temporal conv [1×3] → 5×F1 filters
    - Layer 5: Temporal conv [1×3] → (25×F1)//2 filters
    - Layer 6: Temporal conv [1×3] → 25×F1 filters
    - After Layers 1: BN → LeakyReLU → Dropout
    - After Layers 2-6: BN → LeakyReLU → AvgPool[1×2] → Dropout
    - Flatten → FC → Softmax
    
    Args:
        n_channels: number of EEG channels (C)
        n_classes: number of output classes (e.g. 5 or 6)
        n_timepoints: number of time samples (T)
        F1: base number of filters (default 20 for vowels, 40 for words)
        dropout_iSpeech: dropout probability (default 0.0002)
        semilla: random seed
    """
    def __init__(self,
                 n_channels: int,
                 n_classes: int,
                 n_timepoints: int = 512,
                 F1: int = 20,
                 dropout_iSpeech: float = 0.0002,
                 semilla=None):
        super().__init__()
        
        if semilla is not None:
            torch.manual_seed(int(semilla))
            torch.cuda.manual_seed_all(int(semilla))
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_timepoints = n_timepoints
        self.F1 = F1
        self.dropout_iSpeech = dropout_iSpeech
        
        # Layer 1: Temporal convolution [1×5]
        self.conv1a = nn.Conv2d(1, self.F1, kernel_size=(1, 5), padding=(0, 2))
        self.bn1a = nn.BatchNorm2d(self.F1)
        
        # Layer 2: Spatial convolution [n_channels×1]
        self.conv1b = nn.Conv2d(self.F1, self.F1, kernel_size=(self.n_channels, 1), bias=False)
        self.bn1b = nn.BatchNorm2d(self.F1)
        
        # Layer 3: Temporal convolution [1×5]
        self.conv2 = nn.Conv2d(self.F1, 2*self.F1, kernel_size=(1, 5), padding=(0, 2))
        self.bn2 = nn.BatchNorm2d(2*self.F1)
        
        # Layer 4: Temporal convolution [1×3]
        self.conv3 = nn.Conv2d(2*self.F1, 5*self.F1, kernel_size=(1, 3), padding=(0, 1))
        self.bn3 = nn.BatchNorm2d(5*self.F1)
        
        # Layer 5: Temporal convolution [1×3]
        self.conv4 = nn.Conv2d(5*self.F1, (25*self.F1)//2, kernel_size=(1, 3), padding=(0, 1))
        self.bn4 = nn.BatchNorm2d((25*self.F1)//2)
        
        # Layer 6: Temporal convolution [1×3]
        self.conv5 = nn.Conv2d((25*self.F1)//2, 25*self.F1, kernel_size=(1, 3), padding=(0, 1))
        self.bn5 = nn.BatchNorm2d(25*self.F1)
        
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=self.dropout_iSpeech)
        
        # Calculate flattened size for FC layer
        # After all layers: (batch, 25*F1, 1, n_timepoints // 2^5)
        # Because: 5 pooling operations of factor 2 (after layers 2,3,4,5,6)
        self.pooled_timepoints = self.n_timepoints // (2**5)  # // 32
        self.flattened_size = 25 * self.F1 * 1 * self.pooled_timepoints
        
        self.fc = nn.Linear(self.flattened_size, self.n_classes)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor):
        # Input: (batch, C, T) -> Reshape for Conv2d: (batch, 1, C, T)
        x = x.unsqueeze(1)
        
        # Layer 1: Temporal conv [1×5]
        x = self.conv1a(x)
        x = self.bn1a(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Layer 2: Spatial conv [C×1]
        x = self.conv1b(x)
        x = self.bn1b(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Layer 2 pooling
        x = nn.functional.avg_pool2d(x, kernel_size=(1, 2))  # [1×2] pooling
        
        # Layer 3: Temporal conv [1×5]
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Layer 3 pooling
        x = nn.functional.avg_pool2d(x, kernel_size=(1, 2))
        
        # Layer 4: Temporal conv [1×3]
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Layer 4 pooling
        x = nn.functional.avg_pool2d(x, kernel_size=(1, 2))
        
        # Layer 5: Temporal conv [1×3]
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Layer 5 pooling
        x = nn.functional.avg_pool2d(x, kernel_size=(1, 2))
        
        # Layer 6: Temporal conv [1×3]
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Layer 6 pooling
        x = nn.functional.avg_pool2d(x, kernel_size=(1, 2))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classifier
        logits = self.fc(x)
        return logits

############################################################################################################

# Ensamble de árboles de clasificación binaria paralela según Bolaños y Rufiner
# ---
# N módulos operan paralelamente para clasificar una señal etiquetada.
# Cada módulo es un ensamble de árboles tipo LogitBoost entrenado separadamente
# en una configuración "Uno vs el Resto" (One-vs-All).
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

##################################################################
