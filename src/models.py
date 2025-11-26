# src/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import List, Optional, Callable

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
                 in_ch,
                 n_classes,
                 F1: int = 8,
                 D: int = 2,
                 F2: int | None = None,
                 kernel_length: int = 64,
                 separable_kernel_length: int = 16,
                 pool_time1: int = 4,
                 pool_time2: int = 8,
                 dropout_prob: float = 0.5,
                 hidden_units: int | None = None,
                 max_norm_spatial: float = 1.0,
                 max_norm_dense: float = 0.25,
                 semilla=None):
        super().__init__()

        # ----------------------------------------------------------
        # BLOQUE DE REPRODUCIBILIDAD
        # ----------------------------------------------------------
        if semilla is not None:
            torch.manual_seed(int(semilla))
            torch.cuda.manual_seed_all(int(semilla))
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            self.semilla = int(semilla)
        else:
            self.semilla = None
        # ----------------------------------------------------------

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
        # Block 1: Temporal conv
        # ---------------------
        # Input reshaped to (batch, 1, C, T)
        # Conv2d temporal: in_channels=1, out_channels=F1, kernel_size=(1, kernel_length)
        self.conv_temporal = nn.Conv2d(in_channels=1,
                                       out_channels=self.F1,
                                       kernel_size=(1, self.kernel_length),
                                       padding=(0, self.kernel_length // 2),
                                       bias=False)
        self.bn1 = nn.BatchNorm2d(self.F1)

        # ---------------------
        # Depthwise spatial conv
        # ---------------------
        # Implement depthwise: groups = in_channels (=F1), out_channels = F1 * D
        # Kernel height spans all channels (in_ch) => kernel_size=(in_ch, 1)
        self.depthwise_spatial = nn.Conv2d(in_channels=self.F1,
                                           out_channels=self.F1 * self.D,
                                           kernel_size=(self.in_ch, 1),
                                           groups=self.F1,          # depthwise per temporal filter
                                           bias=False)
        # BatchNorm after depthwise
        self.bn_depth = nn.BatchNorm2d(self.F1 * self.D)

        # ---------------------
        # Pooling + dropout after block 1
        # ---------------------
        self.pool1 = nn.AvgPool2d(kernel_size=(1, self.pool_time1))
        self.dropout1 = nn.Dropout(p=self.dropout_prob)
        self.elu = nn.ELU()

        # ---------------------
        # Block 2: Separable conv
        # Depthwise temporal then pointwise 1x1
        # ---------------------
        # Depthwise temporal conv: groups = in_channels (F1*D), out_channels = same (multiplier=1)
        self.depthwise_time = nn.Conv2d(in_channels=self.F1 * self.D,
                                        out_channels=self.F1 * self.D,
                                        kernel_size=(1, self.separable_kernel_length),
                                        padding=(0, self.separable_kernel_length // 2),
                                        groups=self.F1 * self.D,
                                        bias=False)
        # pointwise 1x1 conv to mix maps -> F2 output maps
        self.pointwise = nn.Conv2d(in_channels=self.F1 * self.D,
                                   out_channels=self.F2,
                                   kernel_size=(1, 1),
                                   bias=False)
        self.bn2 = nn.BatchNorm2d(self.F2)

        self.pool2 = nn.AvgPool2d(kernel_size=(1, self.pool_time2))
        self.dropout2 = nn.Dropout(p=self.dropout_prob)

        # ---------------------
        # Classifier
        # ---------------------
        # We don't know the final temporal length until forward (depends on T and pooling),
        # so we will create the dense layers lazily in forward when flatten size is known.
        self._dense_initialized = False

        # store layers for later init
        self.final_fc = None
        self.hidden_fc = None

        # initialize weights (common practice)
        self._init_weights()

    def _init_weights(self):
        # Kaiming normal for convs
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _create_dense(self, flattened_dim):
        """Create classifier layers once flattened size is known.
        Ensure new layers live on the same device as the rest of the model.
        """
        if self._dense_initialized:
            return

        # device where existing parameters live (conv layers already moved por trainer)
        try:
            device = next(self.parameters()).device
        except StopIteration:
            # fallback to CPU if model has no parameters (shouldn't happen)
            device = torch.device('cpu')

        if self.hidden_units is None:
            # single FC -> class logits (moved to device)
            self.final_fc = nn.Linear(flattened_dim, self.n_classes, bias=True).to(device)
        else:
            self.hidden_fc = nn.Linear(flattened_dim, self.hidden_units, bias=True).to(device)
            self.final_fc = nn.Linear(self.hidden_units, self.n_classes, bias=True).to(device)

        # initialize these weights too (on the right device)
        if self.final_fc is not None:
            nn.init.xavier_uniform_(self.final_fc.weight)
            if self.final_fc.bias is not None:
                nn.init.zeros_(self.final_fc.bias)
        if self.hidden_fc is not None:
            nn.init.xavier_uniform_(self.hidden_fc.weight)
            if self.hidden_fc.bias is not None:
                nn.init.zeros_(self.hidden_fc.bias)

        self._dense_initialized = True

    def forward(self, x: torch.Tensor):
        """
        x: (batch, C, T)
        returns logits (batch, n_classes)
        """
        if x.dim() != 3:
            raise ValueError("Input tensor must be shape (batch, C, T)")

        # reshape to (batch, 1, C, T)
        x = x.unsqueeze(1)

        # Block1: temporal conv
        x = self.conv_temporal(x)       # -> (batch, F1, C, T)
        x = self.bn1(x)
        # Depthwise spatial conv
        x = self.depthwise_spatial(x)   # -> (batch, F1*D, 1, T)
        x = self.bn_depth(x)
        x = self.elu(x)
        x = self.pool1(x)               # -> (batch, F1*D, 1, T//pool_time1)
        x = self.dropout1(x)

        # Block2: separable conv
        x = self.depthwise_time(x)      # -> (batch, F1*D, 1, T' )
        x = self.pointwise(x)           # -> (batch, F2, 1, T' )
        x = self.bn2(x)
        x = self.elu(x)
        x = self.pool2(x)               # -> (batch, F2, 1, T'' )
        x = self.dropout2(x)

        # flatten: remove the singleton spatial dimension (height=1)
        batch = x.shape[0]
        _, nch, h, t = x.shape  # h should be 1
        x = x.reshape(batch, nch * h * t)

        # lazy dense creation according to flattened dimension
        if not self._dense_initialized:
            self._create_dense(flattened_dim=x.shape[1])

        if self.hidden_units is not None:
            x = self.hidden_fc(x)
            x = F.elu(x)

        logits = self.final_fc(x)
        return logits

    def apply_max_norm(self):
        """
        Apply max-norm constraints to:
         - spatial depthwise conv filters (self.depthwise_spatial)
         - final dense layer weights (self.final_fc) if present
        This function should be called AFTER optimizer.step() each training iteration.
        """
        # 1) spatial conv: weights shape -> (out_channels, in_channels_per_group, kernel_h, kernel_w)
        # For depthwise spatial: out_channels = F1 * D, in_channels_per_group = 1 (because groups=F1)
        w = self.depthwise_spatial.weight.data  # shape (F1*D, 1, kernel_h, 1)
        # compute norm over (in_channel, kernel_h, kernel_w) -> per-filter norm
        # flatten dims 1:]
        w_flat = w.view(w.shape[0], -1)
        norms = w_flat.norm(2, dim=1, keepdim=True)
        desired = torch.clamp(norms, max=self.max_norm_spatial)
        # avoid division by zero
        scale = desired / (1e-8 + norms)
        w_flat = w_flat * scale
        self.depthwise_spatial.weight.data = w_flat.view_as(w)

        # 2) final dense layer weights: apply row-wise max-norm (incoming weights per neuron)
        if self.final_fc is not None:
            W = self.final_fc.weight.data  # shape (n_classes, hidden or flattened)
            W_flat = W.view(W.shape[0], -1)
            norms = W_flat.norm(2, dim=1, keepdim=True)
            desired = torch.clamp(norms, max=self.max_norm_dense)
            scale = desired / (1e-8 + norms)
            W_flat = W_flat * scale
            self.final_fc.weight.data = W_flat.view_as(W)

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

