# src/models.py
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import List, Optional, Callable

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

