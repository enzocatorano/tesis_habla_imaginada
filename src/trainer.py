# src/trainer.py
"""
Entrenador y Evaluador mejorados.

Entrenador:
 - registro de scalars en TensorBoard (losses, acc)
 - opción histogram_freq: 0 = nunca, >0 = cada N epochs
 - collect epoch-wise arrays (train_losses, val_losses, val_accs)
 - escribe metrics_epochs.json en el run_specific_log_dir
 - devuelve un dict con las listas para que el script de entrenamiento las guarde donde quiera

Evaluador:
 - evaluar, matriz de confusión, reporte y devolver métricas
"""
import os
import json
import copy
import datetime
from pathlib import Path
from tqdm import tqdm

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
                 histogram_freq: int = 0):
        """
        Args:
            modelo: nn.Module
            optimizador: torch optimizer (si None se crea Adam por defecto)
            func_perdida: loss function (obligatoria)
            device: 'cuda' or 'cpu' o None (auto detect)
            parada_temprana: patience en epocas (None para desactivar)
            log_dir: directorio raíz para TensorBoard (cada run tendrá un subdir timestamp)
            histogram_freq: 0 = nunca, >0 = cada N epochs loguea histogramas
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
                    if nombre_modelo_salida:
                        # guardamos el checkpoint del mejor modelo
                        torch.save(self.modelo.state_dict(), nombre_modelo_salida)
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
            "n_epochs_run": len(train_losses)
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
