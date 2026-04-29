# src/trainer.py
"""
Entrenador y Evaluador Genérico para Múltiples Objetivos de Predicción.
Adaptado para extraer dinámicamente el target (Modalidad, Estímulo o Artefacto)
del vector de metadatos del Dataset.
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
from torch.utils.data import DataLoader

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class Entrenador:
    def __init__(self,
                 modelo: nn.Module,
                 optimizador: optim.Optimizer = None,
                 func_perdida: nn.Module = None,
                 device: str = None,
                 parada_temprana: int = None,
                 log_dir: str = 'runs',
                 target_idx: int = 1,  # 0: Modality, 1: Stimulus, 2: Artifact
                 save_model: bool = True):
        
        # Selección de dispositivo
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = torch.device(device)
            
        print(f"[Entrenador] Usando dispositivo: {self.device} | Objetivo Índice: {target_idx}")

        if func_perdida is None:
            raise ValueError("Debe especificarse func_perdida (ej. nn.CrossEntropyLoss()).")

        self.modelo = modelo.to(self.device)
        self.optimizador = optimizador if optimizador is not None else optim.Adam(modelo.parameters(), lr=1e-3)
        self.func_perdida = func_perdida
        self.parada_temprana = parada_temprana
        self.target_idx = target_idx
        self.save_model = save_model

        # Configuración de Logging
        self.base_log_dir = Path(log_dir)
        self.base_log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_specific_log_dir = self.base_log_dir / f"run_{timestamp}"
        self.run_specific_log_dir.mkdir(parents=True, exist_ok=True)
        self.escritor = SummaryWriter(log_dir=str(self.run_specific_log_dir))

    def _epoca_entrenamiento(self, cargador_entrenamiento: DataLoader, epoca: int):
        self.modelo.train()
        perdida_total = 0.0
        n_samples = 0
        
        for x, y_full in tqdm(cargador_entrenamiento, desc=f"Epoca {epoca} Entrenamiento", position=1, leave=True):
            x = x.to(self.device)
            # Extracción dinámica del objetivo según target_idx
            y_target = y_full[:, self.target_idx].to(self.device)
            
            self.optimizador.zero_grad()
            pred = self.modelo(x)
            loss = self.func_perdida(pred, y_target)
            loss.backward()
            self.optimizador.step()

            if hasattr(self.modelo, "apply_max_norm"):
                self.modelo.apply_max_norm()

            bs = x.shape[0]
            perdida_total += float(loss.item()) * bs
            n_samples += bs
            
        perdida_promedio = perdida_total / n_samples if n_samples > 0 else 0.0
        self.escritor.add_scalar('Perdida/entrenamiento', perdida_promedio, epoca)
        return perdida_promedio

    def _epoca_validacion(self, cargador_validacion: DataLoader, epoca: int):
        self.modelo.eval()
        perdida_total = 0.0
        correctas = 0
        n_samples = 0
        
        with torch.no_grad():
            for x, y_full in tqdm(cargador_validacion, desc=f"Epoca {epoca} Validacion", position=1, leave=True):
                x = x.to(self.device)
                y_target = y_full[:, self.target_idx].to(self.device)
                
                pred = self.modelo(x)
                loss = self.func_perdida(pred, y_target)
                
                bs = x.shape[0]
                perdida_total += float(loss.item()) * bs
                n_samples += bs
                
                _, pred_idx = torch.max(pred, 1)
                correctas += (pred_idx == y_target).sum().item()
                
        perdida_promedio = perdida_total / n_samples
        precision = correctas / n_samples
        
        self.escritor.add_scalar('Perdida/validacion', perdida_promedio, epoca)
        self.escritor.add_scalar('Precision/validacion', precision, epoca)
        return perdida_promedio, precision

    def ajustar(self,
                 cargador_entrenamiento: DataLoader,
                 cargador_validacion: DataLoader = None,
                 epocas: int = 100,
                 nombre_modelo_salida: str = None,
                 early_stop_patience: int = None):
        
        train_losses, val_losses, val_accs = [], [], []
        best_val_loss = float('inf')
        best_epoch = -1
        epochs_no_improve = 0
        best_state = copy.deepcopy(self.modelo.state_dict())
        
        patience = early_stop_patience if early_stop_patience is not None else self.parada_temprana
        
        # Crear barra global de épocas
        pbar = tqdm(total=epocas, desc="Entrenando", position=0, leave=True)
        
        for ep in range(1, epocas + 1):
            t_loss = self._epoca_entrenamiento(cargador_entrenamiento, ep)
            train_losses.append(t_loss)
            
            if cargador_validacion is not None:
                v_loss, v_acc = self._epoca_validacion(cargador_validacion, ep)
                val_losses.append(v_loss)
                val_accs.append(v_acc)
                
                # Actualizar barra con métricas
                pbar.set_postfix({
                    "loss": f"{t_loss:.4f}",
                    "val_loss": f"{v_loss:.4f}",
                    "val_acc": f"{v_acc:.4f}"
                })
                
                if v_loss < best_val_loss:
                    best_val_loss = v_loss
                    best_epoch = ep
                    epochs_no_improve = 0
                    best_state = copy.deepcopy(self.modelo.state_dict())
                    
                    if self.save_model and nombre_modelo_salida:
                        torch.save(best_state, nombre_modelo_salida)
                else:
                    epochs_no_improve += 1
                
                if patience and epochs_no_improve >= patience:
                    print(f"[Entrenador] Parada temprana en época {ep}")
                    pbar.close()
                    break
            else:
                val_losses.append(None)
                val_accs.append(None)
                pbar.set_postfix({"loss": f"{t_loss:.4f}"})
            
            pbar.update(1)
        
        pbar.close()
        
        if cargador_validacion is not None:
            self.modelo.load_state_dict(best_state)
        
        self.escritor.close()
        
        metrics = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accs": val_accs,
            "best_epoch": best_epoch,
            "n_epochs_run": len(train_losses),
            "target_index": self.target_idx
        }
        
        with open(self.run_specific_log_dir / "metrics_epochs.json", "w") as f:
            json.dump(metrics, f, indent=2)
            
        return metrics

class Evaluador:
    def __init__(self, modelo: nn.Module, device: str = None, target_idx: int = 1, nombres_clases: list = None):
        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.modelo = modelo.to(self.device)
        self.target_idx = target_idx
        self.nombres_clases = nombres_clases

    def probar(self, dataloader: DataLoader):
        self.modelo.eval()
        preds_all = []
        y_true_all = []
        
        with torch.no_grad():
            for x, y_full in dataloader:
                x = x.to(self.device)
                y_target = y_full[:, self.target_idx].to(self.device)
                
                out = self.modelo(x)
                pred = torch.argmax(out, dim=1)
                
                preds_all.append(pred.cpu().numpy())
                y_true_all.append(y_target.cpu().numpy())
                
        return np.concatenate(y_true_all), np.concatenate(preds_all)

    def reporte(self, dataloader: DataLoader):
        y_true, y_pred = self.probar(dataloader)
        
        # Mapeo automático de nombres si no se proveen
        if self.nombres_clases is None:
            if self.target_idx == 0:
                self.nombres_clases = ['Imaginada', 'Pronunciada']
            elif self.target_idx == 2:
                self.nombres_clases = ['Limpio', 'Parpadeo']

        print(f"\n--- Reporte de Clasificación (Target Idx: {self.target_idx}) ---")
        print(classification_report(y_true, y_pred, target_names=self.nombres_clases, zero_division=0))
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        return acc, cm
