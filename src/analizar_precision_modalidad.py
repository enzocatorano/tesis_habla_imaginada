#!/usr/bin/env python3
"""
Analiza si existe diferencia significativa en la precisión de predicción del estímulo
entre datos de imaginación y pronunciación.
Reproduce las particiones originales usando las semillas guardadas.
Genera figura con 1×2 subplots (vocales, comandos) para boxplots de precisión,
y figura 2×2 (vocales/comandos) × (imaginada/pronunciada) para matrices de confusión.
"""
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Configuración del experimento a analizar
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = PROJECT_ROOT / "experiments" / "128_splited_EEGNet_20260430-095245_CatoranoBrothers"
DATA_DIR = PROJECT_ROOT / "data" / "preprocessed"

def load_subject_data(subject_name):
    """Carga datos preprocesados del sujeto."""
    file_path = DATA_DIR / f"{subject_name}.npz"
    if not file_path.exists():
        return None, None
    data = np.load(file_path, allow_pickle=True)
    return data['x'], data['y']

def reproducir_splits(X_all, Y_all, subset_name, seed_val, k_folds, val_frac):
    """Reproduce exactamente los splits del experimento original."""
    if subset_name == "vocales":
        stim_min, stim_max = 1, 5
    elif subset_name == "comandos":
        stim_min, stim_max = 6, 11
    else:
        return None
    
    mask = (Y_all[:, 1].astype(int) >= stim_min) & (Y_all[:, 1].astype(int) <= stim_max)
    if mask.sum() == 0:
        return None
    
    X_subset = X_all[mask]
    Y_subset_full = Y_all[mask]
    global_indices = np.where(mask)[0]
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed_val)
    splits = list(skf.split(X_subset, Y_subset_full[:, 1].astype(int)))
    
    result_splits = []
    for fold_idx, (train_idx_local, test_idx_local) in enumerate(splits, start=1):
        train_global = global_indices[train_idx_local]
        test_global = global_indices[test_idx_local]
        
        # Train/Val split
        idx_pool = np.arange(len(train_global))
        if val_frac and val_frac > 0.0:
            idx_train_rel, idx_val_rel = train_test_split(
                idx_pool, test_size=val_frac,
                stratify=Y_subset_full[train_idx_local, 1],
                random_state=seed_val + fold_idx, shuffle=True
            )
            val_global = train_global[idx_val_rel]
            train_final_global = train_global[idx_train_rel]
        else:
            val_global = np.array([], dtype=int)
            train_final_global = train_global
        
        result_splits.append({
            'fold_idx': fold_idx,
            'train_indices': train_final_global,
            'val_indices': val_global,
            'test_indices': test_global
        })
    
    return result_splits

def calcular_precision_modalidad(y_true, y_pred, modality_values):
    """Calcula precisión y retorna datos separados por modalidad."""
    resultados = {}
    
    # Asumimos: modality 1.0 = imaginada, 2.0 = pronunciada
    mask_imag = modality_values == 1.0
    mask_pron = modality_values == 2.0
    
    if mask_imag.sum() > 0:
        acc_imag = accuracy_score(y_true[mask_imag], y_pred[mask_imag])
        resultados['imaginada'] = {'acc': acc_imag, 'y_true': y_true[mask_imag], 'y_pred': y_pred[mask_imag]}
    
    if mask_pron.sum() > 0:
        acc_pron = accuracy_score(y_true[mask_pron], y_pred[mask_pron])
        resultados['pronunciada'] = {'acc': acc_pron, 'y_true': y_true[mask_pron], 'y_pred': y_pred[mask_pron]}
    
    return resultados

def main():
    # Cargar configuración del experimento
    config_path = EXP_ROOT / "experiment_config.json"
    if not config_path.exists():
        print(f"Error: No se encontró {config_path}")
        return
    
    config = json.loads(config_path.read_text(encoding='utf8'))
    seeds = config['seeds_list']
    k_folds = config['k_folds']
    val_frac = config['val_frac']
    
    # Estructuras para almacenar resultados con predicciones por fold
    resultados_vocales = {'imaginada': {'acc_list': [], 'y_true_folds': [], 'y_pred_folds': []}, 
                          'pronunciada': {'acc_list': [], 'y_true_folds': [], 'y_pred_folds': []}}
    resultados_comandos = {'imaginada': {'acc_list': [], 'y_true_folds': [], 'y_pred_folds': []}, 
                           'pronunciada': {'acc_list': [], 'y_true_folds': [], 'y_pred_folds': []}}
    
    print(f"Analizando experimento: {EXP_ROOT.name}")
    print(f"Seeds: {seeds}, Folds: {k_folds}")
    
    # Iterar sobre sujetos
    for subj_dir in sorted(EXP_ROOT.iterdir()):
        if not subj_dir.is_dir() or not subj_dir.name.startswith('S'):
            continue
        
        subj_name = subj_dir.name
        print(f"\n{'='*60}")
        print(f"Procesando {subj_name}...")
        
        # Cargar datos originales
        X_all, Y_all = load_subject_data(subj_name)
        if X_all is None:
            print(f"  No se encontraron datos para {subj_name}")
            continue
        
        # Iterar sobre subsets (vocales, comandos)
        for subset_name in ['vocales', 'comandos']:
            subset_dir = subj_dir / subset_name
            if not subset_dir.exists():
                continue
            
            print(f"  Subset: {subset_name}")
            
            # Iterar sobre seeds y folds
            for seed_idx, seed_val in enumerate(seeds):
                # Reproducir splits para este subset con la semilla actual
                splits = reproducir_splits(X_all, Y_all, subset_name, seed_val, k_folds, val_frac)
                if not splits:
                    continue
                
                for fold_info in splits:
                    fold_idx = fold_info['fold_idx']
                    
                    # Cargar predicciones del fold correspondiente
                    pred_path = subset_dir / f"seed_{seed_idx}" / f"fold_{fold_idx}" / "test_preds.npz"
                    if not pred_path.exists():
                        continue
                    
                    data = np.load(pred_path, allow_pickle=True)
                    y_true = data['y_true']
                    y_pred = data['y_pred']
                    
                    # Obtener modalidad de los trials usando índices reproducidos
                    test_indices = fold_info['test_indices']
                    
                    # Filtrar Y_all por subset
                    if subset_name == "vocales":
                        stim_min, stim_max = 1, 5
                    else:
                        stim_min, stim_max = 6, 11
                    
                    mask = (Y_all[:, 1].astype(int) >= stim_min) & (Y_all[:, 1].astype(int) <= stim_max)
                    Y_subset = Y_all[mask]
                    
                    # Mapear test_indices a Y_subset
                    subset_indices = np.where(mask)[0]
                    global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(subset_indices)}
                    test_local = [global_to_local[g] for g in test_indices if g in global_to_local]
                    
                    if len(test_local) != len(y_true):
                        print(f"    Warning: mismatch en {subset_name} seed_{seed_idx} fold_{fold_idx}")
                        continue
                    
                    modality_test = Y_subset[test_local, 0]  # Columna 0 = modalidad
                    
                    # Calcular precisión por modalidad
                    resultados_fold = calcular_precision_modalidad(y_true, y_pred, modality_test)
                    
                    # Guardar según subset y modalidad
                    target_dict = resultados_vocales if subset_name == 'vocales' else resultados_comandos
                    
                    for mod in ['imaginada', 'pronunciada']:
                        if mod in resultados_fold:
                            target_dict[mod]['acc_list'].append(resultados_fold[mod]['acc'])
                            target_dict[mod]['y_true_folds'].append(resultados_fold[mod]['y_true'])
                            target_dict[mod]['y_pred_folds'].append(resultados_fold[mod]['y_pred'])
    
    print(f"\n{'='*60}")
    print("RESULTADOS FINALES:")
    
    # Función auxiliar para imprimir resultados
    def print_modality_results(name, data_dict, chance):
        print(f"\n{name.upper()} (chance={chance:.3f}):")
        for mod in ['imaginada', 'pronunciada']:
            acc_list = data_dict[mod]['acc_list']
            if acc_list:
                print(f"  {mod.capitalize()}: {len(acc_list)} folds, "
                      f"Precisión: {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
    
    print_modality_results("vocales", resultados_vocales, 1/5)
    print_modality_results("comandos", resultados_comandos, 1/6)
    
    # Guardar en visualization_results/global/ del experimento
    viz_dir = EXP_ROOT / "visualization_results" / "global"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Generar figura 1×2 (boxplots de precisión)
    fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6))
    
    # Datos para cada subplot: (ax, dict_imaginada, dict_pronunciada, title, chance_val)
    boxplot_configs = [
        (axes1[0], resultados_vocales['imaginada']['acc_list'], 
         resultados_vocales['pronunciada']['acc_list'], 'Vocales', 1/5),
        (axes1[1], resultados_comandos['imaginada']['acc_list'], 
         resultados_comandos['pronunciada']['acc_list'], 'Comandos', 1/6),
    ]
    
    for ax, acc_imag, acc_pron, title, chance_val in boxplot_configs:
        if acc_imag or acc_pron:
            all_data = []
            labels = []
            colors = []
            
            if acc_imag:
                all_data.append(acc_imag)
                labels.append(f'Imaginada\n(n={len(acc_imag)})')
                colors.append('lightblue')
            if acc_pron:
                all_data.append(acc_pron)
                labels.append(f'Pronunciada\n(n={len(acc_pron)})')
                colors.append('lightcoral')
            
            bp = ax.boxplot(all_data, tick_labels=labels, patch_artist=True)
            
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
            
            for i, data_list in enumerate(all_data):
                x = np.random.normal(i+1, 0.04, size=len(data_list))
                ax.scatter(x, data_list, alpha=0.6, s=30, edgecolors='black', linewidth=0.5,
                          color=colors[i])
            
            ax.axhline(y=chance_val, color='gray', linestyle=':', linewidth=2,
                        label=f'Chance (~{chance_val:.3f})')
            
            ax.set_title(f'{title}', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=12, fontweight='bold')
    
    fig1.suptitle('Comparación de Precisión: Imaginada vs Pronunciada', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path_boxplot = viz_dir / "analisis_modalidad_precision.png"
    plt.savefig(output_path_boxplot, dpi=200, bbox_inches='tight')
    print(f"\nGráfico de caja guardado en: {output_path_boxplot}")
    plt.close()
    
    # 2. Generar figura 2×2 (matrices de confusión normalizadas por fila)
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 14))
    
    # Configuración de matrices: (ax, resultados_dict, modalidad, título, n_clases, nombres_clases)
    conf_configs = [
        (axes2[0, 0], resultados_vocales, 'imaginada', 'Vocales - Imaginada', 
         5, ['A', 'E', 'I', 'O', 'U']),
        (axes2[0, 1], resultados_vocales, 'pronunciada', 'Vocales - Pronunciada', 
         5, ['A', 'E', 'I', 'O', 'U']),
        (axes2[1, 0], resultados_comandos, 'imaginada', 'Comandos - Imaginada', 
         6, ['up', 'down', 'left', 'right', 'fwd', 'back']),
        (axes2[1, 1], resultados_comandos, 'pronunciada', 'Comandos - Pronunciada', 
         6, ['up', 'down', 'left', 'right', 'fwd', 'back']),
    ]
    
    for ax, results_dict, modality, title, n_classes, class_names in conf_configs:
        y_true_folds = results_dict[modality].get('y_true_folds', [])
        y_pred_folds = results_dict[modality].get('y_pred_folds', [])
        
        if not y_true_folds:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=12, fontweight='bold')
            continue
        
        # Concatenar todas las predicciones para matriz global
        try:
            all_y_true = np.concatenate([np.atleast_1d(np.asarray(x, dtype=int)) for x in y_true_folds])
            all_y_pred = np.concatenate([np.atleast_1d(np.asarray(x, dtype=int)) for x in y_pred_folds])
        except:
            # Fallback
            all_y_true = np.array([int(v) for lst in y_true_folds for v in lst])
            all_y_pred = np.array([int(v) for lst in y_pred_folds for v in lst])
        
        # Filtrar valores fuera de rango (0..n_classes-1)
        valid = (all_y_true >= 0) & (all_y_true < n_classes) & (all_y_pred >= 0) & (all_y_pred < n_classes)
        all_y_true = all_y_true[valid]
        all_y_pred = all_y_pred[valid]
        
        if len(all_y_true) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=12, fontweight='bold')
            continue
        
        # Matriz de confusión global (todas las predicciones juntas)
        cm = confusion_matrix(all_y_true, all_y_pred, labels=range(n_classes))
        
        # Normalizar por fila (cada fila suma 100%) - MATRIZ GLOBAL
        conf_mean = np.zeros((n_classes, n_classes), dtype=float)
        for r in range(n_classes):
            row_sum = cm[r].sum()
            if row_sum > 0:
                conf_mean[r] = (cm[r] / row_sum) * 100
        
        # Calcular std dev por fold (para anotaciones)
        conf_list = []
        for ytf, ypf in zip(y_true_folds, y_pred_folds):
            ytf_int = np.atleast_1d(np.asarray(ytf, dtype=int))
            ypf_int = np.atleast_1d(np.asarray(ypf, dtype=int))
            
            valid = (ytf_int >= 0) & (ytf_int < n_classes) & (ypf_int >= 0) & (ypf_int < n_classes)
            if valid.sum() == 0:
                continue
            
            cm_fold = confusion_matrix(ytf_int[valid], ypf_int[valid], labels=range(n_classes))
            cm_norm = np.zeros((n_classes, n_classes), dtype=float)
            for r in range(n_classes):
                row_sum = cm_fold[r].sum()
                if row_sum > 0:
                    cm_norm[r] = (cm_fold[r] / row_sum) * 100
            conf_list.append(cm_norm)
        
        if conf_list:
            conf_std = np.std(np.stack(conf_list), axis=0)
        else:
            conf_std = np.zeros_like(conf_mean)
        
        # Anotaciones con formato "mean±std" en porcentaje
        annot = np.empty((n_classes, n_classes), dtype=object)
        for r in range(n_classes):
            for c in range(n_classes):
                annot[r, c] = f"{conf_mean[r, c]:.1f}\n±{conf_std[r, c]:.1f}"
        
        # Graficar con seaborn (formato igual a resultados_aug_online.py)
        sns.heatmap(conf_mean, annot=annot, fmt="", cmap="cividis", ax=ax,
                   cbar_kws={'label': 'Accuracy % (row-normalized)'}, linewidths=0.5, linecolor='gray')
        
        ax.set_title(f"{title}\n(n={len(conf_list)} folds)", fontweight='bold')
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names, rotation=0)
        ax.set_xlabel('Predicho')
        ax.set_ylabel('Real')
    
    fig2.suptitle("Matrices de Confusión Normalizadas (Mean ± Std %)", fontsize=14, fontweight='bold')
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path_conf = viz_dir / "analisis_modalidad_confusion.png"
    plt.savefig(output_path_conf, dpi=200, bbox_inches='tight')
    print(f"Matrices de confusión guardadas en: {output_path_conf}")
    plt.close()
    
    # 3. Guardar datos para referencia
    result_data = {
        'vocales': {
            'imaginada': resultados_vocales['imaginada']['acc_list'],
            'pronunciada': resultados_vocales['pronunciada']['acc_list'],
            'imaginada_mean': float(np.mean(resultados_vocales['imaginada']['acc_list'])) if resultados_vocales['imaginada']['acc_list'] else None,
            'pronunciada_mean': float(np.mean(resultados_vocales['pronunciada']['acc_list'])) if resultados_vocales['pronunciada']['acc_list'] else None,
        },
        'comandos': {
            'imaginada': resultados_comandos['imaginada']['acc_list'],
            'pronunciada': resultados_comandos['pronunciada']['acc_list'],
            'imaginada_mean': float(np.mean(resultados_comandos['imaginada']['acc_list'])) if resultados_comandos['imaginada']['acc_list'] else None,
            'pronunciada_mean': float(np.mean(resultados_comandos['pronunciada']['acc_list'])) if resultados_comandos['pronunciada']['acc_list'] else None,
        }
    }
    result_path = viz_dir / "analisis_modalidad_resultados.json"
    with open(result_path, 'w', encoding='utf8') as f:
        json.dump(result_data, f, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
    
    print(f"Resultados guardados en: {result_path}")

if __name__ == "__main__":
    main()
