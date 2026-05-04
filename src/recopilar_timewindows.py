#!/usr/bin/env python3
"""
Recopila información de experimentos con ventanas temporales y genera gráficos.
- Gráfico 1: Boxplots por ventana temporal (vocales vs comandos) con líneas de chance
- Gráfico 2: Boxplots 2x2 (vocales/comandos × imaginada/pronunciada) precisión total
- Gráfico 3: Boxplots 1x2 (vocales, comandos) con 3 grupos (total/imag/pron) x 3 métricas
"""
import re
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold

# Configuración
EXP_ROOT = Path(__file__).resolve().parents[1] / "experiments"
# Directorio con 's' en lugar de '5'
TIME_WINDOWS_DIR = EXP_ROOT / "time_window_128_splitedEEGNet_1s0.2s"

def extract_window_from_name(exp_name):
    """Extrae (start, end) del nombre: timewindowed_128_splitedEEGNet_0.00_1.00_..."""
    match = re.search(r'_(\d+\.\d+)_(\d+\.\d+)_', exp_name)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None

def load_subject_data(subject_name):
    """Carga datos preprocesados del sujeto."""
    file_path = EXP_ROOT.parent / "data" / "preprocessed" / f"{subject_name}.npz"
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
    
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed_val)
    splits = list(skf.split(X_subset, Y_subset_full[:, 1].astype(int)))
    
    result_splits = []
    for fold_idx, (train_idx_local, test_idx_local) in enumerate(splits, start=1):
        train_global = global_indices[train_idx_local]
        test_global = global_indices[test_idx_local]
        
        idx_pool = np.arange(len(train_global))
        if val_frac and val_frac > 0.0:
            from sklearn.model_selection import train_test_split
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

def calculate_metrics(y_true, y_pred):
    """Calcula precision, recall y f1 macro."""
    if len(y_true) == 0:
        return None, None, None
    
    classes = np.unique(np.concatenate([y_true, y_pred]))
    report = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)
    
    precision = report['macro avg']['precision']
    recall = report['macro avg']['recall']
    f1 = report['macro avg']['f1-score']
    
    return precision, recall, f1

def process_experiment(exp_dir):
    """Procesa un experimento y extrae métricas por modalidad y por sujeto."""
    config_path = exp_dir / "experiment_config.json"
    if not config_path.exists():
        return None, None
    
    config = json.loads(config_path.read_text(encoding='utf8'))
    seeds = config['seeds_list']
    k_folds = config['k_folds']
    val_frac = config['val_frac']
    
    results = {
        'vocales': {
            'total': {'precision': [], 'recall': [], 'f1': []},
            'imaginada': {'precision': [], 'recall': [], 'f1': []},
            'pronunciada': {'precision': [], 'recall': [], 'f1': []}
        },
        'comandos': {
            'total': {'precision': [], 'recall': [], 'f1': []},
            'imaginada': {'precision': [], 'recall': [], 'f1': []},
            'pronunciada': {'precision': [], 'recall': [], 'f1': []}
        }
    }
    
    # Estructura para datos por sujeto
    per_subject_results = {}
    
    # Procesar cada sujeto
    for subj_dir in sorted(exp_dir.iterdir()):
        if not subj_dir.is_dir() or not subj_dir.name.startswith('S'):
            continue
        
        subj_name = subj_dir.name
        X_all, Y_all = load_subject_data(subj_name)
        if X_all is None:
            continue
        
        # Inicializar estructura para este sujeto
        if subj_name not in per_subject_results:
            per_subject_results[subj_name] = {
                'vocales': {
                    'total': {'precision': [], 'recall': [], 'f1': []},
                    'imaginada': {'precision': [], 'recall': [], 'f1': []},
                    'pronunciada': {'precision': [], 'recall': [], 'f1': []}
                },
                'comandos': {
                    'total': {'precision': [], 'recall': [], 'f1': []},
                    'imaginada': {'precision': [], 'recall': [], 'f1': []},
                    'pronunciada': {'precision': [], 'recall': [], 'f1': []}
                }
            }
        
        # Procesar subsets
        for subset_name in ['vocales', 'comandos']:
            subset_dir = subj_dir / subset_name
            if not subset_dir.exists():
                continue
            
            # Procesar seeds y folds
            for seed_idx, seed_val in enumerate(seeds):
                # Reproducir splits
                splits = reproducir_splits(X_all, Y_all, subset_name, seed_val, k_folds, val_frac)
                if not splits:
                    continue
                
                for fold_info in splits:
                    fold_idx = fold_info['fold_idx']
                    pred_path = subset_dir / f"seed_{seed_idx}" / f"fold_{fold_idx}" / "test_preds.npz"
                    
                    if not pred_path.exists():
                        continue
                    
                    data = np.load(pred_path, allow_pickle=True)
                    y_true = data['y_true']
                    y_pred = data['y_pred']
                    
                    # Obtener modalidad usando índices reproducidos
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
                        continue
                    
                    modality_test = Y_subset[test_local, 0]  # Columna 0 = modalidad
                    
                    # Calcular métricas totales
                    prec, rec, f1 = calculate_metrics(y_true, y_pred)
                    if prec is not None:
                        results[subset_name]['total']['precision'].append(prec)
                        results[subset_name]['total']['recall'].append(rec)
                        results[subset_name]['total']['f1'].append(f1)
                        # También agregar a datos del sujeto
                        per_subject_results[subj_name][subset_name]['total']['precision'].append(prec)
                        per_subject_results[subj_name][subset_name]['total']['recall'].append(rec)
                        per_subject_results[subj_name][subset_name]['total']['f1'].append(f1)
                    
                    # Separar por modalidad
                    mask_imag = modality_test == 1.0
                    mask_pron = modality_test == 2.0
                    
                    if mask_imag.sum() > 0:
                        prec_i, rec_i, f1_i = calculate_metrics(y_true[mask_imag], y_pred[mask_imag])
                        if prec_i is not None:
                            results[subset_name]['imaginada']['precision'].append(prec_i)
                            results[subset_name]['imaginada']['recall'].append(rec_i)
                            results[subset_name]['imaginada']['f1'].append(f1_i)
                            per_subject_results[subj_name][subset_name]['imaginada']['precision'].append(prec_i)
                            per_subject_results[subj_name][subset_name]['imaginada']['recall'].append(rec_i)
                            per_subject_results[subj_name][subset_name]['imaginada']['f1'].append(f1_i)
                    
                    if mask_pron.sum() > 0:
                        prec_p, rec_p, f1_p = calculate_metrics(y_true[mask_pron], y_pred[mask_pron])
                        if prec_p is not None:
                            results[subset_name]['pronunciada']['precision'].append(prec_p)
                            results[subset_name]['pronunciada']['recall'].append(rec_p)
                            results[subset_name]['pronunciada']['f1'].append(f1_p)
                            per_subject_results[subj_name][subset_name]['pronunciada']['precision'].append(prec_p)
                            per_subject_results[subj_name][subset_name]['pronunciada']['recall'].append(rec_p)
                            per_subject_results[subj_name][subset_name]['pronunciada']['f1'].append(f1_p)
    
    return results, per_subject_results

def generar_precision_tiempo_sujeto(subj_name, subj_data_list, windows_labels, output_dir):
    """Genera gráfico de precisión por ventana temporal para un sujeto."""
    fig, ax = plt.subplots(figsize=(16, 6))
    
    vocab_precision = []
    comandos_precision = []
    
    for exp_data in subj_data_list:
        window_label = exp_data['window_label']
        data = exp_data['data']
        
        # Vocales total precision
        prec_list = data['vocales']['total']['precision']
        if prec_list:
            vocab_precision.append(prec_list)
        else:
            vocab_precision.append([])
        
        # Comandos total precision
        prec_list = data['comandos']['total']['precision']
        if prec_list:
            comandos_precision.append(prec_list)
        else:
            comandos_precision.append([])
    
    # Crear posiciones X para los boxplots
    n_windows = len(windows_labels)
    x_pos = np.arange(n_windows)
    
    # Plotear boxplots
    bp1 = ax.boxplot([vocab_precision[i] for i in range(n_windows)], 
                      positions=x_pos-0.15, widths=0.3, patch_artist=True, 
                      tick_labels=['']*n_windows)
    bp2 = ax.boxplot([comandos_precision[i] for i in range(n_windows)], 
                      positions=x_pos+0.15, widths=0.3, patch_artist=True, 
                      tick_labels=windows_labels)
    
    for patch in bp1['boxes']:
        patch.set_facecolor('blue')
        patch.set_alpha(0.5)
    for patch in bp2['boxes']:
        patch.set_facecolor('orange')
        patch.set_alpha(0.5)
    
    # Agregar puntos individuales
    for i in range(n_windows):
        if vocab_precision[i]:
            x = np.random.normal(i-0.15, 0.04, size=len(vocab_precision[i]))
            ax.scatter(x, vocab_precision[i], alpha=0.6, s=20, color='blue', 
                      edgecolors='black', linewidth=0.5)
        if comandos_precision[i]:
            x = np.random.normal(i+0.15, 0.04, size=len(comandos_precision[i]))
            ax.scatter(x, comandos_precision[i], alpha=0.6, s=20, color='orange', 
                      edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Ventana Temporal (inicio-fin en segundos)')
    ax.set_ylabel('Precisión')
    ax.set_title(f'Precisión por Ventana Temporal - {subj_name}', fontsize=14, fontweight='bold')
    ax.set_xticks(range(n_windows))
    ax.set_xticklabels(windows_labels, rotation=45, ha='right')
    ax.legend(['Vocales', 'Comandos'])
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=1/5, color='blue', linestyle=':', alpha=0.5, label='Chance Vocales (~0.200)')
    ax.axhline(y=1/6, color='orange', linestyle=':', alpha=0.5, label='Chance Comandos (~0.167)')
    ax.legend()
    
    plt.tight_layout()
    output_path = output_dir / f"precision_by_timewindow_{subj_name}.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"  Gráfico de tiempo guardado: {output_path}")
    plt.close()


def generar_metricas_modalidad_sujeto(subj_name, subj_data_list, output_dir):
    """Genera gráfico de métricas por modalidad para un sujeto (1x2 con 3 grupos x 3 métricas)."""
    # Recolectar datos de todos los experimentos del sujeto
    data_vocab = {'total': {'precision': [], 'recall': [], 'f1': []},
                  'imaginada': {'precision': [], 'recall': [], 'f1': []},
                  'pronunciada': {'precision': [], 'recall': [], 'f1': []}}
    data_comandos = {'total': {'precision': [], 'recall': [], 'f1': []},
                     'imaginada': {'precision': [], 'recall': [], 'f1': []},
                     'pronunciada': {'precision': [], 'recall': [], 'f1': []}}
    
    for exp_data in subj_data_list:
        data = exp_data['data']
        for key in ['total', 'imaginada', 'pronunciada']:
            for metric in ['precision', 'recall', 'f1']:
                data_vocab[key][metric].extend(data['vocales'][key][metric])
                data_comandos[key][metric].extend(data['comandos'][key][metric])
    
    # Generar gráfico
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    metrics_names = ['precision', 'recall', 'f1']
    groups = ['total', 'imaginada', 'pronunciada']
    colors = ['lightgreen', 'lightblue', 'lightcoral']
    
    for ax, dataset, title in [(axes[0], data_vocab, 'Vocales'), (axes[1], data_comandos, 'Comandos')]:
        all_plot_data = []
        all_labels = []
        all_colors = []
        
        for group, color in zip(groups, colors):
            for metric in metrics_names:
                data_list = dataset[group][metric]
                if data_list:
                    all_plot_data.append(data_list)
                    all_labels.append(f'{group.capitalize()}\n{metric.capitalize()}\n(n={len(data_list)})')
                    all_colors.append(color)
        
        if all_plot_data:
            bp = ax.boxplot(all_plot_data, tick_labels=all_labels, patch_artist=True)
            
            # Colorear por grupo
            for i, patch in enumerate(bp['boxes']):
                group_idx = i // 3
                patch.set_facecolor(colors[group_idx])
                patch.set_alpha(0.5)
            
            # Agregar puntos
            for i, data_list in enumerate(all_plot_data):
                group_idx = i // 3
                x = np.random.normal(i+1, 0.04, size=len(data_list))
                ax.scatter(x, data_list, alpha=0.6, s=20, color=colors[group_idx], 
                          edgecolors='black', linewidth=0.5)
            
            ax.set_title(f'{title} - {subj_name}', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            ax.set_xticklabels(all_labels, rotation=45, ha='right')
            
            # Línea de chance (solo para precision)
            chance = 1/5 if 'Vocales' in title else 1/6
            ax.axhline(y=chance, color='gray', linestyle=':', linewidth=2, 
                      label=f'Chance (~{chance:.3f})')
            ax.legend()
    
    fig.suptitle(f'Métricas por Modalidad - {subj_name} (Todos los tiempos combinados)', 
                fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = output_dir / f"boxplot_metrics_by_modality_{subj_name}.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"  Gráfico de métricas guardado: {output_path}")
    plt.close()


def main():
    # Encontrar todos los experimentos de ventanas temporales
    experiments = []
    if not TIME_WINDOWS_DIR.exists():
        print(f"No se encontró el directorio: {TIME_WINDOWS_DIR}")
        return
    
    for exp_dir in TIME_WINDOWS_DIR.iterdir():
        if not exp_dir.is_dir() or not exp_dir.name.startswith('timewindowed'):
            continue
        
        start, end = extract_window_from_name(exp_dir.name)
        if start is not None:
            experiments.append((start, end, exp_dir))
    
    if not experiments:
        print(f"No se encontraron experimentos en: {TIME_WINDOWS_DIR}")
        return
    
    # Ordenar por tiempo de inicio
    experiments.sort(key=lambda x: x[0])
    print(f"Encontrados {len(experiments)} experimentos:")
    for start, end, exp_dir in experiments:
        print(f"  {exp_dir.name} (segmento: {start:.2f}-{end:.2f}s)")
    
    # Crear directorios de salida
    viz_dir = TIME_WINDOWS_DIR / "visualization_results"
    global_dir = viz_dir / "global"
    global_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nDirectorio de visualización: {viz_dir}")
    
    # Estructura para almacenar datos de todos los experimentos
    all_data = {
        'windows': [],  # Etiquetas de ventanas
        'experiments': []  # Lista de resultados por experimento
    }
    
    # Estructura para datos por sujeto (across all experiments)
    all_per_subject_data = {}  # subj_name -> lista de resultados por experimento
    
    # Procesar cada experimento
    for start, end, exp_dir in experiments:
        window_label = f"{start:.2f}-{end:.2f}"
        all_data['windows'].append(window_label)
        
        print(f"\n{'='*60}")
        print(f"Procesando: {exp_dir.name}")
        print(f"Segmento: {start:.2f}-{end:.2f}s")
        
        exp_results, exp_per_subject = process_experiment(exp_dir)
        all_data['experiments'].append(exp_results)
        
        # Agregar datos por sujeto
        if exp_per_subject:
            for subj_name, subj_data in exp_per_subject.items():
                if subj_name not in all_per_subject_data:
                    all_per_subject_data[subj_name] = []
                all_per_subject_data[subj_name].append({
                    'window_label': window_label,
                    'start': start,
                    'data': subj_data
                })
    
    print(f"\n{'='*60}")
    print("RECOPILACIÓN COMPLETADA")
    
    # --- GENERAR GRÁFICOS GLOBALES ---
    
    # Gráfico 1: Boxplots por ventana temporal (vocales vs comandos)
    print("\nGenerando Gráfico 1: Boxplots por ventana temporal...")
    fig1, ax1 = plt.subplots(figsize=(16, 6))
    
    # Preparar datos: para cada ventana, tener precision de vocales y comandos
    vocab_precision = []
    comandos_precision = []
    
    for exp_results in all_data['experiments']:
        # Promediar precisión total de vocales y comandos
        if exp_results:
            # Vocales total precision
            prec_list = exp_results['vocales']['total']['precision']
            if prec_list:
                vocab_precision.append(prec_list)
            else:
                vocab_precision.append([])
            
            # Comandos total precision
            prec_list = exp_results['comandos']['total']['precision']
            if prec_list:
                comandos_precision.append(prec_list)
            else:
                comandos_precision.append([])
        else:
            vocab_precision.append([])
            comandos_precision.append([])
    
    # Crear posiciones X para los boxplots
    n_windows = len(all_data['windows'])
    x_pos = np.arange(n_windows)
    
    # Plotear boxplots
    bp1 = ax1.boxplot([vocab_precision[i] for i in range(n_windows)], positions=x_pos-0.15, widths=0.3, patch_artist=True, tick_labels=['']*n_windows)
    bp2 = ax1.boxplot([comandos_precision[i] for i in range(n_windows)], positions=x_pos+0.15, widths=0.3, patch_artist=True, tick_labels=all_data['windows'])
    
    for patch in bp1['boxes']:
        patch.set_facecolor('blue')
        patch.set_alpha(0.5)
    for patch in bp2['boxes']:
        patch.set_facecolor('orange')
        patch.set_alpha(0.5)
    
    # Agregar puntos individuales
    for i in range(n_windows):
        if vocab_precision[i]:
            x = np.random.normal(i-0.15, 0.04, size=len(vocab_precision[i]))
            ax1.scatter(x, vocab_precision[i], alpha=0.6, s=20, color='blue', edgecolors='black', linewidth=0.5)
        if comandos_precision[i]:
            x = np.random.normal(i+0.15, 0.04, size=len(comandos_precision[i]))
            ax1.scatter(x, comandos_precision[i], alpha=0.6, s=20, color='orange', edgecolors='black', linewidth=0.5)
    
    ax1.set_xlabel('Ventana Temporal (inicio-fin en segundos)')
    ax1.set_ylabel('Precisión')
    ax1.set_title('Precisión por Ventana Temporal', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(n_windows))
    ax1.set_xticklabels(all_data['windows'], rotation=45, ha='right')
    ax1.legend(['Vocales', 'Comandos'])
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=1/5, color='blue', linestyle=':', alpha=0.5, label='Chance Vocales (~0.200)')
    ax1.axhline(y=1/6, color='orange', linestyle=':', alpha=0.5, label='Chance Comandos (~0.167)')
    ax1.legend()
    
    plt.tight_layout()
    output_path_1 = global_dir / "precision_by_timewindow.png"
    plt.savefig(output_path_1, dpi=200, bbox_inches='tight')
    print(f"Gráfico 1 guardado en: {output_path_1}")
    plt.close()
    
    # Gráfico 2: Boxplots 2x2 (vocales/comandos × imaginada/pronunciada) - PRECISIÓN TOTAL
    print("\nGenerando Gráfico 2: Boxplots 2x2 (precision por modalidad)...")
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
    
    # Recolectar todos los datos de precisión de todos los experimentos combinados
    vocab_imag_prec = []
    vocab_pron_prec = []
    comandos_imag_prec = []
    comandos_pron_prec = []
    
    for exp_results in all_data['experiments']:
        if not exp_results:
            continue
        # Vocales imaginada
        vocab_imag_prec.extend(exp_results['vocales']['imaginada']['precision'])
        # Vocales pronunciada
        vocab_pron_prec.extend(exp_results['vocales']['pronunciada']['precision'])
        # Comandos imaginada
        comandos_imag_prec.extend(exp_results['comandos']['imaginada']['precision'])
        # Comandos pronunciada
        comandos_pron_prec.extend(exp_results['comandos']['pronunciada']['precision'])
    
    # Configuración de subplots
    plot_configs_2 = [
        (axes2[0, 0], vocab_imag_prec, 'Vocales - Imaginada', 'lightblue'),
        (axes2[0, 1], vocab_pron_prec, 'Vocales - Pronunciada', 'lightcoral'),
        (axes2[1, 0], comandos_imag_prec, 'Comandos - Imaginada', 'lightblue'),
        (axes2[1, 1], comandos_pron_prec, 'Comandos - Pronunciada', 'lightcoral'),
    ]
    
    for ax, data_list, title, color in plot_configs_2:
        if data_list:
            bp = ax.boxplot([data_list], tick_labels=[f'{title}\n(n={len(data_list)})'], patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
            
            # Agregar puntos
            x = np.random.normal(1, 0.04, size=len(data_list))
            ax.scatter(x, data_list, alpha=0.6, s=30, color=color, edgecolors='black', linewidth=0.5)
            
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Línea de chance
            if 'Vocales' in title:
                chance = 1/5
            else:
                chance = 1/6
            ax.axhline(y=chance, color='gray', linestyle=':', linewidth=2, label=f'Chance (~{chance:.3f})')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=12, fontweight='bold')
    
    fig2.suptitle('Precisión por Modalidad (Todos los tiempos combinados)', fontsize=14, fontweight='bold')
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    output_path_2 = global_dir / "boxplot_precision_by_modality.png"
    plt.savefig(output_path_2, dpi=200, bbox_inches='tight')
    print(f"Gráfico 2 guardado en: {output_path_2}")
    plt.close()
    
    # Gráfico 3: Boxplots 1x2 (vocales, comandos) con 3 grupos x 3 métricas
    print("\nGenerando Gráfico 3: Boxplots 1x2 (métricas por modalidad)...")
    fig3, axes3 = plt.subplots(1, 2, figsize=(16, 6))
    
    # Recolectar datos de todos los experimentos combinados
    data_vocab = {'total': {'precision': [], 'recall': [], 'f1': []},
                 'imaginada': {'precision': [], 'recall': [], 'f1': []},
                 'pronunciada': {'precision': [], 'recall': [], 'f1': []}}
    data_comandos = {'total': {'precision': [], 'recall': [], 'f1': []},
                      'imaginada': {'precision': [], 'recall': [], 'f1': []},
                      'pronunciada': {'precision': [], 'recall': [], 'f1': []}}
    
    for exp_results in all_data['experiments']:
        if not exp_results:
            continue
        
        for key in ['total', 'imaginada', 'pronunciada']:
            for metric in ['precision', 'recall', 'f1']:
                data_vocab[key][metric].extend(exp_results['vocales'][key][metric])
                data_comandos[key][metric].extend(exp_results['comandos'][key][metric])
    
    # Configuración de subplots
    metrics_names = ['precision', 'recall', 'f1']
    groups = ['total', 'imaginada', 'pronunciada']
    colors = ['lightgreen', 'lightblue', 'lightcoral']
    
    for ax, dataset, title in [(axes3[0], data_vocab, 'Vocales'), (axes3[1], data_comandos, 'Comandos')]:
        all_plot_data = []
        all_labels = []
        all_colors = []
        
        for group, color in zip(groups, colors):
            for metric in metrics_names:
                data_list = dataset[group][metric]
                if data_list:
                    all_plot_data.append(data_list)
                    all_labels.append(f'{group.capitalize()}\n{metric.capitalize()}\n(n={len(data_list)})')
                    all_colors.append(color)
        
        if all_plot_data:
            bp = ax.boxplot(all_plot_data, tick_labels=all_labels, patch_artist=True)
            
            # Colorear por grupo
            color_idx = 0
            for i, patch in enumerate(bp['boxes']):
                group_idx = i // 3
                patch.set_facecolor(colors[group_idx])
                patch.set_alpha(0.5)
            
            # Agregar puntos
            for i, data_list in enumerate(all_plot_data):
                group_idx = i // 3
                x = np.random.normal(i+1, 0.04, size=len(data_list))
                ax.scatter(x, data_list, alpha=0.6, s=20, color=colors[group_idx], edgecolors='black', linewidth=0.5)
            
            ax.set_title(f'{title} - Precision/Recall/F1', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            ax.set_xticklabels(all_labels, rotation=45, ha='right')
            
            # Línea de chance (solo para precision)
            chance = 1/5 if 'Vocales' in title else 1/6
            ax.axhline(y=chance, color='gray', linestyle=':', linewidth=2, label=f'Chance (~{chance:.3f})')
            ax.legend()
    
    fig3.suptitle('Métricas por Modalidad (Todos los tiempos combinados)', fontsize=14, fontweight='bold')
    fig3.tight_layout(rect=[0, 0, 1, 0.96])
    output_path_3 = global_dir / "boxplot_metrics_by_modality.png"
    plt.savefig(output_path_3, dpi=200, bbox_inches='tight')
    print(f"Gráfico 3 guardado en: {output_path_3}")
    plt.close()
    
    # --- GENERAR GRÁFICOS POR SUJETO ---
    
    print(f"\n{'='*60}")
    print("GENERANDO GRÁFICOS POR SUJETO...")
    
    for subj_name, subj_data_list in all_per_subject_data.items():
        subj_dir = viz_dir / subj_name
        subj_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcesando sujeto: {subj_name}")
        
        # Gráfico 1 por sujeto: Precisión por ventana temporal
        generar_precision_tiempo_sujeto(subj_name, subj_data_list, all_data['windows'], subj_dir)
        
        # Gráfico 2 por sujeto: Métricas por modalidad
        generar_metricas_modalidad_sujeto(subj_name, subj_data_list, subj_dir)
    
    print(f"\n{'='*60}")
    print("TODOS LOS GRÁFICOS GENERADOS")
    print(f"Directorio de salida global: {global_dir}")
    print(f"Directorios por sujeto: {viz_dir}/SXX/")

if __name__ == "__main__":
    main()
