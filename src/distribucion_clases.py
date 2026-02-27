import os
import numpy as np
import matplotlib.pyplot as plt
import json
import glob
from matplotlib.lines import Line2D

# --- CONFIGURACIÓN ---
DATA_PATH = r'.\data\preprocessed'
OUTPUT_DIR = r'.\experiments\distribucion_clases'
os.makedirs(OUTPUT_DIR, exist_ok=True)

VOWEL_CLASSES = [1, 2, 3, 4, 5]          
COMMAND_CLASSES = [6, 7, 8, 9, 10, 11]  

MODALITY_IDX = 0  # 1: Imag, 2: Pron
CLASS_IDX = 1     # 1-11
BLINK_IDX = 2     # 1: Sin, 2: Con

COLORS = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51']
CONDITIONS = ['Total', 'Con Parpadeo', 'Sin Parpadeo', 'Pronunciado', 'Imaginado']

def get_proportions(labels, class_list):
    """
    Calcula proporciones respecto al TOTAL de datos del grupo (Vocales o Comandos) 
    disponibles para el sujeto/dataset, sin importar la condición.
    """
    # Calculamos el denominador: Total de trials de este grupo de clases
    mask_group = np.isin(labels[:, CLASS_IDX], class_list)
    total_group_trials = np.sum(mask_group)
    
    cond_masks = {
        'Total': np.ones(len(labels), dtype=bool),
        'Con Parpadeo': labels[:, BLINK_IDX] == 2,
        'Sin Parpadeo': labels[:, BLINK_IDX] == 1,
        'Pronunciado': labels[:, MODALITY_IDX] == 2,
        'Imaginado': labels[:, MODALITY_IDX] == 1
    }
    
    results = {}
    for name, mask_cond in cond_masks.items():
        props = []
        for cls in class_list:
            # Contamos trials que cumplen la condición Y pertenecen a la clase específica
            count = np.sum((mask_cond) & (labels[:, CLASS_IDX] == cls))
            # Dividimos por el total del grupo (Vocales o Comandos) del sujeto
            props.append(count / total_group_trials if total_group_trials > 0 else 0)
        results[name] = props
    return results

def plot_boxplot_with_jitter(ax, data_list, class_labels, title):
    """
    Crea un boxplot con grupos compactos, puntos grandes y cajas coloreadas (alpha 0.5).
    """
    x_base = np.arange(len(class_labels))
    width = 0.15
    offsets = np.linspace(-0.35, 0.35, len(CONDITIONS))

    for i, cond in enumerate(CONDITIONS):
        series_data = np.array(data_list[cond]) 
        
        for c_idx in range(len(class_labels)):
            pos = x_base[c_idx] + offsets[i]
            points = series_data[:, c_idx]
            
            # Boxplot con patch_artist (alpha 0.5 según pedido)
            ax.boxplot(points, positions=[pos], widths=width, 
                            showfliers=False, manage_ticks=False,
                            patch_artist=True,
                            medianprops=dict(color="orange", linewidth=1.5),
                            whiskerprops=dict(color=COLORS[i], alpha=0.7),
                            capprops=dict(color=COLORS[i], alpha=0.7),
                            boxprops=dict(facecolor=COLORS[i], color=COLORS[i], alpha=0.5))
            
            # Jitter con puntos grandes (s=50)
            jitter = np.random.normal(pos, 0.015, size=len(points))
            ax.scatter(jitter, points, color=COLORS[i], alpha=0.8, s=50, edgecolors='white', linewidths=0.5)

    ax.set_title(title, fontweight='bold', pad=20)
    ax.set_xticks(x_base)
    ax.set_xticklabels(class_labels)
    ax.set_ylim(-0.02, 0.45) 
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# --- PROCESAMIENTO ---
files = sorted(glob.glob(os.path.join(DATA_PATH, "S*_preprocessed.npz")))
subject_stats = {}
global_v_accumulator = {cond: [] for cond in CONDITIONS}
global_c_accumulator = {cond: [] for cond in CONDITIONS}

# 1. FIGURA DE SUJETOS (3x5)
fig_subs = plt.figure(figsize=(25, 18))
gs = fig_subs.add_gridspec(3, 5, hspace=0.5, wspace=0.3)

for idx, file_path in enumerate(files[:15]):
    s_id = os.path.basename(file_path).split('_')[0]
    labels = np.load(file_path)['y']
    
    v_props = get_proportions(labels, VOWEL_CLASSES)
    c_props = get_proportions(labels, COMMAND_CLASSES)
    
    subject_stats[s_id] = {'vowels': v_props, 'commands': c_props}
    for cond in CONDITIONS:
        global_v_accumulator[cond].append(v_props[cond])
        global_c_accumulator[cond].append(c_props[cond])

    inner_gs = gs[idx].subgridspec(2, 1, hspace=0.25)
    ax_v = fig_subs.add_subplot(inner_gs[0])
    ax_c = fig_subs.add_subplot(inner_gs[1])
    
    x_v, x_c = np.arange(len(VOWEL_CLASSES)), np.arange(len(COMMAND_CLASSES))
    w = 0.15
    offs = np.linspace(-w*2, w*2, 5)
    
    for i, cond in enumerate(CONDITIONS):
        ax_v.bar(x_v + offs[i], v_props[cond], w, color=COLORS[i], label=cond if idx==0 else "")
        ax_c.bar(x_c + offs[i], c_props[cond], w, color=COLORS[i])
    
    ax_v.set_title(f"Sujeto: {s_id}", fontsize=10)
    ax_v.set_xticks(x_v); ax_v.set_xticklabels([f'V{c}' for c in VOWEL_CLASSES], fontsize=7)
    ax_c.set_xticks(x_c); ax_c.set_xticklabels([f'C{c}' for c in COMMAND_CLASSES], fontsize=7)
    ax_v.set_ylim(0, 0.4); ax_c.set_ylim(0, 0.3)

fig_subs.legend(loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=5, fontsize=12)
fig_subs.savefig(os.path.join(OUTPUT_DIR, 'distribucion_sujetos.png'), bbox_inches='tight')

# --- 2. FIGURA GLOBAL (Boxplots) ---
fig_global, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
plot_boxplot_with_jitter(ax1, global_v_accumulator, [f'V{c}' for c in VOWEL_CLASSES], "Distribución Global: Vocales")
plot_boxplot_with_jitter(ax2, global_c_accumulator, [f'C{c}' for c in COMMAND_CLASSES], "Distribución Global: Comandos")

legend_elements = [Line2D([0], [0], color=COLORS[i], lw=4, label=CONDITIONS[i]) for i in range(5)]
fig_global.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=5, frameon=False)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'distribucion_total_boxplot.png'), dpi=200, bbox_inches='tight')

# --- 3. GENERAR JSON ---
final_json = {
    "dataset_global": {
        "vowels": {cond: {"mean": np.mean(global_v_accumulator[cond], axis=0).tolist(), 
                          "std": np.std(global_v_accumulator[cond], axis=0).tolist()} for cond in CONDITIONS},
        "commands": {cond: {"mean": np.mean(global_c_accumulator[cond], axis=0).tolist(), 
                            "std": np.std(global_c_accumulator[cond], axis=0).tolist()} for cond in CONDITIONS}
    },
    "por_sujeto": subject_stats
}

with open(os.path.join(OUTPUT_DIR, 'estadisticas.json'), 'w') as f:
    json.dump(final_json, f, indent=4)

print(f"Proceso finalizado. Archivos guardados en: {OUTPUT_DIR}")