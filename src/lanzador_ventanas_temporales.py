#!/usr/bin/env python3
"""
Wrapper que ejecuta experimentos para múltiples segmentos temporales.
Cada experimento es INDEPENDIENTE y se ejecuta secuencialmente.
"""
import subprocess
import sys
from pathlib import Path

# Configuración de segmentos (0.5s cada uno, paso de 0.25s)
SEGMENTS = [
    (0.0, 1.0),
    (0.2, 1.2),
    (0.4, 1.4),
    (0.6, 1.6),
    (0.8, 1.8),
    (1.0, 2.0),
    (1.2, 2.2),
    (1.4, 2.4),
    (1.6, 2.6),
    (1.8, 2.8),
    (2.0, 3.0),
    (2.2, 3.2),
    (2.4, 3.4),
    (2.6, 3.6),
    (2.8, 3.8),
    (3.0, 4.0),
]

# Ventana de 0.5s con desplazamiento de 0.25s (50% traslapo)
WINDOW_DURATION = 1.0
WINDOW_SHIFT = 0.2

# Ruta al script principal
SCRIPT_PATH = Path(__file__).resolve().parent / "experimento_CNNs.py"


def run_experiment(segment):
    start, end = segment
    seg_str = f"{start:.2f}_{end:.2f}"
    
    print(f"\n{'='*80}")
    print(f"INICIANDO EXPERIMENTO PARA SEGMENTO: {seg_str} segundos")
    print(f"Ventana: {WINDOW_DURATION}s, Shift: {WINDOW_SHIFT}s, Segmento: {segment}")
    print(f"{'='*80}")
    
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--trial-segment", str(start), str(end),
        "--experiment-name", f"timewindowed_128_splited_EEGNet_{seg_str}",
        "--window-duration", str(WINDOW_DURATION),
        "--window-shift", str(WINDOW_SHIFT)
    ]
    
    print(f"Comando: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"ERROR en segmento {seg_str}")
        return False
    return True


if __name__ == "__main__":
    print(f"Total de segmentos a procesar: {len(SEGMENTS)}")
    
    for i, seg in enumerate(SEGMENTS):
        print(f"\nProgreso: {i+1}/{len(SEGMENTS)}")
        success = run_experiment(seg)
        
        if not success:
            resp = input(f"FALLO en segmento {seg}. ¿Continuar? (y/n): ")
            if resp.lower() != 'y':
                print("Abortando.")
                break
    
    print("\nTODOS LOS EXPERIMENTOS COMPLETADOS")
