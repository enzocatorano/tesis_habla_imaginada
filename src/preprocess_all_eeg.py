# src/preprocess_all_eeg.py
import os
import argparse
import json
import numpy as np
from scipy.io import loadmat
from glob import glob
import yaml

def process_file(file_path, output_dir, fs, verbose=False):
    if verbose:
        print(f"Procesando archivo: {file_path}")

    try:
        mat = loadmat(file_path)
    except Exception as e:
        print(f"  Error loadmat en {file_path}: {e}")
        return

    # preferencia por 'EEG'
    if 'EEG' in mat:
        data = mat['EEG']
        if verbose:
            print(f"  Variable EEG encontrada, shape={data.shape}")
    else:
        candidates = {k: v for k, v in mat.items() if isinstance(v, np.ndarray) and getattr(v, "ndim", 0) >= 2}
        if not candidates:
            if verbose:
                print(f"  No se encontró variable multidimensional en {file_path}, se saltea.")
            return
        name, data = list(candidates.items())[0]
        if verbose:
            print(f"  Variable elegida por forma: {name}, shape={data.shape}")

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    npz_path = os.path.join(output_dir, f"{base_name}.npz")
    meta_path = os.path.join(output_dir, f"{base_name}.meta.json")

    # Guardar .npz (sobreescribe)
    np.savez(npz_path, data=data)
    if verbose:
        print(f"  Guardado .npz en: {npz_path}")

    # Guardar/actualizar metadata (incluye fs)
    meta = {
        "shape": data.shape,
        "dtype": str(data.dtype),
        "original_file": os.path.abspath(file_path),
        "fs": fs
    }
    with open(meta_path, "w", encoding="utf8") as fh:
        json.dump(meta, fh, indent=4)
    if verbose:
        print(f"  Guardado metadata en: {meta_path}\n")


def resolve_paths_from_cfg(cfg, config_path):
    """
    Busca data_root y processed_root en cfg con tolerancia a nombres distintos.
    Si no encuentra data_root intenta un fallback relativo a config_path.
    """
    # posibles claves para data root
    for key in ("data_root", "dataset_path", "base_data_path"):
        if key in cfg and cfg[key]:
            data_root = cfg[key]
            break
    else:
        # fallback: carpeta ../Base_de_Datos_Habla_Imaginada relativa al config.yml
        config_dir = os.path.dirname(os.path.abspath(config_path))
        data_root = os.path.abspath(os.path.join(config_dir, "..", "Base_de_Datos_Habla_Imaginada"))

    # si la path en cfg es relativa, resolver respecto a la carpeta del config
    if not os.path.isabs(data_root):
        config_dir = os.path.dirname(os.path.abspath(config_path))
        data_root = os.path.abspath(os.path.join(config_dir, data_root))

    # processed_root: preferir ruta en cfg, si no, por defecto data/processed en la raíz del repo (carpeta del config)
    processed_root = cfg.get("processed_root")
    if processed_root:
        if not os.path.isabs(processed_root):
            processed_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(config_path)), processed_root))
    else:
        processed_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(config_path)), "data", "processed"))

    return data_root, processed_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Ruta al YAML de config")
    parser.add_argument('--verbose', action='store_true', help="Imprime info detallada")
    args = parser.parse_args()

    # cargar yaml
    with open(args.config, "r", encoding="utf8") as fh:
        cfg = yaml.safe_load(fh) or {}

    # frecuencia por defecto (si querés tomarla desde cfg, pon cfg['fs'])
    fs = cfg.get("fs", 1024)

    # resolver rutas de datos y processed
    data_root, output_dir = resolve_paths_from_cfg(cfg, args.config)

    if args.verbose:
        print("Configuración leída del YAML:", cfg)
        print("Usando data_root =", data_root)
        print("Usando processed_root =", output_dir)
        print("fs =", fs)

    if not os.path.exists(data_root):
        raise FileNotFoundError(f"data_root no existe: {data_root}")

    os.makedirs(output_dir, exist_ok=True)

    # buscar todos los archivos *_EEG.mat (case-insensitive)
    eeg_files = []
    for p in glob(os.path.join(data_root, "**", "*.*"), recursive=True):
        if p.lower().endswith("_eeg.mat"):
            eeg_files.append(p)
    eeg_files.sort()

    if args.verbose:
        print(f"Encontrados {len(eeg_files)} archivos EEG en {data_root}\n")

    for f in eeg_files:
        process_file(f, output_dir, fs, verbose=args.verbose)

    if args.verbose:
        print("Procesamiento finalizado.")


if __name__ == "__main__":
    main()
