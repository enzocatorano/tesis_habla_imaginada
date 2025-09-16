# src/preprocess_test.py
"""
Script de prueba para cargar archivos .mat (incl. v7.3/HDF5), inspeccionar
su contenido y guardar un ejemplo procesado en data/processed.

Uso:
    python src/preprocess_test.py --config ../config.yml
    python src/preprocess_test.py --data-root "C:\ruta\a\Base_de_Datos_Habla_Imaginada"

Salida:
    data/processed/example_from_<matname>.npz
    data/processed/example_from_<matname>.meta.json
"""
import argparse
import os
import json
import scipy.io as sio
import numpy as np
import yaml

# h5py sólo se importa si hace falta (v7.3)
try:
    import h5py
except Exception:
    h5py = None  # lo chequeamos dinámicamente

COMMON_FS_KEYS = ("fs", "srate", "sfreq", "sampling_rate", "Fs")

def find_mat_files(root):
    mats = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(".mat"):
                mats.append(os.path.join(dirpath, f))
    mats.sort()
    return mats

def is_hdf5_mat(path):
    # MATLAB v7.3 files are HDF5-based; h5py puede abrirlos.
    # Intentamos abrir con h5py para detectarlos sin lanzar excepción fuerte.
    if h5py is None:
        return False
    try:
        with h5py.File(path, "r"):
            return True
    except Exception:
        return False

def inspect_hdf5(path, verbose=False):
    # Devuelve lista de datasets/paths en el archivo HDF5
    items = []
    with h5py.File(path, "r") as f:
        def visitor(name, obj):
            # guardamos solo datasets (no grupos) para no inundar
            if isinstance(obj, h5py.Dataset):
                items.append(name)
        f.visititems(visitor)
        if verbose:
            print("HDF5 keys (datasets):")
            for k in items:
                print("  ", k)
    return items

def read_hdf5_dataset(path, dataset_name):
    with h5py.File(path, "r") as f:
        data = f[dataset_name][()]
    return data

def load_mat_scipy(path, verbose=False):
    m = sio.loadmat(path)
    keys = [k for k in m.keys() if not k.startswith("__")]
    if verbose:
        print("Claves (scipy.loadmat):", keys)
    return m, keys

def guess_signal_from_matdict(mdict, keys, verbose=False):
    # heurística para elegir la variable que probablemente contiene la señal EEG
    # - preferir arrays con >1 dimension (canales x muestras o muestras x canales)
    # - buscar nombres comunes ('EEG','data','signals','X')
    priority_names = ["EEG", "data", "signals", "X", "y", "epoch", "trial"]
    candidates = []
    for k in keys:
        val = mdict.get(k)
        if isinstance(val, np.ndarray):
            shape = getattr(val, "shape", None)
            candidates.append((k, val, shape))
    # chequeo por nombre preferente
    for name in priority_names:
        for (k, val, shape) in candidates:
            if k.lower().startswith(name.lower()):
                if verbose:
                    print(f"Elegida por nombre preferente: {k}, shape={shape}")
                return k, val
    # si no, elegir la primera matriz que tenga al menos 2 dimensiones
    for (k, val, shape) in candidates:
        if shape is not None and len(shape) >= 2:
            if verbose:
                print(f"Elegida por forma (ndim>=2): {k}, shape={shape}")
            return k, val
    # fallback: si hay cualquier ndarray, devolver la primera
    if candidates:
        k, val, shape = candidates[0]
        if verbose:
            print(f"Elegida fallback: {k}, shape={shape}")
        return k, val
    return None, None

def extract_fs_from_matdict(mdict, keys):
    for k in keys:
        lower = k.lower()
        if any(fs_k in lower for fs_k in COMMON_FS_KEYS):
            try:
                val = mdict[k]
                if isinstance(val, np.ndarray):
                    # sacar escalar si es array 1x1
                    if val.size == 1:
                        return float(np.ravel(val)[0])
                    else:
                        # si es array más grande, intentar el primer elemento
                        return float(np.ravel(val)[0])
                else:
                    return float(val)
            except Exception:
                continue
    # no encontrado
    return None

def meta_from_path(mat_path):
    # intentamos sacar sujeto/experiment info desde la ruta: folder padre puede ser sujeto
    folder = os.path.dirname(mat_path)
    parent = os.path.basename(folder)
    return {"subject_folder": parent, "source_file": os.path.basename(mat_path)}

def save_example(processed_root, mat_path, arr, key_used, fs=None, overwrite=False):
    os.makedirs(processed_root, exist_ok=True)
    base = os.path.splitext(os.path.basename(mat_path))[0]
    npz_name = f"example_from_{base}.npz"
    meta_name = f"example_from_{base}.meta.json"
    npz_path = os.path.join(processed_root, npz_name)
    meta_path = os.path.join(processed_root, meta_name)
    if os.path.exists(npz_path) and not overwrite:
        raise FileExistsError(f"{npz_path} ya existe. Usa --overwrite para sobreescribir.")
    # guardamos el array comprimido
    np.savez_compressed(npz_path, data=arr, mat_key=np.array([key_used]))
    # metadatos
    meta = meta_from_path(mat_path)
    meta.update({
        "mat_key": key_used,
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "fs": fs,
        "source": os.path.abspath(mat_path)
    })
    with open(meta_path, "w", encoding="utf8") as fh:
        json.dump(meta, fh, indent=2)
    return npz_path, meta_path

def main():
    parser = argparse.ArgumentParser(description="Prueba de carga .mat y guardado ejemplo procesado")
    parser.add_argument("--config", default="config.yml", help="ruta al config.yml (relativa a src o a la raiz)")
    parser.add_argument("--data-root", default=None, help="ruta alternativa a carpeta con .mat")
    parser.add_argument("--sample-index", type=int, default=0, help="índice del archivo .mat a probar (0..)")
    parser.add_argument("--overwrite", action="store_true", help="sobreescribir si ya existe el ejemplo")
    parser.add_argument("--verbose", action="store_true", help="mensajes detallados")
    args = parser.parse_args()

    # resolver config.yml: primero relativo a src, luego ../config.yml
    cfg_path = args.config
    if not os.path.isabs(cfg_path):
        candidate1 = os.path.join(os.path.dirname(__file__), cfg_path)  # src/config.yml
        candidate2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", cfg_path))  # ../config.yml
        if os.path.exists(candidate1):
            cfg_path = candidate1
        elif os.path.exists(candidate2):
            cfg_path = candidate2
        else:
            # dejar la ruta tal cual para que el mensaje de error sea claro más abajo
            cfg_path = candidate2

    if args.verbose:
        print("Usando config en:", cfg_path)

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"config.yml no encontrado en: {cfg_path}")

    with open(cfg_path, "r", encoding="utf8") as fh:
        cfg = yaml.safe_load(fh)

    data_root = args.data_root if args.data_root else cfg.get("data_root")
    processed_root = cfg.get("processed_root", os.path.join(os.path.dirname(cfg_path), "data", "processed"))

    if not data_root or not os.path.exists(data_root):
        raise FileNotFoundError(f"data_root no encontrado o no existe: {data_root}")

    mats = find_mat_files(data_root)
    if args.verbose:
        print(f"Encontrados {len(mats)} .mat en {data_root}")

    if len(mats) == 0:
        print("No encontré archivos .mat. Revisa data_root en config.yml")
        return

    idx = args.sample_index
    if idx < 0 or idx >= len(mats):
        raise IndexError(f"sample-index fuera de rango: {idx} (hay {len(mats)} archivos)")

    sample = mats[idx]
    if args.verbose:
        print("Probando archivo:", sample)

    # detectamos si es v7.3 HDF5
    if is_hdf5_mat(sample):
        if args.verbose:
            print("Archivo es MATLAB v7.3 (HDF5). Inspeccionando con h5py...")
        if h5py is None:
            raise ImportError("Archivo v7.3 detectado pero h5py no está instalado. Instala con: conda install -c conda-forge h5py")
        items = inspect_hdf5(sample, verbose=args.verbose)
        # heurística: si hay datasets, tomamos el primero grande; de lo contrario devolvemos la lista
        if not items:
            print("No encontré datasets HDF5 dentro del .mat. Miralo manualmente.")
            return
        # intentamos leer el primer dataset razonable que parezca matriz
        chosen = None
        for name in items:
            try:
                arr = read_hdf5_dataset(sample, name)
                if isinstance(arr, np.ndarray) and arr.size > 1:
                    chosen = (name, arr)
                    break
            except Exception:
                continue
        if chosen is None:
            print("No pude extraer un dataset numpy útil automáticamente. Mostré lista de datasets arriba.")
            return
        key_used, arr = chosen
        fs = None  # no intentamos inferir fs desde hdf5 automáticamente aquí
        if args.verbose:
            print(f"Leí dataset HDF5 '{key_used}', shape={arr.shape}, dtype={arr.dtype}")
    else:
        # cargamos con scipy
        m, keys = load_mat_scipy(sample, verbose=args.verbose)
        key_used, arr = guess_signal_from_matdict(m, keys, verbose=args.verbose)
        fs = extract_fs_from_matdict(m, keys)
        if key_used is None:
            print("No encontré una matriz numpy clara en este .mat. Claves encontradas:", keys)
            return
        if args.verbose:
            print(f"Variable elegida: {key_used}, shape={getattr(arr,'shape',None)}, dtype={getattr(arr,'dtype',None)}")
    # normalizamos la forma: si la matriz viene como (n_samples, n_channels) la transponemos a (n_channels, n_samples)
    try:
        shape = arr.shape
        if len(shape) == 2 and shape[0] < shape[1]:
            # asumimos que está (channels, samples) si eje0 < eje1? no siempre es correcto,
            # pero dejamos una heurística: si la primer dim es pequeña (<128) la consideramos canales
            pass  # no transpone por defecto, solo comento la heurística
    except Exception:
        pass

    # guardamos ejemplo y metadatos
    try:
        npz_path, meta_path = save_example(processed_root, sample, arr, key_used, fs=fs, overwrite=args.overwrite)
        print("Guardado .npz en:", npz_path)
        print("Guardado metadata en:", meta_path)
    except FileExistsError as e:
        print(str(e))
        print("Si querés sobrescribir usa --overwrite")

if __name__ == "__main__":
    main()