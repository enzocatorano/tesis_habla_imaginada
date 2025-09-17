#!/usr/bin/env python3
# src/make_meta_featureset.py
"""
Genera un meta_full.json para un features set.
Busca archivos *_features_flat.npy en features/<feat_dir>/per_subject/
y (si existe) lee features/<feat_dir>/meta/meta.json para tomar L, wavelet, target_fs, n_channels.
Si no encuentra esos campos, usa valores por defecto o argumentos CLI.
"""
import json
from pathlib import Path
import numpy as np
import subprocess
import argparse
from datetime import datetime

def get_git_sha():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        sha = None
    return sha

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=str, default="features/preproc_dwt_L5_db4",
                        help="Directorio del feature set (contiene per_subject/ y meta/)")
    parser.add_argument("--defaults", type=str, default=None,
                        help="JSON string o path con defaults para L, wavelet, target_fs, n_channels (opcional)")
    parser.add_argument("--out", type=str, default=None,
                        help="Ruta de salida para meta_full.json (por defecto features_dir/meta/meta_full.json)")
    args = parser.parse_args()

    feat_dir = Path(args.features_dir)
    per_sub = feat_dir / "per_subject"
    meta_dir = feat_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else (meta_dir / "meta_full.json")

    # intentar leer meta existente si hay
    existing_meta_path = meta_dir / "meta.json"
    existing_meta = {}
    if existing_meta_path.exists():
        try:
            existing_meta = json.loads(existing_meta_path.read_text(encoding="utf8"))
        except Exception:
            existing_meta = {}

    # defaults (prioridad: CLI defaults -> existing_meta -> built-ins)
    defaults = {}
    if args.defaults:
        # si args.defaults apunta a archivo
        p = Path(args.defaults)
        if p.exists():
            try:
                defaults = json.loads(p.read_text(encoding="utf8"))
            except Exception:
                defaults = {}
        else:
            try:
                defaults = json.loads(args.defaults)
            except Exception:
                defaults = {}

    # valores finales: buscar en existing_meta, luego defaults, luego valores fijos
    def pick(key, fallback):
        return existing_meta.get(key, defaults.get(key, fallback))

    L = int(pick("L", 5))
    wavelet = str(pick("wavelet", "db4"))
    target_fs = int(pick("target_fs", 128))
    n_channels = int(pick("n_channels", 6))

    per_subject_summary = {}
    total_trials = 0
    n_features = None
    subject_files = sorted(per_sub.glob("*_features_flat.npy"))
    for p in subject_files:
        try:
            arr = np.load(p)
        except Exception as e:
            print(f"Warning: no pude cargar {p}: {e}")
            continue
        stem = p.stem.replace("_features_flat", "")
        per_subject_summary[stem] = {
            "n_trials": int(arr.shape[0]),
            "features_flat_shape": list(arr.shape)
        }
        total_trials += int(arr.shape[0])
        if n_features is None:
            n_features = int(arr.shape[1]) if arr.ndim == 2 else None

    n_subjects = len(per_subject_summary)

    # si existing_meta ya tenía algunos campos útiles (por ejemplo samples_per_channel_after_resample), los conservamos
    extra_from_existing = {}
    for key in ("samples_per_channel_after_resample", "orig_fs_default", "resample_method"):
        if key in existing_meta:
            extra_from_existing[key] = existing_meta[key]

    meta_full = {
        "preproc_id": feat_dir.name,
        "L": L,
        "wavelet": wavelet,
        "target_fs": target_fs,
        "n_channels": n_channels,
        "n_subjects": int(n_subjects),
        "n_total_trials": int(total_trials),
        "n_features_flat": int(n_features) if n_features is not None else None,
        "per_subject_summary": per_subject_summary,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "git_sha": get_git_sha(),
    }
    # añadir extras si existían
    meta_full.update(extra_from_existing)

    out_path.write_text(json.dumps(meta_full, indent=2), encoding="utf8")
    print(f"Wrote meta summary to: {out_path}")

if __name__ == "__main__":
    main()
