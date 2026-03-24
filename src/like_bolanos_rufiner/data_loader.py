"""
data_loader.py
==============
Carga datos preprocesados desde la caché generada por preprocess.py.

Aplica filtros según los parámetros del experimento:
  - modalidad: 1=solo imaginada, 2=solo pronunciada, None=todas
  - subset: "vocales" (1-5), "comandos" (6-11), None=todos

Uso:
  from data_loader import load_subject_data
  X, y, metadata = load_subject_data(
      subject=1,
      data_dir=Path("data/preprocesamiento_segun_bolanos_rufiner"),
      modalidad_filter=1,
      subset="vocales"
  )
"""

from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np

SUBSET_CONFIG = {
    "vocales":  {"stim_min": 1, "stim_max": 5, "n_classes": 5,
                 "names": ["A", "E", "I", "O", "U"]},
    "comandos": {"stim_min": 6, "stim_max": 11, "n_classes": 6,
                 "names": ["up", "down", "left", "right", "forward", "back"]},
}


def load_subject_data(
    subject: int,
    data_dir: Path,
    modalidad_filter: Optional[int] = None,
    subset: Optional[str] = None,
    suffix: str = "_preprocessed",
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Carga los datos preprocesados de un sujeto con filtros aplicados.

    Parameters
    ----------
    subject : int
        Número de sujeto (1-15).
    data_dir : Path
        Directorio con los archivos .npz preprocesados.
    modalidad_filter : int | None
        1 = solo habla imaginada, 2 = solo pronunciada, None = todas.
    subset : str | None
        "vocales" = estímulos 1-5, "comandos" = estímulos 6-11.
        None = todos los estímulos.
    suffix : str
        Sufijo del archivo .npz (default "_preprocessed").

    Returns
    -------
    X : np.ndarray, shape (n_trials, n_features)
    y : np.ndarray, shape (n_trials,) — labels 0-indexed
    metadata : dict
    """
    data_dir = Path(data_dir)
    fixed_subj = f"{subject:02d}"
    path = data_dir / f"S{fixed_subj}{suffix}.npz"

    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")

    data = np.load(path, allow_pickle=True)
    X_raw = data["x"]
    y_raw = data["y"]

    # convertir las 3 etiquetas de y_raw a enteros
    y_raw = y_raw.astype(int)

    mask = np.ones(X_raw.shape[0], dtype=bool)

    if modalidad_filter is not None:
        mask &= (y_raw[:, 0] == modalidad_filter)

    if subset is not None:
        cfg = SUBSET_CONFIG[subset]
        stim_min = cfg["stim_min"]
        stim_max = cfg["stim_max"]
        n_classes = cfg["n_classes"]
        class_names = cfg["names"]
        mask &= (y_raw[:, 1] >= stim_min) & (y_raw[:, 1] <= stim_max)
    else:
        n_classes = None
        class_names = None
        stim_min = None
        stim_max = None

    X = X_raw[mask]
    y_stim = y_raw[mask, 1]

    if subset is not None:
        y = y_stim - stim_min
    else:
        y = y_stim

    if X.shape[0] == 0:
        raise ValueError(f"No hay trials para sujeto {subject} con los filtros dados.")

    metadata = {
        "subject": fixed_subj,
        "n_trials_raw": X_raw.shape[0],
        "n_trials_filtered": X.shape[0],
        "n_features": X.shape[1],
        "modalidad_filter": modalidad_filter,
        "subset": subset,
        "stim_min": int(stim_min) if stim_min is not None else None,
        "stim_max": int(stim_max) if stim_max is not None else None,
        "n_classes": n_classes,
        "class_names": class_names,
        "labels_original": y_stim.tolist(),
    }

    return X, y, metadata


def load_all_subjects(
    subjects: List[int],
    data_dir: Path,
    modalidad_filter: Optional[int] = None,
    subset: Optional[str] = None,
    suffix: str = "_preprocessed",
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """
    Carga y concatena datos de múltiples sujetos.

    Parameters
    ----------
    subjects : list[int]
        Lista de números de sujeto.
    data_dir : Path
        Directorio con los archivos .npz.
    modalidad_filter : int | None
    subset : str | None
    suffix : str

    Returns
    -------
    X : np.ndarray — concatenación de todos los sujetos
    y : np.ndarray — labels 0-indexed
    metadata_list : list[dict] — metadata por sujeto
    """
    X_list = []
    y_list = []
    metadata_list = []

    for subj in subjects:
        X_subj, y_subj, meta_subj = load_subject_data(
            subject=subj,
            data_dir=data_dir,
            modalidad_filter=modalidad_filter,
            subset=subset,
            suffix=suffix,
        )
        X_list.append(X_subj)
        y_list.append(y_subj)
        metadata_list.append(meta_subj)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    return X, y, metadata_list
