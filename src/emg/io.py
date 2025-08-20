from __future__ import annotations
from typing import Any, Dict
from pathlib import Path
import re
import numpy as np
from scipy.io import loadmat

def lower_nonmeta_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    """Lower-case dict keys; drop MATLAB meta keys like '__header__'."""
    return {k.lower(): v for k, v in d.items() if not k.startswith("__")}

def pick_key(d: Dict[str, Any], name: str, required: bool = True):
    """Case-insensitive dict access (for varied MATLAB var names)."""
    key = name.lower()
    if key in d:
        return d[key]
    for kk in d:
        if kk.lower() == key:
            return d[kk]
    if required:
        raise KeyError(f"Key '{name}' not in keys: {list(d.keys())}")
    return None

def read_mat(path: str | Path):
    """Load .mat into a normalized dict (no meta keys, lowercased)."""
    m = loadmat(path, squeeze_me=True, struct_as_record=False)
    return lower_nonmeta_keys(m)

def ensure_samples_channels(X: np.ndarray) -> np.ndarray:
    """
    Ensure shape is (samples, channels). Transpose if clearly (channels, samples).
    Always returns float32.
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]
    if X.shape[0] < X.shape[1] and X.shape[1] > 32:
        X = X.T
    return X.astype(np.float32)

def subject_id_from_path(p: str | Path) -> int:
    """
    Extract integer subject ID from filenames like 'S1_A1_E1.mat' (case-insensitive).
    Returns -1 if not found.
    """
    m = re.search(r"s(\d+)", Path(p).stem.lower())
    return int(m.group(1)) if m else -1

__all__ = [
    "lower_nonmeta_keys", "pick_key", "read_mat", "ensure_samples_channels", "subject_id_from_path"
]