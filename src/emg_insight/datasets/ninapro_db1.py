# src/emg_insight/datasets/ninapro_db1.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Iterator, Optional, Tuple
import re

import numpy as np
from scipy.io import loadmat

DB1_DEFAULT_FS = 100.0  # DB1 Otto Bock EMG sampling (RMS)

def _lower_nonmeta_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    """Lower-case keys and drop MATLAB meta keys."""
    return {k.lower(): v for k, v in d.items() if not k.startswith("__")}

def _pick(d: Dict[str, Any], name: str, required: bool = True):
    """Case-insensitive key access with optional requirement."""
    k = name.lower()
    if k in d:
        return d[k]
    for kk in d.keys():
        if kk.lower() == k:
            return d[kk]
    if required:
        raise KeyError(f"Key '{name}' not found in .mat keys: {list(d.keys())}")
    return None

def read_mat(path: Path) -> Dict[str, Any]:
    """Read a NinaPro DB1 .mat file, return a dict with lower-cased keys."""
    m = loadmat(path, squeeze_me=True, struct_as_record=False)
    return _lower_nonmeta_keys(m)

def _ensure_samples_channels(X: np.ndarray) -> np.ndarray:
    """Ensure EMG is shaped as (samples, channels)."""
    X = np.atleast_2d(X)
    # DB1 is typically already (samples, channels). If clearly transposed, fix it.
    if X.shape[0] < X.shape[1] and X.shape[1] > 32:  # heuristic guard
        X = X.T
    return X

def _infer_subject_exercise(path: Path) -> Tuple[Optional[str], Optional[str]]:
    """Infer subject/exercise from filename like S1_A1_E1.mat (case-insensitive)."""
    stem = path.stem.lower()
    s = re.search(r"s(\d+)", stem)
    e = re.search(r"e(\d+)", stem)
    subj = f"S{s.group(1)}" if s else None
    ex = f"E{e.group(1)}" if e else None
    return subj, ex

@dataclass
class DB1Recording:
    emg: np.ndarray             # (samples, channels) float32
    label: np.ndarray           # chosen label stream (restimulus or stimulus) int32
    stimulus: np.ndarray        # int32
    restimulus: np.ndarray      # int32
    repetition: np.ndarray      # int32
    rerepetition: np.ndarray    # int32
    glove: Optional[np.ndarray] # (samples, glove_dims) or None
    fs_hz: float
    subject: Optional[str]
    exercise: Optional[str]
    file: Path

def load_recording(
    path: str | Path,
    label_stream: str = "restimulus",
    fs_hz: float = DB1_DEFAULT_FS,
) -> DB1Recording:
    """
    Load a single DB1 .mat file.

    Parameters
    ----------
    path : str | Path
        Path to a single .mat file (e.g., S1_A1_E1.mat)
    label_stream : {'restimulus','stimulus'}
        Use refined labels 'restimulus' by default.
    fs_hz : float
        Sampling rate to associate with EMG (DB1 default 100Hz).

    Returns
    -------
    DB1Recording
    """
    p = Path(path)
    d = read_mat(p)

    emg = _pick(d, "emg")
    X = _ensure_samples_channels(np.asarray(emg, dtype=np.float32))

    stim = np.asarray(_pick(d, "stimulus"), dtype=np.int32)
    restim = np.asarray(_pick(d, "restimulus"), dtype=np.int32)
    rep = np.asarray(_pick(d, "repetition"), dtype=np.int32)
    rrep = np.asarray(_pick(d, "rerepetition"), dtype=np.int32)
    glove_arr = _pick(d, "glove", required=False)
    glove = None if glove_arr is None else _ensure_samples_channels(np.asarray(glove_arr, dtype=np.float32))

    if label_stream.lower() not in {"restimulus", "stimulus"}:
        raise ValueError("label_stream must be 'restimulus' or 'stimulus'")
    label = restim if label_stream.lower() == "restimulus" else stim

    subject, exercise = _infer_subject_exercise(p)

    return DB1Recording(
        emg=X,
        label=label,
        stimulus=stim,
        restimulus=restim,
        repetition=rep,
        rerepetition=rrep,
        glove=glove,
        fs_hz=float(fs_hz),
        subject=subject,
        exercise=exercise,
        file=p,
    )

def list_recordings(root: str | Path) -> List[Path]:
    """List all .mat files under a DB1 root (recursively)."""
    root = Path(root)
    return sorted(root.rglob("*.mat"))

def iter_recordings(
    root: str | Path,
    label_stream: str = "restimulus",
    fs_hz: float = DB1_DEFAULT_FS,
) -> Iterator[DB1Recording]:
    """Yield DB1Recording objects for every .mat under root."""
    for f in list_recordings(root):
        yield load_recording(f, label_stream=label_stream, fs_hz=fs_hz)

__all__ = [
    "DB1Recording",
    "DB1_DEFAULT_FS",
    "load_recording",
    "list_recordings",
    "iter_recordings",
    "read_mat",
]