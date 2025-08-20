from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt

def bandpass_filter(x: np.ndarray, fs: float, low: float = 1.0, high: float = 40.0, order: int = 4) -> np.ndarray:
    """
    Butterworth bandpass; DB1-friendly defaults (1–40 Hz).
    """
    nyq = fs / 2.0
    high = min(high, 0.9 * nyq)
    low = max(low, 0.1)
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, x, axis=0)

def rectify_and_zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Full-wave rectify + per-channel z-score.
    """
    xr = np.abs(x)
    m = xr.mean(axis=0, keepdims=True)
    s = xr.std(axis=0, keepdims=True) + eps
    return (xr - m) / s

__all__ = ["bandpass_filter", "rectify_and_zscore"]