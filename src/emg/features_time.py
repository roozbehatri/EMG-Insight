from __future__ import annotations
import numpy as np

# ---- Basic time-domain features ----
def rms(x, axis=0): return np.sqrt((x**2).mean(axis=axis))
def mav(x, axis=0): return np.abs(x).mean(axis=axis)
def wl(x, axis=0):  return np.sum(np.abs(np.diff(x, axis=axis)), axis=axis)
def var(x, axis=0): return x.var(axis=axis)
def std(x, axis=0): return x.std(axis=axis)
def iemg(x, axis=0): return np.abs(x).sum(axis=axis)

# ---- Complexity features ----
def zc(x, axis=0, thresh=0.01):
    """Zero Crossings, thresholded to avoid counting tiny noise flips."""
    ch = x if axis == 0 else np.swapaxes(x, 0, axis)
    s  = np.sign(ch)
    d  = np.diff(ch, axis=0)
    sign_change   = (s[1:] * s[:-1]) < 0
    above_thresh  = np.abs(d) > thresh
    return (sign_change & above_thresh).sum(axis=0)

def ssc(x, axis=0, thresh=0.01):
    """Slope Sign Changes with magnitude threshold."""
    ch = x if axis == 0 else np.swapaxes(x, 0, axis)
    d1 = np.diff(ch, axis=0)
    sign_change  = (d1[1:] * d1[:-1]) < 0
    above_thresh = (np.abs(d1[1:] - d1[:-1]) > thresh)
    return (sign_change & above_thresh).sum(axis=0)

# ---- Higher-order stats ----
def kf(x, axis=0):
    """Kurtosis factor (normalized 4th moment)."""
    m = x.mean(axis=axis)
    s = x.std(axis=axis)
    return ((x - m)**4).mean(axis=axis) / (s**4 + 1e-8)

def skewness(x, axis=0):
    """Skewness (normalized 3rd moment)."""
    m = x.mean(axis=axis)
    s = x.std(axis=axis)
    return ((x - m)**3).mean(axis=axis) / (s**3 + 1e-8)

# ---- Aggregator ----
def time_domain_feature_vector(seg: np.ndarray) -> np.ndarray:
    """
    Concatenate all time-domain features per channel into a single vector.
    seg: (window_len, n_channels)
    """
    return np.concatenate([
        rms(seg, 0), mav(seg, 0), wl(seg, 0), var(seg, 0), std(seg, 0), iemg(seg, 0),
        zc(seg, 0), ssc(seg, 0), kf(seg, 0), skewness(seg, 0)
    ], axis=0)

__all__ = [
    "rms", "mav", "wl", "zc", "ssc", "var", "std", "iemg", "kf", "skewness",
    "time_domain_feature_vector"
]