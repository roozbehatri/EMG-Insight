from __future__ import annotations
import numpy as np
from scipy.signal import welch

EPS = 1e-12  # numerical stability for divides/logs

def _welch_psd(x: np.ndarray, fs: float, nperseg: int | None = None):
    """
    Welch PSD for each channel.
    Returns:
      f: (F,) frequencies in Hz
      P: (F, C) PSD per channel
    """
    if nperseg is None:
        nperseg = int(np.clip(len(x), 64, 1024))
    f, P = welch(x, fs=fs, nperseg=min(nperseg, len(x)), axis=0)
    return f, P

def total_power(x, fs):
    f, P = _welch_psd(x, fs)
    return np.trapezoid(P, f, axis=0)

def bandpower(x, fs, band=(5, 15)):
    f, P = _welch_psd(x, fs)
    idx = (f >= band[0]) & (f <= band[1])
    return np.trapezoid(P[idx, :], f[idx], axis=0)

def mean_frequency(x, fs):
    f, P = _welch_psd(x, fs)
    denom = np.trapezoid(P, f, axis=0) + EPS
    return np.trapezoid(P * f[:, None], f, axis=0) / denom

def median_frequency(x, fs):
    f, P = _welch_psd(x, fs)
    cumsum = np.cumsum(P, axis=0)
    totals = cumsum[-1, :] + EPS
    medf = np.empty(P.shape[1])
    for c in range(P.shape[1]):
        idx = np.searchsorted(cumsum[:, c], 0.5 * totals[c])
        medf[c] = f[min(max(idx, 0), len(f) - 1)]
    return medf

def peak_frequency(x, fs):
    f, P = _welch_psd(x, fs)
    idx = np.argmax(P, axis=0)
    return f[idx]

def spectral_moments(x, fs):
    f, P = _welch_psd(x, fs)
    m0 = np.trapezoid(P, f, axis=0) + EPS
    m1 = np.trapezoid(P * f[:, None], f, axis=0) / m0
    var = np.trapezoid(P * (f[:, None] - m1[None, :])**2, f, axis=0) / m0
    return m0, m1, var

def spectral_entropy(x, fs, base=2):
    f, P = _welch_psd(x, fs)
    Pn = P / (P.sum(axis=0, keepdims=True) + EPS)
    H = -(Pn * np.log(Pn + EPS)).sum(axis=0)
    return H / np.log(2) if base == 2 else H

def spectral_edge_frequency(x, fs, edge=0.95):
    f, P = _welch_psd(x, fs)
    cum = np.cumsum(P, axis=0)
    totals = cum[-1, :] + EPS
    sef = np.empty(P.shape[1])
    for c in range(P.shape[1]):
        idx = np.searchsorted(cum[:, c], edge * totals[c])
        sef[c] = f[min(max(idx, 0), len(f) - 1)]
    return sef

def freq_domain_feature_vector(seg: np.ndarray, fs: float, bands=((5, 15), (15, 30))) -> np.ndarray:
    """
    Concatenate frequency-domain features per channel into a single vector.
    seg: (window_len, n_channels)
    """
    m0, m1, var = spectral_moments(seg, fs)
    tp, mf, v = m0, m1, var
    medf  = median_frequency(seg, fs)
    peakf = peak_frequency(seg, fs)
    sent  = spectral_entropy(seg, fs)
    sef95 = spectral_edge_frequency(seg, fs, 0.95)
    bp_list = [bandpower(seg, fs, band=b) for b in bands]
    feat_per_ch = np.vstack([tp, *(bp_list), mf, medf, peakf, v, sent, sef95])  # (n_feat, C)
    return feat_per_ch.ravel(order="F")

__all__ = [
    "freq_domain_feature_vector", "total_power", "bandpower", "mean_frequency",
    "median_frequency", "peak_frequency", "spectral_moments", "spectral_entropy",
    "spectral_edge_frequency", "_welch_psd", "EPS"
]