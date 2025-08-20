from __future__ import annotations
import numpy as np

def sliding_window(x: np.ndarray, y: np.ndarray, *, window_s: float, step_s: float, fs: float):
    """
    Segment into overlapping windows and assign a label per window
    via majority vote (preferring non-zero).
    Returns:
      Xw: (n_windows, win, n_channels)
      Yw: (n_windows,)
    """
    win, step = int(window_s * fs), int(step_s * fs)
    Xw, Yw = [], []
    for start in range(0, len(x) - win + 1, step):
        seg = x[start:start + win]
        lab = y[start:start + win]
        vals, counts = np.unique(lab, return_counts=True)
        idx = counts.argsort()[::-1]
        chosen = 0
        for j in idx:
            if vals[j] != 0:
                chosen = vals[j]; break
        Xw.append(seg)
        Yw.append(chosen)
    Xw = np.stack(Xw) if Xw else np.empty((0, win, x.shape[1]), dtype=np.float32)
    Yw = np.array(Yw, dtype=np.int32)
    return Xw, Yw

__all__ = ["sliding_window"]