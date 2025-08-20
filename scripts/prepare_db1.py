"""
Prepare DB1 by iterating subject .mat files, optional filtering, and windowing.
Outputs numpy arrays under artifacts/ (X.npy, y.npy) for quick baselines.
"""
from pathlib import Path
import argparse, json
import numpy as np
from emg_insight.datasets.ninapro_db1 import load_recording
from emg_insight.preprocessing.filters import notch_filter, bandpass_filter, rectify_and_normalize
from emg_insight.utils.windows import sliding_window
import yaml

def main(args):
    base = Path(args.data_dir)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.config).read_text())
    fs = cfg.get("fs_hz", 100.0)
    win_ms = cfg.get("window_ms", 200)
    step_ms = cfg.get("step_ms", 50)
    notch = cfg.get("notch_hz", 50)
    bp = cfg.get("bandpass_hz", [20, 450])

    X_all, y_all = [], []
    mats = sorted(base.rglob("*.mat"))
    for m in mats:
        X, y = load_recording(str(m))
        # Filters
        if notch:
            X = notch_filter(X, fs, f0=float(notch))
        if bp:
            X = bandpass_filter(X, fs, low=float(bp[0]), high=float(bp[1]))
        X = rectify_and_normalize(X)

        # Windowing
        win = int(win_ms * fs / 1000)
        step = int(step_ms * fs / 1000)
        Xw, yw = sliding_window(X, y, win, step)
        # Flatten windows for classical ML
        Xf = Xw.reshape(Xw.shape[0], -1)
        X_all.append(Xf); y_all.append(yw)

    if not X_all:
        print("No .mat files found under", base)
        return

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    np.save(out / "X_db1.npy", X)
    np.save(out / "y_db1.npy", y)
    (out / "meta.json").write_text(json.dumps({"fs": fs, "win_ms": win_ms, "step_ms": step_ms, "n": int(X.shape[0])}, indent=2))
    print("Saved:", out / "X_db1.npy", out / "y_db1.npy")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/ninapro/db1")
    p.add_argument("--out_dir", type=str, default="artifacts/db1")
    p.add_argument("--config", type=str, default="configs/filter_config.yaml")
    main(p.parse_args())
