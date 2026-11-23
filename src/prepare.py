"""Task 1: labels 0-5, subject-wise validation split, channel standardisation."""

from pathlib import Path
import json
import numpy as np
import torch
from load_data import load_raw, CHANNELS

VAL_SUBJECTS = [3, 15, 25, 30]
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _count_by_activity(y, names, offset=0):
    counts = {}
    for k, name in enumerate(names):
        counts[name] = int(np.sum(y == (k + offset)))
    return counts


def prepare(data_dir="data"):
    raw = load_raw(data_dir)
    names = raw["activity_names"]

    print("shapes after load_raw:")
    for split in ["train", "test"]:
        d = raw[split]
        print(f"  {split}: X={d['X'].shape} y={d['y'].shape} subject={d['subject'].shape}")

    # labels in the files are 1..6; we use 0..5 everywhere after this
    y_train_raw = raw["train"]["y"].copy()
    y_test_raw = raw["test"]["y"].copy()
    y_train = y_train_raw - 1
    y_test = y_test_raw - 1

    print("\nwindows per activity (train, labels still 1-6):")
    train_counts = _count_by_activity(y_train_raw, names, offset=1)
    test_counts = _count_by_activity(y_test_raw, names, offset=1)
    for name in names:
        print(f"  {name:20s} train={train_counts[name]:5d}  test={test_counts[name]:5d}")
    print(f"  totals: train={len(y_train)} test={len(y_test)}")

    X_all = raw["train"]["X"]
    subj = raw["train"]["subject"]
    is_val = np.isin(subj, VAL_SUBJECTS)
    X_tr, y_tr, s_tr = X_all[~is_val], y_train[~is_val], subj[~is_val]
    X_va, y_va, s_va = X_all[is_val], y_train[is_val], subj[is_val]
    X_te, y_te, s_te = raw["test"]["X"], y_test, raw["test"]["subject"]

    print("\nsplit by person (VAL_SUBJECTS = [3, 15, 25, 30]):")
    print(f"  train windows: {len(y_tr)}  volunteers: {sorted(set(s_tr.tolist()))}")
    print(f"  val   windows: {len(y_va)}  volunteers: {sorted(set(s_va.tolist()))}")
    print(f"  test  windows: {len(y_te)}  volunteers: {sorted(set(s_te.tolist()))}")

    # mean/std over every time step of every training window, per channel
    # X_tr is (N, 128, 9)
    mean = X_tr.reshape(-1, 9).mean(axis=0)
    std = X_tr.reshape(-1, 9).std(axis=0)
    # avoid a divide-by-zero if some channel is somehow constant
    std = np.where(std < 1e-8, 1.0, std)

    print("\nchannel mean / std (training windows only):")
    for i, c in enumerate(CHANNELS):
        print(f"  {c:13s}  mean={mean[i]: .6f}  std={std[i]: .6f}")

    def scale(X):
        return (X - mean) / std

    X_tr_s, X_va_s, X_te_s = scale(X_tr), scale(X_va), scale(X_te)

    tensors = {
        "X_train": torch.from_numpy(X_tr_s.astype(np.float32)),
        "y_train": torch.from_numpy(y_tr.astype(np.int64)),
        "X_val": torch.from_numpy(X_va_s.astype(np.float32)),
        "y_val": torch.from_numpy(y_va.astype(np.int64)),
        "X_test": torch.from_numpy(X_te_s.astype(np.float32)),
        "y_test": torch.from_numpy(y_te.astype(np.int64)),
    }
    for k, t in tensors.items():
        print(f"  {k}: {tuple(t.shape)} {t.dtype}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    scaler = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "channels": CHANNELS,
        "activity_names": names,
        "val_subjects": VAL_SUBJECTS,
        "n_train": int(len(y_tr)),
        "n_val": int(len(y_va)),
        "n_test": int(len(y_te)),
        "train_subjects": sorted(set(int(s) for s in s_tr)),
        "val_subjects_present": sorted(set(int(s) for s in s_va)),
        "test_subjects": sorted(set(int(s) for s in s_te)),
        "train_counts_raw": train_counts,
        "test_counts_raw": test_counts,
    }
    (RESULTS / "scaler.json").write_text(json.dumps(scaler, indent=2))
    torch.save(
        {
            **tensors,
            "subject_train": torch.from_numpy(s_tr.astype(np.int64)),
            "subject_val": torch.from_numpy(s_va.astype(np.int64)),
            "subject_test": torch.from_numpy(s_te.astype(np.int64)),
            "mean": torch.from_numpy(mean.astype(np.float32)),
            "std": torch.from_numpy(std.astype(np.float32)),
            "names": names,
        },
        RESULTS / "prepared.pt",
    )
    print(f"\nwrote {RESULTS / 'prepared.pt'} and scaler.json")
    return tensors, scaler, names


if __name__ == "__main__":
    prepare()
