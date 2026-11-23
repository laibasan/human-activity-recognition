from pathlib import Path
import numpy as np

# The 9 raw signal files, in the channel order used everywhere in this assignment.
CHANNELS = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]


def _find_root(data_dir):
    """Find the dataset folder (the one containing train/ and test/),
    whatever the unzipped folder is called."""
    for d in sorted(Path(data_dir).rglob("Inertial Signals")):
        if "__MACOSX" in d.parts:
            continue
        if d.parent.name == "train" and (d / "body_acc_x_train.txt").exists():
            return d.parent.parent
    raise FileNotFoundError(
        f"No 'train/Inertial Signals' folder under {data_dir}. "
        "Unzip the dataset (and any zip inside it) first."
    )


def load_raw(data_dir="data"):
    """Return {'train': {...}, 'test': {...}, 'activity_names': [...]}.

    Each split is a dict with
    'X' : (N, 128, 9) float32, the raw signals, channels in CHANNELS order
    'y' : (N,) int, activity labels exactly as stored (1..6)
    'subject' : (N,) int, the volunteer who produced each window (1..30)
    Nothing is scaled, split or relabelled here.
    """
    root = _find_root(data_dir)
    out = {}
    for split in ["train", "test"]:
        sig = root / split / "Inertial Signals"
        X = np.stack(
            [np.loadtxt(sig / f"{c}_{split}.txt", dtype=np.float32) for c in CHANNELS],
            axis=-1,
        )
        y = np.loadtxt(root / split / f"y_{split}.txt", dtype=int)
        subject = np.loadtxt(root / split / f"subject_{split}.txt", dtype=int)
        out[split] = {"X": X, "y": y, "subject": subject}
    names = [
        line.split()[1]
        for line in (root / "activity_labels.txt").read_text().splitlines()
        if line.strip()
    ]
    out["activity_names"] = names
    return out


if __name__ == "__main__":
    raw = load_raw()
    for split in ["train", "test"]:
        d = raw[split]
        print(
            f"{split}: X {d['X'].shape}, y {d['y'].shape}, subjects {sorted(set(d['subject'].tolist()))}"
        )
    print("activities:", raw["activity_names"])
