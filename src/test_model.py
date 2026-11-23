"""Tasks 4 and 5: test once, then look at the attention weights."""

import json
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import HARNet
from metrics import accuracy, confusion_matrix, per_class_accuracy
from load_data import CHANNELS

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FS = 50
SHORT = ["Walk", "Up", "Down", "Sit", "Stand", "Lay"]


@torch.no_grad()
def predict_all(model, X, batch=256):
    model.eval()
    logits_all = []
    alpha_all = []
    for start in range(0, X.shape[0], batch):
        logits, alpha = model(X[start : start + batch])
        logits_all.append(logits.cpu())
        alpha_all.append(alpha.cpu())
    logits = torch.cat(logits_all, dim=0)
    alpha = torch.cat(alpha_all, dim=0)
    pred = logits.argmax(dim=1)
    return logits, pred, alpha


def _heatmap(C, names, path):
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(C, cmap="Blues")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            ax.text(j, i, str(C[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _attention_examples(X_raw, pred, y, alpha, names, path):
    """One correctly classified test window per activity."""
    t = np.arange(128) / FS
    fig, axes = plt.subplots(2, 6, figsize=(18, 4.8), sharex=True)
    picks = {}
    for k, name in enumerate(names):
        hits = np.flatnonzero((y == k) & (pred == k))
        if len(hits) == 0:
            hits = np.flatnonzero(y == k)
        i = int(hits[0])
        picks[name] = i
        ax_sig = axes[0, k]
        ax_a = axes[1, k]
        for c in range(3):
            ax_sig.plot(t, X_raw[i, :, c], lw=0.9, label=CHANNELS[c] if k == 0 else None)
        ax_sig.set_title(name.replace("_", " ").title(), fontsize=9)
        ax_a.fill_between(t, alpha[i], color="0.3", alpha=0.85)
        ax_a.set_xlabel("time (s)")
        if k == 0:
            ax_sig.set_ylabel("body acc (g)")
            ax_a.set_ylabel("alpha")
            ax_sig.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return picks


def _mean_attention(alpha, y, names, path):
    t = np.arange(128) / FS
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    peak_means = []
    for k, name in enumerate(names):
        mask = y == k
        mean_a = alpha[mask].mean(axis=0)
        peak_means.append(float(alpha[mask].max(axis=1).mean()))
        ax.plot(t, mean_a, label=name.replace("_", " ").title())
    ax.axhline(1 / 128, color="0.5", ls=":", lw=1, label="flat 1/128")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mean attention")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return peak_means


def test():
    pack = torch.load(RESULTS / "prepared.pt", weights_only=False)
    X = pack["X_test"]
    y = pack["y_test"].numpy()
    names = pack["names"]
    mean = pack["mean"].numpy()
    std = pack["std"].numpy()
    # unstandardise only the body-acc channels for the plots
    X_raw = X.numpy() * std + mean

    torch.manual_seed(0)
    model = HARNet()
    state = torch.load(RESULTS / "harnet_best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    logits, pred_t, alpha_t = predict_all(model, X)
    pred = pred_t.numpy()
    alpha = alpha_t.numpy()
    probs = torch.softmax(logits, dim=1).numpy()

    acc = accuracy(y, pred)
    C = confusion_matrix(y, pred, n_classes=6)
    pca = per_class_accuracy(C)
    print(f"test accuracy: {acc*100:.2f}%   ({int((pred==y).sum())}/{len(y)})")
    print("confusion matrix (rows true, cols pred):")
    print(C)
    print("row sums:", C.sum(axis=1), "total", int(C.sum()))
    print("per-class accuracy:")
    for k, name in enumerate(names):
        print(f"  {name:20s}  {pca[k]*100:6.2f}%")

    _heatmap(C, [n.replace("_", "\n") for n in names], RESULTS / "confusion_matrix.png")
    picks = _attention_examples(X_raw, pred, y, alpha, names, RESULTS / "attention_examples.png")
    peak_means = _mean_attention(alpha, y, names, RESULTS / "attention_mean.png")
    print("mean of max alpha (how peaked):")
    for k, name in enumerate(names):
        print(f"  {name:20s}  {peak_means[k]:.4f}   (flat would be {1/128:.4f})")

    report = {
        "test_accuracy": acc,
        "confusion_matrix": C.tolist(),
        "per_class_accuracy": {names[k]: float(pca[k]) for k in range(6)},
        "mean_max_alpha": {names[k]: peak_means[k] for k in range(6)},
        "n_test": int(len(y)),
        "example_indices": picks,
        "matrix_total": int(C.sum()),
    }
    (RESULTS / "test_results.json").write_text(json.dumps(report, indent=2))

    # a small bundle for the web app: a few real test windows + the weights
    rng = np.random.default_rng(1)
    samples = []
    for k, name in enumerate(names):
        idx = np.flatnonzero(y == k)
        take = rng.choice(idx, size=min(8, len(idx)), replace=False)
        for i in take:
            samples.append(
                {
                    "i": int(i),
                    "activity": name,
                    "y": int(y[i]),
                    "pred": int(pred[i]),
                    "X": X[i].numpy().round(6).tolist(),  # already standardised
                    "X_raw": X_raw[i].round(6).tolist(),
                    "alpha": alpha[i].round(6).tolist(),
                    "probs": probs[i].round(6).tolist(),
                }
            )
    (RESULTS / "web_samples.json").write_text(json.dumps({"samples": samples, "names": names}))
    print("saved test_results.json and web_samples.json")


if __name__ == "__main__":
    test()
