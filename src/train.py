"""Task 3: Adam, batch 64, 30 epochs, clip grad-norm 1.0, keep best val loss."""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import HARNet, count_parameters, param_breakdown

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

EPOCHS = 30
BATCH = 64
LR = 1e-3
CLIP = 1.0


def _batches(n, batch, rng):
    idx = rng.permutation(n)
    for start in range(0, n, batch):
        yield idx[start : start + batch]


@torch.no_grad()
def evaluate(model, X, y, batch=256):
    model.eval()
    total_loss = 0.0
    n_ok = 0
    n = X.shape[0]
    for start in range(0, n, batch):
        xb = X[start : start + batch]
        yb = y[start : start + batch]
        logits, _ = model(xb)
        total_loss += float(F.cross_entropy(logits, yb, reduction="sum"))
        n_ok += int((logits.argmax(dim=1) == yb).sum())
    return total_loss / n, n_ok / n


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    pack = torch.load(RESULTS / "prepared.pt", weights_only=False)
    X_tr, y_tr = pack["X_train"].to(device), pack["y_train"].to(device)
    X_va, y_va = pack["X_val"].to(device), pack["y_val"].to(device)

    torch.manual_seed(0)
    model = HARNet().to(device)
    n_params = count_parameters(model)
    print("trainable parameters:", n_params)
    print("hand count:", param_breakdown())
    if n_params != 46982:
        print("WARNING: expected 46982 parameters")

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # loss on a full pass of train before any update
    init_loss, _ = evaluate(model, X_tr, y_tr)
    print(f"initial train loss: {init_loss:.4f}  (log 6 ~ 1.7918)")

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "initial_loss": init_loss,
        "n_params": n_params,
    }

    best_val = float("inf")
    best_epoch = -1
    best_state = None
    rng = np.random.default_rng(0)
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        seen = 0
        for batch_idx in _batches(X_tr.shape[0], BATCH, rng):
            xb = X_tr[batch_idx]
            yb = y_tr[batch_idx]
            opt.zero_grad()
            logits, _ = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            running += float(loss.item()) * xb.shape[0]
            seen += xb.shape[0]
        tr_loss = running / seen
        va_loss, va_acc = evaluate(model, X_va, y_va)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        mark = ""
        if va_loss < best_val:
            best_val = va_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            mark = "  <- best val so far"
        print(
            f"epoch {epoch:02d}/{EPOCHS}  train {tr_loss:.4f}  "
            f"val {va_loss:.4f}  val acc {va_acc*100:.2f}%{mark}"
        )

    elapsed = time.time() - t0
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val
    history["seconds"] = elapsed
    history["device"] = str(device)
    history["train_loss_at_best"] = history["train_loss"][best_epoch - 1]
    history["val_acc_at_best"] = history["val_acc"][best_epoch - 1]

    RESULTS.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, RESULTS / "harnet_best.pt")
    (RESULTS / "train_history.json").write_text(json.dumps(history, indent=2))
    print(f"best epoch {best_epoch}  val loss {best_val:.4f}  time {elapsed/60:.1f} min")
    print(f"saved {RESULTS / 'harnet_best.pt'}")

    epochs = np.arange(1, EPOCHS + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(epochs, history["train_loss"], label="train")
    ax[0].plot(epochs, history["val_loss"], label="validation")
    ax[0].axvline(best_epoch, color="0.4", ls="--", lw=1)
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("cross-entropy")
    ax[0].set_title("loss")
    ax[0].legend()
    ax[1].plot(epochs, [a * 100 for a in history["val_acc"]], color="C2")
    ax[1].axvline(best_epoch, color="0.4", ls="--", lw=1)
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("accuracy (%)")
    ax[1].set_title("validation accuracy")
    fig.tight_layout()
    fig.savefig(RESULTS / "training_curves.png", dpi=140)
    plt.close(fig)
    print("saved results/training_curves.png")


if __name__ == "__main__":
    train()
