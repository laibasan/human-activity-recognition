"""Dump HARNet weights into JSON so the GitHub Pages page can run without PyTorch."""

import json
from pathlib import Path
import torch
from model import HARNet, count_parameters

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
APP = ROOT / "app"


def tensor_list(t):
    return t.detach().cpu().tolist()


def export():
    torch.manual_seed(0)
    model = HARNet()
    state = torch.load(RESULTS / "harnet_best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    sd = model.state_dict()
    weights = {k: tensor_list(v) for k, v in sd.items()}
    scaler = json.loads((RESULTS / "scaler.json").read_text())
    bundle = {
        "n_params": count_parameters(model),
        "hidden": 64,
        "input": 9,
        "seq": 128,
        "weights": weights,
        "mean": scaler["mean"],
        "std": scaler["std"],
        "channels": scaler["channels"],
        "activity_names": scaler["activity_names"],
    }
    APP.mkdir(parents=True, exist_ok=True)
    (APP / "weights.js").write_text(
        "window.HARNET = " + json.dumps(bundle) + ";\n"
    )
    samples = json.loads((RESULTS / "web_samples.json").read_text())
    (APP / "samples.js").write_text("window.HAR_SAMPLES = " + json.dumps(samples) + ";\n")
    print("wrote app/weights.js and app/samples.js")


if __name__ == "__main__":
    export()
