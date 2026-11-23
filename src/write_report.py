"""Generate a summary report from training and test results."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def write_report():
    # Load training history
    history_path = RESULTS / "train_history.json"
    if not history_path.exists():
        print("No train_history.json found — skipping report.")
        return
    history = json.loads(history_path.read_text())

    # Load test results
    test_path = RESULTS / "test_results.json"
    test = json.loads(test_path.read_text()) if test_path.exists() else None

    lines = []
    lines.append("# Human Activity Recognition — Training Report")
    lines.append("")
    lines.append("## Model")
    lines.append(f"- Architecture: BiLSTM + Attention → Linear(128, 6)")
    lines.append(f"- Trainable parameters: {history['n_params']:,}")
    lines.append(f"- Device: {history.get('device', 'unknown')}")
    lines.append("")
    lines.append("## Training")
    lines.append(f"- Initial loss: {history['initial_loss']:.4f}  (log 6 ≈ 1.7918)")
    lines.append(f"- Best epoch: {history['best_epoch']}")
    lines.append(f"- Best validation loss: {history['best_val_loss']:.4f}")
    lines.append(f"- Train loss at best: {history['train_loss_at_best']:.4f}")
    lines.append(f"- Validation accuracy at best: {history['val_acc_at_best']*100:.2f}%")
    lines.append(f"- Training time: {history['seconds']/60:.1f} minutes")
    lines.append("")

    if test:
        lines.append("## Test Results")
        lines.append(f"- Test accuracy: {test['test_accuracy']*100:.2f}%")
        lines.append(f"- Test samples: {test['n_test']}")
        lines.append("")
        lines.append("### Per-Class Accuracy")
        for name, acc in test["per_class_accuracy"].items():
            lines.append(f"  - {name:20s}  {acc*100:6.2f}%")
        lines.append("")
        lines.append("### Attention Peakedness (mean of max alpha)")
        for name, peak in test["mean_max_alpha"].items():
            lines.append(f"  - {name:20s}  {peak:.4f}")

    report = "\n".join(lines) + "\n"
    report_path = RESULTS / "report.md"
    report_path.write_text(report)
    print(f"wrote {report_path}")


if __name__ == "__main__":
    write_report()
