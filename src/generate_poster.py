"""Generate a high-resolution academic conference poster for the HAR project."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

def create_poster():
    # 48x32 inches at 100 dpi = 4800x3200 pixels (Ultra-HD 4K+ quality)
    fig = plt.figure(figsize=(24, 16), dpi=150)
    fig.patch.set_facecolor("#071120") # Deep midnight blue
    
    # ---------------- HEADER BANNER ----------------
    # Top banner rectangle
    header_ax = fig.add_axes([0.02, 0.88, 0.96, 0.10])
    header_ax.set_facecolor("#0d1e38")
    header_ax.set_xticks([])
    header_ax.set_yticks([])
    for spine in header_ax.spines.values():
        spine.set_color("#f2c744")
        spine.set_linewidth(2)
        
    header_ax.text(0.5, 0.72, "RECOGNIZING HUMAN ACTIVITY FROM PHONE MOTION SENSORS", 
                   ha="center", va="center", color="#ffe58a", fontsize=24, fontweight="bold", family="sans-serif")
    header_ax.text(0.5, 0.42, "Custom Bidirectional LSTM with Temporal Attention Built from Scratch in PyTorch", 
                   ha="center", va="center", color="#8fd0e8", fontsize=15, fontstyle="italic", family="sans-serif")
    header_ax.text(0.5, 0.16, "Applied Machine Learning (Fall 2026)  |  UCI HAR Benchmark  |  Overall Test Accuracy: 91.45%  |  46,982 Trainable Parameters", 
                   ha="center", va="center", color="#b7c6dc", fontsize=11, family="sans-serif")

    # Helper function to create stylish card panel
    def make_panel(ax, title):
        ax.set_facecolor("#0b192e")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#2a5a9a")
            spine.set_linewidth(1.5)
        ax.text(0.04, 0.94, title, transform=ax.transAxes, color="#f2c744", 
                fontsize=13, fontweight="bold", va="top")
        # Gold underline for title
        line = patches.Rectangle((0.04, 0.90), 0.92, 0.005, transform=ax.transAxes, color="#f2c744", alpha=0.6)
        ax.add_patch(line)

    # ---------------- COLUMN 1: DATA & SIGNALS ----------------
    # Panel 1.1: Problem & Inertial Signals
    ax1_1 = fig.add_axes([0.02, 0.54, 0.30, 0.32])
    make_panel(ax1_1, "1. PROBLEM & THE 9 INERTIAL SIGNALS")
    
    col1_text = (
        "• Problem: Classify 6 human activities from 2.56-second sliding windows\n"
        "  (128 samples at 50 Hz) captured by waist-worn smartphone sensors.\n"
        "• 30 Volunteers (aged 19–48) wearing a Samsung Galaxy S II.\n\n"
        "The 9 Raw Input Channels (xt ∈ ℝ⁹):\n"
        "  1-3. Body Acc (x, y, z): Linear body motion with gravity removed (g)\n"
        "  4-6. Body Gyro (x, y, z): Angular velocity around three axes (rad/s)\n"
        "  7-9. Total Acc (x, y, z): Acceleration including Earth's 1g gravity (g)\n\n"
        "Moving vs. Still Dichotomy:\n"
        "  • Dynamic Activities (Walk, Stairs): High amplitude periodic waves (±0.5g).\n"
        "  • Static Postures (Sit, Stand, Lay): Flat traces near zero dynamic acceleration.\n\n"
        "Subject-Independent Validation Partitioning:\n"
        "  • 50% window overlap (1.28 s) causes data leakage if split randomly!\n"
        "  • Solution: Split by volunteer to evaluate unseen human gaits.\n"
        "  • Train: 17 subjects (5,881 windows)  |  Val: 4 subjects (1,471 windows)\n"
        "  • Test: 9 subjects (2,947 windows)  |  Standardized strictly on Train mean/std."
    )
    ax1_1.text(0.05, 0.85, col1_text, transform=ax1_1.transAxes, color="#e8eef8", fontsize=9.2, va="top", family="sans-serif", linespacing=1.35)

    # Panel 1.2: Raw Sensor Signals Plot
    ax1_2 = fig.add_axes([0.02, 0.03, 0.30, 0.49])
    make_panel(ax1_2, "2. INERTIAL WAVEFORMS (BODY ACC & GYRO)")
    ex_img = Image.open(RESULTS / "example_windows.png")
    # Sub-axes for image
    img_ax1 = fig.add_axes([0.03, 0.16, 0.28, 0.31])
    img_ax1.imshow(ex_img)
    img_ax1.axis("off")
    
    caption1 = (
        "Figure 1: Example 128-step windows across the six activities (Task 1.3).\n"
        "Notice the periodic gait spikes in dynamic activities versus the completely\n"
        "quiescent flat-lines in sitting, standing, and laying postures."
    )
    ax1_2.text(0.05, 0.22, caption1, transform=ax1_2.transAxes, color="#b7c6dc", fontsize=8.8, va="top", style="italic")

    # ---------------- COLUMN 2: ARCHITECTURE & TRAINING ----------------
    # Panel 2.1: Model Architecture from First Principles
    ax2_1 = fig.add_axes([0.34, 0.48, 0.32, 0.38])
    make_panel(ax2_1, "3. ARCHITECTURE BUILT FROM SCRATCH")
    
    arch_text = (
        "Strict Requirement: NO black-box nn.LSTM or nn.MultiheadAttention.\n"
        "All recurrent layers implemented from fundamental mathematical equations:\n\n"
        "1. Custom LSTMCell (4 gates packed into unified W, U, b):\n"
        "   [it, ft, gt, ot]ᵀ = [σ, σ, tanh, σ](W xt + U ht-1 + b)\n"
        "   ct = ft ⊙ ct-1 + it ⊙ gt      (additive memory cell highway)\n"
        "   ht = ot ⊙ tanh(ct)            (output hidden state, H = 64)\n\n"
        "2. Bidirectional LSTM (BiLSTM):\n"
        "   Forward LSTM reads x₁ → x₁₂₈; Backward reads x₁₂₈ → x₁.\n"
        "   Reversed backward outputs aligned with forward: ht = [→ht ; ←ht] ∈ ℝ¹²⁸\n\n"
        "3. Temporal Attention Pooling:\n"
        "   et = vᵀ tanh(Wa ht + ba)     (Wa ∈ ℝ⁶⁴ˣ¹²⁸, ba, v ∈ ℝ⁶⁴)\n"
        "   αt = softmax(et)             (normalized temporal weight)\n"
        "   c = ∑ αt ht ∈ ℝ¹²⁸            (context vector) → Linear(128, 6)\n\n"
        "Exact Parameter Hand Count: 46,982 Trainable Parameters\n"
        "  • BiLSTM: 2 × [4×64×9 + 4×64×64 + 4×64] = 37,888\n"
        "  • Attention: 64×128 + 64 + 64 = 8,320  |  Linear: 128×6 + 6 = 774"
    )
    ax2_1.text(0.05, 0.85, arch_text, transform=ax2_1.transAxes, color="#e8eef8", fontsize=9.0, va="top", family="sans-serif", linespacing=1.3)

    # Panel 2.2: Training Curves
    ax2_2 = fig.add_axes([0.34, 0.03, 0.32, 0.43])
    make_panel(ax2_2, "4. TRAINING DYNAMICS (30 EPOCHS, ADAM)")
    train_img = Image.open(RESULTS / "training_curves.png")
    img_ax2 = fig.add_axes([0.35, 0.12, 0.30, 0.28])
    img_ax2.imshow(train_img)
    img_ax2.axis("off")
    
    caption2 = (
        "Figure 2: Cross-entropy loss and validation accuracy curves (Task 3.2).\n"
        "• Initial Loss: 1.8049 (matching theoretical random chance ln 6 ≈ 1.7918).\n"
        "• Best Checkpoint: Epoch 5 | Val Loss: 0.1208 | Val Accuracy: 96.44%.\n"
        "• Gradient Clipping (norm 1.0) prevented gradient explosion."
    )
    ax2_2.text(0.05, 0.18, caption2, transform=ax2_2.transAxes, color="#b7c6dc", fontsize=8.8, va="top", style="italic")

    # ---------------- COLUMN 3: RESULTS, ATTENTION & TASK 6 ----------------
    # Panel 3.1: Results & Confusion Matrix
    ax3_1 = fig.add_axes([0.68, 0.50, 0.30, 0.36])
    make_panel(ax3_1, "5. TEST RESULTS & CONFUSION MATRIX")
    
    cm_img = Image.open(RESULTS / "confusion_matrix.png")
    img_ax3 = fig.add_axes([0.69, 0.52, 0.14, 0.29])
    img_ax3.imshow(cm_img)
    img_ax3.axis("off")
    
    perf_text = (
        "Overall Test Accuracy: 91.45%\n"
        "(2,695 / 2,947 Correct Windows)\n\n"
        "Per-Class Test Accuracy:\n"
        "  • WALKING: 87.70%\n"
        "  • UPSTAIRS: 98.30%\n"
        "  • DOWNSTAIRS: 97.38%\n"
        "  • SITTING: 74.75%\n"
        "  • STANDING: 90.98%\n"
        "  • LAYING: 100.00%\n\n"
        "Confusion Analysis (Task 6.2):\n"
        "Sitting and Standing cause almost all\n"
        "errors (122 sits misclassified as stands)\n"
        "due to identical waist gravity angle\n"
        "and zero dynamic acceleration."
    )
    ax3_1.text(0.51, 0.85, perf_text, transform=ax3_1.transAxes, color="#e8eef8", fontsize=8.8, va="top", family="sans-serif", linespacing=1.25)

    # Panel 3.2: Attention Analysis & Task 6 Summary
    ax3_2 = fig.add_axes([0.68, 0.03, 0.30, 0.45])
    make_panel(ax3_2, "6. ATTENTION INTERPRETABILITY & TASK 6")
    
    attn_img = Image.open(RESULTS / "attention_mean.png")
    img_ax4 = fig.add_axes([0.69, 0.23, 0.28, 0.19])
    img_ax4.imshow(attn_img)
    img_ax4.axis("off")
    
    task6_text = (
        "Task 6 Analytical Findings:\n"
        "• Attention Peakedness (6.4): Walking peaks sharply (mean max αt = 0.2978),\n"
        "  focusing ~30% weight on foot-strike shocks. Static postures diffuse evenly\n"
        "  (0.010 - 0.018 ≈ flat 1/128) because signals are time-invariant.\n"
        "• 100% Laying Accuracy (6.3): Gravity rotates 90° onto total_acc_x axis.\n"
        "• BiLSTM Constraint (6.5): Allowed in offline windowing; disallowed in causal\n"
        "  zero-latency streaming (fall triggers, autoregressive forecasting).\n"
        "• Error Carousel (6.6): ct = ft⊙ct-1 + it⊙gt prevents vanishing gradients over 128 steps.\n\n"
        "🌟 Starry Night Web App: Live client-side pure JS inference (infer.js)\n"
        "Demo: laibasan.github.io/human-activity-recognition/  |  GitHub: laibasan"
    )
    ax3_2.text(0.04, 0.38, task6_text, transform=ax3_2.transAxes, color="#e8eef8", fontsize=8.2, va="top", family="sans-serif", linespacing=1.25)

    # Save outputs
    out_png = RESULTS / "poster.png"
    fig.savefig(out_png, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved high-resolution poster to {out_png}")
    
    # Also save to communication/poster.png
    comm_png = ROOT / "communication" / "poster.png"
    comm_png.write_bytes(out_png.read_bytes())
    print(f"Saved copy to {comm_png}")

if __name__ == "__main__":
    create_poster()
