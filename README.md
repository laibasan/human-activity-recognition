# ✨ Starry Night — Human Activity Recognition

> *A deep learning project dressed in Van Gogh's swirling night sky*

![Van Gogh Starry Night](https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg)

## 🌙 Live Demo & Research Poster

- **[→ Open the Starry Night HAR Demo](https://laibasan.github.io/human-activity-recognition/)**
- **[→ View Interactive Research Poster (HTML / Printable)](https://laibasan.github.io/human-activity-recognition/poster.html)**
- **[→ High-Resolution Poster Image (PNG)](https://raw.githubusercontent.com/laibasan/human-activity-recognition/main/results/poster.png)**

## 📖 About

This project classifies human activities from smartphone sensor data using a **Bidirectional LSTM with Attention**, trained on the [UCI HAR Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones).

The web interface features an animated **Van Gogh Starry Night** theme — complete with swirling nebulae, twinkling stars, a crescent moon, cypress trees, and a distant village — all rendered live on an HTML5 canvas.

### Activities Recognized
| Activity | Description |
|----------|-------------|
| 🚶 Walking | Level ground walking |
| 🔼 Walking Upstairs | Going up stairs |
| 🔽 Walking Downstairs | Coming down stairs |
| 🪑 Sitting | Seated position |
| 🧍 Standing | Upright standing |
| 🛏️ Laying | Lying down |

## 🏗️ Architecture

```
Input (128 steps × 9 channels)
    ↓
BiLSTM (forward + backward, hidden=64)
    ↓
Attention (64-dim, learned weights)
    ↓
Linear → 6 classes
```

- **46,982** trainable parameters
- Custom LSTM cells (no `nn.LSTM`)
- Attention mechanism highlights which time steps matter most
- Runs entirely in the browser via hand-ported JavaScript inference

## 📁 Project Structure

```
├── app/                    # Web interface (GitHub Pages)
│   ├── index.html          # Starry Night themed page
│   ├── style.css           # Van Gogh color palette & glass panels
│   ├── sky.js              # Animated night sky canvas
│   ├── infer.js            # BiLSTM inference in pure JavaScript
│   ├── ui.js               # UI logic & chart rendering
│   ├── weights.js          # Exported model weights
│   └── samples.js          # Test samples for demo
├── src/                    # Python training pipeline
│   ├── download_data.py    # Fetch UCI HAR dataset
│   ├── load_data.py        # Parse raw sensor files
│   ├── plot_windows.py     # Visualize example windows
│   ├── prepare.py          # Train/val/test split & standardization
│   ├── layers.py           # Custom RNN/LSTM/BiLSTM/Attention
│   ├── model.py            # HARNet definition
│   ├── metrics.py          # Accuracy & confusion matrix
│   ├── train.py            # Training loop (30 epochs, Adam)
│   ├── test_model.py       # Evaluation & attention analysis
│   ├── export_web.py       # Export weights to JS
│   └── write_report.py     # Generate training report
├── results/                # Training outputs & plots
├── run_all.py              # Run the full pipeline
└── requirements.txt        # torch, numpy, matplotlib
```

## 🚀 Quick Start

### Train from scratch
```bash
pip install -r requirements.txt
python run_all.py
```

### View the demo locally
```bash
cd app
python serve.py
# Open http://localhost:8000
```

## 🎨 Interface Features

- **Animated Starry Night** background with swirling nebulae
- **Glass-morphism panels** with golden borders
- **Live sensor visualization** — see the 9-channel accelerometer/gyroscope data
- **Attention heatmap** — see which time steps the model focuses on
- **Probability bars** with gradient fills
- **CSV upload** — test with your own sensor data
- **Fully responsive** design

## 📊 Dataset

**UCI Human Activity Recognition Using Smartphones**
- 30 volunteers, ages 19-48
- 6 activities performed wearing a Samsung Galaxy S II
- 50Hz sampling rate
- 128-step sliding windows (2.56 seconds)
- 9 raw inertial signals (body acc, gyro, total acc × xyz)

## 📝 License

Academic project — Applied Machine Learning, Assignment 4.

---

*Interface inspired by The Starry Night (1889) — Vincent van Gogh*
