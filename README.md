#  Pneumonia Detection from Chest X-Rays

A deep learning system that automatically detects pneumonia from chest X-ray images. Built with TensorFlow and Streamlit, this project compares a **custom CNN** against **EfficientNetB0 transfer learning** and includes **Grad-CAM visual explanations** so you can see *exactly* what the model is looking at.

---

##  What This Project Does

Ever wondered how AI can assist radiologists in spotting pneumonia? This project takes a chest X-ray as input and tells you whether it shows signs of pneumonia — along with a confidence score and a heatmap that highlights the suspicious regions. No medical degree required to understand the output.

**Here's the gist:**
- Upload an X-ray → get an instant diagnosis with confidence %
- See a Grad-CAM heatmap showing *where* the model focused
- Two models trained and compared side-by-side
- A clean web interface anyone can use

---

##  Project Structure

```
Pneumonia-detection/
│
├── app/
│   └── app.py                 # Streamlit web application
│
├── src/
│   ├── preprocess.py          # Data loading, augmentation & class weights
│   ├── train.py               # Model architectures & training pipeline
│   ├── predict.py             # Single-image inference utility
│   └── gradcam.py             # Grad-CAM heatmap generation
│
├── models/                    # Saved model weights (.h5)
├── data/                      # Dataset directory (train/val/test splits)
├── requirements.txt           # Python dependencies
└── .gitignore
```

---

##  Models

### 1. Baseline CNN
A custom convolutional neural network built from scratch:
- 4 convolutional blocks with increasing filters (32 → 64 → 128 → 256)
- Batch normalization + dropout at every stage for regularization
- Global average pooling instead of flattening (fewer parameters, less overfitting)
- Trained with Adam optimizer (lr = 5e-4)

### 2. EfficientNetB0 (Transfer Learning)
Leverages a pre-trained EfficientNetB0 backbone:
- **Phase 1:** Backbone frozen — only the classification head trains (lr = 1e-3)
- **Phase 2:** Top 20 backbone layers unfrozen for fine-tuning (lr = 1e-5)
- Uses EfficientNet's native preprocessing (scaling to [-1, 1] to match ImageNet distribution)

> **Why two models?** Comparing a from-scratch model with transfer learning demonstrates the power of pre-trained features — the EfficientNet model converges faster and generalizes better with less data.

---

##  Key Features

| Feature | Details |
|---|---|
| **Data Augmentation** | Rotation, shift, shear, zoom, flip, brightness jitter |
| **Class Imbalance Handling** | Automatic class weights (the dataset is ~3:1 pneumonia-heavy) |
| **Grad-CAM Explainability** | Visual heatmaps showing model attention regions |
| **Early Stopping** | Prevents overfitting by monitoring validation loss |
| **Learning Rate Scheduling** | Reduces LR on plateau for smoother convergence |
| **Interactive Web App** | Upload an X-ray and get results in seconds |

---

##  Dataset

This project uses the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle.

**Structure:**
```
data/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

The dataset contains 5,863 chest X-ray images across two categories. The training set has an approximate 3:1 ratio of pneumonia to normal images, which is handled through computed class weights during training.

---

##  Getting Started

### Prerequisites
- Python 3.9+
- pip

### 1. Clone the repository
```bash
git clone https://github.com/KaranP315/Pneumonia-detection.git
cd Pneumonia-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and extract it into the `data/` folder so the structure matches what's shown above.

### 4. Train the models
```bash
python -m src.train --data_dir data --epochs 20 --batch_size 32
```

This will:
- Train the baseline CNN
- Train EfficientNetB0 (frozen → fine-tuned)
- Save the best weights to `models/`
- Generate comparison plots in `outputs/plots/`
- Save evaluation metrics to `outputs/results/`

### 5. Run the web app
```bash
streamlit run app/app.py
```

Open your browser and upload a chest X-ray. That's it — you'll see the prediction, confidence score, and Grad-CAM heatmap instantly.

### 6. CLI prediction (optional)
```bash
python -m src.predict --image path/to/xray.jpg --model models/efficientnet_transfer.h5
```

---

##  How Grad-CAM Works

Grad-CAM (Gradient-weighted Class Activation Mapping) answers the question: *"What part of the image made the model decide this way?"*

1. The gradients of the prediction flow back to the last convolutional layer
2. These gradients are averaged to get importance weights per feature map
3. A weighted combination of feature maps produces the final heatmap
4. The heatmap is overlaid on the original X-ray for visual interpretation

Bright regions in the heatmap = areas the model relied on most. For pneumonia cases, you'll typically see the model focusing on areas with opacities or consolidations in the lungs.

---

##  Tech Stack

- **TensorFlow / Keras** — Model building, training, and inference
- **EfficientNetB0** — Pre-trained backbone for transfer learning
- **OpenCV** — Image processing and heatmap overlay
- **Streamlit** — Interactive web application
- **Matplotlib** — Training history visualization
- **NumPy / Pillow** — Array manipulation and image handling
- **scikit-learn** — Utility functions

---

##  Pipeline Overview

```
Chest X-Ray Image
       │
       ▼
┌─────────────────┐
│  Preprocessing   │  Resize to 224×224, normalize,
│  & Augmentation  │  apply augmentations (train only)
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│  Baseline CNN    │     │  EfficientNetB0   │
│  (from scratch)  │     │  (transfer learn) │
└────────┬────────┘     └────────┬─────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│         Evaluation & Comparison          │
│  Accuracy, loss, plots, saved metrics    │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│            Streamlit Web App             │
│  Upload → Predict → Grad-CAM Heatmap    │
└─────────────────────────────────────────┘
```

---

##  Disclaimer

This project is built for **educational and demonstration purposes only**. It is not a substitute for professional medical diagnosis. Always consult a qualified healthcare provider for medical advice.

---

##  License

This project is open source and available under the [MIT License](LICENSE).

---

##  Contributing

Found a bug or want to improve something? Feel free to open an issue or submit a pull request. Contributions are always welcome.

---

*Built with ❤️ using TensorFlow and Streamlit*
