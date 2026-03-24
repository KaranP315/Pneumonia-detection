"""
predict.py
----------
Utility for running inference on a single chest X-ray image.

Usage:
    python src/predict.py --image path/to/xray.jpeg --model models/efficientnet_transfer.h5
"""

import argparse
import numpy as np
from tensorflow.keras.models import load_model

from src.preprocess import load_single_image, IMG_SIZE

# Human-readable labels matching the generator's class indices
# (ImageDataGenerator alphabetically: NORMAL=0, PNEUMONIA=1)
LABELS = {0: "Normal", 1: "Pneumonia"}


def predict(model, img_array: np.ndarray):
    """
    Run a forward pass and return the predicted label + confidence.

    Parameters
    ----------
    model : keras.Model
    img_array : np.ndarray  shape (1, H, W, 3), already normalised to [0, 1]

    Returns
    -------
    label : str   ("Normal" or "Pneumonia")
    confidence : float  (0-1)
    """
    prob = float(model.predict(img_array, verbose=0)[0][0])
    label_idx = int(prob >= 0.5)
    confidence = prob if label_idx == 1 else 1 - prob
    return LABELS[label_idx], confidence


def main():
    parser = argparse.ArgumentParser(description="Predict pneumonia from a chest X-ray")
    parser.add_argument("--image", type=str, required=True, help="Path to X-ray image")
    parser.add_argument("--model", type=str, default="models/efficientnet_transfer.h5",
                        help="Path to trained .h5 model")
    args = parser.parse_args()

    model = load_model(args.model)
    img = load_single_image(args.image, IMG_SIZE)
    label, conf = predict(model, img)

    print(f"Prediction : {label}")
    print(f"Confidence : {conf:.2%}")


if __name__ == "__main__":
    main()
