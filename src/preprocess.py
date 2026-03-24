"""
preprocess.py
-------------
Handles dataset loading and image augmentation for the chest X-ray dataset.

Key design decisions:
  - CNN uses simple rescale=1/255 normalisation.
  - EfficientNetB0 uses its own `preprocess_input` (scales to [-1, 1] range
    that matches the ImageNet preprocessing the backbone was trained with).
    Using the wrong preprocessing is the #1 reason transfer-learning accuracy
    tanks — the pretrained weights expect a specific input distribution.
  - Class weights are computed from the training set to handle the ~3:1
    pneumonia-to-normal imbalance.

Expected folder layout (Kaggle "Chest X-Ray Images (Pneumonia)" dataset):
    data/
      train/
        NORMAL/
        PNEUMONIA/
      val/
        NORMAL/
        PNEUMONIA/
      test/
        NORMAL/
        PNEUMONIA/
"""

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess

# ─── Default hyperparams ───────────────────────────────────────────────
IMG_SIZE = (224, 224)
BATCH_SIZE = 32


# ======================================================================
#  Generator builders
# ======================================================================

def build_generators(data_dir: str,
                     img_size: tuple = IMG_SIZE,
                     batch_size: int = BATCH_SIZE,
                     model_type: str = "cnn"):
    """
    Create Keras ImageDataGenerators for train / val / test splits.

    Parameters
    ----------
    model_type : str
        "cnn"  → rescale 1/255  (standard for custom networks)
        "efficientnet" → use EfficientNet's own preprocess_input

    Returns
    -------
    train_gen, val_gen, test_gen : DirectoryIterators
    """

    if model_type == "efficientnet":
        # EfficientNet expects its own preprocessing (scales to [-1, 1])
        train_augmentor = ImageDataGenerator(
            preprocessing_function=effnet_preprocess,
            rotation_range=25,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            zoom_range=0.25,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode="nearest",
        )
        simple_rescaler = ImageDataGenerator(
            preprocessing_function=effnet_preprocess,
        )
    else:
        # Standard CNN: simple rescale to [0, 1]
        train_augmentor = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=25,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            zoom_range=0.25,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode="nearest",
        )
        simple_rescaler = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_augmentor.flow_from_directory(
        f"{data_dir}/train",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True,
    )

    val_gen = simple_rescaler.flow_from_directory(
        f"{data_dir}/val",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )

    test_gen = simple_rescaler.flow_from_directory(
        f"{data_dir}/test",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )

    return train_gen, val_gen, test_gen


# ======================================================================
#  Class weights  (handles the ~3:1 imbalance)
# ======================================================================

def compute_class_weights(data_dir: str):
    """
    Count images per class in the training set and return a weight dict
    that gives the minority class (NORMAL) higher importance during training.
    """
    train_dir = os.path.join(data_dir, "train")
    normal_count = len(os.listdir(os.path.join(train_dir, "NORMAL")))
    pneumonia_count = len(os.listdir(os.path.join(train_dir, "PNEUMONIA")))
    total = normal_count + pneumonia_count

    # class indices: NORMAL=0, PNEUMONIA=1  (alphabetical)
    weights = {
        0: total / (2.0 * normal_count),    # higher weight for minority
        1: total / (2.0 * pneumonia_count),  # lower weight for majority
    }

    print(f"  Class counts — NORMAL: {normal_count}, PNEUMONIA: {pneumonia_count}")
    print(f"  Class weights — {weights}")
    return weights


# ======================================================================
#  Single-image loader  (for inference / Streamlit app)
# ======================================================================

def load_single_image(image_path: str, img_size: tuple = IMG_SIZE,
                      model_type: str = "efficientnet"):
    """
    Load and preprocess a single image for inference.

    Returns a 4-D numpy array ready for model.predict().
    """
    from tensorflow.keras.preprocessing.image import load_img, img_to_array

    img = load_img(image_path, target_size=img_size)
    arr = img_to_array(img)

    if model_type == "efficientnet":
        arr = effnet_preprocess(arr)
    else:
        arr = arr / 255.0

    return np.expand_dims(arr, axis=0)
