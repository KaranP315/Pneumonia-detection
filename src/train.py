import os
import argparse
import json
import matplotlib
matplotlib.use("Agg")  # non-interactive backend so it works on servers
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
)

from src.preprocess import build_generators, compute_class_weights, IMG_SIZE

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "plots")
RESULT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "results")


def build_cnn(img_size: tuple = IMG_SIZE):
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                      input_shape=(*img_size, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 4
        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Classifier head
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=5e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_transfer_model(img_size: tuple = IMG_SIZE):
    base = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*img_size, 3),
    )
    base.trainable = False  # freeze backbone initially

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _get_callbacks(save_path: str, patience: int = 5):
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            save_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def train_model(model, train_gen, val_gen, model_name: str,
                class_weights: dict = None, epochs: int = 20):
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, f"{model_name}.h5")

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=_get_callbacks(save_path, patience=5),
    )

    print(f"\n✓ Best weights saved → {save_path}")
    return history


def fine_tune_transfer_model(model, train_gen, val_gen, model_name: str,
                             class_weights: dict = None, epochs: int = 10,
                             unfreeze_from: int = -20):
    # The backbone is the first layer of our Sequential
    backbone = model.layers[0]
    backbone.trainable = True

    # Re-freeze everything except the last |unfreeze_from| layers
    for layer in backbone.layers[:unfreeze_from]:
        layer.trainable = False

    trainable_count = sum(1 for l in backbone.layers if l.trainable)
    print(f"\n  Fine-tuning: {trainable_count} layers unfrozen in backbone")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-5),  # very low LR
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    save_path = os.path.join(MODEL_DIR, f"{model_name}.h5")

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=_get_callbacks(save_path, patience=4),
    )

    print(f"\n✓ Fine-tuned weights saved → {save_path}")
    return history


def evaluate_model(model, test_gen):
    loss, acc = model.evaluate(test_gen, verbose=0)
    return {"loss": round(loss, 4), "accuracy": round(acc, 4)}


def plot_histories(histories: dict, output_dir: str = PLOT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, hist in histories.items():
        axes[0].plot(hist.history["accuracy"], label=f"{name} — train")
        axes[0].plot(hist.history["val_accuracy"], "--", label=f"{name} — val")

        axes[1].plot(hist.history["loss"], label=f"{name} — train")
        axes[1].plot(hist.history["val_loss"], "--", label=f"{name} — val")

    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    save_to = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(save_to, dpi=150)
    plt.close()
    print(f"✓ Comparison plots saved → {save_to}")


def save_results(results: dict, output_dir: str = RESULT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "evaluation.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Evaluation metrics saved → {path}")

def main():
    parser = argparse.ArgumentParser(description="Train pneumonia detection models")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Root folder containing train/val/test subdirectories")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Maximum training epochs (early stopping may cut short)")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    class_weights = compute_class_weights(args.data_dir)

    histories = {}
    results = {}

    print("\n" + "=" * 60)
    print("  Training: Baseline CNN")
    print("=" * 60)
    cnn_train, cnn_val, cnn_test = build_generators(
        args.data_dir, IMG_SIZE, args.batch_size, model_type="cnn",
    )
    cnn = build_cnn()
    histories["CNN"] = train_model(
        cnn, cnn_train, cnn_val,
        model_name="cnn_baseline",
        class_weights=class_weights,
        epochs=args.epochs,
    )
    results["CNN"] = evaluate_model(cnn, cnn_test)

    print("\n" + "=" * 60)
    print("  Training: EfficientNetB0 — Phase 1 (frozen backbone)")
    print("=" * 60)
    eff_train, eff_val, eff_test = build_generators(
        args.data_dir, IMG_SIZE, args.batch_size, model_type="efficientnet",
    )
    eff = build_transfer_model()
    histories["EfficientNetB0"] = train_model(
        eff, eff_train, eff_val,
        model_name="efficientnet_transfer",
        class_weights=class_weights,
        epochs=args.epochs,
    )

    print("\n" + "=" * 60)
    print("  Training: EfficientNetB0 — Phase 2 (fine-tuning top layers)")
    print("=" * 60)
    ft_history = fine_tune_transfer_model(
        eff, eff_train, eff_val,
        model_name="efficientnet_transfer",
        class_weights=class_weights,
        epochs=10,
    )

    for key in ["accuracy", "val_accuracy", "loss", "val_loss"]:
        histories["EfficientNetB0"].history[key].extend(ft_history.history[key])

    results["EfficientNetB0"] = evaluate_model(eff, eff_test)

    plot_histories(histories)
    save_results(results)

    print("\n── Results ──────────────────────────────────────")
    for name, metrics in results.items():
        print(f"  {name:20s}  acc={metrics['accuracy']:.4f}  loss={metrics['loss']:.4f}")
    print()


if __name__ == "__main__":
    main()
