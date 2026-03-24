"""
gradcam.py
----------
Gradient-weighted Class Activation Mapping (Grad-CAM).

Produces a heatmap that highlights the regions of an X-ray image
that contributed most to the model's prediction — useful for
understanding *why* the model thinks an image shows pneumonia.

Compatible with Keras 3 Sequential models.

Reference:
  Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
  via Gradient-based Localization", ICCV 2017.
"""

import cv2
import numpy as np
import tensorflow as tf


def _flatten_layers(model):
    """
    Yield every leaf layer from a model, recursing into nested
    Sequential / Functional sub-models (e.g. the EfficientNet backbone).
    """
    for layer in model.layers:
        if hasattr(layer, "layers"):
            yield from _flatten_layers(layer)
        else:
            yield layer


def find_last_conv_layer(model):
    """Return the last Conv2D layer object found anywhere in the model."""
    last_conv = None
    for layer in _flatten_layers(model):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer
    if last_conv is None:
        raise ValueError("No Conv2D layer found in the model.")
    return last_conv


def _find_backbone_for_layer(model, target_layer):
    """
    Determine if target_layer lives inside a nested sub-model (backbone)
    or directly in the top-level model.

    Returns (backbone, is_nested):
      - backbone = the sub-model containing the layer (or model itself)
      - is_nested = True if it's inside a sub-model, False if top-level
    """
    for layer in model.layers:
        if layer is target_layer:
            return model, False
        if hasattr(layer, "layers"):
            for sub in layer.layers:
                if sub is target_layer:
                    return layer, True
    raise RuntimeError(
        f"Could not find layer '{target_layer.name}' in the model."
    )


def make_gradcam_heatmap(model, img_array: np.ndarray,
                         target_conv_layer=None):
    """
    Compute the Grad-CAM heatmap for a given image.

    The key challenge with Keras 3 Sequential models is that `.input`
    isn't defined until the model has been called, and creating sub-models
    across nested model boundaries can break the gradient chain.

    Strategy:
      - For nested models (transfer learning): build a single extractor
        that outputs BOTH the conv features and the backbone output from
        the SAME forward pass, then continue through head layers manually.
      - For flat models (CNN): build a single grad-model that outputs
        both the conv features and the final prediction.

    Parameters
    ----------
    model : keras.Model
    img_array : np.ndarray  shape (1, H, W, 3)
    target_conv_layer : keras.layers.Layer or None

    Returns
    -------
    heatmap : np.ndarray  shape (conv_h, conv_w), values in [0, 1]
    """
    if target_conv_layer is None:
        target_conv_layer = find_last_conv_layer(model)

    img_tensor = tf.cast(img_array, tf.float32)

    # Build the model so .input / .output become available
    _ = model(img_tensor)

    backbone, is_nested = _find_backbone_for_layer(model, target_conv_layer)

    if is_nested:
        # ── Transfer-learning model (e.g. Sequential([EfficientNet, ...]))
        # Build an extractor from the backbone that outputs BOTH the
        # intermediate conv features AND the backbone's final output.
        # This ensures both come from the SAME forward pass.
        feature_model = tf.keras.Model(
            inputs=backbone.input,
            outputs=[target_conv_layer.output, backbone.output],
        )

        with tf.GradientTape() as tape:
            conv_output, backbone_out = feature_model(img_tensor)
            tape.watch(conv_output)

            # Continue through the head layers after the backbone
            x = backbone_out
            for layer in model.layers:
                if layer is backbone:
                    continue
                x = layer(x)
            predictions = x
            loss = predictions[:, 0]

    else:
        # ── Flat model (e.g. plain CNN Sequential)
        grad_model = tf.keras.Model(
            inputs=model.input,
            outputs=[target_conv_layer.output, model.output],
        )

        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model(img_tensor)
            tape.watch(conv_output)
            loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_output)

    if grads is None:
        # Safety fallback — shouldn't happen with the above approach
        h, w = int(conv_output.shape[1]), int(conv_output.shape[2])
        return np.ones((h, w), dtype=np.float32) * 0.5

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()

    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() != 0:
        heatmap /= heatmap.max()

    return heatmap


def overlay_heatmap(heatmap: np.ndarray,
                    original_img: np.ndarray,
                    alpha: float = 0.4,
                    colormap: int = cv2.COLORMAP_JET):
    """
    Resize the heatmap to match the original image, apply a colourmap,
    and blend them together.

    Parameters
    ----------
    heatmap : np.ndarray      (conv_h, conv_w), values in [0, 1]
    original_img : np.ndarray  (H, W, 3), uint8 or float [0, 1]
    alpha : float              blending factor for the heatmap

    Returns
    -------
    superimposed : np.ndarray  (H, W, 3) uint8
    """
    if original_img.max() <= 1.0:
        original_img = (original_img * 255).astype(np.uint8)

    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1],
                                            original_img.shape[0]))
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8), colormap
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    superimposed = cv2.addWeighted(original_img, 1 - alpha,
                                   heatmap_colored, alpha, 0)
    return superimposed
