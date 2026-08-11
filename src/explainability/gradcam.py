"""
Grad-CAM utilities for Baseline CNN explainability.

The Baseline CNN expects raw RGB pixel values in [0, 255].
Do NOT normalize images to [0, 1].
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


def load_image(
    image_source,
    image_size: tuple[int, int] = (224, 224),
):
    """
    Load an image for the Baseline CNN.

    IMPORTANT:
    The Baseline CNN expects raw pixel values in [0, 255].
    No /255 normalization is applied.
    """

    if isinstance(image_source, (str, Path)):
        original_image = Image.open(image_source).convert("RGB")
    else:
        original_image = Image.open(image_source).convert("RGB")

    resized = original_image.resize(image_size)

    image_array = np.asarray(
        resized,
        dtype=np.float32,
    )

    # IMPORTANT:
    # Keep pixel values in [0, 255].
    model_input = np.expand_dims(
        image_array,
        axis=0,
    )

    return original_image, model_input


def make_gradcam_heatmap(
    model: tf.keras.Model,
    image_array: np.ndarray,
    last_conv_layer_name: str = "conv2d_4",
) -> np.ndarray:
    """
    Generate a Grad-CAM heatmap for the predicted class.
    """

    last_conv_layer = model.get_layer(last_conv_layer_name)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            last_conv_layer.output,
            model.output,
        ],
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(image_array)

        predicted_class = tf.argmax(
            predictions[0]
        )

        class_channel = predictions[:, predicted_class]

    gradients = tape.gradient(
        class_channel,
        conv_outputs,
    )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2),
    )

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_gradients[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(
        heatmap,
        0,
    )

    max_value = tf.reduce_max(heatmap)

    heatmap = tf.where(
        max_value > 0,
        heatmap / max_value,
        heatmap,
    )

    return heatmap.numpy()


def overlay_gradcam(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> Image.Image:
    """
    Overlay Grad-CAM heatmap on the original image.
    """

    import matplotlib.pyplot as plt

    image = image.convert("RGB")

    heatmap_image = Image.fromarray(
        np.uint8(255 * heatmap)
    ).resize(image.size)

    heatmap_array = np.asarray(
        heatmap_image,
        dtype=np.float32,
    ) / 255.0

    colormap = plt.get_cmap("jet")

    colored_heatmap = colormap(
        heatmap_array
    )[:, :, :3]

    colored_heatmap = np.uint8(
        colored_heatmap * 255
    )

    colored_heatmap = Image.fromarray(
        colored_heatmap
    )

    overlay = Image.blend(
        image,
        colored_heatmap,
        alpha,
    )

    return overlay