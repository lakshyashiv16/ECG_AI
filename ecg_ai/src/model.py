"""Model definitions: 1D CNN, 2D CNN (EfficientNetB0), 1D ResNet with SE blocks."""

from __future__ import annotations

import logging
from typing import List, Literal, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

TaskType = Literal["multiclass", "multilabel"]


# ---------------------------------------------------------------------------
# Helper: build output head
# ---------------------------------------------------------------------------

def _output_head(
    x,
    num_classes: int,
    task: TaskType,
    dropout_rate: float = 0.5,
):
    """Attach a dense output head with appropriate activation."""
    import tensorflow as tf
    from tensorflow.keras import layers

    x = layers.Dropout(dropout_rate)(x)
    activation = "sigmoid" if task == "multilabel" else "softmax"
    x = layers.Dense(num_classes, activation=activation, name="output")(x)
    return x


# ---------------------------------------------------------------------------
# Model A — 1D CNN
# ---------------------------------------------------------------------------

def build_cnn1d(
    n_leads: int,
    n_timesteps: int,
    num_classes: int,
    task: TaskType = "multiclass",
    filters: Optional[List[int]] = None,
    kernel_size: int = 7,
    pool_size: int = 2,
    dense_units: int = 128,
    dropout_rate: float = 0.5,
) -> "tf.keras.Model":
    """Build a 4-block 1D CNN for ECG classification.

    Architecture:
        4 × [Conv1D → BatchNorm → ReLU → MaxPool] with filters 32→64→128→256.
        Global Average Pooling → Dense(128) → Dropout → output.

    Args:
        n_leads: Number of input leads (channels-first: leads are "channels").
        n_timesteps: Length of each time series.
        num_classes: Number of output classes.
        task: 'multiclass' (softmax) or 'multilabel' (sigmoid).
        filters: List of filter counts per block. Defaults to [32,64,128,256].
        kernel_size: Convolution kernel size.
        pool_size: MaxPool stride / pool size.
        dense_units: Units in the pre-output dense layer.
        dropout_rate: Dropout probability before output.

    Returns:
        Compiled Keras model (not yet compiled — caller does that).
    """
    import tensorflow as tf
    from tensorflow.keras import Input, Model, layers

    if filters is None:
        filters = [32, 64, 128, 256]

    inputs = Input(shape=(n_leads, n_timesteps), name="ecg_input")
    # channels_last for Conv1D: (batch, timesteps, channels)
    x = layers.Permute((2, 1))(inputs)  # (batch, n_timesteps, n_leads)

    for n_filt in filters:
        x = layers.Conv1D(n_filt, kernel_size, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling1D(pool_size)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    outputs = _output_head(x, num_classes, task, dropout_rate)

    model = Model(inputs=inputs, outputs=outputs, name="CNN1D")
    return model


# ---------------------------------------------------------------------------
# Model B — 2D CNN via EfficientNetB0 transfer learning
# ---------------------------------------------------------------------------

def ecg_to_image(
    X: np.ndarray,
    image_size: int = 224,
) -> np.ndarray:
    """Render ECG leads as a 2-D grayscale image (leads stacked vertically).

    Each lead is min-max normalised to [0, 1], then the stacked image is
    resized to (image_size, image_size) and broadcast to RGB (3 channels)
    for compatibility with pretrained ImageNet backbones.

    Args:
        X: Signal array (n_samples, n_leads, n_timesteps).
        image_size: Target image resolution.

    Returns:
        Image array (n_samples, image_size, image_size, 3) float32 in [0, 1].
    """
    from PIL import Image

    n_samples, n_leads, n_timesteps = X.shape
    images = np.empty((n_samples, image_size, image_size, 3), dtype=np.float32)

    for i, sample in enumerate(X):
        # Normalise each lead to [0, 1]
        canvas = np.zeros((n_leads, n_timesteps), dtype=np.float32)
        for l in range(n_leads):
            lead = sample[l]
            mn, mx = lead.min(), lead.max()
            canvas[l] = (lead - mn) / (mx - mn + 1e-8)

        # Pillow resize: (width, height)
        img = Image.fromarray((canvas * 255).astype(np.uint8), mode="L")
        img = img.resize((image_size, image_size), Image.BILINEAR)
        img_arr = np.array(img, dtype=np.float32) / 255.0
        images[i] = np.stack([img_arr, img_arr, img_arr], axis=-1)

    return images


def build_cnn2d(
    num_classes: int,
    task: TaskType = "multiclass",
    image_size: int = 224,
    trainable_layers: int = 20,
    dropout_rate: float = 0.5,
) -> "tf.keras.Model":
    """Build an EfficientNetB0-backed 2D CNN for ECG images.

    Args:
        num_classes: Number of output classes.
        task: 'multiclass' or 'multilabel'.
        image_size: Input image size (square).
        trainable_layers: Number of backbone layers to unfreeze from the top.
        dropout_rate: Dropout before output.

    Returns:
        Keras model.
    """
    import tensorflow as tf
    from tensorflow.keras import Input, Model, layers
    from tensorflow.keras.applications import EfficientNetB0

    inputs = Input(shape=(image_size, image_size, 3), name="ecg_image_input")

    backbone = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
    )
    # Freeze all layers, then unfreeze the top N
    backbone.trainable = False
    for layer in backbone.layers[-trainable_layers:]:
        layer.trainable = True

    x = backbone.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    outputs = _output_head(x, num_classes, task, dropout_rate)

    model = Model(inputs=inputs, outputs=outputs, name="CNN2D_EfficientNet")
    return model


# ---------------------------------------------------------------------------
# Model C — 1D ResNet with Squeeze-and-Excitation blocks
# ---------------------------------------------------------------------------

def _se_block(x, filters: int, ratio: int = 16):
    """Squeeze-and-Excitation channel attention block for 1D signals."""
    import tensorflow as tf
    from tensorflow.keras import layers

    # Squeeze: global average pooling across time → (batch, filters)
    se = layers.GlobalAveragePooling1D()(x)
    # Excitation: two FC layers
    se = layers.Dense(max(1, filters // ratio), activation="relu")(se)
    se = layers.Dense(filters, activation="sigmoid")(se)
    # Reshape for broadcast: (batch, 1, filters)
    se = layers.Reshape((1, filters))(se)
    return layers.Multiply()([x, se])


def _residual_block_1d(x, filters: int, kernel_size: int = 7, stride: int = 1, se_ratio: int = 16):
    """Single 1D residual block with optional downsampling and SE attention."""
    import tensorflow as tf
    from tensorflow.keras import layers

    shortcut = x

    x = layers.Conv1D(filters, kernel_size, strides=stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv1D(filters, kernel_size, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    x = _se_block(x, filters, ratio=se_ratio)

    # Projection shortcut when dimensions change
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, strides=stride, padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x


def build_resnet1d(
    n_leads: int,
    n_timesteps: int,
    num_classes: int,
    task: TaskType = "multiclass",
    base_filters: int = 64,
    blocks_per_stage: Optional[List[int]] = None,
    se_ratio: int = 16,
    dropout_rate: float = 0.5,
) -> "tf.keras.Model":
    """Build a 1D ResNet-34 with Squeeze-and-Excitation blocks.

    Architecture mirrors ResNet-34 stages: [3, 4, 6, 3] residual blocks,
    filter counts doubling each stage: 64 → 128 → 256 → 512.

    Args:
        n_leads: Number of input leads.
        n_timesteps: Length of each time series.
        num_classes: Number of output classes.
        task: 'multiclass' or 'multilabel'.
        base_filters: Starting filter count (doubled per stage).
        blocks_per_stage: Blocks in each of the 4 stages.
        se_ratio: SE block reduction ratio.
        dropout_rate: Dropout before output.

    Returns:
        Keras model.
    """
    import tensorflow as tf
    from tensorflow.keras import Input, Model, layers

    if blocks_per_stage is None:
        blocks_per_stage = [3, 4, 6, 3]

    inputs = Input(shape=(n_leads, n_timesteps), name="ecg_input")
    x = layers.Permute((2, 1))(inputs)  # (batch, n_timesteps, n_leads)

    # Stem
    x = layers.Conv1D(base_filters, 15, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(3, strides=2, padding="same")(x)

    # Residual stages
    for stage_idx, n_blocks in enumerate(blocks_per_stage):
        n_filters = base_filters * (2 ** stage_idx)
        for block_idx in range(n_blocks):
            stride = 2 if block_idx == 0 and stage_idx > 0 else 1
            x = _residual_block_1d(x, n_filters, stride=stride, se_ratio=se_ratio)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = _output_head(x, num_classes, task, dropout_rate)

    model = Model(inputs=inputs, outputs=outputs, name="ResNet1D_SE")
    return model


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "cnn1d": build_cnn1d,
    "cnn2d": build_cnn2d,
    "resnet1d": build_resnet1d,
}


def build_model(
    name: str,
    n_leads: int,
    n_timesteps: int,
    num_classes: int,
    task: TaskType = "multiclass",
    **kwargs,
) -> "tf.keras.Model":
    """Build a model by name.

    Args:
        name: One of 'cnn1d', 'cnn2d', 'resnet1d'.
        n_leads: Number of input leads.
        n_timesteps: Input time-series length.
        num_classes: Output classes.
        task: 'multiclass' or 'multilabel'.
        **kwargs: Additional keyword arguments forwarded to the model builder.

    Returns:
        Keras model instance.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_REGISTRY)}")

    builder = MODEL_REGISTRY[name]
    if name == "cnn2d":
        model = builder(num_classes=num_classes, task=task, **kwargs)
    else:
        model = builder(
            n_leads=n_leads,
            n_timesteps=n_timesteps,
            num_classes=num_classes,
            task=task,
            **kwargs,
        )

    # Print summary
    model.summary(print_fn=lambda s: logger.info(s))
    logger.info("Built model '%s' with %d parameters", name, model.count_params())
    return model


def verify_shapes(model: "tf.keras.Model", n_leads: int, n_timesteps: int) -> None:
    """Run a dummy forward pass to verify model input/output shapes.

    Args:
        model: Keras model.
        n_leads: Number of leads.
        n_timesteps: Time-series length.

    Raises:
        AssertionError: If output shape is unexpected.
    """
    import tensorflow as tf

    if "image" in (model.input.name or ""):
        dummy = tf.zeros((2, 224, 224, 3))
    else:
        dummy = tf.zeros((2, n_leads, n_timesteps))

    out = model(dummy, training=False)
    logger.info("Shape check — input: %s, output: %s", dummy.shape, out.shape)
    assert out.shape[0] == 2, "Batch dimension mismatch"
