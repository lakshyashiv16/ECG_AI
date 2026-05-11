"""Central configuration: paths, hyperparameters, class labels, seeds."""

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

DEFAULT_SEED: int = 42


def set_global_seeds(seed: int = DEFAULT_SEED) -> None:
    """Set NumPy, Python random, and TensorFlow global seeds."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent

PATHS = SimpleNamespace(
    root=PROJECT_ROOT,
    data_raw=PROJECT_ROOT / "data" / "raw",
    data_processed=PROJECT_ROOT / "data" / "processed",
    ptbxl=PROJECT_ROOT / "data" / "raw" / "ptb-xl",
    mitbih=PROJECT_ROOT / "data" / "raw" / "mit-bih",
    models=PROJECT_ROOT / "models",
    results=PROJECT_ROOT / "results",
    mlruns=PROJECT_ROOT / "mlruns",
    logs=PROJECT_ROOT / "logs",
)

# Create directories that are guaranteed to exist at import time
for _p in vars(PATHS).values():
    Path(_p).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset class-label mappings
# ---------------------------------------------------------------------------

# PTB-XL superclass mapping (diagnostic_superclass column)
PTBXL_SUPERCLASSES: List[str] = ["NORM", "MI", "STTC", "CD", "HYP"]
PTBXL_LABEL_MAP: Dict[str, int] = {cls: i for i, cls in enumerate(PTBXL_SUPERCLASSES)}

# MIT-BIH AAMI beat classes
MITBIH_CLASSES: List[str] = ["N", "S", "V", "F", "Q"]
MITBIH_LABEL_MAP: Dict[str, int] = {cls: i for i, cls in enumerate(MITBIH_CLASSES)}

# Beat symbol → AAMI class mapping (ANSI/AAMI EC57 standard)
MITBIH_SYMBOL_MAP: Dict[str, str] = {
    # Normal
    "N": "N", ".": "N", "e": "N", "j": "N",
    # Supraventricular ectopic
    "A": "S", "a": "S", "J": "S", "S": "S",
    # Ventricular ectopic
    "V": "V", "E": "V",
    # Fusion
    "F": "F",
    # Unknown / paced
    "/": "Q", "f": "Q", "Q": "Q", "?": "Q",
}


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------

@dataclass
class SignalConfig:
    target_fs: int = 250          # resample target (Hz)
    ptbxl_fs: int = 100           # PTB-XL default sampling rate
    mitbih_fs: int = 360          # MIT-BIH sampling rate
    lowcut: float = 0.5           # bandpass low  (Hz)
    highcut: float = 40.0         # bandpass high (Hz)
    notch_freq: float = 50.0      # powerline notch (Hz)
    notch_quality: float = 30.0   # notch Q factor
    filter_order: int = 4
    baseline_median_ks: int = 201  # median filter kernel size (samples)
    segment_window: int = 187      # samples around R-peak (MIT-BIH beat length)
    do_bandpass: bool = True
    do_notch: bool = True
    do_baseline: bool = True
    do_normalize: bool = True
    do_resample: bool = True


SIGNAL = SignalConfig()


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

@dataclass
class AugConfig:
    enabled: bool = True
    time_shift: int = 10          # ± samples
    amplitude_low: float = 0.9
    amplitude_high: float = 1.1
    noise_snr_db: float = 20.0
    lead_dropout_prob: float = 0.1


AUGMENTATION = AugConfig()


# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    # CNN-1D
    cnn1d_filters: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    cnn1d_kernel_size: int = 7
    cnn1d_pool_size: int = 2
    cnn1d_dense_units: int = 128
    cnn1d_dropout: float = 0.5

    # CNN-2D (EfficientNet)
    cnn2d_image_size: int = 224
    cnn2d_trainable_layers: int = 20

    # ResNet-1D
    resnet_blocks: List[int] = field(default_factory=lambda: [3, 4, 6, 3])
    resnet_base_filters: int = 64
    resnet_se_ratio: int = 16

    # Shared
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 50
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    mixed_precision: bool = True


MODEL = ModelConfig()


# ---------------------------------------------------------------------------
# Training / split
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    patient_level_split: bool = True   # prevent data leakage
    seed: int = DEFAULT_SEED
    n_bootstrap: int = 1000            # CI bootstrap iterations


TRAIN = TrainConfig()


# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------

MLFLOW = SimpleNamespace(
    tracking_uri=str(PATHS.mlruns),
    experiment_name="ecg_ai",
)
