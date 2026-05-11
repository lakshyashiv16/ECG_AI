"""Training loop with MLflow experiment tracking and CLI entry point."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Allow running as a script from within src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """Binary focal loss factory for multi-label tasks.

    Args:
        gamma: Focusing exponent.
        alpha: Balance factor.

    Returns:
        Keras-compatible loss function.
    """
    import tensorflow as tf

    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        fl = alpha * tf.pow(1.0 - pt, gamma) * bce
        return tf.reduce_mean(fl)

    loss_fn.__name__ = "focal_loss"
    return loss_fn


# ---------------------------------------------------------------------------
# Trainer class
# ---------------------------------------------------------------------------

class Trainer:
    """Manages the full training lifecycle for an ECG model.

    Args:
        model: Compiled or uncompiled Keras model.
        dataset_name: Dataset identifier string (for MLflow tagging).
        batch_size: Mini-batch size.
        epochs: Maximum training epochs.
        learning_rate: Initial learning rate.
        use_focal_loss: Use focal loss instead of cross-entropy.
        focal_gamma: Focal loss gamma.
        focal_alpha: Focal loss alpha.
        mixed_precision: Enable TF mixed-precision (float16/float32).
        seed: Global random seed.
        results_dir: Where to save checkpoints and artifacts.
    """

    def __init__(
        self,
        model: "tf.keras.Model",
        dataset_name: str = "ptbxl",
        batch_size: int = 64,
        epochs: int = 50,
        learning_rate: float = 1e-3,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        mixed_precision: bool = True,
        seed: int = 42,
        results_dir: Optional[Path] = None,
    ) -> None:
        import tensorflow as tf
        import config

        self.model = model
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.mixed_precision = mixed_precision
        self.seed = seed
        self.results_dir = Path(results_dir or config.PATHS.models)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        config.set_global_seeds(seed)

        if mixed_precision:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            logger.info("Mixed precision enabled (float16/float32)")

    # ------------------------------------------------------------------
    def compile(self, y_train: np.ndarray) -> None:
        """Compile the model with optimizer, loss, and metrics.

        Args:
            y_train: Training labels — used to determine task and class weights.
        """
        import tensorflow as tf

        is_multilabel = y_train.ndim == 2
        task = "multilabel" if is_multilabel else "multiclass"

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        if self.use_focal_loss:
            loss = focal_loss(self.focal_gamma, self.focal_alpha)
        elif is_multilabel:
            loss = "binary_crossentropy"
        else:
            loss = "sparse_categorical_crossentropy"

        if is_multilabel:
            metrics_list = [
                tf.keras.metrics.AUC(multi_label=True, name="auc"),
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            ]
        else:
            # AUC on sparse labels requires one-hot; use accuracy only during training.
            # Full AUC is computed post-training in evaluate.py.
            metrics_list = [
                tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            ]

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics_list)
        logger.info("Model compiled: loss=%s, task=%s", loss, task)

    # ------------------------------------------------------------------
    def build_callbacks(self, run_id: str, is_multilabel: bool = False) -> list:
        """Build training callbacks.

        Args:
            run_id: MLflow run ID used for naming checkpoint files.
            is_multilabel: If True, monitor val_auc; otherwise monitor val_accuracy.

        Returns:
            List of Keras callbacks.
        """
        import tensorflow as tf
        import config

        checkpoint_path = self.results_dir / f"best_{run_id}.keras"
        log_dir = str(config.PATHS.logs / run_id)
        monitor = "val_auc" if is_multilabel else "val_accuracy"

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                str(checkpoint_path),
                monitor=monitor,
                mode="max",
                save_best_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor,
                mode="max",
                patience=10,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1,
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=0,
            ),
        ]
        self._checkpoint_path = checkpoint_path
        return callbacks

    # ------------------------------------------------------------------
    def compute_class_weights(self, y_train: np.ndarray) -> Optional[dict]:
        """Compute class weights to handle class imbalance.

        Args:
            y_train: Training labels.

        Returns:
            Class weight dictionary or None for multi-label.
        """
        from sklearn.utils.class_weight import compute_class_weight

        if y_train.ndim == 2:
            # Multi-label: class weights not trivially applicable — return None
            return None

        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        return dict(zip(classes.tolist(), weights.tolist()))

    # ------------------------------------------------------------------
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "tf.keras.callbacks.History":
        """Run the full training loop with MLflow logging.

        Args:
            X_train: Training signals (n_samples, n_leads, n_timesteps).
            y_train: Training labels.
            X_val: Validation signals.
            y_val: Validation labels.

        Returns:
            Keras History object.
        """
        import mlflow
        import mlflow.keras
        import tensorflow as tf
        import config

        mlflow.set_tracking_uri(config.MLFLOW.tracking_uri)
        mlflow.set_experiment(config.MLFLOW.experiment_name)

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logger.info("MLflow run: %s", run_id)

            # Log hyperparameters
            mlflow.log_params({
                "model_name": self.model.name,
                "dataset": self.dataset_name,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "learning_rate": self.learning_rate,
                "use_focal_loss": self.use_focal_loss,
                "mixed_precision": self.mixed_precision,
                "seed": self.seed,
            })

            self.compile(y_train)
            is_multilabel = y_train.ndim == 2
            class_weight = self.compute_class_weights(y_train)
            callbacks = self.build_callbacks(run_id, is_multilabel=is_multilabel)

            # MLflow metric callback
            mlflow_cb = _MLflowMetricCallback()
            callbacks.append(mlflow_cb)

            history = self.model.fit(
                X_train,
                y_train,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                class_weight=class_weight,
                verbose=1,
            )

            # Log final metrics
            final_metrics = {k: float(v[-1]) for k, v in history.history.items()}
            mlflow.log_metrics(final_metrics)

            # Save and log model artifact
            if hasattr(self, "_checkpoint_path") and self._checkpoint_path.exists():
                mlflow.log_artifact(str(self._checkpoint_path))

        return history


# ---------------------------------------------------------------------------
# MLflow metric callback
# ---------------------------------------------------------------------------

class _MLflowMetricCallback:
    """Keras callback that logs per-epoch metrics to MLflow.

    Inherits dynamically to avoid importing tensorflow at module level.
    """

    def __new__(cls):
        import tensorflow as tf
        base = tf.keras.callbacks.Callback
        klass = type("_MLflowCB", (base,), {"on_epoch_end": cls._on_epoch_end})
        return klass()

    @staticmethod
    def _on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        import mlflow
        if logs:
            mlflow.log_metrics({k: float(v) for k, v in logs.items()}, step=epoch)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an ECG classification model.")
    parser.add_argument("--model", choices=["cnn1d", "cnn2d", "resnet1d"], default="cnn1d")
    parser.add_argument("--dataset", choices=["ptbxl", "mitbih"], default="ptbxl")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset", type=int, default=None, help="Limit to N samples (for testing)")
    parser.add_argument("--focal_loss", action="store_true")
    parser.add_argument("--no_mixed_precision", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    import config
    from data_loader import DatasetSplitter, load_mitbih, load_ptbxl
    from model import build_model
    from preprocessor import Preprocessor

    # Load dataset
    if args.dataset == "ptbxl":
        dataset = load_ptbxl(config.PATHS.ptbxl, sampling_rate=config.SIGNAL.ptbxl_fs)
        task = "multilabel"
    else:
        dataset = load_mitbih(config.PATHS.mitbih)
        task = "multiclass"

    if args.subset:
        idx = np.random.default_rng(args.seed).choice(len(dataset), args.subset, replace=False)
        from data_loader import ECGDataset
        dataset = ECGDataset(
            X=dataset.X[idx], y=dataset.y[idx], labels=dataset.labels,
            fs=dataset.fs, n_leads=dataset.n_leads,
            meta=dataset.meta.iloc[idx].reset_index(drop=True) if dataset.meta is not None else None,
        )

    splitter = DatasetSplitter(seed=args.seed)
    train_ds, val_ds, test_ds = splitter.split(dataset)

    # Preprocess
    prep = Preprocessor(fs=train_ds.fs, target_fs=config.SIGNAL.target_fs)
    X_train = prep.fit_transform(train_ds.X, augment=True, seed=args.seed)
    X_val = prep.transform(val_ds.X)
    X_test = prep.transform(test_ds.X)

    n_leads, n_timesteps = X_train.shape[1], X_train.shape[2]
    num_classes = len(dataset.labels)

    model = build_model(args.model, n_leads, n_timesteps, num_classes, task=task)

    trainer = Trainer(
        model=model,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        use_focal_loss=args.focal_loss,
        mixed_precision=not args.no_mixed_precision,
        seed=args.seed,
    )

    trainer.train(X_train, train_ds.y, X_val, val_ds.y)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
