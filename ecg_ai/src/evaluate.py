"""Model evaluation: metrics, plots, Grad-CAM, bootstrap CI, run comparison."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    labels: List[str],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute a full suite of classification metrics with bootstrap CIs.

    Args:
        y_true: Ground-truth labels. Shape (n,) for multiclass or (n, C) for multilabel.
        y_pred: Hard predictions. Same shape as y_true.
        y_score: Soft scores / probabilities. Shape (n, C).
        labels: Class name list.
        n_bootstrap: Number of bootstrap iterations for CIs.
        seed: Random seed.

    Returns:
        Dictionary of metric_name → value. CI keys follow the pattern
        ``metric_ci_low`` / ``metric_ci_high``.
    """
    is_multilabel = y_true.ndim == 2
    rng = np.random.default_rng(seed)

    metrics: Dict[str, float] = {}

    # Accuracy
    if is_multilabel:
        acc = float(np.mean(y_pred == y_true))
    else:
        acc = float(np.mean(y_pred == y_true))
    metrics["accuracy"] = acc

    # F1
    avg_kwargs = dict(average="macro", zero_division=0)
    f1_macro = float(f1_score(y_true, y_pred, **avg_kwargs))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    metrics["f1_macro"] = f1_macro
    metrics["f1_weighted"] = f1_weighted

    # ROC-AUC & PR-AUC
    try:
        roc_auc_macro = float(roc_auc_score(y_true, y_score, average="macro", multi_class="ovr"))
        metrics["roc_auc_macro"] = roc_auc_macro
    except Exception:
        metrics["roc_auc_macro"] = float("nan")

    try:
        pr_auc_macro = float(average_precision_score(y_true, y_score, average="macro"))
        metrics["pr_auc_macro"] = pr_auc_macro
    except Exception:
        metrics["pr_auc_macro"] = float("nan")

    # Per-class sensitivity / specificity (multiclass via OvR)
    if not is_multilabel:
        n_classes = len(labels)
        y_true_bin = np.eye(n_classes)[y_true]
        y_pred_bin = np.eye(n_classes)[y_pred]
    else:
        y_true_bin = y_true
        y_pred_bin = y_pred

    for i, cls in enumerate(labels):
        tp = float(np.sum((y_true_bin[:, i] == 1) & (y_pred_bin[:, i] == 1)))
        tn = float(np.sum((y_true_bin[:, i] == 0) & (y_pred_bin[:, i] == 0)))
        fp = float(np.sum((y_true_bin[:, i] == 0) & (y_pred_bin[:, i] == 1)))
        fn = float(np.sum((y_true_bin[:, i] == 1) & (y_pred_bin[:, i] == 0)))
        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        metrics[f"sensitivity_{cls}"] = sensitivity
        metrics[f"specificity_{cls}"] = specificity

    # Bootstrap CIs on accuracy and macro ROC-AUC
    n = len(y_true)
    boot_acc, boot_auc = [], []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, yp, ys = y_true[idx], y_pred[idx], y_score[idx]
        boot_acc.append(float(np.mean(yt == yp) if not is_multilabel else np.mean(yt == yp)))
        try:
            boot_auc.append(float(roc_auc_score(yt, ys, average="macro", multi_class="ovr")))
        except Exception:
            pass

    metrics["accuracy_ci_low"] = float(np.percentile(boot_acc, 2.5))
    metrics["accuracy_ci_high"] = float(np.percentile(boot_acc, 97.5))
    if boot_auc:
        metrics["roc_auc_macro_ci_low"] = float(np.percentile(boot_auc, 2.5))
        metrics["roc_auc_macro_ci_high"] = float(np.percentile(boot_auc, 97.5))

    return metrics


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot and optionally save a normalised confusion matrix heatmap.

    Args:
        y_true: True labels (n,).
        y_pred: Predicted labels (n,).
        labels: Class names.
        save_path: If provided, save the figure here.

    Returns:
        Matplotlib Figure.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) - 1)))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (normalised)")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Confusion matrix saved to %s", save_path)

    return fig


# ---------------------------------------------------------------------------
# ROC curves
# ---------------------------------------------------------------------------

def plot_roc_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    labels: List[str],
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot per-class ROC curves + macro average.

    Args:
        y_true: Binary label matrix (n, n_classes) or one-hot encoded.
        y_score: Probability scores (n, n_classes).
        labels: Class names.
        save_path: Optional save path.

    Returns:
        Matplotlib Figure.
    """
    if y_true.ndim == 1:
        n_classes = len(labels)
        y_true = np.eye(n_classes)[y_true]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

    for i, (cls, col) in enumerate(zip(labels, colors)):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
        auc_val = roc_auc_score(y_true[:, i], y_score[:, i])
        ax.plot(fpr, tpr, color=col, lw=1.5, label=f"{cls} (AUC={auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Per-class ROC Curves")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Precision-Recall curves
# ---------------------------------------------------------------------------

def plot_pr_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    labels: List[str],
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot per-class Precision-Recall curves.

    Args:
        y_true: Binary label matrix (n, n_classes).
        y_score: Probability scores (n, n_classes).
        labels: Class names.
        save_path: Optional save path.

    Returns:
        Matplotlib Figure.
    """
    if y_true.ndim == 1:
        n_classes = len(labels)
        y_true = np.eye(n_classes)[y_true]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

    for i, (cls, col) in enumerate(zip(labels, colors)):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_score[:, i])
        ap = average_precision_score(y_true[:, i], y_score[:, i])
        ax.plot(recall, precision, color=col, lw=1.5, label=f"{cls} (AP={ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Per-class Precision-Recall Curves")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Training history curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot loss and metric training / validation curves.

    Args:
        history: Dictionary from Keras History.history.
        save_path: Optional save path.

    Returns:
        Matplotlib Figure.
    """
    train_keys = [k for k in history if not k.startswith("val_")]
    n_plots = len(train_keys)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    for ax, key in zip(axes, train_keys):
        ax.plot(history[key], label="train")
        if f"val_{key}" in history:
            ax.plot(history[f"val_{key}"], label="val")
        ax.set_title(key)
        ax.set_xlabel("Epoch")
        ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Grad-CAM for 1D CNN
# ---------------------------------------------------------------------------

def grad_cam_1d(
    model: "tf.keras.Model",
    X_sample: np.ndarray,
    class_idx: Optional[int] = None,
    last_conv_layer_name: Optional[str] = None,
) -> np.ndarray:
    """Compute Grad-CAM attribution map for a 1D CNN.

    Finds the last Conv1D layer and computes gradient-weighted activations
    to produce a saliency map over the time axis.

    Args:
        model: Trained Keras model.
        X_sample: Single sample array (n_leads, n_timesteps) or
                  (1, n_leads, n_timesteps).
        class_idx: Class index to explain. If None, uses argmax of prediction.
        last_conv_layer_name: Explicit layer name. Auto-detected if None.

    Returns:
        Normalized saliency map (n_timesteps,) with values in [0, 1].
    """
    import tensorflow as tf

    if X_sample.ndim == 2:
        X_sample = X_sample[np.newaxis]

    # Find last Conv1D layer
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv1D):
                last_conv_layer_name = layer.name
                break
    if last_conv_layer_name is None:
        raise ValueError("No Conv1D layer found in the model.")

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    x_tensor = tf.cast(X_sample, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x_tensor)
        conv_outputs, predictions = grad_model(x_tensor)
        if class_idx is None:
            class_idx = int(tf.argmax(predictions[0]))
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)  # (1, timesteps, filters)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))  # (filters,)
    conv_out = conv_outputs[0]                          # (timesteps, filters)
    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)  # (timesteps,)
    heatmap = tf.nn.relu(heatmap).numpy()

    # Upsample to original n_timesteps if needed
    from scipy.signal import resample
    n_timesteps = X_sample.shape[-1]
    if len(heatmap) != n_timesteps:
        heatmap = resample(heatmap, n_timesteps)

    # Normalize to [0, 1]
    mn, mx = heatmap.min(), heatmap.max()
    if mx > mn:
        heatmap = (heatmap - mn) / (mx - mn)

    return heatmap.astype(np.float32)


# ---------------------------------------------------------------------------
# Full evaluation runner
# ---------------------------------------------------------------------------

def evaluate_model(
    model: "tf.keras.Model",
    X_test: np.ndarray,
    y_test: np.ndarray,
    labels: List[str],
    results_dir: Optional[Union[str, Path]] = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """Run full evaluation and save all plots.

    Args:
        model: Trained Keras model.
        X_test: Test signals.
        y_test: Test labels.
        labels: Class names.
        results_dir: Directory to save output plots.
        n_bootstrap: Bootstrap iterations.
        seed: Random seed.

    Returns:
        Metrics dictionary.
    """
    import tensorflow as tf

    if results_dir is None:
        import config
        results_dir = config.PATHS.results
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    y_score = model.predict(X_test, verbose=0)
    if y_score.ndim == 1:
        y_score = y_score[:, np.newaxis]

    is_multilabel = y_test.ndim == 2
    if is_multilabel:
        y_pred = (y_score >= 0.5).astype(int)
        y_true_flat = np.argmax(y_test, axis=1)
        y_pred_flat = np.argmax(y_pred, axis=1)
    else:
        y_pred_flat = np.argmax(y_score, axis=1)
        y_true_flat = y_test.astype(int)
        n_classes = len(labels)
        y_score = np.exp(y_score) / np.exp(y_score).sum(axis=1, keepdims=True)

    metrics = compute_metrics(
        y_test if is_multilabel else y_true_flat,
        y_pred_flat if not is_multilabel else y_pred,
        y_score,
        labels,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    logger.info("Evaluation metrics: %s", metrics)

    # Plots
    plot_confusion_matrix(y_true_flat, y_pred_flat, labels, results_dir / "confusion_matrix.png")

    y_true_bin = y_test if is_multilabel else np.eye(len(labels))[y_true_flat]
    plot_roc_curves(y_true_bin, y_score, labels, results_dir / "roc_curves.png")
    plot_pr_curves(y_true_bin, y_score, labels, results_dir / "pr_curves.png")

    return metrics


# ---------------------------------------------------------------------------
# Multi-run comparison
# ---------------------------------------------------------------------------

def compare_models(experiment_name: Optional[str] = None) -> "pd.DataFrame":
    """Load all MLflow runs and produce a side-by-side metric DataFrame.

    Args:
        experiment_name: MLflow experiment to query. Uses config default if None.

    Returns:
        Sorted DataFrame of runs with key metrics as columns.
    """
    import mlflow
    import pandas as pd
    import config

    mlflow.set_tracking_uri(config.MLFLOW.tracking_uri)
    exp_name = experiment_name or config.MLFLOW.experiment_name

    runs = mlflow.search_runs(experiment_names=[exp_name])
    if runs.empty:
        logger.warning("No MLflow runs found for experiment '%s'", exp_name)
        return pd.DataFrame()

    keep_cols = ["run_id", "status", "start_time"]
    metric_cols = [c for c in runs.columns if c.startswith("metrics.")]
    param_cols = [c for c in runs.columns if c.startswith("params.model_name") or c.startswith("params.dataset")]
    df = runs[keep_cols + param_cols + metric_cols].copy()
    df.columns = [c.replace("metrics.", "").replace("params.", "") for c in df.columns]
    df.sort_values("roc_auc_macro", ascending=False, inplace=True, na_position="last")
    return df
