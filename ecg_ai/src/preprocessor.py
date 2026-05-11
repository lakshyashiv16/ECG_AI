"""ECG signal preprocessing: filtering, segmentation, normalization, augmentation."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from scipy import signal as sp_signal
from scipy.ndimage import median_filter
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual filter functions
# ---------------------------------------------------------------------------

def bandpass_filter(
    sig: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: int,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter.

    Args:
        sig: Signal array of shape (..., n_timesteps). Filter is applied to
             the last axis.
        lowcut: Low cutoff frequency (Hz).
        highcut: High cutoff frequency (Hz).
        fs: Sampling frequency (Hz).
        order: Filter order.

    Returns:
        Filtered signal with the same shape as input.
    """
    nyq = fs / 2.0
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sp_signal.butter(order, [low, high], btype="band")
    return sp_signal.filtfilt(b, a, sig, axis=-1).astype(np.float32)


def notch_filter(
    sig: np.ndarray,
    notch_freq: float,
    fs: int,
    quality: float = 30.0,
) -> np.ndarray:
    """Remove powerline interference with an IIR notch filter.

    Args:
        sig: Signal array (..., n_timesteps).
        notch_freq: Frequency to remove (Hz).
        fs: Sampling frequency (Hz).
        quality: Quality factor of the notch filter.

    Returns:
        Filtered signal.
    """
    b, a = sp_signal.iirnotch(notch_freq, quality, fs)
    return sp_signal.filtfilt(b, a, sig, axis=-1).astype(np.float32)


def remove_baseline_wander(sig: np.ndarray, kernel_size: int = 201) -> np.ndarray:
    """Subtract median-filter baseline to remove slow baseline wander.

    Args:
        sig: Signal array (..., n_timesteps).
        kernel_size: Median filter window size in samples (must be odd).

    Returns:
        Baseline-corrected signal.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    # Build kernel_size tuple matching signal ndim
    ks_tuple = tuple([1] * (sig.ndim - 1) + [kernel_size])
    baseline = median_filter(sig.astype(np.float64), size=ks_tuple)
    return (sig - baseline).astype(np.float32)


def resample_signal(sig: np.ndarray, orig_fs: int, target_fs: int) -> np.ndarray:
    """Resample signal from orig_fs to target_fs using polyphase resampling.

    Args:
        sig: Signal array (..., n_timesteps).
        orig_fs: Original sampling frequency (Hz).
        target_fs: Target sampling frequency (Hz).

    Returns:
        Resampled signal with last axis length scaled by target_fs/orig_fs.
    """
    if orig_fs == target_fs:
        return sig
    n_orig = sig.shape[-1]
    n_target = int(round(n_orig * target_fs / orig_fs))
    return sp_signal.resample(sig, n_target, axis=-1).astype(np.float32)


def z_score_normalize(
    sig: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-lead z-score normalization.

    Args:
        sig: Signal array (n_samples, n_leads, n_timesteps).
        mean: Pre-computed per-lead means (n_leads,). If None, computed from sig.
        std: Pre-computed per-lead stds (n_leads,). If None, computed from sig.
        eps: Small constant to avoid division by zero.

    Returns:
        Tuple of (normalized_signal, mean, std).
    """
    if mean is None:
        mean = sig.mean(axis=(0, 2), keepdims=False)  # (n_leads,)
    if std is None:
        std = sig.std(axis=(0, 2), keepdims=False)    # (n_leads,)
        std = np.where(std < eps, eps, std)
    norm = (sig - mean[np.newaxis, :, np.newaxis]) / std[np.newaxis, :, np.newaxis]
    return norm.astype(np.float32), mean, std


# ---------------------------------------------------------------------------
# Pan-Tompkins R-peak detector
# ---------------------------------------------------------------------------

def pan_tompkins_rpeaks(sig_1d: np.ndarray, fs: int) -> np.ndarray:
    """Detect R-peaks using a simplified Pan-Tompkins algorithm.

    Steps: bandpass → differentiate → square → moving window integrate →
    adaptive threshold peak detection.

    Args:
        sig_1d: 1-D ECG signal (n_timesteps,).
        fs: Sampling frequency (Hz).

    Returns:
        Array of R-peak sample indices.
    """
    # 1) Bandpass 5–15 Hz
    nyq = fs / 2.0
    b, a = sp_signal.butter(1, [5.0 / nyq, 15.0 / nyq], btype="band")
    filtered = sp_signal.lfilter(b, a, sig_1d)

    # 2) Derivative
    diff = np.ediff1d(filtered, to_begin=filtered[0])

    # 3) Squaring
    squared = diff ** 2

    # 4) Moving window integration (150 ms window)
    win = int(0.15 * fs)
    kernel = np.ones(win) / win
    integrated = np.convolve(squared, kernel, mode="same")

    # 5) Peak detection with minimum distance 200 ms
    min_distance = int(0.2 * fs)
    peaks, _ = sp_signal.find_peaks(integrated, distance=min_distance)
    return peaks


def segment_beats(
    sig: np.ndarray,
    fs: int,
    window: int = 187,
) -> Tuple[np.ndarray, np.ndarray]:
    """Segment a single-lead signal into individual beats around R-peaks.

    Args:
        sig: 1-D signal array.
        fs: Sampling frequency.
        window: Number of samples per segment.

    Returns:
        Tuple of (segments array (n_beats, window), r_peak_indices).
    """
    rpeaks = pan_tompkins_rpeaks(sig, fs)
    half = window // 2
    segs = []
    valid_peaks = []
    for r in rpeaks:
        start = r - half
        end = start + window
        if start >= 0 and end <= len(sig):
            segs.append(sig[start:end])
            valid_peaks.append(r)
    if not segs:
        return np.empty((0, window), dtype=np.float32), np.array([], dtype=np.int32)
    return np.stack(segs, axis=0).astype(np.float32), np.array(valid_peaks, dtype=np.int32)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def augment_signal(
    sig: np.ndarray,
    fs: int,
    time_shift: int = 10,
    amp_low: float = 0.9,
    amp_high: float = 1.1,
    noise_snr_db: float = 20.0,
    lead_dropout_prob: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Apply random augmentations to an ECG signal.

    Args:
        sig: Signal array (n_leads, n_timesteps).
        fs: Sampling frequency.
        time_shift: Maximum shift in samples (±).
        amp_low: Minimum amplitude scale factor.
        amp_high: Maximum amplitude scale factor.
        noise_snr_db: Target SNR for Gaussian noise injection (dB).
        lead_dropout_prob: Probability of zeroing out each lead.
        rng: NumPy random Generator. If None, uses default_rng().

    Returns:
        Augmented signal with the same shape as input.
    """
    if rng is None:
        rng = np.random.default_rng()

    aug = sig.copy()

    # Time shift
    shift = rng.integers(-time_shift, time_shift + 1)
    aug = np.roll(aug, shift, axis=-1)

    # Amplitude scaling
    scale = rng.uniform(amp_low, amp_high)
    aug = aug * scale

    # Gaussian noise injection
    signal_power = np.mean(aug ** 2)
    noise_power = signal_power / (10 ** (noise_snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), aug.shape)
    aug = aug + noise.astype(np.float32)

    # Lead dropout
    for i in range(aug.shape[0]):
        if rng.random() < lead_dropout_prob:
            aug[i] = 0.0

    return aug.astype(np.float32)


# ---------------------------------------------------------------------------
# Main Preprocessor class
# ---------------------------------------------------------------------------

class Preprocessor(BaseEstimator, TransformerMixin):
    """Scikit-learn–style ECG preprocessing pipeline.

    Applies bandpass, notch, baseline removal, resampling, and z-score
    normalization. Fit on training data; apply to all splits.

    Args:
        fs: Input signal sampling frequency (Hz). Required.
        target_fs: Target sampling frequency after resampling. 0 = no resample.
        do_bandpass: Apply bandpass filter.
        do_notch: Apply notch filter.
        do_baseline: Apply baseline wander removal.
        do_normalize: Apply per-lead z-score normalization.
        do_resample: Resample to target_fs.
        lowcut: Bandpass low cutoff (Hz).
        highcut: Bandpass high cutoff (Hz).
        notch_freq: Powerline notch frequency (Hz).
        notch_quality: Notch filter quality factor.
        filter_order: Butterworth filter order.
        baseline_ks: Median filter kernel size for baseline removal.
    """

    def __init__(
        self,
        fs: int = 100,
        target_fs: int = 250,
        do_bandpass: bool = True,
        do_notch: bool = True,
        do_baseline: bool = True,
        do_normalize: bool = True,
        do_resample: bool = True,
        lowcut: float = 0.5,
        highcut: float = 40.0,
        notch_freq: float = 50.0,
        notch_quality: float = 30.0,
        filter_order: int = 4,
        baseline_ks: int = 201,
    ) -> None:
        self.fs = fs
        self.target_fs = target_fs
        self.do_bandpass = do_bandpass
        self.do_notch = do_notch
        self.do_baseline = do_baseline
        self.do_normalize = do_normalize
        self.do_resample = do_resample
        self.lowcut = lowcut
        self.highcut = highcut
        self.notch_freq = notch_freq
        self.notch_quality = notch_quality
        self.filter_order = filter_order
        self.baseline_ks = baseline_ks

        # Learned parameters (set during fit)
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.fitted_: bool = False

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Preprocessor":
        """Compute normalization statistics from training data.

        Args:
            X: Signal array (n_samples, n_leads, n_timesteps).
            y: Ignored.

        Returns:
            self
        """
        X_filt = self._apply_filters(X, self.fs)
        if self.do_resample and self.target_fs and self.target_fs != self.fs:
            X_filt = resample_signal(X_filt, self.fs, self.target_fs)
        if self.do_normalize:
            _, self.mean_, self.std_ = z_score_normalize(X_filt)
        self.fitted_ = True
        return self

    def transform(
        self,
        X: np.ndarray,
        augment: bool = False,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Apply all preprocessing steps to X.

        Args:
            X: Signal array (n_samples, n_leads, n_timesteps).
            augment: If True, apply random augmentation after preprocessing.
            seed: Seed for augmentation RNG.

        Returns:
            Preprocessed signal array.
        """
        current_fs = self.fs
        X_out = self._apply_filters(X, current_fs)

        if self.do_resample and self.target_fs and self.target_fs != current_fs:
            X_out = resample_signal(X_out, current_fs, self.target_fs)

        if self.do_normalize and self.mean_ is not None:
            X_out, _, _ = z_score_normalize(X_out, self.mean_, self.std_)

        if augment:
            rng = np.random.default_rng(seed)
            X_aug = np.empty_like(X_out)
            for i in range(len(X_out)):
                X_aug[i] = augment_signal(X_out[i], self.target_fs or current_fs, rng=rng)
            X_out = X_aug

        return X_out

    def fit_transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        augment: bool = False,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Fit on X then transform X (train split).

        Args:
            X: Signal array (n_samples, n_leads, n_timesteps).
            y: Ignored.
            augment: Apply augmentation after transform.
            seed: Augmentation seed.

        Returns:
            Preprocessed (and optionally augmented) signal.
        """
        self.fit(X)
        return self.transform(X, augment=augment, seed=seed)

    # ------------------------------------------------------------------
    def _apply_filters(self, X: np.ndarray, fs: int) -> np.ndarray:
        """Apply deterministic filter steps (bandpass, notch, baseline)."""
        out = X.copy()
        if self.do_bandpass:
            out = bandpass_filter(out, self.lowcut, self.highcut, fs, self.filter_order)
        if self.do_notch:
            out = notch_filter(out, self.notch_freq, fs, self.notch_quality)
        if self.do_baseline:
            out = remove_baseline_wander(out, self.baseline_ks)
        return out

    # ------------------------------------------------------------------
    def save(self, path: Union[str, Path]) -> None:
        """Serialize the fitted preprocessor to disk."""
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Preprocessor saved to %s", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Preprocessor":
        """Load a previously serialized preprocessor."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("Preprocessor loaded from %s", path)
        return obj
