"""Optional hand-crafted feature extraction from ECG signals."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
from scipy import signal as sp_signal
from scipy.stats import kurtosis, skew

logger = logging.getLogger(__name__)


def extract_time_domain(sig_1d: np.ndarray) -> Dict[str, float]:
    """Extract time-domain statistical features from a 1-D ECG segment.

    Args:
        sig_1d: 1-D signal array (n_timesteps,).

    Returns:
        Dictionary mapping feature name → float value.
    """
    return {
        "mean": float(np.mean(sig_1d)),
        "std": float(np.std(sig_1d)),
        "min": float(np.min(sig_1d)),
        "max": float(np.max(sig_1d)),
        "range": float(np.ptp(sig_1d)),
        "skewness": float(skew(sig_1d)),
        "kurtosis": float(kurtosis(sig_1d)),
        "rms": float(np.sqrt(np.mean(sig_1d ** 2))),
        "zero_crossings": float(np.sum(np.diff(np.sign(sig_1d)) != 0)),
    }


def extract_frequency_domain(sig_1d: np.ndarray, fs: int) -> Dict[str, float]:
    """Extract frequency-domain features via Welch PSD estimation.

    Args:
        sig_1d: 1-D signal array.
        fs: Sampling frequency (Hz).

    Returns:
        Dictionary of spectral features.
    """
    freqs, psd = sp_signal.welch(sig_1d, fs=fs, nperseg=min(256, len(sig_1d)))
    total_power = np.trapz(psd, freqs)
    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs < 0.40)
    lf_power = np.trapz(psd[lf_mask], freqs[lf_mask]) if lf_mask.any() else 0.0
    hf_power = np.trapz(psd[hf_mask], freqs[hf_mask]) if hf_mask.any() else 0.0
    dominant_freq = freqs[np.argmax(psd)] if len(psd) > 0 else 0.0
    return {
        "total_power": float(total_power),
        "lf_power": float(lf_power),
        "hf_power": float(hf_power),
        "lf_hf_ratio": float(lf_power / (hf_power + 1e-8)),
        "dominant_freq_hz": float(dominant_freq),
        "spectral_entropy": float(_spectral_entropy(psd)),
    }


def _spectral_entropy(psd: np.ndarray) -> float:
    """Compute normalized spectral entropy of a PSD."""
    psd_norm = psd / (psd.sum() + 1e-10)
    return -float(np.sum(psd_norm * np.log2(psd_norm + 1e-10)))


def extract_hrv_features(rpeaks: np.ndarray, fs: int) -> Dict[str, float]:
    """Compute basic HRV features from R-peak indices.

    Args:
        rpeaks: Array of R-peak sample indices.
        fs: Sampling frequency (Hz).

    Returns:
        HRV feature dictionary.
    """
    if len(rpeaks) < 2:
        return {"rr_mean_ms": 0.0, "rr_std_ms": 0.0, "rmssd_ms": 0.0, "nn50": 0.0, "pnn50": 0.0}

    rr_intervals = np.diff(rpeaks) / fs * 1000.0  # ms
    successive_diff = np.abs(np.diff(rr_intervals))
    nn50 = int(np.sum(successive_diff > 50))
    pnn50 = nn50 / len(rr_intervals) if len(rr_intervals) > 0 else 0.0
    return {
        "rr_mean_ms": float(np.mean(rr_intervals)),
        "rr_std_ms": float(np.std(rr_intervals)),
        "rmssd_ms": float(np.sqrt(np.mean(successive_diff ** 2))),
        "nn50": float(nn50),
        "pnn50": float(pnn50),
    }


def extract_all_features(
    X: np.ndarray,
    fs: int,
    include_hrv: bool = False,
) -> np.ndarray:
    """Extract a feature vector for every sample in X.

    Args:
        X: Signal array (n_samples, n_leads, n_timesteps).
        fs: Sampling frequency.
        include_hrv: If True, compute HRV features from first lead.

    Returns:
        Feature matrix (n_samples, n_features) as float32.
    """
    from preprocessor import pan_tompkins_rpeaks

    rows = []
    for sample in X:
        feats: Dict[str, float] = {}
        for lead_idx in range(sample.shape[0]):
            lead_sig = sample[lead_idx]
            td = extract_time_domain(lead_sig)
            fd = extract_frequency_domain(lead_sig, fs)
            for k, v in {**td, **fd}.items():
                feats[f"lead{lead_idx}_{k}"] = v
        if include_hrv:
            rpeaks = pan_tompkins_rpeaks(sample[0], fs)
            hrv = extract_hrv_features(rpeaks, fs)
            feats.update(hrv)
        rows.append(list(feats.values()))
    return np.array(rows, dtype=np.float32)
