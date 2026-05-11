"""Unit tests for src/preprocessor.py — each filter step tested independently."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Make sure src/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preprocessor import (
    Preprocessor,
    augment_signal,
    bandpass_filter,
    notch_filter,
    pan_tompkins_rpeaks,
    remove_baseline_wander,
    resample_signal,
    segment_beats,
    z_score_normalize,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FS = 360
N = 1800  # 5 seconds at 360 Hz
N_LEADS = 2


def _sine(freq_hz: float = 1.0, n: int = N, fs: int = FS) -> np.ndarray:
    t = np.arange(n) / fs
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float32)


def _batch(n_samples: int = 8, n_leads: int = N_LEADS, n_t: int = N) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(size=(n_samples, n_leads, n_t)).astype(np.float32)


# ---------------------------------------------------------------------------
# bandpass_filter
# ---------------------------------------------------------------------------

class TestBandpassFilter:
    def test_output_shape(self):
        sig = _batch()
        out = bandpass_filter(sig, lowcut=0.5, highcut=40.0, fs=FS)
        assert out.shape == sig.shape

    def test_attenuates_out_of_band(self):
        # 100 Hz sine should be attenuated by a 0.5–40 Hz bandpass.
        # Use a longer signal (30 s) to minimise edge-effect energy leakage.
        high_freq = _sine(freq_hz=100.0, n=FS * 30)
        out = bandpass_filter(high_freq, 0.5, 40.0, FS)
        assert out.std() < high_freq.std() * 0.3, "High-freq energy not attenuated"

    def test_passes_in_band(self):
        # 10 Hz sine should mostly survive
        sig = _sine(freq_hz=10.0)
        out = bandpass_filter(sig, 0.5, 40.0, FS)
        assert out.std() > 0.5 * sig.std(), "In-band signal excessively attenuated"

    def test_dtype_preserved(self):
        sig = _batch().astype(np.float32)
        out = bandpass_filter(sig, 0.5, 40.0, FS)
        assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# notch_filter
# ---------------------------------------------------------------------------

class TestNotchFilter:
    def test_output_shape(self):
        sig = _batch()
        out = notch_filter(sig, notch_freq=50.0, fs=FS)
        assert out.shape == sig.shape

    def test_attenuates_powerline(self):
        t = np.arange(N) / FS
        powerline = np.sin(2 * np.pi * 50.0 * t).astype(np.float32)
        out = notch_filter(powerline, 50.0, FS)
        assert out.std() < 0.05, f"50 Hz not attenuated: std={out.std():.4f}"

    def test_preserves_signal_away_from_notch(self):
        sig = _sine(freq_hz=10.0)
        out = notch_filter(sig, 50.0, FS)
        corr = float(np.corrcoef(sig, out)[0, 1])
        assert corr > 0.99, f"10 Hz signal damaged: correlation={corr:.4f}"


# ---------------------------------------------------------------------------
# remove_baseline_wander
# ---------------------------------------------------------------------------

class TestBaselineRemoval:
    def test_output_shape(self):
        sig = _batch()
        out = remove_baseline_wander(sig)
        assert out.shape == sig.shape

    def test_removes_slow_drift(self):
        t = np.arange(N) / FS
        drift = (0.5 * np.sin(2 * np.pi * 0.1 * t)).astype(np.float32)
        clean = _sine(freq_hz=10.0)
        combined = clean + drift
        out = remove_baseline_wander(combined[np.newaxis, np.newaxis, :])
        residual = out[0, 0] - clean
        assert residual.std() < 0.1, f"Baseline not removed: std={residual.std():.4f}"

    def test_even_kernel_made_odd(self):
        sig = _batch()
        # Should not raise even for even kernel size
        out = remove_baseline_wander(sig, kernel_size=200)
        assert out.shape == sig.shape


# ---------------------------------------------------------------------------
# resample_signal
# ---------------------------------------------------------------------------

class TestResample:
    def test_upsampling_shape(self):
        sig = _batch(n_t=360)
        out = resample_signal(sig, orig_fs=360, target_fs=500)
        expected_len = int(round(360 * 500 / 360))
        assert out.shape[-1] == expected_len

    def test_downsampling_shape(self):
        sig = _batch(n_t=500)
        out = resample_signal(sig, orig_fs=500, target_fs=250)
        assert out.shape[-1] == 250

    def test_identity_when_same_fs(self):
        sig = _batch()
        out = resample_signal(sig, orig_fs=FS, target_fs=FS)
        np.testing.assert_array_equal(out, sig)


# ---------------------------------------------------------------------------
# z_score_normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_output_shape(self):
        sig = _batch()
        out, mean, std = z_score_normalize(sig)
        assert out.shape == sig.shape
        assert mean.shape == (N_LEADS,)
        assert std.shape == (N_LEADS,)

    def test_zero_mean_unit_std(self):
        sig = _batch(n_samples=256)
        out, _, _ = z_score_normalize(sig)
        # Per-lead mean should be ~0 and std ~1
        per_lead_mean = out.mean(axis=(0, 2))
        per_lead_std = out.std(axis=(0, 2))
        np.testing.assert_allclose(per_lead_mean, 0.0, atol=0.1)
        np.testing.assert_allclose(per_lead_std, 1.0, atol=0.1)

    def test_precomputed_stats_applied(self):
        sig = _batch()
        _, mean, std = z_score_normalize(sig)
        sig2 = _batch(n_samples=4)
        out2, _, _ = z_score_normalize(sig2, mean=mean, std=std)
        assert out2.shape == sig2.shape


# ---------------------------------------------------------------------------
# Pan-Tompkins R-peak detection
# ---------------------------------------------------------------------------

class TestPanTompkins:
    def _synthetic_ecg(self) -> np.ndarray:
        """Create a simple synthetic ECG with R-peaks every 0.8 s."""
        t = np.arange(N) / FS
        ecg = np.zeros(N, dtype=np.float32)
        for peak_t in np.arange(0.2, N / FS, 0.8):
            idx = int(peak_t * FS)
            if idx < N:
                ecg[idx] = 2.0
        # Convolve with Gaussian to smooth
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(ecg, sigma=3)

    def test_returns_array(self):
        sig = self._synthetic_ecg()
        peaks = pan_tompkins_rpeaks(sig, FS)
        assert isinstance(peaks, np.ndarray)

    def test_detects_multiple_peaks(self):
        sig = self._synthetic_ecg()
        peaks = pan_tompkins_rpeaks(sig, FS)
        # Should detect roughly N/FS/0.8 ≈ 6 peaks
        assert len(peaks) >= 3, f"Too few peaks detected: {len(peaks)}"

    def test_peaks_within_bounds(self):
        sig = self._synthetic_ecg()
        peaks = pan_tompkins_rpeaks(sig, FS)
        assert np.all(peaks >= 0)
        assert np.all(peaks < len(sig))


# ---------------------------------------------------------------------------
# segment_beats
# ---------------------------------------------------------------------------

class TestSegmentBeats:
    def test_output_shape(self):
        sig = np.random.default_rng(1).normal(size=N).astype(np.float32)
        beats, peaks = segment_beats(sig, FS, window=187)
        assert beats.ndim == 2
        assert beats.shape[1] == 187
        assert len(beats) == len(peaks)

    def test_no_out_of_bounds(self):
        sig = np.random.default_rng(2).normal(size=N).astype(np.float32)
        beats, peaks = segment_beats(sig, FS, window=187)
        half = 187 // 2
        for p in peaks:
            assert p - half >= 0
            assert p - half + 187 <= len(sig)


# ---------------------------------------------------------------------------
# augment_signal
# ---------------------------------------------------------------------------

class TestAugmentation:
    def test_output_shape(self):
        sig = _batch(n_samples=1)[0]  # (n_leads, n_timesteps)
        out = augment_signal(sig, FS)
        assert out.shape == sig.shape

    def test_output_differs_from_input(self):
        rng = np.random.default_rng(42)
        sig = rng.normal(size=(N_LEADS, N)).astype(np.float32)
        out = augment_signal(sig, FS, rng=rng)
        assert not np.allclose(out, sig), "Augmented signal identical to input"

    def test_dtype_float32(self):
        sig = _batch(n_samples=1)[0]
        out = augment_signal(sig, FS)
        assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# Preprocessor class (integration)
# ---------------------------------------------------------------------------

class TestPreprocessor:
    def test_fit_transform_shape(self):
        X = _batch(n_samples=16, n_leads=2, n_t=360)
        prep = Preprocessor(fs=360, target_fs=250)
        X_out = prep.fit_transform(X)
        # Time axis should be resampled
        expected_t = int(round(360 * 250 / 360))
        assert X_out.shape == (16, 2, expected_t)

    def test_transform_uses_train_stats(self):
        X_train = _batch(n_samples=32, n_leads=2, n_t=360)
        X_test = _batch(n_samples=8, n_leads=2, n_t=360)
        prep = Preprocessor(fs=360, target_fs=360, do_resample=False)
        prep.fit(X_train)
        X_out = prep.transform(X_test)
        assert X_out.shape == X_test.shape

    def test_save_load_roundtrip(self, tmp_path):
        X = _batch(n_samples=16, n_leads=2, n_t=360)
        prep = Preprocessor(fs=360, target_fs=360, do_resample=False)
        prep.fit(X)
        save_path = tmp_path / "prep.pkl"
        prep.save(save_path)
        loaded = Preprocessor.load(save_path)
        np.testing.assert_array_equal(prep.mean_, loaded.mean_)
        np.testing.assert_array_equal(prep.std_, loaded.std_)

    def test_all_flags_disabled(self):
        X = _batch(n_samples=4, n_leads=2, n_t=360)
        prep = Preprocessor(
            fs=360, target_fs=360,
            do_bandpass=False, do_notch=False, do_baseline=False,
            do_normalize=False, do_resample=False,
        )
        X_out = prep.fit_transform(X)
        np.testing.assert_array_almost_equal(X_out, X)

    def test_augmentation_changes_signal(self):
        X = _batch(n_samples=4, n_leads=2, n_t=360)
        prep = Preprocessor(fs=360, target_fs=360, do_resample=False, do_normalize=False,
                            do_bandpass=False, do_notch=False, do_baseline=False)
        prep.fit(X)
        X_aug = prep.transform(X, augment=True, seed=7)
        assert not np.allclose(X, X_aug)
