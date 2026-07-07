"""Extract an ECG signal from a PDF file.

Pipeline:
  PDF page  →  high-res grayscale image  (PyMuPDF)
            →  invert + threshold         (numpy)
            →  suppress grid lines        (scipy.ndimage)
            →  column-wise trace detect   (numpy)
            →  1-D float32 ECGDataset
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from scipy.ndimage import minimum_filter1d

from data_loader import ECGDataset


def _crop_strip(img: np.ndarray) -> np.ndarray:
    """Remove top/bottom white margins; keep rows that contain dark pixels."""
    inv = 255 - img
    thresh = (inv > 80).astype(np.float32)
    row_density = thresh.sum(axis=1)
    row_density /= row_density.max() + 1e-6
    active = np.where(row_density > 0.02)[0]
    if len(active) < 10:
        return img
    top = max(0, int(active[0]) - 10)
    bot = min(img.shape[0], int(active[-1]) + 10)
    return img[top:bot, :]


def _suppress_grid_lines(thresh: np.ndarray, min_height: int = 5) -> np.ndarray:
    """Kill thin horizontal grid lines by keeping only tall dark runs.

    For each column, a pixel survives only if there are at least
    `min_height` consecutive dark pixels in its vertical neighbourhood.
    """
    h, w = thresh.shape
    out = np.zeros_like(thresh)
    for x in range(w):
        col = thresh[:, x].astype(np.uint8)
        # Running sum over a vertical window of size min_height
        cs = np.cumsum(col)
        window_sum = np.empty(h, dtype=np.int32)
        window_sum[:min_height] = cs[:min_height]
        window_sum[min_height:] = cs[min_height:] - cs[:-min_height]
        out[:, x] = (window_sum >= min_height).astype(np.uint8)
    return out


def _extract_trace(img: np.ndarray) -> np.ndarray:
    """Return a 1-D signal from a grayscale ECG strip image."""
    h, w = img.shape

    # Invert: dark trace → bright
    inv = (255 - img).astype(np.float32)

    # Threshold
    thresh = (inv > 100).astype(np.uint8)

    # Remove thin horizontal grid lines
    trace_mask = _suppress_grid_lines(thresh, min_height=4)

    # For each column pick the mean y-position of surviving pixels
    signal = np.full(w, h / 2.0, dtype=np.float32)
    for x in range(w):
        ys = np.where(trace_mask[:, x] > 0)[0]
        if len(ys):
            signal[x] = float(np.mean(ys))

    # Invert Y axis (top of image = high voltage)
    signal = (h - 1) - signal

    # Median smooth to remove remaining spikes (scipy-free fallback via stride tricks)
    k = 7
    pad = k // 2
    padded = np.pad(signal, pad, mode="edge")
    shape = (len(signal), k)
    strides = (padded.strides[0], padded.strides[0])
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    signal = np.median(windows, axis=1).astype(np.float32)

    return signal


def load_pdf_ecg(filepath: Union[str, Path], page: int = 0, dpi: int = 200) -> ECGDataset:
    """Render one page of a PDF and digitise the ECG trace.

    Args:
        filepath: Path to the PDF file.
        page:     Page index (0-based) to render.
        dpi:      Render resolution.

    Returns:
        ECGDataset with X shape (1, 1, n_timesteps).
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise ImportError("PyMuPDF is required: pip install PyMuPDF") from e

    filepath = Path(filepath)
    doc = fitz.open(str(filepath))
    if page >= len(doc):
        page = 0
    pg = doc[page]

    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = pg.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()

    img = _crop_strip(img)
    signal = _extract_trace(img)

    X = signal[np.newaxis, np.newaxis, :]  # (1, 1, n_timesteps)
    return ECGDataset(
        X=X,
        y=np.array([], dtype=np.int32),
        labels=[],
        fs=0,
        n_leads=1,
        meta=pd.DataFrame({"lead": ["PDF"]}),
    )
