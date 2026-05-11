"""Unified data loader for PTB-XL, MIT-BIH, and custom CSV/WFDB datasets."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ECGDataset dataclass — unified return type for all loaders
# ---------------------------------------------------------------------------

@dataclass
class ECGDataset:
    """Container for a loaded ECG dataset.

    Attributes:
        X: Signal array of shape (n_samples, n_leads, n_timesteps).
        y: Label array. Shape (n_samples,) for single-label,
           (n_samples, n_classes) for multi-label.
        labels: Human-readable class names ordered by column index.
        fs: Sampling frequency in Hz.
        n_leads: Number of ECG leads.
        meta: Optional DataFrame with per-sample metadata (patient ID, etc.).
    """

    X: np.ndarray
    y: np.ndarray
    labels: List[str]
    fs: int
    n_leads: int
    meta: Optional[pd.DataFrame] = None

    def __len__(self) -> int:
        return len(self.X)

    def __repr__(self) -> str:
        return (
            f"ECGDataset(n_samples={len(self)}, n_leads={self.n_leads}, "
            f"n_timesteps={self.X.shape[-1]}, n_classes={len(self.labels)}, fs={self.fs})"
        )


# ---------------------------------------------------------------------------
# PTB-XL loader
# ---------------------------------------------------------------------------

def load_ptbxl(
    data_dir: Union[str, Path],
    sampling_rate: int = 100,
) -> ECGDataset:
    """Load the PTB-XL dataset using wfdb and the provided CSV metadata.

    Args:
        data_dir: Directory containing ptbxl_database.csv and the WFDB records.
        sampling_rate: 100 or 500 Hz — selects the corresponding record folder.

    Returns:
        ECGDataset with X of shape (n_samples, 12, n_timesteps) and
        multi-label y of shape (n_samples, 5) for the five superclasses
        NORM, MI, STTC, CD, HYP.
    """
    try:
        import ast
        import wfdb
    except ImportError as exc:
        raise ImportError("wfdb is required: pip install wfdb") from exc

    from config import PTBXL_SUPERCLASSES, PTBXL_LABEL_MAP

    data_dir = Path(data_dir)
    csv_path = data_dir / "ptbxl_database.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"PTB-XL metadata CSV not found at {csv_path}. "
            "Download with: wfdb.dl_database('ptb-xl', data_dir)"
        )

    meta = pd.read_csv(csv_path, index_col="ecg_id")
    meta.scp_codes = meta.scp_codes.apply(ast.literal_eval)

    # Load the SCP statements table to get superclass mappings
    scp_statements_path = data_dir / "scp_statements.csv"
    if scp_statements_path.exists():
        scp_df = pd.read_csv(scp_statements_path, index_col=0)
        scp_superclass_map = scp_df[scp_df.diagnostic == 1]["diagnostic_class"].to_dict()
    else:
        logger.warning("scp_statements.csv not found; falling back to direct superclass labels")
        scp_superclass_map = {}

    def _get_superclasses(scp_codes: Dict[str, float]) -> np.ndarray:
        label = np.zeros(len(PTBXL_SUPERCLASSES), dtype=np.float32)
        for code in scp_codes:
            if code in scp_superclass_map:
                sc = scp_superclass_map[code]
            elif code in PTBXL_LABEL_MAP:
                sc = code
            else:
                continue
            if sc in PTBXL_LABEL_MAP:
                label[PTBXL_LABEL_MAP[sc]] = 1.0
        return label

    meta["superclass_labels"] = meta.scp_codes.apply(_get_superclasses)

    # Keep only samples that have at least one superclass label
    meta = meta[meta["superclass_labels"].apply(lambda x: x.sum() > 0)]

    record_folder = "records100" if sampling_rate == 100 else "records500"

    signals, y_list, valid_idx = [], [], []
    for ecg_id, row in meta.iterrows():
        filename_key = "filename_lr" if sampling_rate == 100 else "filename_hr"
        record_path = data_dir / row[filename_key]
        try:
            signal, _ = wfdb.rdsamp(str(record_path))
            # signal shape: (n_timesteps, n_leads) → transpose to (n_leads, n_timesteps)
            signals.append(signal.T.astype(np.float32))
            y_list.append(row["superclass_labels"])
            valid_idx.append(ecg_id)
        except Exception as exc:
            logger.debug("Skipping record %s: %s", ecg_id, exc)

    if not signals:
        raise RuntimeError("No PTB-XL records could be loaded. Check data_dir and sampling_rate.")

    X = np.stack(signals, axis=0)  # (n_samples, 12, n_timesteps)
    y = np.stack(y_list, axis=0)   # (n_samples, 5)

    valid_meta = meta.loc[valid_idx].copy()
    valid_meta.reset_index(inplace=True)

    logger.info("PTB-XL loaded: %s, fs=%d Hz", X.shape, sampling_rate)
    return ECGDataset(
        X=X,
        y=y,
        labels=PTBXL_SUPERCLASSES,
        fs=sampling_rate,
        n_leads=X.shape[1],
        meta=valid_meta,
    )


# ---------------------------------------------------------------------------
# MIT-BIH loader
# ---------------------------------------------------------------------------

def load_mitbih(data_dir: Union[str, Path]) -> ECGDataset:
    """Load MIT-BIH Arrhythmia database, segmented into individual beats.

    Segments ±93 samples around each annotated R-peak, producing fixed-length
    windows of 187 samples (AAMI standard beat length).

    Args:
        data_dir: Directory containing the MIT-BIH WFDB records (*.dat / *.hea).

    Returns:
        ECGDataset with X of shape (n_beats, 1, 187) and y of shape (n_beats,)
        with integer-encoded AAMI class labels (N=0, S=1, V=2, F=3, Q=4).
    """
    try:
        import wfdb
    except ImportError as exc:
        raise ImportError("wfdb is required: pip install wfdb") from exc

    from config import MITBIH_CLASSES, MITBIH_LABEL_MAP, MITBIH_SYMBOL_MAP, SIGNAL

    data_dir = Path(data_dir)
    record_ids = _find_mitbih_records(data_dir)
    if not record_ids:
        raise FileNotFoundError(
            f"No MIT-BIH records found in {data_dir}. "
            "Download with: wfdb.dl_database('mitdb', data_dir)"
        )

    window = SIGNAL.segment_window  # 187
    half = window // 2

    beats, labels_list, meta_rows = [], [], []
    for rec_id in record_ids:
        record_path = str(data_dir / rec_id)
        try:
            record = wfdb.rdrecord(record_path, channels=[0])
            ann = wfdb.rdann(record_path, "atr")
        except Exception as exc:
            logger.debug("Skipping record %s: %s", rec_id, exc)
            continue

        sig = record.p_signal[:, 0].astype(np.float32)
        fs = record.fs

        for idx, symbol in zip(ann.sample, ann.symbol):
            aami_cls = MITBIH_SYMBOL_MAP.get(symbol)
            if aami_cls is None:
                continue
            start = idx - half
            end = start + window
            if start < 0 or end > len(sig):
                continue
            beat = sig[start:end]
            beats.append(beat)
            labels_list.append(MITBIH_LABEL_MAP[aami_cls])
            meta_rows.append({"record": rec_id, "sample": idx, "symbol": symbol, "aami": aami_cls})

    if not beats:
        raise RuntimeError("No MIT-BIH beats could be extracted. Check data_dir.")

    X = np.stack(beats, axis=0)[:, np.newaxis, :]  # (n_beats, 1, 187)
    y = np.array(labels_list, dtype=np.int32)

    meta_df = pd.DataFrame(meta_rows)
    logger.info("MIT-BIH loaded: %s beats, fs=%d Hz", X.shape, 360)
    return ECGDataset(
        X=X,
        y=y,
        labels=MITBIH_CLASSES,
        fs=360,
        n_leads=1,
        meta=meta_df,
    )


def _find_mitbih_records(data_dir: Path) -> List[str]:
    """Return MIT-BIH record IDs present in data_dir."""
    return sorted({p.stem for p in data_dir.glob("*.hea")})


# ---------------------------------------------------------------------------
# Custom CSV / WFDB loader
# ---------------------------------------------------------------------------

def load_custom(filepath: Union[str, Path]) -> ECGDataset:
    """Load an arbitrary ECG file (CSV or WFDB format).

    CSV format expected columns: lead_name, sample_idx, value.
    The loader pivots this to wide format and infers leads automatically.

    WFDB format: pass any path accepted by wfdb.rdrecord().

    Args:
        filepath: Path to a CSV file or WFDB record (without extension).

    Returns:
        ECGDataset with X of shape (1, n_leads, n_timesteps).
        y is an empty array (no ground-truth labels for arbitrary input).
    """
    try:
        import wfdb
    except ImportError as exc:
        raise ImportError("wfdb is required: pip install wfdb") from exc

    filepath = Path(filepath)

    if filepath.suffix.lower() == ".csv":
        return _load_custom_csv(filepath)
    else:
        return _load_custom_wfdb(filepath)


def _load_custom_csv(filepath: Path) -> ECGDataset:
    df = pd.read_csv(filepath)
    required = {"lead_name", "sample_idx", "value"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"CSV must have columns {required}. Found: {list(df.columns)}"
        )
    wide = df.pivot(index="sample_idx", columns="lead_name", values="value")
    wide.sort_index(inplace=True)
    leads = list(wide.columns)
    sig = wide.values.T.astype(np.float32)  # (n_leads, n_timesteps)
    X = sig[np.newaxis, ...]               # (1, n_leads, n_timesteps)
    return ECGDataset(
        X=X,
        y=np.array([], dtype=np.int32),
        labels=[],
        fs=0,
        n_leads=len(leads),
        meta=pd.DataFrame({"lead": leads}),
    )


def _load_custom_wfdb(filepath: Path) -> ECGDataset:
    import wfdb
    record = wfdb.rdrecord(str(filepath.with_suffix("")))
    sig = record.p_signal.T.astype(np.float32)  # (n_leads, n_timesteps)
    X = sig[np.newaxis, ...]
    leads = record.sig_name or [f"lead_{i}" for i in range(sig.shape[0])]
    return ECGDataset(
        X=X,
        y=np.array([], dtype=np.int32),
        labels=[],
        fs=record.fs,
        n_leads=len(leads),
        meta=pd.DataFrame({"lead": leads}),
    )


# ---------------------------------------------------------------------------
# Dataset splitter
# ---------------------------------------------------------------------------

class DatasetSplitter:
    """Stratified train/val/test split with optional patient-level grouping.

    Args:
        train_frac: Fraction of data for training.
        val_frac: Fraction of data for validation.
        test_frac: Fraction of data for testing.
        patient_level: If True, keep all samples from one patient in one split.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        patient_level: bool = True,
        seed: int = 42,
    ) -> None:
        assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1."
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.patient_level = patient_level
        self.seed = seed

    def split(
        self,
        dataset: ECGDataset,
        patient_col: Optional[str] = "patient_id",
    ) -> Tuple[ECGDataset, ECGDataset, ECGDataset]:
        """Split dataset into train, val, test subsets.

        Args:
            dataset: Full ECGDataset to split.
            patient_col: Column name in dataset.meta for patient IDs.
                         Only used when patient_level=True.

        Returns:
            Three ECGDataset objects: train, val, test.
        """
        n = len(dataset)
        X, y = dataset.X, dataset.y
        meta = dataset.meta

        groups = None
        if self.patient_level and meta is not None and patient_col in (meta.columns if meta is not None else []):
            groups = meta[patient_col].values

        # Multi-label → single-label proxy for stratification
        if y.ndim == 2:
            strat_labels = np.argmax(y, axis=1)
        else:
            strat_labels = y

        indices = np.arange(n)
        train_idx, temp_idx = self._single_split(
            indices, strat_labels, groups,
            test_size=self.val_frac + self.test_frac,
        )
        # Relative val fraction within the temp split
        rel_val = self.val_frac / (self.val_frac + self.test_frac)
        temp_strat = strat_labels[temp_idx]
        temp_groups = groups[temp_idx] if groups is not None else None
        val_idx_rel, test_idx_rel = self._single_split(
            np.arange(len(temp_idx)), temp_strat, temp_groups,
            test_size=1.0 - rel_val,
        )
        val_idx = temp_idx[val_idx_rel]
        test_idx = temp_idx[test_idx_rel]

        def _subset(idx: np.ndarray) -> ECGDataset:
            sub_meta = meta.iloc[idx].reset_index(drop=True) if meta is not None else None
            sub_y = y[idx]
            return ECGDataset(
                X=X[idx],
                y=sub_y,
                labels=dataset.labels,
                fs=dataset.fs,
                n_leads=dataset.n_leads,
                meta=sub_meta,
            )

        train_ds = _subset(train_idx)
        val_ds = _subset(val_idx)
        test_ds = _subset(test_idx)

        logger.info(
            "Split: train=%d val=%d test=%d", len(train_ds), len(val_ds), len(test_ds)
        )
        return train_ds, val_ds, test_ds

    def _single_split(
        self,
        indices: np.ndarray,
        labels: np.ndarray,
        groups: Optional[np.ndarray],
        test_size: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        from sklearn.model_selection import ShuffleSplit

        # Fall back to random split when any class has fewer than 2 members
        min_class_count = int(np.bincount(labels.astype(int)).min()) if labels.ndim == 1 else 2

        if groups is not None:
            splitter: Any = GroupShuffleSplit(
                n_splits=1, test_size=test_size, random_state=self.seed
            )
            train_rel, test_rel = next(splitter.split(indices, labels, groups))
        elif min_class_count >= 2:
            splitter = StratifiedShuffleSplit(
                n_splits=1, test_size=test_size, random_state=self.seed
            )
            train_rel, test_rel = next(splitter.split(indices, labels))
        else:
            splitter = ShuffleSplit(
                n_splits=1, test_size=test_size, random_state=self.seed
            )
            train_rel, test_rel = next(splitter.split(indices))
        return indices[train_rel], indices[test_rel]
