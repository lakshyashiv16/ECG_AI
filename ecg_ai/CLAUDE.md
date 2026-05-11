# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

All commands must be run from `ecg_ai/` with the virtual environment active:

```bash
cd /Users/arnavnadar/vscode/AI-ECG/ecg_ai
source venv/bin/activate
```

## Git & GitHub

Remote: `https://github.com/lakshyashiv16/AI-ECG.git` (branch: `main`)

```bash
# Push changes (from repo root, not ecg_ai/)
cd /Users/arnavnadar/vscode/AI-ECG
git add <files>
git commit -m "message"
git push
```

`venv/` is excluded via `ecg_ai/.gitignore` — never commit it (2.2 GB).
All git commands must be run from `/Users/arnavnadar/vscode/AI-ECG` (the repo root), not from inside `ecg_ai/`.

## Common Commands

```bash
# Run all tests
python -m pytest tests/ -v

# Run a single test class
python -m pytest tests/test_preprocessor.py::TestPreprocessor -v

# Run a single test
python -m pytest tests/test_preprocessor.py::TestBandpassFilter::test_output_shape -v

# Train a model (CLI)
python src/train.py --model cnn1d --dataset mitbih --epochs 30 --batch_size 64 --no_mixed_precision

# Quick smoke-test (100 samples, 2 epochs)
python src/train.py --model cnn1d --dataset mitbih --epochs 2 --subset 100 --no_mixed_precision

# Start dashboard
python dashboard/app.py   # → http://localhost:8050
```

## Architecture

### Data flow

```
data/raw/mit-bih/   (WFDB .dat/.hea/.atr files)
        │
src/data_loader.py  load_mitbih() / load_ptbxl() / load_custom()
        │  returns ECGDataset(X, y, labels, fs, n_leads, meta)
        │
src/preprocessor.py  Preprocessor.fit_transform(X_train)  →  Preprocessor.transform(X_val/test)
        │  bandpass → notch → baseline removal → resample → z-score normalise → augment
        │  scaler stats fitted on train only; serialised with Preprocessor.save()
        │
src/model.py         build_model(name, n_leads, n_timesteps, num_classes, task)
        │  'cnn1d'    → build_cnn1d()     4-block Conv1D + GAP
        │  'resnet1d' → build_resnet1d()  ResNet-34 adapted for 1D with SE blocks
        │  'cnn2d'    → build_cnn2d()     EfficientNetB0 on ECG rendered as image
        │
src/train.py         Trainer.train(X_train, y_train, X_val, y_val)
        │  saves best checkpoint to models/best_<run_id>.keras
        │  logs all params + per-epoch metrics to mlruns/ via MLflow
        │
src/evaluate.py      evaluate_model() / grad_cam_1d() / compare_models()
        │  saves confusion matrix, ROC, PR curve PNGs to results/
        │
dashboard/app.py     Plotly Dash, URL-routed single-process app
```

### Key design rules

**`config.py` is the single source of truth.** Every path, sampling rate, class label, hyperparameter, and seed lives there as typed dataclasses (`SignalConfig`, `ModelConfig`, `TrainConfig`, `AugConfig`) or a `SimpleNamespace` (`PATHS`, `MLFLOW`). No hardcoded values anywhere else.

**`ECGDataset` is the universal container.** All loaders return this dataclass. `X` is always `(n_samples, n_leads, n_timesteps)` float32 — channels-first. Models receive this shape directly; Conv1D layers internally permute to channels-last.

**Preprocessing is split-aware.** `Preprocessor.fit()` computes z-score statistics on training data only. Always call `fit_transform(X_train)` then `transform(X_val)` / `transform(X_test)` — never fit on val/test. The fitted scaler is serialised to disk with `Preprocessor.save()`.

**`DatasetSplitter` prevents data leakage.** Defaults to patient-level stratified splits (70/15/15). Falls back to `ShuffleSplit` when any class has fewer than 2 members in the secondary split (common with rare MIT-BIH classes on small subsets).

**Dashboard uses a server-side cache.** `_DATASET_CACHE` in `dashboard/app.py` holds loaded `(X, y, labels, meta)` tuples keyed by dataset name. This avoids reloading 94k beats on every navigation callback. The cache is process-local; restarting the server clears it.

**Dashboard is URL-routed, not tab-based.** `dcc.Location` + the `route()` callback render different layout functions per URL (`/`, `/explorer`, `/inference`, `/tracker`, `/statistics`). Each page layout function is pure (no side effects). All Dash callbacks referencing per-page component IDs must use `suppress_callback_exceptions=True` (already set).

### MIT-BIH specifics

- 48 records at 360 Hz, MLII lead only → `X.shape = (n_beats, 1, 187)` after segmentation
- Beat window: ±93 samples around each R-peak (Pan-Tompkins detection)
- AAMI classes: N=0, S=1, V=2, F=3, Q=4 — heavily imbalanced (~83% N)
- After `Preprocessor` with `target_fs=250`: time axis becomes 130 samples (`187 × 250/360 ≈ 130`)
- Saved model input shape is therefore `(None, 1, 130)` — the inference callback resamples uploaded beats to match

### Adding a new model

1. Implement `build_mymodel(n_leads, n_timesteps, num_classes, task, **kwargs)` in `src/model.py`
2. Add it to `MODEL_REGISTRY` dict at the bottom of that file
3. It will automatically appear as a `--model` option in `src/train.py` and in the dashboard's checkpoint dropdown once trained

### Adding a new dataset

1. Implement `load_mydata(data_dir) -> ECGDataset` in `src/data_loader.py`
2. Add a download path to `config.PATHS`
3. Wire it into `_load_sample_dataset()` in `dashboard/app.py` and the `if dataset_name ==` branch in `src/train.py::main()`
