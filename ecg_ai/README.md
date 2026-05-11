# ECG AI — End-to-End ECG Interpretation Pipeline

A modular system for ECG signal analysis: data loading, preprocessing, deep-learning model training, evaluation, and an interactive web dashboard.

---

claude --resume 076b2d43-f353-4601-b1c4-bc5331a61561

## Quickstart

If you've already set up the project before, this is all you need:

```bash
cd /Users/arnavnadar/vscode/AI-ECG/ecg_ai
source venv/bin/activate
python dashboard/app.py
```

Then open **http://localhost:8050** in your browser. Press `Ctrl+C` to stop.

---

## First-time Setup

### 1. Create the virtual environment and install dependencies

```bash
cd /Users/arnavnadar/vscode/AI-ECG/ecg_ai

python -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate.bat       # Windows

pip install -r requirements.txt
pip install tensorboard seaborn   # two extras not in requirements.txt
```

### 2. Download the MIT-BIH dataset (≈150 MB, required)

```bash
python -c "import wfdb; wfdb.dl_database('mitdb', 'data/raw/mit-bih')"
```

> **PTB-XL** (12-lead, ≈1.8 GB) is optional. Skip it if you only want to use MIT-BIH.
> ```bash
> python -c "import wfdb; wfdb.dl_database('ptb-xl', 'data/raw/ptb-xl')"
> ```

### 3. Train a model (first time only)

The dashboard's Model Inference tab needs at least one saved checkpoint.

```bash
# Fast training run — takes ~2 minutes on CPU
python src/train.py --model cnn1d --dataset mitbih --epochs 10 --batch_size 64 --no_mixed_precision
```

The best checkpoint is saved automatically to `models/`.

### 4. Start the dashboard

```bash
python dashboard/app.py
# Open http://localhost:8050
```

---

## Every time after that

```bash
cd /Users/arnavnadar/vscode/AI-ECG/ecg_ai
source venv/bin/activate
python dashboard/app.py
```

---

## Running each component

### Dashboard

```bash
python dashboard/app.py
# Open http://localhost:8050
```

The dashboard has five pages accessible from the top navbar:

| Page | What it does |
|---|---|
| **Home** | Landing page with project overview |
| **Data Explorer** | Browse MIT-BIH beats, view waveforms, class distribution |
| **Model Inference** | Upload a beat CSV and get an AAMI class prediction + Grad-CAM |
| **Experiment Tracker** | Compare all MLflow training runs side-by-side |
| **Dataset Statistics** | Class distribution pie chart and signal quality heatmap |

### Training from the CLI

```bash
# Standard training run
python src/train.py --model cnn1d --dataset mitbih --epochs 30 --batch_size 64 --no_mixed_precision

# Train the ResNet variant
python src/train.py --model resnet1d --dataset mitbih --epochs 30 --no_mixed_precision

# Quick smoke-test on a tiny subset
python src/train.py --model cnn1d --dataset mitbih --epochs 2 --subset 100 --no_mixed_precision
```

| Flag | Options | Default |
|---|---|---|
| `--model` | `cnn1d`, `resnet1d`, `cnn2d` | `cnn1d` |
| `--dataset` | `mitbih`, `ptbxl` | `ptbxl` |
| `--epochs` | any integer | `50` |
| `--batch_size` | any integer | `64` |
| `--subset` | integer, limits samples loaded | off |
| `--focal_loss` | flag | off |
| `--no_mixed_precision` | flag (recommended on CPU) | off |

### Running inference on a single beat

Generate a test beat CSV from the dataset, then upload it in the dashboard:

```bash
python - <<'EOF'
import sys; sys.path.insert(0,'src'); sys.path.insert(0,'.')
from data_loader import load_mitbih
import config, pandas as pd

ds = load_mitbih(config.PATHS.mitbih)
sample = ds.X[0, 0]
pd.DataFrame({'lead_name':'MLII','sample_idx':range(len(sample)),'value':sample})\
  .to_csv('/tmp/beat.csv', index=False)
print('Saved to /tmp/beat.csv')
EOF
```

Upload `/tmp/beat.csv` in the **Model Inference** tab, select the checkpoint, and click **Run Inference**.

### Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

| Notebook | Purpose |
|---|---|
| `01_data_exploration.ipynb` | Load datasets, class distributions, lead plots |
| `02_preprocessing.ipynb` | Before/after filter visualisations, R-peak detection |
| `03_model_training.ipynb` | Interactive widget-driven training |
| `04_evaluation.ipynb` | Metrics, ROC curves, Grad-CAM, error analysis |

### Running the test suite

```bash
# All tests
python -m pytest tests/ -v

# Single test class
python -m pytest tests/test_preprocessor.py::TestPreprocessor -v
```

---

## Plugging in a custom dataset

Prepare a CSV with three columns: `lead_name`, `sample_idx`, `value`.

```python
from src.data_loader import load_custom
ds = load_custom('path/to/my_ecg.csv')
print(ds)
```

Or pass a WFDB record path (without extension):
```python
ds = load_custom('path/to/record100')
```

---

## Project Structure

```
ecg_ai/
├── config.py               # Central config: paths, hyperparams, class labels
├── requirements.txt
├── README.md
├── data/
│   ├── raw/                # Downloaded datasets (ptb-xl/, mit-bih/)
│   └── processed/          # Preprocessed .npy arrays (optional cache)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── data_loader.py      # PTB-XL, MIT-BIH, custom loaders + DatasetSplitter
│   ├── preprocessor.py     # Butterworth / notch / baseline / normalise / augment
│   ├── features.py         # Hand-crafted time/frequency/HRV feature extraction
│   ├── model.py            # CNN1D, CNN2D (EfficientNetB0), ResNet1D with SE blocks
│   ├── train.py            # Trainer class + CLI (MLflow logging)
│   └── evaluate.py         # Metrics, ROC/PR plots, Grad-CAM, bootstrap CI
├── dashboard/
│   └── app.py              # Plotly Dash app (4 tabs)
├── tests/
│   └── test_preprocessor.py
├── models/                 # Saved Keras checkpoints
├── results/                # Saved evaluation plots
└── mlruns/                 # MLflow tracking data
```

---

## Reproducibility

All seeds (NumPy, TensorFlow, Python `random`) are set via `config.set_global_seeds()`. Pass `--seed <N>` to `train.py` to override the default (42).

---

## Running the test suite

```bash
pytest tests/ -v
```
