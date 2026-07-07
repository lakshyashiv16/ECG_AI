---
title: ECG AI Platform
emoji: 🫀
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
app_port: 7860
---

# ECG AI Platform

A Plotly Dash dashboard for ECG signal analysis, arrhythmia classification, and model inference using the MIT-BIH Arrhythmia Database.

## Features

- **Data Explorer** — visualise raw ECG beats, filter by class, upload custom CSV or PDF
- **Model Inference** — upload a beat and get a real-time arrhythmia classification
- **Experiment Tracker** — browse MLflow training runs and compare metrics
- **Dataset Statistics** — class distribution, lead quality heatmaps

## Local setup

```bash
cd ecg_ai
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python dashboard/app.py   # → http://localhost:7860
```

## Model

CNN-1D trained on MIT-BIH Arrhythmia Database (AAMI classes: N, S, V, F, Q).
Input shape: `(1, 130)` — single MLII lead, 250 Hz, ±260 ms window around R-peak.
