"""ECG AI — Professional Plotly Dash dashboard.

Pages:  /           → Landing page
        /explorer   → Data Explorer
        /inference  → Model Inference
        /tracker    → Experiment Tracker
        /statistics → Dataset Statistics

Run:  python dashboard/app.py   →   http://localhost:8050
"""

from __future__ import annotations

import base64
import io
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html, dash_table
from dash.exceptions import PreventUpdate

import config

# ── Bootstrap Icons CDN ──────────────────────────────────────────────────────
ICONS_CSS = "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css"
GOOGLE_FONTS = "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap"

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, ICONS_CSS, GOOGLE_FONTS],
    suppress_callback_exceptions=True,
    title="ECG AI Platform",
)

# ── Colour palette ────────────────────────────────────────────────────────────
CLR = dict(
    navy="#0d1b2a",
    blue="#1565c0",
    accent="#00b4d8",
    light_bg="#f8fafc",
    card_bg="#ffffff",
    text="#1e293b",
    muted="#64748b",
    success="#22c55e",
    danger="#ef4444",
)

FONT = "Inter, system-ui, sans-serif"

# ── Server-side dataset cache ─────────────────────────────────────────────────
_DATASET_CACHE: Dict[str, Any] = {}

LEAD_NAMES_12 = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
LEAD_NAMES_1  = ["MLII"]


def _load_sample_dataset(dataset_name: str, n_samples: int = 100):
    if dataset_name in _DATASET_CACHE:
        return _DATASET_CACHE[dataset_name]
    try:
        if dataset_name == "ptbxl":
            from data_loader import load_ptbxl
            ds = load_ptbxl(config.PATHS.ptbxl)
        elif dataset_name == "mitbih":
            from data_loader import load_mitbih
            ds = load_mitbih(config.PATHS.mitbih)
        else:
            return None
        n = min(n_samples, len(ds))
        result = (ds.X[:n], ds.y[:n], ds.labels, ds.meta)
        _DATASET_CACHE[dataset_name] = result
        return result
    except Exception:
        return None


def _list_model_checkpoints() -> List[Dict[str, str]]:
    model_dir = config.PATHS.models
    checkpoints = sorted(
        list(model_dir.glob("*.keras")) + list(model_dir.glob("*.h5"))
    )
    return [{"label": p.name, "value": str(p)} for p in checkpoints]


def _ecg_figure(X_sample: np.ndarray, lead_names: Optional[List[str]] = None,
                title: str = "", bg: str = "white") -> go.Figure:
    n_leads, n_t = X_sample.shape
    if lead_names is None:
        lead_names = [f"Lead {i}" for i in range(n_leads)]
    cols = 2 if n_leads > 4 else 1
    rows = int(np.ceil(n_leads / cols))
    fig = make_subplots(rows=rows, cols=cols, shared_xaxes=True, vertical_spacing=0.015)
    t = np.arange(n_t)
    for i, (lead, name) in enumerate(zip(X_sample, lead_names)):
        fig.add_trace(
            go.Scatter(x=t, y=lead.tolist(), mode="lines", name=name,
                       line=dict(width=1.2, color=CLR["accent"]), showlegend=False),
            row=i // cols + 1, col=i % cols + 1,
        )
        fig.update_yaxes(title_text=name, row=i // cols + 1, col=i % cols + 1,
                         title_font_size=9, tickfont_size=8)
    fig.update_layout(
        title=dict(text=title, font=dict(family=FONT, size=13)),
        height=max(180, 55 * n_leads),
        margin=dict(l=50, r=10, t=35, b=20),
        paper_bgcolor=bg, plot_bgcolor=bg,
        font=dict(family=FONT),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Shared components
# ═══════════════════════════════════════════════════════════════════════════════

def _navbar() -> dbc.Navbar:
    return dbc.Navbar(
        dbc.Container(fluid=True, children=[
            dcc.Link(
                dbc.Row([
                    dbc.Col(html.I(className="bi bi-heart-pulse-fill me-2",
                                   style={"fontSize": "1.4rem", "color": CLR["accent"]})),
                    dbc.Col(dbc.NavbarBrand("ECG AI Platform",
                                            style={"fontFamily": FONT, "fontWeight": 700,
                                                   "fontSize": "1.1rem", "color": "white"})),
                ], align="center"),
                href="/", style={"textDecoration": "none"},
            ),
            dbc.NavbarToggler(id="navbar-toggler"),
            dbc.Collapse(
                dbc.Nav([
                    dbc.NavItem(dcc.Link("Home",        href="/",           className="nav-link text-white fw-500")),
                    dbc.NavItem(dcc.Link("Data Explorer",  href="/explorer",  className="nav-link text-white")),
                    dbc.NavItem(dcc.Link("Model Inference",href="/inference", className="nav-link text-white")),
                    dbc.NavItem(dcc.Link("Experiments",    href="/tracker",   className="nav-link text-white")),
                    dbc.NavItem(dcc.Link("Statistics",     href="/statistics",className="nav-link text-white")),
                ], navbar=True, className="ms-auto"),
                id="navbar-collapse", navbar=True,
            ),
        ]),
        color=CLR["navy"], dark=True, sticky="top",
        style={"boxShadow": "0 2px 12px rgba(0,0,0,0.3)", "fontFamily": FONT},
    )


def _footer() -> html.Footer:
    return html.Footer(
        dbc.Container(fluid=True, children=[
            html.Hr(style={"borderColor": "#334155"}),
            dbc.Row([
                dbc.Col(html.Span("ECG AI Platform — built with Plotly Dash & TensorFlow",
                                   style={"color": CLR["muted"], "fontSize": "0.8rem"}), width="auto"),
                dbc.Col(html.Span("MIT-BIH · PhysioNet · AAMI EC57",
                                   style={"color": CLR["muted"], "fontSize": "0.8rem"}),
                        width="auto", className="ms-auto"),
            ], justify="between"),
        ]),
        style={"background": CLR["navy"], "padding": "1rem 2rem", "marginTop": "3rem"},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Page 0 — Home / Landing
# ═══════════════════════════════════════════════════════════════════════════════

def _hero_ecg_figure() -> go.Figure:
    """Decorative ECG strip for the hero section."""
    t = np.linspace(0, 4 * np.pi, 600)
    ecg = (
        0.08 * np.sin(t)
        + 0.05 * np.sin(3 * t)
        + np.exp(-((t % (2 * np.pi) - 1.0) ** 2) / 0.008) * 0.4   # P
        - np.exp(-((t % (2 * np.pi) - 1.3) ** 2) / 0.002) * 0.15  # Q
        + np.exp(-((t % (2 * np.pi) - 1.5) ** 2) / 0.001) * 1.5   # R
        - np.exp(-((t % (2 * np.pi) - 1.7) ** 2) / 0.003) * 0.25  # S
        + np.exp(-((t % (2 * np.pi) - 2.0) ** 2) / 0.02)  * 0.35  # T
    )
    fig = go.Figure(go.Scatter(
        x=t.tolist(), y=ecg.tolist(),
        mode="lines",
        line=dict(color=CLR["accent"], width=2),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=160,
        showlegend=False,
    )
    return fig


def _feature_card(icon: str, title: str, desc: str, href: str, color: str) -> dbc.Col:
    return dbc.Col(
        dcc.Link(
            dbc.Card([
                dbc.CardBody([
                    html.Div(
                        html.I(className=f"bi {icon}",
                               style={"fontSize": "2rem", "color": color}),
                        className="mb-3",
                    ),
                    html.H5(title, style={"fontWeight": 600, "fontFamily": FONT,
                                          "color": CLR["text"]}),
                    html.P(desc, style={"color": CLR["muted"], "fontSize": "0.875rem",
                                        "lineHeight": "1.6"}),
                    html.Span("Open →", style={"color": color, "fontSize": "0.85rem",
                                                "fontWeight": 600}),
                ])
            ], style={
                "border": "none",
                "borderRadius": "16px",
                "boxShadow": "0 4px 24px rgba(0,0,0,0.07)",
                "transition": "transform .2s, box-shadow .2s",
                "height": "100%",
            }),
            href=href,
            style={"textDecoration": "none"},
        ),
        md=3, sm=6, className="mb-4",
    )


def _stat_card(value: str, label: str, icon: str) -> dbc.Col:
    return dbc.Col(
        dbc.Card(dbc.CardBody([
            html.Div([
                html.I(className=f"bi {icon} me-2",
                       style={"fontSize": "1.3rem", "color": CLR["accent"]}),
                html.Span(value, style={"fontSize": "1.7rem", "fontWeight": 700,
                                        "color": CLR["navy"]}),
            ], className="d-flex align-items-center mb-1"),
            html.P(label, className="mb-0",
                   style={"color": CLR["muted"], "fontSize": "0.8rem"}),
        ]), style={"border": "none", "borderRadius": "12px",
                   "boxShadow": "0 2px 12px rgba(0,0,0,0.06)"}),
        md=3, sm=6, className="mb-3",
    )


def _home_layout() -> html.Div:
    return html.Div([

        # ── Hero ──────────────────────────────────────────────────────────────
        html.Div(
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.Div("AI-Powered", style={
                            "color": CLR["accent"], "fontWeight": 600,
                            "fontSize": "0.9rem", "letterSpacing": "2px",
                            "textTransform": "uppercase", "marginBottom": "0.5rem",
                        }),
                        html.H1("ECG Interpretation\nPlatform",
                                style={"fontFamily": FONT, "fontWeight": 700,
                                       "fontSize": "clamp(2rem, 4vw, 3.2rem)",
                                       "color": "white", "lineHeight": 1.15,
                                       "whiteSpace": "pre-line"}),
                        html.P(
                            "End-to-end arrhythmia detection using deep learning. "
                            "Explore the MIT-BIH dataset, run real-time inference, "
                            "and track model experiments — all in one place.",
                            style={"color": "#94a3b8", "marginTop": "1rem",
                                   "fontSize": "1rem", "maxWidth": "480px",
                                   "lineHeight": 1.7},
                        ),
                        html.Div([
                            dcc.Link(dbc.Button("Explore Data", color="primary", size="lg",
                                                style={"borderRadius": "8px",
                                                       "background": CLR["accent"],
                                                       "border": "none", "fontWeight": 600}),
                                     href="/explorer"),
                            dcc.Link(dbc.Button("Run Inference", outline=True, size="lg",
                                                style={"borderRadius": "8px", "marginLeft": "1rem",
                                                       "color": "white", "borderColor": "#475569",
                                                       "fontWeight": 600}),
                                     href="/inference"),
                        ], style={"marginTop": "2rem"}),
                    ], md=6, className="py-5"),
                    dbc.Col([
                        html.Div([
                            html.Div("Live ECG Preview",
                                     style={"color": "#94a3b8", "fontSize": "0.75rem",
                                            "marginBottom": "0.25rem", "letterSpacing": "1px",
                                            "textTransform": "uppercase"}),
                            dcc.Graph(figure=_hero_ecg_figure(), config={"displayModeBar": False},
                                      style={"borderRadius": "12px",
                                             "background": "rgba(255,255,255,0.04)",
                                             "border": "1px solid rgba(255,255,255,0.08)",
                                             "padding": "1rem"}),
                            html.Div([
                                html.Span("● NORMAL SINUS RHYTHM",
                                          style={"color": CLR["success"], "fontSize": "0.7rem",
                                                 "fontWeight": 600, "letterSpacing": "1px"}),
                                html.Span("  72 BPM",
                                          style={"color": "#94a3b8", "fontSize": "0.7rem",
                                                 "marginLeft": "1rem"}),
                            ], style={"marginTop": "0.5rem"}),
                        ], style={"padding": "2rem 1rem"}),
                    ], md=6),
                ], align="center"),
            ], fluid=True),
            style={
                "background": f"linear-gradient(135deg, {CLR['navy']} 0%, #1a2f4a 100%)",
                "minHeight": "520px",
            },
        ),

        # ── Stats strip ───────────────────────────────────────────────────────
        html.Div(
            dbc.Container([
                dbc.Row([
                    _stat_card("94,138", "MIT-BIH Beat Segments",  "bi-activity"),
                    _stat_card("48",     "Patient Records",         "bi-people-fill"),
                    _stat_card("5",      "AAMI Arrhythmia Classes", "bi-tag-fill"),
                    _stat_card("3",      "Model Architectures",     "bi-cpu-fill"),
                ], className="py-4"),
            ], fluid=True),
            style={"background": CLR["light_bg"], "borderBottom": "1px solid #e2e8f0"},
        ),

        # ── Feature cards ─────────────────────────────────────────────────────
        dbc.Container([
            html.H2("Platform Features",
                    style={"fontFamily": FONT, "fontWeight": 700, "color": CLR["text"],
                            "marginTop": "3rem", "marginBottom": "0.5rem"}),
            html.P("Everything you need for ECG analysis in one dashboard.",
                   style={"color": CLR["muted"], "marginBottom": "2rem"}),
            dbc.Row([
                _feature_card("bi-search",      "Data Explorer",
                              "Browse MIT-BIH beat segments, visualise raw waveforms, "
                              "and inspect class distributions interactively.",
                              "/explorer",   "#1565c0"),
                _feature_card("bi-lightning-charge-fill", "Model Inference",
                              "Upload any ECG beat and get an instant AAMI class prediction "
                              "with per-class confidence scores and Grad-CAM saliency.",
                              "/inference",  "#7c3aed"),
                _feature_card("bi-bar-chart-line-fill", "Experiment Tracker",
                              "Compare all MLflow training runs side-by-side. Sort by "
                              "accuracy, AUC, or loss to find your best checkpoint.",
                              "/tracker",    "#0891b2"),
                _feature_card("bi-pie-chart-fill", "Dataset Statistics",
                              "Class distribution, signal quality heatmaps, and summary "
                              "statistics to understand your data before trusting the model.",
                              "/statistics", "#059669"),
            ]),
        ], fluid=True, style={"padding": "0 2rem"}),

        # ── Pipeline steps ────────────────────────────────────────────────────
        html.Div(
            dbc.Container([
                html.H2("How It Works",
                        style={"fontFamily": FONT, "fontWeight": 700, "color": CLR["text"],
                                "marginTop": "3rem", "marginBottom": "0.5rem"}),
                html.P("Three steps from raw signal to clinical insight.",
                       style={"color": CLR["muted"], "marginBottom": "2.5rem"}),
                dbc.Row([
                    _pipeline_step("1", "bi-download",     "#1565c0",
                                   "Ingest",
                                   "Load MIT-BIH records via wfdb, segment beats "
                                   "around R-peaks using Pan-Tompkins detection."),
                    _pipeline_step("2", "bi-funnel-fill",  "#7c3aed",
                                   "Preprocess",
                                   "Butterworth bandpass, notch filter, baseline "
                                   "wander removal, z-score normalisation, resampling."),
                    _pipeline_step("3", "bi-graph-up-arrow","#059669",
                                   "Classify",
                                   "1D CNN / ResNet with SE blocks predicts AAMI "
                                   "arrhythmia class; Grad-CAM highlights key regions."),
                ]),
            ], fluid=True),
            style={"background": CLR["light_bg"], "padding": "1rem 2rem 3rem"},
        ),

        _footer(),
    ], style={"fontFamily": FONT})


def _pipeline_step(num: str, icon: str, color: str, title: str, desc: str) -> dbc.Col:
    return dbc.Col(
        dbc.Card(dbc.CardBody([
            html.Div([
                html.Span(num, style={"background": color, "color": "white",
                                      "borderRadius": "50%", "width": "2rem",
                                      "height": "2rem", "display": "inline-flex",
                                      "alignItems": "center", "justifyContent": "center",
                                      "fontWeight": 700, "fontSize": "0.85rem",
                                      "marginRight": "0.75rem"}),
                html.I(className=f"bi {icon}",
                       style={"fontSize": "1.3rem", "color": color}),
            ], className="d-flex align-items-center mb-3"),
            html.H5(title, style={"fontWeight": 600, "color": CLR["text"]}),
            html.P(desc, style={"color": CLR["muted"], "fontSize": "0.875rem",
                                 "lineHeight": 1.6, "marginBottom": 0}),
        ]), style={"border": "none", "borderRadius": "16px",
                   "boxShadow": "0 4px 24px rgba(0,0,0,0.06)", "height": "100%"}),
        md=4, className="mb-4",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Page headers (reusable section title bar)
# ═══════════════════════════════════════════════════════════════════════════════

def _page_header(title: str, subtitle: str, icon: str) -> html.Div:
    return html.Div(
        dbc.Container([
            html.Div([
                html.I(className=f"bi {icon} me-3",
                       style={"fontSize": "1.8rem", "color": CLR["accent"]}),
                html.Div([
                    html.H3(title, className="mb-0",
                            style={"fontWeight": 700, "color": "white"}),
                    html.P(subtitle, className="mb-0",
                           style={"color": "#94a3b8", "fontSize": "0.875rem"}),
                ]),
            ], className="d-flex align-items-center py-4"),
        ], fluid=True),
        style={"background": f"linear-gradient(90deg, {CLR['navy']} 0%, #1e3a5f 100%)",
               "borderBottom": f"3px solid {CLR['accent']}"},
    )


def _page_card(children, **kwargs) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(children),
        style={"border": "none", "borderRadius": "12px",
               "boxShadow": "0 2px 16px rgba(0,0,0,0.07)", **kwargs},
        className="mb-4",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Page 1 — Data Explorer
# ═══════════════════════════════════════════════════════════════════════════════

def _explorer_layout() -> html.Div:
    return html.Div([
        _page_header("Data Explorer", "Browse ECG beats and inspect waveforms", "bi-search"),
        dbc.Container([
            _page_card([
                dbc.Row([
                    dbc.Col([
                        html.Label("Dataset", className="fw-600 small text-uppercase text-muted mb-1"),
                        dcc.Dropdown(
                            id="explorer-dataset",
                            options=[{"label": "MIT-BIH Arrhythmia", "value": "mitbih"},
                                     {"label": "PTB-XL (12-lead)", "value": "ptbxl"}],
                            value="mitbih", clearable=False,
                            style={"fontFamily": FONT},
                        ),
                    ], md=3),
                    dbc.Col([
                        html.Label("Upload custom CSV", className="fw-600 small text-uppercase text-muted mb-1"),
                        dcc.Upload(
                            id="explorer-upload",
                            children=html.Div([html.I(className="bi bi-upload me-2"),
                                               "Drag & Drop or ", html.A("Select CSV")]),
                            style={"border": "2px dashed #cbd5e1", "borderRadius": "8px",
                                   "padding": "0.5rem 1rem", "textAlign": "center",
                                   "cursor": "pointer", "color": CLR["muted"]},
                        ),
                    ], md=4),
                    dbc.Col([
                        html.Label("Navigate samples", className="fw-600 small text-uppercase text-muted mb-1"),
                        dbc.InputGroup([
                            dbc.Button(html.I(className="bi bi-chevron-left"),
                                       id="explorer-prev", n_clicks=0, color="outline-secondary", size="sm"),
                            dbc.Input(id="explorer-idx", value="0", type="number", min=0,
                                      style={"textAlign": "center", "maxWidth": "70px"}),
                            dbc.Button(html.I(className="bi bi-chevron-right"),
                                       id="explorer-next", n_clicks=0, color="outline-secondary", size="sm"),
                        ]),
                    ], md=3),
                    dbc.Col([
                        html.Div(id="explorer-label-badge", className="mt-4"),
                    ], md=2),
                ], className="mb-2"),
            ]),

            _page_card([dcc.Graph(id="explorer-ecg-plot", config={"scrollZoom": True})]),

            dbc.Row([
                dbc.Col(_page_card([dcc.Graph(id="explorer-class-dist")]), md=6),
                dbc.Col(_page_card([dcc.Graph(id="explorer-lead-corr")]),  md=6),
            ]),
        ], fluid=True, style={"padding": "1.5rem 2rem"}),
        _footer(),
    ], style={"fontFamily": FONT, "background": CLR["light_bg"], "minHeight": "100vh"})


# ═══════════════════════════════════════════════════════════════════════════════
# Page 2 — Model Inference
# ═══════════════════════════════════════════════════════════════════════════════

def _inference_layout() -> html.Div:
    return html.Div([
        _page_header("Model Inference", "Upload an ECG beat and get an instant prediction", "bi-lightning-charge-fill"),
        dbc.Container([
            _page_card([
                dbc.Row([
                    dbc.Col([
                        html.Label("Upload ECG (CSV or WFDB)",
                                   className="fw-600 small text-uppercase text-muted mb-1"),
                        dcc.Upload(
                            id="inference-upload",
                            children=html.Div([
                                html.I(className="bi bi-file-earmark-medical me-2",
                                       style={"fontSize": "1.2rem"}),
                                "Drag & Drop or ", html.A("Select file"),
                            ]),
                            style={"border": "2px dashed #cbd5e1", "borderRadius": "8px",
                                   "padding": "0.75rem 1rem", "textAlign": "center",
                                   "cursor": "pointer", "color": CLR["muted"]},
                        ),
                        html.Div(id="inference-filename",
                                 className="mt-1 small",
                                 style={"color": CLR["success"]}),
                    ], md=5),
                    dbc.Col([
                        html.Label("Model checkpoint",
                                   className="fw-600 small text-uppercase text-muted mb-1"),
                        dcc.Dropdown(id="inference-model-dropdown",
                                     placeholder="Select a saved checkpoint…",
                                     style={"fontFamily": FONT}),
                    ], md=4),
                    dbc.Col([
                        dbc.Button(
                            [html.I(className="bi bi-play-fill me-2"), "Run Inference"],
                            id="inference-run-btn", color="primary", size="lg",
                            className="mt-4 w-100",
                            style={"background": CLR["blue"], "border": "none",
                                   "borderRadius": "8px", "fontWeight": 600},
                        ),
                    ], md=3),
                ]),
            ]),

            dbc.Spinner(color="primary", children=[
                html.Div(id="inference-result", className="mb-3"),
                dbc.Row([
                    dbc.Col(_page_card([dcc.Graph(id="inference-ecg-plot")]),    md=7),
                    dbc.Col(_page_card([dcc.Graph(id="inference-confidence-plot")]), md=5),
                ]),
            ]),
        ], fluid=True, style={"padding": "1.5rem 2rem"}),
        _footer(),
    ], style={"fontFamily": FONT, "background": CLR["light_bg"], "minHeight": "100vh"})


# ═══════════════════════════════════════════════════════════════════════════════
# Page 3 — Experiment Tracker
# ═══════════════════════════════════════════════════════════════════════════════

def _tracker_layout() -> html.Div:
    return html.Div([
        _page_header("Experiment Tracker", "Compare all MLflow training runs", "bi-bar-chart-line-fill"),
        dbc.Container([
            _page_card([
                dbc.Row([
                    dbc.Col(
                        dbc.Button([html.I(className="bi bi-arrow-clockwise me-2"), "Refresh runs"],
                                   id="tracker-refresh", color="primary",
                                   style={"borderRadius": "8px", "fontWeight": 600,
                                          "background": CLR["blue"], "border": "none"}),
                        width="auto",
                    ),
                    dbc.Col(html.Div(id="tracker-status"), width="auto", className="d-flex align-items-center"),
                ]),
            ]),
            _page_card([html.Div(id="tracker-table-container")]),
            _page_card([dcc.Graph(id="tracker-metric-plot")]),
        ], fluid=True, style={"padding": "1.5rem 2rem"}),
        _footer(),
    ], style={"fontFamily": FONT, "background": CLR["light_bg"], "minHeight": "100vh"})


# ═══════════════════════════════════════════════════════════════════════════════
# Page 4 — Dataset Statistics
# ═══════════════════════════════════════════════════════════════════════════════

def _statistics_layout() -> html.Div:
    return html.Div([
        _page_header("Dataset Statistics", "Signal quality and class distribution analysis", "bi-pie-chart-fill"),
        dbc.Container([
            _page_card([
                dbc.Row([
                    dbc.Col([
                        html.Label("Dataset", className="fw-600 small text-uppercase text-muted mb-1"),
                        dcc.Dropdown(
                            id="stats-dataset",
                            options=[{"label": "MIT-BIH Arrhythmia", "value": "mitbih"},
                                     {"label": "PTB-XL (12-lead)", "value": "ptbxl"}],
                            value="mitbih", clearable=False,
                        ),
                    ], md=3),
                    dbc.Col(
                        dbc.Button([html.I(className="bi bi-bar-chart me-2"), "Load Statistics"],
                                   id="stats-load-btn", color="primary",
                                   style={"borderRadius": "8px", "fontWeight": 600,
                                          "background": CLR["blue"], "border": "none",
                                          "marginTop": "1.6rem"}),
                        md=2,
                    ),
                ]),
            ]),
            dbc.Spinner(color="primary", children=[
                dbc.Row(id="stats-cards", className="mb-2"),
                dbc.Row([
                    dbc.Col(_page_card([dcc.Graph(id="stats-class-pie")]),    md=6),
                    dbc.Col(_page_card([dcc.Graph(id="stats-lead-heatmap")]), md=6),
                ]),
            ]),
        ], fluid=True, style={"padding": "1.5rem 2rem"}),
        _footer(),
    ], style={"fontFamily": FONT, "background": CLR["light_bg"], "minHeight": "100vh"})


# ═══════════════════════════════════════════════════════════════════════════════
# Root layout — URL router
# ═══════════════════════════════════════════════════════════════════════════════

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    _navbar(),
    html.Div(id="page-content"),
    dcc.Store(id="explorer-store"),
])


@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def route(pathname: str):
    if pathname in ("/", ""):
        return _home_layout()
    if pathname == "/explorer":
        return _explorer_layout()
    if pathname == "/inference":
        return _inference_layout()
    if pathname == "/tracker":
        return _tracker_layout()
    if pathname == "/statistics":
        return _statistics_layout()
    return html.Div([
        _page_header("404", "Page not found", "bi-exclamation-triangle"),
        dbc.Container(dcc.Link("← Back to Home", href="/"), className="p-4"),
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# Callbacks — Data Explorer
# ═══════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output("explorer-store", "data"),
    Output("explorer-class-dist", "figure"),
    Input("explorer-dataset", "value"),
    prevent_initial_call=False,
)
def load_explorer_data(dataset_name: str):
    result = _load_sample_dataset(dataset_name)
    if result is None:
        empty_fig = go.Figure(layout=dict(
            title=f"{dataset_name.upper()} not downloaded — see README",
            template="plotly_white",
        ))
        return {"n_samples": 0, "dataset_name": dataset_name}, empty_fig

    X, y, labels, _ = result
    if y.ndim == 2:
        counts = y.sum(axis=0)
    else:
        unique, cnt = np.unique(y, return_counts=True)
        counts = np.zeros(len(labels))
        for u, c in zip(unique, cnt):
            if u < len(labels):
                counts[u] = c

    dist_fig = go.Figure(go.Bar(x=labels, y=counts.tolist(),
                                marker_color=CLR["blue"], marker_line_width=0))
    dist_fig.update_layout(
        title="Class Distribution", xaxis_title="AAMI Class", yaxis_title="Count",
        template="plotly_white", margin=dict(l=40, r=20, t=40, b=40),
        font=dict(family=FONT),
        plot_bgcolor="white",
    )
    return {"n_samples": int(len(X)), "dataset_name": dataset_name}, dist_fig


@app.callback(
    Output("explorer-idx", "value"),
    Input("explorer-prev", "n_clicks"),
    Input("explorer-next", "n_clicks"),
    State("explorer-idx", "value"),
    State("explorer-store", "data"),
    prevent_initial_call=True,
)
def navigate_samples(prev_clicks, next_clicks, current_idx, store):
    if not store or not store.get("n_samples"):
        raise PreventUpdate
    ctx = dash.callback_context
    idx = int(current_idx or 0)
    n = store["n_samples"]
    if ctx.triggered_id == "explorer-next":
        idx = min(idx + 1, n - 1)
    elif ctx.triggered_id == "explorer-prev":
        idx = max(idx - 1, 0)
    return str(idx)


@app.callback(
    Output("explorer-ecg-plot", "figure"),
    Output("explorer-label-badge", "children"),
    Output("explorer-lead-corr", "figure"),
    Input("explorer-idx", "value"),
    Input("explorer-dataset", "value"),
    prevent_initial_call=False,
)
def update_ecg_plot(idx_str, dataset_name):
    empty = go.Figure(layout=dict(template="plotly_white", font=dict(family=FONT)))
    result = _load_sample_dataset(dataset_name)
    if result is None:
        return empty, dbc.Alert(f"{dataset_name.upper()} not downloaded yet.", color="warning"), empty

    X, y, labels, _ = result
    idx = max(0, min(int(idx_str or 0), len(X) - 1))
    sample = X[idx]
    n_leads = sample.shape[0]
    lead_names = LEAD_NAMES_12[:n_leads] if n_leads == 12 else LEAD_NAMES_1[:n_leads]

    ecg_fig = _ecg_figure(sample, lead_names, title=f"Sample {idx}")

    label_int = int(y[idx]) if y.ndim == 1 else int(np.argmax(y[idx]))
    cls_name  = labels[label_int]
    badge_color = {"N": "success", "S": "warning", "V": "danger",
                   "F": "info", "Q": "secondary"}.get(cls_name, "primary")
    badge = dbc.Badge(
        [html.I(className="bi bi-heart-pulse me-1"), f"Class: {cls_name}"],
        color=badge_color, pill=True,
        style={"fontSize": "0.85rem", "padding": "0.4rem 0.8rem"},
    )

    if n_leads == 1:
        t = np.arange(sample.shape[1])
        corr_fig = go.Figure(go.Scatter(x=t.tolist(), y=sample[0].tolist(),
                                        mode="lines", line=dict(color=CLR["blue"], width=1.5)))
        corr_fig.update_layout(title="Signal Detail", template="plotly_white",
                                font=dict(family=FONT), margin=dict(l=40, r=20, t=40, b=40))
    else:
        corr = np.corrcoef(sample)
        corr_fig = go.Figure(go.Heatmap(z=corr.tolist(), x=lead_names, y=lead_names,
                                         colorscale="RdBu", zmid=0))
        corr_fig.update_layout(title="Lead Correlation", template="plotly_white",
                                font=dict(family=FONT), margin=dict(l=60, r=20, t=40, b=60))

    return ecg_fig, badge, corr_fig


# ═══════════════════════════════════════════════════════════════════════════════
# Callbacks — Model Inference
# ═══════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output("inference-model-dropdown", "options"),
    Input("url", "pathname"),
)
def refresh_model_list(pathname):
    return _list_model_checkpoints()


@app.callback(
    Output("inference-filename", "children"),
    Input("inference-upload", "filename"),
    prevent_initial_call=True,
)
def show_upload_filename(filename):
    if not filename:
        return ""
    return [html.I(className="bi bi-check-circle-fill me-1"), html.Strong(filename)]


@app.callback(
    Output("inference-ecg-plot", "figure"),
    Output("inference-confidence-plot", "figure"),
    Output("inference-result", "children"),
    Input("inference-run-btn", "n_clicks"),
    State("inference-upload", "contents"),
    State("inference-upload", "filename"),
    State("inference-model-dropdown", "value"),
    prevent_initial_call=True,
)
def run_inference(n_clicks, contents, filename, model_path):
    empty = go.Figure(layout=dict(template="plotly_white", font=dict(family=FONT)))
    if not contents:
        return empty, empty, dbc.Alert(
            [html.I(className="bi bi-info-circle me-2"), "Upload an ECG file first."],
            color="info")
    if not model_path:
        return empty, empty, dbc.Alert(
            [html.I(className="bi bi-info-circle me-2"), "Select a model checkpoint."],
            color="info")

    try:
        import tempfile
        import tensorflow as tf
        from data_loader import load_custom
        from preprocessor import Preprocessor
        from scipy.signal import resample as sp_resample

        _, content_string = contents.split(",")
        raw_bytes = base64.b64decode(content_string)
        suffix = Path(filename).suffix if filename else ".csv"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name
        try:
            ds = load_custom(tmp_path)
        finally:
            os.unlink(tmp_path)

        sample = ds.X[0]
        model  = tf.keras.models.load_model(model_path)
        n_leads_model = model.input_shape[1]
        n_time_model  = model.input_shape[2]

        input_fs = ds.fs if ds.fs > 0 else 360
        prep = Preprocessor(fs=input_fs, target_fs=config.SIGNAL.target_fs)
        X_proc = prep.fit_transform(ds.X)

        if X_proc.shape[2] != n_time_model:
            X_proc = sp_resample(X_proc, n_time_model, axis=2).astype(np.float32)
        if X_proc.shape[1] < n_leads_model:
            pad = np.zeros((1, n_leads_model - X_proc.shape[1], n_time_model), dtype=np.float32)
            X_proc = np.concatenate([X_proc, pad], axis=1)
        X_proc = X_proc[:, :n_leads_model, :]

        scores     = model.predict(X_proc, verbose=0)[0]
        pred_class = int(np.argmax(scores))
        n_classes  = len(scores)
        class_labels = (config.PTBXL_SUPERCLASSES[:n_classes]
                        if n_classes == len(config.PTBXL_SUPERCLASSES)
                        else config.MITBIH_CLASSES[:n_classes])

        n_leads_sample = sample.shape[0]
        lead_names = LEAD_NAMES_12[:n_leads_sample] if n_leads_sample == 12 \
            else [f"Lead {i}" for i in range(n_leads_sample)]
        ecg_fig = _ecg_figure(sample, lead_names, title=f"Uploaded ECG — {filename}")

        try:
            from evaluate import grad_cam_1d
            saliency = grad_cam_1d(model, X_proc[0], class_idx=pred_class)
            scale = float(np.abs(sample[0]).max()) or 1.0
            ecg_fig.add_trace(
                go.Scatter(x=np.arange(len(saliency)).tolist(),
                           y=(saliency * scale).tolist(),
                           mode="lines", name="Grad-CAM",
                           line=dict(color="#ef4444", width=1.5, dash="dot")),
                row=1, col=1,
            )
        except Exception:
            pass

        bar_colors = [CLR["blue"]] * len(class_labels)
        bar_colors[pred_class] = CLR["accent"]
        conf_fig = go.Figure(go.Bar(x=class_labels, y=scores.tolist(),
                                    marker_color=bar_colors, marker_line_width=0))
        conf_fig.update_layout(
            title="Class Confidence", yaxis_title="Score",
            template="plotly_white", font=dict(family=FONT),
            margin=dict(l=40, r=20, t=40, b=40),
        )

        pred_label = class_labels[pred_class] if pred_class < len(class_labels) else str(pred_class)
        badge_color = {"N": "success", "S": "warning", "V": "danger",
                       "F": "info", "Q": "secondary"}.get(pred_label, "primary")
        result_ui = dbc.Alert([
            html.I(className="bi bi-check-circle-fill me-2"),
            html.Strong("Prediction: "), pred_label,
            dbc.Badge(f"{scores[pred_class]:.1%} confidence",
                      color=badge_color, pill=True, className="ms-2"),
        ], color="success", style={"borderRadius": "10px"})

        return ecg_fig, conf_fig, result_ui

    except Exception as exc:
        return empty, empty, dbc.Alert(
            [html.I(className="bi bi-x-circle me-2"), f"Error: {exc}"],
            color="danger")


# ═══════════════════════════════════════════════════════════════════════════════
# Callbacks — Experiment Tracker
# ═══════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output("tracker-table-container", "children"),
    Output("tracker-status", "children"),
    Output("tracker-metric-plot", "figure"),
    Input("tracker-refresh", "n_clicks"),
    Input("url", "pathname"),
    prevent_initial_call=False,
)
def refresh_tracker(n_clicks, pathname):
    if pathname != "/tracker":
        raise PreventUpdate
    try:
        import mlflow
        mlflow.set_tracking_uri(config.MLFLOW.tracking_uri)
        runs = mlflow.search_runs(experiment_names=[config.MLFLOW.experiment_name])
    except Exception as exc:
        return html.Div(), dbc.Alert(f"MLflow error: {exc}", color="warning"), go.Figure()

    if runs is None or runs.empty:
        return html.P("No runs found. Train a model first.", className="text-muted"), \
               dbc.Badge("0 runs", color="secondary"), go.Figure()

    display_cols = (["run_id", "status", "start_time"]
                    + [c for c in runs.columns if c.startswith("metrics.")][:8]
                    + [c for c in runs.columns
                       if c.startswith("params.model_name") or c.startswith("params.dataset")])
    display_cols = [c for c in display_cols if c in runs.columns]
    df = runs[display_cols].copy()
    df.columns = [c.replace("metrics.", "").replace("params.", "") for c in df.columns]
    df = df.round(4)

    table = dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in df.columns],
        sort_action="native", filter_action="native", page_size=10,
        style_table={"overflowX": "auto"},
        style_cell={"fontFamily": FONT, "fontSize": 12, "padding": "8px 12px",
                    "border": "1px solid #f1f5f9"},
        style_header={"fontWeight": 700, "background": CLR["navy"],
                      "color": "white", "border": "none"},
        style_data_conditional=[
            {"if": {"filter_query": '{status} = "FINISHED"'},
             "backgroundColor": "#f0fdf4"},
            {"if": {"filter_query": '{status} = "FAILED"'},
             "backgroundColor": "#fff5f5"},
        ],
    )

    fig = go.Figure()
    if "metrics.val_accuracy" in runs.columns:
        finished = runs[runs.status == "FINISHED"].copy()
        fig.add_trace(go.Scatter(
            x=finished["start_time"].astype(str).tolist(),
            y=finished["metrics.val_accuracy"].tolist(),
            mode="markers+lines", name="Val Accuracy",
            marker=dict(size=10, color=CLR["accent"]),
            line=dict(color=CLR["blue"]),
        ))
    fig.update_layout(title="Validation Accuracy Over Runs", template="plotly_white",
                      font=dict(family=FONT), margin=dict(l=40, r=20, t=40, b=40))

    n_finished = int((runs.status == "FINISHED").sum())
    status_badge = dbc.Badge(
        f"{len(runs)} runs — {n_finished} finished",
        color="success" if n_finished else "secondary", pill=True,
    )
    return table, status_badge, fig


# ═══════════════════════════════════════════════════════════════════════════════
# Callbacks — Dataset Statistics
# ═══════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output("stats-cards", "children"),
    Output("stats-class-pie", "figure"),
    Output("stats-lead-heatmap", "figure"),
    Input("stats-load-btn", "n_clicks"),
    State("stats-dataset", "value"),
    prevent_initial_call=True,
)
def load_statistics(n_clicks, dataset_name):
    result = _load_sample_dataset(dataset_name, n_samples=500)
    if result is None:
        alert = dbc.Col(dbc.Alert("Dataset not available. Check data/raw/", color="warning"))
        return alert, go.Figure(), go.Figure()

    X, y, labels, _ = result
    n_samples, n_leads, n_timesteps = X.shape

    if y.ndim == 2:
        counts = y.sum(axis=0)
    else:
        unique, cnt = np.unique(y, return_counts=True)
        counts = np.zeros(len(labels))
        for u, c in zip(unique, cnt):
            if u < len(labels):
                counts[u] = c

    def stat_card(val, lbl, icon, color):
        return dbc.Col(dbc.Card(dbc.CardBody([
            html.Div([
                html.I(className=f"bi {icon}",
                       style={"fontSize": "1.5rem", "color": color,
                              "marginRight": "0.75rem"}),
                html.Div([
                    html.H4(str(val), className="mb-0",
                            style={"fontWeight": 700, "color": CLR["text"]}),
                    html.P(lbl, className="mb-0",
                           style={"fontSize": "0.75rem", "color": CLR["muted"]}),
                ]),
            ], className="d-flex align-items-center"),
        ]), style={"border": "none", "borderRadius": "12px",
                   "boxShadow": "0 2px 12px rgba(0,0,0,0.06)"}),
        md=2, sm=4, className="mb-3")

    cards = [
        stat_card(n_samples,   "Samples",       "bi-collection-fill",    CLR["blue"]),
        stat_card(len(labels), "Classes",        "bi-tag-fill",           "#7c3aed"),
        stat_card(n_leads,     "Leads",          "bi-activity",           "#0891b2"),
        stat_card(n_timesteps, "Timesteps",      "bi-clock-fill",         "#059669"),
        stat_card(f"{X.mean():.3f}", "Mean",     "bi-graph-up",           "#d97706"),
        stat_card(f"{X.std():.3f}",  "Std Dev",  "bi-distribute-vertical","#dc2626"),
    ]

    pie_fig = go.Figure(go.Pie(
        labels=labels, values=counts.tolist(), hole=0.4,
        marker=dict(colors=["#1565c0","#7c3aed","#ef4444","#0891b2","#059669"]),
    ))
    pie_fig.update_layout(title="Class Distribution", template="plotly_white",
                          font=dict(family=FONT))

    lead_std = X.std(axis=2)
    lead_names = LEAD_NAMES_12[:n_leads] if n_leads == 12 else [f"L{i}" for i in range(n_leads)]
    heat_fig = go.Figure(go.Heatmap(
        z=lead_std.T.tolist(), y=lead_names,
        colorscale="Blues", colorbar=dict(title="Std"),
    ))
    heat_fig.update_layout(title="Lead Signal Quality (std per sample)",
                           xaxis_title="Sample index",
                           template="plotly_white", font=dict(family=FONT))

    return cards, pie_fig, heat_fig


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True, host="0.0.0.0", port=8050)
