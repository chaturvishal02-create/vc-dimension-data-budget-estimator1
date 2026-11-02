# src/visualizations.py
# ─────────────────────────────────────────────────────────────────
# Interactive Plotly & Matplotlib Data Visualizations.
# Supports both Light and Dark themes for dashboard responsiveness.
# ─────────────────────────────────────────────────────────────────

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config import (
    COLOR_TRAIN, COLOR_VAL, COLOR_THEORY, COLOR_EMPIRICAL,
    COLOR_PRED, COLOR_ACCENT, COLOR_DARK_BG, COLOR_LIGHT_BG
)
from src.curve_fitting import power_law


def _get_plotly_template(is_dark: bool = False) -> Dict[str, Any]:
    """Return styling configuration for Plotly based on active theme."""
    bg_color = "#1E293B" if is_dark else "#FFFFFF"
    paper_color = "#0F172A" if is_dark else "#F8FAFC"
    font_color = "#F1F5F9" if is_dark else "#1E293B"
    grid_color = "#334155" if is_dark else "#E2E8F0"

    return {
        "plot_bgcolor": bg_color,
        "paper_bgcolor": paper_color,
        "font": {"color": font_color, "family": "Inter, -apple-system, sans-serif"},
        "xaxis": {"gridcolor": grid_color, "zerolinecolor": grid_color},
        "yaxis": {"gridcolor": grid_color, "zerolinecolor": grid_color},
    }


# ── 1. Interactive Learning Curve ────────────────────────────────

def plot_plotly_learning_curve(
    lc_result: Dict[str, Any],
    target_error: float,
    model_name: str = "",
    theory_m: Optional[int] = None,
    is_dark: bool = False,
) -> go.Figure:
    """
    Interactive Plotly learning curve with shaded confidence intervals (±1σ)
    for both training and validation errors, target error threshold, and PAC bound line.
    """
    sc = np.array(lc_result.get("sample_counts", []), dtype=float)
    te = np.array(lc_result.get("train_errors", []), dtype=float)
    ve = np.array(lc_result.get("val_errors", []), dtype=float)
    te_std = np.array(lc_result.get("train_errors_std", np.zeros_like(te)), dtype=float)
    ve_std = np.array(lc_result.get("val_errors_std", np.zeros_like(ve)), dtype=float)

    fig = go.Figure()

    if len(sc) > 0:
        # Shaded variance band for Training Error
        if np.any(te_std > 0):
            fig.add_trace(go.Scatter(
                x=np.concatenate([sc, sc[::-1]]),
                y=np.concatenate([np.clip(te + te_std, 0, 1), np.clip(te - te_std, 0, 1)[::-1]]),
                fill="toself",
                fillcolor="rgba(59, 130, 246, 0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
                name="Train ±1σ",
            ))

        # Training Error line
        fig.add_trace(go.Scatter(
            x=sc, y=te,
            mode="lines+markers",
            name="Training Error",
            line=dict(color=COLOR_TRAIN, width=2.5),
            marker=dict(size=7, symbol="circle"),
            hovertemplate="<b>Train Error</b><br>Samples: %{x:,}<br>Error: %{y:.4f}<extra></extra>",
        ))

        # Shaded variance band for Validation Error
        if np.any(ve_std > 0):
            fig.add_trace(go.Scatter(
                x=np.concatenate([sc, sc[::-1]]),
                y=np.concatenate([np.clip(ve + ve_std, 0, 1), np.clip(ve - ve_std, 0, 1)[::-1]]),
                fill="toself",
                fillcolor="rgba(249, 115, 22, 0.18)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
                name="Val ±1σ",
            ))

        # Validation Error line
        fig.add_trace(go.Scatter(
            x=sc, y=ve,
            mode="lines+markers",
            name="Validation Error",
            line=dict(color=COLOR_VAL, width=3),
            marker=dict(size=8, symbol="square"),
            hovertemplate="<b>Validation Error</b><br>Samples: %{x:,}<br>Error: %{y:.4f}<extra></extra>",
        ))

    # Target error horizontal threshold
    fig.add_hline(
        y=target_error,
        line=dict(color="#DC2626", dash="dash", width=2),
        annotation_text=f"Target ε = {target_error:.3f}",
        annotation_position="top right",
    )

    # Theoretical PAC bound vertical marker if reasonably close
    if theory_m is not None and theory_m > 0 and len(sc) > 0:
        if theory_m <= sc.max() * 3.0:
            fig.add_vline(
                x=theory_m,
                line=dict(color=COLOR_THEORY, dash="dot", width=2),
                annotation_text=f"PAC Bound m = {theory_m:,}",
                annotation_position="top left",
            )

    t_cfg = _get_plotly_template(is_dark)
    fig.update_layout(
        title=f"<b>Empirical Learning Curve</b> — {model_name}",
        xaxis_title="Training Subset Size (m)",
        yaxis_title="Error Rate (1 - Accuracy)",
        yaxis_range=[0, min(1.0, max(0.5, float(ve.max() * 1.25) if len(ve) > 0 else 0.5))],
        template="plotly_dark" if is_dark else "plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        **t_cfg,
    )
    return fig


# ── 2. Generalization Gap Plot ────────────────────────────────────

def plot_plotly_generalization_gap(
    lc_result: Dict[str, Any],
    model_name: str = "",
    is_dark: bool = False,
) -> go.Figure:
    """Plot the generalization gap (Val Error - Train Error) vs training sample size."""
    sc = lc_result.get("sample_counts", [])
    gaps = lc_result.get("generalization_gaps", [])

    if not gaps and len(sc) > 0:
        te = np.array(lc_result.get("train_errors", []))
        ve = np.array(lc_result.get("val_errors", []))
        gaps = (ve - te).tolist()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sc,
        y=gaps,
        name="Generalization Gap",
        marker_color=np.where(np.array(gaps) >= 0, "#F59E0B", "#10B981"),
        hovertemplate="<b>Sample Size: %{x:,}</b><br>Generalization Gap: %{y:.4f}<extra></extra>",
    ))

    t_cfg = _get_plotly_template(is_dark)
    fig.update_layout(
        title=f"<b>Generalization Gap (Overfitting Shrinkage)</b> — {model_name}",
        xaxis_title="Training Set Size (m)",
        yaxis_title="Gap: Val Error - Train Error",
        template="plotly_dark" if is_dark else "plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
        **t_cfg,
    )
    return fig


# ── 3. Data Budget Comparison Bar Chart ──────────────────────────

def plot_plotly_budget_comparison(
    comparison_rows: List[Dict[str, Any]],
    current_dataset_size: int = 0,
    is_dark: bool = False,
) -> go.Figure:
    """
    Log-scale comparative bar chart displaying:
    - Theoretical PAC Bound (m_theory)
    - Empirical Requirement (m_emp)
    - Power-law Predicted Requirement (m_pred)
    - Available Training Set Size
    """
    models = [r.get("Model", "Model") for r in comparison_rows]
    theory_vals = []
    emp_vals = []
    pred_vals = []

    for r in comparison_rows:
        def _safe_float(v):
            try:
                f = float(str(v).replace(",", "").replace("×", ""))
                return f if f > 0 else np.nan
            except Exception:
                return np.nan

        theory_vals.append(_safe_float(r.get("Theoretical Bound (m)")))
        emp_vals.append(_safe_float(r.get("Empirical Requirement (m)")))
        pred_vals.append(_safe_float(r.get("Predicted Requirement (m)")))

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Theoretical PAC Bound",
        x=models,
        y=theory_vals,
        marker_color=COLOR_THEORY,
        text=[f"{v:,.0f}" if not np.isnan(v) else "N/A" for v in theory_vals],
        textposition="auto",
        hovertemplate="<b>%{x}</b><br>Theoretical: %{y:,.0f} samples<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        name="Empirically Observed",
        x=models,
        y=emp_vals,
        marker_color=COLOR_EMPIRICAL,
        text=[f"{v:,.0f}" if not np.isnan(v) else "Not met" for v in emp_vals],
        textposition="auto",
        hovertemplate="<b>%{x}</b><br>Empirical: %{y:,.0f} samples<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        name="Power-Law Extrapolated",
        x=models,
        y=pred_vals,
        marker_color=COLOR_PRED,
        text=[f"{v:,.0f}" if not np.isnan(v) else "N/A" for v in pred_vals],
        textposition="auto",
        hovertemplate="<b>%{x}</b><br>Predicted: %{y:,.0f} samples<extra></extra>",
    ))

    if current_dataset_size > 0:
        fig.add_hline(
            y=current_dataset_size,
            line=dict(color="#06B6D4", dash="dash", width=2.5),
            annotation_text=f"Current Data: {current_dataset_size:,}",
            annotation_position="bottom right",
        )

    t_cfg = _get_plotly_template(is_dark)
    fig.update_layout(
        title="<b>Sample Size Requirement Comparison (Log Scale)</b>",
        xaxis_title="Evaluated Model",
        yaxis_title="Required Samples (Log Scale)",
        yaxis_type="log",
        barmode="group",
        template="plotly_dark" if is_dark else "plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        **t_cfg,
    )
    return fig


# ── 4. Multi-Model Leaderboard Comparison ─────────────────────────

def plot_plotly_model_leaderboard(
    comparison_df: pd.DataFrame,
    metric_col: str = "Final Val Accuracy (%)",
    is_dark: bool = False,
) -> go.Figure:
    """Horizontal bar chart comparing models by accuracy or error rate."""
    if comparison_df.empty or "Model" not in comparison_df.columns:
        return go.Figure()

    df = comparison_df.copy()
    vals = pd.to_numeric(df[metric_col].astype(str).str.replace("%", ""), errors="coerce").fillna(0)
    df["numeric_score"] = vals
    df = df.sort_values(by="numeric_score", ascending=True)

    fig = go.Figure(go.Bar(
        x=df["numeric_score"],
        y=df["Model"],
        orientation="h",
        marker=dict(
            color=df["numeric_score"],
            colorscale="Viridis",
            showscale=False,
        ),
        text=[f"{v:.2f}%" for v in df["numeric_score"]],
        textposition="inside",
        hovertemplate="<b>%{y}</b><br>Score: %{x:.2f}%<extra></extra>",
    ))

    t_cfg = _get_plotly_template(is_dark)
    fig.update_layout(
        title=f"<b>Model Performance Comparison</b> ({metric_col})",
        xaxis_title=metric_col,
        yaxis_title="Model",
        template="plotly_dark" if is_dark else "plotly_white",
        margin=dict(l=60, r=40, t=60, b=40),
        **t_cfg,
    )
    return fig


# ── 5. PAC Bound Sensitivity Curves ──────────────────────────────

def plot_plotly_pac_sensitivity(
    sensitivity_data: Dict[str, List[float]],
    param_type: str = "epsilon",
    is_dark: bool = False,
) -> go.Figure:
    """
    Plot sample complexity m vs varying parameter (epsilon, delta, or VC dimension).
    Compares Realizable (Blumer et al.) vs Agnostic (Anthony & Bartlett).
    """
    x_key = "epsilons" if param_type == "epsilon" else ("deltas" if param_type == "delta" else "vc_values")
    x_title = "Error Tolerance (ε)" if param_type == "epsilon" else ("Failure Probability (δ)" if param_type == "delta" else "VC Dimension (d)")

    x_vals = sensitivity_data.get(x_key, [])
    y_real = sensitivity_data.get("realizable_m", [])
    y_agno = sensitivity_data.get("agnostic_m", [])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_real,
        mode="lines+markers",
        name="Realizable Bound (Blumer et al.)",
        line=dict(color=COLOR_THEORY, width=3),
        hovertemplate="<b>Realizable</b><br>Param: %{x}<br>Samples m: %{y:,}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_agno,
        mode="lines+markers",
        name="Agnostic Bound (Anthony & Bartlett)",
        line=dict(color=COLOR_PRED, width=3, dash="dash"),
        hovertemplate="<b>Agnostic</b><br>Param: %{x}<br>Samples m: %{y:,}<extra></extra>",
    ))

    t_cfg = _get_plotly_template(is_dark)
    fig.update_layout(
        title=f"<b>PAC Sample Complexity Sensitivity: m vs {x_title}</b>",
        xaxis_title=x_title,
        yaxis_title="Required Sample Size m (Log Scale)",
        yaxis_type="log",
        template="plotly_dark" if is_dark else "plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        **t_cfg,
    )
    return fig


# ── 6. 2D Feature Projection Scatter Plot ────────────────────────

def plot_plotly_feature_projection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    task_type: str = "classification",
    dataset_name: str = "",
    is_dark: bool = False,
) -> go.Figure:
    """Interactive 2D scatter plot colored by class label or continuous target."""
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    f1 = feature_names[0] if len(feature_names) > 0 else "Feature 1"
    f2 = feature_names[1] if len(feature_names) > 1 else "Feature 2"

    x0 = X_arr[:, 0]
    x1 = X_arr[:, 1] if X_arr.shape[1] > 1 else np.zeros_like(x0)

    fig = go.Figure()
    is_class = (task_type.lower() == "classification")

    if is_class:
        classes = np.unique(y)
        for cls in classes:
            mask = (y == cls)
            fig.add_trace(go.Scatter(
                x=x0[mask],
                y=x1[mask],
                mode="markers",
                name=f"Class {cls}",
                marker=dict(size=8, opacity=0.8),
                hovertemplate=f"<b>Class {cls}</b><br>{f1}: %{{x:.3f}}<br>{f2}: %{{y:.3f}}<extra></extra>",
            ))
    else:
        fig.add_trace(go.Scatter(
            x=x0,
            y=x1,
            mode="markers",
            marker=dict(
                size=8,
                color=y,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Target Value"),
            ),
            hovertemplate=f"{f1}: %{{x:.3f}}<br>{f2}: %{{y:.3f}}<br>Target: %{{marker.color:.3f}}<extra></extra>",
        ))

    t_cfg = _get_plotly_template(is_dark)
    fig.update_layout(
        title=f"<b>2D Feature Space Projection</b> — {dataset_name}",
        xaxis_title=f1,
        yaxis_title=f2,
        template="plotly_dark" if is_dark else "plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
        **t_cfg,
    )
    return fig


# ── 7. Power-Law Fit Overlay ──────────────────────────────────────

def plot_plotly_power_law_fit(
    lc_result: Dict[str, Any],
    fit_result: Dict[str, Any],
    pred_result: Dict[str, Any],
    target_error: float,
    model_name: str = "",
    is_dark: bool = False,
) -> go.Figure:
    """Plot observed empirical validation error alongside fitted power law curve."""
    sc = np.array(lc_result.get("sample_counts", []), dtype=float)
    ve = np.array(lc_result.get("val_errors", []), dtype=float)

    fig = go.Figure()

    # Observed points
    fig.add_trace(go.Scatter(
        x=sc,
        y=ve,
        mode="markers",
        name="Observed Validation Error",
        marker=dict(size=9, color=COLOR_VAL, symbol="diamond"),
        hovertemplate="<b>Observed</b><br>Samples: %{x:,}<br>Error: %{y:.4f}<extra></extra>",
    ))

    # Fitted curve
    if fit_result.get("success") and fit_result.get("params"):
        a, b, c = fit_result["params"]
        m_max = max(sc.max() * 2.5, 500) if len(sc) > 0 else 500
        m_range = np.linspace(sc.min() if len(sc) > 0 else 10, m_max, 300)
        fitted_y = power_law(m_range, a, b, c)

        fig.add_trace(go.Scatter(
            x=m_range,
            y=fitted_y,
            mode="lines",
            name=f"Power Law: {a:.3f}·m^(-{b:.3f}) + {c:.3f}",
            line=dict(color=COLOR_PRED, width=2.5),
            hovertemplate="<b>Fitted Power Law</b><br>m: %{x:,.0f}<br>Error: %{y:.4f}<extra></extra>",
        ))

        # Predicted m*
        m_star = pred_result.get("sample_count")
        if pred_result.get("achieved") and m_star:
            fig.add_vline(
                x=m_star,
                line=dict(color=COLOR_PRED, dash="dot", width=2),
                annotation_text=f"Predicted m* = {m_star:,}",
                annotation_position="top right",
            )

    # Target error line
    fig.add_hline(
        y=target_error,
        line=dict(color="#DC2626", dash="dash", width=2),
        annotation_text=f"Target ε = {target_error:.3f}",
        annotation_position="bottom right",
    )

    t_cfg = _get_plotly_template(is_dark)
    fig.update_layout(
        title=f"<b>Power-Law Extrapolation</b> — {model_name}",
        xaxis_title="Training Subset Size (m)",
        yaxis_title="Validation Error Rate",
        template="plotly_dark" if is_dark else "plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        **t_cfg,
    )
    return fig

