# src/visualization.py
# ─────────────────────────────────────────────────────────────────
# Matplotlib figure factories for the Streamlit UI.
# Every function returns a matplotlib Figure that Streamlit can
# render with st.pyplot().
# ─────────────────────────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.figure import Figure

from config import (COLOR_TRAIN, COLOR_VAL, COLOR_THEORY,
                    COLOR_EMPIRICAL, COLOR_PRED)

# Use a clean, non-interactive backend safe for Streamlit
plt.switch_backend("Agg")

# ── Shared style ──────────────────────────────────────────────────
FIGSIZE   = (9, 5)
TITLE_FS  = 13
LABEL_FS  = 11
TICK_FS   = 9
GRID_KW   = {"linestyle": "--", "alpha": 0.4, "linewidth": 0.8}


def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_title(title, fontsize=TITLE_FS, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.grid(True, **GRID_KW)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── 1. Learning curve (error vs samples) ─────────────────────────

def plot_learning_curve(lc_result: dict, target_error: float,
                         model_name: str = "",
                         theory_m: int | None = None) -> Figure:
    """Plot training error and validation error vs training size with optional error bands."""
    fig, ax = plt.subplots(figsize=FIGSIZE, tight_layout=True)

    sc = np.array(lc_result["sample_counts"], dtype=float)
    te = np.array(lc_result["train_errors"], dtype=float)
    ve = np.array(lc_result["val_errors"], dtype=float)

    te_std = np.array(lc_result.get("train_errors_std", []), dtype=float)
    ve_std = np.array(lc_result.get("val_errors_std", []), dtype=float)

    ax.plot(sc, te, "o-", color=COLOR_TRAIN, label="Training error",
            linewidth=2, markersize=5)
    if len(te_std) == len(sc) and np.any(te_std > 0):
        ax.fill_between(sc, np.clip(te - te_std, 0, 1), np.clip(te + te_std, 0, 1),
                        color=COLOR_TRAIN, alpha=0.18, label="Train ±1σ")

    ax.plot(sc, ve, "s-", color=COLOR_VAL, label="Validation error",
            linewidth=2, markersize=5)
    if len(ve_std) == len(sc) and np.any(ve_std > 0):
        ax.fill_between(sc, np.clip(ve - ve_std, 0, 1), np.clip(ve + ve_std, 0, 1),
                        color=COLOR_VAL, alpha=0.18, label="Val ±1σ")

    ax.axhline(target_error, color="crimson", linestyle="--",
               linewidth=1.5, label=f"Target ε = {target_error:.3f}")

    if theory_m is not None and theory_m > 0:
        if theory_m <= sc.max() * 2.5:
            ax.axvline(theory_m, color=COLOR_THEORY, linestyle="-.",
                       linewidth=1.5, label=f"PAC Bound m = {theory_m:,}")
        else:
            # Theoretical bound is far off to the right (common in PAC theory)
            ax.text(0.98, 0.94, f"Theoretical PAC bound m = {theory_m:,} (off-chart)",
                    transform=ax.transAxes, fontsize=8.5, color=COLOR_THEORY,
                    ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#f3e8fd", ec=COLOR_THEORY, lw=1))

    _style_ax(ax,
              title=f"Learning Curve (Error Rate) – {model_name}",
              xlabel="Training set size (samples)",
              ylabel="Error rate (1 − accuracy)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=TICK_FS, loc="best")
    return fig


def plot_accuracy_curve(lc_result: dict, target_error: float,
                        model_name: str = "") -> Figure:
    """Plot training accuracy and validation accuracy vs training size."""
    fig, ax = plt.subplots(figsize=FIGSIZE, tight_layout=True)

    sc = np.array(lc_result["sample_counts"], dtype=float)
    ta = np.array(lc_result["train_accuracies"], dtype=float)
    va = np.array(lc_result["val_accuracies"], dtype=float)

    ta_std = np.array(lc_result.get("train_accuracies_std", []), dtype=float)
    va_std = np.array(lc_result.get("val_accuracies_std", []), dtype=float)

    target_acc = 1.0 - target_error

    ax.plot(sc, ta, "o-", color=COLOR_TRAIN, label="Training accuracy",
            linewidth=2, markersize=5)
    if len(ta_std) == len(sc) and np.any(ta_std > 0):
        ax.fill_between(sc, np.clip(ta - ta_std, 0, 1), np.clip(ta + ta_std, 0, 1),
                        color=COLOR_TRAIN, alpha=0.18, label="Train ±1σ")

    ax.plot(sc, va, "s-", color=COLOR_EMPIRICAL, label="Validation accuracy",
            linewidth=2, markersize=5)
    if len(va_std) == len(sc) and np.any(va_std > 0):
        ax.fill_between(sc, np.clip(va - va_std, 0, 1), np.clip(va + va_std, 0, 1),
                        color=COLOR_EMPIRICAL, alpha=0.18, label="Val ±1σ")

    ax.axhline(target_acc, color="crimson", linestyle="--",
               linewidth=1.5, label=f"Target 1−ε = {target_acc:.3f} ({target_acc*100:.1f}%)")

    _style_ax(ax,
              title=f"Learning Curve (Accuracy) – {model_name}",
              xlabel="Training set size (samples)",
              ylabel="Accuracy")
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=TICK_FS, loc="lower right")
    return fig


def plot_dataset_2d(X: np.ndarray, y: np.ndarray, feature_names: list,
                    dataset_name: str = "") -> Figure:
    """Plot 2D feature projection or first 2 features with class labels."""
    fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)

    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    # Impute any NaNs in plotting projection
    if np.isnan(X_arr).any():
        col_medians = np.nanmedian(X_arr, axis=0)
        inds = np.where(np.isnan(X_arr))
        X_arr[inds] = np.take(col_medians, inds[1])
        X_arr = np.nan_to_num(X_arr, nan=0.0)

    x0 = X_arr[:, 0]
    if X_arr.shape[1] > 1:
        x1 = X_arr[:, 1]
        f2 = feature_names[1] if len(feature_names) > 1 else "Feature 2"
    else:
        x1 = np.zeros_like(x0)
        f2 = "Index / Zero"

    classes = np.unique(y)
    cmap = plt.get_cmap("tab10")

    for idx, cls in enumerate(classes):
        mask = (y == cls)
        ax.scatter(x0[mask], x1[mask], label=f"Class {cls}",
                   alpha=0.8, edgecolors="none", s=45, color=cmap(idx % 10))

    f1 = feature_names[0] if len(feature_names) > 0 else "Feature 1"

    _style_ax(ax,
              title=f"Feature Space – {dataset_name}",
              xlabel=f1,
              ylabel=f2)
    ax.legend(fontsize=TICK_FS)
    return fig


# ── 2. Train vs validation error ─────────────────────────────────

def plot_train_vs_val(lc_result: dict, model_name: str = "") -> Figure:
    """Scatter: training error (x) vs validation error (y) per size."""
    fig, ax = plt.subplots(figsize=(6, 5), tight_layout=True)

    te = lc_result["train_errors"]
    ve = lc_result["val_errors"]
    sc = lc_result["sample_counts"]

    sc_arr = np.array(sc, dtype=float)
    norm   = plt.Normalize(sc_arr.min(), sc_arr.max())
    cmap   = plt.cm.viridis

    scatter = ax.scatter(te, ve, c=sc_arr, cmap=cmap,
                         norm=norm, s=80, zorder=3)
    for x, y, n in zip(te, ve, sc):
        ax.annotate(str(n), (x, y), textcoords="offset points",
                    xytext=(4, 4), fontsize=7)

    cb = fig.colorbar(scatter, ax=ax)
    cb.set_label("Training size", fontsize=TICK_FS)

    lo = min(min(te), min(ve)) - 0.01
    hi = max(max(te), max(ve)) + 0.01
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="Train = Val")
    ax.legend(fontsize=TICK_FS)
    _style_ax(ax,
              title=f"Training vs Validation Error – {model_name}",
              xlabel="Training error",
              ylabel="Validation error")
    return fig


# ── 3. Power-law fit overlay ──────────────────────────────────────

def plot_power_law_fit(lc_result: dict, fit_result: dict,
                       pred_result: dict, target_error: float,
                       model_name: str = "") -> Figure:
    """Overlay the fitted power-law curve on the learning curve."""
    from src.curve_fitting import power_law

    fig, ax = plt.subplots(figsize=FIGSIZE, tight_layout=True)

    sc = np.array(lc_result["sample_counts"], dtype=float)
    ve = np.array(lc_result["val_errors"],    dtype=float)

    ax.scatter(sc, ve, color=COLOR_VAL, zorder=3, s=60,
               label="Observed validation error")

    if fit_result["success"] and fit_result["params"] is not None:
        a, b, c = fit_result["params"]
        m_range  = np.linspace(sc.min(), max(sc.max() * 2, 500), 400)
        ax.plot(m_range, power_law(m_range, a, b, c),
                color=COLOR_PRED, linewidth=2,
                label=f"Fit: {a:.3f}·m^(-{b:.3f}) + {c:.3f}")

        if pred_result.get("achieved") and pred_result.get("sample_count"):
            m_star = pred_result["sample_count"]
            ax.axvline(m_star, color=COLOR_PRED, linestyle=":",
                       linewidth=1.5,
                       label=f"Predicted m* = {m_star:,}")

    ax.axhline(target_error, color="crimson", linestyle="--",
               linewidth=1.5, label=f"Target ε = {target_error:.3f}")

    _style_ax(ax,
              title=f"Power-Law Fit – {model_name}",
              xlabel="Training set size (samples)",
              ylabel="Validation error")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=TICK_FS)
    return fig


# ── 4. Comparison bar chart ───────────────────────────────────────

def plot_comparison_bars(comparison_rows: list) -> Figure:
    """
    Bar chart: theoretical / empirical / predicted sample counts
    side-by-side for each model.
    `comparison_rows` is a list of dicts from build_comparison_dataframe.
    """
    models    = [r["Model"] for r in comparison_rows]
    theory    = []
    empirical = []
    predicted = []

    for r in comparison_rows:
        def _safe(v):
            try:
                return float(v) if v not in (None, "N/A", "Not achieved",
                                              "Unavailable") else np.nan
            except (ValueError, TypeError):
                return np.nan

        theory.append(_safe(r["Theoretical Bound (m)"]))
        empirical.append(_safe(r["Empirical Requirement (m)"]))
        predicted.append(_safe(r["Predicted Requirement (m)"]))

    x   = np.arange(len(models))
    w   = 0.25
    fig, ax = plt.subplots(figsize=(max(7, len(models) * 2.5), 5),
                            tight_layout=True)

    b1 = ax.bar(x - w, theory,    w, label="Theoretical",
                color=COLOR_THEORY,   alpha=0.85)
    b2 = ax.bar(x,     empirical,  w, label="Empirical",
                color=COLOR_EMPIRICAL, alpha=0.85)
    b3 = ax.bar(x + w, predicted,  w, label="Predicted (fit)",
                color=COLOR_PRED,      alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=TICK_FS)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{int(v):,}" if v >= 1 else f"{v:.3f}"
    ))

    _style_ax(ax,
              title="Theoretical vs Empirical vs Predicted Sample Requirements",
              xlabel="Model",
              ylabel="Sample count (log scale)")
    ax.legend(fontsize=TICK_FS)
    return fig


# ── 5. Multi-model learning curves ───────────────────────────────

def plot_multi_model_curves(results: list, target_error: float) -> Figure:
    """Overlay validation-error curves for all models."""
    colors = [COLOR_TRAIN, COLOR_VAL, COLOR_PRED, COLOR_THEORY]
    markers = ["o", "s", "^", "D"]

    fig, ax = plt.subplots(figsize=FIGSIZE, tight_layout=True)

    for i, r in enumerate(results):
        lc  = r["learning_curve"]
        col = colors[i % len(colors)]
        mk  = markers[i % len(markers)]
        ax.plot(lc["sample_counts"], lc["val_errors"],
                f"{mk}-", color=col, linewidth=2, markersize=5,
                label=r["model_name"])

    ax.axhline(target_error, color="crimson", linestyle="--",
               linewidth=1.5, label=f"Target ε = {target_error:.3f}")

    _style_ax(ax,
              title="Validation Error vs Training Size (All Models)",
              xlabel="Training set size (samples)",
              ylabel="Validation error")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=TICK_FS)
    return fig
