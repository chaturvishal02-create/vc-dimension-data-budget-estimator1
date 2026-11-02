# src/curve_fitting.py
# ─────────────────────────────────────────────────────────────────
# Fits a power-law learning curve of the form
#     error(m) = a * m^(-b) + c
# to the empirical validation-error measurements, then predicts
# the smallest sample size that achieves the target error.
# ─────────────────────────────────────────────────────────────────

import numpy as np

try:
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from config import (CURVE_FIT_P0, CURVE_FIT_BOUNDS, CURVE_FIT_MAXFEV,
                    CURVE_SEARCH_MIN, CURVE_SEARCH_MAX)


# ── Power-law model ───────────────────────────────────────────────

def power_law(m, a, b, c):
    """error(m) = a * m^(-b) + c"""
    return a * np.power(m, -b) + c


# ── Curve fitter ─────────────────────────────────────────────────

def fit_power_law(sample_counts: list, val_errors: list) -> dict:
    """
    Fit a power-law model to validation-error measurements.

    Returns
    -------
    dict with keys:
        success   : bool
        params    : tuple(a, b, c) or None
        covariance: ndarray or None
        error     : str or None  (explanation if fit failed)
        note      : str
    """
    if not SCIPY_AVAILABLE:
        return {
            "success": False,
            "params": None,
            "covariance": None,
            "error": "SciPy is not installed; curve fitting unavailable.",
            "note": "Install scipy to enable power-law fitting.",
        }

    m = np.array(sample_counts, dtype=float)
    e = np.array(val_errors,    dtype=float)

    if len(m) < 3:
        return {
            "success": False,
            "params": None,
            "covariance": None,
            "error": "Fewer than 3 data points – cannot fit a 3-parameter model.",
            "note": "Run the learning curve with more data fractions.",
        }

    # Guard against zero or negative sample counts
    valid = (m > 0) & (e >= 0)
    m, e = m[valid], e[valid]
    if len(m) < 3:
        return {
            "success": False,
            "params": None,
            "covariance": None,
            "error": "Insufficient valid data points after filtering.",
            "note": "",
        }

    try:
        popt, pcov = curve_fit(
            power_law,
            m, e,
            p0=CURVE_FIT_P0,
            bounds=CURVE_FIT_BOUNDS,
            maxfev=CURVE_FIT_MAXFEV,
        )
        a, b, c = popt
        return {
            "success": True,
            "params": (float(a), float(b), float(c)),
            "covariance": pcov,
            "error": None,
            "note": (
                f"Fitted: error(m) = {a:.4f} × m^(-{b:.4f}) + {c:.4f}. "
                "Extrapolated predictions may not be reliable far outside "
                "the observed data range."
            ),
        }
    except Exception as exc:
        return {
            "success": False,
            "params": None,
            "covariance": None,
            "error": str(exc),
            "note": (
                "The power-law fit failed. This may happen when the "
                "learning curve is very flat or noisy."
            ),
        }


# ── Sample predictor ──────────────────────────────────────────────

def predict_sample_size(fit_result: dict, target_error: float) -> dict:
    """
    Predict the smallest sample count m* such that
    power_law(m*, a, b, c) ≤ target_error.

    Returns a dict with 'achieved', 'sample_count', 'note'.
    """
    if not fit_result["success"]:
        return {
            "achieved": False,
            "sample_count": None,
            "note": (
                "Curve fit was unsuccessful; prediction unavailable. "
                + (fit_result.get("error") or "")
            ),
        }

    a, b, c = fit_result["params"]

    # Asymptotic error = c; if target is below c, unachievable
    if target_error <= c:
        return {
            "achieved": False,
            "sample_count": None,
            "note": (
                f"Target error ({target_error:.4f}) is at or below the "
                f"fitted asymptote c={c:.4f}. The power-law model predicts "
                "this error cannot be achieved by adding more data."
            ),
        }

    # Analytic solution: a * m^(-b) = target - c
    #   → m = (a / (target - c))^(1/b)
    try:
        m_star_raw = (a / (target_error - c)) ** (1.0 / b)
        m_star = int(np.ceil(m_star_raw))
        m_star = max(CURVE_SEARCH_MIN, min(m_star, CURVE_SEARCH_MAX))

        # Safety check: verify prediction
        predicted_err = power_law(m_star, a, b, c)

        return {
            "achieved": True,
            "sample_count": m_star,
            "predicted_error": float(predicted_err),
            "note": (
                f"Power-law model predicts ~{m_star:,} training samples "
                f"needed to reach validation error ≤ {target_error:.4f} "
                f"(predicted error at m={m_star:,}: {predicted_err:.4f}). "
                "This is a model-based extrapolation; actual results may differ."
            ),
        }
    except Exception as exc:
        return {
            "achieved": False,
            "sample_count": None,
            "note": f"Prediction failed: {exc}",
        }


# Alias for consistent naming
predict_sample_requirement = predict_sample_size

