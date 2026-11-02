# src/evaluation.py
# ─────────────────────────────────────────────────────────────────
# Orchestrates experiments: executes learning curve generation,
# computes VC dimension, calculates PAC bounds (realizable & agnostic),
# fits power-law extrapolation, and builds comparison dataframes.
# ─────────────────────────────────────────────────────────────────

from typing import List, Dict, Any, Optional
import time
import numpy as np
import pandas as pd

from src.learning_curve import generate_learning_curve, find_empirical_sample_requirement
from src.curve_fitting  import fit_power_law, predict_sample_requirement, predict_sample_size
from src.pac_bound      import calculate_theoretical_samples, calculate_agnostic_samples, calculate_regression_samples
from src.vc_dimension   import estimate_vc_dimension


def run_full_experiment(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    epsilon: float,
    delta:   float,
    n_features: int,
    n_seeds: int = 1,
    task_type: str = "classification",
    **model_kwargs,
) -> Dict[str, Any]:
    """
    Orchestrate full evaluation for a single model:
      1. Learning curve generation across fractions with multi-seed repeats
      2. VC dimension estimation with assumptions and confidence
      3. PAC bound calculation (Realizable, Agnostic, or Regression)
      4. Power-law curve fitting & sample prediction
      5. Empirical requirement finding
      6. Summary metrics compilation
    """
    start_t = time.perf_counter()

    # 1. Learning curve
    lc = generate_learning_curve(
        model_name=model_name,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_seeds=n_seeds,
        task_type=task_type,
        **model_kwargs,
    )

    # 2. VC dimension
    vc = estimate_vc_dimension(model_name, n_features, **model_kwargs)

    # 3. PAC bounds
    if task_type == "regression":
        try:
            target_scale = float(np.ptp(y_train)) if len(y_train) > 0 else 1.0
            pac = calculate_regression_samples(epsilon, delta, vc["vc_dim"], target_scale_M=max(0.5, target_scale))
        except Exception as exc:
            pac = {"m": None, "error": str(exc)}
        pac_agnostic = pac
    else:
        try:
            pac = calculate_theoretical_samples(epsilon, delta, vc["vc_dim"])
        except Exception as exc:
            pac = {"m": None, "error": str(exc)}

        try:
            pac_agnostic = calculate_agnostic_samples(epsilon, delta, vc["vc_dim"])
        except Exception as exc:
            pac_agnostic = {"m": None, "error": str(exc)}

    # 4. Curve fit + prediction
    fit = fit_power_law(lc["sample_counts"], lc["val_errors"])
    pred = predict_sample_size(fit, epsilon)

    # 5. Empirical requirement
    emp = find_empirical_sample_requirement(lc, epsilon)

    total_duration = time.perf_counter() - start_t

    final_tr_err = lc["train_errors"][-1] if lc["train_errors"] else None
    final_va_err = lc["val_errors"][-1] if lc["val_errors"] else None
    final_va_acc = lc["val_accuracies"][-1] if lc["val_accuracies"] else None

    return {
        "model_name":      model_name,
        "task_type":       task_type,
        "learning_curve":  lc,
        "vc_info":         vc,
        "pac_bound":       pac,
        "pac_agnostic":    pac_agnostic,
        "curve_fit":       fit,
        "prediction":      pred,
        "empirical":       emp,
        "final_train_err": final_tr_err,
        "final_val_err":   final_va_err,
        "final_val_acc":   final_va_acc,
        "epsilon":         epsilon,
        "delta":           delta,
        "n_features":      n_features,
        "fit_time_seconds": total_duration,
        "generalization_gap": float(final_va_err - final_tr_err) if (final_va_err is not None and final_tr_err is not None) else 0.0,
    }


def build_comparison_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Build the comparison table from a list of experiment result dicts.
    Strictly preserves legacy column names for existing tests.
    """
    rows = []
    for r in results:
        pac_info = r.get("pac_bound") or {}
        emp_info = r.get("empirical") or {}
        pred_info = r.get("prediction") or {}
        vc_info = r.get("vc_info") or {}

        pac_m = pac_info.get("m")
        emp_m = emp_info.get("sample_count")
        pred_m = pred_info.get("sample_count")

        # Theoretical-to-empirical ratio
        ratio = None
        if pac_m is not None and emp_m is not None and emp_m > 0:
            ratio = pac_m / emp_m

        final_train_err = r.get("final_train_err")
        final_val_err   = r.get("final_val_err")
        final_val_acc   = r.get("final_val_acc")

        rows.append({
            "Model":                      r.get("model_name", "Unknown"),
            "VC-Dim Estimate":            vc_info.get("vc_dim", "N/A"),
            "VC-Dim Label":               vc_info.get("label", "N/A"),
            "Theoretical Bound (m)":      pac_m if pac_m is not None else "N/A",
            "Empirical Requirement (m)":  emp_m if emp_m is not None else "Not achieved",
            "Predicted Requirement (m)":  pred_m if pred_m is not None else "Unavailable",
            "Final Train Error":          f"{final_train_err:.4f}" if final_train_err is not None else "N/A",
            "Final Val Error":            f"{final_val_err:.4f}"   if final_val_err   is not None else "N/A",
            "Final Val Accuracy (%)":     f"{final_val_acc*100:.2f}" if final_val_acc is not None else "N/A",
            "Target Error (ε)":           r.get("epsilon", "N/A"),
            "Failure Prob (δ)":           r.get("delta", "N/A"),
            "Theory/Empirical Ratio":     f"{ratio:.1f}×" if ratio is not None else "N/A",
            "Training Time (s)":          f"{r.get('fit_time_seconds', 0.0):.2f}",
            "Generalization Gap":         f"{r.get('generalization_gap', 0.0):.4f}",
        })

    return pd.DataFrame(rows)
