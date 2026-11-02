# src/data_budget.py
# ─────────────────────────────────────────────────────────────────
# Data Budget Estimator: Combines theoretical PAC bounds,
# empirical learning curve observations, and power-law curve fitting
# to produce actionable data acquisition recommendations and costs.
# ─────────────────────────────────────────────────────────────────

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from src.curve_fitting import fit_power_law, predict_sample_requirement, power_law


def calculate_fit_r2(sample_counts: List[int], val_errors: List[float], params: Tuple[float, float, float]) -> float:
    """Calculate R² goodness-of-fit for the power-law parameters on observed points."""
    if not params or len(sample_counts) < 3:
        return 0.0
    a, b, c = params
    y_true = np.array(val_errors)
    y_pred = power_law(np.array(sample_counts), a, b, c)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0
    r2 = 1.0 - (ss_res / ss_tot)
    return float(max(-1.0, min(1.0, r2)))


def estimate_data_budget(
    theoretical_m: int,
    empirical_achieved: bool,
    empirical_m: Optional[int],
    current_dataset_size: int,
    lc_result: Dict[str, Any],
    target_error: float,
    cost_per_sample: float = 1.0,
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """
    Produce a rigorous, comparative data budget report comparing theoretical PAC requirements
    with observed empirical requirements and fitted power-law projections.

    Parameters
    ----------
    theoretical_m : int
        PAC sample complexity bound.
    empirical_achieved : bool
        Whether the target error was observed directly in the learning curve.
    empirical_m : int or None
        Smallest sample size achieving target error empirically.
    current_dataset_size : int
        Total size of available dataset.
    lc_result : dict
        Results of generate_learning_curve.
    target_error : float
        Target error tolerance epsilon.
    cost_per_sample : float
        Estimated acquisition/annotation cost in dollars per unit.
    confidence_level : float
        Theoretical confidence 1 - delta.

    Returns
    -------
    Dictionary of budget estimates, safety flags, and financial calculations.
    """
    sample_counts = lc_result.get("sample_counts", [])
    val_errors = lc_result.get("val_errors", [])

    # 1. Fit power-law curve to extrapolate if not achieved
    fit_res = fit_power_law(sample_counts, val_errors)
    pred_res = predict_sample_requirement(fit_res, target_error)

    fit_r2 = 0.0
    if fit_res.get("success") and fit_res.get("params"):
        fit_r2 = calculate_fit_r2(sample_counts, val_errors, fit_res["params"])

    predicted_m = pred_res.get("sample_count")
    irreducible_floor = fit_res.get("params", (0, 0, 0))[2] if fit_res.get("params") else 0.0

    # 2. Determine best empirical / projected estimate
    extrapolation_needed = False
    is_extrapolation_safe = True

    if empirical_achieved and empirical_m is not None:
        recommended_empirical_m = empirical_m
        recommendation_source = "Direct Empirical Observation"
        additional_samples = max(0, empirical_m - current_dataset_size)
    elif predicted_m is not None:
        recommended_empirical_m = predicted_m
        recommendation_source = f"Power-Law Fit Projection (R²={fit_r2:.2f})"
        extrapolation_needed = True
        max_observed = max(sample_counts, default=1)
        # Check if extrapolating > 10x beyond observed data
        if predicted_m > max_observed * 10:
            is_extrapolation_safe = False
        additional_samples = max(0, predicted_m - current_dataset_size)
    else:
        recommended_empirical_m = None
        recommendation_source = "Target Not Achievable within Fit Ceiling"
        additional_samples = None

    # Check if target error is below irreducible error floor c
    floor_warning = None
    if fit_res.get("success") and target_error <= irreducible_floor:
        floor_warning = (
            f"Target error ({target_error:.3f}) is at or below the fitted irreducible error floor "
            f"(c = {irreducible_floor:.3f}). The model may never achieve this accuracy regardless of data size."
        )

    # 3. Calculate financial data budget
    theoretical_cost = float(theoretical_m * cost_per_sample)
    empirical_cost = (
        float(recommended_empirical_m * cost_per_sample)
        if recommended_empirical_m is not None
        else None
    )
    additional_cost = (
        float(additional_samples * cost_per_sample)
        if additional_samples is not None
        else None
    )

    # 4. Conservatism ratio: PAC bound vs Empirical
    ratio = None
    if recommended_empirical_m is not None and recommended_empirical_m > 0:
        ratio = round(theoretical_m / recommended_empirical_m, 1)

    # 5. Recommendation Reliability Rating
    if empirical_achieved:
        reliability_rating = "High (Direct Observation)"
        reliability_color = "#10B981"
        reliability_desc = "Target error was achieved empirically on observed validation data."
    elif floor_warning:
        reliability_rating = "Low (Asymptotic Limit)"
        reliability_color = "#EF4444"
        reliability_desc = "Target error is near or below the irreducible error floor."
    elif recommended_empirical_m is not None:
        proj_ratio = recommended_empirical_m / max(1, current_dataset_size)
        if fit_r2 >= 0.85 and proj_ratio <= 5.0:
            reliability_rating = "Moderate (High R² Fit)"
            reliability_color = "#F59E0B"
            reliability_desc = f"Extrapolated with strong power-law fit (R²={fit_r2:.2f}) within {proj_ratio:.1f}x of observed data."
        elif proj_ratio > 5.0:
            reliability_rating = "Low (Far Extrapolation)"
            reliability_color = "#EF4444"
            reliability_desc = f"Predicted requirement is {proj_ratio:.1f}x beyond observed data. Treat as an indicative estimate."
        else:
            reliability_rating = "Moderate (Fair Fit)"
            reliability_color = "#F59E0B"
            reliability_desc = f"Fit quality R²={fit_r2:.2f}. Additional data recommended to verify trajectory."
    else:
        reliability_rating = "Uncertain"
        reliability_color = "#64748B"
        reliability_desc = "Target error could not be reached within search bounds."

    return {
        "theoretical_m": theoretical_m,
        "empirical_m": empirical_m,
        "empirical_achieved": empirical_achieved,
        "predicted_m": predicted_m,
        "recommended_empirical_m": recommended_empirical_m,
        "recommendation_source": recommendation_source,
        "current_dataset_size": current_dataset_size,
        "additional_samples_needed": additional_samples,
        "cost_per_sample": cost_per_sample,
        "theoretical_cost": theoretical_cost,
        "empirical_cost": empirical_cost,
        "additional_cost": additional_cost,
        "conservatism_ratio": ratio,
        "fit_r2": fit_r2,
        "irreducible_floor": irreducible_floor,
        "floor_warning": floor_warning,
        "extrapolation_needed": extrapolation_needed,
        "is_extrapolation_safe": is_extrapolation_safe,
        "reliability_rating": reliability_rating,
        "reliability_color": reliability_color,
        "reliability_desc": reliability_desc,
        "power_law_fit": fit_res,
        "prediction_result": pred_res,
    }


