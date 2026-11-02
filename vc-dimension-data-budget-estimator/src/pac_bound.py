# src/pac_bound.py
# ─────────────────────────────────────────────────────────────────
# Backwards compatibility proxy for pac_learning.py
# ─────────────────────────────────────────────────────────────────

from src.pac_learning import (
    calculate_theoretical_samples,
    calculate_agnostic_samples,
    calculate_regression_samples,
    get_pac_bounds,
    generate_epsilon_sensitivity_curve,
    generate_delta_sensitivity_curve,
    generate_vc_sensitivity_curve,
)

__all__ = [
    "calculate_theoretical_samples",
    "calculate_agnostic_samples",
    "calculate_regression_samples",
    "get_pac_bounds",
    "generate_epsilon_sensitivity_curve",
    "generate_delta_sensitivity_curve",
    "generate_vc_sensitivity_curve",
]
