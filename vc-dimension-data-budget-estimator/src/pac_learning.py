# src/pac_learning.py
# ─────────────────────────────────────────────────────────────────
# PAC Learning Sample-Complexity Theory & Calculators.
# Computes distribution-free sample bounds for classification
# (Realizable & Agnostic) and real-valued regression.
# ─────────────────────────────────────────────────────────────────

import math
from typing import Dict, Any, List, Tuple


def calculate_theoretical_samples(epsilon: float, delta: float, vc_dimension: int) -> Dict[str, Any]:
    """
    Calculate the Realizable PAC / VC-dimension theoretical sample-size lower bound
    based on Blumer, Ehrenfeucht, Haussler, and Warmuth (1989).

    Formula:
        m ≥ (1/ε) × [ 4 × log₂(2/δ) + 8 × d × log₂(13/ε) ]
    """
    if not (0 < epsilon < 1):
        raise ValueError(f"epsilon must be strictly between 0 and 1, got {epsilon}.")
    if not (0 < delta < 1):
        raise ValueError(f"delta must be strictly between 0 and 1, got {delta}.")
    if not isinstance(vc_dimension, (int, float)) or vc_dimension <= 0:
        raise ValueError(f"vc_dimension must be a positive number, got {vc_dimension}.")

    d = int(vc_dimension)

    term1 = 4.0 * math.log2(2.0 / delta)
    term2 = 8.0 * d * math.log2(13.0 / epsilon)
    m_raw = (1.0 / epsilon) * (term1 + term2)
    m = math.ceil(m_raw)

    substituted = (
        f"(1 / {epsilon:.4f}) × [4 × log₂(2 / {delta:.4f}) + 8 × {d} × log₂(13 / {epsilon:.4f})]\n"
        f"  = (1 / {epsilon:.4f}) × [{term1:.4f} + {term2:.4f}]\n"
        f"  = (1 / {epsilon:.4f}) × {term1 + term2:.4f}\n"
        f"  = {m_raw:,.2f}  →  ⌈m⌉ = {m:,} samples"
    )

    return {
        "m": m,
        "m_raw": m_raw,
        "term1": term1,
        "term2": term2,
        "epsilon": epsilon,
        "delta": delta,
        "confidence": 1.0 - delta,
        "vc_dimension": d,
        "setting": "realizable",
        "setting_label": "Realizable Setting (Zero-Noise / Separable)",
        "formula": "m ≥ (1/ε) × [ 4 × log₂(2/δ) + 8 × d × log₂(13/ε) ]",
        "latex_formula": r"m \ge \frac{1}{\epsilon} \left[ 4 \log_2\left(\frac{2}{\delta}\right) + 8 d \log_2\left(\frac{13}{\epsilon}\right) \right]",
        "substituted": substituted,
        "assumptions": [
            "Realizability: The true target concept c* is contained in hypothesis class H (zero training error achievable).",
            "I.I.D. Sampling: All training and test instances are drawn independently from identical distribution D.",
            "Distribution-Free: Guarantee holds for ANY underlying probability distribution D.",
            "Deterministic binary ground-truth labels (zero label noise).",
        ],
        "explanation": (
            f"Under the realizable assumption, any hypothesis with true error > {epsilon} that is consistent with the "
            f"training data will be eliminated with probability at least 1 - {delta:.2f} ({(1-delta)*100:.0f}% confidence) "
            f"once at least {m:,} training samples are observed. The sample size scales as O((d/ε)·log(1/ε) + (1/ε)·log(1/δ))."
        ),
        "reference": (
            "Blumer, A., Ehrenfeucht, A., Haussler, D., & Warmuth, M. K. (1989). "
            "Learnability and the Vapnik-Chervonenkis dimension. Journal of the ACM, 36(4), 929–965."
        ),
        "note": "Worst-case distribution-free upper bound. Practical requirements are typically far smaller.",
    }


def calculate_agnostic_samples(epsilon: float, delta: float, vc_dimension: int) -> Dict[str, Any]:
    """
    Calculate the Agnostic (Non-Realizable) PAC sample-complexity bound
    based on Anthony & Bartlett (1999) and Vapnik (1998).

    Formula:
        m ≥ (8 / ε²) × [ 2 × d × ln(8e / ε) + ln(4 / δ) ]
    """
    if not (0 < epsilon < 1):
        raise ValueError(f"epsilon must be strictly between 0 and 1, got {epsilon}.")
    if not (0 < delta < 1):
        raise ValueError(f"delta must be strictly between 0 and 1, got {delta}.")
    if not isinstance(vc_dimension, (int, float)) or vc_dimension <= 0:
        raise ValueError(f"vc_dimension must be a positive number, got {vc_dimension}.")

    d = int(vc_dimension)
    e_const = math.e

    term1 = 2.0 * d * math.log((8.0 * e_const) / epsilon)
    term2 = math.log(4.0 / delta)
    scale = 8.0 / (epsilon ** 2)
    m_raw = scale * (term1 + term2)
    m = math.ceil(m_raw)

    substituted = (
        f"(8 / {epsilon:.4f}²) × [2 × {d} × ln(8e / {epsilon:.4f}) + ln(4 / {delta:.4f})]\n"
        f"  = {scale:,.2f} × [{term1:.4f} + {term2:.4f}]\n"
        f"  = {scale:,.2f} × {term1 + term2:.4f}\n"
        f"  = {m_raw:,.2f}  →  ⌈m⌉ = {m:,} samples"
    )

    return {
        "m": m,
        "m_raw": m_raw,
        "term1": term1,
        "term2": term2,
        "epsilon": epsilon,
        "delta": delta,
        "confidence": 1.0 - delta,
        "vc_dimension": d,
        "setting": "agnostic",
        "setting_label": "Agnostic Setting (Noisy Labels / Model Misspecification)",
        "formula": "m ≥ (8 / ε²) × [ 2 × d × ln(8e / ε) + ln(4 / δ) ]",
        "latex_formula": r"m \ge \frac{8}{\epsilon^2} \left[ 2 d \ln\left(\frac{8e}{\epsilon}\right) + \ln\left(\frac{4}{\delta}\right) \right]",
        "substituted": substituted,
        "assumptions": [
            "No Realizability Required: The optimal hypothesis in H may have non-zero Bayes risk or misclassification.",
            "Arbitrary noise in target labels is tolerated.",
            "I.I.D. Sampling from an arbitrary joint probability distribution P(X, Y).",
            "Uniform convergence of empirical risk to true risk over all h in H.",
        ],
        "explanation": (
            f"In the agnostic setting, we cannot assume that any hypothesis achieves zero error. To guarantee that the "
            f"empirical risk minimizer (ERM) is within {epsilon} excess error of the best-in-class hypothesis with "
            f"confidence {1-delta:.2f}, Hoeffding-style concentration requires 1/ε² scaling rather than 1/ε. "
            f"This produces a conservative requirement of {m:,} samples."
        ),
        "reference": (
            "Anthony, M., & Bartlett, P. L. (1999). Neural Network Learning: "
            "Theoretical Foundations. Cambridge University Press."
        ),
        "note": "Agnostic bounds are larger than realizable bounds due to the quadratic 1/ε² dependence.",
    }


def calculate_regression_samples(
    epsilon: float,
    delta: float,
    pseudo_dim: int,
    target_scale_M: float = 1.0
) -> Dict[str, Any]:
    """
    Calculate PAC sample complexity for real-valued regression based on Pollard's
    pseudo-dimension bound (Pollard 1984; Alon et al. 1997).

    Formula (bounded target in [0, M]):
        m ≥ (128 · M² / ε²) × [ 2 · d_P · ln(16 e M / ε) + ln(8 / δ) ]
    """
    if not (0 < epsilon < 100):
        raise ValueError("epsilon must be positive.")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0, 1).")
    if pseudo_dim <= 0:
        raise ValueError("pseudo_dim must be positive.")

    d = int(pseudo_dim)
    M = max(0.1, float(target_scale_M))
    e_const = math.e

    scale = (128.0 * (M ** 2)) / (epsilon ** 2)
    inner_log = max(1.01, (16.0 * e_const * M) / epsilon)
    term1 = 2.0 * d * math.log(inner_log)
    term2 = math.log(8.0 / delta)
    m_raw = scale * (term1 + term2)
    m = math.ceil(m_raw)

    substituted = (
        f"(128 × {M:.2f}² / {epsilon:.4f}²) × [2 × {d} × ln(16e × {M:.2f} / {epsilon:.4f}) + ln(8 / {delta:.4f})]\n"
        f"  = {scale:,.2f} × [{term1:.4f} + {term2:.4f}]\n"
        f"  = {m_raw:,.2f}  →  ⌈m⌉ = {m:,} samples"
    )

    return {
        "m": m,
        "m_raw": m_raw,
        "term1": term1,
        "term2": term2,
        "epsilon": epsilon,
        "delta": delta,
        "confidence": 1.0 - delta,
        "vc_dimension": d,
        "target_scale_M": M,
        "setting": "regression",
        "setting_label": "Regression Setting (Pollard's Pseudo-Dimension)",
        "formula": "m ≥ (128 · M² / ε²) × [ 2 · d_P · ln(16 e M / ε) + ln(8 / δ) ]",
        "latex_formula": r"m \ge \frac{128 M^2}{\epsilon^2} \left[ 2 d_P \ln\left(\frac{16 e M}{\epsilon}\right) + \ln\left(\frac{8}{\delta}\right) \right]",
        "substituted": substituted,
        "assumptions": [
            f"Bounded target domain: Outputs y and predictions f(x) reside in interval [-{M:.2f}, {M:.2f}].",
            "Loss function is Lipschitz continuous (e.g., bounded squared error or absolute error).",
            "Hypothesis class has Pollard pseudo-dimension d_P.",
        ],
        "explanation": (
            f"For continuous regression, Pollard's generalization of VC dimension bounds uniform convergence "
            f"of continuous empirical loss. To guarantee that expected loss is within {epsilon:.3f} of optimal "
            f"with confidence {1-delta:.2f}, the bound requires {m:,} samples."
        ),
        "reference": (
            "Pollard, D. (1984). Convergence of Stochastic Processes. Springer. "
            "Alon, N. et al. (1997). Scale-sensitive Dimensions and Uniform Convergence. JACM."
        ),
        "note": "Continuous bounded regression PAC guarantee.",
    }


def get_pac_bounds(epsilon: float, delta: float, vc_dimension: int) -> Dict[str, Any]:
    """Compute both Realizable and Agnostic bounds for classification."""
    return {
        "realizable": calculate_theoretical_samples(epsilon, delta, vc_dimension),
        "agnostic": calculate_agnostic_samples(epsilon, delta, vc_dimension),
    }


# ── Sensitivity Analysis Generators ──────────────────────────────

def generate_epsilon_sensitivity_curve(
    delta: float,
    vc_dim: int,
    eps_min: float = 0.02,
    eps_max: float = 0.30,
    steps: int = 25
) -> Dict[str, List[float]]:
    """Generate sample requirements across varying epsilon values."""
    epsilons = [eps_min + (eps_max - eps_min) * (i / (steps - 1)) for i in range(steps)]
    realizable_m = [calculate_theoretical_samples(eps, delta, vc_dim)["m"] for eps in epsilons]
    agnostic_m = [calculate_agnostic_samples(eps, delta, vc_dim)["m"] for eps in epsilons]
    return {
        "epsilons": epsilons,
        "realizable_m": realizable_m,
        "agnostic_m": agnostic_m,
    }


def generate_delta_sensitivity_curve(
    epsilon: float,
    vc_dim: int,
    delta_min: float = 0.01,
    delta_max: float = 0.30,
    steps: int = 25
) -> Dict[str, List[float]]:
    """Generate sample requirements across varying failure probability delta values."""
    deltas = [delta_min + (delta_max - delta_min) * (i / (steps - 1)) for i in range(steps)]
    confidences = [(1.0 - d) * 100 for d in deltas]
    realizable_m = [calculate_theoretical_samples(epsilon, d, vc_dim)["m"] for d in deltas]
    agnostic_m = [calculate_agnostic_samples(epsilon, d, vc_dim)["m"] for d in deltas]
    return {
        "deltas": deltas,
        "confidences": confidences,
        "realizable_m": realizable_m,
        "agnostic_m": agnostic_m,
    }


def generate_vc_sensitivity_curve(
    epsilon: float,
    delta: float,
    vc_min: int = 1,
    vc_max: int = 50,
    steps: int = 25
) -> Dict[str, List[float]]:
    """Generate sample requirements across varying VC dimensions."""
    step_size = max(1, (vc_max - vc_min) // (steps - 1))
    vc_values = list(range(vc_min, vc_max + 1, step_size))
    realizable_m = [calculate_theoretical_samples(epsilon, delta, v)["m"] for v in vc_values]
    agnostic_m = [calculate_agnostic_samples(epsilon, delta, v)["m"] for v in vc_values]
    return {
        "vc_values": vc_values,
        "realizable_m": realizable_m,
        "agnostic_m": agnostic_m,
    }

