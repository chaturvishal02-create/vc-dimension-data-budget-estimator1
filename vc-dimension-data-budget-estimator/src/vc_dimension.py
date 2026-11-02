# src/vc_dimension.py
# ─────────────────────────────────────────────────────────────────
# Comprehensive VC-Dimension & Statistical Capacity Estimator.
#
# References
# ──────────
# • Vapnik, V. N. (1998). Statistical Learning Theory. Wiley.
# • Blumer, A. et al. (1989). Learnability and the Vapnik-Chervonenkis
#   dimension. JACM, 36(4), 929-965.
# • Anthony, M., & Bartlett, P. L. (1999). Neural Network Learning:
#   Theoretical Foundations. Cambridge University Press.
# • Bartlett, P. L., & Shawe-Taylor, J. (1999). Generalization Performance
#   of Support Vector Machines and Other Pattern Classifiers.
# • Devroye, L., Gyorfi, L., & Lugosi, G. (1996). A Probabilistic Theory
#   of Pattern Recognition. Springer.
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations
import math
from typing import Dict, Any, List, Optional


# ── 1. Linear Halfspaces / Perceptron ─────────────────────────────

def vc_perceptron(n_features: int, bias: bool = True) -> Dict[str, Any]:
    """
    Exact VC dimension of linear threshold functions (Perceptron) in R^n.
    d = n + 1 (with bias) or d = n (homogenous / through origin).
    """
    if n_features < 1:
        raise ValueError("n_features must be ≥ 1")
    d = n_features + 1 if bias else n_features
    bias_str = "+ 1 (intercept)" if bias else "(no intercept)"
    return {
        "vc_dim": d,
        "type": "exact",
        "label": "Exact (Radon's Theorem / Linear Halfspaces in General Position)",
        "confidence": "High (Exact Mathematical Theorem)",
        "family": "Linear Threshold Functions",
        "formula": "d = n + 1" if bias else "d = n",
        "substituted": f"d = {n_features} {bias_str} = {d}",
        "explanation": (
            f"By Radon's Theorem, any set of n + 2 points in R^{n_features} can be partitioned into "
            "two subsets whose convex hulls intersect, making them unshatterable by a hyperplane. "
            f"Conversely, any n + 1 points in general position can be shattered. Thus, VC dimension is exactly {d}."
        ),
        "assumptions": [
            "Data points are in general position (no n+1 points lie on an (n-1)-dimensional affine subspace).",
            "Homogeneous or affine linear decision boundary.",
            "Binary classification target.",
        ],
        "note": "This is an exact, distribution-free theoretical result (Vapnik, 1998; Blumer et al., 1989).",
    }


# ── 2. Logistic Regression ────────────────────────────────────────

def vc_logistic_regression(n_features: int, bias: bool = True) -> Dict[str, Any]:
    """
    VC dimension of Logistic Regression for binary classification.
    The decision rule sign(w^T x + b) is identical to linear halfspaces, so d = n + 1.
    """
    if n_features < 1:
        raise ValueError("n_features must be ≥ 1")
    d = n_features + 1 if bias else n_features
    return {
        "vc_dim": d,
        "type": "exact",
        "label": "Exact (Linear Separating Hyperplane in R^n)",
        "confidence": "High (Exact Mathematical Theorem)",
        "family": "Generalized Linear Models (Logit link)",
        "formula": "d = n + 1",
        "substituted": f"d = {n_features} + 1 = {d}",
        "explanation": (
            "Although logistic regression outputs calibrated posterior probabilities via the sigmoid function "
            "sigma(z) = 1 / (1 + e^-z), classification at threshold p = 0.5 corresponds to the linear sign test "
            "w^T x + b >= 0. The hypothesis class of decision regions is isomorphic to linear halfspaces."
        ),
        "assumptions": [
            "Standard classification decision threshold at p = 0.5.",
            "Linear log-odds (no non-linear feature basis expansions without adding dimensions).",
            "Input features in general position.",
        ],
        "note": "Exact capacity for binary decision boundaries with linear log-odds.",
    }


# ── 3. Linear Regression (Pollard's Pseudo-Dimension) ─────────────

def vc_linear_regression(n_features: int, bias: bool = True) -> Dict[str, Any]:
    """
    For real-valued linear regression, the standard generalization of VC dimension
    is Pollard's pseudo-dimension (subgraph capacity), which equals n + 1.
    """
    if n_features < 1:
        raise ValueError("n_features must be ≥ 1")
    d = n_features + 1 if bias else n_features
    return {
        "vc_dim": d,
        "type": "exact",
        "label": "Exact (Pollard's Pseudo-Dimension for Hyperplanes)",
        "confidence": "High (Exact Pseudo-Dimension)",
        "family": "Linear Regression / Real-Valued Hyperplanes",
        "formula": "dim_P(H) = n + 1",
        "substituted": f"dim_P(H) = {n_features} + 1 = {d}",
        "explanation": (
            "For real-valued function classes, Pollard (1984) defined pseudo-dimension as the VC dimension "
            "of the subgraph class {(x, y): f(x) >= y}. For linear functions f(x) = w^T x + b, the subgraphs "
            f"are halfspaces in R^{n_features + 1}, which has VC dimension exactly n + 1."
        ),
        "assumptions": [
            "Target is continuous real-valued scalar.",
            "Ordinary least-squares or ridge/lasso linear hypothesis.",
            "Loss function is bounded or sub-Gaussian for sample bounds.",
        ],
        "note": "Used in Pollard's PAC sample complexity bounds for regression.",
    }


# ── 4. Polynomial Classifiers ─────────────────────────────────────

def vc_polynomial_classifier(n_features: int, degree: int = 2) -> Dict[str, Any]:
    """
    VC dimension of polynomial threshold functions of degree k.
    Via the Veronese embedding into dimension N = binom(n + k, k) - 1,
    the VC dimension is binom(n + k, k).
    """
    if n_features < 1:
        raise ValueError("n_features must be ≥ 1")
    if degree < 1:
        raise ValueError("degree must be ≥ 1")
    d = math.comb(n_features + degree, degree)
    return {
        "vc_dim": d,
        "type": "exact",
        "label": f"Exact (Polynomial Halfspaces via Veronese Embedding, degree={degree})",
        "confidence": "High (Exact Linear Embedding)",
        "family": "Polynomial Threshold Functions",
        "formula": "d = C(n + k, k)",
        "substituted": f"C({n_features} + {degree}, {degree}) = {d}",
        "explanation": (
            f"A polynomial of degree {degree} in {n_features} variables is a linear combination of all monomials "
            f"up to degree {degree}. The number of such monomials is the binomial coefficient C(n + k, k). "
            "Since linear halfspaces in dimension D have VC dimension D, this transformed class has exact VC dimension D."
        ),
        "assumptions": [
            "Full polynomial expansion including all interaction terms up to degree k.",
            "General position of monomial feature vectors.",
        ],
        "note": "Exact capacity for polynomial decision boundaries.",
    }


# ── 5. Decision Trees ─────────────────────────────────────────────

def vc_decision_tree(
    n_features: int,
    max_depth: Optional[int] = 3,
    max_leaf_nodes: Optional[int] = None
) -> Dict[str, Any]:
    """
    VC dimension bounds for axis-aligned decision trees.
    For a tree of depth D in n dimensions with axis-aligned cuts:
    Upper bound: d <= 2 (2^D - 1) log2(e * n) + 1.
    If depth is unlimited, VC dimension can be unbounded (equal to sample size).
    """
    if n_features < 1:
        raise ValueError("n_features must be ≥ 1")

    depth = max_depth if max_depth is not None else 4
    n_leaves = 2 ** depth if max_leaf_nodes is None else min(2 ** depth, max_leaf_nodes)
    n_internal = n_leaves - 1

    # Theoretical upper bound for axis-aligned splits (Devroye, Gyorfi, Lugosi, 1996)
    log_term = math.log2(math.e * n_features) if n_features > 1 else 1.5
    raw_d = int(math.ceil(2 * n_internal * log_term + 1))
    d = max(raw_d, n_features + 1)

    return {
        "vc_dim": d,
        "type": "bound",
        "label": f"Known Upper Bound (Axis-aligned tree, depth={depth}, leaves<={n_leaves})",
        "confidence": "Medium (Upper Bound on Restricted Tree)",
        "family": "Decision Trees (Axis-Aligned Splits)",
        "formula": "d <= 2 * (2^D - 1) * log2(e * n) + 1",
        "substituted": f"Depth={depth}, Splits={n_internal}, n={n_features} → d <= {d}",
        "explanation": (
            f"An axis-aligned binary decision tree of depth {depth} partitions R^{n_features} into at most {n_leaves} "
            "hyper-rectangles using orthogonal threshold cuts. The number of possible tree topologies and split sequences "
            f"yields an information-theoretic capacity bound of O(2^D * log(n)). Without depth constraints, an arbitrary "
            "decision tree can shatter any finite dataset, yielding an infinite theoretical VC dimension."
        ),
        "assumptions": [
            f"Strict maximum depth restriction D = {depth}.",
            "Axis-aligned orthogonal splits (x_i <= theta).",
            "Discrete leaf assignments (binary 0/1 labels).",
        ],
        "note": "Upper bound derived from Devroye et al. (1996). Depth pruning is required for finite VC dimension.",
    }


# ── 6. Random Forest ──────────────────────────────────────────────

def vc_random_forest(
    n_features: int,
    n_estimators: int = 50,
    max_depth: Optional[int] = 4
) -> Dict[str, Any]:
    """
    VC dimension bound for an ensemble of voting decision trees.
    For an ensemble of T base classifiers with VC dimension d_base,
    Schapire & Singer / Bartlett established:
        d <= O(T * d_base * log(T * d_base))
    """
    if n_features < 1:
        raise ValueError("n_features must be ≥ 1")
    if n_estimators < 1:
        raise ValueError("n_estimators must be ≥ 1")

    base_tree_info = vc_decision_tree(n_features, max_depth=max_depth)
    d_base = base_tree_info["vc_dim"]
    product = n_estimators * d_base
    log_val = math.log2(max(product, 2))
    d = int(math.ceil(product * log_val))

    return {
        "vc_dim": d,
        "type": "bound",
        "label": f"Theoretical Ensemble Upper Bound (T={n_estimators} trees, max_depth={max_depth})",
        "confidence": "Medium (Proven Ensemble Upper Bound)",
        "family": "Ensemble Methods (Bagged Decision Trees)",
        "formula": "d <= T * d_base * log2(T * d_base)",
        "substituted": f"T={n_estimators}, d_base={d_base} → d <= {d:,}",
        "explanation": (
            f"A majority-vote ensemble of {n_estimators} decision trees combines individual tree hypotheses. "
            f"Since each base tree has capacity bound d_base = {d_base}, the VC dimension of majority votes over T trees "
            "is upper bounded by T * d_base * log2(T * d_base). Random subspace projection further constrains actual empirical capacity."
        ),
        "assumptions": [
            f"Ensemble consists of T = {n_estimators} voting trees.",
            f"Each individual tree has depth capped at {max_depth}.",
            "Majority voting decision rule.",
        ],
        "note": "Worst-case upper bound (Schapire & Singer, 1999). Empirical needs are significantly smaller in practice.",
    }


# ── 7. Support Vector Machines (Linear & RBF) ─────────────────────

def vc_svm(
    n_features: int,
    kernel: str = "linear",
    C: float = 1.0,
    margin_gamma: Optional[float] = None,
    radius_R: float = 1.0
) -> Dict[str, Any]:
    """
    VC dimension of Support Vector Machines.
    - Linear SVM: Exactly n + 1.
    - Non-linear RBF SVM: Structurally infinite without margin.
      With margin gamma in a bounding ball of radius R:
      fat-shattering dimension / margin bound <= min(n, ceil(R^2 / gamma^2)).
    """
    if n_features < 1:
        raise ValueError("n_features must be ≥ 1")

    k_clean = str(kernel).lower()
    if k_clean == "linear":
        d = n_features + 1
        return {
            "vc_dim": d,
            "type": "exact",
            "label": "Exact (Linear Separating Hyperplane)",
            "confidence": "High (Exact Mathematical Value)",
            "family": "Maximum-Margin Linear Classifiers",
            "formula": "d = n + 1",
            "substituted": f"d = {n_features} + 1 = {d}",
            "explanation": (
                f"A linear SVM maximizes the geometric margin in the original input space R^{n_features}. "
                "The geometric boundary is a hyperplane, giving an exact VC dimension of n + 1."
            ),
            "assumptions": [
                "Linear kernel (no non-linear feature maps).",
                "Data in general position in input space.",
            ],
            "note": "Exact capacity for linear SVM (Vapnik, 1998).",
        }
    else:
        # RBF / Kernel SVM: Margin-based fat-shattering dimension (Vapnik, 1998; Bartlett & Shawe-Taylor)
        gamma_eff = margin_gamma if margin_gamma is not None else (1.0 / math.sqrt(n_features))
        fat_shat = int(math.ceil((radius_R ** 2) / (gamma_eff ** 2)))
        # Guard against zero or extreme negative
        fat_shat = max(fat_shat, n_features + 1)
        return {
            "vc_dim": fat_shat,
            "type": "heuristic",
            "label": f"Fat-Shattering Margin Bound (RBF Kernel, R={radius_R:.1f}, gamma~{gamma_eff:.2f})",
            "confidence": "Medium (Margin-Dependent Capacity)",
            "family": "Kernel Methods / Reproducing Kernel Hilbert Spaces",
            "formula": "d_fat <= min(N, ceil(R^2 / gamma^2))",
            "substituted": f"R={radius_R:.1f}, gamma={gamma_eff:.2f} → d_fat <= {fat_shat}",
            "explanation": (
                "An RBF kernel maps data into an infinite-dimensional Hilbert space, meaning the unconstrained VC dimension "
                "is theoretically infinite (it can shatter any finite dataset for sufficiently small bandwidth sigma). "
                f"However, the effective capacity is controlled by the margin gamma: fat-shattering dimension is bounded by (R/gamma)^2 = {fat_shat}."
            ),
            "assumptions": [
                f"Points reside in a bounding sphere of radius R = {radius_R:.1f} in feature space.",
                f"Effective geometric margin gamma >= {gamma_eff:.2f} achieved by soft-margin penalty C={C}.",
            ],
            "note": "Unconstrained RBF has d = infinity. We report the distribution-dependent margin-bounded capacity (Bartlett & Shawe-Taylor, 1999).",
        }


# ── 8. K-Nearest Neighbors ────────────────────────────────────────

def vc_knn(n_features: int, k: int = 1) -> Dict[str, Any]:
    """
    Capacity analysis for K-Nearest Neighbors.
    1-NN has infinite VC dimension because any finite dataset can be memorized/shattered
    if point locations can be chosen arbitrarily.
    With metric space covering numbers, the effective metric dimension is approximately n.
    """
    if n_features < 1:
        raise ValueError("n_features must be ≥ 1")
    if k < 1:
        raise ValueError("k must be ≥ 1")

    # Effective local degrees of freedom heuristic
    eff_capacity = max(n_features + 1, int(math.ceil(2 ** (min(n_features, 10)))))
    return {
        "vc_dim": eff_capacity,
        "type": "heuristic",
        "label": f"Effective Metric Capacity (k={k}, unconstrained VC is infinite)",
        "confidence": "Low (Non-Parametric / Metric Bound)",
        "family": "Non-Parametric Metric Classifiers",
        "formula": "d_exact = ∞ (effective local metric capacity ~ O(2^n))",
        "substituted": f"n={n_features}, k={k} → d_eff ≈ {eff_capacity}",
        "explanation": (
            f"Because 1-NN constructs Voronoi cells around each stored sample, it can shatter an arbitrarily large set "
            "of points by placing points appropriately. Hence, the standard distribution-free VC dimension of k-NN is INFINITE. "
            f"In practice, under metric regularity and compact domains, the effective metric capacity scales exponentially with dimension (d_eff ~ {eff_capacity})."
        ),
        "assumptions": [
            "Euclidean metric space with compact support.",
            "Smooth Bayes decision boundary.",
            "Local Voronoi cell capacity heuristic.",
        ],
        "note": "Strict theoretical VC dimension is infinite (Devroye, Gyorfi, Lugosi, 1996). We display the metric-entropy effective proxy.",
    }


# ── 9. Naive Bayes ────────────────────────────────────────────────

def vc_naive_bayes(n_features: int, variant: str = "gaussian") -> Dict[str, Any]:
    """
    VC dimension of Gaussian Naive Bayes.
    When variances are shared across classes, the log-likelihood ratio is strictly linear,
    yielding VC dimension = n + 1.
    When variances are distinct, the decision boundary is a quadratic quadric surface,
    yielding VC dimension = n(n + 3)/2 + 1.
    """
    if n_features < 1:
        raise ValueError("n_features must be ≥ 1")

    d_linear = n_features + 1
    d_quad = int(n_features * (n_features + 3) / 2 + 1)

    return {
        "vc_dim": d_linear,
        "type": "exact",
        "label": "Exact under Shared Variance (d = n + 1) / Quadratic if Separate Variance",
        "confidence": "High (Linear Boundary under Conditional Independence)",
        "family": "Probabilistic Generative Classifiers",
        "formula": "d = n + 1 (shared covariance)",
        "substituted": f"d = {n_features} + 1 = {d_linear} (or {d_quad} with separate variances)",
        "explanation": (
            f"Gaussian Naive Bayes applies Bayes' rule under feature conditional independence. "
            f"When feature variances are equal across classes, the log-posterior odds log(P(C1|x)/P(C0|x)) is a linear "
            f"function of x, producing an exact VC dimension of n + 1 = {d_linear}. If variances differ per class, "
            f"the boundary is an axis-aligned quadric with capacity up to {d_quad}."
        ),
        "assumptions": [
            "Conditional independence of features given class label.",
            "Normal (Gaussian) feature likelihoods.",
            "Shared feature variance across classes (linear boundary).",
        ],
        "note": "Exact for linear log-odds decision regions (Duda, Hart, Stork, 2001).",
    }


# ── 10. Finite Hypothesis Classes ─────────────────────────────────

def vc_finite_class(cardinality: int) -> Dict[str, Any]:
    """
    For any finite hypothesis class H, VC(H) <= log2(|H|).
    """
    if cardinality < 1:
        raise ValueError("cardinality must be ≥ 1")
    d = int(math.floor(math.log2(cardinality)))
    return {
        "vc_dim": d,
        "type": "bound",
        "label": f"Exact Upper Bound for Finite Class (|H| = {cardinality:,})",
        "confidence": "High (Fundamental Theorem of VC Theory)",
        "family": "Finite Hypothesis Spaces",
        "formula": "d <= floor(log2(|H|))",
        "substituted": f"|H| = {cardinality:,} → d <= {d}",
        "explanation": (
            f"A hypothesis class H of size |H| can label a set of m points in at most |H| distinct ways. "
            f"To shatter m points, H must produce all 2^m possible labellings. Thus 2^m <= |H|, which implies "
            f"m <= log2(|H|). Hence VC(H) <= floor(log2(|H|)) = {d}."
        ),
        "assumptions": ["Hypothesis class H has strictly finite cardinality."],
        "note": "Universal theorem for finite hypothesis classes (Blumer et al., 1989).",
    }


# ── 11. Legacy MLP & AdaBoost (Backward Compatibility) ───────────

def vc_mlp(n_features: int, hidden_layer_sizes: tuple = (100,)) -> dict:
    """Backward compatible MLP VC estimator."""
    if n_features < 1:
        raise ValueError("n_features must be ≥ 1")
    layer_sizes = [n_features] + list(hidden_layer_sizes) + [1]
    n_params = sum(
        layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1]
        for i in range(len(layer_sizes) - 1)
    )
    d = int(n_params)
    arch_str = " → ".join(str(s) for s in layer_sizes)
    return {
        "vc_dim": d,
        "type": "heuristic",
        "label": "Approximate / Illustrative (≈ # trainable parameters)",
        "confidence": "Medium (Parameter Proxy)",
        "family": "Feedforward Neural Networks",
        "formula": "d ≈ Σ (units_i × units_{i+1} + units_{i+1})",
        "substituted": f"Architecture: {arch_str} → d ≈ {d:,}",
        "explanation": "Illustrative proxy based on total trainable parameters (Anthony & Bartlett, 1999).",
        "assumptions": ["Fully connected architecture with ReLU / threshold activations."],
        "note": "Serves as an illustrative parameter count proxy.",
    }


def vc_adaboost(n_features: int, n_estimators: int = 50, base_vc: int | None = None) -> dict:
    """Backward compatible AdaBoost VC estimator."""
    if n_features < 1:
        raise ValueError("n_features must be ≥ 1")
    if n_estimators < 1:
        raise ValueError("n_estimators must be ≥ 1")
    d_base = base_vc if base_vc is not None else 2 * n_features
    product = n_estimators * d_base
    d = int(product * math.log(max(product, 2)))
    return {
        "vc_dim": d,
        "type": "bound",
        "label": "Approximate / Illustrative (ensemble VC bound)",
        "confidence": "Medium (Ensemble Bound)",
        "family": "Boosting Ensembles",
        "formula": "d ≈ T × d_base × log(T × d_base)",
        "substituted": f"T={n_estimators}, d_base={d_base} → d ≈ {d:,}",
        "explanation": "Illustrative upper-bound estimate for AdaBoost with decision stumps.",
        "assumptions": ["Decision stump base learners", "Majority voting combination"],
        "note": "Upper bound (Schapire & Singer, 1999).",
    }


# ── Unified Dispatcher ───────────────────────────────────────────

def estimate_vc_dimension(model_name: str, n_features: int, **kwargs) -> Dict[str, Any]:
    """
    Unified dispatcher estimating VC dimension for any of the 8 required models
    plus legacy models and finite classes.
    """
    m = model_name.strip()

    if m in ("Perceptron", "Linear Perceptron"):
        bias = kwargs.get("bias", True)
        return vc_perceptron(n_features, bias=bias)

    elif m in ("Logistic Regression", "Logit"):
        bias = kwargs.get("bias", True)
        return vc_logistic_regression(n_features, bias=bias)

    elif m in ("Linear Regression", "Ridge", "Lasso", "OLS"):
        bias = kwargs.get("bias", True)
        return vc_linear_regression(n_features, bias=bias)

    elif m in ("Polynomial Classifier", "Polynomial Regression"):
        degree = kwargs.get("degree", 2)
        return vc_polynomial_classifier(n_features, degree=degree)

    elif m in ("Decision Tree", "CART", "DecisionTreeClassifier", "DecisionTreeRegressor"):
        max_depth = kwargs.get("max_depth", 3)
        return vc_decision_tree(n_features, max_depth=max_depth)

    elif m in ("Random Forest", "RandomForestClassifier", "RandomForestRegressor"):
        n_estimators = kwargs.get("n_estimators", 50)
        max_depth = kwargs.get("max_depth", 4)
        return vc_random_forest(n_features, n_estimators=n_estimators, max_depth=max_depth)

    elif m in ("Support Vector Machine", "SVM", "SVC", "SVR"):
        kernel = kwargs.get("kernel", "linear")
        C = kwargs.get("C", 1.0)
        return vc_svm(n_features, kernel=kernel, C=C)

    elif m in ("K-Nearest Neighbors", "KNN", "KNeighborsClassifier", "KNeighborsRegressor"):
        k = kwargs.get("k", kwargs.get("n_neighbors", 3))
        return vc_knn(n_features, k=k)

    elif m in ("Naive Bayes", "GaussianNB"):
        variant = kwargs.get("variant", "gaussian")
        return vc_naive_bayes(n_features, variant=variant)

    elif m == "MLP (Multilayer Perceptron)":
        hidden = kwargs.get("hidden_layer_sizes", (100,))
        return vc_mlp(n_features, hidden_layer_sizes=hidden)

    elif m == "AdaBoost":
        n_est = kwargs.get("n_estimators", 50)
        return vc_adaboost(n_features, n_estimators=n_est)

    elif m == "Finite Class":
        card = kwargs.get("cardinality", 1000)
        return vc_finite_class(card)

    else:
        raise ValueError(f"Unknown model: '{model_name}'")
