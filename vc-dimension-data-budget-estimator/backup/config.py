# config.py
# ─────────────────────────────────────────────────────────────────
# Global configuration constants for the VC-Dimension Data Budget
# Estimator project. Adjust defaults here; do not scatter magic
# numbers throughout the codebase.
# ─────────────────────────────────────────────────────────────────

# ── PAC / VC defaults ────────────────────────────────────────────
DEFAULT_EPSILON = 0.10   # target error tolerance (10 %)
DEFAULT_DELTA   = 0.05   # acceptable failure probability (5 %)

# ── Learning-curve fractions ─────────────────────────────────────
TRAIN_FRACTIONS = [0.10, 0.20, 0.30, 0.40, 0.50,
                   0.60, 0.70, 0.80, 0.90, 1.00]

# ── Train / test split ───────────────────────────────────────────
TEST_SIZE      = 0.20   # 20 % held-out test set
RANDOM_STATE   = 42

# ── Cross-validation folds & curve repeats ─────────────────────────
CV_FOLDS = 5
DEFAULT_N_SEEDS = 3   # number of seeds to average for error bars

# ── Power-law curve fitting ──────────────────────────────────────
# error(m) = a * m^(-b) + c
CURVE_FIT_P0     = [1.0, 0.5, 0.01]   # initial guess: [a, b, c]
CURVE_FIT_BOUNDS = ([0, 0, 0], [1e6, 10, 1])  # lower / upper
CURVE_FIT_MAXFEV = 5000                # max evaluations

# Search range for predicted sample size from curve fit
CURVE_SEARCH_MIN = 10
CURVE_SEARCH_MAX = 1_000_000

# ── Built-in dataset options ─────────────────────────────────────
BUILTIN_DATASETS = [
    "Iris",
    "Breast Cancer Wisconsin",
    "Wine Classification",
    "Synthetic Linearly Separable",
    "Synthetic Non-Linearly Separable",
]

# ── Models available ─────────────────────────────────────────────
AVAILABLE_MODELS = ["Perceptron", "MLP (Multilayer Perceptron)", "AdaBoost"]

# ── Minimum samples for a reliable experiment ────────────────────
MIN_SAMPLES_REQUIRED = 30

# ── UI colour palette (used in Matplotlib plots) ─────────────────
COLOR_TRAIN    = "#4C9BE8"   # blue
COLOR_VAL      = "#E8834C"   # orange
COLOR_THEORY   = "#9B59B6"   # purple
COLOR_EMPIRICAL= "#2ECC71"   # green
COLOR_PRED     = "#E74C3C"   # red
