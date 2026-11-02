# config.py
# ─────────────────────────────────────────────────────────────────
# Global configuration constants for the VC-Dimension Data Budget
# Estimator project.
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
# Primary 8 required models:
PRIMARY_MODELS = [
    "Decision Tree",
    "Random Forest",
    "Support Vector Machine",
    "K-Nearest Neighbors",
    "Logistic Regression",
    "Linear Regression",
    "Naive Bayes",
    "Perceptron",
]

# All available models including legacy variants for full backwards compatibility
AVAILABLE_MODELS = [
    "Decision Tree",
    "Random Forest",
    "Support Vector Machine",
    "K-Nearest Neighbors",
    "Logistic Regression",
    "Linear Regression",
    "Naive Bayes",
    "Perceptron",
    "MLP (Multilayer Perceptron)",
    "AdaBoost",
]

# Classification-compatible models
CLASSIFICATION_MODELS = [
    "Decision Tree",
    "Random Forest",
    "Support Vector Machine",
    "K-Nearest Neighbors",
    "Logistic Regression",
    "Naive Bayes",
    "Perceptron",
    "MLP (Multilayer Perceptron)",
    "AdaBoost",
]

# Regression-compatible models
REGRESSION_MODELS = [
    "Decision Tree",
    "Random Forest",
    "Support Vector Machine",
    "K-Nearest Neighbors",
    "Linear Regression",
    "Perceptron",  # implemented via SGDRegressor
]

# ── Minimum samples for a reliable experiment ────────────────────
MIN_SAMPLES_REQUIRED = 30

# ── UI colour palette (Plotly & Matplotlib) ─────────────────────
COLOR_TRAIN     = "#3B82F6"   # vibrant blue
COLOR_VAL       = "#F97316"   # bright orange
COLOR_THEORY    = "#8B5CF6"   # violet/purple
COLOR_EMPIRICAL = "#10B981"   # emerald green
COLOR_PRED      = "#EF4444"   # coral red
COLOR_ACCENT    = "#06B6D4"   # cyan
COLOR_DARK_BG   = "#0F172A"   # slate 900
COLOR_LIGHT_BG  = "#F8FAFC"   # slate 50

# Supported upload file formats
SUPPORTED_UPLOAD_EXTENSIONS = ["csv", "xlsx", "xls", "json", "txt", "parquet"]
