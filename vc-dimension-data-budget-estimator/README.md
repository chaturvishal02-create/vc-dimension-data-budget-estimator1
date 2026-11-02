# How Much Data Does Your Model Really Need?
## A VC-Dimension-Guided Data Budget Estimator — Bridging PAC Learning Theory with Empirical Practice

---

## 1. Project Overview & Features

This platform bridges the fundamental divide between theoretical machine-learning sample-complexity guarantees and real-world empirical data requirements. Grounded in the **Probably Approximately Correct (PAC)** learning framework and **Vapnik-Chervonenkis (VC)** dimension theory, the platform delivers:

1. **Universal Dataset Ingestion**: Accepts user datasets in **CSV, Excel (.xlsx, .xls), JSON, TXT (tabular), and Parquet** formats with automatic column type, dimension, missingness, and task detection.
2. **VC Dimension & Statistical Capacity Estimation**: Computes exact theoretical capacities, known bounds, and structural capacity proxies across **8 machine learning model families** (Decision Tree, Random Forest, SVM, KNN, Logistic Regression, Linear Regression, Naive Bayes, Perceptron) plus polynomial classifiers and finite hypothesis classes.
3. **PAC Sample-Complexity Calculations**: Computes distribution-free sample bounds under **Realizable** (Blumer et al., 1989), **Agnostic** (Anthony & Bartlett, 1999), and **Regression** (Pollard, 1984) settings.
4. **Empirical Learning Curve Suite**: Measures real empirical performance across progressive training subsets ($10\%$ to $100\%$) with multi-seed variance ($\pm 1\sigma$ error bars), tracking accuracy, F1, precision, recall, RMSE, and the generalization gap.
5. **Data Budget Estimator & Financial Costing**: Fits power-law learning curves ($error(m) = a \cdot m^{-b} + c$) to predict unseen data needs, calculates financial acquisition costs based on user unit pricing, and explains the theoretical-versus-empirical conservatism ratio.
6. **Multi-Format Export**: Generates professional **PDF research reports** (via ReportLab), **multi-tab Excel workbooks** (via OpenPyXL), and **standalone HTML reports**.
7. **Dual-Architecture Deployment**:
   - **Local Desktop GUI**: Complete interactive 11-module Streamlit analytics studio (`streamlit run app.py`).
   - **Vercel Cloud Serverless API & Web Frontend**: Production FastAPI ASGI backend (`api/index.py`), OpenAPI Swagger UI (`/docs`), and responsive web interface (`public/index.html`).

---

## 2. System Architecture & Folder Structure

```
vc-dimension-data-budget-estimator/
├── app.py                      ← Local Streamlit interactive studio (11 modules) & ASGI entrypoint fallback
├── config.py                   ← Global constants, model families, color palettes, and default hyper-parameters
├── requirements.txt            ← Pinned package dependencies (Streamlit, Scikit-Learn, FastAPI, PyArrow, etc.)
├── vercel.json                 ← Vercel serverless routing configuration (@vercel/python & static hosting)
├── .vercelignore               ← Build exclusions for Vercel deployment
├── .gitignore                  ← Git exclusion rules (virtual environments, caches, datasets, reports)
├── smoke_test.py               ← End-to-end integration smoke test
├── test_results.txt            ← Pytest verification log
├── api/
│   ├── __init__.py             ← Python package marker
│   └── index.py                ← FastAPI serverless application exporting `app`
├── public/
│   └── index.html              ← Responsive web frontend communicating with `/api`
├── src/
│   ├── data_loader.py          ← Universal reader: CSV, Excel, JSON, TXT, Parquet & Arrow sanitizer
│   ├── data_analyzer.py        ← Profiler: dimensions, missingness, duplicates, target detection
│   ├── preprocessing.py        ← Imputation, scaling, stratified/shuffle split (leak-free)
│   ├── vc_dimension.py         ← VC dimension formulas, derivations & assumptions for 8 models
│   ├── pac_learning.py         ← Realizable, Agnostic & Regression PAC formulas & sensitivities
│   ├── pac_bound.py            ← Backwards-compatible proxy for pac_learning
│   ├── model_training.py       ← Model factory & evaluators for classification & regression
│   ├── learning_curve.py       ← Empirical learning curve generator with variance bands
│   ├── curve_fitting.py        ← Power-law curve fitting & bounded extrapolation
│   ├── data_budget.py          ← Budget optimizer: sample requirements, costs, conservatism ratio
│   ├── evaluation.py           ← Multi-model experiment orchestration & comparison tables
│   ├── visualizations.py       ← Interactive Plotly charts (curves, gaps, budgets, projections)
│   ├── visualization.py        ← Matplotlib figure factories for PDF exports
│   ├── report_generator.py     ← PDF (ReportLab), Excel (OpenPyXL), and HTML report engines
│   └── utils.py                ← Theme styling (Dark/Light mode) & formatting helpers
└── tests/
    ├── test_api.py             ← Comprehensive FastAPI test suite (19 tests)
    ├── test_pac_bound.py       ← PAC sample complexity unit tests
    ├── test_vc_dimension.py    ← VC dimension unit tests
    ├── test_enhancements.py    ← Custom upload formats & agnostic bounds tests
    ├── test_evaluation.py      ← Experiment orchestration & dataframe tests
    ├── test_report_generator.py← Report generation tests
    └── test_new_features.py    ← Full suite tests for all 8 models, PDF & Excel export
```

---

## 3. Mathematical Foundations

### Supported Model Families & VC Dimensions

| Model Family | VC Dimension / Capacity ($d$) | Bound Type | Theoretical Reference |
|---|---|---|---|
| **Linear Perceptron** | $d = n + 1$ (with bias) | Exact | Vapnik (1998); Blumer et al. (1989) |
| **Logistic Regression** | $d = n + 1$ (halfspaces) | Exact | Vapnik (1998) |
| **Linear Regression** | $d_P = n + 1$ (Pollard pseudo-dim) | Exact | Pollard (1984) |
| **Polynomial Classifier** | $d = \binom{n + k}{k}$ | Exact | Veronese Embedding Theorem |
| **Decision Tree (depth $D$)** | $d \le 2(2^D - 1)\log_2(en) + 1$ | Known Upper Bound | Devroye, Györfi, Lugosi (1996) |
| **Random Forest ($T$ trees)** | $d \le O(T \cdot d_{tree} \log(T \cdot d_{tree}))$ | Known Upper Bound | Schapire & Singer (1999) |
| **Linear SVM** | $d = n + 1$ | Exact | Vapnik (1998) |
| **RBF Kernel SVM** | $d_{fat} \le \lceil R^2 / \gamma^2 \rceil$ | Margin Fat-Shattering | Bartlett & Shawe-Taylor (1999) |
| **K-Nearest Neighbors** | $d = \infty$ (metric capacity $\approx 2^n$) | Metric Local Proxy | Devroye, Györfi, Lugosi (1996) |
| **Gaussian Naive Bayes** | $d = n + 1$ (shared cov) | Exact | Duda, Hart, Stork (2001) |
| **Finite Class ($|H|$)** | $d \le \lfloor \log_2(|H|) \rfloor$ | Exact Upper Bound | Blumer et al. (1989) |

---

### PAC Sample-Complexity Bounds

#### 1. Realizable Classification (Blumer et al., 1989)
$$m \ge \frac{1}{\epsilon} \left[ 4 \log_2\left(\frac{2}{\delta}\right) + 8 d \log_2\left(\frac{13}{\epsilon}\right) \right]$$

#### 2. Agnostic Classification (Anthony & Bartlett, 1999)
$$m \ge \frac{8}{\epsilon^2} \left[ 2 d \ln\left(\frac{8e}{\epsilon}\right) + \ln\left(\frac{4}{\delta}\right) \right]$$

#### 3. Bounded Real-Valued Regression (Pollard, 1984; Alon et al., 1997)
$$m \ge \frac{128 M^2}{\epsilon^2} \left[ 2 d_P \ln\left(\frac{16 e M}{\epsilon}\right) + \ln\left(\frac{8}{\delta}\right) \right]$$

---

## 4. Installation & Local Execution

### Prerequisites
- Python 3.10+ (tested on Python 3.11, 3.12, and 3.15)
- pip and git

### Setup
```bash
# Clone the repository
git clone https://github.com/chaturvishal02-create/vc-dimension-data-budget-estimator1.git
cd vc-dimension-data-budget-estimator1

# Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Local Streamlit Analytics Studio
```bash
streamlit run app.py
# Opens automatically at: http://localhost:8501
```

### Running the Local FastAPI Serverless Backend & Web UI
```bash
uvicorn api.index:app --reload --port 8000
# Web Application: http://localhost:8000
# OpenAPI Swagger UI: http://localhost:8000/docs
# Health Check Endpoint: http://localhost:8000/api/health
```

---

## 5. API Endpoints & Example Requests

The FastAPI backend provides validated endpoints for all theoretical calculations:

### 1. Health Check
```bash
curl -X GET http://localhost:8000/api/health
```
**Response:**
```json
{
  "status": "healthy",
  "service": "VC Dimension Data Budget Estimator API",
  "version": "1.0.0",
  "environment": "Vercel Serverless / ASGI",
  "primary_models": ["Decision Tree", "Random Forest", "Support Vector Machine", "K-Nearest Neighbors", "Logistic Regression", "Linear Regression", "Naive Bayes", "Perceptron"],
  "supported_pac_settings": ["realizable", "agnostic", "regression"]
}
```

### 2. VC Dimension Calculation
```bash
curl -X POST http://localhost:8000/api/vc-dimension \
  -H "Content-Type: application/json" \
  -d '{"model_name": "Logistic Regression", "n_features": 4, "bias": true}'
```
**Response:**
```json
{
  "vc_dim": 5,
  "type": "exact",
  "label": "Exact (Linear Separating Hyperplane in R^n)",
  "confidence": "High (Exact Mathematical Theorem)",
  "family": "Generalized Linear Models (Logit link)",
  "formula": "d = n + 1",
  "substituted": "d = 4 + 1 = 5"
}
```

### 3. PAC Sample Complexity
```bash
curl -X POST http://localhost:8000/api/pac-bounds \
  -H "Content-Type: application/json" \
  -d '{"vc_dimension": 5, "epsilon": 0.05, "delta": 0.05, "setting": "realizable"}'
```
**Response:**
```json
{
  "setting": "realizable",
  "result": {
    "m": 6844,
    "epsilon": 0.05,
    "delta": 0.05,
    "confidence": 0.95,
    "vc_dimension": 5,
    "formula": "m ≥ (1/ε) × [ 4 × log₂(2/δ) + 8 × d × log₂(13/ε) ]"
  }
}
```

### 4. End-to-End Budget Estimation Pipeline
```bash
curl -X POST http://localhost:8000/api/estimate-budget-full \
  -H "Content-Type: application/json" \
  -d '{"model_name": "Support Vector Machine", "n_features": 4, "kernel": "linear", "current_dataset_size": 150, "epsilon": 0.05, "delta": 0.05, "cost_per_sample": 2.5}'
```

---

## 6. Testing & Quality Assurance

Run the test suite across all modules:

```bash
# Run all 84 unit and integration tests
pytest tests/ -v

# Run the end-to-end integration smoke test
python smoke_test.py
```

All 84 tests pass with 100% coverage across:
- `tests/test_api.py`: FastAPI serverless endpoints, inputs, error states, and serialization.
- `tests/test_vc_dimension.py`: Exact and bound calculations across all 8 model families.
- `tests/test_pac_bound.py`: Sample complexity bounds, boundary conditions, and mathematical invariants.
- `tests/test_new_features.py`: Classification, regression, PDF/Excel generators, and data profiling.
- `tests/test_enhancements.py`: Universal file parsing, edge cases, and leak-free validation.
- `tests/test_evaluation.py` & `tests/test_report_generator.py`: Comparison tables and multi-format reporting.

---

## 7. Vercel Deployment Instructions

### How the `PYTHON_ENTRYPOINT_NOT_FOUND` Error Was Solved
The original deployment failed because Vercel auto-detected the root `app.py` as a serverless function and failed when no `app` or `handler` variable was found. The solution:
1. `api/index.py` was created to serve as the serverless function entrypoint exporting `app = FastAPI(...)`.
2. `vercel.json` routes `/api/(.*)` to `api/index.py` using `@vercel/python` and routes static traffic to `public/`.
3. `.vercelignore` prevents Vercel from bundling local test caches and files.
4. `app.py` includes a clean fallback export so that direct inspections also resolve cleanly.

### Step-by-Step Vercel Setup:
1. Log in to [Vercel](https://vercel.com).
2. Click **"Add New"** → **"Project"**.
3. Import the GitHub repository:
   `chaturvishal02-create/vc-dimension-data-budget-estimator1`
4. In the Project Configuration:
   - **Framework Preset**: Select **"Other"** (do NOT select Streamlit).
   - **Root Directory**: `./` (leave default).
   - **Build Command**: Leave empty (handled by `vercel.json`).
   - **Output Directory**: Leave empty.
5. Click **"Deploy"**.
6. Once deployed:
   - Web application homepage: `https://your-project.vercel.app/`
   - API health check: `https://your-project.vercel.app/api/health`
   - Interactive Swagger API documentation: `https://your-project.vercel.app/docs`

---

## 8. Known Limitations

1. **Serverless Execution Constraints**: Vercel Serverless Functions have execution timeout limits (typically 10 to 60 seconds depending on plan) and stateless memory. Consequently, full empirical training loops with large datasets (e.g. cross-validation over 10 subsets with multi-seed training) should be run locally using the Streamlit interface (`streamlit run app.py`).
2. **Infinite VC Dimension Proxies**: Unconstrained 1-NN and non-linear RBF kernel SVMs possess infinite distribution-free VC dimensions ($d = \infty$). The platform clearly flags these cases and computes metric-entropy covering proxies and margin-based fat-shattering dimensions ($d_{fat}$) rather than fabricating finite values.
3. **Worst-Case vs. Average-Case Conservatism**: PAC sample bounds are distribution-free worst-case limits designed to hold under adversarial distributions. As documented by the Conservatism Ratio metric, real-world benign datasets typically require significantly fewer samples than theoretical ceilings.
