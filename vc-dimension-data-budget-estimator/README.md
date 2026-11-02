# How Much Data Does Your Model Really Need?
## A VC-Dimension-Guided Data Budget Estimator — Bridging PAC Learning Theory with Empirical Practice

---

## Abstract

This platform bridges the fundamental divide between theoretical machine-learning sample-complexity guarantees and real-world empirical data requirements. Grounded in the Probably Approximately Correct (PAC) learning framework and Vapnik-Chervonenkis (VC) dimension theory, the platform:

1. **Universal Dataset Ingestion**: Accepts user datasets in **CSV, Excel (.xlsx, .xls), JSON, TXT (tabular), and Parquet** formats with automatic column type, dimension, missingness, and task detection.
2. **VC Dimension & Statistical Capacity Estimation**: Computes exact theoretical capacities, known bounds, and structural capacity proxies across **8 machine learning model families** (Decision Tree, Random Forest, SVM, KNN, Logistic Regression, Linear Regression, Naive Bayes, Perceptron).
3. **PAC Sample-Complexity Calculations**: Computes distribution-free sample bounds under **Realizable** (Blumer et al., 1989), **Agnostic** (Anthony & Bartlett, 1999), and **Regression** (Pollard, 1984) settings.
4. **Empirical Learning Curve Suite**: Measures real empirical performance across progressive training subsets ($10\%$ to $100\%$) with multi-seed variance ($\pm 1\sigma$ error bars), tracking accuracy, F1, precision, recall, RMSE, and the generalization gap.
5. **Data Budget Estimator & Financial Costing**: Fits power-law learning curves ($error(m) = a \cdot m^{-b} + c$) to predict unseen data needs, calculates financial acquisition costs based on user unit pricing, and explains the theoretical-versus-empirical conservatism ratio.
6. **Multi-Format Export**: Generates professional **PDF research reports** (via ReportLab), **multi-tab Excel workbooks** (via OpenPyXL), and **standalone HTML reports**.

---

## System Architecture

```
vc-dimension-data-budget-estimator/
├── app.py                      ← Interactive Streamlit dashboard (11 modules)
├── config.py                   ← Global constants, models, color palettes & defaults
├── requirements.txt            ← Pinned package dependencies
├── smoke_test.py               ← End-to-end integration smoke test
├── test_results.txt            ← Pytest verification logs (65 passing tests)
├── src/
│   ├── data_loader.py          ← Universal reader: CSV, Excel, JSON, TXT, Parquet
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
    ├── test_pac_bound.py       ← PAC sample complexity unit tests
    ├── test_vc_dimension.py    ← VC dimension unit tests
    ├── test_enhancements.py    ← Custom upload formats & agnostic bounds tests
    ├── test_evaluation.py      ← Experiment orchestration & dataframe tests
    ├── test_report_generator.py← Report generation tests
    └── test_new_features.py    ← Full suite tests for all 8 models, PDF & Excel export
```

---

## Theory: PAC Learning and VC Dimension

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

## Installation & Execution

### Prerequisites
- Python 3.11+ (tested on Python 3.15)
- pip

### Setup
```bash
# 1. Navigate to project root
cd "vc-dimension-data-budget-estimator"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run complete test suite (65 tests)
py -m pytest tests/ -v

# 4. Run end-to-end integration smoke test
py smoke_test.py

# 5. Launch interactive platform
streamlit run app.py
```

The application opens at: **http://localhost:8501**

---

## Interactive Dashboard Modules

The platform is organized into 11 responsive, dark/light theme-aware views:

1. **🏠 Home Dashboard**: System overview, active KPI cards, 5-step guided roadmap, problem statement.
2. **📤 Dataset Upload**: Multi-format file uploader (CSV, Excel, JSON, TXT, Parquet) with target detection and task classifier.
3. **🔍 Dataset Explorer**: Deep statistical profiling, missing values breakdown, 2D feature projection, class balance/target histograms.
4. **🤖 Model Selection**: Selection across 8 models, hyperparameter expanders, test split sizing, and execution engine.
5. **📐 VC Dimension Calculator**: Interactive capacity studio with derivations, assumptions, and confidence ratings.
6. **🎯 PAC Sample Complexity**: Sliders for $\epsilon$, $\delta$, formula substitution, and sensitivity curves ($m$ vs $\epsilon$, $\delta$, $d$).
7. **📈 Learning Curve Analysis**: Interactive Plotly curves with $\pm 1\sigma$ variance bands, generalization gap metrics, and tabular logs.
8. **💰 Data Budget Estimator**: Side-by-side comparison of PAC bound, empirical observations, power-law projections, and financial collection costs.
9. **🏆 Model Comparison**: Executive leaderboard and log-scale bar charts comparing data hunger and performance across models.
10. **🔬 Mathematical Explanation**: Academic derivations, bias-variance analysis, synthetic label noise simulator, and citations.
11. **📑 Download Reports**: Instant downloads for **PDF Reports** (ReportLab), **Excel Workbooks** (OpenPyXL), and **HTML Reports**.

---

## References

1. **Blumer, A., Ehrenfeucht, A., Haussler, D., & Warmuth, M. K.** (1989). Learnability and the Vapnik-Chervonenkis dimension. *Journal of the ACM, 36*(4), 929–965.
2. **Vapnik, V. N.** (1998). *Statistical Learning Theory*. John Wiley & Sons.
3. **Anthony, M., & Bartlett, P. L.** (1999). *Neural Network Learning: Theoretical Foundations*. Cambridge University Press.
4. **Pollard, D.** (1984). *Convergence of Stochastic Processes*. Springer-Verlag.
5. **Devroye, L., Györfi, L., & Lugosi, G.** (1996). *A Probabilistic Theory of Pattern Recognition*. Springer.
6. **Bartlett, P. L., & Shawe-Taylor, J.** (1999). Generalization performance of Support Vector Machines and other pattern classifiers. *Advances in Kernel Methods*, 43–54.
7. **Schapire, R. E., & Singer, Y.** (1999). Improved boosting algorithms using confidence-rated predictions. *Machine Learning, 37*(3), 297–336.
