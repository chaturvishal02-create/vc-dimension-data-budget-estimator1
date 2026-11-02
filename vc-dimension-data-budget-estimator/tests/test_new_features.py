# tests/test_new_features.py
# ─────────────────────────────────────────────────────────────────
# Unit tests for the upgraded platform:
# 1. 8 Machine Learning Models (Classification & Regression)
# 2. Comprehensive VC Dimension Formulas
# 3. Data Analyzer & Profiler
# 4. Data Budget Estimator
# 5. PDF & Excel Report Generators
# ─────────────────────────────────────────────────────────────────

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from config import PRIMARY_MODELS
from src.model_training import build_model, train_and_evaluate_model
from src.vc_dimension import (
    estimate_vc_dimension, vc_perceptron, vc_logistic_regression,
    vc_linear_regression, vc_polynomial_classifier, vc_decision_tree,
    vc_random_forest, vc_svm, vc_knn, vc_naive_bayes, vc_finite_class
)
from src.pac_learning import calculate_regression_samples, generate_epsilon_sensitivity_curve
from src.data_analyzer import analyze_dataframe, profile_target_variable
from src.data_budget import estimate_data_budget, calculate_fit_r2
from src.report_generator import generate_pdf_report, generate_excel_report


@pytest.fixture
def synthetic_data():
    X_c, y_c = make_classification(n_samples=60, n_features=5, n_informative=3, n_redundant=1, n_classes=2, random_state=42)
    X_r, y_r = make_regression(n_samples=60, n_features=5, noise=0.1, random_state=42)
    return {
        "X_c": X_c, "y_c": y_c,
        "X_r": X_r, "y_r": y_r,
    }


# ── 1. Model Training Tests ───────────────────────────────────────

def test_primary_models_classification(synthetic_data):
    X = synthetic_data["X_c"]
    y = synthetic_data["y_c"]
    X_tr, X_te = X[:40], X[40:]
    y_tr, y_te = y[:40], y[40:]

    for m in PRIMARY_MODELS:
        res = train_and_evaluate_model(m, X_tr, y_tr, X_te, y_te, task_type="classification")
        assert res["task_type"] == "classification"
        assert 0.0 <= res["accuracy"] <= 1.0
        assert res["fit_time_seconds"] >= 0.0


def test_primary_models_regression(synthetic_data):
    X = synthetic_data["X_r"]
    y = synthetic_data["y_r"]
    X_tr, X_te = X[:40], X[40:]
    y_tr, y_te = y[:40], y[40:]

    reg_models = ["Decision Tree", "Random Forest", "Support Vector Machine", "K-Nearest Neighbors", "Linear Regression", "Perceptron"]
    for m in reg_models:
        res = train_and_evaluate_model(m, X_tr, y_tr, X_te, y_te, task_type="regression")
        assert res["task_type"] == "regression"
        assert "rmse" in res
        assert "r2_score" in res


# ── 2. VC Dimension Tests ─────────────────────────────────────────

def test_vc_dimension_all_models():
    n = 5
    assert vc_perceptron(n)["vc_dim"] == 6
    assert vc_logistic_regression(n)["vc_dim"] == 6
    assert vc_linear_regression(n)["vc_dim"] == 6
    assert vc_polynomial_classifier(n, degree=2)["vc_dim"] == 21
    assert vc_decision_tree(n, max_depth=3)["vc_dim"] > 0
    assert vc_random_forest(n, n_estimators=10)["vc_dim"] > 0
    assert vc_svm(n, kernel="linear")["vc_dim"] == 6
    assert vc_svm(n, kernel="rbf")["vc_dim"] >= 6
    assert vc_knn(n, k=3)["vc_dim"] > 0
    assert vc_naive_bayes(n)["vc_dim"] == 6
    assert vc_finite_class(1024)["vc_dim"] == 10

    # Dispatcher verification
    for m in PRIMARY_MODELS:
        est = estimate_vc_dimension(m, n)
        assert "vc_dim" in est
        assert est["vc_dim"] > 0
        assert "confidence" in est
        assert "formula" in est


# ── 3. PAC Regression & Sensitivity ───────────────────────────────

def test_regression_pac_bounds():
    pac_reg = calculate_regression_samples(epsilon=0.10, delta=0.05, pseudo_dim=5, target_scale_M=1.0)
    assert pac_reg["m"] > 0
    assert pac_reg["setting"] == "regression"

    sens = generate_epsilon_sensitivity_curve(delta=0.05, vc_dim=5, steps=5)
    assert len(sens["epsilons"]) == 5
    assert len(sens["realizable_m"]) == 5
    # Smaller epsilon requires more samples
    assert sens["realizable_m"][0] > sens["realizable_m"][-1]


# ── 4. Data Analyzer Tests ────────────────────────────────────────

def test_data_analyzer():
    df = pd.DataFrame({
        "num1": np.arange(50, dtype=float),
        "num2": np.random.randn(50),
        "cat1": ["A", "B"] * 25,
        "target": [0, 1] * 25,
    })
    analysis = analyze_dataframe(df)
    assert analysis["n_rows"] == 50
    assert analysis["n_cols"] == 4
    assert "num1" in analysis["numeric_cols"]
    assert "cat1" in analysis["categorical_cols"]
    assert len(analysis["target_candidates"]) > 0

    target_profile = profile_target_variable(df, "target")
    assert target_profile["task_type"] == "classification"
    assert target_profile["unique_values"] == 2


# ── 5. Data Budget Estimator Tests ────────────────────────────────

def test_data_budget_estimator():
    lc_res = {
        "sample_counts": [10, 20, 30, 40, 50],
        "val_errors": [0.40, 0.25, 0.15, 0.10, 0.08],
    }
    budget = estimate_data_budget(
        theoretical_m=2000,
        empirical_achieved=True,
        empirical_m=40,
        current_dataset_size=50,
        lc_result=lc_res,
        target_error=0.10,
        cost_per_sample=2.5,
    )
    assert budget["theoretical_m"] == 2000
    assert budget["recommended_empirical_m"] == 40
    assert budget["conservatism_ratio"] == 50.0  # 2000 / 40
    assert budget["theoretical_cost"] == 5000.0  # 2000 * 2.5


# ── 6. PDF and Excel Report Tests ─────────────────────────────────

def test_generate_pdf_report():
    mock_results = [{
        "model_name": "Decision Tree",
        "vc_info": {"vc_dim": 15, "label": "Upper Bound"},
        "pac_bound": {"m": 4500},
        "empirical": {"sample_count": 50},
        "prediction": {"sample_count": 55},
        "final_train_err": 0.02,
        "final_val_err": 0.08,
        "final_val_acc": 0.92,
        "epsilon": 0.10,
        "delta": 0.05,
        "fit_time_seconds": 0.04,
        "generalization_gap": 0.06,
    }]
    pdf_bytes = generate_pdf_report(
        results=mock_results,
        dataset_name="Test Dataset",
        dataset_info={"n_samples": 100, "n_features": 4, "task_type": "classification"},
        epsilon=0.10,
        delta=0.05,
    )
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 500
    assert pdf_bytes.startswith(b"%PDF")


def test_generate_excel_report():
    mock_results = [{
        "model_name": "Random Forest",
        "vc_info": {"vc_dim": 120, "label": "Ensemble Bound"},
        "pac_bound": {"m": 12000},
        "empirical": {"sample_count": 60},
        "prediction": {"sample_count": 65},
        "final_train_err": 0.01,
        "final_val_err": 0.05,
        "final_val_acc": 0.95,
        "epsilon": 0.10,
        "delta": 0.05,
        "fit_time_seconds": 0.12,
        "generalization_gap": 0.04,
        "learning_curve": {
            "sample_counts": [20, 40, 60],
            "train_errors": [0.05, 0.02, 0.01],
            "val_errors": [0.15, 0.08, 0.05],
            "train_errors_std": [0.01, 0.01, 0.005],
            "val_errors_std": [0.02, 0.01, 0.01],
            "generalization_gaps": [0.10, 0.06, 0.04],
        }
    }]
    excel_bytes = generate_excel_report(
        results=mock_results,
        dataset_name="Test Dataset",
        dataset_info={"n_samples": 100, "n_features": 4},
        epsilon=0.10,
        delta=0.05,
        cost_per_sample=1.5,
    )
    assert isinstance(excel_bytes, bytes)
    assert len(excel_bytes) > 1000
