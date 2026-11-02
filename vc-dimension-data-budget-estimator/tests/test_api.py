# tests/test_api.py
# ─────────────────────────────────────────────────────────────────
# Comprehensive test suite for the Vercel FastAPI backend.
# Tests health check, model discovery, VC dimension mathematical dispatch,
# PAC sample complexity bounds, and data budget estimation.
# ─────────────────────────────────────────────────────────────────

import pytest
from fastapi.testclient import TestClient
from api.index import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "service" in data
    assert "primary_models" in data
    assert len(data["primary_models"]) >= 8
    assert "supported_pac_settings" in data


def test_api_root_endpoint():
    response = client.get("/api")
    assert response.status_code == 200
    data = response.json()
    assert "endpoints" in data
    assert data["endpoints"]["health"] == "/api/health"
    assert data["endpoints"]["vc_dimension"] == "/api/vc-dimension"
    assert data["endpoints"]["pac_bounds"] == "/api/pac-bounds"


def test_list_models_endpoint():
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "primary_models" in data
    assert "Decision Tree" in data["primary_models"]
    assert "Perceptron" in data["primary_models"]
    assert "Logistic Regression" in data["primary_models"]
    assert "Linear Regression" in data["primary_models"]


def test_vc_dimension_perceptron():
    # Perceptron with 5 features and bias: d = 5 + 1 = 6
    payload = {"model_name": "Perceptron", "n_features": 5, "bias": True}
    response = client.post("/api/vc-dimension", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["vc_dim"] == 6
    assert data["type"] == "exact"
    assert "Radon" in data["label"] or "Exact" in data["label"]


def test_vc_dimension_linear_regression():
    # Linear regression with 10 features without bias: d = 10
    payload = {"model_name": "Linear Regression", "n_features": 10, "bias": False}
    response = client.post("/api/vc-dimension", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["vc_dim"] == 10
    assert data["type"] == "exact"


def test_vc_dimension_logistic_regression():
    # Logistic regression with 4 features and bias: d = 5
    payload = {"model_name": "Logistic Regression", "n_features": 4, "bias": True}
    response = client.post("/api/vc-dimension", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["vc_dim"] == 5
    assert data["type"] == "exact"


def test_vc_dimension_decision_tree():
    payload = {"model_name": "Decision Tree", "n_features": 4, "max_depth": 3}
    response = client.post("/api/vc-dimension", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["vc_dim"] > 0
    assert "bound" in data["type"].lower() or "exact" in data["type"].lower() or "heuristic" in data["type"].lower()


def test_vc_dimension_svm():
    # Linear SVM: d = n + 1 = 4 + 1 = 5
    payload = {"model_name": "Support Vector Machine", "n_features": 4, "kernel": "linear"}
    response = client.post("/api/vc-dimension", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["vc_dim"] == 5

    # RBF SVM
    payload_rbf = {"model_name": "Support Vector Machine", "n_features": 4, "kernel": "rbf"}
    res_rbf = client.post("/api/vc-dimension", json=payload_rbf)
    assert res_rbf.status_code == 200
    assert res_rbf.json()["vc_dim"] >= 5


def test_vc_dimension_knn():
    payload = {"model_name": "K-Nearest Neighbors", "n_features": 3, "k": 3}
    response = client.post("/api/vc-dimension", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["vc_dim"] > 0
    assert "metric" in data["label"].lower() or "effective" in data["label"].lower()


def test_vc_dimension_finite_class():
    # |H| = 1024 -> d = floor(log2(1024)) = 10
    payload = {"model_name": "Finite Class", "n_features": 1, "cardinality": 1024}
    response = client.post("/api/vc-dimension", json=payload)
    assert response.status_code == 200
    assert response.json()["vc_dim"] == 10


def test_vc_dimension_invalid_inputs():
    # Negative features
    payload = {"model_name": "Perceptron", "n_features": 0}
    response = client.post("/api/vc-dimension", json=payload)
    assert response.status_code == 422  # Pydantic validation error

    # Unknown model
    payload_unknown = {"model_name": "QuantumTransformerXYZ", "n_features": 4}
    res_unknown = client.post("/api/vc-dimension", json=payload_unknown)
    assert res_unknown.status_code == 400


def test_pac_bounds_realizable():
    payload = {"vc_dimension": 5, "epsilon": 0.05, "delta": 0.05, "setting": "realizable"}
    response = client.post("/api/pac-bounds", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["setting"] == "realizable"
    res = data["result"]
    assert res["m"] > 0
    assert res["epsilon"] == 0.05
    assert res["delta"] == 0.05
    assert "formula" in res
    assert "substituted" in res


def test_pac_bounds_agnostic():
    payload = {"vc_dimension": 5, "epsilon": 0.05, "delta": 0.05, "setting": "agnostic"}
    response = client.post("/api/pac-bounds", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["setting"] == "agnostic"
    assert data["result"]["m"] > 0


def test_pac_bounds_regression():
    payload = {"vc_dimension": 5, "epsilon": 0.1, "delta": 0.05, "setting": "regression", "y_bound": 1.0}
    response = client.post("/api/pac-bounds", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["setting"] == "regression"
    assert data["result"]["m"] > 0


def test_pac_bounds_all():
    payload = {"vc_dimension": 5, "epsilon": 0.05, "delta": 0.05, "setting": "all"}
    response = client.post("/api/pac-bounds", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["setting"] == "all"
    assert "realizable" in data["bounds"]
    assert "agnostic" in data["bounds"]
    assert "regression" in data["bounds"]


def test_pac_bounds_invalid_parameters():
    # Epsilon out of bounds
    res_eps = client.post("/api/pac-bounds", json={"vc_dimension": 5, "epsilon": 1.5, "delta": 0.05})
    assert res_eps.status_code == 422

    # Delta out of bounds
    res_del = client.post("/api/pac-bounds", json={"vc_dimension": 5, "epsilon": 0.05, "delta": 0.0})
    assert res_del.status_code == 422

    # Invalid setting
    res_set = client.post("/api/pac-bounds", json={"vc_dimension": 5, "epsilon": 0.05, "delta": 0.05, "setting": "magical"})
    assert res_set.status_code == 400


def test_data_budget_endpoint():
    payload = {
        "theoretical_m": 2500,
        "empirical_achieved": True,
        "empirical_m": 120,
        "current_dataset_size": 150,
        "target_error": 0.05,
        "cost_per_sample": 2.5,
        "confidence_level": 0.95,
        "sample_counts": [15, 30, 60, 90, 120],
        "val_errors": [0.35, 0.20, 0.12, 0.07, 0.04],
    }
    response = client.post("/api/data-budget", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["theoretical_m"] == 2500
    assert data["empirical_m"] == 120
    assert data["empirical_achieved"] is True
    assert data["current_dataset_size"] == 150
    assert data["cost_per_sample"] == 2.5
    assert data["theoretical_cost"] == 2500 * 2.5
    assert "conservatism_ratio" in data
    assert "reliability_rating" in data


def test_full_pipeline_endpoint():
    payload = {
        "model_name": "Logistic Regression",
        "n_features": 4,
        "current_dataset_size": 150,
        "epsilon": 0.05,
        "delta": 0.05,
        "cost_per_sample": 1.5,
    }
    response = client.post("/api/estimate-budget-full", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "Logistic Regression"
    assert data["vc_dimension"]["vc_dim"] == 5
    assert "realizable_bound" in data
    assert "agnostic_bound" in data
    summary = data["budget_summary"]
    assert summary["current_dataset_size"] == 150
    assert summary["theoretical_m"] > 0
    assert summary["total_theoretical_cost"] == summary["theoretical_m"] * 1.5


def test_pac_sensitivity_endpoint():
    # Epsilon sensitivity
    res_eps = client.post("/api/pac-sensitivity", json={"param_type": "epsilon", "vc_dimension": 5, "delta": 0.05})
    assert res_eps.status_code == 200
    assert len(res_eps.json()["curve_points"]) > 0

    # Delta sensitivity
    res_del = client.post("/api/pac-sensitivity", json={"param_type": "delta", "vc_dimension": 5, "epsilon": 0.05})
    assert res_del.status_code == 200
    assert len(res_del.json()["curve_points"]) > 0

    # VC sensitivity
    res_vc = client.post("/api/pac-sensitivity", json={"param_type": "vc", "epsilon": 0.05, "delta": 0.05})
    assert res_vc.status_code == 200
    assert len(res_vc.json()["curve_points"]) > 0

    # Invalid
    res_inv = client.post("/api/pac-sensitivity", json={"param_type": "invalid_param"})
    assert res_inv.status_code == 400

