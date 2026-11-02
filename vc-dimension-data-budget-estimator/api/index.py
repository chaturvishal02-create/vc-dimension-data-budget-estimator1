# api/index.py
# ─────────────────────────────────────────────────────────────────
# Vercel Serverless Entrypoint: FastAPI REST API for
# VC-Dimension-Guided Data Budget Estimator.
# Exports top-level ASGI variable: `app`
# ─────────────────────────────────────────────────────────────────

import sys
import os

# Ensure the root repository directory is on sys.path so that `src` and `config` resolve
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from typing import Dict, Any, List, Optional, Union
import math
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import config
from src.vc_dimension import estimate_vc_dimension
from src.pac_learning import (
    calculate_theoretical_samples,
    calculate_agnostic_samples,
    calculate_regression_samples,
    get_pac_bounds,
    generate_epsilon_sensitivity_curve,
    generate_delta_sensitivity_curve,
    generate_vc_sensitivity_curve,
)
from src.data_budget import estimate_data_budget


def to_serializable(val: Any) -> Any:
    """Recursively convert NumPy objects and non-finite floats to JSON-serializable types."""
    if isinstance(val, dict):
        return {k: to_serializable(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple)):
        return [to_serializable(v) for v in val]
    elif isinstance(val, (np.ndarray, pd.Series)):
        return val.tolist()
    elif isinstance(val, (np.integer,)):
        return int(val)
    elif isinstance(val, (np.floating,)):
        return float(val)
    elif isinstance(val, (np.bool_,)):
        return bool(val)
    elif isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return str(val)
        return val
    return val


# Initialize FastAPI application
app = FastAPI(
    title="VC Dimension Data Budget Estimator API",
    description=(
        "Production REST API bridging PAC Learning Theory with Empirical Practice. "
        "Calculates mathematical VC dimensions, distribution-free sample bounds, "
        "and comparative data acquisition budgets."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Enable CORS for universal frontend accessibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Schemas ────────────────────────────────────

class VCDimensionRequest(BaseModel):
    model_name: str = Field(..., description="Name of the ML model family")
    n_features: int = Field(..., ge=1, description="Number of input features (n >= 1)")
    bias: bool = Field(True, description="Whether linear model includes an intercept term")
    degree: int = Field(2, ge=1, le=10, description="Degree for polynomial classifier")
    max_depth: int = Field(3, ge=1, le=20, description="Maximum tree depth for Decision Tree / Random Forest")
    n_estimators: int = Field(50, ge=1, le=1000, description="Ensemble size for Random Forest / AdaBoost")
    kernel: str = Field("linear", description="Kernel function for SVM ('linear' or 'rbf')")
    C: float = Field(1.0, gt=0, description="Regularization parameter C for SVM")
    k: int = Field(3, ge=1, description="Number of neighbors for KNN")
    variant: str = Field("gaussian", description="Naive Bayes variant ('gaussian')")
    cardinality: int = Field(1000, ge=2, description="Hypothesis space cardinality |H| for Finite Class")


class PACBoundRequest(BaseModel):
    vc_dimension: int = Field(..., ge=1, description="Vapnik-Chervonenkis dimension (d >= 1)")
    epsilon: float = Field(0.05, gt=0.0, lt=1.0, description="Target generalization error tolerance (0 < ε < 1)")
    delta: float = Field(0.05, gt=0.0, lt=1.0, description="PAC failure probability risk (0 < δ < 1, confidence = 1 - δ)")
    setting: str = Field(
        "realizable",
        description="Learning setting: 'realizable', 'agnostic', 'regression', or 'all'"
    )
    y_bound: float = Field(1.0, gt=0.0, description="Target output bound [-M, M] for real-valued regression")


class DataBudgetRequest(BaseModel):
    theoretical_m: int = Field(..., ge=1, description="Theoretical PAC sample requirement bound")
    empirical_achieved: bool = Field(False, description="Whether target error was directly observed in empirical trials")
    empirical_m: Optional[int] = Field(None, ge=1, description="Observed empirical sample size (if achieved)")
    current_dataset_size: int = Field(..., ge=1, description="Total number of currently available samples")
    target_error: float = Field(0.05, gt=0.0, lt=1.0, description="Desired error tolerance ε")
    cost_per_sample: float = Field(1.0, ge=0.0, description="Estimated annotation/collection cost in dollars ($/sample)")
    confidence_level: float = Field(0.95, gt=0.0, lt=1.0, description="Theoretical PAC confidence (1 - δ)")
    sample_counts: Optional[List[int]] = Field(default_factory=list, description="Observed subset sample sizes")
    val_errors: Optional[List[float]] = Field(default_factory=list, description="Observed validation errors per subset")


class FullEstimateRequest(BaseModel):
    model_name: str = Field("Logistic Regression", description="Selected ML model")
    n_features: int = Field(4, ge=1, description="Number of input features")
    current_dataset_size: int = Field(150, ge=1, description="Current sample count")
    epsilon: float = Field(0.05, gt=0.0, lt=1.0, description="Target error tolerance ε")
    delta: float = Field(0.05, gt=0.0, lt=1.0, description="Failure probability risk δ")
    cost_per_sample: float = Field(1.0, ge=0.0, description="Unit sample cost ($)")
    bias: bool = Field(True, description="Linear model bias flag")
    degree: int = Field(2, ge=1, le=10, description="Polynomial degree")
    max_depth: int = Field(3, ge=1, le=20, description="Tree max depth")
    n_estimators: int = Field(50, ge=1, le=1000, description="Random Forest trees")
    kernel: str = Field("linear", description="SVM kernel")
    C: float = Field(1.0, gt=0, description="SVM C")
    k: int = Field(3, ge=1, description="KNN k")
    variant: str = Field("gaussian", description="Naive Bayes variant")
    cardinality: int = Field(1000, ge=2, description="Finite class size")


class SensitivityRequest(BaseModel):
    param_type: str = Field("epsilon", description="Sensitivity parameter: 'epsilon', 'delta', or 'vc'")
    vc_dimension: int = Field(5, ge=1, description="Base VC dimension for sensitivity curve")
    epsilon: float = Field(0.05, gt=0.0, lt=1.0, description="Base epsilon")
    delta: float = Field(0.05, gt=0.0, lt=1.0, description="Base delta")


# ── Core Endpoints ────────────────────────────────────────────────

@app.get("/api/health", status_code=status.HTTP_200_OK, tags=["System"])
def health_check() -> Dict[str, Any]:
    """
    Health check and metadata endpoint for Vercel deployment verification.
    """
    return {
        "status": "healthy",
        "service": "VC Dimension Data Budget Estimator API",
        "version": "1.0.0",
        "environment": "Vercel Serverless / ASGI",
        "primary_models": getattr(config, "PRIMARY_MODELS", [
            "Decision Tree", "Random Forest", "Support Vector Machine",
            "K-Nearest Neighbors", "Logistic Regression", "Linear Regression",
            "Naive Bayes", "Perceptron"
        ]),
        "supported_pac_settings": ["realizable", "agnostic", "regression"],
    }


@app.get("/api", status_code=status.HTTP_200_OK, tags=["System"])
def api_root() -> Dict[str, Any]:
    """
    API discovery and navigation directory.
    """
    return {
        "message": "Welcome to the VC Dimension Data Budget Estimator API",
        "endpoints": {
            "health": "/api/health",
            "models": "/api/models",
            "vc_dimension": "/api/vc-dimension",
            "pac_bounds": "/api/pac-bounds",
            "data_budget": "/api/data-budget",
            "full_estimate": "/api/estimate-budget-full",
            "pac_sensitivity": "/api/pac-sensitivity",
            "documentation": "/docs",
            "openapi": "/openapi.json",
        },
        "note": "For the full local GUI with dataset upload and learning curves, run `streamlit run app.py`.",
    }


@app.get("/api/models", status_code=status.HTTP_200_OK, tags=["Models"])
def list_supported_models() -> Dict[str, Any]:
    """
    List all supported ML model families with their theoretical classification/regression properties.
    """
    primary = getattr(config, "PRIMARY_MODELS", [
        "Decision Tree", "Random Forest", "Support Vector Machine",
        "K-Nearest Neighbors", "Logistic Regression", "Linear Regression",
        "Naive Bayes", "Perceptron"
    ])
    classification_models = getattr(config, "CLASSIFICATION_MODELS", [
        "Decision Tree", "Random Forest", "Support Vector Machine",
        "K-Nearest Neighbors", "Logistic Regression", "Naive Bayes", "Perceptron"
    ])
    regression_models = getattr(config, "REGRESSION_MODELS", [
        "Decision Tree", "Random Forest", "Support Vector Machine",
        "K-Nearest Neighbors", "Linear Regression", "Perceptron"
    ])
    return {
        "primary_models": primary,
        "classification_models": classification_models,
        "regression_models": regression_models,
        "extended_families": ["Polynomial Classifier", "Finite Class", "MLP (Multilayer Perceptron)", "AdaBoost"],
    }


@app.post("/api/vc-dimension", status_code=status.HTTP_200_OK, tags=["Theory"])
def calculate_vc_dimension(req: VCDimensionRequest) -> Dict[str, Any]:
    """
    Calculate or bound the Vapnik-Chervonenkis (VC) dimension for a selected model family.
    Reuses existing mathematical functions from `src/vc_dimension.py`.
    """
    try:
        vc_info = estimate_vc_dimension(
            model_name=req.model_name,
            n_features=req.n_features,
            bias=req.bias,
            degree=req.degree,
            max_depth=req.max_depth,
            n_estimators=req.n_estimators,
            kernel=req.kernel,
            C=req.C,
            k=req.k,
            variant=req.variant,
            cardinality=req.cardinality,
        )
        return vc_info
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error evaluating VC dimension: {str(exc)}"
        )


@app.post("/api/pac-bounds", status_code=status.HTTP_200_OK, tags=["Theory"])
def calculate_pac_sample_complexity(req: PACBoundRequest) -> Dict[str, Any]:
    """
    Calculate theoretical PAC sample-size bounds based on VC dimension, epsilon, and delta.
    Supports Realizable (Blumer et al.), Agnostic (Anthony & Bartlett), and Regression settings.
    """
    setting = req.setting.lower()
    try:
        if setting == "realizable":
            res = calculate_theoretical_samples(
                epsilon=req.epsilon, delta=req.delta, vc_dimension=req.vc_dimension
            )
            return to_serializable({"setting": "realizable", "result": res})
        elif setting == "agnostic":
            res = calculate_agnostic_samples(
                epsilon=req.epsilon, delta=req.delta, vc_dimension=req.vc_dimension
            )
            return to_serializable({"setting": "agnostic", "result": res})
        elif setting == "regression":
            res = calculate_regression_samples(
                epsilon=req.epsilon, delta=req.delta, pseudo_dim=req.vc_dimension, target_scale_M=req.y_bound
            )
            return to_serializable({"setting": "regression", "result": res})
        elif setting == "all":
            all_bounds = get_pac_bounds(
                epsilon=req.epsilon, delta=req.delta, vc_dimension=req.vc_dimension
            )
            all_bounds["regression"] = calculate_regression_samples(
                epsilon=req.epsilon, delta=req.delta, pseudo_dim=req.vc_dimension, target_scale_M=req.y_bound
            )
            return to_serializable({"setting": "all", "bounds": all_bounds})
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid setting '{req.setting}'. Choose 'realizable', 'agnostic', 'regression', or 'all'."
            )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error computing PAC bound: {str(exc)}"
        )


@app.post("/api/data-budget", status_code=status.HTTP_200_OK, tags=["Budget"])
def estimate_budget(req: DataBudgetRequest) -> Dict[str, Any]:
    """
    Calculate comparative data budget, financial acquisition cost, and reliability metrics.
    Reuses `src/data_budget.py:estimate_data_budget`.
    """
    try:
        lc_result = {
            "sample_counts": req.sample_counts or [],
            "val_errors": req.val_errors or [],
        }
        budget_report = estimate_data_budget(
            theoretical_m=req.theoretical_m,
            empirical_achieved=req.empirical_achieved,
            empirical_m=req.empirical_m,
            current_dataset_size=req.current_dataset_size,
            lc_result=lc_result,
            target_error=req.target_error,
            cost_per_sample=req.cost_per_sample,
            confidence_level=req.confidence_level,
        )
        return to_serializable(budget_report)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error computing data budget: {str(exc)}"
        )


@app.post("/api/estimate-budget-full", status_code=status.HTTP_200_OK, tags=["Full Pipeline"])
def estimate_budget_full(req: FullEstimateRequest) -> Dict[str, Any]:
    """
    One-click end-to-end mathematical budget pipeline:
    1. Computes VC dimension for the selected model.
    2. Calculates Realizable and Agnostic PAC bounds.
    3. Calculates required additional samples and financial budget.
    """
    try:
        # 1. VC dimension
        vc_info = estimate_vc_dimension(
            model_name=req.model_name,
            n_features=req.n_features,
            bias=req.bias,
            degree=req.degree,
            max_depth=req.max_depth,
            n_estimators=req.n_estimators,
            kernel=req.kernel,
            C=req.C,
            k=req.k,
            variant=req.variant,
            cardinality=req.cardinality,
        )
        d = vc_info["vc_dim"]

        # 2. PAC bounds
        realizable_res = calculate_theoretical_samples(
            epsilon=req.epsilon, delta=req.delta, vc_dimension=d
        )
        agnostic_res = calculate_agnostic_samples(
            epsilon=req.epsilon, delta=req.delta, vc_dimension=d
        )

        m_theory = realizable_res["m"]
        additional_needed = max(0, m_theory - req.current_dataset_size)
        total_cost = m_theory * req.cost_per_sample
        additional_cost = additional_needed * req.cost_per_sample

        return to_serializable({
            "model_name": req.model_name,
            "n_features": req.n_features,
            "vc_dimension": vc_info,
            "realizable_bound": realizable_res,
            "agnostic_bound": agnostic_res,
            "budget_summary": {
                "current_dataset_size": req.current_dataset_size,
                "theoretical_m": m_theory,
                "additional_samples_needed": additional_needed,
                "cost_per_sample": req.cost_per_sample,
                "total_theoretical_cost": total_cost,
                "additional_acquisition_cost": additional_cost,
                "confidence_percentage": f"{(1 - req.delta) * 100:.1f}%",
                "target_error_percentage": f"{req.epsilon * 100:.1f}%",
            },
        })
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Full pipeline calculation error: {str(exc)}"
        )


@app.post("/api/pac-sensitivity", status_code=status.HTTP_200_OK, tags=["Theory"])
def pac_sensitivity(req: SensitivityRequest) -> Dict[str, Any]:
    """
    Generate sensitivity curve data points for charting PAC sample complexity across variations.
    """
    try:
        ptype = req.param_type.lower()
        if ptype == "epsilon":
            data = generate_epsilon_sensitivity_curve(
                delta=req.delta, vc_dim=req.vc_dimension
            )
            return to_serializable({"parameter": "epsilon", "curve_points": data})
        elif ptype == "delta":
            data = generate_delta_sensitivity_curve(
                epsilon=req.epsilon, vc_dim=req.vc_dimension
            )
            return to_serializable({"parameter": "delta", "curve_points": data})
        elif ptype in ("vc", "vc_dimension"):
            data = generate_vc_sensitivity_curve(
                epsilon=req.epsilon, delta=req.delta
            )
            return to_serializable({"parameter": "vc_dimension", "curve_points": data})
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid param_type '{req.param_type}'. Choose 'epsilon', 'delta', or 'vc'."
            )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sensitivity calculation error: {str(exc)}"
        )


# ── Mount static files for local testing & preview ───────────────
from fastapi.staticfiles import StaticFiles

public_dir = os.path.join(ROOT_DIR, "public")
if os.path.isdir(public_dir):
    app.mount("/", StaticFiles(directory=public_dir, html=True), name="static")

