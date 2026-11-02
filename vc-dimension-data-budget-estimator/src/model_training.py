# src/model_training.py
# ─────────────────────────────────────────────────────────────────
# Unified ML Model Factory & Evaluator for both Classification
# and Regression tasks.
# ─────────────────────────────────────────────────────────────────

import time
from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    AdaBoostClassifier, AdaBoostRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge,
    Perceptron as SkPerceptron, SGDRegressor, BayesianRidge
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score
)

from config import RANDOM_STATE


def build_model(model_name: str, task_type: str = "classification", **kwargs):
    """
    Instantiate the specified model family configured for classification or regression.
    """
    rs = kwargs.get("random_state", RANDOM_STATE)
    is_class = (task_type.lower() == "classification")
    m = model_name.strip()

    if m in ("Decision Tree", "CART"):
        max_d = kwargs.get("max_depth", 4)
        min_samples_split = kwargs.get("min_samples_split", 2)
        if is_class:
            return DecisionTreeClassifier(max_depth=max_d, min_samples_split=min_samples_split, random_state=rs)
        return DecisionTreeRegressor(max_depth=max_d, min_samples_split=min_samples_split, random_state=rs)

    elif m in ("Random Forest", "RandomForest"):
        n_est = kwargs.get("n_estimators", 50)
        max_d = kwargs.get("max_depth", 4)
        if is_class:
            return RandomForestClassifier(n_estimators=n_est, max_depth=max_d, random_state=rs, n_jobs=-1)
        return RandomForestRegressor(n_estimators=n_est, max_depth=max_d, random_state=rs, n_jobs=-1)

    elif m in ("Support Vector Machine", "SVM", "SVC", "SVR"):
        kernel = kwargs.get("kernel", "linear")
        C = kwargs.get("C", 1.0)
        if is_class:
            return SVC(kernel=kernel, C=C, random_state=rs)
        return SVR(kernel=kernel, C=C)

    elif m in ("K-Nearest Neighbors", "KNN"):
        k = kwargs.get("n_neighbors", kwargs.get("k", 3))
        weights = kwargs.get("weights", "uniform")
        if is_class:
            return KNeighborsClassifier(n_neighbors=k, weights=weights, n_jobs=-1)
        return KNeighborsRegressor(n_neighbors=k, weights=weights, n_jobs=-1)

    elif m in ("Logistic Regression", "Logit"):
        C = kwargs.get("C", 1.0)
        max_iter = kwargs.get("max_iter", 1000)
        if is_class:
            return LogisticRegression(C=C, max_iter=max_iter, random_state=rs)
        return LinearRegression()

    elif m in ("Linear Regression", "OLS", "Ridge"):
        if is_class:
            # Classification analog: LogisticRegression
            return LogisticRegression(random_state=rs, max_iter=1000)
        alpha = kwargs.get("alpha", 1.0)
        if kwargs.get("use_ridge", False):
            return Ridge(alpha=alpha, random_state=rs)
        return LinearRegression()

    elif m in ("Naive Bayes", "GaussianNB"):
        if is_class:
            return GaussianNB()
        # Generative linear Bayesian regression equivalent
        return BayesianRidge()

    elif m in ("Perceptron", "Linear Perceptron"):
        if is_class:
            max_iter = kwargs.get("max_iter", 1000)
            tol = kwargs.get("tol", 1e-3)
            return SkPerceptron(max_iter=max_iter, tol=tol, random_state=rs)
        # Regression analog: linear SGD with squared loss
        return SGDRegressor(loss="squared_error", max_iter=1000, random_state=rs)

    elif m == "MLP (Multilayer Perceptron)":
        hl = kwargs.get("hidden_layer_sizes", (100,))
        max_iter = kwargs.get("max_iter", 500)
        if is_class:
            return MLPClassifier(hidden_layer_sizes=hl, max_iter=max_iter, random_state=rs, early_stopping=True)
        return MLPRegressor(hidden_layer_sizes=hl, max_iter=max_iter, random_state=rs, early_stopping=True)

    elif m == "AdaBoost":
        n_est = kwargs.get("n_estimators", 50)
        base = DecisionTreeClassifier(max_depth=1, random_state=rs) if is_class else DecisionTreeRegressor(max_depth=1, random_state=rs)
        if is_class:
            return AdaBoostClassifier(estimator=base, n_estimators=n_est, random_state=rs, algorithm="SAMME")
        return AdaBoostRegressor(estimator=base, n_estimators=n_est, random_state=rs)

    else:
        raise ValueError(f"Unknown model name: '{model_name}' for task: '{task_type}'")


def train_and_evaluate_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_type: str = "classification",
    **kwargs
) -> Dict[str, Any]:
    """
    Train model and evaluate on both training and test partitions.
    Returns performance metrics, error, and fit timing.
    """
    model = build_model(model_name, task_type=task_type, **kwargs)

    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    fit_duration = time.perf_counter() - start_time

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    is_class = (task_type.lower() == "classification")

    if is_class:
        if np.asarray(y_train_pred).dtype.kind in ('f', 'c'):
            y_train_pred = (np.asarray(y_train_pred) >= 0.5).astype(int)
        if np.asarray(y_test_pred).dtype.kind in ('f', 'c'):
            y_test_pred = (np.asarray(y_test_pred) >= 0.5).astype(int)

        train_acc = float(accuracy_score(y_train, y_train_pred))
        test_acc = float(accuracy_score(y_test, y_test_pred))
        train_err = float(1.0 - train_acc)
        test_err = float(1.0 - test_acc)

        # F1, precision, recall (macro averaged across classes)
        test_f1 = float(f1_score(y_test, y_test_pred, average="macro", zero_division=0))
        test_prec = float(precision_score(y_test, y_test_pred, average="macro", zero_division=0))
        test_rec = float(recall_score(y_test, y_test_pred, average="macro", zero_division=0))

        res = {
            "model_name": model_name,
            "task_type": "classification",
            "fit_time_seconds": fit_duration,
            "train_score": train_acc,
            "val_score": test_acc,
            "train_error": train_err,
            "val_error": test_err,
            "accuracy": test_acc,
            "f1_score": test_f1,
            "precision": test_prec,
            "recall": test_rec,
            "generalization_gap": float(test_err - train_err),
            "estimator": model,
        }
    else:
        # Regression metrics
        train_mse = float(mean_squared_error(y_train, y_train_pred))
        test_mse = float(mean_squared_error(y_test, y_test_pred))
        train_rmse = float(np.sqrt(train_mse))
        test_rmse = float(np.sqrt(test_mse))
        test_mae = float(mean_absolute_error(y_test, y_test_pred))
        train_r2 = float(r2_score(y_train, y_train_pred))
        test_r2 = float(r2_score(y_test, y_test_pred))

        # Normalized bounded error proxy for learning curve: 1 - max(0, R^2)
        norm_train_err = float(max(0.0, 1.0 - max(0.0, train_r2)))
        norm_val_err = float(max(0.0, 1.0 - max(0.0, test_r2)))

        res = {
            "model_name": model_name,
            "task_type": "regression",
            "fit_time_seconds": fit_duration,
            "train_score": train_r2,
            "val_score": test_r2,
            "train_error": norm_train_err,
            "val_error": norm_val_err,
            "rmse": test_rmse,
            "mae": test_mae,
            "r2_score": test_r2,
            "generalization_gap": float(norm_val_err - norm_train_err),
            "estimator": model,
        }

    return res
