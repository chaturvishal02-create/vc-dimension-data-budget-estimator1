# src/learning_curve.py
# ─────────────────────────────────────────────────────────────────
# Empirical Learning Curve Generator for both Classification and
# Regression models with multi-seed variance and error bars.
# ─────────────────────────────────────────────────────────────────

from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score
)

from config import TRAIN_FRACTIONS, RANDOM_STATE
from src.model_training import build_model


def _stratified_subset(y: np.ndarray, n: int, seed: int = RANDOM_STATE) -> np.ndarray:
    """
    Return indices of a stratified subset of size `n` from `y`.
    Falls back to a random subset if stratification fails or for regression.
    """
    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)

    if n_classes < 2 or n < n_classes:
        rng = np.random.default_rng(seed)
        return rng.choice(len(y), size=min(n, len(y)), replace=False)

    # Proportional allocation per class
    proportions = counts / counts.sum()
    allocations  = np.floor(proportions * n).astype(int)
    deficit      = n - allocations.sum()
    order = np.argsort(-proportions)
    for i in range(int(deficit)):
        allocations[order[i % n_classes]] += 1

    rng = np.random.default_rng(seed)
    selected = []
    for cls, alloc in zip(classes, allocations):
        cls_idx = np.where(y == cls)[0]
        if len(cls_idx) == 0:
            continue
        chosen = rng.choice(cls_idx, size=min(alloc, len(cls_idx)), replace=False)
        selected.extend(chosen.tolist())

    # In case rounding missed any samples
    if len(selected) < n:
        remaining = [i for i in range(len(y)) if i not in selected]
        if remaining:
            fill_count = min(n - len(selected), len(remaining))
            selected.extend(rng.choice(remaining, size=fill_count, replace=False).tolist())

    return np.array(selected)


def generate_learning_curve(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    fractions: Optional[List[float]] = None,
    n_seeds: int = 1,
    task_type: str = "classification",
    **model_kwargs,
) -> Dict[str, Any]:
    """
    Train the model on growing subsets of the training data, recording
    training and validation performance across fractions.

    Returns
    -------
    dict with keys:
        sample_counts       : list[int]
        train_errors        : list[float] (mean)
        val_errors          : list[float] (mean)
        train_accuracies    : list[float] (mean score: acc or R²)
        val_accuracies      : list[float] (mean score: acc or R²)
        train_errors_std    : list[float] (std dev)
        val_errors_std      : list[float] (std dev)
        train_accuracies_std: list[float] (std dev)
        val_accuracies_std  : list[float] (std dev)
        generalization_gaps : list[float]
        f1_scores           : list[float] (classification)
        precision_scores    : list[float] (classification)
        recall_scores       : list[float] (classification)
        rmse_scores         : list[float] (regression)
        r2_scores           : list[float] (regression)
        model_name          : str
        task_type           : str
    """
    if fractions is None:
        fractions = TRAIN_FRACTIONS

    n_train = len(X_train)
    sample_counts        = []
    train_errors         = []
    val_errors           = []
    train_accuracies     = []
    val_accuracies       = []
    train_errors_std     = []
    val_errors_std       = []
    train_accuracies_std = []
    val_accuracies_std   = []
    generalization_gaps  = []

    f1_scores            = []
    prec_scores          = []
    rec_scores           = []
    rmse_scores          = []
    r2_scores            = []

    base_rs = model_kwargs.pop("random_state", RANDOM_STATE)
    is_class = (task_type.lower() == "classification")

    for frac in fractions:
        n = max(int(frac * n_train), 2)
        n = min(n, n_train)

        seed_tr_err = []
        seed_va_err = []
        seed_tr_acc = []
        seed_va_acc = []
        seed_f1     = []
        seed_prec   = []
        seed_rec    = []
        seed_rmse   = []
        seed_r2     = []

        for s_idx in range(max(1, n_seeds)):
            seed = base_rs + s_idx

            if is_class:
                idx = _stratified_subset(y_train, n, seed=seed)
                # Check for at least 2 distinct classes in subset
                if len(np.unique(y_train[idx])) < 2:
                    continue
            else:
                rng = np.random.default_rng(seed)
                idx = rng.choice(n_train, size=n, replace=False)

            X_sub = X_train[idx]
            y_sub = y_train[idx]

            try:
                model = build_model(model_name, task_type=task_type, random_state=seed, **model_kwargs)
                model.fit(X_sub, y_sub)

                y_tr_pred = model.predict(X_sub)
                y_va_pred = model.predict(X_test)

                if is_class:
                    tr_acc = float(accuracy_score(y_sub, y_tr_pred))
                    va_acc = float(accuracy_score(y_test, y_va_pred))
                    tr_err = float(1.0 - tr_acc)
                    va_err = float(1.0 - va_acc)

                    f1 = float(f1_score(y_test, y_va_pred, average="macro", zero_division=0))
                    prec = float(precision_score(y_test, y_va_pred, average="macro", zero_division=0))
                    rec = float(recall_score(y_test, y_va_pred, average="macro", zero_division=0))

                    seed_tr_acc.append(tr_acc)
                    seed_va_acc.append(va_acc)
                    seed_tr_err.append(tr_err)
                    seed_va_err.append(va_err)
                    seed_f1.append(f1)
                    seed_prec.append(prec)
                    seed_rec.append(rec)
                else:
                    tr_r2 = float(r2_score(y_sub, y_tr_pred))
                    va_r2 = float(r2_score(y_test, y_va_pred))
                    tr_err = float(max(0.0, 1.0 - max(0.0, tr_r2)))
                    va_err = float(max(0.0, 1.0 - max(0.0, va_r2)))

                    rmse = float(np.sqrt(mean_squared_error(y_test, y_va_pred)))

                    seed_tr_acc.append(tr_r2)
                    seed_va_acc.append(va_r2)
                    seed_tr_err.append(tr_err)
                    seed_va_err.append(va_err)
                    seed_rmse.append(rmse)
                    seed_r2.append(va_r2)

            except Exception:
                continue

        if seed_va_err:
            m_tr_err = float(np.mean(seed_tr_err))
            m_va_err = float(np.mean(seed_va_err))

            sample_counts.append(n)
            train_errors.append(m_tr_err)
            val_errors.append(m_va_err)
            train_accuracies.append(float(np.mean(seed_tr_acc)))
            val_accuracies.append(float(np.mean(seed_va_acc)))

            train_errors_std.append(float(np.std(seed_tr_err)))
            val_errors_std.append(float(np.std(seed_va_err)))
            train_accuracies_std.append(float(np.std(seed_tr_acc)))
            val_accuracies_std.append(float(np.std(seed_va_acc)))

            generalization_gaps.append(float(m_va_err - m_tr_err))

            if is_class:
                f1_scores.append(float(np.mean(seed_f1)))
                prec_scores.append(float(np.mean(seed_prec)))
                rec_scores.append(float(np.mean(seed_rec)))
            else:
                rmse_scores.append(float(np.mean(seed_rmse)))
                r2_scores.append(float(np.mean(seed_r2)))

    return {
        "sample_counts":        sample_counts,
        "train_errors":         train_errors,
        "val_errors":           val_errors,
        "train_accuracies":     train_accuracies,
        "val_accuracies":       val_accuracies,
        "train_errors_std":     train_errors_std,
        "val_errors_std":       val_errors_std,
        "train_accuracies_std": train_accuracies_std,
        "val_accuracies_std":   val_accuracies_std,
        "generalization_gaps":  generalization_gaps,
        "f1_scores":            f1_scores,
        "precision_scores":     prec_scores,
        "recall_scores":        rec_scores,
        "rmse_scores":          rmse_scores,
        "r2_scores":            r2_scores,
        "model_name":           model_name,
        "task_type":            task_type,
    }


def find_empirical_sample_requirement(
    lc_result: Dict[str, Any],
    target_error: float,
) -> Dict[str, Any]:
    """
    Return the smallest training-set size at which validation error first dropped
    to or below target_error.
    """
    for n, err in zip(lc_result.get("sample_counts", []), lc_result.get("val_errors", [])):
        if err <= target_error:
            return {
                "achieved": True,
                "sample_count": n,
                "val_error": err,
                "note": (
                    f"Target validation error ≤ {target_error:.4f} first achieved "
                    f"with {n:,} training samples (observed error = {err:.4f})."
                ),
            }

    max_tested = max(lc_result.get("sample_counts", [0]), default=0)
    return {
        "achieved": False,
        "sample_count": None,
        "val_error": None,
        "note": (
            f"Target validation error ≤ {target_error:.4f} was NOT achieved "
            f"within the tested range (maximum training size = {max_tested:,})."
        ),
    }
