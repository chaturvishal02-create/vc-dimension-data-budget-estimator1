# src/preprocessing.py
# ─────────────────────────────────────────────────────────────────
# Builds scikit-learn preprocessing pipelines and performs the
# train / test split. All transformers are fit on training data
# only to prevent data leakage.
# ─────────────────────────────────────────────────────────────────

from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from config import TEST_SIZE, RANDOM_STATE


def build_preprocessor() -> Pipeline:
    """
    Return a leak-free numeric pipeline:
      1. Median imputation for missing values
      2. Standard scaling (zero mean, unit variance)
    Features are numeric after data_loader encoding.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])


def split_and_preprocess(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    task_type: str = "classification"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Pipeline]:
    """
    Split into train / test, fit the preprocessor strictly on the
    training portion, and return transformed numpy arrays.

    Parameters
    ----------
    X : pd.DataFrame
        Clean feature dataframe.
    y : pd.Series
        Target array / series.
    test_size : float
        Fraction reserved for held-out evaluation.
    random_state : int
        Seed for reproducibility.
    task_type : str
        'classification' (attempts stratified split) or 'regression'.

    Returns
    -------
    X_train_proc, X_test_proc : np.ndarray
        Preprocessed feature matrices.
    y_train, y_test : np.ndarray
        Labels or continuous targets.
    preprocessor : fitted Pipeline
    """
    is_classification = (task_type.lower() == "classification")

    if is_classification:
        # Stratified split when possible (classification)
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state,
                stratify=y,
            )
        except (ValueError, TypeError):
            # Fall back to random split if stratification fails (e.g. rare class count < 2)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state,
            )
    else:
        # Standard shuffle split for regression
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )

    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)

    # Cast to float for regression or maintain numeric types
    y_tr_arr = np.asarray(y_train)
    y_te_arr = np.asarray(y_test)

    return (
        X_train_proc,
        X_test_proc,
        y_tr_arr,
        y_te_arr,
        preprocessor,
    )
