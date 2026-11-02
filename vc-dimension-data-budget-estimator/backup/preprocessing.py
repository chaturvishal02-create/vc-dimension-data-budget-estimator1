# src/preprocessing.py
# ─────────────────────────────────────────────────────────────────
# Builds scikit-learn preprocessing pipelines and performs the
# train / test split.  All transformers are fit on training data
# only to prevent data leakage.
# ─────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from config import TEST_SIZE, RANDOM_STATE


def build_preprocessor() -> Pipeline:
    """
    Return a simple numeric pipeline:
      1. Median imputation for missing values
      2. Standard scaling (zero mean, unit variance)
    Features are assumed to be numeric after data_loader encoding.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])


def split_and_preprocess(X: pd.DataFrame, y: pd.Series,
                          test_size: float = TEST_SIZE,
                          random_state: int = RANDOM_STATE):
    """
    Split into train / test, fit the preprocessor on the training
    portion, and return transformed arrays.

    Returns
    -------
    X_train_proc, X_test_proc : np.ndarray
        Preprocessed feature matrices.
    y_train, y_test : np.ndarray
        Integer label arrays.
    preprocessor : fitted Pipeline
        Can be reused for scoring new samples.
    """
    # Stratified split when possible (classification)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
    except ValueError:
        # Fall back to random split if stratification fails
        # (e.g. class with only 1 member)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
        )

    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)

    return (
        X_train_proc,
        X_test_proc,
        np.array(y_train),
        np.array(y_test),
        preprocessor,
    )
