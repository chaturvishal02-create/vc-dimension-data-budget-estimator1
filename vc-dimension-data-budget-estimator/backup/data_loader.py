# src/data_loader.py
# ─────────────────────────────────────────────────────────────────
# Loads built-in datasets and validates user-uploaded CSV files.
# Returns a unified dict with keys: X, y, feature_names,
# target_name, dataset_name, n_samples, n_features.
# ─────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, make_classification, make_moons


# ── Built-in loaders ─────────────────────────────────────────────

def load_iris_dataset() -> dict:
    """Return the Iris classification dataset."""
    ds = load_iris(as_frame=True)
    return {
        "X": ds.data,
        "y": ds.target,
        "feature_names": list(ds.feature_names),
        "target_name": "species",
        "dataset_name": "Iris",
        "n_samples": len(ds.target),
        "n_features": ds.data.shape[1],
    }


def load_breast_cancer_dataset() -> dict:
    """Return the Breast Cancer Wisconsin dataset."""
    ds = load_breast_cancer(as_frame=True)
    return {
        "X": ds.data,
        "y": ds.target,
        "feature_names": list(ds.feature_names),
        "target_name": "diagnosis",
        "dataset_name": "Breast Cancer Wisconsin",
        "n_samples": len(ds.target),
        "n_features": ds.data.shape[1],
    }


def load_wine_dataset() -> dict:
    """Return the Wine classification dataset."""
    ds = load_wine(as_frame=True)
    return {
        "X": ds.data,
        "y": ds.target,
        "feature_names": list(ds.feature_names),
        "target_name": "cultivar",
        "dataset_name": "Wine Classification",
        "n_samples": len(ds.target),
        "n_features": ds.data.shape[1],
    }


def load_synthetic_linear_dataset() -> dict:
    """Return a clean 2D linearly separable synthetic classification dataset."""
    X_arr, y_arr = make_classification(
        n_samples=300,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=1.8,
        random_state=42,
    )
    feature_names = ["feature_x1", "feature_x2"]
    X = pd.DataFrame(X_arr, columns=feature_names)
    y = pd.Series(y_arr, name="class_label")
    return {
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "target_name": "class_label",
        "dataset_name": "Synthetic Linearly Separable",
        "n_samples": 300,
        "n_features": 2,
    }


def load_synthetic_nonlinear_dataset() -> dict:
    """Return a 2D non-linearly separable two-moons synthetic dataset."""
    X_arr, y_arr = make_moons(
        n_samples=300,
        noise=0.15,
        random_state=42,
    )
    feature_names = ["feature_x1", "feature_x2"]
    X = pd.DataFrame(X_arr, columns=feature_names)
    y = pd.Series(y_arr, name="moon_class")
    return {
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "target_name": "moon_class",
        "dataset_name": "Synthetic Non-Linearly Separable",
        "n_samples": 300,
        "n_features": 2,
    }


# ── CSV uploader / validator ─────────────────────────────────────

def load_csv_dataset(uploaded_file, target_column: str) -> dict:
    """
    Parse a user-uploaded CSV, validate it, and return the
    unified dataset dict.  Raises ValueError with a user-friendly
    message on failure.
    """
    # Read
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        raise ValueError(f"Could not parse CSV file: {exc}") from exc

    # Basic shape check
    if df.empty:
        raise ValueError("The uploaded CSV file is empty.")
    if len(df.columns) < 2:
        raise ValueError("The CSV must contain at least two columns (features + target).")

    # Target column check
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Drop rows with missing target
    before = len(df)
    df = df.dropna(subset=[target_column])
    dropped = before - len(df)

    # Separate features and target
    y_raw = df[target_column]
    X_raw = df.drop(columns=[target_column])

    # Encode categorical target to integer labels
    y_series = pd.Categorical(y_raw)
    y = pd.Series(y_series.codes, name=target_column)

    # Keep only numeric / encode categorical features
    X_numeric = X_raw.select_dtypes(include=[np.number])
    X_cat = X_raw.select_dtypes(exclude=[np.number])
    if not X_cat.empty:
        X_cat_enc = pd.get_dummies(X_cat, drop_first=True)
        X = pd.concat([X_numeric, X_cat_enc], axis=1)
    else:
        X = X_numeric

    if X.shape[1] == 0:
        raise ValueError("No usable feature columns found after encoding.")

    # Minimum-samples guard
    if len(X) < 30:
        raise ValueError(
            f"Dataset has only {len(X)} samples after cleaning. "
            "Please provide at least 30 samples for a meaningful experiment."
        )

    return {
        "X": X,
        "y": y,
        "feature_names": list(X.columns),
        "target_name": target_column,
        "dataset_name": "Uploaded CSV",
        "n_samples": len(X),
        "n_features": X.shape[1],
        "dropped_rows": dropped,
    }


# ── Dispatcher ───────────────────────────────────────────────────

def get_dataset(name: str, uploaded_file=None, target_column: str = "") -> dict:
    """
    Dispatcher: return the correct dataset dict by name.
    Supported: "Iris", "Breast Cancer Wisconsin", "Wine Classification",
    "Synthetic Linearly Separable", "Synthetic Non-Linearly Separable", "CSV Upload".
    """
    if name == "Iris":
        return load_iris_dataset()
    elif name == "Breast Cancer Wisconsin":
        return load_breast_cancer_dataset()
    elif name == "Wine Classification":
        return load_wine_dataset()
    elif name == "Synthetic Linearly Separable":
        return load_synthetic_linear_dataset()
    elif name == "Synthetic Non-Linearly Separable":
        return load_synthetic_nonlinear_dataset()
    elif name == "CSV Upload":
        if uploaded_file is None:
            raise ValueError("Please upload a CSV file.")
        if not target_column:
            raise ValueError("Please select a target column.")
        return load_csv_dataset(uploaded_file, target_column)
    else:
        raise ValueError(f"Unknown dataset name: '{name}'")


# ── Descriptive statistics helper ────────────────────────────────

def describe_dataset(ds: dict) -> dict:
    """
    Return a dict of human-readable statistics about the dataset.
    """
    X, y = ds["X"], ds["y"]
    missing = int(X.isnull().sum().sum())
    class_counts = y.value_counts().to_dict()
    total_samples = len(y)
    class_proportions = {
        cls: (count / total_samples) * 100
        for cls, count in class_counts.items()
    }
    return {
        "n_samples": ds["n_samples"],
        "n_features": ds["n_features"],
        "missing_values": missing,
        "class_distribution": class_counts,
        "class_proportions": class_proportions,
        "n_classes": len(class_counts),
        "feature_names": ds["feature_names"],
        "target_name": ds["target_name"],
    }
