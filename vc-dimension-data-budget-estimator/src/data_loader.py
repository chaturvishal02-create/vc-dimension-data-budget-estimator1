# src/data_loader.py
# ─────────────────────────────────────────────────────────────────
# Universal dataset loader supporting CSV, Excel (.xlsx, .xls),
# JSON (.json), TXT (tabular), and Parquet (.parquet).
# Handles both classification and regression datasets without leakage.
# ─────────────────────────────────────────────────────────────────

import io
import json
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_iris, load_breast_cancer, load_wine,
    make_classification, make_moons, fetch_california_housing
)
from sklearn.preprocessing import LabelEncoder


# ── Built-in benchmark dataset loaders ───────────────────────────

def load_iris_dataset() -> dict:
    """Load the standard 3-class Fisher Iris dataset."""
    ds = load_iris(as_frame=True)
    return {
        "X": ds.data,
        "y": ds.target,
        "feature_names": list(ds.feature_names),
        "target_name": "species",
        "dataset_name": "Iris",
        "n_samples": ds.data.shape[0],
        "n_features": ds.data.shape[1],
        "task_type": "classification",
        "is_custom": False,
    }


def load_breast_cancer_dataset() -> dict:
    """Load the binary Breast Cancer Wisconsin dataset."""
    ds = load_breast_cancer(as_frame=True)
    return {
        "X": ds.data,
        "y": ds.target,
        "feature_names": list(ds.feature_names),
        "target_name": "diagnosis",
        "dataset_name": "Breast Cancer Wisconsin",
        "n_samples": ds.data.shape[0],
        "n_features": ds.data.shape[1],
        "task_type": "classification",
        "is_custom": False,
    }


def load_wine_dataset() -> dict:
    """Load the 3-class Wine recognition dataset."""
    ds = load_wine(as_frame=True)
    return {
        "X": ds.data,
        "y": ds.target,
        "feature_names": list(ds.feature_names),
        "target_name": "wine_cultivar",
        "dataset_name": "Wine Classification",
        "n_samples": ds.data.shape[0],
        "n_features": ds.data.shape[1],
        "task_type": "classification",
        "is_custom": False,
    }


def load_synthetic_linear_dataset(n_samples: int = 300, random_state: int = 42) -> dict:
    """Generate a 2D linearly separable binary dataset (optimal for Perceptron)."""
    X_arr, y_arr = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=2.0,
        random_state=random_state,
    )
    feature_names = ["feature_x1", "feature_x2"]
    X = pd.DataFrame(X_arr, columns=feature_names)
    y = pd.Series(y_arr, name="linear_class")
    return {
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "target_name": "linear_class",
        "dataset_name": "Synthetic Linearly Separable",
        "n_samples": n_samples,
        "n_features": 2,
        "task_type": "classification",
        "is_custom": False,
    }


def load_synthetic_nonlinear_dataset(n_samples: int = 300, noise: float = 0.15, random_state: int = 42) -> dict:
    """Generate a 2D two-interleaving-moons binary dataset (non-linear)."""
    X_arr, y_arr = make_moons(
        n_samples=n_samples,
        noise=noise,
        random_state=random_state,
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
        "n_samples": n_samples,
        "n_features": 2,
        "task_type": "classification",
        "is_custom": False,
    }


# ── Universal File Reader (CSV, Excel, JSON, TXT, Parquet) ────────

def read_uploaded_file(uploaded_file, filename: Optional[str] = None) -> pd.DataFrame:
    """
    Read an uploaded dataset supporting:
    - CSV (.csv)
    - Excel (.xlsx, .xls)
    - JSON (.json)
    - Tabular Text (.txt with auto-detected delimiter: comma, tab, whitespace, pipe)
    - Apache Parquet (.parquet)

    Accepts a Streamlit UploadedFile, file-like object, bytes, or file path.
    """
    fname = filename or getattr(uploaded_file, "name", "")
    fname_lower = fname.lower()

    if hasattr(uploaded_file, "seek"):
        try:
            uploaded_file.seek(0)
        except Exception:
            pass

    try:
        if fname_lower.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        elif fname_lower.endswith(".parquet"):
            df = pd.read_parquet(uploaded_file)
        elif fname_lower.endswith(".json"):
            try:
                df = pd.read_json(uploaded_file)
            except Exception:
                if hasattr(uploaded_file, "seek"):
                    uploaded_file.seek(0)
                raw_json = json.load(uploaded_file)
                if isinstance(raw_json, list):
                    df = pd.json_normalize(raw_json)
                elif isinstance(raw_json, dict):
                    try:
                        df = pd.DataFrame(raw_json)
                    except Exception:
                        df = pd.json_normalize(raw_json)
                else:
                    raise ValueError("JSON must represent an array of records or a table dictionary.")
        elif fname_lower.endswith(".txt"):
            # Tabular text: attempt auto-detection of separator
            try:
                df = pd.read_csv(uploaded_file, sep=None, engine="python")
            except Exception:
                if hasattr(uploaded_file, "seek"):
                    uploaded_file.seek(0)
                # Fallback to tab
                try:
                    df = pd.read_csv(uploaded_file, sep="\t")
                except Exception:
                    if hasattr(uploaded_file, "seek"):
                        uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file)
        else:
            # Default fallback: CSV
            df = pd.read_csv(uploaded_file)
    except Exception as exc:
        raise ValueError(f"Could not parse uploaded dataset '{fname}': {exc}") from exc

    if df is None or df.empty:
        raise ValueError("The uploaded file is empty.")

    # Clean whitespace in column names
    df.columns = [str(c).strip() for c in df.columns]
    return sanitize_dataframe_for_arrow(df)


def sanitize_dataframe_for_arrow(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitize dataframe columns so they are strictly Arrow-compatible and Streamlit-safe.
    Converts mixed object columns (mix of bytes, numbers, strings) into clean types.
    """
    if df is None or df.empty:
        return df

    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            has_mixed_or_bytes = False
            types_seen = set()
            for val in df[col].dropna():
                types_seen.add(type(val))
                if isinstance(val, (bytes, bytearray)) or len(types_seen) > 1:
                    has_mixed_or_bytes = True
                    break
            
            if has_mixed_or_bytes:
                def _clean_val(v):
                    if pd.isna(v):
                        return v
                    if isinstance(v, (bytes, bytearray)):
                        try:
                            return v.decode("utf-8", errors="replace")
                        except Exception:
                            return str(v)
                    return str(v)
                
                df[col] = df[col].apply(_clean_val)
    return df


# ── Task Type Detection Helper ────────────────────────────────────

def detect_task_type(y_series: pd.Series) -> str:
    """
    Automatically identify whether the target variable corresponds
    to a 'classification' or 'regression' problem.
    """
    # Drop NAs for detection
    y_clean = y_series.dropna()
    if y_clean.empty:
        return "classification"

    # Discrete types
    if pd.api.types.is_object_dtype(y_clean) or isinstance(y_clean.dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(y_clean):
        return "classification"

    unique_vals = y_clean.nunique()
    total_vals = len(y_clean)

    # If small number of unique values (<= 20) or very small proportion of dataset, it's classification
    if unique_vals <= 20:
        return "classification"

    if pd.api.types.is_integer_dtype(y_clean):
        if (unique_vals / total_vals) < 0.10:
            return "classification"
        return "regression"

    # Floats with many unique values -> Regression
    return "regression"


# ── Custom Dataset Preparation & Leak-Free Encoding ───────────────

def prepare_custom_dataset(
    df: pd.DataFrame,
    target_column: str,
    dataset_name: str = "Uploaded Dataset",
    task_type: Optional[str] = None,
    min_samples: int = 30,
) -> dict:
    """
    Validate, clean, encode, and prepare a custom user-uploaded dataset for ML training.
    Supports both classification and regression targets.
    Strictly separates features from target to eliminate data leakage.
    """
    if df is None or df.empty:
        raise ValueError("The uploaded dataset is empty.")

    df = sanitize_dataframe_for_arrow(df)

    if len(df.columns) < 2:
        raise ValueError(
            f"The dataset must contain at least 2 columns (features + target). "
            f"Found {len(df.columns)} column(s): {list(df.columns)}"
        )

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # 1. Drop rows with missing target values (cannot train or evaluate without ground truth)
    init_rows = len(df)
    df_clean = df.dropna(subset=[target_column]).copy()
    dropped_target_rows = init_rows - len(df_clean)

    if len(df_clean) < min_samples:
        raise ValueError(
            f"Dataset has only {len(df_clean)} valid rows after dropping missing targets. "
            f"At least {min_samples} samples are required for a meaningful ML experiment and learning curves."
        )

    # 2. Separate features and target (strictly prevent target leakage)
    y_raw = df_clean[target_column]
    X_raw = df_clean.drop(columns=[target_column])

    # 3. Determine or validate task type
    detected_task = detect_task_type(y_raw) if task_type is None else task_type.lower()
    final_task_type = detected_task

    target_classes = []
    n_classes = 0

    if final_task_type == "classification":
        unique_classes = y_raw.unique()
        n_classes = len(unique_classes)
        if n_classes < 2:
            raise ValueError(
                f"Target column '{target_column}' contains only {n_classes} class: {list(unique_classes)}. "
                f"Classification requires at least 2 distinct classes."
            )

        # Encode target labels to consecutive integer labels [0, ..., K-1]
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_raw.astype(str))
        y = pd.Series(y_encoded, name=target_column, index=df_clean.index)
        target_classes = [str(c) for c in le.classes_]
    else:
        # Regression target: convert to float numeric values
        try:
            y = pd.to_numeric(y_raw, errors="coerce")
            if y.isnull().any():
                valid_mask = y.notnull()
                df_clean = df_clean[valid_mask]
                y_raw = y_raw[valid_mask]
                X_raw = X_raw[valid_mask]
                y = y[valid_mask]
            y = pd.Series(y.astype(float), name=target_column, index=df_clean.index)
        except Exception as exc:
            raise ValueError(f"Could not convert regression target '{target_column}' to numerical values: {exc}")

    # 4. Process features: identify numeric vs categorical
    all_null_cols = [c for c in X_raw.columns if X_raw[c].isnull().all()]
    if all_null_cols:
        X_raw = X_raw.drop(columns=all_null_cols)

    numeric_cols = list(X_raw.select_dtypes(include=[np.number]).columns)
    cat_cols = [c for c in X_raw.columns if c not in numeric_cols]

    X_processed_parts = []

    if numeric_cols:
        X_num = X_raw[numeric_cols].copy()
        X_processed_parts.append(X_num)

    if cat_cols:
        X_cat = X_raw[cat_cols].copy()
        for c in cat_cols:
            mode_val = X_cat[c].mode()
            fill_val = mode_val.iloc[0] if not mode_val.empty else "Missing"
            X_cat[c] = X_cat[c].fillna(fill_val).astype(str)

        # One-hot encode categorical features with drop_first=True to avoid multicollinearity
        X_cat_enc = pd.get_dummies(X_cat, prefix=cat_cols, drop_first=True, dtype=float)
        X_processed_parts.append(X_cat_enc)

    if not X_processed_parts:
        raise ValueError("No usable feature columns found in the dataset.")

    X = pd.concat(X_processed_parts, axis=1)

    if X.shape[1] == 0:
        raise ValueError("No usable feature columns found after encoding.")

    return {
        "X": X,
        "y": y,
        "raw_df": df,
        "raw_X": X_raw,
        "raw_y": y_raw,
        "feature_names": list(X.columns),
        "raw_feature_names": list(df.columns),
        "categorical_columns": cat_cols,
        "numeric_columns": numeric_cols,
        "target_name": target_column,
        "target_classes": target_classes,
        "task_type": final_task_type,
        "dataset_name": dataset_name,
        "n_samples": len(X),
        "n_features": X.shape[1],
        "dropped_rows": dropped_target_rows,
        "is_custom": True,
        "n_classes": n_classes if final_task_type == "classification" else None,
    }


def load_custom_dataset(
    uploaded_file,
    target_column: str,
    dataset_name: str = "",
    task_type: Optional[str] = None
) -> dict:
    """
    Parse an uploaded file (CSV, Excel, JSON, TXT, Parquet), clean, and return unified dataset dictionary.
    """
    fname = getattr(uploaded_file, "name", "Uploaded Dataset")
    ds_name = dataset_name or f"Custom ({fname})"
    df = read_uploaded_file(uploaded_file, filename=fname)
    return prepare_custom_dataset(df, target_column, dataset_name=ds_name, task_type=task_type)


def load_csv_dataset(uploaded_file, target_column: str) -> dict:
    """Backward compatible wrapper around load_custom_dataset."""
    return load_custom_dataset(uploaded_file, target_column, dataset_name="Uploaded CSV")


# ── Unified Dispatcher ───────────────────────────────────────────

def get_dataset(
    name: str,
    uploaded_file=None,
    target_column: str = "",
    task_type: Optional[str] = None
) -> dict:
    """
    Dispatcher: return the correct dataset dict by name.
    Supported built-ins: 'Iris', 'Breast Cancer Wisconsin', 'Wine Classification',
    'Synthetic Linearly Separable', 'Synthetic Non-Linearly Separable'.
    Supported custom options: 'Custom Upload (CSV / Excel / JSON / Parquet)', 'CSV Upload', etc.
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
    elif any(k in name for k in ("Upload", "Custom", "CSV")):
        if uploaded_file is None:
            raise ValueError("Please upload a dataset file (CSV, Excel, JSON, TXT, or Parquet).")
        if not target_column:
            raise ValueError("Please select a target column.")
        return load_custom_dataset(
            uploaded_file,
            target_column,
            dataset_name="Uploaded Dataset",
            task_type=task_type,
        )
    else:
        raise ValueError(f"Unknown dataset name: '{name}'")


# ── Descriptive statistics helper ────────────────────────────────

def describe_dataset(ds: dict) -> dict:
    """
    Return a comprehensive dict of statistics about the dataset.
    """
    X, y = ds["X"], ds["y"]
    missing = int(X.isnull().sum().sum())

    if "raw_df" in ds and isinstance(ds["raw_df"], pd.DataFrame):
        duplicates = int(ds["raw_df"].duplicated().sum())
    else:
        duplicates = int(X.duplicated().sum())

    task_type = ds.get("task_type", "classification")

    if task_type == "classification":
        class_counts = y.value_counts().to_dict()
        total_samples = len(y)
        class_proportions = {
            cls: (count / total_samples) * 100
            for cls, count in class_counts.items()
        }
        n_classes = len(class_counts)
    else:
        class_counts = {}
        class_proportions = {}
        n_classes = 0

    return {
        "n_samples": ds["n_samples"],
        "n_features": ds["n_features"],
        "task_type": task_type,
        "missing_values": missing,
        "duplicate_rows": duplicates,
        "class_distribution": class_counts,
        "class_proportions": class_proportions,
        "n_classes": n_classes,
        "feature_names": ds["feature_names"],
        "target_name": ds["target_name"],
        "target_classes": ds.get("target_classes", [str(c) for c in class_counts.keys()]),
        "numeric_columns": ds.get("numeric_columns", []),
        "categorical_columns": ds.get("categorical_columns", []),
        "is_custom": ds.get("is_custom", False),
    }
