# src/data_analyzer.py
# ─────────────────────────────────────────────────────────────────
# Comprehensive Dataset Profiler & Inspector.
# Analyzes raw DataFrames to extract dimensional, statistical,
# missingness, duplication, and target candidate information.
# ─────────────────────────────────────────────────────────────────

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from src.data_loader import detect_task_type


def analyze_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform deep static analysis on a pandas DataFrame.
    Returns metadata required for UI display and model ingestion.
    """
    if df is None or df.empty:
        return {
            "n_rows": 0,
            "n_cols": 0,
            "columns": [],
            "numeric_cols": [],
            "categorical_cols": [],
            "datetime_cols": [],
            "missing_per_col": {},
            "total_missing": 0,
            "duplicate_count": 0,
            "duplicate_pct": 0.0,
            "target_candidates": [],
            "summary_stats": pd.DataFrame(),
        }

    n_rows, n_cols = df.shape

    # Detect column data types
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    datetime_cols = list(df.select_dtypes(include=["datetime", "datetimetz"]).columns)
    categorical_cols = [c for c in df.columns if c not in numeric_cols and c not in datetime_cols]

    # Missing value statistics
    missing_series = df.isnull().sum()
    missing_per_col = missing_series.to_dict()
    missing_pct_per_col = (missing_series / max(1, n_rows) * 100).round(2).to_dict()
    total_missing = int(missing_series.sum())

    # Duplicate records
    duplicate_count = int(df.duplicated().sum())
    duplicate_pct = round((duplicate_count / max(1, n_rows)) * 100, 2)

    # Heuristic detection for probable target column candidates
    target_keywords = ["target", "label", "class", "outcome", "y", "status", "price", "churn", "diagnosis", "survived"]
    target_candidates = []

    for col in df.columns:
        col_lower = str(col).lower()
        score = 0
        if any(kw == col_lower or col_lower.endswith(f"_{kw}") or col_lower.startswith(f"{kw}_") for kw in target_keywords):
            score += 10
        # If it's the last column, convention often places target at the end
        if col == df.columns[-1]:
            score += 4
        # Binary or low cardinality categorical
        if df[col].nunique() in (2, 3):
            score += 3
        if score > 0:
            target_candidates.append((col, score))

    target_candidates.sort(key=lambda x: x[1], reverse=True)
    sorted_candidates = [c[0] for c in target_candidates]
    if not sorted_candidates and len(df.columns) > 0:
        sorted_candidates = [df.columns[-1]]

    # Statistical summary (numeric and categorical)
    summary_stats = df.describe(include="all").transpose()

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "columns": list(df.columns),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "datetime_cols": datetime_cols,
        "missing_per_col": missing_per_col,
        "missing_pct_per_col": missing_pct_per_col,
        "total_missing": total_missing,
        "duplicate_count": duplicate_count,
        "duplicate_pct": duplicate_pct,
        "target_candidates": sorted_candidates,
        "summary_stats": summary_stats,
    }


def profile_target_variable(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """
    Inspect the selected target column to determine task type,
    class distribution, or regression distribution properties.
    """
    if target_col not in df.columns:
        return {"error": f"Target '{target_col}' not in DataFrame"}

    series = df[target_col].dropna()
    task_type = detect_task_type(series)

    info = {
        "target_name": target_col,
        "task_type": task_type,
        "valid_count": len(series),
        "missing_count": int(df[target_col].isnull().sum()),
        "unique_values": int(series.nunique()),
    }

    if task_type == "classification":
        val_counts = series.value_counts()
        total = len(series)
        info["class_counts"] = val_counts.to_dict()
        info["class_proportions"] = {
            str(k): round((v / total) * 100, 2)
            for k, v in val_counts.items()
        }
        info["is_imbalanced"] = (val_counts.min() / max(1, val_counts.max())) < 0.20
    else:
        # Regression stats
        num_s = pd.to_numeric(series, errors="coerce").dropna()
        info["mean"] = float(num_s.mean()) if not num_s.empty else 0.0
        info["std"] = float(num_s.std()) if not num_s.empty else 0.0
        info["min"] = float(num_s.min()) if not num_s.empty else 0.0
        info["max"] = float(num_s.max()) if not num_s.empty else 0.0
        info["skew"] = float(num_s.skew()) if len(num_s) > 2 else 0.0

    return info

