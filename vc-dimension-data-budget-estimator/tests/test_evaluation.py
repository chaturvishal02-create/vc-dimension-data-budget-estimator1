# tests/test_evaluation.py
import pytest
import numpy as np
import pandas as pd
from src.evaluation import build_comparison_dataframe, run_full_experiment
from src.data_loader import get_dataset
from src.preprocessing import split_and_preprocess

def test_build_comparison_dataframe_empty():
    df = build_comparison_dataframe([])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0

def test_build_comparison_dataframe_with_results():
    mock_results = [{
        "model_name": "Perceptron",
        "vc_info": {"vc_dim": 5, "label": "exact"},
        "pac_bound": {"m": 3022},
        "empirical": {"sample_count": 48},
        "prediction": {"sample_count": 50},
        "final_train_err": 0.05,
        "final_val_err": 0.08,
        "final_val_acc": 0.92,
        "epsilon": 0.10,
        "delta": 0.05,
    }]
    df = build_comparison_dataframe(mock_results)
    assert len(df) == 1
    assert df.iloc[0]["Model"] == "Perceptron"
    assert df.iloc[0]["Theoretical Bound (m)"] == 3022
    assert df.iloc[0]["Empirical Requirement (m)"] == 48
    assert "63.0×" in df.iloc[0]["Theory/Empirical Ratio"] or "x" in df.iloc[0]["Theory/Empirical Ratio"].lower()

def test_run_full_experiment_smoke():
    ds = get_dataset("Iris")
    X_tr, X_te, y_tr, y_te, _ = split_and_preprocess(ds["X"], ds["y"])
    res = run_full_experiment(
        "Perceptron", X_tr, y_tr, X_te, y_te,
        epsilon=0.10, delta=0.05, n_features=ds["n_features"]
    )
    assert res["model_name"] == "Perceptron"
    assert res["pac_bound"]["m"] > 0
    assert "sample_counts" in res["learning_curve"]
