# tests/test_enhancements.py
import io
import json
import pytest
import numpy as np
import pandas as pd
from src.data_loader import (
    get_dataset, describe_dataset, read_uploaded_file,
    prepare_custom_dataset, load_custom_dataset, load_csv_dataset,
)
from src.pac_bound import calculate_agnostic_samples, get_pac_bounds
from src.learning_curve import generate_learning_curve
from src.preprocessing import split_and_preprocess

def test_load_wine_dataset():
    ds = get_dataset("Wine Classification")
    assert ds["dataset_name"] == "Wine Classification"
    assert ds["n_samples"] == 178
    assert ds["n_features"] == 13
    assert len(ds["feature_names"]) == 13

def test_load_synthetic_linear_dataset():
    ds = get_dataset("Synthetic Linearly Separable")
    assert ds["dataset_name"] == "Synthetic Linearly Separable"
    assert ds["n_samples"] == 300
    assert ds["n_features"] == 2

def test_load_synthetic_nonlinear_dataset():
    ds = get_dataset("Synthetic Non-Linearly Separable")
    assert ds["dataset_name"] == "Synthetic Non-Linearly Separable"
    assert ds["n_samples"] == 300
    assert ds["n_features"] == 2

def test_describe_dataset():
    ds = get_dataset("Synthetic Linearly Separable")
    info = describe_dataset(ds)
    assert info["n_samples"] == 300
    assert info["n_features"] == 2
    assert "class_proportions" in info

def test_calculate_agnostic_samples():
    res = calculate_agnostic_samples(0.1, 0.05, 5)
    assert "m" in res
    assert res["m"] > 0
    # Agnostic bound should be significantly higher than realizable bound due to 1/eps^2
    assert res["setting"] == "agnostic"
    assert "Anthony" in res["reference"]

def test_agnostic_invalid_inputs():
    with pytest.raises(ValueError):
        calculate_agnostic_samples(0.0, 0.05, 5)
    with pytest.raises(ValueError):
        calculate_agnostic_samples(0.1, 0.0, 5)
    with pytest.raises(ValueError):
        calculate_agnostic_samples(0.1, 0.05, 0)

def test_learning_curve_with_variance():
    ds = get_dataset("Synthetic Linearly Separable")
    X_tr, X_te, y_tr, y_te, _ = split_and_preprocess(ds["X"], ds["y"])
    lc = generate_learning_curve("Perceptron", X_tr, y_tr, X_te, y_te, fractions=[0.5, 1.0], n_seeds=2)
    assert "train_errors_std" in lc
    assert "val_errors_std" in lc
    assert len(lc["sample_counts"]) > 0

def test_read_uploaded_file_csv():
    csv_text = "f1,f2,target\n" + "\n".join(f"{i*0.1},{i*0.2},{i%2}" for i in range(40))
    f = io.StringIO(csv_text)
    f.name = "data.csv"
    df = read_uploaded_file(f)
    assert len(df) == 40
    assert list(df.columns) == ["f1", "f2", "target"]

def test_read_uploaded_file_json():
    records = [{"x1": i, "x2": i * 1.5, "label": f"c_{i % 2}"} for i in range(35)]
    f = io.StringIO(json.dumps(records))
    f.name = "data.json"
    df = read_uploaded_file(f)
    assert len(df) == 35
    assert "x1" in df.columns
    assert "label" in df.columns

def test_read_uploaded_file_excel():
    f = io.BytesIO()
    f.name = "data.xlsx"
    df_raw = pd.DataFrame({
        "num1": np.arange(40, dtype=float),
        "num2": np.arange(40, dtype=float) * 2,
        "target": ["A" if i % 2 == 0 else "B" for i in range(40)]
    })
    df_raw.to_excel(f, index=False)
    f.seek(0)
    df_parsed = read_uploaded_file(f)
    assert len(df_parsed) == 40
    assert "num1" in df_parsed.columns

def test_read_uploaded_file_empty():
    f = io.StringIO("")
    f.name = "empty.csv"
    with pytest.raises(ValueError, match="empty"):
        read_uploaded_file(f)

def test_prepare_custom_dataset_success():
    df = pd.DataFrame({
        "num": np.random.randn(50),
        "category": ["red", "blue"] * 25,
        "target": ["yes", "no"] * 25,
    })
    ds = prepare_custom_dataset(df, target_column="target", dataset_name="ColorData")
    assert ds["dataset_name"] == "ColorData"
    assert ds["n_samples"] == 50
    assert ds["is_custom"] is True
    assert "category" in ds["categorical_columns"]
    assert "num" in ds["numeric_columns"]
    assert len(ds["target_classes"]) == 2
    assert any("category" in col for col in ds["feature_names"])

def test_prepare_custom_dataset_missing_target():
    df = pd.DataFrame({
        "num": range(40),
        "target": ["pos", "neg"] * 19 + [np.nan, None],
    })
    ds = prepare_custom_dataset(df, target_column="target")
    assert ds["n_samples"] == 38
    assert ds["dropped_rows"] == 2

def test_prepare_custom_dataset_validations():
    # Empty df
    with pytest.raises(ValueError, match="empty"):
        prepare_custom_dataset(pd.DataFrame(), "col")

    # Less than 2 columns
    with pytest.raises(ValueError, match="at least 2 columns"):
        prepare_custom_dataset(pd.DataFrame({"x": range(40)}), "x")

    # Missing target column
    with pytest.raises(ValueError, match="not found"):
        prepare_custom_dataset(pd.DataFrame({"x": range(40), "y": range(40)}), "z")

    # Less than 30 samples
    df_small = pd.DataFrame({"x": range(20), "target": [0, 1] * 10})
    with pytest.raises(ValueError, match="At least 30 samples"):
        prepare_custom_dataset(df_small, "target")

    # Single class target
    df_single = pd.DataFrame({"x": range(40), "target": ["only_class"] * 40})
    with pytest.raises(ValueError, match="requires at least 2 distinct classes"):
        prepare_custom_dataset(df_single, "target")

def test_load_custom_dataset_and_csv_wrapper():
    csv_text = "x1,x2,y\n" + "\n".join(f"{i},{i*2},{i%2}" for i in range(40))
    f = io.StringIO(csv_text)
    f.name = "upload.csv"
    ds = load_csv_dataset(f, target_column="y")
    assert ds["n_samples"] == 40
    assert ds["is_custom"] is True

    f.seek(0)
    ds2 = get_dataset("Custom Upload (CSV / Excel / JSON)", uploaded_file=f, target_column="y")
    assert ds2["n_samples"] == 40
    assert ds2["is_custom"] is True

def test_describe_dataset_custom():
    df = pd.DataFrame({
        "x": [1.0, 2.0] * 20,
        "y": ["catA", "catB"] * 20,
        "target": [0, 1] * 20,
    })
    ds = prepare_custom_dataset(df, target_column="target")
    info = describe_dataset(ds)
    assert info["is_custom"] is True
    assert info["n_samples"] == 40
    assert info["duplicate_rows"] > 0
    assert "x" in info["numeric_columns"]
    assert "y" in info["categorical_columns"]
