# tests/test_report_generator.py
import os
import pytest
import matplotlib.pyplot as plt
from src.report_generator import figure_to_base64, generate_html_report, save_report_to_disk


def test_figure_to_base64():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6])
    uri = figure_to_base64(fig)
    plt.close(fig)

    assert isinstance(uri, str)
    assert uri.startswith("data:image/png;base64,")
    assert len(uri) > 50


def test_generate_html_report_valid():
    results = [
        {
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
        }
    ]

    html = generate_html_report(
        results=results,
        dataset_name="Iris Test",
        dataset_info={"n_samples": 150, "n_features": 4, "n_classes": 3},
        epsilon=0.10,
        delta=0.05,
        bound_setting="realizable",
    )

    assert "<!DOCTYPE html>" in html
    assert "Iris Test" in html
    assert "Perceptron" in html
    assert "3,022" in html
    assert "48" in html
    assert "0.10" in html
    assert "@media print" in html
    assert "window.print()" in html


def test_generate_html_report_with_figures():
    fig, ax = plt.subplots()
    ax.bar(["Perceptron"], [48])
    figures = {"Comparison Chart": fig}

    html = generate_html_report(
        results=[],
        dataset_name="Synthetic",
        figures=figures,
    )
    plt.close(fig)

    assert "Comparison Chart</h3>" in html
    assert 'data:image/png;base64,' in html


def test_generate_html_report_empty_results():
    html = generate_html_report(
        results=[],
        dataset_name="Empty Experiment",
    )
    assert "<!DOCTYPE html>" in html
    assert "No experiment results recorded." in html


def test_save_report_to_disk(tmp_path):
    out_dir = str(tmp_path / "test_reports")
    html_content = "<html><body>Test Report</body></html>"
    saved_path = save_report_to_disk(html_content, output_dir=out_dir, filename="test_rep.html")

    assert os.path.exists(saved_path)
    with open(saved_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert content == html_content
