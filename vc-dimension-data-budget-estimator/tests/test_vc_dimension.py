# tests/test_vc_dimension.py
# ─────────────────────────────────────────────────────────────────
# Unit tests for src/vc_dimension.py
# Run with:  py -m pytest tests/test_vc_dimension.py -v
# ─────────────────────────────────────────────────────────────────

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.vc_dimension import (vc_perceptron, vc_mlp, vc_adaboost,
                               estimate_vc_dimension)


class TestVCPerceptron:
    def test_d_equals_n_plus_1(self):
        assert vc_perceptron(4)["vc_dim"]  == 5
        assert vc_perceptron(10)["vc_dim"] == 11
        assert vc_perceptron(1)["vc_dim"]  == 2

    def test_label_is_exact(self):
        assert "Exact" in vc_perceptron(4)["label"]

    def test_zero_features_raises(self):
        with pytest.raises(ValueError):
            vc_perceptron(0)

    def test_negative_features_raises(self):
        with pytest.raises(ValueError):
            vc_perceptron(-1)


class TestVCMLP:
    def test_returns_positive_int(self):
        r = vc_mlp(4, (100,))
        assert isinstance(r["vc_dim"], int)
        assert r["vc_dim"] > 0

    def test_label_is_approximate(self):
        r = vc_mlp(4, (100,))
        assert "Approximate" in r["label"] or "Illustrative" in r["label"]

    def test_larger_hidden_gives_larger_d(self):
        d_small = vc_mlp(4, (10,))["vc_dim"]
        d_large = vc_mlp(4, (200, 100))["vc_dim"]
        assert d_large > d_small

    def test_zero_features_raises(self):
        with pytest.raises(ValueError):
            vc_mlp(0)


class TestVCAdaBoost:
    def test_returns_positive_int(self):
        r = vc_adaboost(4, n_estimators=50)
        assert isinstance(r["vc_dim"], int)
        assert r["vc_dim"] > 0

    def test_label_is_approximate(self):
        r = vc_adaboost(4, n_estimators=50)
        assert "Approximate" in r["label"] or "Illustrative" in r["label"]

    def test_more_estimators_gives_larger_d(self):
        d1 = vc_adaboost(4, n_estimators=10)["vc_dim"]
        d2 = vc_adaboost(4, n_estimators=100)["vc_dim"]
        assert d2 > d1

    def test_zero_features_raises(self):
        with pytest.raises(ValueError):
            vc_adaboost(0)

    def test_zero_estimators_raises(self):
        with pytest.raises(ValueError):
            vc_adaboost(4, n_estimators=0)


class TestEstimateVCDimension:
    def test_perceptron_dispatch(self):
        r = estimate_vc_dimension("Perceptron", 4)
        assert r["vc_dim"] == 5

    def test_mlp_dispatch(self):
        r = estimate_vc_dimension("MLP (Multilayer Perceptron)", 4,
                                   hidden_layer_sizes=(50,))
        assert r["vc_dim"] > 0

    def test_adaboost_dispatch(self):
        r = estimate_vc_dimension("AdaBoost", 4, n_estimators=50)
        assert r["vc_dim"] > 0

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError):
            estimate_vc_dimension("RandomForest", 4)
