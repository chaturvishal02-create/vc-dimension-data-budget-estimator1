# tests/test_pac_bound.py
# ─────────────────────────────────────────────────────────────────
# Unit tests for src/pac_bound.py
# Run with:  py -m pytest tests/test_pac_bound.py -v
# ─────────────────────────────────────────────────────────────────

import math
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pac_bound import calculate_theoretical_samples


class TestCalculateTheoreticalSamples:
    """Tests for the PAC/VC sample-complexity formula."""

    def test_basic_output_keys(self):
        result = calculate_theoretical_samples(0.1, 0.05, 5)
        assert "m"           in result
        assert "formula"     in result
        assert "substituted" in result
        assert "reference"   in result
        assert "note"        in result

    def test_result_is_positive_integer(self):
        result = calculate_theoretical_samples(0.1, 0.05, 5)
        assert isinstance(result["m"], int)
        assert result["m"] > 0

    def test_result_is_ceil(self):
        result = calculate_theoretical_samples(0.1, 0.05, 5)
        assert result["m"] == math.ceil(result["m_raw"])

    def test_larger_d_gives_larger_m(self):
        m1 = calculate_theoretical_samples(0.1, 0.05, 5)["m"]
        m2 = calculate_theoretical_samples(0.1, 0.05, 50)["m"]
        assert m2 > m1

    def test_smaller_epsilon_gives_larger_m(self):
        m1 = calculate_theoretical_samples(0.2, 0.05, 10)["m"]
        m2 = calculate_theoretical_samples(0.05, 0.05, 10)["m"]
        assert m2 > m1

    def test_smaller_delta_gives_larger_m(self):
        m1 = calculate_theoretical_samples(0.1, 0.1, 10)["m"]
        m2 = calculate_theoretical_samples(0.1, 0.01, 10)["m"]
        assert m2 > m1

    # ── Invalid-input tests ───────────────────────────────────────

    def test_epsilon_zero_raises(self):
        with pytest.raises(ValueError, match="epsilon"):
            calculate_theoretical_samples(0.0, 0.05, 5)

    def test_epsilon_one_raises(self):
        with pytest.raises(ValueError, match="epsilon"):
            calculate_theoretical_samples(1.0, 0.05, 5)

    def test_epsilon_negative_raises(self):
        with pytest.raises(ValueError, match="epsilon"):
            calculate_theoretical_samples(-0.1, 0.05, 5)

    def test_delta_zero_raises(self):
        with pytest.raises(ValueError, match="delta"):
            calculate_theoretical_samples(0.1, 0.0, 5)

    def test_delta_one_raises(self):
        with pytest.raises(ValueError, match="delta"):
            calculate_theoretical_samples(0.1, 1.0, 5)

    def test_delta_negative_raises(self):
        with pytest.raises(ValueError, match="delta"):
            calculate_theoretical_samples(0.1, -0.05, 5)

    def test_vc_dim_zero_raises(self):
        with pytest.raises(ValueError, match="vc_dimension"):
            calculate_theoretical_samples(0.1, 0.05, 0)

    def test_vc_dim_negative_raises(self):
        with pytest.raises(ValueError, match="vc_dimension"):
            calculate_theoretical_samples(0.1, 0.05, -1)

    def test_formula_constants_unchanged(self):
        """
        Verify the formula uses the exact constants specified:
        (1/epsilon) * [4 * log2(2/delta) + 8 * d * log2(13/epsilon)]
        """
        epsilon, delta, d = 0.1, 0.05, 5
        term1 = 4.0 * math.log2(2.0 / delta)
        term2 = 8.0 * d * math.log2(13.0 / epsilon)
        expected_raw = (1.0 / epsilon) * (term1 + term2)
        expected_m   = math.ceil(expected_raw)

        result = calculate_theoretical_samples(epsilon, delta, d)
        assert result["m"] == expected_m, (
            f"Formula mismatch: expected {expected_m}, got {result['m']}"
        )

    def test_reference_contains_blumer(self):
        result = calculate_theoretical_samples(0.1, 0.05, 5)
        assert "Blumer" in result["reference"]
