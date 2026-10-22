"""Stata weight-semantics tests (``[U] 20.24``)."""

from __future__ import annotations

import numpy as np
import pytest

from westata.data.weights import normalize_weights
from westata.errors import WeightError


def test_fweight_keeps_raw_weights_and_uses_sum():
    w = np.array([1.0, 2.0, 3.0])
    spec = normalize_weights(w, "fweight")
    np.testing.assert_array_equal(spec.weights, w)
    assert spec.effective_n == 6.0
    assert spec.implies_robust is False


def test_aweight_normalizes_to_n():
    w = np.array([1.0, 2.0, 3.0])
    spec = normalize_weights(w, "aweight")
    np.testing.assert_allclose(spec.weights.sum(), 3.0, rtol=1e-12)
    np.testing.assert_allclose(spec.weights, w * 3.0 / 6.0, rtol=1e-12)
    assert spec.effective_n == 3.0
    assert spec.implies_robust is False


def test_pweight_normalizes_like_aweight_but_implies_robust():
    w = np.array([1.0, 2.0, 3.0])
    aweighted = normalize_weights(w, "aweight")
    pweighted = normalize_weights(w, "pweight")
    np.testing.assert_allclose(pweighted.weights, aweighted.weights, rtol=1e-12)
    assert pweighted.effective_n == aweighted.effective_n == 3.0
    # Stata: "specifying pweights implies the vce(robust) option"
    assert pweighted.implies_robust is True


def test_iweight_is_not_normalized():
    w = np.array([0.5, 1.5, 2.0])
    spec = normalize_weights(w, "iweight")
    np.testing.assert_array_equal(spec.weights, w)
    assert spec.effective_n == 4.0
    assert spec.implies_robust is False


def test_weight_type_is_case_insensitive():
    spec = normalize_weights([1.0, 1.0], "PWEIGHT")
    assert spec.weight_type == "pweight"


@pytest.mark.parametrize("bad", ["weight", "fweights", "", "freq"])
def test_unknown_weight_type_raises(bad):
    with pytest.raises(WeightError, match="weight_type"):
        normalize_weights([1.0, 2.0], bad)


@pytest.mark.parametrize(
    "w, message",
    [
        ([1.0, -1.0], "non-negative"),
        ([0.0, 0.0], "sum to zero"),
        ([1.0, np.nan], "missing"),
        ([1.0, np.inf], "finite"),
    ],
)
def test_invalid_weights_raise(w, message):
    with pytest.raises(WeightError, match=message):
        normalize_weights(w, "aweight")


def test_empty_weights_raise():
    with pytest.raises(WeightError, match="empty"):
        normalize_weights([], "aweight")
