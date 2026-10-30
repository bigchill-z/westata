"""Tests for Stata's ``estat vce`` / ``estat vce, correlation``.

The matrix itself is not recomputed here -- it is ``cov_params()`` -- so what is
asserted is (a) that the rendering round-trips the matrix exactly, (b) that the
correlation form is ``D^-1/2 V D^-1/2`` with a zeroed diagonal, and (c) that the
labels use Stata's spelling (``_cons``, bare ``cut1``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.postestimation.vce import estat_vce


@pytest.fixture()
def auto():
    df = pd.read_stata("tests/data/auto.dta")
    df["dom"] = (df["foreign"] == "Foreign").astype(int)
    return df


def _permute_to_display_order(result):
    """Indices mapping ``cov_params``' order onto ``estat vce``'s displayed order.

    The printed matrix follows Stata's row order (``_cons`` LAST), which since round
    123 differs from ``cov_params``' patsy order.  Comparing by POSITION would report a
    correct reordering as a corrupted matrix, so the comparison is made by name.
    """

    names = list(result.estat_vce().names)
    stored = [str(name) for name in result.param_names]
    lookup = {}
    for index, name in enumerate(stored):
        key = "_cons" if name in ("Intercept", "intercept") else (
            name[1:] if name.startswith("/") else name
        )
        lookup[key.replace(":", "#")] = index
    return [lookup[name] for name in names]


def test_estat_vce_round_trips_cov_params(auto) -> None:
    result = ws.OLS.from_formula("mpg ~ weight + foreign", auto).fit()
    rendered = result.estat_vce()
    positions = _permute_to_display_order(result)
    expected = np.asarray(result.cov_params())[np.ix_(positions, positions)]

    np.testing.assert_allclose(
        rendered.to_frame().to_numpy(),
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_estat_vce_correlation_is_scaled_covariance(auto) -> None:
    result = ws.OLS.from_formula("mpg ~ weight + foreign", auto).fit()
    positions = _permute_to_display_order(result)
    cov = np.asarray(result.cov_params())[np.ix_(positions, positions)]
    se = np.sqrt(np.diag(cov))
    expected = cov / np.outer(se, se)
    np.fill_diagonal(expected, 0.0)

    np.testing.assert_allclose(
        result.estat_vce(correlation=True).to_frame().to_numpy(),
        expected,
        rtol=1e-15,
        atol=1e-15,
    )


def test_estat_vce_correlation_entries_are_bounded(auto) -> None:
    result = ws.OLS.from_formula("mpg ~ weight + foreign", auto).fit()
    corr = result.estat_vce(correlation=True).to_frame().to_numpy()

    assert np.all(np.diag(corr) == 0.0)
    off_diagonal = corr[~np.eye(corr.shape[0], dtype=bool)]
    assert np.all(np.abs(off_diagonal) <= 1.0 + 1e-12)


def test_estat_vce_labels_use_stata_spelling(auto) -> None:
    result = ws.OLS.from_formula("mpg ~ weight + foreign", auto).fit()
    names = result.estat_vce().names

    assert "_cons" in names
    assert "Intercept" not in names


def test_estat_vce_ordered_cutpoints_are_bare(auto) -> None:
    """Stata prints ``cut1``; westata's parameter spelling is ``/cut1``."""

    result = ws.OrderedLogit.from_formula("rep78 ~ weight", auto).fit()
    names = result.estat_vce().names

    assert not any(name.startswith("/") for name in names), names
    assert "cut1" in names


def test_estat_vce_renders_lower_triangle_only(auto) -> None:
    result = ws.OLS.from_formula("mpg ~ weight + foreign", auto).fit()
    text = str(result.estat_vce())

    assert "Covariance matrix" in text
    # The first labelled data row must not contain a value in the last column
    # (that is the upper triangle), which Stata leaves blank.
    rows = [line for line in text.splitlines() if line.startswith("_cons")]
    assert len(rows) == 1
    assert "foreign[T.Foreign]" not in rows[0].split("_cons", 1)[1]


def test_estat_vce_works_on_nonlinear_models(auto) -> None:
    """The matrix follows Stata's row order for an ML fit too: ``_cons`` LAST.

    This test used to assert ``["_cons", "weight", "mpg"]`` -- patsy's order.  Stata
    prints the constant last in every estimation table, and ``estat vce`` reproduces
    that, so the old expectation locked in an ordering no Stata user would see.
    """

    result = ws.Logit.from_formula("dom ~ weight + mpg", auto).fit()
    rendered = result.estat_vce()

    assert rendered.names == ["weight", "mpg", "_cons"]
    positions = _permute_to_display_order(result)
    np.testing.assert_allclose(
        rendered.to_frame().to_numpy(),
        np.asarray(result.cov_params())[np.ix_(positions, positions)],
        rtol=0.0,
        atol=0.0,
    )


def test_estat_vce_module_function_matches_method(auto) -> None:
    result = ws.OLS.from_formula("mpg ~ weight", auto).fit()

    np.testing.assert_allclose(
        estat_vce(result).to_frame().to_numpy(),
        result.estat_vce().to_frame().to_numpy(),
        rtol=0.0,
        atol=0.0,
    )


def test_estat_vce_rejects_objects_without_param_names() -> None:
    with pytest.raises(TypeError, match="param_names"):
        estat_vce(object())
