"""Multiway clustering must be the CGM combination, not an approximation of it.

``replication/papers/adhm2020/WESTATA_GAPS.md`` recorded two gaps: ``IV2SLS`` refused
multiway clusters outright, and ``OLS`` rejected a two-column ``groups`` with "must be
one-dimensional" -- the single cell where westata was behind a *vanilla* Stata, since
``regress`` accepts ``vce(cluster a b)`` (``regress.sthlp:196``).

Both are fixed by one implementation, ``westata.vce.cluster.cov_multiway_cluster``, which
applies Cameron-Gelbach-Miller inclusion-exclusion:

    V = V_A + V_B - V_{A & B}

The tests below pin the property that makes this a DECOMPOSITION rather than a new
estimator: the two-way matrix must equal that arithmetic combination of three one-way
fits, to machine precision. "It runs and the number looks plausible" is exactly what a
wrong combination step would also produce, so the identity is the assertion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


def _frame(n: int = 600, seed: int = 11501) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    covariate = rng.normal(size=n)
    instrument = rng.normal(size=n)
    endogenous = 0.8 * instrument + rng.normal(size=n) * 0.5
    return pd.DataFrame(
        {
            "y": 1.0 + 0.6 * endogenous + rng.normal(size=n),
            "x": covariate,
            "endog": endogenous,
            "z": instrument,
            "grp_a": np.repeat(np.arange(20), n // 20),
            "grp_b": np.tile(np.arange(12), n // 12),
        }
    )


@pytest.fixture(scope="module")
def frame():
    return _frame()


def _variance_ols(frame: pd.DataFrame, groups, **fit_kwargs) -> np.ndarray:
    fitted = ws.OLS.from_formula("y ~ x", frame).fit(
        cov_type="cluster", cov_kwds={"groups": groups}, **fit_kwargs
    )
    return np.asarray(fitted.cov_params(), dtype=float)


def _variance_iv(frame: pd.DataFrame, groups) -> np.ndarray:
    fitted = ws.IV2SLS.from_formula("y ~ 1 + [endog ~ z]", frame).fit(
        cov_type="cluster", cov_kwds={"clusters": groups}
    )
    return np.asarray(fitted.cov_params(), dtype=float)


# ------------------------------------------------------- OLS (GAP 2, closed)
def test_ols_accepts_two_columns(frame) -> None:
    matrix = _variance_ols(frame, frame[["grp_a", "grp_b"]])

    assert matrix.shape == (2, 2)
    assert np.all(np.isfinite(matrix))


def test_ols_two_way_is_the_cgm_combination(frame) -> None:
    """THE assertion: the arithmetic identity, to machine precision."""

    two_way = frame[["grp_a", "grp_b"]]
    composite = (frame["grp_a"].astype(str) + "_" + frame["grp_b"].astype(str)).to_numpy()

    combined = _variance_ols(frame, two_way)
    expected = (
        _variance_ols(frame, frame["grp_a"])
        + _variance_ols(frame, frame["grp_b"])
        - _variance_ols(frame, composite)
    )

    np.testing.assert_allclose(combined, expected, rtol=1e-12, atol=0.0)


def test_ols_two_way_differs_from_either_one_way(frame) -> None:
    """Guard against "accepted and silently used one dimension"."""

    two_way = _variance_ols(frame, frame[["grp_a", "grp_b"]])

    only_a = _variance_ols(frame, frame["grp_a"])
    only_b = _variance_ols(frame, frame["grp_b"])

    assert not np.allclose(two_way, only_a)
    assert not np.allclose(two_way, only_b)


def test_a_numpy_two_column_array_works_too(frame) -> None:
    """The API accepted arrays before it accepted DataFrames; both must keep working."""

    columns = np.column_stack([frame["grp_a"].to_numpy(), frame["grp_b"].to_numpy()])

    from_frame = _variance_ols(frame, frame[["grp_a", "grp_b"]])
    from_array = _variance_ols(frame, columns)

    np.testing.assert_allclose(from_frame, from_array, rtol=1e-12, atol=0.0)


# -------------------------------------------------------- IV (GAP 1, closed)
def test_iv_accepts_two_columns(frame) -> None:
    """This raised ``UnsupportedError`` before the fix."""

    matrix = _variance_iv(frame, frame[["grp_a", "grp_b"]])

    assert matrix.shape == (2, 2)
    assert np.all(np.isfinite(matrix))


def test_iv_two_way_is_the_cgm_combination(frame) -> None:
    two_way = frame[["grp_a", "grp_b"]]
    composite = (frame["grp_a"].astype(str) + "_" + frame["grp_b"].astype(str)).to_numpy()

    combined = _variance_iv(frame, two_way)
    expected = (
        _variance_iv(frame, frame["grp_a"])
        + _variance_iv(frame, frame["grp_b"])
        - _variance_iv(frame, composite)
    )

    np.testing.assert_allclose(combined, expected, rtol=1e-12, atol=0.0)


def test_ols_and_iv_share_the_same_combination_rule(frame) -> None:
    """The inconsistency in §4 of the gaps document is gone.

    One implementation serves both, so their two-way matrices must both equal their own
    one-way combination -- which means neither can drift into its own convention.
    """

    for variance in (_variance_ols, _variance_iv):
        composite = (
            frame["grp_a"].astype(str) + "_" + frame["grp_b"].astype(str)
        ).to_numpy()
        expected = (
            variance(frame, frame["grp_a"])
            + variance(frame, frame["grp_b"])
            - variance(frame, composite)
        )
        actual = variance(frame, frame[["grp_a", "grp_b"]])
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.0)


# ----------------------------------------------------------- the bookkeeping
def test_n_clust_reports_the_smallest_dimension(frame) -> None:
    """Stata reports the smallest per-dimension group count for a multiway fit."""

    fitted = ws.OLS.from_formula("y ~ x", frame).fit(
        cov_type="cluster", cov_kwds={"groups": frame[["grp_a", "grp_b"]]}
    )

    smallest = min(frame["grp_a"].nunique(), frame["grp_b"].nunique())
    assert float(fitted.e["N_clust"]) == float(smallest)


def test_iv_n_clust_also_uses_the_smallest_dimension(frame) -> None:
    """The IV path used ``np.unique`` on the 2-D array, which raised ``TypeError``
    once the two label columns had different dtypes."""

    fitted = ws.IV2SLS.from_formula("y ~ 1 + [endog ~ z]", frame).fit(
        cov_type="cluster", cov_kwds={"clusters": frame[["grp_a", "grp_b"]]}
    )

    smallest = min(frame["grp_a"].nunique(), frame["grp_b"].nunique())
    assert float(fitted.e["N_clust"]) == float(smallest)


# --------------------------------------------------- one-way is unaffected
def test_one_way_clustering_is_unchanged(frame) -> None:
    """The multiway branch must not alter the ordinary path."""

    fitted = ws.OLS.from_formula("y ~ x", frame).fit(
        cov_type="cluster", cov_kwds={"groups": frame["grp_a"]}
    )
    matrix = np.asarray(fitted.cov_params(), dtype=float)

    assert matrix.shape == (2, 2)
    assert float(fitted.e["N_clust"]) == float(frame["grp_a"].nunique())


def test_a_single_column_dataframe_is_still_one_way(frame) -> None:
    """A one-column frame must not be mistaken for a multiway spec."""

    fitted = ws.OLS.from_formula("y ~ x", frame).fit(
        cov_type="cluster", cov_kwds={"groups": frame[["grp_a"]]}
    )

    assert float(fitted.e["N_clust"]) == float(frame["grp_a"].nunique())


# ------------------------------------------------------- three dimensions
def test_three_dimensions_use_inclusion_exclusion(frame) -> None:
    """The general CGM sum, not a hard-coded two-way formula.

    For three dimensions the signs alternate as +A +B +C -AB -AC -BC +ABC. A two-way-only
    implementation would silently drop or mis-sign the higher-order terms.
    """

    data = _frame(n=600, seed=11502)
    data["grp_c"] = np.tile(np.arange(6), 100)
    dimensions = data[["grp_a", "grp_b", "grp_c"]]

    combined = _variance_ols(data, dimensions)

    def key(*names):
        return data[list(names)].astype(str).agg("_".join, axis=1).to_numpy()

    expected = (
        _variance_ols(data, data["grp_a"])
        + _variance_ols(data, data["grp_b"])
        + _variance_ols(data, data["grp_c"])
        - _variance_ols(data, key("grp_a", "grp_b"))
        - _variance_ols(data, key("grp_a", "grp_c"))
        - _variance_ols(data, key("grp_b", "grp_c"))
        + _variance_ols(data, key("grp_a", "grp_b", "grp_c"))
    )

    np.testing.assert_allclose(combined, expected, rtol=1e-10, atol=0.0)