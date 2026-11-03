"""Tests for ``estat hettest, mtest()`` -- multiple-testing adjustment.

Transcribed from ``hettest.ado:160-190`` and ``_mtest.ado:189-296``; the
per-variable and simultaneous statistics are checked against
``tests/conformance/stata_fixtures/hettest_auto.json`` (captured from StataNow
19.5), so the numbers are Stata's, not a second reading of the same ado.

``hettest`` with ``mtest()`` runs ONE auxiliary regression per regressor and
adjusts the resulting family of p-values:

    bonferroni: adjp[i] = min(1, m * p[i])
    sidak:      adjp[i] = 1 - (1 - p[i])^m
    holm:       step-down with multiplier m, m-1, ..., 1 in ascending p order,
                floored at the PREVIOUS ADJUSTED value (not the previous raw p)

The Holm floor is the part that is easy to omit, because it only binds when the
raw p-values are close together.  One test constructs exactly that case, since a
plain step-down on well-separated p-values looks correct without it.

Two option-interaction facts are pinned because both are silent:

* ``mtest()`` needs a MULTI-regressor auxiliary regression.  ``hettest.ado:
  126-128`` clears ``mtmethod`` when ``nvar == 1``, so the no-varlist default
  (fitted values) drops it and returns only the simultaneous test.
* ``, rhs`` is how a multi-column family is requested, since the varlist-less
  call is a single column.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import westata as ws
from westata.postestimation.mtest import MTEST_METHODS, adjust_pvalues

FIXTURE = (
    Path(__file__).resolve().parent
    / "conformance"
    / "stata_fixtures"
    / "hettest_auto.json"
)


@pytest.fixture(scope="module")
def reference() -> dict:
    assert FIXTURE.exists(), (
        f"{FIXTURE} is missing; regenerate it with "
        "scripts/capture_hettest_fixtures.py"
    )
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def frames():
    auto = pd.read_stata("tests/data/auto.dta")
    m1 = auto[["mpg", "weight", "foreign"]].dropna()
    m2 = auto[["mpg", "weight", "length", "displacement"]].dropna()
    return {
        "M1": (m1, ws.OLS.from_formula("mpg ~ weight + foreign", m1).fit()),
        "M2": (
            m2,
            ws.OLS.from_formula(
                "mpg ~ weight + length + displacement", m2
            ).fit(),
        ),
    }


# ------------------------------------------------------------- adjustment maths
def test_bonferroni_matches_the_ado_formula() -> None:
    p = np.array([0.001, 0.02, 0.3, 0.9])

    assert np.allclose(adjust_pvalues(p, "bonferroni"), np.minimum(1.0, p.size * p))


def test_sidak_matches_the_ado_formula() -> None:
    p = np.array([0.001, 0.02, 0.3, 0.9])

    assert np.allclose(adjust_pvalues(p, "sidak"), 1.0 - (1.0 - p) ** p.size)


def test_holm_matches_the_step_down() -> None:
    p = np.array([0.001, 0.02, 0.3, 0.9])
    expected = np.minimum(1.0, p * np.array([4, 3, 2, 1]))

    assert np.allclose(adjust_pvalues(p, "holm"), expected)


def test_holm_monotonicity_floor_actually_binds() -> None:
    """Without the floor the adjustment would DECREASE, which is invalid."""

    p = np.array([0.01, 0.013, 0.02])
    naive = p * np.array([3, 2, 1])
    assert np.all(np.diff(naive) < 0), "the raw steps must decrease here"

    expected = np.maximum.accumulate(naive)

    assert np.allclose(adjust_pvalues(p, "holm"), expected)
    assert not np.allclose(adjust_pvalues(p, "holm"), naive)


def test_noadjust_returns_the_raw_values() -> None:
    p = np.array([0.04, 0.5, 0.001])

    assert np.allclose(adjust_pvalues(p, "noadjust"), p)


def test_adjustments_preserve_test_order() -> None:
    p = np.array([0.5, 0.001, 0.2, 0.05])
    order = np.argsort(p)

    for method in ("bonferroni", "holm", "sidak"):
        adjusted = adjust_pvalues(p, method)
        assert np.allclose(adjusted[order], adjust_pvalues(p[order], method)), method


def test_unknown_method_is_refused() -> None:
    with pytest.raises(ValueError, match="not supported"):
        adjust_pvalues(np.array([0.1, 0.2]), "fdr")


def test_the_four_methods_are_the_ado_ones() -> None:
    assert set(MTEST_METHODS) == {"bonferroni", "holm", "sidak", "noadjust"}


# ------------------------------------------------------------- end-to-end
def test_the_family_matches_stata(frames, reference) -> None:
    """Every per-variable statistic and p-value equals Stata's ``r(mtest)``."""

    for model, entry in reference["mtest"].items():
        _, result = frames[model]
        table = result.estat_hettest(mtest="noadjust", rhs=True).table
        assert len(table) == entry["rows"], model
        for variable in entry["variables"]:
            row = table.loc[variable["name"]]
            assert row["chi2"] == pytest.approx(variable["chi2"], rel=1e-9), variable["name"]
            assert row["df"] == variable["df"], variable["name"]
            assert row["p"] == pytest.approx(variable["pvalue"], rel=1e-9), variable["name"]


def test_the_family_has_one_row_per_regressor(frames, reference) -> None:
    """One row per regressor, in ``e(b)`` order (the e(b) stripes, per the ado).

    Stata's ``r(mtest)`` rownames come from ``e(b)``, so the order follows the
    formula's written order rather than patsy's alphabetical one; the SET is what
    is pinned here, and ``test_the_family_matches_stata`` pins the values by name.
    """

    for model, entry in reference["mtest"].items():
        table = frames[model][1].estat_hettest(mtest="noadjust", rhs=True).table
        assert sorted(table.index) == sorted(v["name"] for v in entry["variables"]), model
        assert len(table) == entry["rows"], model


def test_per_variable_statistics_do_not_depend_on_the_method(frames) -> None:
    result = frames["M2"][1]
    raw = result.estat_hettest(mtest="noadjust", rhs=True)

    for method in ("bonferroni", "holm", "sidak"):
        adjusted = result.estat_hettest(mtest=method, rhs=True)
        assert np.allclose(
            adjusted.table["chi2"].to_numpy(), raw.table["chi2"].to_numpy()
        )
        assert np.allclose(adjusted.table["p"].to_numpy(), raw.table["p"].to_numpy())


def test_adjusted_values_match_the_standalone_function(frames) -> None:
    result = frames["M2"][1]
    raw_p = result.estat_hettest(mtest="noadjust", rhs=True).table["p"].to_numpy()

    for method in ("bonferroni", "holm", "sidak"):
        got = result.estat_hettest(mtest=method, rhs=True).table["adjustp"].to_numpy()
        assert np.allclose(got, adjust_pvalues(raw_p, method)), method


def test_each_variable_gets_its_own_auxiliary_regression(frames) -> None:
    """The statistic is that ONE regressor's ``e(mss)/2``, not the full model's.

    Uses M2 (three continuous regressors) so the reference computation needs no
    factor decoding: a categorical column cannot be cast to float, and
    ``hettest`` only ever sees the design matrix.
    """

    frame, result = frames["M2"]
    table = result.estat_hettest(mtest="noadjust", rhs=True).table

    residuals = np.asarray(result.resid, dtype=float)
    scale = float(residuals @ residuals) / residuals.size
    squared = residuals**2 / scale

    for name in ("weight", "length", "displacement"):
        column = frame[name].to_numpy(float)
        design = np.column_stack([np.ones(column.size), column])
        beta, *_ = np.linalg.lstsq(design, squared, rcond=None)
        resid = squared - design @ beta
        rss = float(resid @ resid)
        centred = squared - squared.mean()
        mss = float(centred @ centred) - rss
        assert table.loc[name, "chi2"] == pytest.approx(mss / 2.0, rel=1e-10), name


def test_degrees_of_freedom_are_one_per_single_regressor(frames) -> None:
    table = frames["M2"][1].estat_hettest(mtest="noadjust", rhs=True).table

    assert np.all(table["df"].to_numpy() == 1.0)


def test_pvalue_is_the_chi2_upper_tail(frames) -> None:
    table = frames["M2"][1].estat_hettest(mtest="noadjust", rhs=True).table

    for name, row in table.iterrows():
        assert row["p"] == pytest.approx(
            float(stats.chi2.sf(row["chi2"], row["df"])), rel=1e-12
        ), name


def test_r_matrix_shape(frames) -> None:
    """``r(mtest)`` is nvar x 3, gaining a fourth column when adjusted."""

    result = frames["M1"][1]

    assert result.estat_hettest(mtest="noadjust", rhs=True).to_matrix().shape == (2, 3)
    assert result.estat_hettest(mtest="holm", rhs=True).to_matrix().shape == (2, 4)


def test_the_simultaneous_statistic_matches_the_rhs_test(frames, reference) -> None:
    """``hettest.ado`` reports the family AND the joint test from one regression."""

    for model, entry in reference["mtest"].items():
        result = frames[model][1]
        simultaneous = result.estat_hettest(rhs=True).statistic
        assert simultaneous == pytest.approx(
            entry["simultaneous"]["chi2"], rel=1e-9
        ), model
        assert result.estat_hettest(rhs=True).df == entry["simultaneous"]["df"]


# ---------------------------------------------- mtest() is silently dropped
def test_mtest_is_dropped_when_only_one_regressor(frames, reference) -> None:
    """``hettest.ado:126-128``: ``if `nvar' == 1 { local mtmethod }``.

    Stata does not error; it reports only the simultaneous test and returns no
    ``r(mtest)``.  The no-varlist call is exactly this case, because it regresses
    on the fitted values -- one column.
    """

    result = frames["M1"][1]
    rendered = result.estat_hettest(mtest="holm")

    expected = reference["mtest_single_regressor"]
    assert type(rendered).__name__ == "HetTestResult"
    assert expected["r_mtest_missing"] is True
    assert rendered.statistic == pytest.approx(expected["chi2"], rel=1e-9)
    assert rendered.df == expected["df"]


def test_explicit_single_variable_also_drops_mtest(frames) -> None:
    result = frames["M1"][1]
    rendered = result.estat_hettest(variables=["weight"], mtest="holm")

    assert type(rendered).__name__ == "HetTestResult"
    assert rendered.df == 1


# --------------------------------------------------------------- refusals
def test_iid_with_mtest_is_refused(frames) -> None:
    with pytest.raises(ValueError, match="iid may not be specified"):
        frames["M1"][1].estat_hettest(method="iid", mtest="holm", rhs=True)


def test_fstat_with_mtest_is_refused(frames) -> None:
    with pytest.raises(ValueError, match="fstat may not be specified"):
        frames["M1"][1].estat_hettest(method="fstat", mtest="holm", rhs=True)


def test_unknown_mtest_method_is_refused(frames) -> None:
    with pytest.raises(ValueError, match="not supported"):
        frames["M1"][1].estat_hettest(mtest="fdr", rhs=True)


def test_robust_vce_is_still_refused_with_mtest(frames) -> None:
    frame, _ = frames["M1"]
    robust = ws.OLS.from_formula("mpg ~ weight + foreign", frame).fit(cov_type="HC1")

    with pytest.raises(ValueError, match="robust"):
        robust.estat_hettest(mtest="holm", rhs=True)


def test_output_lists_every_variable_and_the_adjustment(frames) -> None:
    text = str(frames["M2"][1].estat_hettest(mtest="holm", rhs=True))

    assert "holm" in text
    for name in ("weight", "length", "displacement"):
        assert name in text
    assert "adjustp" in text
