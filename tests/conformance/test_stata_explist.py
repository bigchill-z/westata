"""E3 test for the expression-list bootstrap path.

Stata on this data:

    bootstrap r(mean), reps(2000): summarize y
        _bs_1   1.838715   .073721

westata through the expression list: SE 0.0735211625879815, a 0.27% gap at 2000
replicates -- well inside the roughly 1.6% Monte Carlo error at that count, and consistent
with the seed-band reasoning recorded for sqreg.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.expressions.explist import ExpListError, StoredResults, parse_exp_list

STATA_RMEAN = 1.838715
STATA_RMEAN_SE = 0.073721


def _frame(n: int = 400, seed: int = 401) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "y": rng.normal(loc=2.0, scale=1.5, size=n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        }
    )


def test_bootstrap_of_r_mean_matches_stata():
    frame = _frame()
    model = ws.Bootstrap.from_expressions(
        "(m: value=r(mean))",
        frame,
        formula="y ~ 1",
        estimator="mean",
        reps=2000,
        seed=20261023,
    )
    result = model.fit()

    assert list(result.param_names) == ["m:value"]
    # Stata prints 1.838715 -- six decimals -- so six decimals is the precision of the
    # reference, not of the computation.
    assert float(result.params.iloc[0]) == pytest.approx(STATA_RMEAN, abs=1e-6)
    # 1% is the Monte Carlo band at 2000 replicates, not implementation slack.
    assert float(result.bse.iloc[0]) == pytest.approx(STATA_RMEAN_SE, rel=1e-2)


def test_expressions_over_regression_results():
    """The four tables _b/_se/e() reach must all be populated for a regression."""

    frame = _frame()
    model = ws.Bootstrap.from_expressions(
        "tstat=(_b[x1]/_se[x1]) rmse=(e(rmse)) r2=(e(r2)) nobs=(e(N))",
        frame,
        formula="y ~ x1 + x2",
        estimator="regress",
        reps=200,
        seed=7,
    )
    result = model.fit()

    assert set(result.param_names) == {"tstat", "rmse", "r2", "nobs"}
    # e(N) is constant across replicates, so its bootstrap SE must be exactly zero.
    assert float(result.params["nobs"]) == 400.0
    assert float(result.bse["nobs"]) == pytest.approx(0.0, abs=1e-12)
    # The observed t statistic matches the plain fit.
    plain = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()
    expected = float(plain.params["x1"]) / float(plain.bse["x1"])
    assert float(result.params["tstat"]) == pytest.approx(expected, rel=1e-10)


def test_the_manual_summarise_group_form():
    """``(location: mean=r(mean) median=r(p50))`` needs median in the r() table."""

    frame = _frame()
    model = ws.Bootstrap.from_expressions(
        "(location: mean=r(mean))", frame, formula="y ~ 1", estimator="mean", reps=50, seed=1
    )
    result = model.fit()
    assert list(result.param_names) == ["location:mean"]


def test_an_invalid_expression_fails_before_resampling():
    """A typo must not cost 2000 replicate fits first."""

    with pytest.raises(ExpListError):
        ws.Bootstrap.from_expressions(
            "a=(r(nonexistent_thing))",
            _frame(),
            formula="y ~ 1",
            estimator="mean",
            reps=2000,
        )


def test_expressions_are_reproducible_with_a_seed():
    frame = _frame()
    first = ws.Bootstrap.from_expressions(
        "rmse=(e(rmse))", frame, formula="y ~ x1", estimator="regress", reps=100, seed=42
    ).fit()
    second = ws.Bootstrap.from_expressions(
        "rmse=(e(rmse))", frame, formula="y ~ x1", estimator="regress", reps=100, seed=42
    ).fit()
    np.testing.assert_allclose(
        np.asarray(first.cov_params()), np.asarray(second.cov_params()), rtol=1e-12
    )


def test_the_parser_alone_covers_the_manual_examples():
    """The parser is verified independently of any estimator."""

    stored = StoredResults(
        r={"mean": 5.0, "p50": 4.0, "p25": 2.0, "p75": 7.0, "sd": 2.5, "max": 9.0, "min": 1.0},
        e={"rmse": 1.5},
        b={"mpg": -0.5},
        se={},
    )
    parsed = parse_exp_list(
        "(location: mean=r(mean) median=r(p50)) "
        "(scale: sd=r(sd) iqr=(r(p75)-r(p25)) range=(r(max)-r(min)))"
    )
    values = parsed.evaluate(stored)
    assert values["location:mean"] == 5.0
    assert values["location:median"] == 4.0
    assert values["scale:iqr"] == 5.0
    assert values["scale:range"] == 8.0

    exponential = parse_exp_list("or=(exp(_b[mpg]))")
    assert exponential.evaluate(stored)["or"] == pytest.approx(math.exp(-0.5))
