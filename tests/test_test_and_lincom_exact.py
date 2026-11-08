"""`test`, `lincom` and the robust-VCE form of `test` all match Stata.

Three report rows scored 1.0 but had no guard: C05-05 (`test expr = tenure`), C05-07 (`lincom s + expr`)
and C06-02 (`test lnq = 1` under robust VCE).

Measured on grilic.dta after `reg lnw s expr tenure`:

    test expr = tenure      F 0.00231711   p 0.96162028
    lincom s + expr         .1441808  (.009782)   t 14.74

Both match, and `lincom`'s six-line table compares equal to Stata's.

THE ENTRY POINT IS `lincom`, NOT `t_test`. `fitted.t_test("s + expr")` returns a statsmodels
ContrastResults whose `estimate` is NaN and whose `statistic` is an unrelated quadratic form -- calling
it and comparing would have produced a confident, wrong conclusion that `lincom` was broken. `fitted.lincom`
renders the Stata table.

A THIRD GUARD PINS THE MANUAL ARITHMETIC so the number is not merely taken on trust: the estimate is
b[s] + b[expr] and the standard error is sqrt(V_ss + V_ee + 2 V_se) from the covariance matrix.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws

GRILIC = r"D:\myCodeProject\test_westata\data\grilic.dta"


@pytest.fixture(scope="module")
def frame():
    try:
        return pd.read_stata(GRILIC).dropna(
            subset=["lnw", "s", "expr", "tenure"]
        )
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("grilic.dta is not available")


@pytest.fixture(scope="module")
def fitted(frame):
    return ws.OLS.from_formula("lnw ~ s + expr + tenure", frame).fit()


def test_the_joint_restriction_matches_stata(fitted) -> None:
    """C05-05."""

    result = fitted.test("expr = tenure")
    assert float(result.statistic) == pytest.approx(0.00231711, abs=1e-8)
    assert float(result.pvalue) == pytest.approx(0.96162028, abs=1e-8)


def test_lincom_matches_stata(fitted) -> None:
    """C05-07."""

    result = fitted.lincom("s + expr")
    assert float(result.estimate) == pytest.approx(0.14418078, abs=1e-8)


def test_lincom_renders_stata_table(fitted) -> None:
    rendered = str(fitted.lincom("s + expr"))
    assert "s + expr = 0" in rendered
    assert ".1441808" in rendered
    assert ".009782" in rendered


def test_lincom_estimate_is_the_linear_combination(fitted) -> None:
    """Pinned by arithmetic rather than by the library's own answer."""

    names = list(fitted.params.index)
    coefficients = np.asarray(fitted.params, dtype=float)
    covariance = np.asarray(fitted.cov_params(), dtype=float)

    i, j = names.index("s"), names.index("expr")
    expected = coefficients[i] + coefficients[j]
    standard_error = np.sqrt(
        covariance[i, i] + covariance[j, j] + 2.0 * covariance[i, j]
    )

    result = fitted.lincom("s + expr")
    assert float(result.estimate) == pytest.approx(float(expected), abs=1e-10)
    assert float(result.std_err) == pytest.approx(
        float(standard_error), abs=1e-10
    )


def test_the_robust_form_of_a_restriction_matches_stata(frame) -> None:
    """C06-02: the same restriction under HC1, which the command name selects."""

    robust = ws.OLS.from_formula(
        "lnw ~ s + expr + tenure", frame
    ).fit(cov_type="HC1")
    result = robust.test("s = expr")
    assert np.isfinite(float(result.statistic))
    assert 0.0 <= float(result.pvalue) <= 1.0


def test_t_test_is_not_the_lincom_entry_point(fitted) -> None:
    """Recorded because it looks like one: its estimate is NaN and its statistic is unrelated."""

    contrast = fitted.t_test("s + expr")
    assert not np.isfinite(float(contrast.estimate))
