"""Stata's ``test`` accepts SEVERAL constraints in one string, separated by commas.

    test x1 = 0, x2 = 0

is a JOINT test of two restrictions (``df = 2``), not one restriction on a coefficient
called ``0,x2``. The parser used to treat the whole string as a single hypothesis, so the
comma form failed with

    KeyError: coefficient '0,x2' not found

while the two-argument form worked. Both spellings must now agree, because Stata users
write the comma form.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

import westata as ws


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(739)
    n = 300
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    frame = pd.DataFrame(
        {"y": 1.0 + 0.8 * x1 - 0.3 * x2 + rng.normal(size=n), "x1": x1, "x2": x2}
    )
    return ws.OLS.from_formula("y ~ x1 + x2", frame).fit()


def test_the_comma_form_is_a_joint_test(fitted) -> None:
    """Two restrictions give ``df = 2``."""

    tested = fitted.test("x1 = 0, x2 = 0")

    assert float(tested.df) == 2.0
    assert np.isfinite(float(tested.statistic))
    assert 0.0 <= float(tested.pvalue) <= 1.0


def test_the_comma_form_matches_the_two_argument_form(fitted) -> None:
    """The two spellings describe the same hypothesis and must agree exactly."""

    comma = fitted.test("x1 = 0, x2 = 0")
    separate = fitted.test("x1 = 0", "x2 = 0")

    assert float(comma.statistic) == pytest.approx(float(separate.statistic), rel=1e-12)
    assert float(comma.df) == float(separate.df)
    assert float(comma.pvalue) == pytest.approx(float(separate.pvalue), rel=1e-12)


def test_a_joint_test_is_not_the_product_of_the_singles(fitted) -> None:
    """Guard: the joint statistic must actually be the 2-df Wald form.

    If the comma had been split into two SEPARATE tests and only one kept, the result
    would equal a single-restriction statistic -- still plausible, and wrong.
    """

    joint = fitted.test("x1 = 0, x2 = 0")
    single = fitted.test("x1 = 0")

    assert float(joint.df) != float(single.df)
    assert float(joint.statistic) != pytest.approx(float(single.statistic), rel=1e-6)


def test_the_joint_statistic_is_the_quadratic_form(fitted) -> None:
    """``F = (Rb)' (R V R')^-1 (Rb) / q`` for the two zero restrictions.

    Computed from the stored estimates, so the test does not merely restate the
    implementation's own arithmetic.
    """

    names = list(fitted.param_names)
    params = np.asarray(fitted.params, dtype=float)
    cov = np.asarray(fitted.cov_params(), dtype=float)
    positions = [names.index("x1"), names.index("x2")]

    restricted = params[positions]
    block = cov[np.ix_(positions, positions)]
    wald = float(restricted @ np.linalg.solve(block, restricted))
    expected = wald / len(positions)

    assert float(fitted.test("x1 = 0, x2 = 0").statistic) == pytest.approx(
        expected, rel=1e-10
    )


def test_a_trailing_comma_is_tolerated(fitted) -> None:
    """``test x1 = 0,`` should not become an empty second hypothesis."""

    tested = fitted.test("x1 = 0,")

    assert float(tested.df) == 1.0


def test_equality_chains_still_work(fitted) -> None:
    """``x1 = x2`` is one restriction, not a comma-separated list."""

    tested = fitted.test("x1 = x2")

    assert float(tested.df) == 1.0
    assert np.isfinite(float(tested.statistic))


def test_three_constraints(fitted) -> None:
    """The split handles more than two parts."""

    tested = fitted.test("x1 = 0, x2 = 0, x1 = x2")

    assert float(tested.df) == 3.0
    assert np.isfinite(float(tested.statistic))


def test_an_unknown_name_still_names_itself(fitted) -> None:
    """Splitting must not swallow the error for a genuinely bad coefficient."""

    with pytest.raises(Exception) as caught:
        fitted.test("nope = 0, x2 = 0")

    assert "nope" in str(caught.value)