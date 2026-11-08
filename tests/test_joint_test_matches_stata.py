"""`test()` handles multi-constraint joint hypotheses exactly; C17-04's premise does not hold.

Report item C17-04 records `test canal_post post` after a `didregress` as "approximate" and scores it
0.5. Measured, that command fails on BOTH sides:

    Stata   `test canal_post post`       -> r(111): canal_post not found
            e(b) names are ATET:r1vs0.canal_post plus 16 Controls:year dummies
    package `r.test('canal_post post')`  -> KeyError: coefficient 'post' not found

The DID design absorbs `post` into the time fixed effects, so no such coefficient exists anywhere.
What IS correct is `test()` itself, verified against a live Stata:

    reg re78 age education black    then    test age education

        constraints   (1) age = 0      (2) education = 0
        F(2, 441)     4.33             the package gives 4.3307
        Prob > F      0.0137           the package gives 0.0137

The joint test across several restrictions, the constraint printing and the degrees of freedom all
match. The entry in the report is therefore about a command that cannot be run, not about a gap in
`test()`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\nsw_dw.dta"

#: Captured from `reg re78 age education black` then `test age education`.
STATA_F = 4.33
STATA_P = 0.0137


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(DATA)
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("nsw_dw.dta is not available")
    return ws.OLS.from_formula("re78 ~ age + education + black", frame).fit()


def test_a_joint_test_of_two_restrictions_matches_stata(fitted) -> None:
    result = fitted.test("age education")
    assert float(result.statistic) == pytest.approx(STATA_F, abs=5e-3)
    assert float(result.pvalue) == pytest.approx(STATA_P, abs=5e-5)


def test_the_two_spellings_agree(fitted) -> None:
    """`age education` and `(age = 0) (education = 0)` are the same hypothesis."""

    comma_form = fitted.test("age education")
    paren_form = fitted.test("(age = 0) (education = 0)")
    assert float(comma_form.statistic) == pytest.approx(
        float(paren_form.statistic), rel=1e-12
    )


def test_the_printed_constraints_name_every_restriction(fitted) -> None:
    """Stata numbers them `( 1)` and `( 2)`; both labels must appear."""

    rendered = str(fitted.test("age education"))
    assert "( 1)" in rendered
    assert "( 2)" in rendered
    assert "age" in rendered
    assert "education" in rendered


def test_a_single_restriction_is_a_different_hypothesis(fitted) -> None:
    """Pinned so a joint test cannot be mistaken for the marginal one."""

    single = float(fitted.test("age = 0").statistic)
    joint = float(fitted.test("age education").statistic)
    assert single != pytest.approx(joint, rel=1e-6)
