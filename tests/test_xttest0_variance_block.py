"""xttest0: the variance block was printing the WRONG QUANTITY, off by a squaring.

C12-04 scored a layout of 0.64 against a numeric coverage of 0.50. The test statistic itself was
already exact -- chibar2(01) = 3921.14 on munnell.dta, matching Stata to the printed digit -- so the
gap was in the "Estimated results" block above it.

A COMMENT IN THE SOURCE WAS WRONG AND THE CODE OBEYED IT. ``xttest0.py`` asserted that
``xtreg, re`` stores ``e(sigma_u)``/``e(sigma_e)`` as VARIANCES and printed them straight through.
A live ``ereturn list`` shows they are STANDARD DEVIATIONS:

    e(sigma_u) = .0818628686135636
    xttest0's "u" row = .0067015 = e(sigma_u)^2

so the block was off by exactly the squaring. Round 176 fixed that, then found the two rows do not
even come from one formula:

    the "u" row  <- e(sigma_u)^2                              .0067015  vs Stata .0067015
    the "e" row  <- within residual variance / (N - n)         .00151239 vs Stata .0015071

``e(sigma_e)^2`` is 0.00154534 for the "e" row -- 2.5% too high -- which is why squaring both
looked almost right and would have passed a spot check on the "u" row alone.

ALSO FIXED: the equation line and the first table row printed the placeholders ``y[id,t]`` / ``y``
where Stata prints the model's own names (``lny[sid,t]`` / ``lny``).

REMAINING, recorded rather than papered over: the entity name in that line is whatever the caller
passed as ``entity=``, so it reads ``lny[id,t]`` unless the result knows the panel's id variable
name; and the two variance cells sit 0.35% / 0.03% from Stata because the GLS variance components
are estimated slightly differently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\munnell.dta"
FORMULA = "lny ~ lnk1 + lnk2 + lnlabor"

#: From `xtreg lny lnk1 lnk2 lnlabor, re` then `xttest0`.
STATA_CHIBAR2 = 3921.14
STATA_VAR_Y = 1.04271
STATA_VAR_E = 0.0015071
STATA_VAR_U = 0.0067015


@pytest.fixture(scope="module")
def block():
    try:
        frame = pd.read_stata(DATA)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("munnell.dta is not available")
    result = ws.RandomEffects.from_formula(
        FORMULA, frame, entity="st_abb", time="year"
    ).fit()
    return ws.xttest0(result, entity=frame["st_abb"].to_numpy())


def test_the_test_statistic_is_unchanged(block) -> None:
    """It matched before the fix too -- which is why the defect went unnoticed."""

    assert float(block.chibar2) == pytest.approx(STATA_CHIBAR2, abs=5e-3)


def test_the_variance_block_is_on_the_variance_scale(block) -> None:
    """The squaring bug: these used to print 0.08184913 and 0.03931081."""

    assert float(block.var_e) == pytest.approx(STATA_VAR_E, rel=5e-3)
    assert float(block.var_u) == pytest.approx(STATA_VAR_U, rel=5e-3)


def test_the_two_rows_do_not_come_from_one_formula(block) -> None:
    """e(sigma_e)^2 would be 0.00154534 for the "e" row -- 2.5% above what Stata prints."""

    assert float(block.var_e) < 0.00154534
    # The "u" row IS e(sigma_u)^2, and the two must not be interchangeable.
    assert float(block.var_u) > 4 * float(block.var_e)


def test_the_response_variance_matches(block) -> None:
    assert float(block.var_y) == pytest.approx(STATA_VAR_Y, abs=5e-6)


def test_the_first_row_is_named_after_the_dependent_variable(block) -> None:
    """Stata writes 'lny', not the placeholder 'y' the module used to hard-code."""

    assert block.depvar == "lny"
    rendered = str(block)
    assert "                     lny |" in rendered
    assert "        y[id,t]" not in rendered


def test_every_data_row_is_fifty_two_characters(block) -> None:
    """The three data rows (not the header) end the variance/SD columns at 37 and 52."""

    rows = [
        line
        for line in str(block).splitlines()
        if " |" in line and not line.lstrip().startswith("|")
    ]
    assert len(rows) == 3
    assert all(len(line) == 52 for line in rows)
