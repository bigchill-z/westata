"""``ivregress ..., first``: the first-stage regression table (v2 report C10-01).

The package had ``estat firststage`` -- the DIAGNOSTICS -- but nothing that printed the
first-stage EQUATION the way Stata's ``, first`` does. Stata offers both and they are not
substitutes: one reports R-squared and the instrument F, the other reports the coefficients.

Reference, ``ivregress 2sls lnw s expr (tenure = rns smsa), first`` on grilic (n = 758)::

    First-stage regressions
    -----------------------

                                                            Number of obs =    758
                                                            F(4, 753)     =  11.15
                                                            Prob > F      = 0.0000
                                                            R-squared     = 0.0559
                                                            Adj R-squared = 0.0509
                                                            Root MSE      = 1.6305

    ------------------------------------------------------------------------------
          tenure | Coefficient  Std. err.      t    P>|t|     [95% conf. interval]
    -------------+----------------------------------------------------------------
               s |   .0006668   .0275293     0.02   0.981    -.0533765    .0547102
            expr |   .1846684   .0290088     6.37   0.000     .1277207    .2416161
             rns |  -.1210981   .1354721    -0.89   0.372    -.3870459    .1448497
            smsa |   .1304456   .1320924     0.99   0.324    -.1288676    .3897587
           _cons |   1.442411   .3970731     3.63   0.000     .6629091    2.221913
    ------------------------------------------------------------------------------

The column positions were measured off those lines with re.finditer, not derived; the number
formatter is the one already pinned against ``display %9.0g``, because a local re-derivation
printed .0006668274 where Stata prints .0006668.
"""

from __future__ import annotations

import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\grilic.dta"
STATA_DATA_ROWS = [
    "           s |   .0006668   .0275293     0.02   0.981    -.0533765    .0547102",
    "        expr |   .1846684   .0290088     6.37   0.000     .1277207    .2416161",
    "         rns |  -.1210981   .1354721    -0.89   0.372    -.3870459    .1448497",
    "        smsa |   .1304456   .1320924     0.99   0.324    -.1288676    .3897587",
    "       _cons |   1.442411   .3970731     3.63   0.000     .6629091    2.221913",
]


@pytest.fixture(scope="module")
def rendered():
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False).dropna()
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")
    fitted = ws.IV2SLS.from_formula(
        "lnw ~ s + expr + [tenure ~ rns + smsa]", frame
    ).fit()
    return str(fitted.first_stage())


def test_the_header_block_matches_stata(rendered) -> None:
    assert "First-stage regressions" in rendered
    assert "      tenure | Coefficient  Std. err.      t    P>|t|     [95% conf. interval]" in rendered
    assert "F(4, 753)     =  11.15" in rendered or "F(4, 753) =  11.15" in rendered


@pytest.mark.parametrize("row", STATA_DATA_ROWS)
def test_each_coefficient_row_matches_stata(rendered, row: str) -> None:
    assert row in rendered, row


def test_every_row_is_78_characters(rendered) -> None:
    for line in rendered.splitlines():
        if "|" in line:
            assert len(line) == 78, repr(line)


def test_the_constant_is_labelled_cons(rendered) -> None:
    """patsy calls it Intercept; Stata prints _cons."""

    assert "_cons" in rendered
    assert "Intercept" not in rendered


def test_the_diagnostics_are_a_separate_output() -> None:
    """estat_firststage must still report R-squared and F, not coefficients."""

    try:
        frame = pd.read_stata(DATA, convert_categoricals=False).dropna()
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("grilic.dta is not available")
    fitted = ws.IV2SLS.from_formula(
        "lnw ~ s + expr + [tenure ~ rns + smsa]", frame
    ).fit()
    diagnostics = str(fitted.estat_firststage())
    assert "First-stage regression summary statistics" in diagnostics
    assert "Partial" in diagnostics
    assert "tenure |  0.0559" in diagnostics
