"""Pin ``teffects_psmatch`` to Stata, including the tie rule that decides the ATET.

Reference, from a live ``teffects psmatch (re78) (treat <8 covariates>), atet`` on
nsw_dw.dta (n = 445)::

    default nn(1):  ATET 2639.865, SE 714.7758, Matches min=1 max=8
    nn(4):          ATET 2330.108, SE 680.0515, Matches min=4 max=11

The estimator was reconstructed from Stata's ``generate(m)`` option, which stored EIGHT
match-ID variables for a requested ``nneighbor(1)``. The help says "The number of variables
generated may be more than nneighbor() because of tied distances". Using only the single
nearest control gives 2569.745; taking EVERY control tied at the minimum distance reproduces
2639.865 exactly. The header's "max = 8" is that tie count.

Verified with Stata's own exported propensity scores AND with westata's Logit predictions
(differ by 2.75e-8); tie tolerances from 1e-8 to 1e-6 all give the same two ATETs. The
logit coefficients agree with Stata to 1e-7, so none of this is a model discrepancy.

NOT reproduced: the Abadie-Imbens AI-robust standard error, which needs a local-linear
variance estimator rather than a closed form. The inferential fields are NaN and the table
prints Stata's dot placeholder there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\nsw_dw.dta"
COVS = [
    "age", "education", "black", "hispanic", "married",
    "nodegree", "re74", "re75",
]


@pytest.fixture(scope="module")
def sample():
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("nsw_dw.dta is not available")
    return (
        frame["re78"].to_numpy(float),
        frame["treat"].to_numpy(float),
        frame[COVS].to_numpy(float),
    )


def test_default_atet_matches_stata(sample) -> None:
    y, d, x = sample
    result = ws.teffects_psmatch(y, d, x)
    assert result.ate == pytest.approx(2639.8646, abs=5e-3)
    assert result.treated == 185
    assert result.controls == 260
    assert result.nobs == 445


def test_nn4_atet_matches_stata(sample) -> None:
    y, d, x = sample
    result = ws.teffects_psmatch(y, d, x, nneighbor=4)
    assert result.ate == pytest.approx(2330.1077, abs=5e-3)


def test_the_tie_counts_match_statas_header(sample) -> None:
    """requested=1 gives min=1/max=8; requested=4 gives min=4/max=11."""

    y, d, x = sample
    one = ws.teffects_psmatch(y, d, x, nneighbor=1)
    four = ws.teffects_psmatch(y, d, x, nneighbor=4)
    assert (one.matches_min, one.matches_max) == (1, 8)
    assert (four.matches_min, four.matches_max) == (4, 11)


def test_using_only_one_neighbour_would_be_wrong(sample) -> None:
    """The gap the report recorded: 2569.745 against Stata's 2639.865."""

    y, d, x = sample
    from westata.estimators.discrete import Logit

    design = np.column_stack([np.ones(x.shape[0]), x])
    score = np.asarray(Logit(d, design).fit().predict(), dtype=float)
    controls = np.flatnonzero(d == 0)
    treated = np.flatnonzero(d == 1)
    smallest = float(
        np.mean([y[i] - y[controls[np.argmin(np.abs(score[i] - score[controls]))]]
                 for i in treated])
    )
    assert smallest == pytest.approx(2569.745, abs=1.0)
    assert abs(smallest - 2639.865) > 60  # the ties are not a rounding detail


def test_the_standard_error_is_reported_as_unavailable(sample) -> None:
    """AI robust needs a local-linear variance estimator; a guess is not published."""

    y, d, x = sample
    result = ws.teffects_psmatch(y, d, x)
    assert not result.variance_matches_stata
    assert np.isnan(result.se)
    assert np.isnan(result.z)
    assert np.isnan(result.pvalue)


def test_the_table_matches_stata_layout(sample) -> None:
    y, d, x = sample
    rendered = [
        line for line in str(ws.teffects_psmatch(y, d, x, outcome="re78")).splitlines()
        if line.strip()
    ]
    expected = [
        "Treatment-effects estimation                   Number of obs      =        445",
        "Estimator      : propensity-score matching     Matches: requested =          1",
        "Outcome model  : matching                                     min =          1",
        "Treatment model: logit                                        max =          8",
        "-" * 78,
        "             |              AI robust",
        "        re78 | Coefficient  std. err.      z    P>|z|     [95% conf. interval]",
        "-" * 13 + "+" + "-" * 64,
        "ATET         |",
        "       treat |",
        "-" * 78,
    ]
    for line in expected:
        assert line in rendered, line
    data_row = rendered[10]
    assert len(data_row) == 78
    assert "2639.865" in data_row
    assert data_row.count(".") == 6  # estimate's decimal + five dot placeholders


def test_bad_treatment_coding_is_refused(sample) -> None:
    y, d, x = sample
    bad = np.where(d > 0, 1, 2)
    with pytest.raises(Exception):
        ws.teffects_psmatch(y, bad, x)
