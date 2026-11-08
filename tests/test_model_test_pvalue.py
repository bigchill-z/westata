"""The printed model-test p-value is chi2.sf(chi2, df_m), which is not always e(p).

Round 208 found `zip`'s header contradicting itself:

    'LR chi2(1)    =   0.23'
    'Prob > chi2   =  0.9727'

A chi-square of 0.23 on one degree of freedom gives 0.6320, which is what Stata prints. Measuring Stata's own
e() explained the gap rather than contradicting it:

    e(chi2) 0.2293346    e(df_m) 1    e(rank) 4    e(p) 0.97272244

`e(p)` really is 0.9727 -- it uses rank-1 degrees of freedom, and the package was faithfully copying it. But THE
HEADER PRINTS chi2.sf(chi2, df_m), not e(p). The two differ only for the zero-inflated family, and there the
header is what a reader compares against Stata.

For Logit, Probit and Poisson, e(p) already equals chi2.sf(chi2, df_m) to twelve digits, so recomputing the
printed value is a no-op for them; all three were re-checked against live captures after the change.

This is the distinction that took an extra round to see: `e()` is a storage contract, the header is a display
contract, and they can legitimately disagree.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws


def _frame(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(n)
    frame = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
    frame["cnt"] = rng.poisson(2.0, n).astype(float)
    return frame


def _header_lines() -> list[str]:
    fitted = ws.ZeroInflatedPoisson.from_formula(
        "cnt ~ x", _frame(), inflate="z"
    ).fit()
    return [
        line
        for line in str(fitted.summary(style="stata")).splitlines()
        if line.strip()
    ]


def test_the_printed_p_matches_its_own_chi_square() -> None:
    """0.63 for chi2 0.23 on one degree of freedom -- not the stored e(p)."""

    lines = _header_lines()
    prob_row = next(line for line in lines if "Prob > chi2" in line)
    assert "0.6320" in prob_row, prob_row


def test_the_stored_e_p_is_still_the_rank_minus_one_value() -> None:
    """e() and the header are separate contracts; e(p) keeps Stata's own definition."""

    fitted = ws.ZeroInflatedPoisson.from_formula(
        "cnt ~ x", _frame(), inflate="z"
    ).fit()
    assert abs(float(fitted.e["p"]) - 0.97272) < 1e-4, fitted.e["p"]


def test_the_two_rows_no_longer_contradict_each_other() -> None:
    """chi2 0.23 beside p 0.97 was the visible symptom; p must exceed 0.6 and stay below 0.7."""

    lines = _header_lines()
    chi_row = next(line for line in lines if "LR chi2" in line)
    prob_row = next(line for line in lines if "Prob > chi2" in line)
    assert "0.23" in chi_row
    printed = float(prob_row.split("=")[-1].strip())
    assert 0.6 < printed < 0.7, printed


@pytest.mark.parametrize(
    "estimator,formula",
    [("Logit", "hi ~ x"), ("Probit", "hi ~ x"), ("Poisson", "cnt ~ x")],
)
def test_the_three_other_estimators_still_agree_with_their_chi_square(
    estimator, formula
) -> None:
    rng = np.random.default_rng(800)
    frame = pd.DataFrame({"y": rng.normal(size=800), "x": rng.normal(size=800)})
    frame["hi"] = (frame["y"] > 0).astype(float)
    frame["cnt"] = rng.poisson(2.0, 800).astype(float)
    fitted = getattr(ws, estimator).from_formula(formula, frame).fit()
    printed = next(
        line
        for line in str(fitted.summary(style="stata")).splitlines()
        if "Prob > chi2" in line
    )
    expected = f"{float(__import__('scipy.stats', fromlist=['chi2']).chi2.sf(fitted.e['chi2'], fitted.e['df_m'])):.4f}"
    assert expected in printed, (estimator, printed, expected)
