"""`zinb` shares zip's header rules, and both are now byte-identical through the column headings.

Rounds 209-210 fixed zip's printed p-value and its log-likelihood digits by model name, listing "zinb" among the
seven-figure family. This round confirmed that empirically on a negative-binomial sample, comparing the first
seven header lines against a live `zinb`:

    0  'Zero-inflated negative binomial regression              Number of obs =  1,000'
    1  'Inflation model: logit                                  Nonzero obs   =    838'
    2  '                                                        Zero obs      =    162'
    3  '                                                        LR chi2(1)    =   ...'
    4  'Log likelihood = -2228.662                              Prob > chi2   =  0....'
    5  separator
    6  '         cnt | Coefficient  Std. err.      z    P>|z|     ...'

ALL SEVEN MATCH, including the seven-figure likelihood.

WHAT THIS FILE DOES NOT COVER: the coefficient block below row 6 differs, because Stata prefixes it with a
"cnt |" equation heading for the two-equation commands and the package does not. That is a separate structural
difference in the ml table, recorded rather than folded in here.

`ztp` and `ztnb` ARE NOT IMPLEMENTED in this package, so there is nothing to compare; they are recorded as missing
functionality and skipped, per the task's scope.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws


def _fit(n: int = 1000):
    rng = np.random.default_rng(n)
    frame = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
    frame["cnt"] = rng.negative_binomial(2, 0.4, n).astype(float)
    return ws.ZeroInflatedNegativeBinomial.from_formula(
        "cnt ~ x", frame, inflate="z"
    ).fit()


def _header() -> list[str]:
    # Option C puts an `Iteration 0` line (and possibly a `note:`) before the banner, so the
    # first non-blank line is no longer necessarily the header.
    return [
        line
        for line in str(_fit().summary(style="stata")).splitlines()
        if line.strip()
        and not line.startswith(("Iteration", "note:"))
    ]


def test_the_banner_names_the_negative_binomial_model() -> None:
    assert _header()[0].startswith("Zero-inflated negative binomial regression")


def test_the_count_row_lands_on_seventy() -> None:
    row = next(line for line in _header() if "Number of obs" in line)
    assert row.index("=") == 70, row
    assert "1,000" in row


def test_the_log_likelihood_keeps_seven_figures() -> None:
    """-2228.662 matches Stata; eight figures would give -2228.6618."""

    row = next(line for line in _header() if "Log likelihood" in line)
    assert "-2228.662" in row, row


def test_the_printed_p_matches_its_chi_square() -> None:
    """The zip fix applies here too: e(p) uses rank-1 df, the header uses df_m."""

    fitted = _fit()
    row = next(
        line
        for line in str(fitted.summary(style="stata")).splitlines()
        if "Prob > chi2" in line
    )
    from scipy import stats

    expected = f"{float(stats.chi2.sf(float(fitted.e['chi2']), float(fitted.e['df_m']))):.4f}"
    assert expected in row, (row, expected)


def test_the_inflation_rows_are_present() -> None:
    lines = _header()
    assert any("Inflation model: logit" in line for line in lines)
    assert any("Nonzero obs" in line for line in lines)
    assert any("Zero obs" in line for line in lines)
