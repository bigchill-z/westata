"""Negative aweight, the `(sum of wgt is ...)` line, and the closing border for a plain regress.

C07-04 of the v3 report is `reg lntc lnq lnpl lnpk lnpf [aw=1/lntc]` on nerlove.dta, scored 0.25.
Three separate defects accounted for it:

1. THE COMMAND COULD NOT RUN AT ALL. `1/lntc` is negative for 21 of 145 observations (lntc spans
   -2.501 to 4.9375) and `normalize_weights` refused any negative weight. Stata does not refuse:
   for aweight and pweight it DROPS those rows and reports e(N) = 124. `fweight` is the type that
   raises r(401) ("may not use noninteger frequency weights"), and it still does.

2. THE DEGREES OF FREEDOM WERE WRONG. Once the command ran, df_resid came out 140 (145 rows minus
   5 parameters) where Stata has 119 (124 minus 5), because effective_n counted rows the fit never
   used. That changes F's denominator and therefore its p-value.

3. TWO LINES WERE MISSING FROM THE OUTPUT: `(sum of wgt is ...)`, which aweight and pweight
   prepend, and the table's closing border -- which a plain `regress` prints and every ordinary
   regression was ending one line short of Stata's.

    before: 0.3333 line-match
    after:  0.8889 (16 of 18), the rest floating-point last digits

e(N) = 124 and b(lnq) = .7059381 both match Stata exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.data import normalize_weights
from westata.errors import WeightError

NERLOVE = r"D:\myCodeProject\test_westata\data\nerlove.dta"


@pytest.fixture(scope="module")
def frame():
    try:
        return pd.read_stata(NERLOVE)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("nerlove.dta is not available")


@pytest.fixture(scope="module")
def weighted(frame):
    return ws.OLS.from_formula(
        "lntc ~ lnq + lnpl + lnpk + lnpf",
        frame,
        weights=1.0 / frame["lntc"],
        weight_type="aweight",
    ).fit()


def test_a_negative_aweight_drops_the_row() -> None:
    spec = normalize_weights([1.0, -2.0, 3.0], "aweight")
    assert spec.dropped == 1
    assert spec.effective_n == 2.0
    assert spec.kept.tolist() == [True, False, True]


def test_a_negative_fweight_still_raises() -> None:
    """Stata: 'may not use noninteger frequency weights', r(401)."""

    with pytest.raises(WeightError, match="non-negative"):
        normalize_weights([1.0, -2.0], "fweight")


def test_the_weighted_fit_survives_the_negative_weights(weighted) -> None:
    assert int(weighted.nobs) == 124


def test_it_matches_stata(weighted) -> None:
    """From `reg lntc lnq lnpl lnpk lnpf [aw=1/lntc]`."""

    assert float(weighted.params["lnq"]) == pytest.approx(0.7059381, abs=5e-7)


def test_the_degrees_of_freedom_exclude_the_dropped_rows(weighted) -> None:
    """119 = 124 - 5, not 140 = 145 - 5. Getting this wrong moves F's p-value."""

    assert int(weighted.df_resid) == 119


def test_the_weight_sum_line_is_printed(weighted) -> None:
    rendered = str(weighted.summary(style="stata"))
    assert rendered.startswith("(sum of wgt is ")


def test_the_weight_sum_line_is_absent_for_fweight(frame) -> None:
    """Only aweight and pweight prepend it."""

    fitted = ws.OLS.from_formula(
        "lntc ~ lnq + lnpl",
        frame,
        weights=np.ones(len(frame)),
        weight_type="fweight",
    ).fit()
    assert not str(fitted.summary(style="stata")).startswith("(sum of wgt")


def test_a_plain_regress_closes_its_table(frame) -> None:
    """Stata prints a full border under the last coefficient row."""

    fitted = ws.OLS.from_formula("lntc ~ lnq + lnpl", frame).fit()
    lines = str(fitted.summary(style="stata")).splitlines()
    assert set(lines[-1]) == {"-"}


def test_a_panel_fit_does_not_gain_that_border(frame) -> None:
    """`xtreg, fe` puts a SPLIT rule there; the full border comes after rho."""

    panel = pd.read_stata(r"D:\myCodeProject\test_westata\data\munnell.dta")
    panel["sid"] = pd.Categorical(panel["st_abb"]).codes
    fitted = ws.PanelOLS.from_formula(
        "y ~ k1 + C(year)", panel, entity="sid", time="year"
    ).fit()
    lines = str(fitted.summary(style="stata")).splitlines()
    # The block ends with the F test, not with a bare rule.
    assert set(lines[-1]) != {"-"}
    assert len(lines) == 42
