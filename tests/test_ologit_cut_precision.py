"""ologit: cut points differ in the 8th digit while the LIKELIHOOD is identical.

Round 210 found `< /cut1 >` differed by one printed digit. Following it down to full precision:

    Stata      s = -0.06945370989890     cut1 = 0.05379984507472    ll = -439.79902573776917
    westata    s = -0.06945370593431     cut1 = 0.05379990129543    ll = -439.79902573776906

THE LIKELIHOOD AGREES TO FIFTEEN DIGITS -- the two implementations are at the SAME maximum. The
slope agrees to eight. Only the cut point, whose estimate is a threshold on the latent scale rather
than a fitted slope, differs by 5.6e-8, which is enough to flip the eighth printed figure.

That rules out a formula error and rules out a different likelihood surface. What remains is that
the maximum is flat enough in the cut direction that two optimisers stop at points indistinguishable
in likelihood but distinguishable in the raw parameter. Supporting evidence: refitting the same
model with a different optimiser moves the cut by more than the gap to Stata --

    statsmodels bfgs   0.05378388765707
    statsmodels newton 0.05379980280619      <- closest to Stata's 0.05379984507472
    statsmodels lbfgs  0.05372872391532

the spread across optimisers (7e-5) is three orders of magnitude larger than the remaining
difference to Stata (4e-8).

This is the same class as round 200's odds-ratio last digit: reported, not chased. Pinning agreement
would mean nudging a converged estimate, and the quantity the reader cares about -- the likelihood and
the slopes -- already matches.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\grilic.dta"

#: From `ologit rns s expr` at full precision.
STATA_SLOPE_S = -0.06945370989890
STATA_CUT1 = 0.05379984507472
STATA_LOGLIKE = -439.79902573776917


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(DATA).dropna(subset=["rns", "s", "expr"])
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")
    return ws.OrderedLogit.from_formula("rns ~ s + expr", frame).fit()


def test_the_log_likelihood_is_identical(fitted) -> None:
    """Fifteen digits -- both implementations sit at the same maximum."""

    assert float(fitted.llf) == pytest.approx(STATA_LOGLIKE, abs=1e-10)


def test_the_slopes_match_to_eight_digits(fitted) -> None:
    assert float(fitted.params["s"]) == pytest.approx(STATA_SLOPE_S, abs=1e-8)


def test_the_cut_point_differs_in_the_eighth_digit(fitted) -> None:
    """Recorded as KNOWN. It is inside the spread between optimisers on the same likelihood.

    If a future round pins it exactly, replace this with the equality assertion.
    """

    cut = float(fitted.params["/cut1"])
    gap = abs(cut - STATA_CUT1)
    assert gap < 1e-6, "a larger gap would mean a formula error, not optimiser noise"
    assert gap > 1e-9, "if this fails the cut point now matches -- pin the equality instead"


def test_optimiser_choice_moves_the_cut_far_more_than_stata_does(fitted) -> None:
    """Which is what shows the remaining gap is a property of the maximum, not of this code."""

    from statsmodels.miscmodels.ordinal_model import OrderedModel

    frame = pd.read_stata(DATA).dropna(subset=["rns", "s", "expr"])
    model = OrderedModel(
        frame["rns"].to_numpy(float),
        frame[["s", "expr"]].to_numpy(float),
        distr="logit",
    )
    cuts = {
        method: float(model.fit(method=method, disp=False).params[-1])
        for method in ("bfgs", "newton")
    }
    spread = abs(cuts["bfgs"] - cuts["newton"])
    gap_to_stata = abs(float(fitted.params["/cut1"]) - STATA_CUT1)
    assert spread > 100 * gap_to_stata
