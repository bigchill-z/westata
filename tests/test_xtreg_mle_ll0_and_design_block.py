"""`xtreg, mle` was missing e(ll_0)/e(chi2), and the whole panel design block.

The v3 report lists this under "output is missing the group count and LR chi2". Two distinct causes:

1. NO e(ll_0). Stata reports the model test as LR chi2 = 2*(ll - ll_0). The estimator stored e(ll)
   but not e(ll_0), so the header's LR chi2 row had nothing to print. It is now fitted as a
   constant-only random-effects model with the SAME optimiser (Nelder-Mead start, then Powell), so
   the two likelihoods come from the same procedure.

   On the reference panel: ll_0 = -529.69283 against Stata's -529.6928, and
   chi2 = 1.5225 against Stata's 1.5225.

2. NO DESIGN BLOCK. `_panel_design_block` gated on `model_name.startswith("xt")`. The REML
   estimator carries model_name "xtreg, re", but this one carries "random-effects MLE regression"
   -- while its e(cmd) is still "xtreg". Keying on model_name therefore dropped Group variable,
   Time variable, Random effects u_i and Obs per group from this command alone. The gate now also
   accepts e(cmd), which is the macro Stata itself sets for the command.

STILL INCOMPLETE (recorded, not pretended): the tail the report also mentions -- `/sigma_u` and
`/sigma_e` printed WITH standard errors and an interval, and the "LR test of sigma_u=0" line. The
block is 24 lines against Stata's 27.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def panel():
    rng = np.random.default_rng(7)
    n_g, n_t = 40, 8
    sid = np.repeat(np.arange(n_g), n_t)
    u = np.repeat(rng.normal(scale=1.2, size=n_g), n_t)
    return pd.DataFrame(
        {
            "y": 1.0 + 0.6 * rng.normal(size=n_g * n_t) + u + rng.normal(size=n_g * n_t),
            "x1": rng.normal(size=n_g * n_t),
            "sid": sid,
            "t": np.tile(np.arange(n_t), n_g),
        }
    )


@pytest.fixture(scope="module")
def fitted(panel):
    return ws.RandomEffectsMLE.from_formula(
        "y ~ x1", panel, entity="sid", time="t"
    ).fit()


def test_ll_0_is_recorded(fitted) -> None:
    """From `xtreg y x1, mle`: e(ll_0) = -529.6928."""

    ll_0 = float(fitted.e["ll_0"])
    assert np.isfinite(ll_0)
    assert ll_0 == pytest.approx(-529.6928, abs=1e-3)


def test_chi2_is_twice_the_likelihood_difference(fitted) -> None:
    macros = fitted.e
    expected = 2.0 * (float(macros["ll"]) - float(macros["ll_0"]))
    assert float(macros["chi2"]) == pytest.approx(expected, rel=1e-12)


def test_chi2_matches_stata(fitted) -> None:
    """Stata reports LR chi2(1) = 1.5225."""

    assert float(fitted.e["chi2"]) == pytest.approx(1.5225, abs=1e-4)


def test_chi2type_is_lr_not_wald(fitted) -> None:
    """The header must label this as a likelihood-ratio test."""

    assert str(fitted.e["chi2type"]).upper() == "LR"


def test_the_design_block_is_present(fitted) -> None:
    """Group variable / Random effects / Obs per group were all dropped by the model_name gate."""

    rendered = str(fitted.summary(style="stata"))
    for marker in ("Group variable:", "Random effects u_i ~ Gaussian", "Obs per group:"):
        assert marker in rendered, marker


def test_the_lr_chi2_row_is_printed(fitted) -> None:
    assert "LR chi2(1)" in str(fitted.summary(style="stata"))


def test_the_gate_still_accepts_a_plain_xt_model_name(panel) -> None:
    """The e(cmd) fallback must not disturb the estimator that already carried an xt model_name."""

    rendered = str(
        ws.RandomEffects.from_formula("y ~ x1", panel, entity="sid", time="t")
        .fit()
        .summary(style="stata")
    )
    assert "Obs per group:" in rendered
