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

ROUND 6 ADDED the rest of the tail:

  * `/sigma_u` and `/sigma_e` now carry the SLASH prefix Stata uses for ML parameters, and the
    "~ (fraction of variance due to u_i)" suffix is dropped for this variant (Stata prints it only
    for the linear RE regression, not for the ML fit);
  * "LR test of sigma_u=0: chibar2(01)" is implemented. It compares this fit against the POOLED
    OLS likelihood, NOT against the constant-only model -- 2*(-528.931579 - -591.455135) = 125.05,
    which is Stata's chibar2(01), while the header's LR chi2(1) of 1.5225 is the constant-only
    comparison. The p-value halves the chi2(1) tail, as the other boundary tests do;
  * the "Integration method / Integration pts." rows are GONE. `xtreg ..., mle` is a linear mixed
    model solved in closed form, so Stata prints no integration row for it -- the gate previously
    keyed only on "has sigma_u", which every RE estimator satisfies;
  * "Prob > chi2" is now printed beside "Log likelihood", and "LR chi2(1)" on the line above it.

The block is now 23 lines, matching `xtreg y x1, mle nolog` line for line in structure.

STILL DIFFERENT, and outside this task: the coefficient and sigma values differ in the last one or
two digits (optimiser precision), and /sigma_u //sigma_e still lack their standard-error column --
those need Stata's own parameterisation for the delta method, and a plausible-looking number there
would be worse than the omission.
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


def test_the_sigma_rows_use_the_ml_slash_prefix(fitted) -> None:
    """ML parameters print as /sigma_u and /sigma_e; the linear RE fit uses bare names."""

    rendered = str(fitted.summary(style="stata"))
    assert "/sigma_u" in rendered
    assert "/sigma_e" in rendered


def test_the_ml_fit_does_not_print_an_integration_row(fitted) -> None:
    """xtreg, mle is closed form; Stata prints no Integration method/pts. for it."""

    rendered = str(fitted.summary(style="stata"))
    assert "Integration method" not in rendered
    assert "Integration pts." not in rendered


def test_the_sigma_u_test_is_against_the_pooled_fit(fitted) -> None:
    """chibar2(01) compares this fit with POOLED OLS, not with the constant-only model."""

    macros = fitted.e
    pooled = 2.0 * (float(macros["ll"]) - float(macros["ll_pooled"]))
    constant_only = float(macros["chi2"])
    assert pooled != pytest.approx(constant_only, rel=1e-6)
    assert pooled == pytest.approx(125.05, abs=0.05)


def test_the_sigma_u_test_line_is_printed(fitted) -> None:
    assert "LR test of sigma_u=0" in str(fitted.summary(style="stata"))


def test_prob_chi2_is_printed_beside_the_likelihood(fitted) -> None:
    lines = str(fitted.summary(style="stata")).splitlines()
    pair = next(line for line in lines if "Log likelihood" in line)
    assert "Prob > chi2" in pair
