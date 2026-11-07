"""Likelihood fits now close their coefficient table, and `nbreg` was found to be broken.

THE BORDER. Stata's `logit` ends with a full-width rule under `_cons`; this package stopped at
`_cons`. The gate that closes a table covered `regress` and the variance-decomposition estimators,
so every likelihood fit was one line short. Measured before and after, counting only the summary
block (Stata run with `nolog` so the iteration log does not inflate the comparison):

    logit      1.0000  (13 of 13)
    probit     1.0000  (13 of 13)
    poisson    1.0000  (13 of 13)
    nbreg      0.5000  -- see below

The xt likelihood family is deliberately excluded: its tail is the sigma/rho decomposition, which
`_closes_with_border` already handles, or the `LR test of rho=0` line.

A MEASUREMENT ERROR WORTH RECORDING: the first scoring script hardcoded `ws.OLS.from_formula` for
every case, so it compared Stata's logit against this package's OLS and reported 0.28 line-match
for logit/probit/poisson. The classes were right all along; the harness was wrong. A stale
__pycache__ made the first border fix look inert for the same kind of reason.

NBREG IS BROKEN, found by this comparison and NOT fixed here:

    quantity      westata        Stata
    LR chi2       0.00           69.88
    Log lik.      -302.32947     -196.12988
    se(weight)    .0001969       .000041
    /lnalpha      .05            -20.69024
    alpha         1.051271       1.03e-09

The optimiser lands at alpha ~ 1 where Stata drives it to ~ 1e-9; with alpha that large the
negative binomial has not collapsed to Poisson, so every standard error is inflated and the model
test is meaningless. That is an estimation defect, not a layout one, and it needs its own round.
"""

from __future__ import annotations

import pandas as pd
import pytest

import westata as ws

GIRLIC = r"D:\myCodeProject\test_westata\data\grilic.dta"
AUTO = r"tests\data\auto.dta"


@pytest.fixture(scope="module")
def grilic():
    try:
        return pd.read_stata(GIRLIC).dropna(subset=["rns", "s", "expr"])
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("grilic.dta is not available")


@pytest.fixture(scope="module")
def auto():
    try:
        frame = pd.read_stata(AUTO).dropna(subset=["mpg", "weight"])
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("auto.dta is not available")
    frame = frame.copy()
    frame["fgn"] = (
        frame["foreign"]
        .astype(str)
        .str.lower()
        .str.startswith(("foreign", "1"))
        .astype(float)
    )
    return frame


@pytest.mark.parametrize(
    "klass, formula, frame_name",
    [
        (ws.Logit, "rns ~ s + expr", "grilic"),
        (ws.Probit, "rns ~ s + expr", "grilic"),
        (ws.Poisson, "mpg ~ weight + fgn", "auto"),
    ],
)
def test_a_likelihood_table_closes_with_a_rule(klass, formula, frame_name, request) -> None:
    frame = request.getfixturevalue(frame_name)
    fitted = klass.from_formula(formula, frame).fit()
    lines = str(fitted.summary(style="stata")).splitlines()
    assert set(lines[-1].strip()) == {"-"}, lines[-1][:40]


def test_the_block_length_matches_stata(grilic) -> None:
    """`logit rns s expr, nolog` prints 13 lines; this package now prints the same 13."""

    fitted = ws.Logit.from_formula("rns ~ s + expr", grilic).fit()
    assert len(str(fitted.summary(style="stata")).splitlines()) == 13


def test_a_panel_likelihood_fit_does_not_gain_the_rule(grilic) -> None:
    """`xtlogit, re` ends with its sigma/rho block, not a bare rule."""

    import numpy as np

    rng = np.random.default_rng(3)
    n_g, n_t = 30, 8
    frame = pd.DataFrame(
        {
            "y": (rng.uniform(size=n_g * n_t) < 0.5).astype(float),
            "x1": rng.normal(size=n_g * n_t),
            "sid": np.repeat(np.arange(n_g), n_t),
            "t": np.tile(np.arange(n_t), n_g),
        }
    )
    fitted = ws.XtLogit.from_formula(
        "y ~ x1", frame, entity="sid", time="t", method="re"
    ).fit()
    lines = str(fitted.summary(style="stata")).splitlines()
    assert set(lines[-1].strip()) != {"-"}


def test_nbreg_still_does_not_match_stata() -> None:
    """Pins the KNOWN defect so it cannot be mistaken for coverage.

    The optimiser stops near alpha = 1 where Stata reaches alpha ~ 1e-9, so the likelihood, the
    model test and every standard error differ. When this starts passing, the optimiser was fixed
    and this test should be replaced with the equality assertions in the module docstring.
    """

    frame = pd.read_stata(AUTO).dropna(subset=["mpg", "weight"]).copy()
    frame["fgn"] = (
        frame["foreign"]
        .astype(str)
        .str.lower()
        .str.startswith(("foreign", "1"))
        .astype(float)
    )
    fitted = ws.NegativeBinomial.from_formula("mpg ~ weight + fgn", frame).fit()
    ll = float(fitted.e["ll"])
    # Stata: -196.12988. Landing at -302 means alpha never collapsed.
    assert ll < -250.0, f"nbreg improved to {ll}; the optimiser may have been fixed"
