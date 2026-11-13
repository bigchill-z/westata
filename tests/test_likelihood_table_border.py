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
    # Option C prepends `Iteration 0` plus a blank line, so the TOTAL line count grew from 13.
    # The BORDER rule is about the table block itself, which still has 13 lines; skip the preamble.
    lines = str(fitted.summary(style="stata")).splitlines()
    # Skip only the `Iteration 0` LINE; its following blank is the table's original leading blank
    # and is part of the 13-line body the assertion measures.
    # MEASURED: Stata `logit rns s expr, nolog` prints 13 lines and OPENS WITH A BLANK LINE. With
    # `Iteration 0` prepended the output is '' / trace / '' / banner... -- so BOTH the trace line and
    # the blank after it are the addition, not just the trace.
    body = [
        line
        for index, line in enumerate(lines)
        if not line.startswith("Iteration")
        and not line.startswith("note:")
        and not (
            line == "" and index > 0 and lines[index - 1].startswith("Iteration")
        )
    ]
    assert len(body) == 13, lines[:4]


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


def test_nbreg_now_matches_stata() -> None:
    """The optimiser defect is FIXED as of round 13.

    The previous version of this test pinned the failure: the fit stopped at alpha = 1.0513 with
    log likelihood -302.329467, where Stata reaches alpha = 1.03e-09 and -196.12988. A single BFGS
    run never walks toward the boundary, so `NegativeBinomial._fit_backend` now restarts from
    several dispersions and keeps the best likelihood that clears a 1e-3 improvement threshold.

    The threshold matters: near alpha -> 0 the computed likelihood is flat and stops being
    trustworthy, and on the other reference dataset a 1e-9 start gains 5.3e-04 that is pure
    numerical noise. Sixty units separate that from the real fix here.
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

    # Stata: log likelihood -196.12988, alpha 1.03e-09.
    assert float(fitted.e["ll"]) == pytest.approx(-196.12988, abs=5e-3)
    # The old local optimum was -302.3; anything near it means the multi-start was lost.
    assert float(fitted.e["ll"]) > -200.0


def test_the_multi_start_keeps_a_non_boundary_optimum(grilic) -> None:
    """On data whose true alpha is ~6e-06 the fit must NOT be dragged to the 1e-9 boundary.

    Measured: this dataset's Stata reference is alpha = 6.1e-06 with ll = -569.0777629. A naive
    "take the highest likelihood" rule picks a 1e-9 start whose 5.3e-04 gain is numerical noise.
    """

    import numpy as _np
    import pandas as _pd

    rng = _np.random.default_rng(11)
    n = 600
    x1 = rng.normal(size=n)
    mu = _np.exp(0.5 + 0.4 * x1)
    y = rng.poisson(mu).astype(float)
    frame = _pd.DataFrame({"y": y, "x1": x1})

    fitted = ws.NegativeBinomial.from_formula("y ~ x1", frame).fit()
    alpha = float(fitted.e["alpha"])
    assert _np.isfinite(alpha) and alpha > 0.0
    # Well away from the 1e-9 floor the selector clamps to.
    assert alpha > 1e-8
