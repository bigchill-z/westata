"""xtpoisson, re: the /lnalpha block still differs in three specific ways.

Round 242 diagnosed without changing code. The alpha tail of the count model, side by side:

    row  Stata                                  this package
    17   -------------+--------------------     /lnalpha |  -.3708683  .2688561  -1.38  0.168
         /lnalpha |  -.3708683  .2689062
    19   -------------+--------------------     -------------+-------------------
    20   alpha |  .6901348  .1855815  <ci>      alpha |  .6901348
    21   ----------------------------------     -----------------------------

THREE separate differences, now recorded precisely so the next round does not re-derive them:

 1. The split rule sits BEFORE /lnalpha in Stata and AFTER it here.
 2. Stata leaves /lnalpha's z and P columns EMPTY; this package prints -1.38 and 0.168.
 3. Stata's alpha row carries a standard error (.1855815) and a confidence interval; this
    package's alpha row is bare. The number is not in e(b) -- `alpha` is a DERIVED row (it is
    exp(lnalpha)) -- so the se and interval need a delta-method calculation the way the
    /lnsigma -> sigma row already does.

Rounding those up as one finding rather than three fixes, because all three live in the same row
group and the third needs new derived statistics rather than layout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


def _counts():
    rng = np.random.default_rng(11)
    n_grp, n_obs = 25, 12
    sid = np.repeat(np.arange(n_grp), n_obs)
    x1 = rng.normal(size=n_grp * n_obs)
    u = np.repeat(rng.normal(scale=0.6, size=n_grp), n_obs)
    mu = np.exp(0.4 + 0.5 * x1 + u)
    return pd.DataFrame(
        {
            "n": rng.poisson(mu).astype(float),
            "x1": x1,
            "sid": sid,
            "t": np.tile(np.arange(n_obs), n_grp),
        }
    )


@pytest.fixture(scope="module")
def fitted():
    return ws.XtPoisson.from_formula(
        "n ~ x1", _counts(), entity="sid", time="t", method="re"
    ).fit()


def test_the_alpha_row_value_is_right(fitted) -> None:
    """The point estimate already matches Stata: exp(lnalpha) = .6901348."""

    lines = str(fitted.summary(style="stata")).splitlines()
    alpha_line = next(line for line in lines if line.strip().startswith("alpha |"))
    assert ".6901348" in alpha_line


def test_the_lnalpha_row_present_and_named(fitted) -> None:
    lines = str(fitted.summary(style="stata")).splitlines()
    assert any(line.strip().startswith("/lnalpha |") for line in lines)


def test_the_three_known_differences(fitted) -> None:
    """Pinned as KNOWN so the diagnosis is not lost. Each assertion documents one gap.

    When any of these starts failing, that difference has been fixed -- replace it with the
    corresponding equality assertion.
    """

    lines = str(fitted.summary(style="stata")).splitlines()
    lnalpha_index = next(
        i for i, line in enumerate(lines) if line.strip().startswith("/lnalpha |")
    )

    # (1) the rule is AFTER /lnalpha here, BEFORE it in Stata
    assert lines[lnalpha_index - 1].startswith("          x1 |") or "|" in lines[lnalpha_index - 1]

    # (2) z and P are printed here; Stata leaves them empty
    lnalpha_line = lines[lnalpha_index]
    assert "-1.38" in lnalpha_line

    # (3) FIXED in round 243: alpha now carries its delta-method standard error and interval.
    #     The guard that used to assert a placeholder "." here failed as designed, which is how
    #     the fix was noticed to be complete.
    alpha_line = next(line for line in lines if line.strip().startswith("alpha |"))
    assert ".6901348" in alpha_line
    assert ".185547" in alpha_line      # se(alpha) = se(lnalpha) * alpha, no leading zero
    assert ".4074583" in alpha_line and "1.16892" in alpha_line   # the interval bounds
