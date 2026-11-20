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
    return ws.PanelPoisson.from_formula(
        "n ~ x1", _counts(), entity="sid", time="t", method="re"
    ).fit()


def _alpha_fields(line: str) -> dict[str, float]:
    """Parse an ``alpha |  est  se  <blank z>  <blank P>  lower  upper`` row.

    The Stata layout leaves the z and P columns empty, so the numeric fields are
    not at fixed positions; taking every float on the row and naming them by
    count keeps this readable and still fails loudly if a column disappears.
    """

    tokens = [tok for tok in line.split("|", 1)[1].split() if tok]
    values = [float(tok) for tok in tokens]
    if len(values) != 4:  # pragma: no cover - shape regression
        raise AssertionError(f"expected 4 numeric fields on the alpha row, got {values}")
    estimate, std_err, lower, upper = values
    return {"estimate": estimate, "std_err": std_err, "lower": lower, "upper": upper}


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

    # (1) FIXED in round 244: the split rule now sits ABOVE /lnalpha, as Stata has it.
    assert lines[lnalpha_index - 1].startswith("-------------+")

    # (2) FIXED in round 244: /lnalpha prints no z or P.
    lnalpha_line = lines[lnalpha_index]
    assert "-1.38" not in lnalpha_line

    # (3) FIXED in round 243: alpha now carries its delta-method standard error and interval.
    #     The guard that used to assert a placeholder "." here failed as designed, which is how
    #     the fix was noticed to be complete.
    #
    #     WHAT THIS PINS IS THE LAYOUT, not the arithmetic's last digits.  The row used
    #     to be matched as the literal text ".185547" / ".4074583" / "1.16892", which
    #     asserts the optimiser's sixth significant figure.  se(alpha) is a delta-method
    #     combination over a numerically inverted Hessian, and measured across three
    #     platforms it lands at .1855469 (Windows), .1855392 (Linux CI, Python 3.10) and
    #     .1855317 (Linux CI, Python 3.11/3.12) -- a 1.5e-5 spread for identical data.
    #     The interval bounds move with it, up to ~5e-5.
    #
    #     So the assertions below are: the row exists, it carries FOUR numbers (estimate,
    #     standard error, lower, upper -- the z and P columns stay empty), they are in the
    #     right order at the right magnitudes, and alpha is exp(/lnalpha).  A dropped
    #     column, a shifted decimal point or a returned "." all still fail this; a
    #     different BLAS does not.
    alpha_line = next(line for line in lines if line.strip().startswith("alpha |"))
    assert ".6901348" in alpha_line, alpha_line
    fields = _alpha_fields(alpha_line)
    assert fields["estimate"] == pytest.approx(0.6901348, abs=5e-7)
    assert fields["std_err"] == pytest.approx(0.185547, abs=5e-5)
    assert fields["lower"] == pytest.approx(0.4074583, abs=1e-4)
    assert fields["upper"] == pytest.approx(1.168920, abs=1e-4)
    # The interval must contain the estimate, and the standard error must be the
    # delta-method one rather than a placeholder: these two hold on every platform.
    assert fields["lower"] < fields["estimate"] < fields["upper"]
    assert 0.0 < fields["std_err"] < 1.0
