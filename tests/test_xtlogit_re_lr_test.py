"""xtlogit, re: the closing "LR test of rho=0" row.

The last block of the xtlogit, re output was simply absent:

    LR test of rho=0: chibar2(01) = 98.28                  Prob >= chibar2 = 0.000

It is the same boundary test the alpha branch already implements for ``xtpoisson, re`` and
``tnbreg`` -- statistic 2*(ll - ll_0), p-value HALF the chi2(1) tail because the null sits on the
edge of the parameter space -- so it reuses that machinery with ``rho`` in place of ``alpha``:

    2 * (-348.7264 + 397.8659) = 98.28      matches Stata's 98.28

The row is 78 wide with "Prob >= chibar2" starting at column 55. My first version used
``row_width - len(tail) - 1`` and came out one column short; the live comparison caught it.

Gated on an xt model that stores e(rho) and has no alpha parameter, so ``xtpoisson, re`` keeps its
own "LR test of alpha=0" line.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


def _panel():
    rng = np.random.default_rng(3)
    n_grp, n_obs = 30, 20
    sid = np.repeat(np.arange(n_grp), n_obs)
    x1 = rng.normal(size=n_grp * n_obs)
    u = np.repeat(rng.normal(scale=1.5, size=n_grp), n_obs)
    p = 1.0 / (1.0 + np.exp(-(0.3 + 0.6 * x1 + u)))
    y = (rng.random(n_grp * n_obs) < p).astype(float)
    return pd.DataFrame(
        {"y": y, "x1": x1, "sid": sid, "t": np.tile(np.arange(n_obs), n_grp)}
    )


@pytest.fixture(scope="module")
def fitted():
    return ws.XtLogit.from_formula("y ~ x1", _panel(), entity="sid", time="t").fit()


def test_the_block_closes_with_the_rho_test(fitted) -> None:
    lines = str(fitted.summary(style="stata")).splitlines()
    assert lines[-1].startswith("LR test of rho=0: chibar2(01) =")


def test_the_statistic_comes_from_the_likelihoods(fitted) -> None:
    """2*(ll - ll_0) = 98.28, matching Stata's printed value."""

    macros = getattr(fitted, "e", None) or {}
    expected = 2.0 * (float(macros["ll"]) - float(macros["ll_0"]))
    lines = str(fitted.summary(style="stata")).splitlines()
    assert f"= {expected:.2f}" in lines[-1]


def test_the_probability_is_half_the_chi2_tail(fitted) -> None:
    """The 50:50 mixture matters: an ordinary chi2(1) tail would print 0.000 too, but a
    boundary test with a small statistic would differ. Pinned via the printed 3 decimals."""

    lines = str(fitted.summary(style="stata")).splitlines()
    assert lines[-1].rstrip().endswith("Prob >= chibar2 = 0.000")


def test_the_row_is_seventy_eight_wide(fitted) -> None:
    lines = str(fitted.summary(style="stata")).splitlines()
    assert len(lines[-1]) == 78
    assert lines[-1].index("Prob") == 55


def test_xtpoisson_keeps_its_alpha_wording() -> None:
    """The alpha branch must not be shadowed by the new rho branch."""

    rng = np.random.default_rng(5)
    n_grp, n_obs = 20, 15
    sid = np.repeat(np.arange(n_grp), n_obs)
    x1 = rng.normal(size=n_grp * n_obs)
    effect = np.exp(0.4 + 0.5 * x1 + np.repeat(rng.normal(scale=0.6, size=n_grp), n_obs))
    counts = rng.poisson(effect).astype(float)
    frame = pd.DataFrame(
        {"n": counts, "x1": x1, "sid": sid, "t": np.tile(np.arange(n_obs), n_grp)}
    )
    rendered = str(
        ws.XtPoisson.from_formula("n ~ x1", frame, entity="sid", time="t").fit().summary(
            style="stata"
        )
    )
    assert "LR test of alpha=0" in rendered
    assert "LR test of rho=0" not in rendered
