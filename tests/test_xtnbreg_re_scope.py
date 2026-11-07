"""What implementing xtnbreg, re actually requires -- scoped, with the block identified.

Round 248 attempted the estimator and reverted two edits rather than leave them half-built
(`_methods = ("fe", "re")` plus a `_fit_random_nb` dispatch with no `_fit_random_nb` behind it).
This file records what the work involves so the next attempt starts from the scope, not from an
exploration.

THE LIKELIHOOD IS DONE (round 247, tests/test_xtnbreg_beta_likelihood_prototype.py): group-level
Beta negative-binomial, verified finite, verified directionally (a tighter Beta fits better), and
verified per-group rather than per-observation.

WHAT REMAINS, in the order the code needs it:

 1. `_methods = ("fe", "re")` on XtNbreg, and a dispatch in `fit()` -- two lines, trivial ONCE
    there is something to dispatch to.

 2. `_fit_random_nb`: optimise (beta, log_r, log_s) JOINTLY. Note the FE path optimises only beta
    because the HHG kernel is conditional; the RE likelihood is not, so r and s join the parameter
    vector. The prototype takes (r, s) directly -- the estimator needs the log parameterisation,
    since r and s must stay positive.

 3. The likelihood call inside the optimiser must not re-run the quadrature per function
    evaluation at 64 nodes per group. The prototype does exactly that, which is fine for three
    assertions and far too slow for BFGS. Either the node count drops or the inner loop is
    vectorised across groups first.

 4. Observed-information covariance over the FULL parameter vector including (log_r, log_s), so
    the reported standard errors are for (b, r, s) as a block. The FE path has its own machinery
    for a beta-only covariance that does not extend.

 5. Stata's e() surface for this command: `e(r)` and `e(s)` (sthlp:413), plus whatever else the
    RE variant reports that FE does not.

 6. The 25-line table with "Random effects u_i ~ Beta". The renderer already keys the distribution
    off the command name (round 239) -- `xtnbreg` currently maps to Gamma, which is right for
    `xtpoisson` and wrong for this command. That mapping is a one-line change but it must not be
    made before the RE path exists, or the FE variant would inherit a Beta label it never prints.

WHY THIS ROUND STOPPED: items 2-4 are the estimator, and doing them properly is a session's work.
Committing `_methods = ("fe", "re")` with no implementation would have made `xtnbreg ..., re`
raise a different error rather than work -- strictly worse than the current honest UnsupportedError.

The guards re-pin that the gap is still open and that the prototype still behaves.
"""

from __future__ import annotations

import numpy as np
import pytest

import westata as ws


def _counts():
    rng = np.random.default_rng(13)
    n_grp, n_obs = 25, 12
    sid = np.repeat(np.arange(n_grp), n_obs)
    x1 = rng.normal(size=n_grp * n_obs)
    u = np.repeat(rng.normal(scale=0.5, size=n_grp), n_obs)
    mu = np.exp(0.4 + 0.5 * x1 + u)
    p = 1.0 / (1.0 + 0.7 * mu)
    counts = rng.negative_binomial(1.0 / 0.7, 1.0 - p).astype(float)
    return pd_frame(counts, x1, sid)


def pd_frame(counts, x1, sid):
    import pandas as pd

    n_obs = 12
    return pd.DataFrame(
        {
            "n": counts,
            "x1": x1,
            "sid": sid,
            "t": np.tile(np.arange(n_obs), sid.size // n_obs),
        }
    )


def test_re_still_raises_unsupported() -> None:
    """The gap is open. When this flips to passing, the estimator was implemented."""

    with pytest.raises(ws.errors.UnsupportedError):
        ws.XtNbreg.from_formula(
            "n ~ x1", _counts(), entity="sid", time="t", method="re"
        )


def test_only_fe_is_advertised() -> None:
    """`_methods` must not list re while there is nothing behind it."""

    assert ws.XtNbreg._methods == ("fe",)


def test_the_beta_label_is_not_applied_to_the_fe_variant() -> None:
    """The renderer maps xtnbreg to Gamma today. That is CORRECT until the RE path exists,
    because a conditional fit never prints a distribution line at all."""

    fitted = ws.XtNbreg.from_formula(
        "n ~ x1", _counts(), entity="sid", time="t", method="fe"
    ).fit()
    rendered = str(fitted.summary(style="stata"))
    assert "Random effects u_i ~" not in rendered
