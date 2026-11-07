"""xtlogit, fe: the conditional header is now aligned with Stata, row for row.

Round 238 finished what rounds 235-237 went back and forth on. The answer came from printing the
two column lists at the assembly loop rather than reasoning about indices, which is the same method
that settled the ML leading blank (199) and the ml_left overlap (227).

The row map, measured:

    idx 8   (blank)              (blank)
    idx 9   (blank)              LR chi2(1)
    idx 10  Log likelihood = ... Prob > chi2

So there are TWO blanks between "max" and the LR row, and the left column carries one entry more
than the right before the log likelihood is placed -- the two wrong turns earlier rounds took were
one blank too many (ll on row 11) and one too few (ll on row 9).

A conditional fit has no distribution line for u_i and integrates nothing out, so unlike the
random-effects variant it prints NO "Random effects u_i ~ Gaussian" and NO "Integration method".
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
    return ws.XtLogit.from_formula("y ~ x1", _panel(), entity="sid", time="t", method="fe").fit()


def test_log_likelihood_sits_on_the_prob_row(fitted) -> None:
    lines = str(fitted.summary(style="stata")).splitlines()
    ll_index = next(i for i, line in enumerate(lines) if line.startswith("Log likelihood"))
    prob_index = next(i for i, line in enumerate(lines) if "Prob > chi2" in line)
    assert ll_index == prob_index


def test_the_lr_row_has_an_empty_left_cell(fitted) -> None:
    """Stata leaves it blank; the log likelihood goes on the row BELOW."""

    lines = str(fitted.summary(style="stata")).splitlines()
    lr_index = next(i for i, line in enumerate(lines) if "LR chi2(" in line)
    assert lines[lr_index][:53].strip() == ""


def test_the_gap_between_max_and_the_lr_row(fitted) -> None:
    """One blank sits between the per-group block and the LR row, and the LR row's own
    left cell is blank too -- so the log likelihood lands two rows below "max"."""

    lines = str(fitted.summary(style="stata")).splitlines()
    lr_index = next(i for i, line in enumerate(lines) if "LR chi2(" in line)
    assert "max" in lines[lr_index - 2]
    assert lines[lr_index - 1].strip() == ""


def test_the_conditional_variant_has_no_random_effects_rows(fitted) -> None:
    rendered = str(fitted.summary(style="stata"))
    assert "Random effects u_i ~ Gaussian" not in rendered
    assert "Integration method:" not in rendered


def test_the_random_effects_variant_is_unaffected() -> None:
    """re keeps its own map: Gaussian and Integration rows present, ll on the Prob row."""

    rendered = str(
        ws.XtLogit.from_formula("y ~ x1", _panel(), entity="sid", time="t", method="re")
        .fit()
        .summary(style="stata")
    )
    assert "Random effects u_i ~ Gaussian" in rendered
    assert "Integration method: mvaghermite" in rendered
    lines = rendered.splitlines()
    ll_index = next(i for i, line in enumerate(lines) if line.startswith("Log likelihood"))
    prob_index = next(i for i, line in enumerate(lines) if "Prob > chi2" in line)
    assert ll_index == prob_index
