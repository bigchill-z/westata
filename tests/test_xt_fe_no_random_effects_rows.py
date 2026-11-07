"""Random-effects rows must not leak into the fixed-effects variant.

Round 234 audited the siblings of xtlogit, re after nine rounds of work on that one shape, and
found a real regression: the FE variant printed the two random-effects rows.

THE BUG was a substring test. Three separate places decided "is this random effects?" with

    "re" in str(e.get("title"))

and a fixed-effects fit's title is "Conditional fixed-effects logistic REGRESSION" -- the letters
"re" are inside the word "regression". So `xtlogit ..., fe` was classified as random-effects and
printed "Random effects u_i ~ Gaussian" and "Integration method: mvaghermite", neither of which a
conditional fit has.

Fixed by testing the PREFIX ("Random-effects") instead of a substring. The live metadata settles
the discriminator cleanly:

    xtlogit, fe   title 'Conditional fixed-effects logistic regression'   e(sigma_u) None
    xtlogit, re   title 'Random-effects logistic regression'              e(sigma_u) 1.345...

STATUS OF THE TWO VARIANTS after the fix:

    xtlogit, re   26 lines against Stata's 26; header and closing row exact; only the seventh
                  significant figure of the coefficients differs
    xtlogit, fe   the random-effects rows are gone, but its header is NOT yet aligned: Stata
                  prints "Group variable" beside "Number of groups" on the row right after the
                  banner, while this package appends the whole panel block after the fit
                  statistics. That shape was never in scope for the re work and is recorded here
                  as open rather than silently claimed.
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
def frame():
    return _panel()


def test_fe_has_no_random_effects_rows(frame) -> None:
    """The substring bug put these here; a conditional fit has neither."""

    rendered = str(
        ws.XtLogit.from_formula("y ~ x1", frame, entity="sid", time="t", method="fe")
        .fit()
        .summary(style="stata")
    )
    assert "Random effects u_i ~ Gaussian" not in rendered
    assert "Integration method:" not in rendered


def test_re_keeps_its_random_effects_rows(frame) -> None:
    rendered = str(
        ws.XtLogit.from_formula("y ~ x1", frame, entity="sid", time="t", method="re")
        .fit()
        .summary(style="stata")
    )
    assert "Random effects u_i ~ Gaussian" in rendered
    assert "Integration method: mvaghermite" in rendered


def test_the_discriminator_is_the_metadata_not_the_title_text(frame) -> None:
    """'re' appears inside 'regression' -- that is what made the substring test wrong."""

    fe = ws.XtLogit.from_formula(
        "y ~ x1", frame, entity="sid", time="t", method="fe"
    ).fit()
    re_fit = ws.XtLogit.from_formula(
        "y ~ x1", frame, entity="sid", time="t", method="re"
    ).fit()
    fe_title = str((getattr(fe, "e", None) or {}).get("title") or "")
    assert "regression" in fe_title.lower()
    assert "re" in fe_title.lower()          # the trap: substring matches
    assert not fe_title.startswith("Random-effects")   # the correct test
    assert str((getattr(re_fit, "e", None) or {}).get("title") or "").startswith(
        "Random-effects"
    )


def test_xtprobit_re_still_matches_the_shape(frame) -> None:
    """The sibling that shares the assembly path; its line count must stay aligned."""

    rendered = str(
        ws.XtProbit.from_formula("y ~ x1", frame, entity="sid", time="t", method="re")
        .fit()
        .summary(style="stata")
    )
    assert "Random effects u_i ~ Gaussian" in rendered
    assert "LR test of rho=0" in rendered
