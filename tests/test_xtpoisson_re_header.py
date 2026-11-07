"""xtpoisson, re: the count-shape header is aligned; only the fit statistics differ.

Rounds 239-241 took three passes because I kept putting the blank row in the wrong COLUMN. The
answer came from printing both lists at the assembly loop:

    idx 8   ''                       ''
    idx 9   ''                       'Wald chi2(1)'
    idx 10  'Log likelihood = ...'   'Prob > chi2'

The blank belongs in ml_right as well as ml_left. Rounds 239 and 240 added and removed it only on
the left, which is why nothing moved. With one blank on each side the block is 23 lines against
Stata's 23.

The count shape also differs from the logit shape in three ways, all now handled:

  * the random effect is GAMMA, not Gaussian ("Random effects u_i ~ Gamma");
  * a Gamma random effect is CLOSED FORM, so there is no "Integration method" row and no
    "Integration pts." -- two rows fewer than logit/probit;
  * the entity and outcome names had to be stored (round 239) or the header read
    "Group variable: ."

WHAT REMAINS is numeric, not layout: Wald chi2 121.39 against 121.38, the standard errors and
/lnalpha differ in their last digits, and the alpha row's separator differs. Those are the
optimiser-precision differences already recorded for other ML families.
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
    counts = rng.poisson(mu).astype(float)
    return pd.DataFrame(
        {"n": counts, "x1": x1, "sid": sid, "t": np.tile(np.arange(n_obs), n_grp)}
    )


@pytest.fixture(scope="module")
def fitted():
    return ws.XtPoisson.from_formula(
        "n ~ x1", _counts(), entity="sid", time="t", method="re"
    ).fit()


def test_the_entity_and_outcome_names_are_stored(fitted) -> None:
    macros = getattr(fitted, "e", None) or {}
    assert str(macros.get("ivar")) == "sid"
    assert str(macros.get("depvar")) == "n"


def test_the_header_names_the_group(fitted) -> None:
    lines = str(fitted.summary(style="stata")).splitlines()
    assert lines[2].startswith("Group variable: sid")


def test_the_random_effect_is_gamma_not_gaussian(fitted) -> None:
    rendered = str(fitted.summary(style="stata"))
    assert "Random effects u_i ~ Gamma" in rendered
    assert "Random effects u_i ~ Gaussian" not in rendered


def test_there_is_no_integration_method_row(fitted) -> None:
    """A Gamma random effect is closed form -- nothing is integrated numerically."""

    rendered = str(fitted.summary(style="stata"))
    assert "Integration method:" not in rendered
    assert "Integration pts." not in rendered


def test_the_block_is_twenty_three_lines(fitted) -> None:
    assert len(str(fitted.summary(style="stata")).splitlines()) == 23


def test_a_blank_separates_max_from_wald(fitted) -> None:
    lines = str(fitted.summary(style="stata")).splitlines()
    wald_index = next(i for i, line in enumerate(lines) if "Wald chi2(" in line)
    assert lines[wald_index - 1].strip() == ""
    assert "max" in lines[wald_index - 2]


def test_the_logit_variant_keeps_its_gaussian_and_integration_rows() -> None:
    """The count-shape rows must not leak into the binary family."""

    rng = np.random.default_rng(3)
    n_grp, n_obs = 30, 20
    sid = np.repeat(np.arange(n_grp), n_obs)
    x1 = rng.normal(size=n_grp * n_obs)
    u = np.repeat(rng.normal(scale=1.5, size=n_grp), n_obs)
    p = 1.0 / (1.0 + np.exp(-(0.3 + 0.6 * x1 + u)))
    y = (rng.random(n_grp * n_obs) < p).astype(float)
    frame = pd.DataFrame(
        {"y": y, "x1": x1, "sid": sid, "t": np.tile(np.arange(n_obs), n_grp)}
    )
    rendered = str(
        ws.XtLogit.from_formula("y ~ x1", frame, entity="sid", time="t", method="re")
        .fit()
        .summary(style="stata")
    )
    assert "Random effects u_i ~ Gaussian" in rendered
    assert "Integration method: mvaghermite" in rendered
