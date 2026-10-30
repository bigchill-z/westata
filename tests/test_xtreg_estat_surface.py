"""The complete ``estat`` surface for ``xtreg``, checked against the dispatcher.

``xtreg_estat.ado:5`` names the subcommands it handles beyond ``estat_default``:

    local allowed_butsum "hausman mundlak vce"

and ``:20-25`` routes ``su``/``ic`` to ``estat_default`` while rejecting anything else
with "{bf:estat} {bf:`subcmd'} not valid" (exit 321).  So the surface is exactly five
entries -- no more, and the refusals matter as much as the successes:

| subcommand | offered |
|---|---|
| `summarize` | yes (via `estat_default`) |
| `ic` | yes (via `estat_default`) |
| `vce` | yes |
| `hausman` | yes |
| `mundlak` | yes |
| everything else | refused |

Round 170 added ``mundlak``, which was the last missing entry. Rounds 84 and 102 showed
that inferring this list from ``e()`` contents errs in both directions, so the ado is the
authority.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def panel_fits():
    rng = np.random.default_rng(960)
    entities, periods = 50, 6
    entity = np.repeat(np.arange(entities), periods)
    n = entity.size
    covariate = rng.normal(size=n)
    effect = rng.normal(size=entities)
    frame = pd.DataFrame(
        {
            "y": 1.0 + 0.8 * covariate + effect[entity] + rng.normal(size=n),
            "x1": covariate,
            "firm": entity,
            "year": np.tile(np.arange(periods), entities),
        }
    )
    random_effects = ws.RandomEffects.from_formula(
        "y ~ x1", frame, entity="firm", time="year"
    ).fit()
    fixed_effects = ws.PanelOLS.from_formula(
        "y ~ x1", frame, entity="firm", time="year"
    ).fit()
    return frame, random_effects, fixed_effects


# --------------------------------------------------------------- the five entries
def test_summarize_works(panel_fits) -> None:
    assert panel_fits[1].estat_summarize() is not None


def test_ic_works(panel_fits) -> None:
    assert panel_fits[1].estat_ic() is not None


def test_vce_works(panel_fits) -> None:
    assert panel_fits[1].estat_vce() is not None


def test_hausman_works(panel_fits) -> None:
    _, random_effects, fixed_effects = panel_fits

    rendered = fixed_effects.hausman(random_effects)

    assert np.isfinite(float(rendered.statistic))


def test_mundlak_works(panel_fits) -> None:
    _, random_effects, _ = panel_fits

    rendered = random_effects.estat_mundlak(entity="firm", time="year")

    assert np.isfinite(float(rendered.statistic))


# ------------------------------------------------------------------- the refusals
@pytest.mark.parametrize(
    "method",
    ["estat_hettest", "estat_ovtest", "estat_imtest", "estat_vif", "estat_gof"],
)
def test_commands_outside_the_list_are_refused(panel_fits, method) -> None:
    """``xtreg_estat.ado:22-25`` rejects anything not in the five.

    A number here would be an invention: these tests are defined for other estimators,
    and ``estat vif`` returning the pooled-regression VIF was a real bug once (round 102).
    """

    with pytest.raises((ValueError, NotImplementedError)):
        getattr(panel_fits[1], method)()


def test_the_refusal_is_explained_not_bare(panel_fits) -> None:
    """``estat_hettest`` names the estimator it is unavailable for."""

    with pytest.raises(ValueError) as caught:
        panel_fits[1].estat_hettest()

    message = str(caught.value).lower()
    assert "xtreg" in message or "panel" in message or "not available" in message


def test_the_dispatch_list_is_pinned_to_the_ado() -> None:
    """Read the authority directly, so a future addition has to argue with it."""

    from pathlib import Path

    ado = (
        Path(r"D:\Program Files\StataNow19\ado\base\x\xtreg_estat.ado")
    ).read_text(encoding="utf-8", errors="ignore")

    assert 'local allowed_butsum "hausman mundlak vce"' in ado
    # And the two routed to estat_default.
    assert 'substr(`"`subcmd\'"\',1, 2) == "su"' in ado
    assert '`"`subcmd\'"\' == "ic"' in ado


def test_mundlak_is_not_offered_for_a_plain_ols_fit() -> None:
    """It is an ``xtreg`` command; a cross-section has no within-group means to test."""

    rng = np.random.default_rng(962)
    frame = pd.DataFrame({"y": rng.normal(size=100), "x": rng.normal(size=100)})
    plain = ws.OLS.from_formula("y ~ x", frame).fit()

    with pytest.raises(NotImplementedError):
        plain.estat_mundlak(entity="x")