"""``F_f`` -- the "F test that all u_i = 0" statistic for ``xtreg, fe``.

The formula is in ``xtreg_fe.ado:505-506, 524`` and it contains a trap:

```stata
/* not the same as e(r2) returned from _regress */
scalar `r2' = 1 - e(rss)/e(tss)
ereturn scalar F_f = ((`r2'-`r2c')/e(df_a))/((1-`r2')/e(df_r))
```

The ``r2`` in that formula is ``1 - rss/tss``, NOT ``e(r2)``. Measured on
``xtreg_fe.json`` the two are 0.9531797 and 0.9299367 -- and substituting
``e(r2)`` produces 17.075 where the fixture records 28.085.

That failure mode is visible without a fixture at all: using ``e(r2)`` on weakly
clustered data gave a NEGATIVE F statistic (-0.00891), which cannot exist.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

import westata as ws

FIXTURE = Path("tests/conformance/stata_fixtures/xtreg_fe.json")


def _fixture_scalars() -> dict[str, float]:
    return json.loads(FIXTURE.read_text())["scalars"]


def test_f_f_matches_the_fixture_arithmetic() -> None:
    """Recompute the fixture's own F_f from its own stored pieces.

    ``1 - rss/tss`` (not ``e(r2)``), then the df-scaled ratio.  This pins the
    FORMULA against captured Stata output without needing a re-run.
    """

    s = _fixture_scalars()
    r2_used = 1.0 - s["rss"] / s["tss"]

    expected = ((r2_used - s["r2_c"]) / s["df_a"]) / ((1.0 - r2_used) / s["df_r"])

    assert expected == pytest.approx(s["F_f"], rel=1e-9)


def test_the_two_r2_values_really_differ_in_the_fixture() -> None:
    """If they were equal the distinction would be academic; they are not."""

    s = _fixture_scalars()
    r2_formula = 1.0 - s["rss"] / s["tss"]

    assert r2_formula != pytest.approx(s["r2"], rel=1e-6)
    assert r2_formula == pytest.approx(0.9531797308802162, rel=1e-9)


def test_using_e_r2_instead_gives_a_different_wrong_number() -> None:
    """The wrong substitution is not a small error -- it is 17.075 against 28.085.

    Recording the wrong value makes the trap concrete: a future refactor that
    "simplifies" the code to read ``e(r2)`` must fail this test.
    """

    s = _fixture_scalars()
    wrong = ((s["r2"] - s["r2_c"]) / s["df_a"]) / ((1.0 - s["r2"]) / s["df_r"])

    assert wrong == pytest.approx(17.075086283267392, rel=1e-9)
    assert not np.isclose(wrong, s["F_f"], rtol=1e-3)


def test_the_stored_p_f_confirms_the_degrees_of_freedom() -> None:
    """``f.sf(F_f, df_a, df_r)`` reproduces the fixture's ``p_f`` exactly.

    That is what identifies 29 and 148 as the right degrees of freedom rather than
    guessing from the counts.
    """

    s = _fixture_scalars()

    assert scipy_stats.f.sf(s["F_f"], s["df_a"], s["df_r"]) == pytest.approx(
        s["p_f"], rel=1e-9
    )


# -------------------------------------------------------------- live fits
def _panel_frame(*, entity_effect: float, seed: int) -> pd.DataFrame:
    """A panel whose within-entity scatter scales with ``entity_effect``.

    A LARGE effect means entity means absorb little of the within variation, so the
    within R2 dominates the pooled one and ``F_f`` is large.  A small effect makes
    the two comparable -- the case that produced a negative statistic when the
    wrong R2 was substituted.
    """

    rng = np.random.default_rng(seed)
    entities, periods = 40, 5
    firm = np.repeat(np.arange(entities), periods)
    n = firm.size
    effect = rng.normal(scale=entity_effect, size=entities)[firm]
    x1 = rng.normal(size=n)
    y = 0.8 * x1 + effect + rng.normal(size=n)
    return pd.DataFrame(
        {"y": y, "x1": x1, "firm": firm, "year": np.tile(np.arange(periods), entities)}
    )


@pytest.mark.parametrize("effect", [0.0, 0.5, 3.0])
def test_f_f_is_never_negative(effect) -> None:
    """An F statistic is a ratio of mean squares, so it cannot be below zero.

    With the wrong R2 substituted this failed at ``effect=0`` (giving -0.00891);
    it is the cheapest possible guard against the mistake.
    """

    frame = _panel_frame(entity_effect=effect, seed=311)
    e = ws.PanelOLS.from_formula(
        "y ~ x1", frame, entity="firm", time="year"
    ).fit().e

    assert np.isfinite(e["F_f"])
    assert e["F_f"] >= 0.0


def test_a_strong_entity_effect_gives_a_large_f_f() -> None:
    """The statistic tests whether the entity effects are jointly zero.

    With a strong effect it must be large; with none it must be small.  Both ends
    are asserted so the direction cannot silently invert.
    """

    strong = _panel_frame(entity_effect=3.0, seed=313)
    weak = _panel_frame(entity_effect=0.0, seed=313)

    strong_f = ws.PanelOLS.from_formula(
        "y ~ x1", strong, entity="firm", time="year"
    ).fit().e["F_f"]
    weak_f = ws.PanelOLS.from_formula(
        "y ~ x1", weak, entity="firm", time="year"
    ).fit().e["F_f"]

    assert strong_f > weak_f
    assert strong_f > 1.0


def test_r2_c_is_the_pooled_ols_r_squared() -> None:
    """``r2_c`` comes from a plain pooled regression on the same regressors.

    ``xtreg_fe.ado:517-518`` fits ``_regress y xvars`` and stores its ``e(r2)``.
    """

    frame = _panel_frame(entity_effect=1.0, seed=317)
    e = ws.PanelOLS.from_formula(
        "y ~ x1", frame, entity="firm", time="year"
    ).fit().e
    pooled = ws.OLS.from_formula("y ~ x1", frame).fit().e["r2"]

    assert e["r2_c"] == pytest.approx(pooled, rel=1e-10)


def test_df_a_is_the_number_of_entities_minus_one() -> None:
    """``e(df_a)`` is the absorbed degrees of freedom."""

    frame = _panel_frame(entity_effect=1.0, seed=319)
    e = ws.PanelOLS.from_formula(
        "y ~ x1", frame, entity="firm", time="year"
    ).fit().e

    assert e["df_a"] == float(frame["firm"].nunique() - 1)


def test_the_random_effects_estimator_has_no_f_f() -> None:
    """``F_f`` belongs to the absorbed-effects block, which is fixed effects only."""

    frame = _panel_frame(entity_effect=1.0, seed=323)
    e = ws.RandomEffects.from_formula(
        "y ~ x1", frame, entity="firm", time="year"
    ).fit().e

    assert "F_f" not in e


# ---------------------------------------------------------------- rendering
def test_the_test_is_printed_below_the_table() -> None:
    """``whatsnew13.sthlp:1165-1168`` names the printed text EXACTLY.

    "This affected only the value of the test statistic reported below the results
    table as 'F test that all u_i=0'."  That is Stata's own wording, so the render
    must use it rather than a paraphrase like "F test of u_i=0".
    """

    frame = _panel_frame(entity_effect=1.0, seed=331)
    rendered = ws.PanelOLS.from_formula(
        "y ~ x1", frame, entity="firm", time="year"
    ).fit().summary(style="stata")

    assert "F test that all u_i=0" in rendered
    assert "F test of u_i=0" not in rendered  # the paraphrase must not sneak back


def test_the_rendered_statistic_is_the_stored_value() -> None:
    """The printed F and p must be ``e(F_f)`` / ``e(p_f)``, not recomputed."""

    from westata.results.summary import _stata_coef

    frame = _panel_frame(entity_effect=1.0, seed=337)
    result = ws.PanelOLS.from_formula(
        "y ~ x1", frame, entity="firm", time="year"
    ).fit()
    e = result.e
    rendered = result.summary(style="stata")

    assert _stata_coef(float(e["F_f"]), 8).strip() in rendered
    assert f"{e['p_f']:.4f}" in rendered


def test_the_test_line_sits_after_the_coefficient_table() -> None:
    """Stata reports it BELOW the results table, after the decomposition block."""

    frame = _panel_frame(entity_effect=1.0, seed=347)
    rendered = ws.PanelOLS.from_formula(
        "y ~ x1", frame, entity="firm", time="year"
    ).fit().summary(style="stata")
    lines = rendered.splitlines()

    f_index = next(i for i, line in enumerate(lines) if "F test that all u_i=0" in line)
    cons_index = next(i for i, line in enumerate(lines) if line.strip().startswith("_cons"))
    sigma_index = next(i for i, line in enumerate(lines) if "sigma_u |" in line)

    assert cons_index < sigma_index < f_index


def test_the_degrees_of_freedom_pair_is_dfa_and_dfr() -> None:
    """``F(df_a, df_r)`` -- the absorbed df and the residual df."""

    frame = _panel_frame(entity_effect=1.0, seed=349)
    result = ws.PanelOLS.from_formula(
        "y ~ x1", frame, entity="firm", time="year"
    ).fit()
    e = result.e
    rendered = result.summary(style="stata")

    assert f"F({int(e['df_a'])}, {int(e['df_r'])})" in rendered


def test_a_random_effects_fit_prints_no_such_test() -> None:
    """No ``F_f`` means no line -- not a blank one."""

    frame = _panel_frame(entity_effect=1.0, seed=353)
    rendered = ws.RandomEffects.from_formula(
        "y ~ x1", frame, entity="firm", time="year"
    ).fit().summary(style="stata")

    assert "F test that all u_i=0" not in rendered


def test_an_ols_fit_prints_no_panel_test() -> None:
    """``regress`` has neither ``sigma_u`` nor ``F_f``, so neither block appears."""

    frame = _panel_frame(entity_effect=1.0, seed=359)
    rendered = ws.OLS.from_formula("y ~ x1", frame).fit().summary(style="stata")

    assert "F test that all u_i=0" not in rendered
    assert "sigma_u |" not in rendered