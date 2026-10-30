"""Tests for the Stata-ordered `b` / `V` accessors.

`regress mpg weight foreign` reports `e(b)` in the order **`weight foreign
_cons`** -- the written order with the constant last. patsy, which builds this
package's design matrices, alphabetises and puts the intercept first, so
`result.params` reads `[Intercept, foreign[T.Foreign], weight]`.

The NUMBERS are identical: reordering `regress_auto.json`'s design against our
output agrees to 1.6e-12 (`b`) and 1.6e-15 (`V`). But a caller comparing
`result.params` with Stata's `e(b)` element by element sees a difference of ~41.7
in the first slot and reasonably concludes the fit is wrong. `stata_params` and
`stata_cov_params` remove that trap.

Recovering the written order needs the formula, because patsy destroys it -- so
the estimator records `_formula_rhs_order` and passes it to the result.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import westata as ws

FIXTURES = Path("tests/conformance/stata_fixtures")


@pytest.fixture()
def auto():
    return pd.read_stata("tests/data/auto.dta")


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


# ------------------------------------------------------------ names and order
def test_stata_param_names_put_the_constant_last(auto) -> None:
    result = ws.OLS.from_formula("mpg ~ weight + foreign", auto).fit()

    assert result.stata_param_names[-1] == "_cons"
    # And the patsy spelling is gone.
    assert not any("[T." in name for name in result.stata_param_names)


def test_stata_param_names_follow_the_written_order(auto) -> None:
    """``weight + foreign`` must stay in that order, not patsy's alphabetical one."""

    result = ws.OLS.from_formula("mpg ~ weight + foreign", auto).fit()

    assert result.stata_param_names == ["weight", "foreign", "_cons"]


def test_reversing_the_formula_reverses_the_names(auto) -> None:
    """The order is read from the formula, not from the data or the alphabet."""

    result = ws.OLS.from_formula("mpg ~ foreign + weight", auto).fit()

    assert result.stata_param_names == ["foreign", "weight", "_cons"]


def test_names_match_the_fixture_exactly(auto) -> None:
    result = ws.OLS.from_formula("mpg ~ weight + foreign", auto).fit()

    assert result.stata_param_names == _fixture("regress_auto.json")["names"]


# ------------------------------------------------------------- values
def test_stata_params_reproduce_the_fixture_coefficients(auto) -> None:
    result = ws.OLS.from_formula("mpg ~ weight + foreign", auto).fit()
    expected = np.asarray(_fixture("regress_auto.json")["b"], dtype=float)

    np.testing.assert_allclose(result.stata_params.to_numpy(), expected, rtol=1e-10)


def test_stata_cov_params_reproduce_the_fixture_variance(auto) -> None:
    result = ws.OLS.from_formula("mpg ~ weight + foreign", auto).fit()
    expected = np.asarray(_fixture("regress_auto.json")["V"], dtype=float)

    np.testing.assert_allclose(result.stata_cov_params, expected, rtol=1e-10)


def test_robust_fixture_also_aligns(auto) -> None:
    result = ws.OLS.from_formula("mpg ~ weight + foreign", auto).fit(cov_type="HC1")
    fixture = _fixture("regress_auto_robust.json")

    assert result.stata_param_names == fixture["names"]
    np.testing.assert_allclose(
        result.stata_params.to_numpy(),
        np.asarray(fixture["b"], dtype=float),
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        result.stata_cov_params,
        np.asarray(fixture["V"], dtype=float),
        rtol=1e-8,
    )


def test_the_unnamed_order_and_the_stata_order_hold_the_same_numbers(auto) -> None:
    """Reordering is a permutation, not a recomputation."""

    result = ws.OLS.from_formula("mpg ~ weight + foreign", auto).fit()

    assert sorted(result.stata_params.to_numpy()) == pytest.approx(
        sorted(np.asarray(result.params, dtype=float))
    )
    # The constant is the same number, just moved.
    assert result.stata_params["_cons"] == pytest.approx(
        float(result.params["Intercept"])
    )


def test_covariance_stays_symmetric_and_positive(auto) -> None:
    result = ws.OLS.from_formula("mpg ~ weight + foreign", auto).fit()
    covariance = result.stata_cov_params

    assert np.allclose(covariance, covariance.T)
    assert np.all(np.diag(covariance) > 0)


# ---------------------------------------------------- the trap this removes
def test_raw_params_would_mislead_a_naive_comparison(auto) -> None:
    """Record WHY these accessors exist: the raw order is genuinely confusing.

    Asserting the trap keeps the accessors justified. If a future refactor made
    ``params`` Stata-ordered, this test would fail and the accessors could be
    reconsidered -- which is the right moment to reconsider them.
    """

    result = ws.OLS.from_formula("mpg ~ weight + foreign", auto).fit()
    fixture = np.asarray(_fixture("regress_auto.json")["b"], dtype=float)
    raw = np.asarray(result.params, dtype=float)

    # Element-wise the raw order disagrees badly ...
    assert np.max(np.abs(raw - fixture)) > 1.0
    # ... while the Stata order agrees.
    assert np.max(np.abs(result.stata_params.to_numpy() - fixture)) < 1e-9


# -------------------------------------------------------- built from arrays
def test_a_result_without_a_formula_still_puts_cons_last(auto) -> None:
    """With no recorded RHS order the slopes keep patsy's order; ``_cons`` last."""

    y = auto["mpg"].to_numpy(float)
    X = np.column_stack(
        [np.ones(len(auto)), auto["weight"].to_numpy(float)]
    )
    result = ws.OLS(y, X, param_names=["Intercept", "weight"]).fit()

    assert result.stata_param_names == ["weight", "_cons"]


# ------------------------------------------------- bare factor vs explicit C()
def test_a_bare_factor_column_is_named_after_the_variable(auto) -> None:
    """``foreign`` is a string column, but Stata reports one row named ``foreign``.

    ``regress_auto.json``'s names are ``['weight', 'foreign', '_cons']``, so the
    patsy spelling ``foreign[T.Foreign]`` must collapse to the bare variable name
    -- not be treated as an ambiguous factor expansion.
    """

    result = ws.OLS.from_formula("mpg ~ weight + foreign", auto).fit()

    assert result.stata_factor_terms == []
    assert result.stata_param_names == ["weight", "foreign", "_cons"]
    assert result.stata_param_names == _fixture("regress_auto.json")["names"]


def test_an_explicit_C_term_is_refused_rather_than_guessed(auto) -> None:
    """``i.foreign`` expands to one row per level plus a zero base row.

    ``regress_auto_factor.json`` has FOUR names -- ``weight``, ``0b.foreign``,
    ``1.foreign``, ``_cons`` -- with ``b = [-0.00659, 0.0, -1.65003, 41.6797]``,
    the base carrying 0. This result has three columns because patsy absorbs the
    base into the intercept.

    Returning an invented fourth row would look like a valid correspondence to
    Stata's ``e(b)`` when none exists, so the accessors refuse instead.
    """

    result = ws.OLS.from_formula("mpg ~ weight + C(foreign)", auto).fit()

    assert result.stata_factor_terms == ["foreign"]
    with pytest.raises(NotImplementedError, match="factor"):
        _ = result.stata_param_names


def test_the_covariance_accessor_refuses_the_same_cases(auto) -> None:
    """A matrix whose rows silently mismatch Stata's would be worse than an error."""

    result = ws.OLS.from_formula("mpg ~ weight + C(foreign)", auto).fit()

    with pytest.raises(NotImplementedError, match="factor"):
        _ = result.stata_cov_params


def test_the_refusal_names_the_offending_term(auto) -> None:
    result = ws.OLS.from_formula("mpg ~ weight + C(foreign)", auto).fit()

    with pytest.raises(NotImplementedError) as excinfo:
        _ = result.stata_param_names

    assert "foreign" in str(excinfo.value)
    # And points at what to use instead.
    assert "params" in str(excinfo.value)


def test_plain_results_are_unaffected_by_the_factor_rule(auto) -> None:
    result = ws.OLS.from_formula("mpg ~ weight + length", auto).fit()

    assert result.stata_factor_terms == []
    assert result.stata_param_names == ["weight", "length", "_cons"]


# ------------------------------------------- multi-level factors are ambiguous
def _multi_level_frame():
    """A five-level string column, so a bare ``g`` expands to four columns."""

    rng = np.random.default_rng(1)
    entities, periods = 20, 5
    firm = np.repeat(np.arange(entities), periods)
    n = firm.size
    x1 = rng.normal(size=n)
    g = np.tile(["a", "b", "c", "d", "e"], entities)[:n]
    effect = rng.normal(size=entities)[firm]
    y = 1.0 + 0.5 * x1 + effect + rng.normal(scale=0.4, size=n)
    return pd.DataFrame({"y": y, "x1": x1, "g": g, "firm": firm})


def test_a_bare_multi_level_column_is_refused() -> None:
    """``y ~ x1 + g`` with five levels expands to four columns, not one.

    Without this rule all four ``g[T.x]`` columns collapsed to the single name
    ``g``, so four different coefficients shared one label -- a silently wrong
    table rather than an error.
    """

    result = ws.FEOLS.from_formula(
        "y ~ x1 + g | firm", _multi_level_frame()
    ).fit()

    assert result.stata_factor_terms == ["g"]
    with pytest.raises(NotImplementedError, match="factor"):
        _ = result.stata_param_names


def test_a_binary_explicit_factor_is_still_refused(auto) -> None:
    """``i.foreign`` reports a zero base row even though it has two levels.

    Only one level column survives in this design (the base joins the intercept),
    so a "more than one level" test alone would pass it through. The ``C(...)``
    prefix is what marks it as a genuine level expansion.
    """

    result = ws.OLS.from_formula("mpg ~ weight + C(foreign)", auto).fit()

    assert result.stata_factor_terms == ["foreign"]


def test_a_bare_binary_column_still_works(auto) -> None:
    """The exclusion must not over-reach: one level column is an ordinary dummy."""

    result = ws.OLS.from_formula("mpg ~ weight + foreign", auto).fit()

    assert result.stata_factor_terms == []
    assert result.stata_param_names == _fixture("regress_auto.json")["names"]


def test_reghdfe_with_only_numeric_regressors_is_unaffected() -> None:
    """``reghdfe``'s own fixtures use plain numeric regressors."""

    frame = _multi_level_frame()
    result = ws.FEOLS.from_formula("y ~ x1 | firm", frame).fit()

    assert result.stata_factor_terms == []
    assert result.stata_param_names == ["x1"]


def test_reghdfe_preserves_the_written_order_without_a_recorded_formula() -> None:
    """patsy leaves purely numeric columns in written order, so no record is needed.

    ``OLS`` needs ``_formula_rhs_order`` because patsy alphabetises; with numeric
    columns only there is nothing to alphabetise.
    """

    frame = _multi_level_frame()
    forward = ws.FEOLS.from_formula("y ~ x1 | firm", frame).fit()

    assert getattr(forward, "_formula_rhs_order", None) is None
    assert forward.stata_param_names == ["x1"]


# ------------------------------------------------ interaction term separators
def test_interaction_terms_use_statats_hash_separator(auto) -> None:
    """Stata writes ``x1#x2``; patsy writes ``x1:x2``.

    Only the separator differs -- the position in ``e(b)`` is already right -- so
    renaming is safe rather than a guess about Stata's layout.
    """

    result = ws.OLS.from_formula("mpg ~ weight * length", auto).fit()

    assert "weight#length" in result.stata_param_names
    assert not any(":" in name for name in result.stata_param_names)


def test_interaction_order_follows_the_formula(auto) -> None:
    forward = ws.OLS.from_formula("mpg ~ weight * length", auto).fit()
    reverse = ws.OLS.from_formula("mpg ~ length * weight", auto).fit()

    assert forward.stata_param_names == [
        "weight",
        "length",
        "weight#length",
        "_cons",
    ]
    assert reverse.stata_param_names == [
        "length",
        "weight",
        "length#weight",
        "_cons",
    ]


def test_stata_params_after_an_interaction_stay_finite(auto) -> None:
    result = ws.OLS.from_formula("mpg ~ weight * length", auto).fit()

    values = result.stata_params.to_numpy()

    assert np.all(np.isfinite(values))
    # Same multiset as the unrenamed accessor: a rename, not a recomputation.
    assert sorted(values) == pytest.approx(
        sorted(np.asarray(result.params, dtype=float))
    )


# --------------------------------------------------- numeric multi-valued column
def test_a_numeric_three_valued_column_is_a_continuous_regressor(auto) -> None:
    """Stata treats a bare numeric ``rep78`` as continuous, so no levels expand.

    This is the mirror of the string case: the same column written as a STRING
    would factor-code under patsy and be refused, while as a number it is one
    coefficient and must pass.
    """

    result = ws.OLS.from_formula("mpg ~ weight + rep78", auto).fit()

    assert result.stata_factor_terms == []
    assert result.stata_param_names == ["weight", "rep78", "_cons"]


# --------------------------------------------- auxiliary parameters come last
def test_auxiliary_parameters_lose_the_display_slash(auto) -> None:
    """Stata DISPLAYS ``/lnalpha``, ``/cut1`` but stores them without the slash.

    ``nbreg.json``'s names are ``['x1', '_cons', 'lnalpha']`` and
    ``ologit.json``'s are ``['x1', 'x2', 'cut1', 'cut2']`` -- no slashes in
    either.  The slash is a formatting convention of the printed table, not part
    of ``e(b)``.
    """

    nbreg = ws.NegativeBinomial.from_formula("mpg ~ weight", auto).fit()
    ordered = ws.OrderedLogit.from_formula("rep78 ~ weight", auto).fit()

    # The internal spelling keeps the slash ...
    assert any(name.startswith("/") for name in nbreg.param_names)
    assert any(name.startswith("/") for name in ordered.param_names)
    # ... while Stata's e(b) does not.
    assert not any(name.startswith("/") for name in nbreg.stata_param_names)
    assert not any(name.startswith("/") for name in ordered.stata_param_names)
    assert "lnalpha" in nbreg.stata_param_names


def test_the_dispersion_follows_the_constant(auto) -> None:
    """``nbreg.json``: ``['x1', '_cons', 'lnalpha']`` -- aux parameters are LAST."""

    result = ws.NegativeBinomial.from_formula("mpg ~ weight", auto).fit()

    assert result.stata_param_names == ["weight", "_cons", "lnalpha"]


def test_ordinal_cut_points_come_after_the_slopes(auto) -> None:
    result = ws.OrderedLogit.from_formula("rep78 ~ weight", auto).fit()
    names = result.stata_param_names

    assert names[0] == "weight"
    assert names[1:] == ["cut1", "cut2", "cut3", "cut4"]


def test_auxiliary_parameters_do_not_disturb_the_plain_ols_order(auto) -> None:
    """Regression guard: the aux rule must not move ordinary regressors."""

    result = ws.OLS.from_formula("mpg ~ weight + foreign", auto).fit()

    assert result.stata_param_names == _fixture("regress_auto.json")["names"]


def test_logit_has_no_auxiliary_parameters_to_move(auto) -> None:
    frame = auto.assign(dom=(auto["foreign"] == "Foreign").astype(int))
    result = ws.Logit.from_formula("dom ~ weight + mpg", frame).fit()

    assert result.stata_param_names[-1] == "_cons"
    assert all(name == "_cons" or not name.startswith("cut") for name in result.stata_param_names)


# ---------------------------------------- IV reports endogenous regressors first
@pytest.fixture()
def iv_frame():
    rng = np.random.default_rng(1)
    n = 200
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    u = rng.normal(size=n)
    x = 0.7 * z1 + 0.3 * z2 + 0.6 * u + rng.normal(size=n)
    y = 1.0 + 2.0 * x + u
    return pd.DataFrame({"y": y, "x": x, "z1": z1, "z2": z2})


def test_iv_puts_the_endogenous_regressor_first(iv_frame) -> None:
    """``iv_liml.json``: cmdline ``ivregress liml y x2 (x = z1 z2)``, names
    ``['x', 'x2', '_cons']`` -- the endogenous ``x`` precedes the exogenous
    ``x2``.

    This package builds the design as ``exog + endog``, so the two orders differ
    whenever both blocks are present.
    """

    result = ws.IV2SLS.from_formula("y ~ z1 + [x ~ z2]", iv_frame).fit()

    # The design order is exogenous-first ...
    assert list(result.param_names) == ["Intercept", "z1", "x"]
    # ... while Stata's is endogenous-first.
    assert result.stata_param_names == ["x", "z1", "_cons"]


def test_iv_without_included_exogenous_regressors(iv_frame) -> None:
    result = ws.IV2SLS.from_formula("y ~ [x ~ z1 + z2]", iv_frame).fit()

    assert result.stata_param_names == ["x", "_cons"]


def test_iv_rhs_order_excludes_the_constant(iv_frame) -> None:
    result = ws.IV2SLS.from_formula("y ~ z1 + [x ~ z2]", iv_frame).fit()

    assert result._formula_rhs_order == ["x", "z1"]
    assert "Intercept" not in result._formula_rhs_order


def test_iv_stata_params_hold_the_same_numbers(iv_frame) -> None:
    """Reordering must be a permutation, not a different fit."""

    result = ws.IV2SLS.from_formula("y ~ z1 + [x ~ z2]", iv_frame).fit()

    assert sorted(result.stata_params.to_numpy()) == pytest.approx(
        sorted(np.asarray(result.params, dtype=float))
    )
    # ``x`` is the same coefficient, just moved to the front.
    assert result.stata_params["x"] == pytest.approx(float(result.params["x"]))


def test_iv_covariance_permutes_consistently(iv_frame) -> None:
    result = ws.IV2SLS.from_formula("y ~ z1 + [x ~ z2]", iv_frame).fit()
    covariance = result.stata_cov_params

    assert covariance.shape == (3, 3)
    assert np.allclose(covariance, covariance.T)
    # The variance of ``x`` is unchanged by the permutation.
    assert covariance[0, 0] == pytest.approx(
        float(np.diag(np.asarray(result.cov_params(), dtype=float))[2])
    )
