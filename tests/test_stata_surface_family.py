"""Cross-estimator consistency of the Stata coefficient surface.

Rounds 41-43 changed the Stata coefficient ordering in three places: auxiliary
parameters (`nbreg`/`ologit`), IV's endogenous-first rule, and the interaction
separator. Each was verified on its own; this file checks the properties that must
hold for EVERY estimator, so the guarantees cannot drift apart again.

The properties:

1. ``stata_param_names`` has no duplicates -- the round-39 defect was four
   coefficients sharing the label ``g``;
2. ``stata_params`` is a permutation of ``params`` -- a rename, never a refit;
3. ``stata_cov_params`` is symmetric with the same diagonal multiset;
4. ``_cons`` precedes any auxiliary parameter (``cut1``.., ``lnalpha``);
5. results that genuinely cannot be mapped raise ``NotImplementedError`` rather
   than returning an invented row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

N = 300
ENTITIES, PERIODS = 50, 6


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(43)
    n = N
    firm = np.repeat(np.arange(ENTITIES), PERIODS)
    year = np.tile(np.arange(PERIODS), ENTITIES)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    effect = rng.normal(size=ENTITIES)[firm]
    y = 1.0 + 0.6 * x1 - 0.4 * x2 + effect + rng.normal(scale=0.5, size=n)
    y_bin = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.8 * x1)))).astype(float)
    y_count = rng.poisson(np.exp(0.4 + 0.4 * x1)).astype(float)
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = 0.5 * u[t - 1] + rng.normal()
    x_endo = 0.7 * z1 + 0.3 * z2 + 0.5 * u + rng.normal(size=n)
    y_iv = 1.0 + 2.0 * x_endo + u
    return pd.DataFrame(
        {
            "y": y,
            "y_bin": y_bin,
            "y_count": y_count,
            "y_iv": y_iv,
            "x1": x1,
            "x2": x2,
            "x_endo": x_endo,
            "z1": z1,
            "z2": z2,
            "firm": firm,
            "year": year,
            "rep": rng.integers(1, 4, size=n).astype(float),
        }
    )


def _estimators(frame):
    return {
        "OLS": lambda: ws.OLS.from_formula("y ~ x1 + x2", frame).fit(),
        "OLS interaction": lambda: ws.OLS.from_formula("y ~ x1 * x2", frame).fit(),
        "FEOLS one-way": lambda: ws.FEOLS.from_formula(
            "y ~ x1 + x2 | firm", frame
        ).fit(),
        "FEOLS two-way": lambda: ws.FEOLS.from_formula(
            "y ~ x1 + x2 | firm + year", frame
        ).fit(),
        "IV2SLS": lambda: ws.IV2SLS.from_formula(
            "y_iv ~ z1 + [x_endo ~ z2]", frame
        ).fit(),
        "Logit": lambda: ws.Logit.from_formula("y_bin ~ x1 + x2", frame).fit(),
        "Probit": lambda: ws.Probit.from_formula("y_bin ~ x1 + x2", frame).fit(),
        "Poisson": lambda: ws.Poisson.from_formula("y_count ~ x1", frame).fit(),
        "NBreg": lambda: ws.NegativeBinomial.from_formula(
            "y_count ~ x1", frame
        ).fit(),
        "OrderedLogit": lambda: ws.OrderedLogit.from_formula("rep ~ x1", frame).fit(),
        "OrderedProbit": lambda: ws.OrderedProbit.from_formula(
            "rep ~ x1", frame
        ).fit(),
        "PanelOLS fe": lambda: ws.PanelOLS.from_formula(
            "y ~ x1 + x2", frame, entity="firm", time="year"
        ).fit(),
        "RandomEffects": lambda: ws.RandomEffects.from_formula(
            "y ~ x1 + x2", frame, entity="firm", time="year"
        ).fit(),
        "BetweenOLS": lambda: ws.BetweenOLS.from_formula(
            "y ~ x1 + x2", frame, entity="firm", time="year"
        ).fit(),
    }


_LABELS = [
    "OLS",
    "OLS interaction",
    "FEOLS one-way",
    "FEOLS two-way",
    "IV2SLS",
    "Logit",
    "Probit",
    "Poisson",
    "NBreg",
    "OrderedLogit",
    "OrderedProbit",
    "PanelOLS fe",
    "RandomEffects",
    "BetweenOLS",
]


@pytest.mark.parametrize("label", _LABELS)
def test_stata_names_are_unique_and_non_empty(frame, label) -> None:
    result = _estimators(frame)[label]()

    names = list(result.stata_param_names)

    assert names, label
    assert len(set(names)) == len(names), (label, names)
    assert all(isinstance(name, str) and name for name in names), label


@pytest.mark.parametrize("label", _LABELS)
def test_stata_params_is_a_permutation_of_params(frame, label) -> None:
    """Reordering must never change a number, only its position."""

    result = _estimators(frame)[label]()

    reordered = np.sort(np.asarray(result.stata_params.to_numpy(), dtype=float))
    original = np.sort(np.asarray(result.params, dtype=float))

    assert np.allclose(reordered, original), label


@pytest.mark.parametrize("label", [
    "OLS",
    "FEOLS one-way",
    "IV2SLS",
    "PanelOLS fe",
    "NBreg",
])
def test_stata_cov_params_is_symmetric_with_a_permuted_diagonal(frame, label) -> None:
    result = _estimators(frame)[label]()

    covariance = np.asarray(result.stata_cov_params, dtype=float)
    base = np.asarray(result.cov_params(), dtype=float)

    assert covariance.shape == base.shape, label
    assert np.allclose(covariance, covariance.T), label
    assert np.allclose(np.sort(np.diag(covariance)), np.sort(np.diag(base))), label


def test_the_constant_precedes_auxiliary_parameters(frame) -> None:
    """``nbreg.json``: ``['x1', '_cons', 'lnalpha']``."""

    nbreg = ws.NegativeBinomial.from_formula("y_count ~ x1", frame).fit()
    names = nbreg.stata_param_names

    assert names.index("_cons") < names.index("lnalpha")

    ordered = ws.OrderedLogit.from_formula("rep ~ x1", frame).fit()
    ordered_names = ordered.stata_param_names
    # Ordinal models have no _cons, but the cuts follow the slopes.
    assert ordered_names[0] == "x1"
    assert all(name.startswith("cut") for name in ordered_names[1:])


def test_iv_reports_the_endogenous_regressor_first(frame) -> None:
    result = ws.IV2SLS.from_formula("y_iv ~ z1 + [x_endo ~ z2]", frame).fit()

    assert result.stata_param_names == ["x_endo", "z1", "_cons"]


def test_xtreg_fixtures_do_report_a_constant(frame) -> None:
    """The panel fixtures include ``_cons``, so ``PanelOLS`` must too.

    Checked against ``xtreg_fe.json``/``xtreg_re.json``/``xtreg_be.json``, whose
    names are all ``['x1', 'x2', '_cons']``.
    """

    for estimator in (ws.PanelOLS, ws.RandomEffects, ws.BetweenOLS):
        result = estimator.from_formula(
            "y ~ x1 + x2", frame, entity="firm", time="year"
        ).fit()
        assert result.stata_param_names == ["x1", "x2", "_cons"], estimator.__name__


def test_unmappable_terms_raise_rather_than_invent(frame) -> None:
    """The round-38/39 rule still holds inside the family."""

    explicit = ws.OLS.from_formula("y ~ x1 + C(firm)", frame).fit()

    with pytest.raises(NotImplementedError):
        _ = explicit.stata_param_names
