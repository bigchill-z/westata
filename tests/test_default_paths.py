"""Every estimator must run on its DEFAULT path.

Round 70 found `Sqreg(...).fit()` raising `NameError: DEFAULT_SEED is not
defined`. The whole suite missed it because every existing test passed
``seed=...``: **a test that always supplies an optional argument never exercises
the default.**

That is the path a new user hits first, so it deserves its own guard. These tests
make the smallest legal call for each estimator and only require that it runs --
they assert nothing about the numbers, which other files already cover.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(3)
    n = 150
    groups, periods = 30, 5
    firm = np.repeat(np.arange(groups), periods)
    year = np.tile(np.arange(periods), groups)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    effect = rng.normal(size=groups)[firm]
    y = 1.0 + 0.6 * x1 - 0.4 * x2 + effect + rng.normal(size=n)
    return pd.DataFrame(
        {
            "y": y,
            "x1": x1,
            "x2": x2,
            "firm": firm,
            "year": year,
            "yb": (y > y.mean()).astype(int),
            "yc": rng.poisson(3.0, size=n),
            "rep": rng.integers(1, 4, size=n).astype(float),
            "z1": rng.normal(size=n),
            "t": np.arange(n),
        }
    )


#: label -> callable taking the shared frame.  Each call supplies only what the
#: estimator cannot infer (the entity/time columns for a panel, the lag for
#: newey -- both REQUIRED by Stata too), and nothing else.
CASES = {
    "Regress": lambda f: ws.Regress.from_formula("y ~ x1", f).fit(),
    "IV2SLS": lambda f: ws.IV2SLS.from_formula("y ~ x1 + [x2 ~ z1]", f).fit(),
    "IVLIML": lambda f: ws.IVLIML.from_formula("y ~ x1 + [x2 ~ z1]", f).fit(),
    "IVGMM": lambda f: ws.IVGMM.from_formula("y ~ x1 + [x2 ~ z1]", f).fit(),
    "RegressHighDimFixedEffects": lambda f: ws.RegressHighDimFixedEffects.from_formula("y ~ x1 | firm", f).fit(),
    "Logit": lambda f: ws.Logit.from_formula("yb ~ x1", f).fit(),
    "Probit": lambda f: ws.Probit.from_formula("yb ~ x1", f).fit(),
    "Poisson": lambda f: ws.Poisson.from_formula("yc ~ x1", f).fit(),
    "NBreg": lambda f: ws.NegativeBinomial.from_formula("yc ~ x1", f).fit(),
    "OrderedLogit": lambda f: ws.OrderedLogit.from_formula("rep ~ x1", f).fit(),
    "OrderedProbit": lambda f: ws.OrderedProbit.from_formula("rep ~ x1", f).fit(),
    "PanelRegressFE": lambda f: ws.PanelRegressFE.from_formula(
        "y ~ x1", f, entity="firm", time="year"
    ).fit(),
    "PanelRegressRE": lambda f: ws.PanelRegressRE.from_formula(
        "y ~ x1", f, entity="firm", time="year"
    ).fit(),
    "PanelRegressBE": lambda f: ws.PanelRegressBE.from_formula(
        "y ~ x1", f, entity="firm", time="year"
    ).fit(),
    "PanelRegressFD": lambda f: ws.PanelRegressFD.from_formula(
        "y ~ x1", f, entity="firm", time="year"
    ).fit(),
    "QuantileRegress": lambda f: ws.QuantileRegress.from_formula(
        "y ~ x1", f
    ).fit(),
    "RobustRegress": lambda f: ws.RobustRegress.from_formula("y ~ x1", f).fit(),
    "MultivariateRegress": lambda f: ws.MultivariateRegress.from_formula("y x1 = x2", f).fit(),
    "Manova": lambda f: ws.Manova.from_formula("y x1 = firm", f).fit(),
    "SeeminglyUnrelatedRegress": lambda f: ws.SeeminglyUnrelatedRegress.from_formula(["y ~ x1", "x1 ~ x2"], f).fit(),
    "newey_west": lambda f: ws.newey_west("y ~ x1", f, lag=1, time=f["t"])._results,
}


@pytest.mark.parametrize("label", sorted(CASES))
def test_the_smallest_legal_call_runs(frame, label) -> None:
    """No crash on the default path; the numbers are other files' business."""

    result = CASES[label](frame)

    assert result is not None


@pytest.mark.parametrize("label", ["SimultaneousQuantileRegress", "BootstrappedQuantileRegress"])
def test_bootstrap_estimators_run_without_a_seed(frame, label) -> None:
    """The round-70 failure mode, pinned per estimator.

    ``Sqreg``/``Bsqreg`` share the bootstrap machinery whose default branch read an
    undefined ``DEFAULT_SEED``.  Calling them with no ``seed`` is the regression
    test.
    """

    if label == "SimultaneousQuantileRegress":
        result = ws.SimultaneousQuantileRegress.from_formula(
            "y ~ x1", frame, quantiles=(0.25, 0.75)
        ).fit()
    else:
        result = ws.BootstrappedQuantileRegress.from_formula("y ~ x1", frame).fit()

    assert np.all(np.isfinite(np.asarray(result.cov_params())))


def test_an_unseeded_bootstrap_is_not_silently_deterministic(frame) -> None:
    """The fix must not have replaced "crash" with "always the same draw".

    Stata's unseeded behaviour follows ``c(seed)``; pinning a constant would make
    every run identical, which is a different wrong answer.
    """

    first = ws.BootstrappedQuantileRegress.from_formula("y ~ x1", frame).fit()
    second = ws.BootstrappedQuantileRegress.from_formula("y ~ x1", frame).fit()

    assert not np.allclose(
        np.asarray(first.cov_params()), np.asarray(second.cov_params())
    )


def test_eivreg_refuses_without_reliability_ratios(frame) -> None:
    """A documented REFUSAL, not a default-path crash.

    ``eivreg`` without ``reliab`` degenerates to plain OLS while still claiming to
    be errors-in-variables, so the estimator rejects it.  Listing it here keeps the
    distinction visible: "does not run by default" and "must not run by default"
    are different outcomes.
    """

    with pytest.raises(ws.ConsistencyError, match="reliab"):
        ws.ErrorsInVariablesRegress.from_formula("y ~ x1", frame).fit()
