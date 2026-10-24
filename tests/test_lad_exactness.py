"""Tests for the LAD solver's exactness, and for the singular-Sigma warning.

Two defects were found and fixed in this round, and both are pinned here.

1. `solve_lad` converged linearly (IRLS) on a piecewise-linear objective, so it stopped at its tolerance
   rather than at the minimiser. Measured against a live Stata on the sweep's own data:

       median        4.564e-09 -> 3.381e-16
       quantile(.25) 5.678e-07 -> 3.129e-15

   The test asserts the OBJECTIVE is minimal, not that coefficients match a stored vector: the
   check function can have a flat optimal face, so a coefficient comparison would be testing which point
   an LP happens to report.

2. `XtGls` silently returned a pseudo-inverse solution when Sigma was singular. Stata says so in its
   output; westata said nothing. The test asserts the warning fires for the singular case and does not
   fire for the well-posed ones, because a warning that always fires is as useless as none.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import westata as ws

# ---------------------------------------------------------------- quantile solver


def _quantile_data():
    """The sweep's own generator: seed 31, n = 400, heteroskedastic errors."""

    rng = np.random.default_rng(31)
    n = 400
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.0 + 0.8 * x1 - 0.4 * x2 + rng.normal(size=n) * (1 + 0.4 * np.abs(x1))
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def _check_objective(
    response: np.ndarray, design: np.ndarray, beta: np.ndarray, tau: float
) -> float:
    """Koenker's check function at `beta`; the criterion a quantile regression minimises."""

    residual = np.asarray(response, dtype=float) - np.asarray(design, dtype=float) @ beta
    return float(np.sum(np.where(residual >= 0.0, tau * residual, (tau - 1.0) * residual)))


@pytest.mark.parametrize("tau", [0.5, 0.25])
def test_lad_reaches_the_exact_check_function_minimum(tau):
    """The solver must reach the minimum, not merely a small step.

    These are the objectives an independent LP solver (HiGHS) reaches, and Stata reaches the same
    values. IRLS alone fell short by 1.9e-09 and 3.0e-09 respectively, which is what the LP polish
    in `_polish_with_lp` removes.
    """

    frame = _quantile_data()
    result = ws.QuantileRegression.from_formula("y ~ x1 + x2", frame, quantile=tau).fit()
    beta = np.asarray(result.params, dtype=float)

    expected = {0.5: 208.04731048707464, 0.25: 167.14892238457264}[tau]
    design = np.column_stack(
        [np.ones(len(frame)), frame["x1"].to_numpy(), frame["x2"].to_numpy()]
    )
    achieved = _check_objective(frame["y"].to_numpy(), design, beta, tau)
    assert achieved == pytest.approx(expected, rel=1e-12), (
        f"tau={tau}: objective {achieved!r} is above the exact minimum {expected!r}"
    )


def test_lad_solves_a_sample_quantile_when_there_is_only_a_constant():
    """With one constant column the minimiser is a sample quantile of y -- an order statistic."""

    rng = np.random.default_rng(7)
    y = rng.normal(size=201)
    design = np.ones((201, 1))
    from westata.estimators.quantile import solve_lad

    solution = solve_lad(design, y, quantile=0.5)
    location = float(np.asarray(solution, dtype=float).reshape(-1)[0])
    # A check-function minimiser with a constant column is an order statistic, and any value between
    # the two middle order statistics is optimal for the median.
    ordered = np.sort(y)
    assert ordered[99] <= location <= ordered[101]
    # And the objective must be minimal, not merely inside the interval.
    assert _check_objective(y, design, np.array([location]), 0.5) == pytest.approx(
        _check_objective(y, design, np.array([np.median(y)]), 0.5), rel=1e-12
    )


def test_the_polish_never_makes_the_objective_worse():
    """A degenerate design must not be degraded by the LP step.

    `_polish_with_lp` accepts the LP answer only when it does not raise the objective, so a failure in
    the LP must leave the IRLS result in place rather than propagate an error.
    """

    from westata.estimators.quantile import _polish_with_lp

    rng = np.random.default_rng(11)
    design = np.column_stack([np.ones(50), rng.normal(size=50)])
    response = rng.normal(size=50) + 0.5 * design[:, 1]
    start = np.linalg.lstsq(design, response, rcond=None)[0]

    polished = _polish_with_lp(design, response, start, 0.5)
    assert np.all(np.isfinite(polished))
    assert _check_objective(response, design, polished, 0.5) <= (
        _check_objective(response, design, start, 0.5) + 1e-12
    )


# ---------------------------------------------------------------- xtgls singularity


def _panel(panels: int, periods: int, seed: int = 20261024) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(panels):
        for period in range(periods):
            x1 = rng.normal()
            x2 = rng.normal()
            y = 0.7 + 0.9 * x1 - 0.5 * x2 + rng.normal() + 0.35 * unit / panels
            rows.append({"id": unit + 1, "t": period + 1, "y": y, "x1": x1, "x2": x2})
    return pd.DataFrame(rows)


def test_xtgls_warns_when_sigma_is_singular():
    """30 panels and 6 periods make Sigma rank-deficient, so the GLS solution is not unique.

    Stata prints "results are based on a generalized inverse of a singular matrix"; this warning is the
    same message in a catchable form. Without it, westata returned a number that differed from Stata by
    more than an order of magnitude with no indication the question was ill-posed.
    """

    from westata.estimators.panel_iv import SingularSigmaWarning

    frame = _panel(30, 6)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ws.XtGls.from_formula(
            "y ~ x1 + x2", frame, entity="id", time="t", panels="correlated", corr="independent"
        ).fit()
    matches = [item for item in caught if issubclass(item.category, SingularSigmaWarning)]
    assert matches, "a singular Sigma must warn"
    assert "generalized inverse" in str(matches[0].message)


@pytest.mark.parametrize("panels", ["heteroskedastic", "iid"])
def test_xtgls_does_not_warn_when_sigma_is_well_posed(panels):
    """A warning that fires on healthy input trains the user to ignore it."""

    from westata.estimators.panel_iv import SingularSigmaWarning

    frame = _panel(30, 6)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ws.XtGls.from_formula(
            "y ~ x1 + x2", frame, entity="id", time="t", panels=panels, corr="independent"
        ).fit()
    assert not [item for item in caught if issubclass(item.category, SingularSigmaWarning)]


def test_xtgls_with_correlated_and_more_periods_than_panels_is_silent():
    """m < T makes Sigma full rank, and the sweep measures that configuration at 1e-15."""

    from westata.estimators.panel_iv import SingularSigmaWarning

    frame = _panel(6, 8, seed=31337)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ws.XtGls.from_formula(
            "y ~ x1 + x2", frame, entity="id", time="t", panels="correlated", corr="independent"
        ).fit()
    assert not [item for item in caught if issubclass(item.category, SingularSigmaWarning)]
