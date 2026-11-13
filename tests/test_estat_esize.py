"""``estat esize`` -- eta/epsilon/omega-squared effect sizes for a linear model.

``regress_postestimation.sthlp:599-614``:

    By default, estat esize reports eta-squared estimates (Kerlinger 1964), which are
    equivalent to R-squared estimates.  If the option epsilon is specified, estat esize
    reports epsilon-squared estimates ... If the option omega is specified, estat esize
    reports omega-squared estimates ... Both epsilon-squared and omega-squared are
    adjusted R-squared estimates.  Confidence intervals for eta-squared estimates are
    estimated by using the noncentral F distribution (Smithson 2001).

``esize.ado`` hands the arithmetic to compiled builtins, so the formulas come from the
sources the help file cites:

    eta2     = SS_effect / SS_total                      ( == R^2 )
    epsilon2 = (SS_effect - df_effect * MS_error) / SS_total
    omega2   = (SS_effect - df_effect * MS_error) / (SS_total + MS_error)

The two adjusted forms differ only in the denominator, which is what fixes their order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

import westata as ws


def _fit(*, n: int = 150, seed: int = 3000, signal: float = 0.8):
    rng = np.random.default_rng(seed)
    covariate = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "y": 1.0 + signal * covariate + rng.normal(size=n),
            "x1": covariate,
            "x2": rng.normal(size=n),
        }
    )
    return frame, ws.OLS.from_formula("y ~ x1 + x2", frame).fit()


@pytest.fixture(scope="module")
def fitted():
    return _fit()


# ------------------------------------------------------------------- eta2
def test_eta_squared_equals_r_squared(fitted) -> None:
    """``sthlp:602-603``: "eta-squared estimates ... equivalent to R-squared"."""

    reported = fitted[1].estat_esize()

    assert reported.estimate == pytest.approx(
        float(fitted[1].e["r2"]), rel=1e-12
    )


def test_eta_squared_is_mss_over_total(fitted) -> None:
    """``regress`` stores mss and rss but NOT tss, so the total is their sum."""

    e = fitted[1].e
    expected = float(e["mss"]) / (float(e["mss"]) + float(e["rss"]))

    assert fitted[1].estat_esize().estimate == pytest.approx(expected, rel=1e-12)


def test_the_interval_brackets_the_estimate(fitted) -> None:
    reported = fitted[1].estat_esize()

    assert reported.ci_lower is not None
    assert reported.ci_upper is not None
    assert reported.ci_lower < reported.estimate < reported.ci_upper


def test_the_interval_is_within_the_unit_range(fitted) -> None:
    """An effect size is a share of variance and cannot escape [0, 1] in this form."""

    reported = fitted[1].estat_esize()

    assert reported.ci_lower >= 0.0
    assert reported.ci_upper <= 1.0


def test_a_wider_level_gives_a_wider_interval(fitted) -> None:
    """`level` is a FRACTION here, like every other `level` in the package (R3.3).

    This used to pass ``level=90`` / ``level=99``, which read as percentages because
    ``estat_esize`` was the one command that took them that way.  It now takes
    ``0.90`` / ``0.99``; passing 90 would ask for a 9000% interval.
    """

    narrow = fitted[1].estat_esize(level=0.90)
    wide = fitted[1].estat_esize(level=0.99)

    assert np.isfinite(narrow.ci_upper) and np.isfinite(wide.ci_upper)
    assert wide.ci_lower <= narrow.ci_lower
    assert wide.ci_upper > narrow.ci_upper


def test_the_lower_bound_is_zero_not_nan_when_f_is_small(fitted) -> None:
    """A small F puts the whole noncentrality domain outside the bracket.

    ``lower_gap`` is then positive everywhere, so there is no root to solve for and
    the mathematically correct lower limit is noncentrality 0 (effect size 0), NOT
    NaN.  This was a real defect: every ordinary fit reported ``ci_lower=nan``.
    """

    reported = fitted[1].estat_esize()
    assert np.isfinite(reported.ci_lower), "ci_lower must be a number, not nan"
    assert reported.ci_lower >= 0.0


def test_the_interval_narrows_with_more_data() -> None:
    """Guard: an interval that ignored the sample size would pass the level test.

    Comparing two independent samples of the same design is what makes the width a
    function of ``N`` rather than of the requested level alone.
    """

    small = _fit(n=60, seed=3002)[1].estat_esize()
    large = _fit(n=600, seed=3002)[1].estat_esize()

    assert (large.ci_upper - large.ci_lower) < (small.ci_upper - small.ci_lower)


# --------------------------------------------------------- epsilon and omega
def test_epsilon_is_the_adjusted_form(fitted) -> None:
    """``(SS_effect - df_effect*MS_error) / SS_total``."""

    e = fitted[1].e
    mss = float(e["mss"])
    rss = float(e["rss"])
    df_effect = float(e["df_m"])
    df_residual = float(e["df_r"])
    mean_square_error = rss / df_residual
    expected = (mss - df_effect * mean_square_error) / (mss + rss)

    assert fitted[1].estat_esize(epsilon=True).estimate == pytest.approx(
        expected, rel=1e-12
    )


def test_omega_adds_one_mean_square_to_the_denominator(fitted) -> None:
    """``(SS_effect - df_effect*MS_error) / (SS_total + MS_error)``.

    The larger denominator is what makes omega the smaller of the two adjusted forms.
    """

    e = fitted[1].e
    mss = float(e["mss"])
    rss = float(e["rss"])
    df_effect = float(e["df_m"])
    df_residual = float(e["df_r"])
    mean_square_error = rss / df_residual
    expected = (mss - df_effect * mean_square_error) / (
        (mss + rss) + mean_square_error
    )

    assert fitted[1].estat_esize(omega=True).estimate == pytest.approx(
        expected, rel=1e-12
    )


def test_eta_exceeds_epsilon_which_exceeds_omega(fitted) -> None:
    """The adjustment can only shrink the share.

    This ordering is a property of the formulas for any fit with a positive effect, so it
    is asserted rather than eyeballed.
    """

    eta = fitted[1].estat_esize().estimate
    epsilon = fitted[1].estat_esize(epsilon=True).estimate
    omega = fitted[1].estat_esize(omega=True).estimate

    assert eta > epsilon > omega


def test_the_adjusted_forms_carry_no_interval(fitted) -> None:
    """``sthlp:610-612`` documents intervals for ETA-squared only."""

    for kwargs in ({"epsilon": True}, {"omega": True}):
        reported = fitted[1].estat_esize(**kwargs)
        assert reported.ci_lower is None
        assert reported.ci_upper is None


# ---------------------------------------------------------------- refusals
def test_epsilon_and_omega_together_are_refused(fitted) -> None:
    """``esize.ado:14``: ``opts_exclusive "`epsilon' `omega'"``."""

    with pytest.raises(ValueError, match="mutually exclusive"):
        fitted[1].estat_esize(epsilon=True, omega=True)


def test_a_non_linear_model_is_refused() -> None:
    """``sthlp:599-601``: "after regress or anova"."""

    rng = np.random.default_rng(3004)
    frame = pd.DataFrame(
        {"yb": (rng.normal(size=120) > 0).astype(int), "x1": rng.normal(size=120)}
    )
    logit = ws.Logit.from_formula("yb ~ x1", frame).fit()

    with pytest.raises(NotImplementedError, match="regress or anova"):
        logit.estat_esize()


# ---------------------------------------------------------------- the r() names
def test_to_dict_reports_only_the_computed_kind(fitted) -> None:
    """``esize.sthlp:284-285`` names ``r(eta2)``/``r(epsilon2)``/``r(omega2)``.

    Only the requested one is returned, so a caller checking for a key learns which
    estimate the command produced.
    """

    assert set(fitted[1].estat_esize().to_dict()) == {"eta2"}
    assert set(fitted[1].estat_esize(epsilon=True).to_dict()) == {"epsilon2"}
    assert set(fitted[1].estat_esize(omega=True).to_dict()) == {"omega2"}


def test_the_rendered_text_names_the_estimate(fitted) -> None:
    assert "Eta-squared" in str(fitted[1].estat_esize())
    assert "Epsilon-squared" in str(fitted[1].estat_esize(epsilon=True))
    assert "Omega-squared" in str(fitted[1].estat_esize(omega=True))