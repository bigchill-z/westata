"""``estat firststage, forcenonrobust`` -- the minimum-eigenvalue override.

``ivregress_postestimation.sthlp:416-422``:

    {opt forcenonrobust} requests that the minimum eigenvalue statistic and its critical
    values be reported even though a robust VCE was used at estimation time.  The
    reported critical values assume that the errors are independent and identically
    distributed normal, so the user must determine whether the critical values are
    appropriate for a given application.

Two things follow, and both were missing:

* the statistic and its critical values are SUPPRESSED under a robust VCE by default;
* ``forcenonrobust`` is the explicit override.

``reports_mineig`` exposes the decision, because ``mineig=None`` cannot distinguish
"not available for this model shape" from "not applicable under this VCE".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(900)
    n = 300
    instrument = rng.normal(size=n)
    endogenous = 1.2 * instrument + rng.normal(size=n) * 0.4
    exog = rng.normal(size=n)
    outcome = 1.0 + 0.8 * exog + 0.5 * endogenous + rng.normal(size=n)
    return pd.DataFrame(
        {"y": outcome, "x": exog, "e": endogenous, "z": instrument}
    )


@pytest.fixture(scope="module")
def classical(frame):
    return ws.IV2SLS.from_formula("y ~ x + [e ~ z]", frame).fit()


@pytest.fixture(scope="module")
def robust(frame):
    return ws.IV2SLS.from_formula("y ~ x + [e ~ z]", frame).fit(cov_type="HC1")


def test_the_classical_fit_reports_the_statistic(classical) -> None:
    assert classical.estat_firststage().reports_mineig is True


def test_the_robust_fit_suppresses_it_by_default(robust) -> None:
    """``sthlp:417-418``: it is only reported under a robust VCE on request."""

    assert robust.estat_firststage().reports_mineig is False


def test_forcenonrobust_restores_it(robust) -> None:
    assert robust.estat_firststage(forcenonrobust=True).reports_mineig is True


def test_the_option_is_harmless_on_a_classical_fit(classical) -> None:
    """There is nothing to force: the statistic is already reported."""

    assert classical.estat_firststage(forcenonrobust=True).reports_mineig is True


def test_the_vce_kind_is_detected(robust, classical) -> None:
    """The suppression keys off the covariance actually used, not a caller flag."""

    assert robust.estat_firststage().robust_vce is True
    assert classical.estat_firststage().robust_vce is False


def test_the_value_is_unchanged_by_the_option(robust) -> None:
    """``forcenonrobust`` changes whether the number is DISPLAYED, not its value.

    The Cragg-Donald statistic does not depend on the covariance estimator, so a version
    that recomputed it on the robust fit would be answering a different question.
    """

    suppressed = robust.estat_firststage()
    forced = robust.estat_firststage(forcenonrobust=True)

    assert suppressed.mineig == pytest.approx(forced.mineig, rel=1e-12)


def test_the_critical_values_are_unchanged_too(robust) -> None:
    suppressed = robust.estat_firststage()
    forced = robust.estat_firststage(forcenonrobust=True)

    assert np.array_equal(
        suppressed.mineigcv, forced.mineigcv, equal_nan=True
    )


def test_all_and_forcenonrobust_compose(frame) -> None:
    """Two independent switches must not cancel each other out."""

    rng = np.random.default_rng(902)
    n = 300
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    e1 = 0.7 * z1 + rng.normal(size=n) * 0.3
    e2 = 0.7 * z2 + rng.normal(size=n) * 0.3
    exog = rng.normal(size=n)
    outcome = 1.0 + 0.8 * exog + 0.5 * e1 + 0.4 * e2 + rng.normal(size=n)
    multi = pd.DataFrame(
        {"y": outcome, "x": exog, "e1": e1, "e2": e2, "z1": z1, "z2": z2}
    )
    fitted = ws.IV2SLS.from_formula("y ~ x + [e1 + e2 ~ z1 + z2]", multi).fit(
        cov_type="HC1"
    )

    rendered = fitted.estat_firststage(all=True, forcenonrobust=True)

    assert rendered.show_all is True
    assert rendered.reports_mineig is True
    assert "R-sq." in str(rendered)


def test_clustered_fits_count_as_robust(frame) -> None:
    """A clustered VCE is also robust, so the same suppression applies."""

    rng = np.random.default_rng(904)
    n = len(frame)
    groups = pd.Series(rng.integers(0, 40, size=n), name="g")
    clustered = ws.IV2SLS.from_formula("y ~ x + [e ~ z]", frame).fit(
        cov_type="cluster", cov_kwds={"clusters": groups}
    )

    assert clustered.estat_firststage().reports_mineig is False
    assert (
        clustered.estat_firststage(forcenonrobust=True).reports_mineig is True
    )