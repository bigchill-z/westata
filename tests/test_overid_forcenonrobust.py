"""``estat overid, forcenonrobust`` -- Sargan/Basmann under a robust VCE.

``ivregress_postestimation.sthlp:443-451``:

    {opt forcenonrobust} requests that the Sargan and Basmann tests of overidentifying
    restrictions be performed after 2SLS or LIML estimation even though a robust VCE was
    used at estimation time.  These tests assume that the errors are independent and
    identically distributed normal, so the user must determine whether the critical
    values are appropriate for a given application.

By default a robust VCE replaces them with Wooldridge's score test, and LIML has no
robust form at all.  ``forcenonrobust`` is the documented override for both.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(910)
    n = 300
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    endogenous = 0.8 * z1 + rng.normal(size=n) * 0.4
    exog = rng.normal(size=n)
    outcome = 1.0 + 0.8 * exog + 0.5 * endogenous + rng.normal(size=n)
    return pd.DataFrame(
        {"y": outcome, "x": exog, "e": endogenous, "z1": z1, "z2": z2}
    )


@pytest.fixture(scope="module")
def robust(frame):
    return ws.IV2SLS.from_formula("y ~ x + [e ~ z1 + z2]", frame).fit(cov_type="HC1")


@pytest.fixture(scope="module")
def classical(frame):
    return ws.IV2SLS.from_formula("y ~ x + [e ~ z1 + z2]", frame).fit()


def test_a_robust_vce_uses_the_wooldridge_score_by_default(robust) -> None:
    assert robust.estat_overid().test == "Wooldridge score"


def test_forcenonrobust_switches_to_sargan(robust) -> None:
    """The reported test is Sargan, and Basmann comes back with it."""

    forced = robust.estat_overid(forcenonrobust=True)

    assert forced.test == "Sargan"
    assert forced.basmann is not None
    assert np.isfinite(float(forced.basmann))


def test_the_forced_numbers_match_the_classical_fit(robust, classical) -> None:
    """Sargan/Basmann do not depend on the VCE -- only on the residuals.

    So forcing them on a robust fit must reproduce exactly the values a classical fit
    reports.  A version that computed something else under robustness would fail here.
    """

    forced = robust.estat_overid(forcenonrobust=True)
    plain = classical.estat_overid()

    assert float(forced.statistic) == pytest.approx(
        float(plain.statistic), rel=1e-10
    )
    assert float(forced.basmann) == pytest.approx(
        float(plain.basmann), rel=1e-10
    )


def test_the_wooldridge_statistic_is_a_different_number(robust) -> None:
    """Guard: the two routes must not accidentally coincide."""

    default = robust.estat_overid()
    forced = robust.estat_overid(forcenonrobust=True)

    assert float(default.statistic) != pytest.approx(float(forced.statistic), rel=1e-6)


def test_a_classical_fit_is_unaffected(classical) -> None:
    """Nothing to force: it already reports Sargan."""

    assert classical.estat_overid().test == "Sargan"
    assert classical.estat_overid(forcenonrobust=True).test == "Sargan"


def test_clustering_is_still_refused(robust) -> None:
    """``ado:290-296`` refuses the test after ``vce(cluster)`` outright.

    ``forcenonrobust`` is about the Sargan/Basmann assumption, not about clustered
    errors, so it must not open this path.
    """

    rng = np.random.default_rng(912)
    frame = pd.DataFrame(
        {
            "y": rng.normal(size=200),
            "x": rng.normal(size=200),
            "e": rng.normal(size=200),
            "z1": rng.normal(size=200),
            "z2": rng.normal(size=200),
        }
    )
    groups = pd.Series(np.arange(200) % 20, name="g")
    clustered = ws.IV2SLS.from_formula("y ~ x + [e ~ z1 + z2]", frame).fit(
        cov_type="cluster", cov_kwds={"clusters": groups}
    )

    with pytest.raises(ValueError, match="clustered"):
        clustered.estat_overid()


def test_liml_with_robust_vce_is_refused_by_default(frame) -> None:
    """``ado:354-357``: no robust overidentification test exists for LIML."""

    liml = ws.IVLIML.from_formula("y ~ x + [e ~ z1 + z2]", frame).fit(cov_type="HC1")

    with pytest.raises(ValueError, match="LIML"):
        liml.estat_overid()


def test_liml_accepts_forcenonrobust(frame) -> None:
    """The ado's message is literally "use forcenonrobust to override"."""

    liml = ws.IVLIML.from_formula("y ~ x + [e ~ z1 + z2]", frame).fit(cov_type="HC1")

    forced = liml.estat_overid(forcenonrobust=True)

    assert np.isfinite(float(forced.statistic))


def test_the_degrees_of_freedom_are_the_restriction_count(robust) -> None:
    """Two instruments for one endogenous regressor leaves one restriction."""

    forced = robust.estat_overid(forcenonrobust=True)

    assert float(forced.df) == 1.0