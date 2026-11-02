"""``intreg``'s interval statistics: ``pr``, ``e``, ``ystar``.

``intreg.ado:421-425``:

    ereturn local marginsok default XB Pr(passthru) E(passthru) YStar(passthru)

``intreg_postestimation.sthlp:133-145`` defines the three identically to ``tobit``:

    pr(a,b)      Pr(a < y < b)
    e(a,b)       E(xb + u | a < xb + u < b)
    ystar(a,b)   E(y*), y* = a below a, b above b, xb+u in between

Only ``xb``/``stdp``/``stdf``/``fitted`` were registered here, so the command's own macro
named three capabilities that did not exist.

**The trap this file guards.** ``intreg`` stores its scale as ``/lnsigma`` while ``tobit``
stores ``sigma``. Reading the last parameter as sigma gives ~0.0006 for a typical intreg
fit -- which produces perfectly plausible-looking probabilities near 0.5 and an
expectation equal to the linear index. Nothing raises; the numbers are simply wrong. The
scale is therefore resolved by parameter NAME (``_scale_sigma``), and the tests below
pin the ON SCALE property that a log-scale misread destroys.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

import westata as ws


def _frame(n: int = 400, seed: int = 5600):
    rng = np.random.default_rng(seed)
    covariate = rng.normal(size=n)
    latent = 1.0 + 0.8 * covariate + rng.normal(size=n)
    # An interval-censored outcome: the true value is known only to lie in (lo, hi).
    return pd.DataFrame(
        {"lo": latent - 0.3, "hi": latent + 0.3, "x": covariate}
    )


@pytest.fixture(scope="module")
def intreg_fit():
    return ws.Intreg.from_formula("lo hi ~ x", _frame()).fit()


# ------------------------------------------------------------- the registration
@pytest.mark.parametrize("kind", ["pr", "e", "ystar"])
def test_the_three_statistics_are_registered(kind: str) -> None:
    from westata.postestimation.predict import supported_types

    assert kind in supported_types("intreg")


def test_the_scale_parameter_is_stored_as_a_log(intreg_fit) -> None:
    """Pin the premise: ``intreg``'s last parameter is ``/lnsigma``, not sigma.

    If a future change renamed it, ``_scale_sigma`` would silently stop exponentiating
    and the probabilities would collapse toward 0.5.
    """

    assert str(intreg_fit.param_names[-1]).lstrip("/").lower() == "lnsigma"


def test_the_resolved_sigma_is_on_the_sigma_scale(intreg_fit) -> None:
    """``exp(lnsigma)`` must be a plausible standard deviation, not ~0.0006."""

    from westata.postestimation.predict import _scale_sigma

    raw_last = float(np.asarray(intreg_fit.params, dtype=float)[-1])
    resolved = _scale_sigma(intreg_fit)

    assert resolved > 0.01
    assert resolved == pytest.approx(float(np.exp(raw_last)), rel=1e-12)
    # And the raw value really is the log, i.e. the two differ.
    assert resolved != pytest.approx(abs(raw_last), rel=1e-6)


# ------------------------------------------------------------------ the values
def test_the_unbounded_probability_is_one(intreg_fit) -> None:
    probability = np.asarray(intreg_fit.predict(kind="pr"), dtype=float)

    assert np.allclose(probability, 1.0)


@pytest.mark.parametrize("lower,upper", [(-1.0, 2.0), (0.0, 1.0), (0.5, 2.5)])
def test_the_probability_lies_in_the_unit_interval(
    intreg_fit, lower: float, upper: float
) -> None:
    probability = np.asarray(
        intreg_fit.predict(kind="pr", lower=lower, upper=upper), dtype=float
    )

    assert np.all(probability >= 0.0)
    assert np.all(probability <= 1.0)
    assert np.all(np.isfinite(probability))


def test_the_probability_is_not_a_constant_near_one_half(intreg_fit) -> None:
    """THE log-scale check.

    Reading ``lnsigma`` as sigma makes ``(a - xb)/sigma`` enormous, so the normal
    probability saturates to 0 or 1 for every observation -- a constant-looking answer.
    A genuine sigma gives a spread of values.
    """

    probability = np.asarray(
        intreg_fit.predict(kind="pr", lower=0.0, upper=1.5), dtype=float
    )

    assert float(probability.std()) > 1e-3


def test_a_wider_interval_is_more_likely(intreg_fit) -> None:
    wide = np.asarray(intreg_fit.predict(kind="pr", lower=-2.0, upper=2.0), dtype=float)
    narrow = np.asarray(intreg_fit.predict(kind="pr", lower=-0.5, upper=0.5), dtype=float)

    assert np.all(wide >= narrow - 1e-12)


def test_the_conditional_expectation_moves_with_the_interval(intreg_fit) -> None:
    """``e(a,b)`` is a truncated-normal mean, so tightening the interval moves it."""

    low = np.asarray(intreg_fit.predict(kind="e", lower=-3.0, upper=-1.0), dtype=float)
    high = np.asarray(intreg_fit.predict(kind="e", lower=1.0, upper=3.0), dtype=float)

    assert float(low.mean()) < float(high.mean())


def test_ystar_brackets_the_linear_index(intreg_fit) -> None:
    """``y*`` censors the latent value at a and b, so it cannot run away."""

    lower, upper = 0.0, 2.0
    star = np.asarray(
        intreg_fit.predict(kind="ystar", lower=lower, upper=upper), dtype=float
    )

    assert np.all(star >= lower - 1e-9)
    assert np.all(star <= upper + 1e-9)


def test_the_three_statistics_are_distinct(intreg_fit) -> None:
    """Guard: implementing one in terms of another would be invisible per-test."""

    probability = np.asarray(
        intreg_fit.predict(kind="pr", lower=0.0, upper=2.0), dtype=float
    )
    expectation = np.asarray(
        intreg_fit.predict(kind="e", lower=0.0, upper=2.0), dtype=float
    )
    star = np.asarray(
        intreg_fit.predict(kind="ystar", lower=0.0, upper=2.0), dtype=float
    )

    assert not np.allclose(probability, expectation, rtol=1e-6)
    assert not np.allclose(expectation, star, rtol=1e-6)


# ----------------------------------------------------------- tobit still works
def test_tobit_keeps_its_own_sigma_scale() -> None:
    """``tobit`` stores sigma DIRECTLY; exponentiating it would be the mirror error."""

    from westata.postestimation.predict import _scale_sigma

    rng = np.random.default_rng(5602)
    n = 300
    covariate = rng.normal(size=n)
    frame = pd.DataFrame(
        {"y": np.clip(1.0 + 0.8 * covariate + rng.normal(size=n), 0.0, None), "x": covariate}
    )
    fitted = ws.Tobit.from_formula("y ~ x", frame, lower=0).fit()

    # tobit's scale is named "var(e.<depvar>)" in Stata's e(b), so the suffix test would
    # fail on the name alone. The assertion that matters is that the stored value IS the
    # scale, not its exponent -- checked numerically below and here by the spelling family.
    _scale_name = str(fitted.param_names[-1])
    assert _scale_name == "var(e.y)" or _scale_name.lstrip("/").lower() == "sigma"
    assert _scale_sigma(fitted) == pytest.approx(
        float(np.asarray(fitted.params, dtype=float)[-1]), rel=1e-12
    )