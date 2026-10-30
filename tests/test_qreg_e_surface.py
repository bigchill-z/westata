"""``qreg``'s stored results, read from the ado rather than from the help file.

`qreg.ado` sets its e() entries two ways, and they mean opposite things:

* ``ereturn local name "value"`` -- the macro exists with that value;
* ``ereturn local name`` with NO value -- the macro is CLEARED.

Reading only the second form as "documented" is how a key gets invented. In
`qreg.ado:175-194`, the `fvops` branch issues a block of bare
``ereturn local r2``/``mss``/``rss``/``rmse``/``r2_a``/``ll``/``ll_0`` statements
with no values -- they erase anything inherited, they do not report results.

Conditional entries are also real in this command (`qreg.ado:266-273`):

    ereturn local vce `vce'
    if "`vce'" != "iid" { ereturn local vcetype "`vcetype'" }
    if "`method'" == "kernel" { ereturn local kernel ... ; ereturn scalar kbwidth = `kband' }

so with the defaults the first of those is present and the other two are not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(107)
    n = 200
    x1 = rng.normal(size=n)
    y = 1.0 + 0.8 * x1 + (0.5 + 0.3 * np.abs(x1)) * rng.normal(size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": rng.normal(size=n)})


@pytest.fixture(scope="module")
def fitted(frame):
    return ws.QuantileRegression.from_formula("y ~ x1 + x2", frame).fit()


def test_the_scalar_keys_are_present(fitted) -> None:
    """``qreg.ado`` sets these with ``ereturn scalar`` (real values)."""

    for key in ("df_m", "df_r", "f_r", "bwidth", "convcode"):
        assert key in fitted.e, key


def test_the_pseudo_r2_and_deviations_are_present(fitted) -> None:
    """``qreg`` reports a PSEUDO R-squared and the deviation sums.

    Those are its own fit measures -- it has no ``mss``/``rss``/``rmse`` in the
    ordinary sense, because it minimises absolute rather than squared deviations.
    """

    for key in ("r2_p", "sum_rdev", "sum_adev", "q_v"):
        assert key in fitted.e, key


def test_cleared_macros_are_absent(fitted) -> None:
    """A bare ``ereturn local name`` CLEARS the macro; it does not set one.

    ``qreg.ado:175-194`` issues
    ``ereturn local r2``/``mss``/``rss``/``rmse``/``r2_a``/``ll``/``ll_0``/
    ``title``/``model`` with no values inside the ``fvops`` branch.  Treating those
    lines as documentation of stored results is exactly how an invented key gets
    added -- there is nothing to report.
    """

    for key in (
        "mss",
        "rss",
        "rmse",
        "ll",
        "ll_0",
        "r2_a",
        "title",
        "model",
        "estat_cmd",
    ):
        assert key not in fitted.e, key


def test_the_conditional_vcetype_is_absent_at_the_default_vce(fitted) -> None:
    """``qreg.ado:267``: ``vcetype`` is set only when ``vce`` is not ``iid``.

    The default is ``vce(iid)``, so the macro must not exist.  Reporting it anyway
    would claim a VCE title the default fit does not have.
    """

    assert fitted.e["vce"] == "iid"
    assert "vcetype" not in fitted.e


def test_the_conditional_kernel_keys_are_absent_at_the_default_method(
    fitted,
) -> None:
    """``qreg.ado:270-273``: ``kernel`` and ``kbwidth`` require ``method(kernel)``.

    ``bwidth`` IS present because it belongs to the default fitted method; the
    kernel ones are separate.
    """

    assert "bwidth" in fitted.e
    assert "kernel" not in fitted.e


def test_the_pseudo_r2_is_in_the_unit_interval(fitted) -> None:
    """A pseudo R-squared outside [0, 1] would signal a formula error."""

    value = float(fitted.e["r2_p"])

    assert 0.0 <= value <= 1.0


def test_r2_is_not_confused_with_the_pseudo_r2(fitted) -> None:
    """``qreg`` stores ``e(r2_p)``, not ``e(r2)``.

    ``e(r2)`` appears in the ado only as a bare ``ereturn local r2`` (a clear), so
    a plain ``r2`` key on a quantile result would be a key with no source.
    """

    assert "r2_p" in fitted.e
    assert "r2" not in fitted.e


def test_sum_adev_never_exceeds_sum_rdev(fitted) -> None:
    """``sum_adev`` is the MINIMAL sum of absolute deviations.

    ``qreg`` minimises absolute deviation, so its achieved value cannot exceed the
    raw sum about the fitted quantile -- if it did, the optimiser would not have
    minimised anything.
    """

    assert float(fitted.e["sum_adev"]) <= float(fitted.e["sum_rdev"]) + 1e-12


def test_sum_w_is_the_sum_of_weights(fitted) -> None:
    """``qreg.ado:279`` sets ``sum_w``; ``qreg.sthlp:507`` calls it "sum of weights".

    Unweighted, every observation contributes 1, so it equals the sample size.  The
    key was missing entirely, which reads as "qreg has no weights" rather than "the
    count was not reported".
    """

    assert "sum_w" in fitted.e
    assert float(fitted.e["sum_w"]) == float(fitted.e["N"])


def test_the_kernel_bandwidth_is_conditional(fitted) -> None:
    """``qreg.ado:270-272``: ``kbwidth`` comes with ``method(kernel)`` only.

    ``bwidth`` belongs to the default fitted method and IS present; ``kbwidth`` is a
    separate key for the kernel smoother, so its absence at the default is correct
    rather than a gap.
    """

    assert "bwidth" in fitted.e
    assert "kbwidth" not in fitted.e


def test_rreg_does_not_carry_the_qreg_keys(frame) -> None:
    """``sum_w``/``q``/``sum_rdev`` are ``qreg``'s, not ``rreg``'s.

    ``rreg.ado`` contains no ``sum_w`` at all.  Copying quantile keys onto the robust
    regression would invent results for a command that reports different ones.
    """

    result = ws.RobustRegression.from_formula("y ~ x1", frame).fit()

    assert "sum_w" not in result.e
    assert "q" not in result.e
    assert "sum_rdev" not in result.e