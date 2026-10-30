"""``tobit`` stores no ``e(aic)`` -- and that matches Stata.

``tobit.sthlp``'s stored-results block contains no ``aic`` at all (a grep gives zero), so
the macro a ``regress`` user might reach for does not exist there. What DOES work is
``estat ic``, which computes the criterion on demand from ``e(ll)`` and the parameter
count -- that is a separate command, not a stored macro.

So the surface is:

=================  =============  =================
estimator          ``e(aic)``     ``estat_ic()``
=================  =============  =================
``regress``        yes (not Stata's)  yes
``logit`` etc.     yes (Stata's)      yes
``tobit``          **no**             **yes**
=================  =============  =================

The middle column being absent is not a gap to be filled: adding the key would contradict
the help file. The third column is how a caller gets the number.

This is the same shape as ``e(predict)`` for ``regress`` (documented but unfindable) and
``e(title)`` for ``poisson`` (present but set indirectly) -- the three show that "which
macros exist" is a per-command question with no safe default.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import westata as ws

BASE = Path(r"D:\Program Files\StataNow19\ado\base")


def _frame(n: int = 300, seed: int = 8402):
    rng = np.random.default_rng(seed)
    covariate = rng.normal(size=n)
    return pd.DataFrame(
        {"yc": np.clip(1.0 + 0.8 * covariate + rng.normal(size=n), 0.0, None), "x": covariate}
    )


@pytest.fixture(scope="module")
def tobit_fit():
    return ws.Tobit.from_formula("yc ~ x", _frame(), lower=0).fit()


# --------------------------------------------------------------- the absence
def test_tobit_stores_no_aic(tobit_fit) -> None:
    """``tobit.sthlp`` has no ``e(aic)``, so the key stays absent."""

    assert "aic" not in tobit_fit.e
    assert "bic" not in tobit_fit.e


def test_the_help_file_really_has_none() -> None:
    """Read the authority rather than trusting the transcription above."""

    text = (BASE / "t/tobit.sthlp").read_text(encoding="utf-8", errors="ignore")

    assert "e(aic)" not in text


# ------------------------------------------------------- estat_ic still works
def test_estat_ic_is_available(tobit_fit) -> None:
    """The criterion is computed on demand, which is what Stata's ``estat ic`` does."""

    rendered = tobit_fit.estat_ic()

    assert np.isfinite(float(rendered.aic))
    assert np.isfinite(float(rendered.bic))


def test_estat_ic_uses_ll_and_the_parameter_count(tobit_fit) -> None:
    """``estimates.ado:985-1002``: ``AIC = -2 e(ll) + 2 e(df)``."""

    rendered = tobit_fit.estat_ic()
    e = tobit_fit.e

    assert float(rendered.aic) == pytest.approx(
        -2.0 * float(e["ll"]) + 2.0 * float(rendered.df), rel=1e-10
    )


# --------------------------------------------------- the contrast with regress
def test_regress_does_store_the_keys() -> None:
    """Guard: the absence is specific to tobit, not a package-wide omission."""

    rng = np.random.default_rng(8404)
    frame = pd.DataFrame({"y": rng.normal(size=250), "x": rng.normal(size=250)})
    fitted = ws.OLS.from_formula("y ~ x", frame).fit()

    assert "aic" in fitted.e
    assert "bic" in fitted.e


def test_logit_stores_the_keys() -> None:
    rng = np.random.default_rng(8406)
    frame = pd.DataFrame(
        {"yb": (rng.uniform(size=250) < 0.5).astype(int), "x": rng.normal(size=250)}
    )
    fitted = ws.Logit.from_formula("yb ~ x", frame).fit()

    assert "aic" in fitted.e


# ------------------------------------------------------- not collateral damage
def test_tobit_still_has_the_core_macros(tobit_fit) -> None:
    e = tobit_fit.e

    for key in ("N", "ll", "rank", "df_m"):
        assert key in e, key


def test_the_summary_still_renders(tobit_fit) -> None:
    """The absence of ``aic`` must not break the table."""

    rendered = str(tobit_fit.summary(style="stata"))

    assert rendered
    assert "Coefficient" in rendered