"""``e(aic)`` is this package's convenience value, NOT a Stata macro.

``regress.sthlp`` stores no ``e(aic)``/``e(bic)`` -- a grep for "aic" over the whole
stored-results block returns nothing. Only a handful of commands in ``ado/base`` document
``e(aic)`` at all (``binreg``, ``glm``, ``factor``), and the estimator families this
package implements are not among them.

Those two keys are still EXposed, because removing them would break callers and the
tests that pin them, but they are on the Gaussian-deviance scale::

    aic = N log(rss/N) + 2k
    bic = N log(rss/N) + k log N

which omits the ``N (1 + log 2pi)`` constant that ``e(ll)`` carries. That omission is
exactly why ``e(aic)`` and ``estat_ic``'s AIC differ by a constant -- the difference is
NOT a bug in either, but a reader who assumed they were the same number would be badly
misled.

``estat_ic`` is the Stata-conformant surface: it follows ``estimates.ado:985-1002``,
``AIC = -2 e(ll) + 2 e(df)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


def _frame(n: int = 300, seed: int = 4900):
    rng = np.random.default_rng(seed)
    covariate = rng.normal(size=n)
    return pd.DataFrame(
        {"y": 1.0 + 0.8 * covariate + rng.normal(size=n), "x": covariate}
    )


@pytest.fixture(scope="module")
def fitted():
    return ws.OLS.from_formula("y ~ x", _frame()).fit()


# ----------------------------------------------------------- the two quantities
def test_estat_ic_follows_the_documented_formula(fitted) -> None:
    """``estimates.ado:985-1002``: ``AIC = -2 e(ll) + 2 e(df)``."""

    rendered = fitted.estat_ic()
    e = fitted.e

    assert rendered.aic == pytest.approx(
        -2.0 * float(e["ll"]) + 2.0 * float(rendered.df), rel=1e-12
    )
    assert rendered.bic == pytest.approx(
        -2.0 * float(e["ll"]) + float(rendered.df) * np.log(float(e["N"])),
        rel=1e-12,
    )


def test_the_estimate_side_aic_is_the_gaussian_deviance_scale(fitted) -> None:
    """``N log(rss/N) + 2k`` -- deliberately WITHOUT ``N(1 + log 2pi)``."""

    e = fitted.e
    n = float(e["N"])
    k = float(e["rank"])
    rss = float(e["rss"])

    assert float(e["aic"]) == pytest.approx(n * np.log(rss / n) + 2.0 * k, rel=1e-12)


def test_the_two_aics_differ_by_the_missing_constant(fitted) -> None:
    """The gap is ``N (1 + log 2pi)`` -- a CONSTANT, not noise.

    Asserting the exact gap is what proves the difference is the omitted constant rather
    than an arithmetic mistake on one side.
    """

    e = fitted.e
    n = float(e["N"])
    table = fitted.estat_ic()

    gap = float(table.aic) - float(e["aic"])

    assert gap == pytest.approx(n * (1.0 + np.log(2.0 * np.pi)), rel=1e-10)


def test_they_are_not_the_same_number(fitted) -> None:
    """Guard: if the two coincided, the gap test above would prove nothing."""

    assert float(fitted.e["aic"]) != pytest.approx(fitted.estat_ic().aic, rel=1e-6)


# ------------------------------------------------ against the Stata help files
def test_regress_help_documents_no_aic() -> None:
    """Pin the evidence: ``regress.sthlp`` has no aic in its stored-results block.

    This is the reason the key cannot be presented as Stata's: it does not exist there.
    The check reads the file so a future Stata upgrade that adds it would be noticed.
    """

    from pathlib import Path

    help_file = Path(r"D:\Program Files\StataNow19\ado\base\r\regress.sthlp")
    text = help_file.read_text(encoding="utf-8", errors="ignore").lower()

    assert "aic" not in text
    assert "bic" not in text


def test_logit_help_documents_no_aic() -> None:
    from pathlib import Path

    help_file = Path(r"D:\Program Files\StataNow19\ado\base\l\logit.sthlp")
    text = help_file.read_text(encoding="utf-8", errors="ignore").lower()

    assert "e(aic)" not in text


# ------------------------------------------------- what a caller should use
def test_estat_ic_is_the_stata_conformant_surface(fitted) -> None:
    """A reader who wants Stata's information criteria calls ``estat ic``.

    Both surfaces are internally consistent; only one matches Stata. This test records
    which.
    """

    rendered = fitted.estat_ic()
    e = fitted.e

    # estat_ic traces back to e(ll) and e(N); e(aic) does not.
    assert rendered.n == float(e["N"])
    assert rendered.ll_model == pytest.approx(float(e["ll"]), rel=1e-12)


def test_e_ll_and_e_rank_are_the_stata_macros(fitted) -> None:
    """These ARE documented for ``regress`` and are what a Stata-tied number needs."""

    e = fitted.e

    assert "ll" in e
    assert "rank" in e
    assert np.isfinite(float(e["ll"]))

    # And their combination reproduces estat ic exactly.
    recomputed = -2.0 * float(e["ll"]) + 2.0 * float(e["rank"])
    assert recomputed == pytest.approx(fitted.estat_ic().aic, rel=1e-12)