"""The summary header must mirror the command's OWN table, not one template for all.

Stata does not print one header for every estimator. Two shapes exist:

**OLS-style** (``regress``, ``areg``)::

    No. Observations  R-squared
    Df Residuals      Adj. R-squared
    Df Model          F-statistic
                      Log-Likelihood

**ML-style** (``logit``, ``probit``, ``poisson``, ``tobit``)::

    No. Observations  Pseudo R-squared
                      LR chi2
                      Log-Likelihood

This package printed the OLS block for everything, so an ML fit showed ``Df Residuals:
250`` -- a number its own table omits -- next to ``R-squared: .``, ``Adj. R-squared: .``
and ``F-statistic: .``. A reader saw a broken regression table rather than a discrete-choice
one.

**``tobit`` is the case that rules out keying on the distribution.** Its tests use a t
distribution (``dist == "t"``), but its table reports ``LR chi2`` and ``Pseudo R2`` with no
F statistic -- so the block is chosen by WHAT THE COMMAND REPORTS, not by ``dist``. Keying
on ``dist`` gave tobit the OLS block and reproduced the ``.``-filled table exactly.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

import westata as ws

OLS_LABELS = ("R-squared:", "Adj. R-squared:", "F-statistic:", "Df Residuals:")
ML_LABELS = ("Pseudo R-squared:", "LR chi2:")

#: Labels that are SUBSTRINGS of the ML ones and so need a line-start check.
#: ``"R-squared:"`` occurs inside ``"Pseudo R-squared:"``; asserting its absence with
#: ``in`` therefore fails on a correct ML table.
PREFIX_SENSITIVE = ("R-squared:",)


def _has_standalone_label(rendered: str, label: str) -> bool:
    """True when ``label`` appears NOT preceded by ``"Pseudo "``.

    The two-column layout puts the right-hand labels mid-line, so a line-start check does
    not apply.  What distinguishes the two blocks is only the prefix -- ``R-squared:``
    versus ``Pseudo R-squared:`` -- so a negative lookbehind for ``Pseudo `` is precise
    for both.
    """

    return re.search(r"(?<!Pseudo )" + re.escape(label), rendered) is not None


def _frame(n: int = 250, seed: int = 6400):
    rng = np.random.default_rng(seed)
    covariate = rng.normal(size=n)
    latent = 0.4 + 0.9 * covariate
    continuous = 1.0 + 0.8 * covariate + rng.normal(size=n)
    return pd.DataFrame(
        {
            "y": continuous,
            "yc": np.clip(continuous, 0.0, None),
            "yb": (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-latent))).astype(int),
            "ycnt": rng.poisson(np.exp(0.3 + 0.4 * covariate)),
            "x": covariate,
        }
    )


@pytest.fixture(scope="module")
def frame():
    return _frame()


# ------------------------------------------------------------- the OLS block
def test_regress_uses_the_ols_block(frame) -> None:
    rendered = str(ws.OLS.from_formula("y ~ x", frame).fit().summary())

    for label in OLS_LABELS:
        assert _has_standalone_label(rendered, label), label
    for label in ML_LABELS:
        assert label not in rendered, label


# -------------------------------------------------------------- the ML block
@pytest.mark.parametrize(
    "builder",
    [
        lambda frame: ws.Logit.from_formula("yb ~ x", frame).fit(),
        lambda frame: ws.Probit.from_formula("yb ~ x", frame).fit(),
        lambda frame: ws.Poisson.from_formula("ycnt ~ x", frame).fit(),
        lambda frame: ws.Tobit.from_formula("yc ~ x", frame, lower=0).fit(),
    ],
    ids=["logit", "probit", "poisson", "tobit"],
)
def test_the_ml_families_use_the_ml_block(frame, builder) -> None:
    rendered = str(builder(frame).summary())

    for label in ML_LABELS:
        assert label in rendered, label
    for label in OLS_LABELS:
        assert not _has_standalone_label(rendered, label), label


def test_tobit_is_the_case_that_rules_out_keying_on_dist(frame) -> None:
    """``tobit`` has ``dist == 't'`` yet reports LR chi2 -- the block follows the REPORTS.

    Pinning this stops a later "simplify to ``if dist == 'z'``" edit, which is exactly the
    mistake that produced the original defect.
    """

    fitted = ws.Tobit.from_formula("yc ~ x", frame, lower=0).fit()

    assert fitted.dist == "t"
    assert "LR chi2:" in str(fitted.summary())
    assert "F-statistic:" not in str(fitted.summary())


def test_no_dots_are_printed_for_statistics_the_command_does_not_have(frame) -> None:
    """The visible symptom: a row of ``.`` values where an ML table should be.

    ``R-squared: .`` next to ``Adj. R-squared: .`` and ``F-statistic: .`` is what an ML
    fit used to show; none of those labels should appear at all now.
    """

    rendered = str(ws.Logit.from_formula("yb ~ x", frame).fit().summary())

    assert "R-squared:                 ." not in rendered
    assert "F-statistic:               ." not in rendered


# ------------------------------------------------------------ what stays true
@pytest.mark.parametrize(
    "builder",
    [
        lambda frame: ws.OLS.from_formula("y ~ x", frame).fit(),
        lambda frame: ws.Logit.from_formula("yb ~ x", frame).fit(),
        lambda frame: ws.Poisson.from_formula("ycnt ~ x", frame).fit(),
    ],
    ids=["regress", "logit", "poisson"],
)
def test_every_table_keeps_the_essentials(frame, builder) -> None:
    rendered = str(builder(frame).summary())

    assert "No. Observations:" in rendered
    assert "Log-Likelihood:" in rendered
    assert "coef" in rendered
    assert "std err" in rendered


@pytest.mark.parametrize(
    "builder,stat_column",
    [
        (lambda frame: ws.OLS.from_formula("y ~ x", frame).fit(), "P>|t|"),
        (lambda frame: ws.Logit.from_formula("yb ~ x", frame).fit(), "P>|z|"),
        (lambda frame: ws.Poisson.from_formula("ycnt ~ x", frame).fit(), "P>|z|"),
    ],
    ids=["regress", "logit", "poisson"],
)
def test_the_test_statistic_column_follows_the_distribution(
    frame, builder, stat_column: str
) -> None:
    """OLS reports t; the ML families report z. That part was already right."""

    rendered = str(builder(frame).summary())

    assert stat_column in rendered