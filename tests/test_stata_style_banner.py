"""The Stata-style banner is the command's own ``e(title)``, not a fixed string.

``render_summary(result, style="stata")`` prints a text banner above the table. A single
fixed banner ("Stata-style regression results") is wrong for every command whose real
banner differs, and Stata's banner differs per command:

    regress.ado:176    "Linear regression"
    logit.ado:253      "Logistic regression"
    probit.ado:252     "Probit regression"
    tobit.ado:127      "Tobit regression"

Those values are already recorded in ``e(title)`` (rounds 223 and 233), so the renderer
now reads the macro instead of hard-coding one line.

``poisson`` keeps the generic banner: ``poisson.ado`` contains no
``ereturn local title`` at all, so there is no value to print and a guess would be worse
than the neutral fallback.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.results.summary import render_summary

GENERIC_BANNER = "Stata-style regression results"



# Option C prints Stata's `Iteration 0` line (and count models may print a `note:`)
# BEFORE the banner, so the first non-blank line is no longer necessarily the banner.
def _is_preamble(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith(("Iteration", "note:")) or stripped.startswith("      ")

def _frame(n: int = 250, seed: int = 6500):
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


def _banner(rendered: str) -> str:
    """The banner text, whether it stands alone or shares its line with a statistic.

    ``tobit`` and ``quantile`` print the banner on the SAME line as "Number of obs"
    (measured from live runs), so taking the whole first line returns
    "Tobit regression            Number of obs = ..." instead of the title. Splitting on
    the run of spaces that separates the two columns yields the banner either way.
    """

    # Skip the LEADING blank line the renderer now emits (round 199 restored it; the raw Stata
    # capture starts with a newline for every estimator checked).
    lines = rendered.splitlines()
    first = next((line for line in lines if line.strip() and not _is_preamble(line)), "") if lines else ""
    return re.split(r"\s{2,}", first.strip())[0].strip()


# --------------------------------------------------------------- per command
@pytest.mark.parametrize(
    "builder,expected",
    [
        (
            lambda frame: ws.Logit.from_formula("yb ~ x", frame).fit(),
            "Logistic regression",
        ),
        (
            lambda frame: ws.Probit.from_formula("yb ~ x", frame).fit(),
            "Probit regression",
        ),
        (
            lambda frame: ws.Tobit.from_formula("yc ~ x", frame, lower=0).fit(),
            "Tobit regression",
        ),
    ],
    ids=["logit", "probit", "tobit"],
)
def test_the_banner_is_the_commands_title(frame, builder, expected: str) -> None:
    assert _banner(render_summary(builder(frame), style="stata")) == expected


def test_plain_regress_prints_no_banner(frame) -> None:
    """``regress`` stores ``e(title) = "Linear regression"`` but never prints it.

    Measured from a live ``regress y k``: the table opens directly at
    "      Source |       SS           df       MS". The ROBUST variant DOES print the title,
    heading its two-column block -- so the banner is not absent from the command altogether,
    only from the plain form.

    This test used to be part of the parametrised list above, asserting "Linear regression"
    was the banner. That pinned the extra header line the gap report's C04-01 flagged.
    """

    fitted = ws.OLS.from_formula("y ~ x", frame).fit()
    rendered = render_summary(fitted, style="stata")

    assert fitted.e["title"] == "Linear regression"
    assert "Linear regression" not in rendered
    # The renderer prefixes one blank line; Stata's own output has it too.
    first = next(line for line in rendered.splitlines() if line.strip() and not _is_preamble(line))
    assert first.lstrip().startswith("Source")


def test_robust_regress_does_print_the_banner(frame) -> None:
    """The counterpart: with a robust VCE the title heads the two-column block."""

    fitted = ws.OLS.from_formula("y ~ x", frame).fit(cov_type="HC1")
    rendered = render_summary(fitted, style="stata")

    assert "Linear regression" in rendered


def test_poisson_uses_its_own_title(frame) -> None:
    """``poisson.ado:241`` passes ``title(Poisson regression)`` to ``ml``.

    Round 233 read the absence of an ``ereturn local title`` line as "no title" and this
    test asserted the generic fallback.  The value exists; it just travels as an ``ml``
    option rather than an ``ereturn local`` assignment, and ``poisson.sthlp:247``
    documents the key.
    """

    fitted = ws.Poisson.from_formula("ycnt ~ x", frame).fit()

    assert fitted.e["title"] == "Poisson regression"
    assert _banner(render_summary(fitted, style="stata")) == "Poisson regression"


# ------------------------------------------------------------------ the values
def test_the_banner_matches_the_macro_exactly(frame) -> None:
    """Guard against a paraphrase: the banner IS ``e(title)``."""

    fitted = ws.Logit.from_formula("yb ~ x", frame).fit()

    assert _banner(render_summary(fitted, style="stata")) == fitted.e["title"]


def test_the_banners_are_not_all_the_same(frame) -> None:
    """Guard: a fixed banner would satisfy the fallback test and fail this one."""

    banners = {
        _banner(render_summary(ws.OLS.from_formula("y ~ x", frame).fit(), style="stata")),
        _banner(
            render_summary(ws.Logit.from_formula("yb ~ x", frame).fit(), style="stata")
        ),
        _banner(
            render_summary(ws.Tobit.from_formula("yc ~ x", frame, lower=0).fit(), style="stata")
        ),
    }

    assert len(banners) == 3


# --------------------------------------------------- the rest of the table holds
def test_the_stata_style_table_still_has_its_parts(frame) -> None:
    """The table body is unchanged; the two scaffolding lines above it are gone.

    ``Model:`` and ``Covariance type:`` were this package's own header rows.  Stata
    prints neither -- the captured output at
    ``tests/conformance/stata_fixtures/p10_outputs.txt:97-98`` opens the table directly
    with ``Source | SS df MS`` and the ``Number of obs`` line beside it.  Round 247
    removed them.
    """

    rendered = render_summary(
        ws.OLS.from_formula("y ~ x", frame).fit(), style="stata"
    )

    assert "Model:" not in rendered
    assert "Covariance type:" not in rendered
    assert "Coefficient" in rendered
    assert "P>|t|" in rendered
    assert "_cons" in rendered
    assert "Source |" in rendered


def test_the_logit_stata_table_has_no_anova_block(frame) -> None:
    """Stata's ``logit`` output has no Source/SS/MS table; nor should this."""

    rendered = render_summary(ws.Logit.from_formula("yb ~ x", frame).fit(), style="stata")

    assert "Source |" not in rendered
    assert "LR chi2" in rendered
    assert "Pseudo R2" in rendered


def test_the_regress_stata_table_KEEPS_its_anova_block(frame) -> None:
    """Guard on the other direction: ``regress`` must keep Source/SS/MS."""

    rendered = render_summary(ws.OLS.from_formula("y ~ x", frame).fit(), style="stata")

    assert "Source |" in rendered
    assert "Number of obs" in rendered


def test_the_statsmodels_style_is_unaffected(frame) -> None:
    """The default style never showed a Stata banner, and still does not."""

    rendered = ws.OLS.from_formula("y ~ x", frame).fit().summary()

    assert GENERIC_BANNER not in rendered
    assert "Linear regression" not in rendered