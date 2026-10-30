"""The Stata-style coefficient header, matched to CAPTURED Stata output.

This file replaces an earlier version that asserted the opposite. The history matters,
because the earlier conclusion came from a plausible-but-wrong authority.

Round 237 read ``docs/superpowers/specs/2026-10-22-westata-design.md:209``, which says the
table is ``Coef. / Std. Err. / t or z / P>|t| / [95% Conf. Interval]``, and capitalised the
renderer to match. Round 248 checked the committed captures of real Stata runs instead:

    p8_outputs.txt:25   (qreg)   y | Coefficient  Std. err.      t  P>|t|  [95% conf. interval]
    p10_outputs.txt:106 (nl)     y | Coefficient  Std. err.      t  P>|t|  [95% conf. interval]
    p6p7_outputs.txt:18 (logit)  y | Coefficient  Std. err.      z  P>|z|  [95% conf. interval]

Every one prints ``Coefficient`` (not ``Coef.``), ``Std. err.`` with a LOWERCASE ``err``,
and an all-lowercase ``[95% conf. interval]``. Captured output outranks the design
document: the document described an intention, the captures record what Stata printed.

The level must still not be hard-coded -- that part of round 237 was right and is kept.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.results.summary import _ci_label, render_summary

FIXTURES = Path(__file__).resolve().parent / "conformance" / "stata_fixtures"


def _frame(n: int = 250, seed: int = 7800):
    rng = np.random.default_rng(seed)
    covariate = rng.normal(size=n)
    latent = 0.4 + 0.9 * covariate
    return pd.DataFrame(
        {
            "y": 1.0 + 0.8 * covariate + rng.normal(size=n),
            "yb": (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-latent))).astype(int),
            "x": covariate,
        }
    )


@pytest.fixture(scope="module")
def frame():
    return _frame()


@pytest.mark.parametrize(
    "builder",
    [
        lambda frame: ws.OLS.from_formula("y ~ x", frame).fit(),
        lambda frame: ws.Logit.from_formula("yb ~ x", frame).fit(),
        lambda frame: ws.QuantileRegression.from_formula("y ~ x", frame).fit(),
    ],
    ids=["regress", "logit", "qreg"],
)
def test_the_header_matches_the_captures(frame, builder) -> None:
    rendered = render_summary(builder(frame), style="stata")
    header = next(line for line in rendered.splitlines() if "Coefficient" in line)

    assert "Coefficient" in header
    assert "Std. err." in header
    assert "conf. interval]" in header


def test_the_first_column_is_spelled_out(frame) -> None:
    rendered = render_summary(ws.OLS.from_formula("y ~ x", frame).fit(), style="stata")

    assert "Coefficient" in rendered


def test_the_error_label_is_lowercase(frame) -> None:
    rendered = render_summary(ws.OLS.from_formula("y ~ x", frame).fit(), style="stata")

    assert "Std. err." in rendered
    assert "Std. Err." not in rendered


def test_the_interval_label_is_all_lowercase(frame) -> None:
    rendered = render_summary(ws.OLS.from_formula("y ~ x", frame).fit(), style="stata")

    assert "[95% conf. interval]" in rendered
    assert "[95% Conf. Interval]" not in rendered


def test_the_four_labels_appear_in_order(frame) -> None:
    rendered = render_summary(ws.OLS.from_formula("y ~ x", frame).fit(), style="stata")
    header = next(line for line in rendered.splitlines() if "Coefficient" in line)

    positions = [
        header.index("Coefficient"),
        header.index("Std. err."),
        header.index("P>|t|"),
        header.index("[95% conf. interval]"),
    ]
    assert positions == sorted(positions)


@pytest.mark.parametrize(
    "relative", ["p8_outputs.txt", "p10_outputs.txt", "p6p7_outputs.txt"]
)
def test_the_captures_really_contain_this_header(relative: str) -> None:
    text = (FIXTURES / relative).read_text(encoding="utf-8", errors="ignore")

    assert "Coefficient  Std. err." in text, relative
    assert "conf. interval]" in text, relative


def test_no_capture_uses_the_capitalised_forms() -> None:
    for name in ("p8_outputs.txt", "p10_outputs.txt", "p6p7_outputs.txt"):
        text = (FIXTURES / name).read_text(encoding="utf-8", errors="ignore")
        assert "Std. Err." not in text, name
        assert "Conf. Interval" not in text, name


@pytest.mark.parametrize(
    "level,expected",
    [
        (None, "[95% conf. interval]"),
        (0.90, "[90% conf. interval]"),
        (0.95, "[95% conf. interval]"),
        (0.99, "[99% conf. interval]"),
        (0.999, "[99.9% conf. interval]"),
    ],
)
def test_the_label_follows_the_level(level, expected: str) -> None:
    assert _ci_label(level) == expected


def test_the_rendered_table_uses_the_results_own_level(frame) -> None:
    fitted = ws.OLS.from_formula("y ~ x", frame).fit()
    rendered = render_summary(fitted, style="stata")

    assert _ci_label(getattr(fitted, "level", None)) in rendered


def test_the_statsmodels_style_is_unaffected(frame) -> None:
    rendered = ws.OLS.from_formula("y ~ x", frame).fit().summary()

    assert "std err" in rendered
    assert "Coefficient" not in rendered
