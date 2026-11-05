"""Pin ``estat vce`` and ``estat gof`` to Stata's printed output.

``estat vce`` reference, from a live ``logit y x1 x2`` (n = 500)::

    Covariance matrix of coefficients of logit model
                 | y
            e(V) |         x1          x2       _cons
    -------------+-----------------------------------
    y            |
              x1 |   .0168814
              x2 | -.00338524   .01251233
           _cons |  .00121321  -.00025332   .01065644

``estat gof`` reference on the same fit::

    Goodness-of-fit test after logistic model
    Variable: y
          Number of observations =    500
    Number of covariate patterns =    500
               Pearson chi2(497) = 500.43
                     Prob > chi2 = 0.4484

The vce geometry that took several attempts: the pipe is at column 13,每一 cell is twelve
characters wide (value right-aligned in eleven plus one space), rows STOP at the diagonal
with no trailing padding, and the separator is exactly as wide as the header line.

The cell formatter is width-driven: seven significant figures where they fit in ten
characters, six where they do not (``-.00338524`` is already ten characters).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


def _frame():
    rng = np.random.default_rng(31337)
    n = 500
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    latent = 0.9 * x1 - 0.4 * x2 + rng.normal(scale=1.2, size=n)
    y = (latent > 0).astype(int)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


@pytest.fixture(scope="module")
def fitted():
    return ws.Logit.from_formula("y ~ x1 + x2", _frame()).fit()


VCE_LINES = [
    "Covariance matrix of coefficients of logit model",
    "             | y",
    "        e(V) |         x1          x2       _cons",
    "-------------+-----------------------------------",
    "y            |",
    "          x1 |   .0168814",
    "          x2 | -.00338524   .01251233",
    "       _cons |  .00121321  -.00025332   .01065644",
]

GOF_LINES = [
    "Goodness-of-fit test after logistic model",
    "Variable: y",
    "      Number of observations =    500",
    "Number of covariate patterns =    500",
    "           Pearson chi2(497) = 500.43",
    "                 Prob > chi2 = 0.4484",
]


def test_estat_vce_matches_stata_line_for_line(fitted) -> None:
    rendered = str(ws.estat_vce(fitted)).splitlines()
    assert rendered == VCE_LINES


def test_estat_vce_names_the_model_and_the_equation(fitted) -> None:
    """Three pieces were missing entirely: the model name, the column header and e(V)."""

    rendered = str(ws.estat_vce(fitted))
    assert "of logit model" in rendered
    assert "e(V)" in rendered
    assert "| y" in rendered


def test_estat_gof_matches_stata_line_for_line(fitted) -> None:
    rendered = str(ws.estat_gof(fitted)).splitlines()
    assert rendered == GOF_LINES


def test_estat_gof_reports_both_counts(fitted) -> None:
    """chi2(497) means nothing without knowing 497 = 500 - 3 covariate patterns."""

    rendered = str(ws.estat_gof(fitted))
    assert "Number of observations =    500" in rendered
    assert "Number of covariate patterns =    500" in rendered


def test_vce_rows_stop_at_the_diagonal(fitted) -> None:
    """Stata pads nothing past the diagonal; trailing blanks would break the widths."""

    rendered = str(ws.estat_vce(fitted)).splitlines()
    for line in rendered:
        assert line == line.rstrip(), line


def test_the_cell_formatter_is_width_driven() -> None:
    """Seven figures where they fit in ten characters, six where they do not."""

    from westata.postestimation.vce import _format_entry

    assert _format_entry(0.016881396, 11).strip() == ".0168814"
    assert _format_entry(-0.003385236, 11).strip() == "-.00338524"
    assert _format_entry(0.012512333, 11).strip() == ".01251233"
