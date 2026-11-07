"""`nbreg` reports e(df_m) NET of the dispersion parameter.

Measured on `nbreg mpg weight fgn` -- TWO slopes -- against a live Stata:

    e(df_m) = 1
    e(chi2) = 69.876594
    e(k_aux) = 1

and the printed header says "LR chi2(1)". This package reported df_m = 2, so the header read
"LR chi2(2)" beside a statistic whose p-value had been computed for two degrees of freedom: the
label and the number disagreed with Stata and with each other.

`_model_test` counts `_slope_positions`, which correctly excludes `/lnalpha` and the constant, so
two slopes really are two positions -- Stata's convention is the additional step of subtracting
`e(k_aux)`. That is now applied in `_DiscreteBase`, gated on `_aux_count` so the models with no
auxiliary parameter are untouched.

The chi2 value itself was already right: 69.875793 here against Stata's 69.876594, a 8e-4 gap from
the optimiser's stopping point rather than from the test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

AUTO = r"tests\data\auto.dta"


@pytest.fixture(scope="module")
def frame():
    try:
        base = pd.read_stata(AUTO).dropna(subset=["mpg", "weight"]).copy()
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("auto.dta is not available")
    base["fgn"] = (
        base["foreign"]
        .astype(str)
        .str.lower()
        .str.startswith(("foreign", "1"))
        .astype(float)
    )
    return base


@pytest.fixture(scope="module")
def nbreg(frame):
    return ws.NegativeBinomial.from_formula("mpg ~ weight + fgn", frame).fit()


def test_df_m_excludes_the_dispersion(nbreg) -> None:
    """Two slopes, one auxiliary parameter: Stata reports 1."""

    assert int(nbreg.e["df_m"]) == 1


def test_k_aux_records_the_dispersion(nbreg) -> None:
    assert int(nbreg.e["k_aux"]) == 1


def test_the_rank_still_counts_every_parameter(nbreg) -> None:
    """e(rank) is the FULL parameter count, so it stays above df_m."""

    assert int(nbreg.e["rank"]) == 4
    assert int(nbreg.e["rank"]) > int(nbreg.e["df_m"])


def test_the_chi2_value_is_near_stata(nbreg) -> None:
    """Stata: 69.876594. The gap is the optimiser's stopping point, not the test."""

    assert float(nbreg.e["chi2"]) == pytest.approx(69.876594, abs=5e-3)


def test_the_header_labels_the_reduced_degrees_of_freedom(nbreg) -> None:
    rendered = str(nbreg.summary(style="stata"))
    assert "LR chi2(1)" in rendered
    assert "LR chi2(2)" not in rendered


def test_a_model_without_an_auxiliary_parameter_is_unchanged(frame) -> None:
    """`poisson` has no k_aux, so its df_m must stay at the slope count."""

    fitted = ws.Poisson.from_formula("mpg ~ weight + fgn", frame).fit()
    assert int(fitted.e["df_m"]) == 2
    assert "LR chi2(2)" in str(fitted.summary(style="stata"))
