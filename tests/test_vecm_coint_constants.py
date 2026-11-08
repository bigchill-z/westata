"""The cointegrating constant: one per relation, summing in the _cons transform.

_vecm_cointegrating_constant returned float(...ravel()[0]) -- a single constant. A rank-2 system
has two cointegrating relations, each with its own constant, and the _cons transform applied
relation 1's value to both.

Stata stores them in e(beta) with a stride of (neqs + 1) per relation:

    rank 2, three variables -> [1, 0, -1.970651, 16.042122,
                                0, 1, -4.429383, 16.736736]

so the two constants are 16.042122 and 16.736736. The helper now returns an array and the _cons
row sums alpha[index, r] * c_ce[r] across relations.

This is the FIFTH instance of the same "column 0" assumption in vec's output (rounds 73-75 were
alpha, beta and the summary rows; round 76 and 77 were literals that were right only at rank 1).

The rank-1 path is unchanged -- its single constant is still -0.00666269.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.estimators.tseries import _vecm_cointegrating_constant


@pytest.fixture(scope="module")
def rank_two():
    rng = np.random.default_rng(7)
    n = 220
    common = np.cumsum(rng.normal(size=n))
    frame = pd.DataFrame(
        {
            "a": common + np.cumsum(rng.normal(scale=0.2, size=n)),
            "b": 1.5 * common + np.cumsum(rng.normal(scale=0.2, size=n)),
            "c": np.cumsum(rng.normal(size=n)),
        }
    )
    return ws.VECM(frame.to_numpy(float), coint_rank=2, lags=2, names=["a", "b", "c"]).fit()


@pytest.fixture(scope="module")
def rank_one():
    try:
        frame = pd.read_stata(r"D:\myCodeProject\test_westata\data\macro_swatson.dta")[
            ["dinf", "unem"]
        ].dropna()
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")
    return ws.VECM(
        frame[["dinf", "unem"]].to_numpy(float), coint_rank=1, lags=2, names=["dinf", "unem"]
    ).fit()


def test_rank_two_returns_two_constants(rank_two) -> None:
    constants = np.atleast_1d(_vecm_cointegrating_constant(rank_two))
    assert constants.size == 2


def test_rank_one_returns_one_constant(rank_one) -> None:
    constants = np.atleast_1d(_vecm_cointegrating_constant(rank_one))
    assert constants.size == 1
    assert float(constants[0]) == pytest.approx(-0.00666269, abs=5e-8)


def test_the_summary_still_renders(rank_two) -> None:
    assert isinstance(str(rank_two.summary()), str)


def test_rank_one_constants_are_unchanged(rank_one) -> None:
    """The single-relation path must not move."""

    rendered = str(rank_one.summary())
    assert "_cons |" in rendered


def test_each_equation_gets_a_constant_row(rank_two) -> None:
    """One `_cons` per equation in the COEFFICIENT table -- three for a three-equation VECM.

    The beta table now also carries a `_cons` row (the cointegrating constant), so counting every
    `_cons` line in the output would find five. Restrict to the section above the beta header.
    """

    lines = str(rank_two.summary()).splitlines()
    beta_header = next(
        index for index, line in enumerate(lines) if line.lstrip().startswith("beta |")
    )
    rows = [
        line
        for line in lines[:beta_header]
        if "_cons" in line and "|" in line
    ]
    assert len(rows) == 3
