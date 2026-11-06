"""A rank-2 system shows _ce1 AND _ce2 inside every d() equation.

alpha is (equations x relations), so each differenced equation gets one block per cointegrating
relation. The renderer read alpha[:, 0], so a rank-2 model showed only _ce1 and silently skipped
_ce2 -- the third instance of the same defect in vec's output (rounds 73, 74, 75).

Against a live ``vec a b c, lags(2) rank(2)`` on three independent random walks, the d(a)
equation reads:

    D_a          |
            _ce1 |
             L1. |   .1279021   .0578459     2.21   0.027     .0145262     .241278
                 |
            _ce2 |
             L1. |  -.1163404   .0509856    -2.28   0.022    -.2162703   -.0164105

The COEFFICIENTS match exactly (.1279021 and -.1163404); the standard errors differ in the third
digit because westata's stderr_alpha carries the ML covariance where Stata's uses the
degrees-of-freedom one. That scaling is a separate open item -- the structure is what this test
pins.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(11)
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


def test_each_equation_has_both_ce_blocks(fitted) -> None:
    rendered = str(fitted.summary()).splitlines()
    start = next(index for index, line in enumerate(rendered) if line.startswith("D_a") and "|" in line)
    block = rendered[start:start + 12]
    labels = [line.strip() for line in block if line.strip().endswith("|")]
    assert "_ce1 |" in labels
    assert "_ce2 |" in labels


def test_the_adjustment_coefficients_match_stata(fitted) -> None:
    rendered = str(fitted.summary())
    assert ".1279021" in rendered
    assert "-.1163404" in rendered


def test_rank_one_still_shows_a_single_block() -> None:
    try:
        frame = pd.read_stata(r"D:\myCodeProject\test_westata\data\macro_swatson.dta")[
            ["dinf", "unem"]
        ].dropna()
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")
    single = ws.VECM(
        frame[["dinf", "unem"]].to_numpy(float), coint_rank=1, lags=2, names=["dinf", "unem"]
    ).fit()
    assert "_ce2" not in str(single.summary())


def test_no_alpha_column_is_ignored(fitted) -> None:
    """alpha has one column per relation; the render must consume all of them."""

    alpha = np.asarray(fitted.alpha)
    assert alpha.ndim == 2 and alpha.shape[1] == 2
    rendered = str(fitted.summary())
    assert "_ce2" in rendered
