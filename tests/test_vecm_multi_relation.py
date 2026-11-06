"""The cointegrating block on a MULTI-RELATION system.

Round 72 fixed the fabricated chi2 for a rank-1 model. Testing rank 2 exposed two more faults,
both of which only appear when a system has more than one cointegrating relation:

  * the block printed ONE row where Stata prints one per relation, with a statistic pooling all
    of them;
  * a relation whose free coefficient had a zero standard error raised ZeroDivisionError and
    killed the entire summary.

With independent random walks (a, b, c with two shared trends), rank 2:

    _ce1                  1   1.156643   0.2822
    _ce2                  1   1.141828   0.2853

Two indexing details decide whether this matches:

  * beta is (variables x relations), so each relation is a COLUMN -- reading the array flattened
    gave a statistic 0.9% high;
  * the stored stderr_beta is the ML covariance, while Stata's V_beta divides by the degrees of
    freedom. The ratio is exactly N / (N - df_eq), and the correction DIVIDES the statistic
    (multiplying moved it further away: 1.1894 -> 1.2230).
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


def test_one_row_per_relation(fitted) -> None:
    rows = [
        line
        for line in str(fitted.summary()).splitlines()
        if line.startswith("_ce") and "|" not in line and "P>chi2" not in line
    ]
    assert len(rows) == 2
    assert rows[0].startswith("_ce1")
    assert rows[1].startswith("_ce2")


def test_both_statistics_match_stata(fitted) -> None:
    rendered = str(fitted.summary())
    assert "_ce1                  1   1.156643   0.2822" in rendered
    assert "_ce2                  1   1.141828   0.2853" in rendered


def test_the_summary_does_not_raise(fitted) -> None:
    """A zero standard error used to raise ZeroDivisionError here."""

    assert isinstance(str(fitted.summary()), str)


def test_the_degrees_of_freedom_correction_divides(fitted) -> None:
    """Multiplying instead pushes the statistic to 1.2230."""

    rendered = str(fitted.summary())
    assert "1.2230" not in rendered
    assert "1.189379" not in rendered


def test_rank_one_still_matches(fitted) -> None:
    """The rank-1 case must not regress while fixing rank 2."""

    frame = pd.read_stata(r"D:\myCodeProject\test_westata\data\macro_swatson.dta")[
        ["dinf", "unem"]
    ].dropna()
    try:
        single = ws.VECM(
            frame[["dinf", "unem"]].to_numpy(float), coint_rank=1, lags=2,
            names=["dinf", "unem"],
        ).fit()
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")
    assert "_ce1                  1   8.358789   0.0038" in str(single.summary())
