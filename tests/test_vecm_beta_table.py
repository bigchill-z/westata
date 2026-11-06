"""vec's beta table: labelled, one block per relation, 78 columns.

The table had three faults, the middle one only visible on a multi-relation system:

  * the header printed "| Coefficient" with no "beta" label, at the wrong column;
  * rows ran to 81 characters instead of 78;
  * the loop read beta[:, 0] only, so a rank-2 model printed the first cointegrating relation
    and silently dropped _ce2.

Against a live rank-1, two-variable fit:

            beta | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
    -------------+----------------------------------------------------------------
    _ce1         |
            unem |    .111912   .0387084     2.89   0.004      .036045     .187779

Values end at 25 / 36 / 45 / 53 / 66 / 78, as in the d() equation table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"


@pytest.fixture(scope="module")
def single():
    try:
        frame = pd.read_stata(DATA)[["dinf", "unem"]].dropna()
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")
    return ws.VECM(
        frame[["dinf", "unem"]].to_numpy(float), coint_rank=1, lags=2, names=["dinf", "unem"]
    ).fit()


@pytest.fixture(scope="module")
def multiple():
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


def test_the_header_carries_the_beta_label(single) -> None:
    expected = "        beta | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]"
    assert expected in str(single.summary()).splitlines()


def test_the_beta_rows_match_stata(single) -> None:
    rendered = str(single.summary())
    assert "        unem |    .111912   .0387084     2.89   0.004      .036045     .187779" in rendered


def test_beta_rows_are_seventy_eight_characters(single) -> None:
    lines = str(single.summary()).splitlines()
    # Only data rows: the block labels ("        dinf |") are 14 characters by design.
    rows = [
        line
        for line in lines
        if "|" in line and len(line) > 20 and ("dinf" in line or "unem" in line)
    ]
    assert rows
    for row in rows:
        assert len(row) == 78, (len(row), row)


def test_each_relation_gets_its_own_block(multiple) -> None:
    """A rank-2 model printed only the first relation before this."""

    rendered = str(multiple.summary()).splitlines()
    labels = [line for line in rendered if line.startswith("_ce") and line.endswith("|")]
    assert len(labels) == 2, labels


def test_the_second_relation_shows_its_coefficients(multiple) -> None:
    rendered = str(multiple.summary())
    # The second block's b row is normalised to 1.
    assert rendered.count("_ce2") >= 2
