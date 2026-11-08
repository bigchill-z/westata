"""Pin the ``predict`` + ``summarize`` workflow that two report cases described as missing.

C04-04 ("predict residuals returns an ndarray, no Stata summary table"), C05-08 ("predict xb
numerically identical but you have to summarize yourself") and C09-04 ("leverage identical, no
Stata summarize table") were all blocked by the same thing, and none of them named it: there
was no ``summarize`` command. Round 92 added it, so the whole workflow can now be checked end
to end.

Reference, ``regress y k l`` then ``predict`` and ``summarize xb lev ck`` on cobb_douglas.dta::

        Variable |        Obs        Mean    Std. dev.       Min        Max
    -------------+---------------------------------------------------------
              xb |         24    165.9167    42.47537   102.6653   236.5989
             lev |         24        .125    .1107656   .0421769   .5036705
              ck |         24     .187198    .5508187   2.40e-06   2.253077

The three predict kinds and the descriptive table are checked separately and together, because
either half being wrong produces a table that looks plausible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\cobb_douglas.dta"

#: kind -> (mean, std dev, min, max) as Stata's ``summarize`` prints them.
STATA_STATS = {
    "xb": (165.9167, 42.47537, 102.6653, 236.5989),
    "leverage": (0.125, 0.1107656, 0.0421769, 0.5036705),
    "cooksd": (0.187198, 0.5508187, 2.40e-06, 2.253077),
}


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("cobb_douglas.dta is not available")
    return ws.OLS.from_formula("y ~ k + l", frame).fit()


@pytest.mark.parametrize("kind", ["xb", "leverage", "cooksd"])
def test_each_predict_kind_matches_statas_summary(fitted, kind: str) -> None:
    values = np.asarray(fitted.predict(kind=kind), dtype=float)
    mean, deviation, minimum, maximum = STATA_STATS[kind]
    assert float(values.mean()) == pytest.approx(mean, rel=1e-4), "mean"
    assert float(values.std(ddof=1)) == pytest.approx(deviation, rel=1e-4), "std dev"
    # Cook's D's minimum is 2.4044677e-06, which %9.0g prints as "2.4e-06" -- two significant
    # figures. Comparing against the PRINTED value therefore needs a loose relative tolerance;
    # a tight one fails on a value that actually agrees.
    assert float(values.min()) == pytest.approx(minimum, rel=2e-2), "min"
    assert float(values.max()) == pytest.approx(maximum, rel=1e-4), "max"


def test_the_residuals_sum_to_zero(fitted) -> None:
    """An OLS property that must survive into the predict output."""

    residuals = np.asarray(fitted.predict(kind="residuals"), dtype=float)
    assert float(residuals.sum()) == pytest.approx(0.0, abs=1e-9)


def test_the_combined_table_matches_stata_line_for_line(fitted) -> None:
    frame = pd.DataFrame(
        {
            "xb": np.asarray(fitted.predict(kind="xb"), dtype=float),
            "lev": np.asarray(fitted.predict(kind="leverage"), dtype=float),
            "ck": np.asarray(fitted.predict(kind="cooksd"), dtype=float),
        }
    )
    rendered = str(ws.summarize(frame)).splitlines()
    expected = [
        "    Variable |        Obs        Mean    Std. dev.       Min        Max",
        "-------------+---------------------------------------------------------",
        "          xb |         24    165.9167    42.47537   102.6653   236.5989",
        "         lev |         24        .125    .1107656   .0421769   .5036705",
        "          ck |         24     .187198    .5508187   2.40e-06   2.253077",
    ]
    for line in expected:
        assert line in rendered, line
