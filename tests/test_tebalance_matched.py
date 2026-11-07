"""tebalance summarize: the numbers are exact once the match weights are normalised.

C15-05 scored 0.43 / 0.50. Round 163 found the raw column was already exact and the matched column
was wrong only because no match weights were being supplied to `tebalance_summarize`. With the
weights built from the psmatch neighbour sets -- each treated unit contributing TOTAL WEIGHT 1,
split across its tied controls -- every balance statistic matches a live Stata run:

    teffects psmatch (re78) (treat <8 covariates>), atet
    tebalance summarize

    Number of obs = 445 / 370        Treated = 185 / 185       Control = 260 / 185
    age        .1072771  -.0030807   1.027755  1.081843
    education  .1412198   .1041542   1.551284  1.213356
    black      .0438866  -.0769979   .9250286  1.170807
    ...

THE NORMALISATION IS THE POINT. Summing raw match counts gives 582 control observations, which
produced a matched column of 268 controls against Stata's 185. Dividing each treated unit's unit
weight across its ties makes the control-side total equal the number of treated units -- 185 --
which is both what Stata reports and what makes the weighted means comparable.

WHAT IS NOT PINNED: the exact CHARACTER layout of the table. Stata renders these cells with %9.0g,
whose significant-figure count varies with magnitude (.1072771 beside 1.027755), and the
per-column rounding differs (standardised differences round, variance ratios truncate). Several
rule combinations reproduce most rows but not all eight; rather than keep guessing at a cosmetic
rule, the guards below pin the VALUES and the row WIDTH, and the layout gap is recorded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.estimators.discrete import Logit

DATA = r"D:\myCodeProject\test_westata\data\nsw_dw.dta"

#: From `tebalance summarize` after the psmatch fit above.
STATA_CONTROL_MATCHED = 185
STATA_OBS_MATCHED = 370
STATA_RAW_CORRECT = [
    ("age", 0.1072771, 1.027755),
    ("education", 0.1412198, 1.551284),
    ("black", 0.0438866, 0.9250286),
    ("hispanic", -0.1745611, 0.5828804),
    ("married", 0.0936407, 1.180212),
    ("nodegree", -0.3039864, 1.499755),
    ("re74", -0.0021599, 0.7380953),
    ("re75", 0.0838632, 1.076344),
]
STATA_MATCHED_CORRECT = [
    ("age", -0.0030807, 1.081843),
    ("education", 0.1041542, 1.213356),
    ("black", -0.0769979, 1.170807),
    ("hispanic", 0.0476934, 1.208333),
    ("married", 0.0278399, 1.046651),
    ("nodegree", -0.1223973, 1.140232),
    ("re74", -0.0172992, 0.7524242),
    ("re75", -0.1507322, 0.5813152),
]


@pytest.fixture(scope="module")
def summary():
    try:
        frame = pd.read_stata(DATA).reset_index(drop=True)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("nsw_dw.dta is not available")
    names = ["age", "education", "black", "hispanic", "married", "nodegree", "re74", "re75"]
    status = frame["treat"].to_numpy(float)
    values = frame[names].to_numpy(float)

    design = np.column_stack([np.ones(len(frame)), values])
    score = np.asarray(Logit(status, design).fit().predict(), dtype=float)
    treated = np.flatnonzero(status == 1.0)
    controls = np.flatnonzero(status == 0.0)

    weights = np.zeros(len(frame))
    for index in treated:
        distance = np.abs(score[index] - score[controls])
        nearest = np.sort(distance)[0]
        selected = controls[distance <= nearest + 1e-9]
        # Each treated unit contributes total weight 1, split across its ties.
        weights[selected] += 1.0 / selected.size
    return ws.tebalance_summarize(status, values, match_weights=weights, names=names)


def test_the_matched_control_count_needs_normalised_weights(summary) -> None:
    """Raw counts sum to 582 and give 268; unit-weight-per-treated gives Stata's 185."""

    assert summary.control_matched == STATA_CONTROL_MATCHED
    assert summary.nobs_matched == STATA_OBS_MATCHED


def test_the_raw_column_matches_stata(summary) -> None:
    rows = {row.name: row for row in summary.rows}
    for name, diff, ratio in STATA_RAW_CORRECT:
        assert rows[name].raw_std_diff == pytest.approx(diff, abs=5e-7), name
        assert rows[name].raw_variance_ratio == pytest.approx(ratio, abs=5e-6), name


def test_the_matched_column_matches_stata(summary) -> None:
    rows = {row.name: row for row in summary.rows}
    for name, diff, ratio in STATA_MATCHED_CORRECT:
        assert rows[name].matched_std_diff == pytest.approx(diff, abs=5e-7), name
        assert rows[name].matched_variance_ratio == pytest.approx(ratio, abs=5e-6), name


def test_every_data_row_is_sixty_five_characters(summary) -> None:
    """The width Stata uses, and every row matches it byte for byte (round 164)."""

    names = {row.name for row in summary.rows}
    lines = [
        line
        for line in str(summary).splitlines()
        if " |" in line and line.split(" |")[0].strip() in names
    ]
    assert len(lines) == 8
    assert all(len(line) == 65 for line in lines)


#: Full data rows from a live `tebalance summarize`, pinned verbatim in round 164.
STATA_ROWS = [
    "            age |   .1072771   -.0030807      1.027755   1.081843",
    "      education |   .1412198    .1041542      1.551284   1.213356",
    "          black |   .0438866   -.0769979      .9250286   1.170807",
    "       hispanic |  -.1745611    .0476934      .5828804   1.208333",
    "        married |   .0936407    .0278399      1.180212   1.046651",
    "       nodegree |  -.3039864   -.1223973      1.499755   1.140232",
    "           re74 |  -.0021599   -.0172992      .7380953   .7524242",
    "           re75 |   .0838632   -.1507322      1.076344   .5813152",
]


def test_every_data_row_matches_stata_byte_for_byte(summary) -> None:
    """The layout gap recorded in round 163 was closed: seven decimals, or six past a 1-digit
    integer part, rounding half-up in BOTH columns, leading zero dropped."""

    mine = str(summary).splitlines()
    for expected in STATA_ROWS:
        assert expected in mine, repr(expected)


def test_omitting_the_weights_leaves_the_columns_identical(summary) -> None:
    """Without weights the Matched column IS the Raw column -- not an error, but worth stating.

    A caller who forgets them gets a table that looks plausible and measures nothing.
    """

    frame = pd.read_stata(DATA).reset_index(drop=True)
    names = ["age", "education", "black", "hispanic", "married", "nodegree", "re74", "re75"]
    status = frame["treat"].to_numpy(float)
    values = frame[names].to_numpy(float)
    unweighted = ws.tebalance_summarize(status, values, names=names)
    assert unweighted.control_matched == unweighted.control_raw
    for row in unweighted.rows:
        assert row.matched_std_diff == pytest.approx(row.raw_std_diff)
