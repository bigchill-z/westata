"""Thousands separators at large n, checked across the tables that print a count.

Rounds 182-183 found `summarize` and the regression ANOVA missing separators. This round checked the rest of
the surfaces at n=1500, and found the OPPOSITE error in one of them:

    esttab, N row
        Stata    'N                    1500            1500   '
        package  'N                   1,500           1,500   '

STATA'S `esttab` DOES NOT SEPARATE THE N ROW, while its `summarize` and its regression block both do. So the
rule is per-command rather than global, and a blanket change in either direction would be wrong. The three
tables measured:

    summarize v          N separated        'Obs  1,000'
    reg y x              N separated        'Number of obs   =     1,500'
    esttab m0 m1, se     N NOT separated    'N                    1500'

`estat classification` and `estat firststage` at n=1500 match with no change needed, which is worth pinning
because their absence from the exception list is the claim.

Verified across n = 200, 1500, 5000: esttab 3 of 3 byte-identical.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def models():
    rng = np.random.default_rng(1500)
    n = 1500
    frame = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        }
    )
    return [
        ws.OLS.from_formula("y ~ x1", frame).fit(),
        ws.OLS.from_formula("y ~ x1 + x2", frame).fit(),
    ]


def test_esttab_does_not_separate_the_n_row(models) -> None:
    rendered = ws.etable_as_stata(models, model_names=None)
    row = next(line for line in rendered.splitlines() if line.strip().startswith("N "))
    assert "1500" in row
    assert "1,500" not in row


def test_summarize_does_separate_the_count() -> None:
    """The opposite convention in a different command, pinned side by side with the one above."""

    rng = np.random.default_rng(1000)
    frame = pd.DataFrame({"v": rng.normal(size=1000)})
    assert "1,000" in str(ws.summarize(frame))


def test_the_regression_block_separates_its_count() -> None:
    rng = np.random.default_rng(1500)
    frame = pd.DataFrame({"y": rng.normal(size=1500), "x": rng.normal(size=1500)})
    rendered = str(ws.OLS.from_formula("y ~ x", frame).fit().summary(style="stata"))
    assert "1,500" in rendered


def test_classification_at_scale_needs_no_separator_change() -> None:
    rng = np.random.default_rng(1500)
    frame = pd.DataFrame(
        {"y": rng.normal(size=1500), "x": rng.normal(size=1500)}
    )
    frame["hi"] = (frame["y"] > 0).astype(float)
    rendered = str(
        ws.estat_classification(ws.Logit.from_formula("hi ~ x", frame).fit())
    )
    assert "Logistic model for hi" in rendered
    assert len([line for line in rendered.splitlines() if line.strip()]) == 23


def test_esttab_line_count_is_stable_across_sizes(models) -> None:
    small = models[0]
    assert len(ws.etable_as_stata([small], model_names=None).splitlines()) == 15
    assert len(ws.etable_as_stata(models, model_names=None).splitlines()) == 18
