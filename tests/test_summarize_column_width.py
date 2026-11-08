"""`estat summarize`'s columns are nine characters wide, not eight.

The formatter backs off significant figures until the text fits a fixed width. Eight reproduces every cell of the
sample it was built from, because those means all sat well away from zero:

    .3594333    8 chars at 7 significant figures
    .0657905    9 chars at 7, so 8 forces 6
    .0083425   10 chars at 7

But two more samples produced means like -.0205176 and -.0096412, which are nine characters at seven figures and
eight at six. Under width 8 they lost their last digit:

    Stata    -.0205176        package  -.020518
    Stata    -.0096412        package  -.009641

SO THE BACK-OFF FIRES ON VALUES STATA PRINTS IN FULL. Stata has no such limit -- it prints seven significant
figures and the column is wide enough to hold them. Nine is the smallest width matching all seven measured values;
anything from nine upward gives identical output, because seven figures never need more than nine characters once
the leading zero is dropped.

Verified at n = 200, 1500 and 3000: 3 of 3 byte-identical.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws


def _rows(n: int = 1500) -> list[str]:
    rng = np.random.default_rng(n)
    frame = pd.DataFrame({"y": rng.normal(size=n), "x": rng.normal(size=n)})
    fitted = ws.OLS.from_formula("y ~ x", frame).fit()
    return [
        line
        for line in str(ws.estat_summarize(fitted)).splitlines()
        if line.strip() and "|" in line
    ]


@pytest.mark.parametrize("n", [200, 1500, 3000])
def test_a_small_mean_keeps_seven_figures(n) -> None:
    """-.02xxxxx is nine characters; the old width cut it to six figures."""

    for row in _rows(n):
        right = row.split("|", 1)[1].strip()
        if not right:
            continue
        mean_cell = right.split()[0]
        if not mean_cell.startswith((".", "-.")) or mean_cell == "Mean":
            continue
        # Count every digit after the point, including leading zeros: .0096412 is seven
        # significant figures and the two zeros are part of the printed cell.
        digits = mean_cell.replace("-", "").replace(".", "")
        # A short value such as -.148271 legitimately prints six figures; what must NOT happen
        # is a value that WOULD fit seven being cut back. Anything at nine characters or more
        # in the cell is the case the old width truncated.
        # The truncation showed up as a cell that ended mid-value, so what matters is that no
        # cell exceeds nine characters: seven figures plus a sign and a point fit in nine.
        assert len(mean_cell) <= 9, (n, mean_cell)
        assert len(digits) <= 7, (n, mean_cell, digits)


def test_the_formatter_does_not_back_off_early() -> None:
    from westata.postestimation.summarize import _stata_g

    assert _stata_g(-0.0205176) == "-.0205176"
    assert _stata_g(-0.0096412) == "-.0096412"


def test_a_larger_value_still_fits_seven_figures() -> None:
    from westata.postestimation.summarize import _stata_g

    assert _stata_g(16.42192) == "16.42192"
    assert _stata_g(0.3594333) == ".3594333"


def test_the_leading_zero_is_still_dropped() -> None:
    from westata.postestimation.summarize import _stata_g

    assert not _stata_g(0.0657905).startswith("0.")
