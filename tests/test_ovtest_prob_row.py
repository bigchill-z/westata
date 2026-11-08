"""`estat ovtest`'s Prob row is as wide as the F row above it.

The comment in the renderer said "Stata indents this line by ONE space", which is true at every sample where
the degrees-of-freedom count has three characters. At n=1500 the F row grows by one:

    n=200    'F(3, 194) =   1.63'    (18)    ' Prob > F = 0.1845'   (18)
    n=1500   'F(3, 1494) =   0.34'   (19)    '  Prob > F = 0.7969'  (19)

SO 'Prob' IS RIGHT-ALIGNED TO THE F ROW'S WIDTH rather than indented by a fixed amount, and the two rows grow
together. A fixed one-space indent is correct for the first case only.

Verified at n = 200, 500, 1500: 3 of 3 byte-identical.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws


def _lines(n: int) -> list[str]:
    rng = np.random.default_rng(n)
    frame = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        }
    )
    return [
        line
        for line in str(
            ws.estat_ovtest(ws.OLS.from_formula("y ~ x1 + x2", frame).fit())
        ).splitlines()
        if line.strip()
    ]


@pytest.mark.parametrize("n", [200, 500, 1500])
def test_the_prob_row_is_as_wide_as_the_f_row(n) -> None:
    lines = _lines(n)
    f_row = next(line for line in lines if line.startswith("F("))
    prob_row = next(line for line in lines if "Prob > F" in line)
    assert len(f_row) == len(prob_row), (n, f_row, prob_row)


@pytest.mark.parametrize("n", [200, 500, 1500])
def test_the_prob_label_is_right_aligned(n) -> None:
    """Its leading spaces are whatever makes the row match the F row above."""

    lines = _lines(n)
    f_row = next(line for line in lines if line.startswith("F("))
    prob_row = next(line for line in lines if "Prob > F" in line)
    expected_padding = len(f_row) - len(prob_row.lstrip())
    assert prob_row[:expected_padding].strip() == "", (n, prob_row)


def test_a_four_digit_df_widens_both_rows() -> None:
    """n=1500 gives F(3, 1494), four digits, and the Prob row follows."""

    small = next(line for line in _lines(200) if line.startswith("F("))
    large = next(line for line in _lines(1500) if line.startswith("F("))
    assert len(large) == len(small) + 1


@pytest.mark.parametrize("n", [200, 1500])
def test_the_heading_lines_are_present(n) -> None:
    lines = _lines(n)
    assert "Ramsey RESET test for omitted variables" in lines[0]
    assert any(line.startswith("Omitted:") for line in lines)
