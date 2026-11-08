"""The narrow `irf_table` form -- one impulse/response pair -- matches Stata line for line.

The wide form was aligned in rounds 147-148. Narrowing with `impulse()`/`response()` takes a different
branch, and that branch still carried two defects:

  * it printed six DECIMALS (1.004387) where Stata prints six SIGNIFICANT figures (1.00439);
  * its two header rows had no trailing spaces, and the word row's cell was one column wide.

Captured from `irf table oirf, noci impulse(y1) response(y1)` after `var y1 y2, lags(1/2)`:

    Results from var1
    ---------------------
             |      (1)  
        Step |     oirf  
    ---------+-----------
           0 |  1.00439
           1 |  .971741
           2 |  .959237
           3 |  .948666
    ---------------------
    (1) irfname = var1, impulse = y1, and response = y1.

Eleven lines, all equal -- the trailing blanks on the two header rows included, which is why the
comparison uses rstrip only on the RIGHT of the whole line and the assertion spells them out.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def var_fit():
    rng = np.random.default_rng(3)
    n = 200
    frame = pd.DataFrame(
        {
            "y1": np.cumsum(rng.normal(size=n)),
            "y2": np.cumsum(rng.normal(size=n)),
        }
    )
    return ws.VAR(frame[["y1", "y2"]], lags=2).fit()


@pytest.fixture(scope="module")
def narrow(var_fit):
    return var_fit.irf_table(
        statistic="oirf", impulse="y1", response="y1", periods=3
    )


def test_the_two_header_rows_carry_two_trailing_spaces(narrow) -> None:
    lines = narrow.splitlines()
    assert lines[2] == "         |      (1)  "
    assert lines[3] == "    Step |     oirf  "


def test_the_step_zero_value_uses_six_significant_figures(narrow) -> None:
    """1.00439, not 1.004387 -- the same rule the wide form needed in round 147."""

    assert "0 |  1.00439" in narrow
    assert "1.004387" not in narrow


def test_the_sub_one_values_keep_six_decimals(narrow) -> None:
    """Row order is: title, rule, (1), oirf, separator, then steps 0..3, rule, footnote."""

    lines = narrow.splitlines()
    assert lines[5] == "       0 |  1.00439"
    assert lines[6] == "       1 |  .971741"
    assert lines[7] == "       2 |  .959237"
    assert lines[8] == "       3 |  .948666"


def test_the_rules_and_footnote_match(narrow) -> None:
    lines = narrow.splitlines()
    assert lines[0] == "Results from var1"
    assert lines[1] == "-" * 21
    assert lines[4] == "-" * 9 + "+" + "-" * 11
    assert lines[9] == "-" * 21
    assert lines[10] == "(1) irfname = var1, impulse = y1, and response = y1."
    assert len(lines) == 11


def test_the_pair_selects_a_different_column(narrow, var_fit) -> None:
    """Picking y2 as the response must give a different series, not the same one relabelled."""

    other = var_fit.irf_table(
        statistic="oirf", impulse="y1", response="y2", periods=3
    )
    assert other.splitlines()[5] != narrow.splitlines()[5]
