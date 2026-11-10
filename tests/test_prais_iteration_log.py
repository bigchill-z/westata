"""`prais` prints Stata's iteration log, and reproducing it needed the ado's pass order.

The log is the first thing a reader sees and it had never been emitted. Three conditions decide it,
and only the third was missing:

  * rho = sum(e_t e_t-1)/sum(e_t-1 squared) -- `_crcar1.ado:37-42`, the DEFAULT branch. Already used.
  * corc drops the transformed first row -- `prais_e.ado:28-33`. Already done.
  * rho is estimated from the PREVIOUS fit's residual, printed, and only THEN used to refit
    (`prais.ado:172-217`). This is what every earlier attempt got wrong.

Measured on icecream, `prais consumption temp price income`:

    corc   : 8 lines, 7 iterations, rho .4009256
    default: 42 lines, 41 iterations, rho .8002264

`display e(ic)` on the default form reports 41, which is why `STATA_ITERATIONS` moved from 40 to 41:
the old value was one short because the loop stopped a pass early.
"""

import pathlib
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\icecream.dta"
FORMULA = "consumption ~ temp + price + income"

pytestmark = pytest.mark.skipif(
    not pathlib.Path(DATA).exists(),
    reason="icecream.dta is not available on this machine",
)


def _lines(corc: bool) -> list[str]:
    fitted = ws.prais(pd.read_stata(DATA), FORMULA, corc=corc)
    return str(fitted).splitlines()


def test_the_corc_log_matches_the_capture() -> None:
    expected = [
        "Iteration 0:  rho = 0.0000",
        "Iteration 1:  rho = 0.4006",
        "Iteration 2:  rho = 0.4008",
        "Iteration 3:  rho = 0.4009",
        "Iteration 4:  rho = 0.4009",
        "Iteration 5:  rho = 0.4009",
        "Iteration 6:  rho = 0.4009",
        "Iteration 7:  rho = 0.4009",
    ]
    lines = _lines(True)
    assert lines[1 : 1 + len(expected)] == expected, lines[:10]


def test_the_log_is_preceded_and_followed_by_a_blank_line() -> None:
    lines = _lines(True)
    assert lines[0] == ""
    count = sum(1 for line in lines if line.startswith("Iteration"))
    assert lines[count + 1] == "", lines[count : count + 2]


def test_the_spacing_narrows_at_two_digits() -> None:
    """`prais.ado:181-184`: two spaces below 10, one from 10 on, so the column stays aligned."""

    lines = _lines(False)
    assert "Iteration 9:  rho = " in "\n".join(lines)
    assert "Iteration 10: rho = " in "\n".join(lines)


def test_the_default_form_iterates_forty_one_times() -> None:
    """`display e(ic)` reports 41 on a live run."""

    fitted = ws.prais(pd.read_stata(DATA), FORMULA)
    assert int(fitted.iterations) == 41
    assert len(fitted.rho_trace) == 42


def test_the_corc_form_iterates_seven_times() -> None:
    fitted = ws.prais(pd.read_stata(DATA), FORMULA, corc=True)
    assert int(fitted.iterations) == 7
    assert len(fitted.rho_trace) == 8


def test_the_trace_starts_at_zero() -> None:
    """`Iteration 0` is a LITERAL zero (`prais.ado:201`), not the starting estimate."""

    assert ws.prais(pd.read_stata(DATA), FORMULA).rho_trace[0] == 0.0
