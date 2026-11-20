"""`varsoc`'s configuration options, measured against Stata rather than assumed.

The other five commands (`varlmar`, `varstable`, `varwle`, `vargranger`, and the `irf` suite) were
already byte-identical; `varsoc` was the one with gaps.

WHICH OPTIONS EXIST was PROBED, not recalled. Stata 19.5 accepts:

    maxlag(#)   exog(varlist)   sep(#)   constant

and REJECTS `noforsc`, `dots` and `dfk` with ``r(198)``, so those are deliberately not modelled.
`sep()`` only spaces the printed block and changes no number.

Two rules about the lag-0 row only became visible through the `noconstant` captures:

* the row disappears only when there is NOTHING to regress on -- no constant AND no exog;
* when it does disappear, the first printed row still carries ``.`` in the LR and p columns, with df
  showing the equation count.
"""

import contextlib
import io
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "replication")

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\gdp_china.dta"


def _series(n: int = 220, seed: int = 7) -> pd.DataFrame:
    """A well-behaved VAR(2) process.

    Generated rather than taken from the fixture: gdp_china's derived bivariate series produced a
    degenerate third lag (det(Sigma) around 1e-22) and Stata's own varsoc table jumped by a factor of
    six there, which makes it useless for comparing formatting.
    """

    rng = np.random.default_rng(seed)
    values = np.zeros((n, 2))
    for t in range(2, n):
        values[t, 0] = (
            0.5 * values[t - 1, 0] - 0.2 * values[t - 2, 0] + 0.1 * values[t - 1, 1] + rng.normal()
        )
        values[t, 1] = (
            0.2 * values[t - 1, 0] + 0.4 * values[t - 1, 1] - 0.1 * values[t - 2, 1] + rng.normal()
        )
    frame = pd.DataFrame(values, columns=["a", "b"])
    frame["t"] = np.arange(1, n + 1)
    frame["x"] = rng.normal(size=n)
    return frame


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return _series()


def _lines(text: str) -> list[str]:
    return [line for line in text.split("\n") if line.strip()]


def _stata(frame: pd.DataFrame, command: str) -> list[str]:
    try:
        from lib.stata_session import StataSession
    except Exception:  # noqa: BLE001
        pytest.skip("live Stata is not available")

    try:
        with StataSession() as stata:
            stata.push(frame)
            with contextlib.redirect_stdout(io.StringIO()):
                stata.run("tsset t")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                stata.run(command)
            return _lines(out.getvalue())
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"live Stata unavailable: {exc}")


# ---------------------------------------------------------------- the four accepted options
def test_the_plain_table_matches(frame) -> None:
    ours = _lines(str(ws.var_select_order(frame[["a", "b"]].to_numpy(), names=["a", "b"], maxlag=3)))
    assert ours == _stata(frame, "varsoc a b, maxlag(3)")


def test_exog_enters_every_equation(frame) -> None:
    """`exog(x)` puts x in each design and names it in the footnote."""

    ours = _lines(
        str(
            ws.var_select_order(
                frame[["a", "b"]].to_numpy(), names=["a", "b"], maxlag=3, exog=frame[["x"]]
            )
        )
    )
    assert ours == _stata(frame, "varsoc a b, maxlag(3) exog(x)")


def test_exog_names_come_from_the_frame(frame) -> None:
    """A generated x1/x2 would print a footnote Stata never shows."""

    result = ws.var_select_order(
        frame[["a", "b"]].to_numpy(), names=["a", "b"], maxlag=2, exog=frame[["x"]]
    )
    assert result.exog_names == ("x",)
    assert "Exogenous: x  _cons" in str(result)


def test_noconstant_matches(frame) -> None:
    ours = _lines(
        str(ws.var_select_order(frame[["a", "b"]].to_numpy(), names=["a", "b"], maxlag=3, constant=False))
    )
    assert ours == _stata(frame, "varsoc a b, maxlag(3) noconstant")


def test_noconstant_with_exog_keeps_the_lag_zero_row(frame) -> None:
    """Measured: lag 0 survives when `x` is a regressor, even without a constant."""

    ours = _lines(
        str(
            ws.var_select_order(
                frame[["a", "b"]].to_numpy(),
                names=["a", "b"],
                maxlag=2,
                exog=frame[["x"]],
                constant=False,
            )
        )
    )
    assert ours == _stata(frame, "varsoc a b, maxlag(2) exog(x) noconstant")
    assert any(line.strip().startswith("|   0 |") for line in ours)


def test_noconstant_alone_drops_the_lag_zero_row(frame) -> None:
    ours = _lines(
        str(ws.var_select_order(frame[["a", "b"]].to_numpy(), names=["a", "b"], maxlag=3, constant=False))
    )
    assert not any(line.strip().startswith("|   0 |") for line in ours)


def test_the_noconstant_first_row_carries_dots(frame) -> None:
    """Measured on `varsoc a b, maxlag(3) noconstant`."""

    rows = [
        line
        for line in _lines(
            str(ws.var_select_order(frame[["a", "b"]].to_numpy(), names=["a", "b"], maxlag=3, constant=False))
        )
        if line.strip().startswith("|   1 |")
    ]
    assert rows
    row = rows[0]
    assert "      . " in row          # the LR placeholder
    assert "    4      . " in row     # df is the equation count, then the p placeholder


def test_the_empty_exogenous_label_keeps_its_space(frame) -> None:
    """Stata prints `    Exogenous: ` with a trailing space when noconstant empties it."""

    text = str(ws.var_select_order(frame[["a", "b"]].to_numpy(), names=["a", "b"], maxlag=3, constant=False))
    assert "    Exogenous: " in text


# ---------------------------------------------------------------- rejected options
def test_rejected_options_are_not_offered() -> None:
    """`noforsc`, `dots` and `dfk` all raise r(198) in Stata, so the signature must not accept them."""

    import inspect

    parameters = set(inspect.signature(ws.var_select_order).parameters)
    for absent in ("noforsc", "dots", "dfk", "sep"):
        assert absent not in parameters


# ---------------------------------------------------------------- the other five commands
@pytest.fixture(scope="module")
def fitted(frame):
    return ws.VAR(frame[["a", "b"]].to_numpy(), lags=2, names=["a", "b"]).fit()


def test_varlmar_matches(frame, fitted) -> None:
    assert _lines(str(ws.var_lmar(fitted))) == [
        line for line in _stata(frame, "quietly var a b, lags(1/2)\nvarlmar")
    ]


def test_varstable_matches(frame, fitted) -> None:
    assert _lines(str(ws.var_stable(fitted))) == [
        line for line in _stata(frame, "quietly var a b, lags(1/2)\nvarstable")
    ]


def test_varwle_matches(frame, fitted) -> None:
    assert _lines(str(ws.var_wle(fitted))) == [
        line for line in _stata(frame, "quietly var a b, lags(1/2)\nvarwle")
    ]


def test_vargranger_matches(frame, fitted) -> None:
    assert _lines(str(ws.var_granger(fitted))) == [
        line for line in _stata(frame, "quietly var a b, lags(1/2)\nvargranger")
    ]
