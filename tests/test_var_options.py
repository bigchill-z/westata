"""`var`'s configuration options, every one measured against a live Stata run.

The option list was PROBED, not recalled. Stata 19.5 accepts `lags()`, `exog()`, `noconstant`,
`dfk`, `small` and `level()` on `var`; it REJECTS `nobalc`, `noback`, `noprint`, `nofootnote`,
`noheader`, `novce`, `ml`, `mle`, `tmin()` and `tmax()` with ``r(198)``. Only the accepted six are
modelled here.

Four of the corrections were arithmetic that had to be measured:

* `dfk` multiplies the standard errors by ``sqrt(N/(N-k))`` and NOTHING else -- the log likelihood
  and the criteria do not move;
* `small` divides the equation Wald statistic by the number of non-constant coefficients
  (`lags*neqs + n_exog`), so `exog(x)` makes the divisor 5 where a plain fit uses 4;
* `noconstant` switches R-squared to the UNCENTRED denominator ``sum(y^2)``
  (.2401/.1972 against the centred .2146/.1690);
* the exogenous columns belong in `k_per_equation`, which drives `ml_factor` and therefore
  ``e(Sigma)``, ``e(detsig)`` and every Wald statistic. Leaving them out reported .6929486 where
  Stata prints .6849608.
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


def _frame(n: int = 180, seed: int = 7) -> pd.DataFrame:
    """A well-behaved bivariate VAR(2) with an exogenous column."""

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
    frame["x2"] = rng.normal(size=n)
    return frame


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return _frame()


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


def _fit(frame: pd.DataFrame, exog=None, **init):
    return ws.VAR(
        frame[["a", "b"]].to_numpy(),
        lags=2,
        names=["a", "b"],
        time=frame["t"],
        exog=None if exog is None else frame[list(exog)],
        **init,
    ).fit()


# ---------------------------------------------------------------- every combination
BASE = "var a b, lags(1/2)"

CASES = [
    (BASE, {}, {}),
    (BASE + " exog(x)", {"exog": ["x"]}, {}),
    (BASE + " noconstant", {"trend": "n"}, {}),
    (BASE + " small", {}, {"small": True}),
    (BASE + " dfk", {}, {"dfk": True}),
    (BASE + " level(99)", {}, {"level": 0.99}),
    (BASE + " exog(x) small", {"exog": ["x"]}, {"small": True}),
    (BASE + " exog(x) dfk", {"exog": ["x"]}, {"dfk": True}),
    (BASE + " exog(x) dfk level(99)", {"exog": ["x"]}, {"dfk": True, "level": 0.99}),
    (BASE + " exog(x x2)", {"exog": ["x", "x2"]}, {}),
]


@pytest.mark.parametrize("command,init,summary", CASES)
def test_every_option_combination_matches(frame, command, init, summary) -> None:
    ours = _lines(str(_fit(frame, **init).summary(**summary)))
    assert ours == _stata(frame, command)


# ---------------------------------------------------------------- the arithmetic, without Stata
def test_dfk_scales_only_the_standard_errors(frame) -> None:
    """MEASURED: ll and the criteria are identical; the SEs grow by sqrt(N/(N-k))."""

    plain = _fit(frame)
    adjusted = _fit(frame)
    plain_text = _lines(str(plain.summary()))[4:7]
    adjusted_text = _lines(str(adjusted.summary(dfk=True)))[4:7]
    assert plain_text == adjusted_text  # log likelihood, FPE, Det and the criteria

    assert plain.ll == pytest.approx(adjusted.ll)


def test_small_divides_by_the_non_constant_coefficients(frame) -> None:
    """With exog(x) the divisor is 5, without it 4."""

    plain = _fit(frame)
    with_exog = _fit(frame, exog=["x"])

    chi2_plain = float(plain.e["chi2_1"])
    chi2_exog = float(with_exog.e["chi2_1"])

    # Pick the equation row by its LABEL: line 8 is equation b, so indexing by position read the
    # wrong equation and compared a 12.8 against a 9.7.
    small_plain = next(
        row for row in _lines(str(plain.summary(small=True))) if row.startswith("a ")
    )
    small_exog = next(
        row for row in _lines(str(with_exog.summary(small=True))) if row.startswith("a ")
    )

    # Compare numerically rather than by formatted text: the cell is printed with Stata's
    # significant-figure rule, so a %.6g rendering of the same value does not string-match it.
    field_plain = float(small_plain.split()[4])
    field_exog = float(small_exog.split()[4])
    assert field_plain == pytest.approx(chi2_plain / 4, rel=1e-6)
    assert field_exog == pytest.approx(chi2_exog / 5, rel=1e-6)


def test_noconstant_uses_the_uncentred_r_squared(frame) -> None:
    """MEASURED: .2401/.1972, where the centred formula gives .2146/.1690."""

    fitted = _fit(frame, trend="n")
    assert float(fitted.e["r2_1"]) == pytest.approx(0.2401, abs=5e-5)
    assert float(fitted.e["r2_2"]) == pytest.approx(0.1972, abs=5e-5)


def test_the_exogenous_columns_count_as_parameters(frame) -> None:
    """This drives ml_factor, e(Sigma), e(detsig) and every Wald statistic."""

    plain = _fit(frame)
    with_exog = _fit(frame, exog=["x"])

    assert plain.k_per_equation == 5
    assert with_exog.k_per_equation == 6


def test_exog_shifts_the_parameter_layout(frame) -> None:
    """The lag block starts after the trend AND the exogenous columns."""

    fitted = _fit(frame, exog=["x"])
    names = list(fitted.stata_param_names)[:6]
    assert names == ["L1.a", "L2.a", "L1.b", "L2.b", "x", "_cons"]
    values = np.asarray(fitted.stata_params, dtype=float).ravel()[:6]
    # First equation's coefficients, measured: L1.a=.5161514 L2.a=-.1081222 ...
    assert values[0] == pytest.approx(0.5161514, abs=1e-6)
    assert values[4] == pytest.approx(0.0761699, abs=1e-6)


def test_the_exogenous_row_has_no_group_heading(frame) -> None:
    """Stata prints `x | ...` directly; a bare `x |` heading above it would be an extra line."""

    rows = _lines(str(_fit(frame, exog=["x"]).summary()))
    headings = [row for row in rows if row.rstrip().endswith("|") and "x" in row and len(row) < 20]
    assert headings == []


def test_noconstant_prints_no_constant_row(frame) -> None:
    rows = _lines(str(_fit(frame, trend="n").summary()))
    assert not any("_cons" in row for row in rows)


def test_dfk_adds_the_note(frame) -> None:
    rows = _lines(str(_fit(frame).summary(dfk=True)))
    assert rows[-2:] == [
        "Note: Small-sample degrees-of-freedom adjustment applied when estimating",
        "      covariance matrix of residuals.",
    ]


def test_rejected_options_are_not_parameters() -> None:
    """Stata rejects these with r(198), so the API must not pretend to accept them."""

    import inspect

    parameters = set(inspect.signature(ws.VAR.__init__).parameters) | set(
        inspect.signature(ws.VAR.fit).parameters
    )
    for absent in ("nobalc", "noback", "noprint", "nofootnote", "noheader", "novce", "mle", "tmin", "tmax"):
        assert absent not in parameters
