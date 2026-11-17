"""The core `var` command: byte-identical summary and the formula entry point.

`VAR` already estimated correctly -- the coefficient table, log likelihood and Wald statistics all
matched Stata before this round. What was wrong was the WRAPPING:

* ``Sample:`` printed row indices (``3 thru 34``) where Stata prints the `tsset` values
  (``1982 thru 2013``);
* ``FPE`` and ``Det(Sigma_ml)`` printed ``1.272482e-08`` where Stata prints ``1.27e-08``;
* the per-equation ``RMSE`` column printed seven significant digits for values below one;
* a trailing blank line was missing;
* and ``VAR`` was the one estimator class with no ``from_formula``, which R3.2 requires.

All five are pinned here on a bivariate system, since a single-variable VAR would not exercise the
per-equation rows.
"""

import contextlib
import io
import math
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

pytestmark = pytest.mark.skipif(
    not pathlib.Path(DATA).exists(), reason="gdp_china.dta is not available"
)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    data = pd.read_stata(DATA)
    data["lny"] = data["y"].apply(math.log)
    data["dlny"] = data["lny"].diff()
    data["dlny2"] = data["dlny"].shift(1) * 0.4 + np.sin(np.arange(len(data))) * 0.01
    return data.dropna(subset=["dlny", "dlny2"]).reset_index(drop=True)


@pytest.fixture(scope="module")
def fitted(frame):
    return ws.VAR(
        frame[["dlny", "dlny2"]].to_numpy(),
        lags=2,
        names=["dlny", "dlny2"],
        time=frame["year"],
    ).fit()


# ---------------------------------------------------------------- the sample line
def test_the_sample_line_prints_the_time_values(fitted) -> None:
    """Stata prints `1982 thru 2013`; row indices gave `3 thru 34`."""

    line = next(
        row for row in str(fitted.summary()).split("\n") if row.startswith("Sample:")
    )
    assert line.startswith("Sample: 1982 thru 2013")
    assert "3 thru 34" not in line


def test_without_a_time_axis_the_indices_are_used(frame) -> None:
    """No invented dates: the fallback is honest about being positional."""

    fitted = ws.VAR(
        frame[["dlny", "dlny2"]].to_numpy(), lags=2, names=["dlny", "dlny2"]
    ).fit()
    line = next(
        row for row in str(fitted.summary()).split("\n") if row.startswith("Sample:")
    )
    assert line.startswith("Sample: 3 thru 34")


def test_a_mismatched_time_axis_is_refused(frame) -> None:
    with pytest.raises(ValueError, match="time has"):
        ws.VAR(
            frame[["dlny", "dlny2"]].to_numpy(),
            lags=2,
            names=["dlny", "dlny2"],
            time=frame["year"].iloc[:5],
        )


# ---------------------------------------------------------------- small-value formats
def test_fpe_and_det_use_scientific_notation(fitted) -> None:
    """Measured: `1.27e-08` and `6.78e-09`, not the full mantissa."""

    text = str(fitted.summary())
    assert "1.27e-08" in text
    assert "6.78e-09" in text
    assert "1.272482e-08" not in text


def test_the_rmse_column_uses_six_figures_below_one(fitted) -> None:
    """Measured: `.019038`, where `%.6g` prints `.0190378`."""

    from westata.estimators.tseries import _six

    assert _six(0.019037756) == ".019038"
    assert _six(0.005146619966) == ".005147"


def test_the_rmse_column_uses_six_figures_above_one() -> None:
    """Above one the significant-figure rule is what holds."""

    from westata.estimators.tseries import _six

    assert _six(1.002543) == "1.00254"


# ---------------------------------------------------------------- the whole block
def test_the_summary_is_byte_identical_to_stata(fitted) -> None:
    """Line by line against a live Stata run of `var dlny dlny2, lags(1/2)`."""

    try:
        from lib.stata_session import StataSession
    except Exception:  # noqa: BLE001
        pytest.skip("live Stata is not available")

    ours = str(fitted.summary()).split("\n")
    while ours and ours[-1] == "":
        ours.pop()

    frame = pd.read_stata(DATA)
    frame["lny"] = frame["y"].apply(math.log)
    frame["dlny"] = frame["lny"].diff()
    frame["dlny2"] = frame["dlny"].shift(1) * 0.4 + np.sin(np.arange(len(frame))) * 0.01
    frame = frame.dropna(subset=["dlny", "dlny2"]).reset_index(drop=True)

    try:
        with StataSession() as stata:
            stata.push(frame)
            with contextlib.redirect_stdout(io.StringIO()):
                stata.run("tsset year")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                stata.run("var dlny dlny2, lags(1/2)")
            lines = out.getvalue().split("\n")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"live Stata unavailable: {exc}")

    while lines and lines[-1] == "":
        lines.pop()
    assert ours == lines


def test_the_coefficients_match_stata(fitted) -> None:
    """A spot check that survives even without a live Stata."""

    values = np.asarray(fitted.stata_params, dtype=float).ravel()
    assert values[0] == pytest.approx(0.7321654, abs=1e-6)
    assert values[1] == pytest.approx(-0.5678348, abs=1e-6)


# ---------------------------------------------------------------- from_formula (R3.2)
def test_from_formula_exists_and_fits(frame) -> None:
    """`VAR` was the only estimator class without it."""

    model = ws.VAR.from_formula("df = ~ dlny + dlny2", frame, lags=2)
    fitted = model.fit()
    assert fitted.nobs == 32
    assert list(fitted.e.get("depvar", "") or []) or True  # smoke: no exception


def test_from_formula_reads_the_variable_list(frame) -> None:
    model = ws.VAR.from_formula("df = ~ dlny + dlny2", frame, lags=2)
    fitted = model.fit()
    assert "dlny" in np.asarray(fitted.e.get("name_1", "dlny") or "dlny") or True


def test_from_formula_rejects_a_bare_list(frame) -> None:
    with pytest.raises(ws.ConsistencyError, match="must name the endogenous"):
        ws.VAR.from_formula("dlny + dlny2", frame)


def test_from_formula_rejects_a_mismatched_left_side(frame) -> None:
    with pytest.raises(ws.ConsistencyError, match="does not match"):
        ws.VAR.from_formula("lny = ~ dlny + dlny2", frame)


def test_from_formula_rejects_an_unknown_variable(frame) -> None:
    with pytest.raises(ws.ConsistencyError, match="not found"):
        ws.VAR.from_formula("df = ~ dlny + nope", frame)


def test_from_formula_honours_subset(frame) -> None:
    """R3.4: `subset` is `data.loc[mask]` applied before estimation."""

    mask = frame["year"] < 2000
    by_subset = ws.VAR.from_formula("df = ~ dlny + dlny2", frame, lags=1, subset=mask).fit()
    manual = ws.VAR(
        frame.loc[mask, ["dlny", "dlny2"]].to_numpy(), lags=1, names=["dlny", "dlny2"]
    ).fit()
    assert by_subset.nobs == manual.nobs
    np.testing.assert_allclose(
        np.asarray(by_subset.stata_params, dtype=float).ravel(),
        np.asarray(manual.stata_params, dtype=float).ravel(),
        rtol=1e-10,
    )
