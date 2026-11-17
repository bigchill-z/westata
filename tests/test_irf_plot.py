"""`ws.irf_table(res).plot()` and the multi-lag IRF it rests on.

Two things were wrong before this and are pinned here:

* the companion matrix only ever held ONE lag block, so a VAR(2) produced `.594687` at step 2 where
  Stata reports `.145969` -- the recursion is `s[h] = A1 s[h-1] + A2 s[h-2] + ...`;
* `lags=p` alone is NOT an ordinary VAR(p). Stata treats it as a single block at distance p, so the
  response is zero on every step that is not a multiple of p. `lags(1 2)` -- which is also the bare
  command's default -- is the ordinary VAR(2).

`lags` therefore stores the argument verbatim (`lag_orders`) while `lags` keeps the block count.
"""

import pathlib
import warnings

warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")  # headless: no display in CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\gdp_china.dta"

pytestmark = pytest.mark.skipif(
    not pathlib.Path(DATA).exists(), reason="gdp_china.dta is not available"
)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    frame = pd.read_stata(DATA)
    frame["lny"] = np.log(frame["y"])
    frame["dlny"] = frame["lny"].diff()
    return frame


@pytest.fixture(scope="module")
def fitted(panel):
    return ws.varbasic(panel, "dlny", time="year", subset=panel["year"] < 2013)


# ---------------------------------------------------------------- the IRF numbers
def test_the_default_fits_lag_one_and_two(fitted) -> None:
    """A bare `varbasic` keeps L1 and L2, which is Stata's default `lags(1 2)`."""

    assert fitted.regressor_names == ("L1.", "L2.", "_cons")
    assert fitted.nobs == 32


def test_the_var_two_recursion_is_ar2(fitted) -> None:
    """Measured: s[h] = A1 s[h-1] + A2 s[h-2] gives 1, .771159, .145969, -.233467, ..."""

    values = ws.irf_table(fitted).responses[0].values
    expected = [1.0, 0.771159, 0.145969, -0.233467, -0.245539, -0.084589, 0.044946, 0.072617, 0.035831]
    assert values == pytest.approx(expected, abs=1e-5)


def test_a_single_lag_is_a_recurrence_at_that_distance(panel) -> None:
    """`lags(2)` alone keeps one block, so every odd step is exactly zero."""

    fitted = ws.varbasic(panel, "dlny", lags=2, time="year")
    values = ws.irf_table(fitted).responses[0].values
    assert values[0] == pytest.approx(1.0)
    assert values[1] == 0.0 and values[3] == 0.0 and values[5] == 0.0
    assert values[2] != 0.0


def test_consecutive_lags_are_not_zeroed(panel) -> None:
    """`lags(1, 2)` is an ordinary VAR(2): no step is forced to zero."""

    fitted = ws.varbasic(panel, "dlny", lags=(1, 2), time="year")
    values = ws.irf_table(fitted).responses[0].values
    assert all(value != 0.0 for value in values[1:])


def test_lag_orders_records_the_argument(panel) -> None:
    """The block count alone cannot tell `lags(2)` from a VAR(2), so the argument is kept."""

    single = ws.varbasic(panel, "dlny", lags=2, time="year")
    consecutive = ws.varbasic(panel, "dlny", lags=(1, 2), time="year")
    assert single.lag_orders == 2
    assert consecutive.lag_orders == (1, 2)
    # lags keeps the BLOCK COUNT: one block at distance 2 versus two consecutive blocks.
    assert single.lags == 1
    assert consecutive.lags == 2


# ---------------------------------------------------------------- irf_table options
def test_all_returns_every_pair(panel) -> None:
    """One variable gives one pair; the option still works."""

    fitted = ws.varbasic(panel, "dlny", time="year")
    assert len(ws.irf_table(fitted, all=True).responses) == 1


def test_steps_option_is_honoured(fitted) -> None:
    for steps in (0, 4, 12):
        table = ws.irf_table(fitted, steps=steps)
        assert len(table.responses[0].steps) == steps + 1


def test_an_unknown_name_is_refused(fitted) -> None:
    with pytest.raises(ws.ConsistencyError, match="not in the VAR"):
        ws.irf_table(fitted, impulse="nope")


# ---------------------------------------------------------------- plot
def test_plot_returns_a_figure(fitted) -> None:
    table = ws.irf_table(fitted)
    figure = table.plot()
    try:
        assert isinstance(figure, plt.Figure)
        assert len(figure.axes) >= 1
    finally:
        plt.close(figure)


def test_plot_draws_one_panel_per_pair(fitted) -> None:
    table = ws.irf_table(fitted, all=True)
    figure = table.plot()
    try:
        visible = [axis for axis in figure.axes if axis.get_visible()]
        assert len(visible) == len(table.responses)
    finally:
        plt.close(figure)


def test_plot_has_a_line_and_a_band(fitted) -> None:
    figure = ws.irf_table(fitted).plot()
    try:
        axis = figure.axes[0]
        # One line for the response path, one for the zero reference.
        assert len(axis.lines) == 2
        assert len(axis.collections) == 1
    finally:
        plt.close(figure)


def test_plot_can_omit_the_band(fitted) -> None:
    figure = ws.irf_table(fitted).plot(show_ci=False)
    try:
        assert len(figure.axes[0].collections) == 0
    finally:
        plt.close(figure)


def test_plot_accepts_an_existing_axis(fitted) -> None:
    _figure, axis = plt.subplots()
    try:
        returned = ws.irf_table(fitted).plot(ax=axis)
        assert returned is axis.figure
    finally:
        plt.close(_figure)


def test_plot_titles_name_the_pair(fitted) -> None:
    figure = ws.irf_table(fitted).plot()
    try:
        assert "dlny" in figure.axes[0].get_title()
    finally:
        plt.close(figure)


def test_plot_can_save_to_a_file(fitted, tmp_path) -> None:
    """The whole point of a plot: something a user can keep."""

    table = ws.irf_table(fitted)
    figure = table.plot()
    target = tmp_path / "irf.png"
    try:
        figure.savefig(target, dpi=80)
        assert target.exists() and target.stat().st_size > 1000
    finally:
        plt.close(figure)


# ---------------------------------------------------------------- dependency handling
def test_plot_raises_a_clear_error_without_matplotlib(monkeypatch, fitted) -> None:
    """`matplotlib` is optional; the message must say how to get it."""

    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("no matplotlib")
        return real_import(name, *args, **kwargs)

    table = ws.irf_table(fitted)
    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match="westata\\[plot\\]"):
        table.plot()
