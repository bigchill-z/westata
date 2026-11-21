"""`irf_graph` panels share one y axis, because Stata's do.

THE BUG THIS PINS

Each panel had its own y range, so a pair whose band was 50x larger set the scale for itself alone
and the small ones looked flat. Four panels on four different scales cannot be compared by eye, and
comparing magnitudes is the whole point of the `oirf` grid.

THE EVIDENCE that Stata shares the range: exporting its own `irf graph oirf` for
`var dinf unem, lags(1/2)` shows all four subgraphs reading -.01 to .02. No option is involved --
`ycommon`, `sameyaxis` and `individual` are all rejected with r(198) -- so it is the default
behaviour, not a choice.

`same_y` therefore defaults to True; `same_y=False` still gives independent scales for anyone who
wants them.
"""

import pathlib
import warnings

warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.postestimation import irf_graph

DATA = r"D:\DeskBox\文件夹\计量经济学\python实现\data\macro_swatson.dta"


@pytest.fixture(autouse=True)
def clean_registry():
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def fitted():
    if not pathlib.Path(DATA).exists():
        pytest.skip("macro_swatson.dta is not available")
    frame = pd.read_stata(DATA).dropna(subset=["dinf", "unem"]).reset_index(drop=True)
    return ws.VAR(
        frame[["dinf", "unem"]].to_numpy(), lags=2, names=["dinf", "unem"]
    ).fit()


def _limits(figure):
    return [tuple(round(value, 10) for value in axis.get_ylim()) for axis in figure.axes]


def test_the_panels_share_one_y_range_by_default(fitted) -> None:
    figure = irf_graph(fitted, statistic="oirf", periods=8)
    try:
        limits = _limits(figure)
        assert len(limits) == 4
        assert len(set(limits)) == 1, limits
    finally:
        plt.close(figure)


def test_same_y_false_still_allows_independent_scales(fitted) -> None:
    figure = irf_graph(fitted, statistic="oirf", periods=8, same_y=False)
    try:
        limits = _limits(figure)
        assert len(set(limits)) > 1, limits
    finally:
        plt.close(figure)


def test_the_shared_range_covers_every_panel(fitted) -> None:
    """The union must actually contain each panel's data, not merely be equal everywhere."""

    independent = irf_graph(fitted, statistic="oirf", periods=8, same_y=False)
    separate = {
        axis.get_title(): axis.get_ylim() for axis in independent.axes
    }
    plt.close(independent)

    shared = irf_graph(fitted, statistic="oirf", periods=8)
    try:
        low, high = shared.axes[0].get_ylim()
        for title, (panel_low, panel_high) in separate.items():
            assert low <= panel_low + 1e-12, (title, low, panel_low)
            assert high >= panel_high - 1e-12, (title, high, panel_high)
    finally:
        plt.close(shared)


def test_the_irf_statistic_shares_the_range_too(fitted) -> None:
    """The rule is a property of the grid, not of one statistic."""

    for statistic in ("irf", "oirf", "cirf"):
        figure = irf_graph(fitted, statistic=statistic, periods=6)
        try:
            assert len(set(_limits(figure))) == 1, statistic
        finally:
            plt.close(figure)


def test_the_shared_range_is_not_degenerate(fitted) -> None:
    """A zero-height axis would mean the union went wrong."""

    figure = irf_graph(fitted, statistic="oirf", periods=8)
    try:
        low, high = figure.axes[0].get_ylim()
        assert high - low > 0.01
    finally:
        plt.close(figure)
