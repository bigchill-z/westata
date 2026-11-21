"""Narrowing to one combination: Stata's `i(impulse) r(response)`.

The equivalence under test, on `var dinf unem, lags(1/2)`:

    irf graph oirf, i(unem) r(dinf)   ==   i(2) r(1)

because Stata's `i()`/`r()` are **1-BASED**. The previous build treated the integer as 0-based, so
`impulse=2` raised "index 2 is out of range for ['dinf', 'unem']" on a two-variable VAR and every
other value silently selected the wrong series.
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

# Stata's oirf for impulse=unem, response=dinf.
STATA_UNEM_DINF = [0.0, -0.005521, -0.001434, 0.000372]


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


def _curve(figure):
    """The plotted response path (the series line, not the zero rule)."""

    axis = figure.axes[0]
    candidates = [line for line in axis.lines if len(line.get_ydata()) > 2]
    assert candidates
    return [float(value) for value in candidates[0].get_ydata()]


def test_narrowing_to_one_pair_gives_one_panel(fitted) -> None:
    figure = irf_graph(
        fitted, statistic="oirf", periods=3, impulse="unem", response="dinf"
    )
    try:
        assert len([axis for axis in figure.axes if axis.get_visible()]) == 1
    finally:
        plt.close(figure)


def test_the_panel_shows_the_requested_pair(fitted) -> None:
    figure = irf_graph(
        fitted, statistic="oirf", periods=3, impulse="unem", response="dinf"
    )
    try:
        assert "unem" in figure.axes[0].get_title()
        assert "dinf" in figure.axes[0].get_title()
        assert _curve(figure) == pytest.approx(STATA_UNEM_DINF, abs=5e-6, rel=1e-4)
    finally:
        plt.close(figure)


def test_positions_are_one_based_and_match_the_names(fitted) -> None:
    """`i(2) r(1)` must equal `i(unem) r(dinf)`."""

    by_name = irf_graph(
        fitted, statistic="oirf", periods=3, impulse="unem", response="dinf"
    )
    by_position = irf_graph(
        fitted, statistic="oirf", periods=3, impulse=2, response=1
    )
    try:
        assert by_name.axes[0].get_title() == by_position.axes[0].get_title()
        assert _curve(by_name) == pytest.approx(_curve(by_position), rel=1e-12)
    finally:
        plt.close(by_name)
        plt.close(by_position)


def test_position_one_is_the_first_variable(fitted) -> None:
    figure = irf_graph(
        fitted, statistic="oirf", periods=3, impulse=1, response=2
    )
    try:
        title = figure.axes[0].get_title()
        assert "dinf" in title and "unem" in title
        assert title.index("dinf") < title.index("unem")
    finally:
        plt.close(figure)


def test_an_out_of_range_position_is_refused(fitted) -> None:
    with pytest.raises(ws.ConsistencyError, match="1-based"):
        irf_graph(fitted, statistic="oirf", periods=3, impulse=3, response=1)


def test_zero_is_rejected_because_positions_start_at_one(fitted) -> None:
    with pytest.raises(ws.ConsistencyError):
        irf_graph(fitted, statistic="oirf", periods=3, impulse=0, response=1)


def test_one_impulse_with_several_responses_gives_several_panels(fitted) -> None:
    figure = irf_graph(
        fitted, statistic="oirf", periods=3, impulse="unem", response=["dinf", "unem"]
    )
    try:
        visible = [axis for axis in figure.axes if axis.get_visible()]
        assert len(visible) == 2
    finally:
        plt.close(figure)


def test_a_list_of_one_behaves_like_a_scalar(fitted) -> None:
    scalar = irf_graph(
        fitted, statistic="oirf", periods=3, impulse="unem", response="dinf"
    )
    listed = irf_graph(
        fitted, statistic="oirf", periods=3, impulse=["unem"], response=["dinf"]
    )
    try:
        assert _curve(scalar) == pytest.approx(_curve(listed), rel=1e-12)
    finally:
        plt.close(scalar)
        plt.close(listed)


def test_an_unknown_name_names_the_available_ones(fitted) -> None:
    with pytest.raises(ws.ConsistencyError, match="dinf"):
        irf_graph(fitted, statistic="oirf", periods=3, impulse="nope", response="dinf")
