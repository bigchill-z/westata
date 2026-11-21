"""Several named IRF result sets: Stata's IRF file, and why it can show two identical titles.

THE BEHAVIOUR THIS PINS

`irf graph` draws one panel per **(irfname, impulse, response)**. `i()`/`r()` narrow the VARIABLES
but not the NAME, so a session holding two `irf create` names shows two panels whose titles carry the
same impulse/response -- the screenshot's puzzle. The name is narrowed with `irf(irfnames)`, NOT
`irfname()` (Stata rejects that with r(198)).

Measured on macro_swatson: `iu` (lags 1/2) gives -.005521 at step 1 where `iu3` (lags 1/3) gives
-.006555, so the two panels are genuinely different results that share a label.
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

STATA_IU = [0.0, -0.005521, -0.001434]
STATA_IU3 = [0.0, -0.006555, 0.000029]


@pytest.fixture(autouse=True)
def clean_registry():
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def values():
    if not pathlib.Path(DATA).exists():
        pytest.skip("macro_swatson.dta is not available")
    frame = pd.read_stata(DATA).dropna(subset=["dinf", "unem"]).reset_index(drop=True)
    return frame[["dinf", "unem"]].to_numpy()


@pytest.fixture(scope="module")
def store(values):
    two = ws.VAR(values, lags=2, names=["dinf", "unem"]).fit()
    three = ws.VAR(values, lags=3, names=["dinf", "unem"]).fit()
    return ws.IrfStore().add(two, "iu").add(three, "iu3")


def _titles(figure) -> list[str]:
    return [axis.get_title() for axis in figure.axes if axis.get_visible()]


# ---------------------------------------------------------------- the store itself
def test_the_store_lists_names_in_order(store) -> None:
    assert store.names == ("iu", "iu3")
    assert len(store) == 2


def test_a_repeated_name_replaces_rather_than_appends(values, store) -> None:
    """`irf create ..., replace` keeps one entry per name."""

    clone = ws.IrfStore().add(ws.VAR(values, lags=2, names=["dinf", "unem"]).fit(), "iu")
    clone.add(ws.VAR(values, lags=3, names=["dinf", "unem"]).fit(), "iu")
    assert clone.names == ("iu",)


def test_an_unknown_name_reports_what_exists(store) -> None:
    with pytest.raises(ws.ConsistencyError, match="iu"):
        store.get("nope")


def test_remove_and_clear(store) -> None:
    clone = ws.IrfStore().add(store.get("iu"), "a").add(store.get("iu3"), "b")
    clone.remove("a")
    assert clone.names == ("b",)
    clone.clear()
    assert clone.names == ()
    assert len(clone) == 0


# ---------------------------------------------------------------- the panel rule
def test_two_names_give_two_panels_for_one_pair(store) -> None:
    """This is exactly the screenshot: same impulse/response, two panels."""

    figure = irf_graph(
        store, statistic="oirf", periods=8, impulse="unem", response="dinf"
    )
    try:
        titles = _titles(figure)
        assert len(titles) == 2
        # Both titles name the same pair, distinguished only by the irfname prefix.
        assert all("unem" in title and "dinf" in title for title in titles)
        assert titles[0].startswith("iu,")
        assert titles[1].startswith("iu3,")
    finally:
        plt.close(figure)


def test_the_two_panels_carry_different_numbers(store) -> None:
    """Same label, different results -- which is why drawing both is not redundant."""

    figure = irf_graph(
        store, statistic="oirf", periods=8, impulse="unem", response="dinf"
    )
    try:
        curves = [
            [float(value) for value in axis.lines[0].get_ydata()[:3]] for axis in figure.axes
        ]
        assert curves[0] == pytest.approx(STATA_IU, abs=5e-6, rel=1e-4)
        assert curves[1] == pytest.approx(STATA_IU3, abs=5e-6, rel=1e-4)
        assert curves[0] != pytest.approx(curves[1], rel=1e-3)
    finally:
        plt.close(figure)


def test_irf_selects_one_name(store) -> None:
    figure = irf_graph(
        store, statistic="oirf", periods=8, impulse="unem", response="dinf", irf="iu"
    )
    try:
        titles = _titles(figure)
        assert len(titles) == 1
        assert titles[0].startswith("iu,")
        curve = [float(value) for value in figure.axes[0].lines[0].get_ydata()[:3]]
        assert curve == pytest.approx(STATA_IU, abs=5e-6, rel=1e-4)
    finally:
        plt.close(figure)


def test_irf_accepts_a_sequence(store) -> None:
    figure = irf_graph(
        store,
        statistic="oirf",
        periods=8,
        impulse="unem",
        response="dinf",
        irf=["iu", "iu3"],
    )
    try:
        assert len(_titles(figure)) == 2
    finally:
        plt.close(figure)


def test_a_single_result_still_gives_one_panel(values) -> None:
    """The store is optional: a bare VAR result must behave as before."""

    alone = ws.VAR(values, lags=2, names=["dinf", "unem"]).fit()
    figure = irf_graph(alone, statistic="oirf", periods=6)
    try:
        assert len(_titles(figure)) == 4  # both impulses x both responses
        # No name prefix on the title when there is no name.
        assert all("iu" not in title for title in _titles(figure))
    finally:
        plt.close(figure)


def test_irf_rejects_a_bare_result(values) -> None:
    """`irf=` is meaningless with one unnamed result, so it says so instead of ignoring it."""

    alone = ws.VAR(values, lags=2, names=["dinf", "unem"]).fit()
    with pytest.raises(ws.ConsistencyError, match="IrfStore"):
        irf_graph(alone, statistic="oirf", periods=4, irf="iu")


def test_names_and_pair_multiply(store) -> None:
    """2 names x 1 impulse x 2 responses = 4 panels."""

    figure = irf_graph(
        store, statistic="oirf", periods=6, impulse="unem", response=["dinf", "unem"]
    )
    try:
        assert len(_titles(figure)) == 4
    finally:
        plt.close(figure)


def test_the_store_plot_helper(store) -> None:
    figure = store.plot(statistic="oirf", periods=6, impulse="unem", response="dinf")
    try:
        assert len(_titles(figure)) == 2
    finally:
        plt.close(figure)
