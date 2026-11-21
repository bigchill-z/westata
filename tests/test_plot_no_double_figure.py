"""Every `plot()` must leave pyplot's registry EMPTY.

THE BUG THIS PINS

In a Jupyter cell -- including VS Code's Jupyter extension, which uses the same inline backend --
one call to `ws.ac_table(series).plot()` showed TWO images. Two independent renderers fired:

  1. the inline backend draws every figure still REGISTERED in pyplot when the cell ends;
  2. the returned ``Figure`` is displayed too, because it is the cell's expression value.

`plt.subplots` registers the figure, so both paths ran on the same object.

THE FIX AND WHY THIS SHAPE

`plt.subplots` gives a real backend canvas; a bare ``matplotlib.figure.Figure()`` does not (it carries
``FigureCanvasBase``, whose ``print_figure`` is not a renderer), which could leave an interactive
session unable to draw. So the figure is built normally and then only its pyplot MANAGER is
destroyed: the canvas, `savefig`, `show` and inline display all keep working, while the registry stays
empty. `westata.plotting.release` does that, and `make_figure`/`make_axes` call it.

These tests assert the registry is empty after EVERY entry point, which is the observable symptom the
user saw, rather than the private implementation.
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

DATA = r"docs\data\auto.csv"


@pytest.fixture(autouse=True)
def clean_registry():
    """Each test starts and ends with no open figure, so a leak cannot hide."""

    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def series() -> np.ndarray:
    generator = np.random.default_rng(1)
    values = np.zeros(200)
    for t in range(1, 200):
        values[t] = 0.6 * values[t - 1] + generator.normal()
    return values


@pytest.fixture(scope="module")
def regress_fit():
    if not pathlib.Path(DATA).exists():
        pytest.skip("auto.csv is not available")
    frame = pd.read_csv(DATA)
    return ws.Regress.from_formula("price ~ mpg + weight", frame).fit()


@pytest.fixture(scope="module")
def var_fit():
    values = np.random.default_rng(3).normal(size=(120, 2))
    return ws.VAR(values, lags=2, names=["a", "b"]).fit()


# ---------------------------------------------------------------- the regression plots
def test_rvf_plot_leaves_no_figure(regress_fit) -> None:
    figure = ws.rvf_table(regress_fit).plot()
    assert plt.get_fignums() == []
    assert figure.axes  # still a usable figure


def test_rvp_plot_leaves_no_figure(regress_fit) -> None:
    assert ws.rvp_table(regress_fit, "mpg").plot() is not None
    assert plt.get_fignums() == []


def test_av_plot_leaves_no_figure(regress_fit) -> None:
    assert ws.av_table(regress_fit, "mpg").plot() is not None
    assert plt.get_fignums() == []


def test_avplots_leaves_no_figure(regress_fit) -> None:
    figure = ws.avplots_table(regress_fit).plot()
    assert plt.get_fignums() == []
    assert len(figure.axes) == 2


def test_cpr_and_acpr_leave_no_figure(regress_fit) -> None:
    assert ws.cpr_table(regress_fit, "mpg").plot() is not None
    assert plt.get_fignums() == []
    assert ws.acpr_table(regress_fit, "mpg").plot() is not None
    assert plt.get_fignums() == []


def test_lvr2_leaves_no_figure(regress_fit) -> None:
    assert ws.lvr2_table(regress_fit).plot() is not None
    assert plt.get_fignums() == []


# ---------------------------------------------------------------- the correlograms
def test_ac_plot_leaves_no_figure(series) -> None:
    figure = ws.ac_table(series, lags=8).plot()
    assert plt.get_fignums() == []
    assert figure.axes


def test_pac_plot_leaves_no_figure(series) -> None:
    assert ws.pac_table(series, lags=8).plot() is not None
    assert plt.get_fignums() == []


# ---------------------------------------------------------------- the VAR plots
def test_varstable_leaves_no_figure(var_fit) -> None:
    assert ws.varstable(var_fit).plot() is not None
    assert plt.get_fignums() == []


def test_varstable_table_leaves_no_figure(var_fit) -> None:
    assert ws.varstable_table(var_fit).plot() is not None
    assert plt.get_fignums() == []


# ---------------------------------------------------------------- the returned figure still works
def test_a_deregistered_figure_can_still_be_saved(regress_fit, tmp_path) -> None:
    """Removing the manager must NOT break savefig -- that is the reason for this approach."""

    figure = ws.rvf_table(regress_fit).plot()
    target = tmp_path / "rvf.png"
    figure.savefig(target, dpi=70)
    assert target.exists() and target.stat().st_size > 1000


def test_a_deregistered_figure_keeps_a_real_canvas(regress_fit) -> None:
    """A bare Figure() would carry FigureCanvasBase, which cannot render; this must not."""

    figure = ws.rvf_table(regress_fit).plot()
    assert type(figure.canvas).__name__ != "FigureCanvasBase"


def test_the_axis_still_accepts_drawing_after_release(regress_fit) -> None:
    figure = ws.rvf_table(regress_fit).plot()
    axis = figure.axes[0]
    axis.axhline(123.0, color="red")
    assert axis.lines[-1].get_ydata()[0] == 123.0


# ---------------------------------------------------------------- the helper itself
def test_make_figure_deregisters() -> None:
    from westata.plotting import make_figure

    figure, axis = make_figure(figsize=(4, 3))
    assert plt.get_fignums() == []
    axis.plot([1, 2, 3])
    assert type(figure.canvas).__name__ != "FigureCanvasBase"


def test_make_axes_deregisters_and_flattens() -> None:
    from westata.plotting import make_axes

    figure, flat = make_axes(2, 2, figsize=(6, 6))
    assert plt.get_fignums() == []
    assert len(flat) == 4
    assert len(figure.axes) == 4


def test_release_is_idempotent(regress_fit) -> None:
    from westata.plotting import release

    figure = ws.rvf_table(regress_fit).plot()
    assert release(figure) is figure
    assert plt.get_fignums() == []


def test_an_axis_caller_can_still_own_the_pyplot_figure() -> None:
    """When the caller passes `ax=`, the figure belongs to them and must NOT be deregistered."""

    _figure, axis = plt.subplots()
    try:
        ws.ac_table(np.random.default_rng(5).normal(size=100), lags=5).plot(ax=axis)
        assert plt.get_fignums() == [_figure.number]
    finally:
        plt.close(_figure)
