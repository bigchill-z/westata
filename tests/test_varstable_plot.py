"""`VarStableResult.plot()`: the figure Stata's `varstable, graph` draws.

Stata's `, graph` calls `_vec_grcroots` (`varstable_w2.ado` -> `_vec_grcroots.ado`), which draws the
eigenvalues of the companion matrix in the COMPLEX PLANE -- real part on x, imaginary part on y --
with the unit circle behind them, titled "Roots of the companion matrix". It is not a time-series
plot.

Two things here are measured rather than assumed:

* THE POINTS ARE THE COMPANION MATRIX'S SPECTRUM. A bivariate VAR(2) contributes FOUR points
  (`lags * equations`), not two. `r(Re)`/`r(Im)` from the ado confirm it, and a plot that drew the
  per-equation coefficient eigenvalues would silently show half the spectrum.
* NEAR-COINCIDENT ROOTS ARE NORMAL, so label placement has to survive them. On the reference fit the
  two real roots are `.82182` and `.7970223` -- 0.025 apart, both at angle 0 -- and the first
  implementation printed them as one smeared string. The guards below pin the collision handling
  instead of relying on the eye.

`matplotlib` stays optional (R1.1): it is imported inside the method, and the error names the extra.
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")  # headless: no display in CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.postestimation.varstable import VarStableResults

DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"

pytestmark = pytest.mark.skipif(
    not __import__("pathlib").Path(DATA).exists(),
    reason="macro_swatson.dta is not available on this machine",
)


@pytest.fixture(scope="module")
def stable_result():
    frame = pd.read_stata(DATA)
    values = frame[["dinf", "unem"]].dropna()
    return ws.var_stable(ws.VAR(values, lags=2).fit())


@pytest.fixture(scope="module")
def unstable_result() -> VarStableResults:
    """One root outside the circle -- the case a reader most needs to see."""

    return VarStableResults(
        eigenvalues=(0.35 + 0.0j, 1.08 + 0.0j, -0.2 + 0.55j, -0.2 - 0.55j),
        modulus=(0.35, 1.08, 0.58523, 0.58523),
    )


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------- the figure exists and is right
def test_plot_returns_a_figure(stable_result) -> None:
    figure = stable_result.plot()
    assert isinstance(figure, matplotlib.figure.Figure)
    assert len(figure.axes) == 1


def test_every_eigenvalue_becomes_one_point(stable_result) -> None:
    """A bivariate VAR(2) has FOUR roots; drawing two would hide half the spectrum."""

    figure = stable_result.plot()
    offsets = figure.axes[0].collections[0].get_offsets()
    assert len(offsets) == len(stable_result.eigenvalues) == 4


def test_the_points_are_the_complex_values(stable_result) -> None:
    """x is the real part, y the imaginary part -- not the modulus."""

    figure = stable_result.plot()
    offsets = np.asarray(figure.axes[0].collections[0].get_offsets(), dtype=float)
    expected = np.array([(value.real, value.imag) for value in stable_result.eigenvalues])
    assert np.allclose(offsets, expected)


def test_the_unit_circle_is_drawn(stable_result) -> None:
    figure = stable_result.plot()
    circle = figure.axes[0].lines[0]
    radius = np.hypot(circle.get_xdata(), circle.get_ydata())
    assert np.allclose(radius, 1.0)


def test_the_circle_can_be_turned_off(stable_result) -> None:
    figure = stable_result.plot(unit_circle=False)
    # Only the two zero-crossing axes remain.
    assert len(figure.axes[0].lines) == 2


def test_the_axes_are_square(stable_result) -> None:
    """An equal aspect is what keeps the unit circle a circle rather than an ellipse."""

    figure = stable_result.plot()
    assert figure.axes[0].get_aspect() == 1.0
    x_low, x_high = figure.axes[0].get_xlim()
    y_low, y_high = figure.axes[0].get_ylim()
    assert (y_high - y_low) == pytest.approx(x_high - x_low)


def test_the_axis_labels_and_title_match_statas_plot(stable_result) -> None:
    axis = stable_result.plot().axes[0]
    assert axis.get_xlabel() == "Real"
    assert axis.get_ylabel() == "Imaginary"
    assert axis.get_title() == "Roots of the companion matrix"


def test_the_title_can_be_overridden(stable_result) -> None:
    assert stable_result.plot(title="custom").axes[0].get_title() == "custom"


# ---------------------------------------------------------------- the unstable case stays readable
def test_an_unstable_root_is_inside_the_visible_axes(unstable_result) -> None:
    """A root at 1.08 must not be clipped -- it is the finding the figure exists to show."""

    assert unstable_result.stable is False
    axis = unstable_result.plot(labels=True).axes[0]
    x_low, x_high = axis.get_xlim()
    y_low, y_high = axis.get_ylim()
    assert x_low < 1.08 < x_high
    assert y_low < 0.0 < y_high


def test_one_point_per_eigenvalue_when_unstable(unstable_result) -> None:
    figure = unstable_result.plot()
    assert len(figure.axes[0].collections[0].get_offsets()) == 4


# ---------------------------------------------------------------- labels
def test_labels_are_off_by_default(stable_result) -> None:
    figure = stable_result.plot()
    assert len(figure.axes[0].texts) == 0


def test_labels_write_the_modulus(stable_result) -> None:
    figure = stable_result.plot(labels=True)
    texts = [item.get_text() for item in figure.axes[0].texts]
    assert len(texts) == 4
    for text in texts:
        value = float(text)
        assert any(abs(value - modulus) < 1e-3 for modulus in stable_result.modulus), (
            text,
            stable_result.modulus,
        )


def test_distance_labels_are_one_minus_the_modulus(stable_result) -> None:
    """Stata's `dlabel`: the distance from the unit circle."""

    figure = stable_result.plot(labels="distance")
    texts = sorted(float(item.get_text()) for item in figure.axes[0].texts)
    expected = sorted(1.0 - modulus for modulus in stable_result.modulus)
    assert texts == pytest.approx(expected, abs=1e-3)


def test_a_format_string_controls_the_label(stable_result) -> None:
    figure = stable_result.plot(labels=".2e")
    texts = [item.get_text() for item in figure.axes[0].texts]
    assert all("e" in text for text in texts), texts


def test_coincident_roots_get_separate_labels(stable_result) -> None:
    """The two real roots are 0.025 apart at the SAME angle.

    A purely radial offset placed both labels on the same spot, so they printed on top of each
    other. The vertical stacking is what fixes it, and this pins the separation rather than the
    pixels.
    """

    figure = stable_result.plot(labels=True)
    # `Text.get_position()` is the (x, y) of the ANCHOR in data coordinates: the root itself, not the
    # offset label position. The rendered offset lives in `get_transform()`, so the separation is
    # measured from the annotation transform's offset arguments instead.
    offsets = []
    for item in figure.axes[0].texts:
        transform = item.get_transform()
        # `Annotation` keeps its offset in an `offset points` transform; read the accumulated Bbox
        # of the drawn text, which is the only place the final placement is visible.
        window = item.get_window_extent(renderer=figure.canvas.get_renderer())
        offsets.append((window.x0, window.y0, window.x1, window.y1))
    assert len(offsets) == 4

    # The two labels of the real roots share a root region; their drawn boxes must not coincide.
    boxes = sorted(offsets, key=lambda box: box[0])
    overlaps = 0
    for i in range(len(boxes) - 1):
        left, right = boxes[i], boxes[i + 1]
        if right[0] < left[2] and right[1] < left[3] and left[1] < right[3]:
            overlaps += 1
    assert overlaps == 0, boxes


def test_labels_stay_inside_the_axes(stable_result) -> None:
    """The outermost labels must not be clipped by the axis limits."""

    figure = stable_result.plot(labels=True)
    axis = figure.axes[0]
    figure.canvas.draw()
    axes_box = axis.get_window_extent()
    for item in axis.texts:
        box = item.get_window_extent(renderer=figure.canvas.get_renderer())
        assert axes_box.x0 <= box.x0 and box.x1 <= axes_box.x1, (axes_box, box)


# ---------------------------------------------------------------- plumbing
def test_drawing_into_an_existing_axis_returns_its_figure(stable_result) -> None:
    figure, axis = plt.subplots(figsize=(4, 4))
    assert stable_result.plot(ax=axis) is figure


def test_figsize_is_honoured(stable_result) -> None:
    figure = stable_result.plot(figsize=(3.0, 3.0))
    width, height = figure.get_size_inches()
    assert width == pytest.approx(3.0) and height == pytest.approx(3.0)


def test_scatter_keywords_pass_through(stable_result) -> None:
    figure = stable_result.plot(s=120, color="crimson")
    collection = figure.axes[0].collections[0]
    assert collection.get_sizes()[0] == 120


def test_plot_needs_matplotlib_and_says_how_to_get_it(monkeypatch, stable_result) -> None:
    """R1.1: plotting must not be a forced dependency."""

    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("no matplotlib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match=r"westata\[plot\]"):
        stable_result.plot()


def test_plot_does_not_change_the_result(stable_result) -> None:
    """It is a view over stored numbers, not a re-estimation."""

    before = stable_result.to_frame()
    stable_result.plot()
    after = stable_result.to_frame()
    assert before.equals(after)


def test_a_single_eigenvalue_still_plots() -> None:
    """No divide-by-zero or empty-extent guard: one point is a legitimate figure."""

    single = VarStableResults(eigenvalues=(0.5 + 0.0j,), modulus=(0.5,))
    figure = single.plot()
    assert len(figure.axes[0].collections[0].get_offsets()) == 1
    assert figure.axes[0].get_xlim()[1] > 0.5
