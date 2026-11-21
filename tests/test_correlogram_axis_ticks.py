"""The x axis of a correlogram is an INTEGER lag index.

THE BUG THIS PINS

matplotlib's default locator is built for a continuous axis, so a 20-lag correlogram was labelled
`0.0  2.5  5.0  7.5 ... 22.5` -- half-steps that no lag occupies -- and the axis ran past the last
lag to 22.5. Stata labels whole lags only.

The fix sets `MaxNLocator(integer=True, prune="both")` with the limits applied FIRST, which is what
stops the locator emitting a tick beyond the view. Order matters: setting the locator before the
limits still produced a 21 on a 0.4..20.6 window.

`varstable` is deliberately NOT touched: its x axis is the real part of the eigenvalues, a genuinely
continuous quantity, so fractional ticks are correct there.
"""

import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import westata as ws


@pytest.fixture(autouse=True)
def clean_registry():
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


@pytest.mark.parametrize("lags", [4, 8, 12, 20, 40])
def test_ac_ticks_are_whole_lags(series, lags) -> None:
    figure = ws.ac_table(series, lags=lags).plot()
    try:
        ticks = figure.axes[0].get_xticks()
        assert all(float(tick).is_integer() for tick in ticks), ticks
    finally:
        plt.close(figure)


@pytest.mark.parametrize("lags", [4, 8, 12, 20])
def test_pac_ticks_are_whole_lags(series, lags) -> None:
    figure = ws.pac_table(series, lags=lags).plot()
    try:
        ticks = figure.axes[0].get_xticks()
        assert all(float(tick).is_integer() for tick in ticks), ticks
    finally:
        plt.close(figure)


@pytest.mark.parametrize("lags", [8, 20])
def test_no_tick_falls_outside_the_lag_range(series, lags) -> None:
    """A 0.4..20.6 window must not label 0 or 21 -- the locator has to prune, not merely round."""

    for builder in (ws.ac_table, ws.pac_table):
        figure = builder(series, lags=lags).plot()
        try:
            ticks = [float(tick) for tick in figure.axes[0].get_xticks()]
            assert min(ticks) >= 1.0, ticks
            assert max(ticks) <= lags, ticks
        finally:
            plt.close(figure)


def test_the_limits_hold_the_data(series) -> None:
    """The axis stops near the last lag instead of running on to 22.5."""

    figure = ws.ac_table(series, lags=20).plot()
    try:
        low, high = figure.axes[0].get_xlim()
        assert low >= 0.0
        assert high <= 21.0
    finally:
        plt.close(figure)


def test_a_short_correlogram_is_labelled_every_lag(series) -> None:
    """With few lags every integer fits, so none is skipped."""

    figure = ws.ac_table(series, lags=8).plot()
    try:
        ticks = [int(tick) for tick in figure.axes[0].get_xticks()]
        assert ticks == [1, 2, 3, 4, 5, 6, 7, 8]
    finally:
        plt.close(figure)


def test_labels_match_the_integer_ticks(series) -> None:
    """The rendered LABELS must be whole numbers too, not `3.0`."""

    figure = ws.ac_table(series, lags=20).plot()
    try:
        figure.canvas.draw()
        labels = [text.get_text() for text in figure.axes[0].get_xticklabels() if text.get_text()]
        assert labels
        for label in labels:
            assert "." not in label, labels
    finally:
        plt.close(figure)


def test_varstable_keeps_fractional_ticks() -> None:
    """Its x axis is the eigenvalue real part -- a continuous quantity, so decimals are correct."""

    values = np.random.default_rng(3).normal(size=(120, 2))
    fitted = ws.VAR(values, lags=2, names=["a", "b"]).fit()
    figure = ws.varstable(fitted).plot()
    try:
        ticks = figure.axes[0].get_xticks()
        assert any(not float(tick).is_integer() for tick in ticks), ticks
    finally:
        plt.close(figure)
