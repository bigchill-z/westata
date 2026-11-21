"""`ac` and `pac`: the autocorrelation correlograms.

Both commands DRAW rather than tabulate, so the numbers were recovered from Stata's own
``generate(newvar)`` and cross-checked against the ``corrgram`` table, which reports the same two
columns. The reference series is an AR(1) with phi = 0.6 over 200 observations (seed 1).

TWO DEFINITIONS THAT A PLAUSIBLE IMPLEMENTATION GETS WRONG:

* ``pac`` is NOT Yule-Walker. The recursion gives .42822423 at lag 1 where Stata prints .42937017.
  Stata takes the LAST coefficient of an OLS regression of x_t on a CONSTANT and the first k lags;
  adding the intercept reproduces .42937017 and then all twelve lags match.
* the two bands differ in KIND, not just in width: ``ac`` uses Bartlett's formula for MA(q)
  (``sqrt((1 + 2*sum rho_j^2)/n)``), which WIDENS with the lag, while ``pac`` uses the constant
  ``1/sqrt(n)``.
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

# Measured from Stata: `ac y, lags(12) generate(rho)` and `pac y, lags(12) generate(prho)`.
STATA_AC = [
    0.42822423, 0.06899586, 0.04978896, 0.05504390, -0.00370413, -0.10755702,
    -0.14592822, -0.23804167, -0.23240357, -0.19818889, -0.07703128, -0.02466803,
]
STATA_PAC = [
    0.42937017, -0.14058180, 0.09532018, -0.00079455, -0.03732481, -0.11677536,
    -0.06484940, -0.21665331, -0.06485808, -0.11528968, 0.06819676, -0.03197955,
]


@pytest.fixture(scope="module")
def series() -> np.ndarray:
    generator = np.random.default_rng(1)
    values = np.zeros(200)
    for t in range(1, 200):
        values[t] = 0.6 * values[t - 1] + generator.normal()
    return values


# ---------------------------------------------------------------- values
def test_autocorrelations_match_stata(series) -> None:
    table = ws.ac_table(series, lags=12)
    assert table.values == pytest.approx(STATA_AC, abs=5e-8)


def test_partial_autocorrelations_match_stata(series) -> None:
    table = ws.pac_table(series, lags=12)
    assert table.values == pytest.approx(STATA_PAC, abs=5e-8)


def test_pac_is_the_regression_coefficient_not_yule_walker(series) -> None:
    """Yule-Walker gives .42822423 at lag 1; Stata (and this) give .42937017."""

    table = ws.pac_table(series, lags=1)
    assert table.values[0] == pytest.approx(0.42937017, abs=5e-8)

    # The Yule-Walker value is exactly the lag-1 autocorrelation; assert they DIFFER, so a future
    # regression to the recursion is caught.
    acf_first = ws.ac_table(series, lags=1).values[0]
    assert abs(table.values[0] - acf_first) > 1e-3


def test_lag_one_autocorrelation_is_the_obvious_ratio(series) -> None:
    deviations = series - series.mean()
    expected = float(deviations[:-1] @ deviations[1:]) / float(deviations @ deviations)
    assert ws.ac_table(series, lags=1).values[0] == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------- bands
def test_ac_band_uses_bartlett_and_widens(series) -> None:
    """se(k)^2 = (1 + 2*sum_{j<k} rho_j^2)/n, so the band grows as the lag does."""

    table = ws.ac_table(series, lags=12)
    assert table.upper[0] == pytest.approx(1.959963984540054 / np.sqrt(200), rel=1e-9)
    # Strictly increasing: each successive band adds a squared autocorrelation.
    assert np.all(np.diff(table.upper) > 0)


def test_pac_band_is_constant(series) -> None:
    """Stata's help: `pac` uses a standard error of 1/sqrt(n) throughout."""

    table = ws.pac_table(series, lags=12)
    assert np.allclose(table.upper, 1.959963984540054 / np.sqrt(200), rtol=1e-12)
    assert np.allclose(np.diff(table.upper), 0.0)


def test_the_band_is_symmetric(series) -> None:
    for table in (ws.ac_table(series, lags=8), ws.pac_table(series, lags=8)):
        assert np.allclose(table.lower, -table.upper)


def test_level_widens_the_band(series) -> None:
    narrow = ws.ac_table(series, lags=8, level=0.90)
    wide = ws.ac_table(series, lags=8, level=0.99)
    assert np.all(wide.upper > narrow.upper)


# ---------------------------------------------------------------- inputs
def test_a_frame_with_one_column_works(series) -> None:
    frame = pd.DataFrame({"y": series})
    assert ws.ac_table(frame, lags=3).values == pytest.approx(STATA_AC[:3], abs=5e-8)


def test_a_frame_needs_a_column_when_wide(series) -> None:
    frame = pd.DataFrame({"y": series, "z": series * 2})
    with pytest.raises(ws.ConsistencyError, match="several columns"):
        ws.ac_table(frame, lags=3)


def test_a_column_can_be_named(series) -> None:
    frame = pd.DataFrame({"y": series, "z": series * 2})
    assert ws.ac_table(frame, column="z", lags=3).values == pytest.approx(
        STATA_AC[:3], abs=5e-8
    )


def test_a_constant_series_is_refused() -> None:
    with pytest.raises(ws.ConsistencyError, match="constant"):
        ws.ac_table(np.ones(50), lags=4)


def test_missing_values_are_dropped(series) -> None:
    dirty = series.copy()
    dirty[10] = np.nan
    assert ws.ac_table(dirty, lags=3).values.shape == (3,)


# ---------------------------------------------------------------- plot
def test_ac_plot_is_a_stem_inside_a_band(series) -> None:
    figure = ws.ac_table(series, lags=10).plot()
    try:
        axis = figure.axes[0]
        assert len(axis.collections) >= 2  # the band fill plus the stems
        assert len(axis.lines) >= 3        # zero line, two band edges, the markers
    finally:
        plt.close(figure)


def test_pac_plot_uses_a_constant_band(series) -> None:
    figure = ws.pac_table(series, lags=10).plot()
    try:
        axis = figure.axes[0]
        upper = [line for line in axis.lines if len(set(np.round(line.get_ydata(), 12))) == 1]
        assert upper  # a horizontal edge, which only a constant band produces
    finally:
        plt.close(figure)


def test_plot_accepts_an_existing_axis(series) -> None:
    _figure, axis = plt.subplots()
    try:
        returned = ws.ac_table(series, lags=6).plot(ax=axis)
        assert returned is axis.figure
    finally:
        plt.close(_figure)


def test_to_frame_round_trips(series) -> None:
    frame = ws.ac_table(series, lags=5).to_frame()
    assert list(frame.columns) == ["lag", "ac", "lower", "upper"]
    assert frame["ac"].to_numpy() == pytest.approx(STATA_AC[:5], abs=5e-8)

    pac = ws.pac_table(series, lags=5).to_frame()
    assert list(pac.columns) == ["lag", "pac", "lower", "upper"]


def test_plots_can_be_saved(series, tmp_path) -> None:
    target = tmp_path / "ac.png"
    figure = ws.ac_table(series, lags=8).plot()
    try:
        figure.savefig(target, dpi=70)
        assert target.exists() and target.stat().st_size > 1000
    finally:
        plt.close(figure)


def test_missing_matplotlib_raises_a_helpful_error(series, monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("no matplotlib")
        return real_import(name, *args, **kwargs)

    table = ws.ac_table(series, lags=4)
    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match=r"westata\[plot\]"):
        table.plot()


def test_the_package_exports_both(series) -> None:
    for name in ("ac_table", "pac_table", "AcTable", "PacTable"):
        assert name in ws.__all__, name
