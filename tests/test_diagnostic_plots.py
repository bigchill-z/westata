"""`regress` diagnostic plots, in the house ``xxx_table(result).plot()`` spelling.

The seven commands Stata ships for `regress` postestimation -- `rvfplot`, `rvpplot`, `avplot`,
`avplots`, `cprplot`, `acprplot`, `lvr2plot` -- return NOTHING: they draw and forget, so they cannot
be compared output-for-output the way a table can. Each one is therefore pinned by the algebraic
identity that defines it, checked against quantities from a live Stata `predict`:

    predict fitted, xb         mean 6165.2568   (rvfplot's x axis)
    predict resid, residuals   sd   2479.3506   (rvfplot's y axis)
    predict hat, hat           mean .0405405406 == 3/74  (lvr2plot's normalisation)

The strongest check is the added-variable plot: the slope through its partial-regression residuals
must equal that predictor's own regression coefficient. It does, to six decimals.
"""

import contextlib
import io
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "replication")

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"docs\data\auto.csv"

pytestmark = pytest.mark.skipif(
    not pathlib.Path(DATA).exists(), reason="auto.csv is not available"
)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return pd.read_csv(DATA)


@pytest.fixture(scope="module")
def fitted(frame):
    return ws.Regress.from_formula("price ~ mpg + weight", frame).fit()


# ---------------------------------------------------------------- rvfplot
def test_rvf_table_holds_residuals_against_fitted(fitted) -> None:
    table = ws.rvf_table(fitted)
    assert table.fitted.size == table.residuals.size == 74
    # Measured on the same fit in Stata: the first three fitted values.
    assert table.fitted[:3] == pytest.approx([5974.2181, 6955.3341, 5467.7160], abs=1e-3)


def test_rvf_plot_draws_a_scatter_and_a_zero_line(fitted) -> None:
    figure = ws.rvf_table(fitted).plot()
    try:
        axis = figure.axes[0]
        assert len(axis.collections) == 1
        assert len(axis.lines) == 1  # the reference line
        assert axis.lines[0].get_ydata()[0] == 0.0
    finally:
        plt.close(figure)


# ---------------------------------------------------------------- rvpplot
def test_rvp_table_uses_the_predictor_column(fitted, frame) -> None:
    table = ws.rvp_table(fitted, "mpg")
    assert table.values == pytest.approx(frame["mpg"].to_numpy(dtype=float), abs=1e-9)


def test_rvp_rejects_an_unknown_predictor(fitted) -> None:
    with pytest.raises(ws.ConsistencyError, match="not in the model"):
        ws.rvp_table(fitted, "nope")


# ---------------------------------------------------------------- avplot / avplots
def test_avplot_slope_equals_the_coefficient(fitted) -> None:
    """The partial-regression residuals must recover b_j -- the defining property."""

    table = ws.av_table(fitted, "mpg")
    coefficient = float(np.asarray(fitted.params, dtype=float)[1])
    assert table.slope == pytest.approx(coefficient, rel=1e-9)
    assert table.slope == pytest.approx(-49.512221, abs=1e-6)


def test_avplot_residuals_are_orthogonal_to_the_other_regressors(fitted) -> None:
    """Partialling out removes every other column: the residuals must be uncorrelated with them."""

    table = ws.av_table(fitted, "mpg")
    assert abs(float(table.x.mean())) < 1e-9
    assert abs(float(table.y.mean())) < 1e-9


def test_avplots_returns_one_panel_per_predictor(fitted) -> None:
    table = ws.avplots_table(fitted)
    assert [panel.predictor for panel in table.panels] == ["mpg", "weight"]


def test_avplots_draws_every_panel(fitted) -> None:
    figure = ws.avplots_table(fitted).plot()
    try:
        visible = [axis for axis in figure.axes if axis.get_visible()]
        assert len(visible) == 2
    finally:
        plt.close(figure)


# ---------------------------------------------------------------- cprplot / acprplot
def test_cpr_uses_the_component_plus_residual(fitted) -> None:
    table = ws.cpr_table(fitted, "mpg")
    coefficient = float(np.asarray(fitted.params, dtype=float)[1])
    residuals = np.asarray(fitted.predict(kind="residuals"), dtype=float).reshape(-1)
    assert table.slope == pytest.approx(coefficient, rel=1e-9)
    assert table.component_plus_residual == pytest.approx(
        coefficient * table.values + residuals, abs=1e-9
    )


def test_acpr_adds_the_mallows_augmentation(fitted) -> None:
    """acpr - cpr == b * (x - mean(x)), exactly -- checked to 0.0."""

    cpr = ws.cpr_table(fitted, "mpg")
    acpr = ws.acpr_table(fitted, "mpg")
    expected = cpr.slope * (cpr.values - cpr.values.mean())
    difference = acpr.component_plus_residual - cpr.component_plus_residual
    assert float(np.abs(difference - expected).max()) < 1e-12


def test_acpr_keeps_the_same_slope(fitted) -> None:
    assert ws.acpr_table(fitted, "mpg").slope == pytest.approx(
        ws.cpr_table(fitted, "mpg").slope, rel=1e-12
    )


# ---------------------------------------------------------------- lvr2plot
def test_lvr2_normalisation_centres_both_axes_on_one(fitted) -> None:
    """Both means are 1 by construction; Stata draws its reference lines there."""

    table = ws.lvr2_table(fitted)
    assert float(table.normalised_leverage.mean()) == pytest.approx(1.0, abs=1e-12)
    assert float(table.normalised_residual.mean()) == pytest.approx(1.0, abs=1e-12)


def test_lvr2_leverage_mean_is_k_over_n(fitted) -> None:
    """Measured in Stata: mean(hat) = .0405405406 = 3/74."""

    table = ws.lvr2_table(fitted)
    assert table.leverage_mean == pytest.approx(3 / 74, abs=1e-9)


def test_lvr2_plot_draws_both_reference_lines(fitted) -> None:
    figure = ws.lvr2_table(fitted).plot()
    try:
        axis = figure.axes[0]
        horizontal = [line for line in axis.lines if len(set(line.get_ydata())) == 1]
        vertical = [line for line in axis.lines if len(set(line.get_xdata())) == 1]
        assert horizontal and vertical
    finally:
        plt.close(figure)


# ---------------------------------------------------------------- the house convention
def test_every_table_follows_the_xxx_table_plot_shape(fitted) -> None:
    """`rvf_table(res).plot()`, not `ws.rvfplot(res)`."""

    builders = [
        (ws.rvf_table, (fitted,)),
        (ws.rvp_table, (fitted, "mpg")),
        (ws.av_table, (fitted, "mpg")),
        (ws.avplots_table, (fitted,)),
        (ws.cpr_table, (fitted, "mpg")),
        (ws.acpr_table, (fitted, "mpg")),
        (ws.lvr2_table, (fitted,)),
    ]
    for builder, arguments in builders:
        table = builder(*arguments)
        assert hasattr(table, "to_frame"), builder.__name__
        assert callable(getattr(table, "plot", None)), builder.__name__
        assert isinstance(table.to_frame(), pd.DataFrame), builder.__name__


def test_varstable_table_is_the_same_builder(fitted) -> None:
    """`varstable_table(res)` exists so every plotting entry point reads alike."""

    from westata.estimators.tseries import VAR

    values = np.random.default_rng(3).normal(size=(120, 2))
    result = VAR(values, lags=2, names=["a", "b"]).fit()
    assert ws.varstable_table(result).to_frame().equals(ws.varstable(result).to_frame())


def test_a_non_regress_result_is_refused(fitted) -> None:
    class NotARegression:
        def predict(self, kind=None):  # noqa: D102
            return np.zeros(3)

        model_name = "logit"

    with pytest.raises(ws.ConsistencyError, match="belong to regress"):
        ws.rvf_table(NotARegression())


def test_plots_can_be_saved(fitted, tmp_path) -> None:
    target = tmp_path / "rvf.png"
    figure = ws.rvf_table(fitted).plot()
    try:
        figure.savefig(target, dpi=70)
        assert target.exists() and target.stat().st_size > 1000
    finally:
        plt.close(figure)


def test_missing_matplotlib_raises_a_helpful_error(fitted, monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("no matplotlib")
        return real_import(name, *args, **kwargs)

    table = ws.rvf_table(fitted)
    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match=r"westata\[plot\]"):
        table.plot()
