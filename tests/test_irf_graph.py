"""`irf graph`: the five statistics a `var` can supply, and the figures built from them.

Every number here was captured from StataNow 19.5 with `var dinf unem, lags(1/2)` on
macro_swatson.dta, then `irf table <stat>, noci` at steps 0..4. Stata's four columns are

    (1) impulse=dinf response=dinf      (2) impulse=dinf response=unem
    (3) impulse=unem response=dinf      (4) impulse=unem response=unem

Three findings are load-bearing, and each cost a measurement rather than a guess:

* `dm`, `cdm`, `sirf`, `csirf` and `sfevd` are ACCEPTED by `irf graph`'s syntax but are
  undefined after a plain `var` -- Stata prints `.` in every cell. They raise here, naming
  the model they belong to, instead of plotting zeros.
* `coirf` is the cumulative sum of the ORTHOGONALISED responses. statsmodels'
  `orth_cum_effects` is NOT that: it is off by `2.1e-04` at step 4, which is larger than
  every printed digit.
* statsmodels' FEVD `decomp` is `(response, step, impulse)` and starts at step 1. Getting
  the permutation wrong swaps `.121271` with `.002094` -- an error of `1.2e-01` that still
  looks like a plausible variance share, so it is pinned by a test.
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
from westata.errors import ConsistencyError

DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"

pytestmark = pytest.mark.skipif(
    not __import__("pathlib").Path(DATA).exists(),
    reason="macro_swatson.dta is not available on this machine",
)

#: StataNow 19.5, steps 0..4, columns (1)..(4) as documented above.
STATA = {
    "irf": [
        [1, 0, 0, 1],
        [-0.470944, 0.030522, -2.24205, 1.63803],
        [-0.247676, 0.024305, -0.582498, 1.94213],
        [0.313099, 0.017054, 0.151132, 2.08716],
        [-0.036921, 0.023947, -0.566462, 2.12379],
    ],
    "oirf": [
        [0.013603, -0.000188, 0, 0.002462],
        [-0.005984, 0.000107, -0.005521, 0.004033],
        [-0.003259, -0.000035, -0.001434, 0.004782],
        [0.004231, -0.000161, 0.000372, 0.005139],
        [-0.000396, -0.000074, -0.001395, 0.00523],
    ],
    "cirf": [
        [1, 0, 0, 1],
        [0.529056, 0.030522, -2.24205, 2.63803],
        [0.281381, 0.054827, -2.82455, 4.58016],
        [0.59448, 0.071881, -2.67342, 6.66732],
        [0.557558, 0.095828, -3.23988, 8.79111],
    ],
    "coirf": [
        [0.013603, -0.000188, 0, 0.002462],
        [0.007619, -0.000082, -0.005521, 0.006496],
        [0.00436, -0.000117, -0.006955, 0.011278],
        [0.008591, -0.000279, -0.006583, 0.016418],
        [0.008195, -0.000353, -0.007978, 0.021647],
    ],
    "fevd": [
        [0, 0, 0, 0],
        [1, 0.005822, 0, 0.994178],
        [0.878729, 0.002094, 0.121271, 0.997906],
        [0.876761, 0.001063, 0.123239, 0.998937],
        [0.884151, 0.001034, 0.115849, 0.998966],
    ],
}

#: Stata's column order as (response, impulse) index pairs, for names [dinf, unem].
COLUMNS = [(0, 0), (1, 0), (0, 1), (1, 1)]


@pytest.fixture(scope="module")
def fitted():
    frame = pd.read_stata(DATA)
    return ws.VAR(frame[["dinf", "unem"]].dropna(), lags=2).fit()


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _grid(series) -> np.ndarray:
    return np.column_stack([series.values[:, r, i] for r, i in COLUMNS])


# ---------------------------------------------------------------- values match Stata
@pytest.mark.parametrize("statistic", sorted(STATA))
def test_every_statistic_matches_the_stata_capture(fitted, statistic: str) -> None:
    """Stata prints six significant figures, so <5e-6 is the same printed value."""

    series = ws.irf_series(fitted, statistic=statistic, periods=4)
    worst = float(np.max(np.abs(_grid(series) - np.asarray(STATA[statistic]))))
    assert worst < 5e-6, (statistic, worst)


@pytest.mark.parametrize("statistic", sorted(STATA))
def test_the_axes_are_step_response_impulse(fitted, statistic: str) -> None:
    """One index convention for all five, so a caller never has to remember a special case."""

    series = ws.irf_series(fitted, statistic=statistic, periods=4)
    assert series.values.shape == (5, 2, 2)
    # A specific cell that would move if response and impulse were swapped.
    assert series.values[2, 0, 0] == pytest.approx(
        np.asarray(STATA[statistic])[2][0], abs=5e-6
    )


def test_oirf_starts_at_the_cholesky_impact(fitted) -> None:
    """`unem` cannot move `dinf` on impact when dinf is ordered first."""

    series = ws.irf_series(fitted, statistic="oirf", periods=4)
    assert series.values[0, 0, 1] == pytest.approx(0.0, abs=1e-12)
    assert series.values[0, 0, 0] == pytest.approx(0.013603, abs=5e-7)


def test_coirf_is_not_statsmodels_orth_cum_effects(fitted) -> None:
    """The documented trap: `orth_cum_effects` is off by 2.1e-04 at step 4."""

    analysis = fitted.irf(periods=4)
    ours = ws.irf_series(fitted, statistic="coirf", periods=4).values
    theirs = np.asarray(analysis.orth_cum_effects, dtype=float)
    # Stata's step-4 dinf->dinf is .008195; ours matches, statsmodels does not.
    assert ours[4, 0, 0] == pytest.approx(0.008195, abs=5e-7)
    assert abs(float(theirs[4, 0, 0]) - 0.008195) > 1e-4


def test_fevd_off_diagonal_cells_are_not_swapped(fitted) -> None:
    """`.121271` and `.002094` live in different cells; a wrong permutation swaps them.

    Stata's column (3) is `impulse=unem, response=dinf` = `.121271`, and column (2) is
    `impulse=dinf, response=unem` = `.002094`. With `values[step, response, impulse]` those
    are `values[2, 0, 1]` and `values[2, 1, 0]` respectively.
    """

    series = ws.irf_series(fitted, statistic="fevd", periods=4)
    # response=dinf, impulse=unem  -> Stata column (3)
    assert series.values[2, 0, 1] == pytest.approx(0.121271, abs=5e-6)
    # response=unem, impulse=dinf  -> Stata column (2)
    assert series.values[2, 1, 0] == pytest.approx(0.002094, abs=5e-6)


def test_fevd_has_a_zero_first_step(fitted) -> None:
    """Stata prints step 0 as all zeros; statsmodels starts at step 1."""

    series = ws.irf_series(fitted, statistic="fevd", periods=4)
    assert np.allclose(series.values[0], 0.0)


def test_fevd_carries_no_confidence_band(fitted) -> None:
    series = ws.irf_series(fitted, statistic="fevd", periods=4)
    assert series.lower is None and series.upper is None


# ---------------------------------------------------------------- refusals
@pytest.mark.parametrize(
    "statistic,expected",
    [
        ("dm", "ARDL"),
        ("cdm", "ARDL"),
        ("sirf", "svar"),
        ("csirf", "svar"),
        ("sfevd", "svar"),
    ],
)
def test_statistics_undefined_after_var_are_refused(
    fitted, statistic: str, expected: str
) -> None:
    """Stata prints `.` for these; plotting zeros would be a silent fabrication.

    Each refusal names the model the statistic actually belongs to, so a reader can tell
    "not implemented here" from "not applicable to this model".
    """

    with pytest.raises(ConsistencyError, match=expected):
        ws.irf_series(fitted, statistic=statistic, periods=4)


def test_the_refusal_names_the_model_the_statistic_belongs_to(fitted) -> None:
    with pytest.raises(ConsistencyError) as caught:
        ws.irf_series(fitted, statistic="dm", periods=4)
    assert "ARDL" in str(caught.value)


def test_an_unknown_statistic_lists_the_choices(fitted) -> None:
    with pytest.raises(ConsistencyError) as caught:
        ws.irf_series(fitted, statistic="nope", periods=4)
    message = str(caught.value)
    for choice in ("irf", "oirf", "cirf", "coirf", "fevd"):
        assert choice in message


def test_a_bad_variable_name_is_refused(fitted) -> None:
    with pytest.raises(ConsistencyError, match="not one of"):
        ws.irf_graph(fitted, statistic="oirf", impulse="nonexistent")


# ---------------------------------------------------------------- names
def test_panel_names_come_from_the_fit_not_from_statsmodels(fitted) -> None:
    """statsmodels' IRAnalysis carries no names at all; reading them there yields y1/y2."""

    series = ws.irf_series(fitted, statistic="oirf", periods=4)
    assert list(series.names) == ["dinf", "unem"]


def test_an_analysis_object_is_accepted_too(fitted) -> None:
    analysis = fitted.irf(periods=4)
    series = ws.irf_series(analysis, statistic="oirf", periods=4)
    assert series.values.shape == (5, 2, 2)


# ---------------------------------------------------------------- figures
def test_one_panel_per_combination_by_default(fitted) -> None:
    figure = ws.irf_graph(fitted, statistic="oirf", periods=6)
    assert len(figure.axes) == 4
    titles = [axis.get_title() for axis in figure.axes]
    assert titles == ["dinf → dinf", "unem → dinf", "dinf → unem", "unem → unem"]


def test_narrowing_to_one_pair_draws_one_panel(fitted) -> None:
    figure = ws.irf_graph(fitted, statistic="oirf", impulse="unem", response="dinf", periods=6)
    assert len(figure.axes) == 1
    assert figure.axes[0].get_title() == "unem → dinf"


def test_noci_removes_the_band(fitted) -> None:
    with_band = ws.irf_graph(fitted, statistic="oirf", periods=6)
    without = ws.irf_graph(fitted, statistic="oirf", periods=6, show_ci=False)
    assert all(len(axis.collections) == 1 for axis in with_band.axes)
    assert all(len(axis.collections) == 0 for axis in without.axes)


def test_individual_gives_one_figure_per_combination(fitted) -> None:
    figures = ws.irf_graph(fitted, statistic="oirf", periods=6, individual=True)
    assert isinstance(figures, list) and len(figures) == 4


def test_the_band_widens_with_the_confidence_level(fitted) -> None:
    narrow = ws.irf_series(fitted, statistic="oirf", periods=6, level=0.90)
    wide = ws.irf_series(fitted, statistic="oirf", periods=6, level=0.99)
    narrow_width = float(np.mean(wide.upper - wide.lower))
    small_width = float(np.mean(narrow.upper - narrow.lower))
    assert narrow_width > small_width


def test_ax_needs_exactly_one_pair(fitted) -> None:
    figure, axis = plt.subplots()
    with pytest.raises(ConsistencyError, match="exactly one"):
        ws.irf_graph(fitted, statistic="oirf", ax=axis)
    plt.close(figure)


def test_drawing_into_an_axis_returns_its_figure(fitted) -> None:
    figure, axis = plt.subplots()
    returned = ws.irf_graph(
        fitted, statistic="oirf", impulse="dinf", response="dinf", ax=axis
    )
    assert returned is figure


def test_same_y_shares_one_scale(fitted) -> None:
    figure = ws.irf_graph(fitted, statistic="oirf", periods=6, same_y=True)
    limits = {axis.get_ylim() for axis in figure.axes}
    assert len(limits) == 1


def test_fevd_panels_are_labelled_as_shares(fitted) -> None:
    figure = ws.irf_graph(fitted, statistic="fevd", periods=6)
    assert figure.axes[0].get_ylabel() == "share"


def test_plot_needs_matplotlib(monkeypatch, fitted) -> None:
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("no matplotlib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match=r"westata\[plot\]"):
        ws.irf_graph(fitted, statistic="oirf")


def test_the_series_to_frame_is_long_and_complete(fitted) -> None:
    frame = ws.irf_series(fitted, statistic="oirf", periods=4).to_frame()
    assert len(frame) == 4 * 5  # 4 combinations x 5 steps
    assert set(frame.columns) == {"step", "impulse", "response", "value", "lower", "upper"}


# ---------------------------------------------------------------- lstep / ustep
def test_lstep_and_ustep_trim_the_reported_steps(fitted) -> None:
    """Stata's `lstep(#)` / `ustep(#)` narrow the axis without recomputing anything."""

    series = ws.irf_series(fitted, statistic="oirf", periods=8, lstep=2, ustep=5)
    assert series.steps == (2, 3, 4, 5)
    assert series.values.shape[0] == 4


def test_trimming_does_not_restart_a_cumulative_series(fitted) -> None:
    """`coirf` at a trimmed step must equal the FULL-horizon cumulation.

    Summing from the trimmed start would silently drop the earlier steps -- the same trap
    `_qsur.ado:54` sets for the lag sample.
    """

    full = ws.irf_series(fitted, statistic="coirf", periods=8)
    trimmed = ws.irf_series(fitted, statistic="coirf", periods=8, lstep=2, ustep=5)
    assert trimmed.values[0, 0, 0] == pytest.approx(full.values[2, 0, 0], abs=1e-12)


def test_negative_lstep_is_refused(fitted) -> None:
    with pytest.raises(ConsistencyError, match="lstep"):
        ws.irf_series(fitted, statistic="oirf", periods=8, lstep=-1)


def test_ustep_below_lstep_is_refused(fitted) -> None:
    with pytest.raises(ConsistencyError, match="ustep"):
        ws.irf_series(fitted, statistic="oirf", periods=8, lstep=5, ustep=3)


def test_ustep_beyond_the_horizon_is_refused(fitted) -> None:
    """Silently clamping would hide a typo; the message names the real last step."""

    with pytest.raises(ConsistencyError, match="beyond the reported horizon"):
        ws.irf_series(fitted, statistic="oirf", periods=8, ustep=99)


def test_irf_graph_honours_the_trim(fitted) -> None:
    figure = ws.irf_graph(fitted, statistic="oirf", periods=10, lstep=3, ustep=7)
    xdata = np.asarray(figure.axes[0].lines[0].get_xdata(), dtype=float)
    assert xdata.min() == 3.0 and xdata.max() == 7.0


# ---------------------------------------------------------------- irf ograph (overlay)
def test_ograph_overlays_several_statistics_on_one_axis(fitted) -> None:
    """`irf ograph` puts them on ONE axis so shapes can be compared, unlike `irf graph`."""

    figure = ws.irf_ograph(
        fitted, impulse="dinf", response="unem", statistic=["oirf", "coirf"], periods=8
    )
    assert len(figure.axes) == 1
    axis = figure.axes[0]
    labels = [item.get_text() for item in axis.get_legend().get_texts()]
    assert labels == ["oirf", "coirf"]
    assert axis.get_title() == "dinf → unem"


def test_ograph_accepts_statas_spec_triples(fitted) -> None:
    """Stata spells a spec `(impulse response stat)`; the triples map onto that directly."""

    figure = ws.irf_ograph(
        fitted,
        ographs=[("dinf", "dinf", "oirf"), ("unem", "unem", "oirf")],
        periods=8,
    )
    axis = figure.axes[0]
    labels = [item.get_text() for item in axis.get_legend().get_texts()]
    assert labels == ["dinf→dinf oirf", "unem→unem oirf"]


def test_ograph_draws_the_band_only_for_a_single_spec(fitted) -> None:
    """Several overlapping confidence regions on one axis are unreadable."""

    single = ws.irf_ograph(fitted, ographs=[("dinf", "unem", "oirf")], periods=8)
    assert len(single.axes[0].collections) == 1

    plt.close("all")
    several = ws.irf_ograph(
        fitted, impulse="dinf", response="unem", statistic=["oirf", "coirf"], periods=8
    )
    assert len(several.axes[0].collections) == 0


def test_ograph_shows_which_pair_when_several_are_overlaid(fitted) -> None:
    figure = ws.irf_ograph(
        fitted,
        ographs=[("dinf", "dinf", "oirf"), ("unem", "unem", "oirf")],
        periods=8,
    )
    assert figure.axes[0].get_title() == "Overlaid impulse responses"


def test_ograph_honours_the_trim(fitted) -> None:
    figure = ws.irf_ograph(
        fitted, ographs=[("dinf", "unem", "oirf")], periods=10, lstep=2, ustep=6
    )
    xdata = np.asarray(figure.axes[0].lines[0].get_xdata(), dtype=float)
    assert xdata.min() == 2.0 and xdata.max() == 6.0


def test_ograph_refuses_an_empty_spec_list(fitted) -> None:
    with pytest.raises(ConsistencyError, match="empty"):
        ws.irf_ograph(fitted, ographs=[])


def test_ograph_refuses_an_empty_statistic_list(fitted) -> None:
    with pytest.raises(ConsistencyError, match="no statistic"):
        ws.irf_ograph(fitted, statistic=[], impulse="dinf", response="unem")


def test_ograph_refuses_a_refused_statistic(fitted) -> None:
    """Overlay must go through the same gate as the single-statistic path."""

    with pytest.raises(ConsistencyError, match="ARDL"):
        ws.irf_ograph(fitted, ographs=[("dinf", "unem", "dm")])


def test_ograph_draws_into_an_existing_axis(fitted) -> None:
    figure, axis = plt.subplots()
    returned = ws.irf_ograph(fitted, ographs=[("dinf", "unem", "oirf")], periods=6, ax=axis)
    assert returned is figure


# ---------------------------------------------------------------- irf cgraph (combined)
def test_cgraph_gives_each_spec_its_own_panel(fitted) -> None:
    """`irf cgraph` combines SPECS as subgraphs; `irf ograph` overlays, `irf graph` splits."""

    figure = ws.irf_cgraph(
        fitted, specs=[("dinf", "unem", "oirf"), ("dinf", "unem", "fevd")], periods=6
    )
    assert len(figure.axes) == 2


def test_cgraph_lets_one_spec_hold_several_statistics(fitted) -> None:
    """Stata's spec takes one OR MORE statistics; several share that panel with a legend."""

    figure = ws.irf_cgraph(fitted, specs=[("dinf", "dinf", ["oirf", "cirf"])], periods=6)
    assert len(figure.axes) == 1
    labels = [item.get_text() for item in figure.axes[0].get_legend().get_texts()]
    assert labels == ["oirf", "cirf"]


def test_cgraph_convenience_form_is_one_panel_per_combination(fitted) -> None:
    figure = ws.irf_cgraph(fitted, statistic="oirf", periods=6)
    assert len(figure.axes) == 4


def test_cgraph_overlays_when_the_convenience_form_lists_several_statistics(fitted) -> None:
    figure = ws.irf_cgraph(
        fitted, statistic=["oirf", "coirf"], impulse="dinf", response="unem", periods=6
    )
    assert len(figure.axes) == 1
    labels = [item.get_text() for item in figure.axes[0].get_legend().get_texts()]
    assert labels == ["oirf", "coirf"]


def test_cgraph_refuses_an_empty_spec_list(fitted) -> None:
    with pytest.raises(ConsistencyError, match="empty"):
        ws.irf_cgraph(fitted, specs=[])


def test_cgraph_refuses_a_spec_without_a_statistic(fitted) -> None:
    with pytest.raises(ConsistencyError, match="no statistic"):
        ws.irf_cgraph(fitted, specs=[("dinf", "unem", [])])


def test_cgraph_uses_the_same_statistic_gate(fitted) -> None:
    with pytest.raises(ConsistencyError, match="ARDL"):
        ws.irf_cgraph(fitted, specs=[("dinf", "unem", "dm")])


def test_the_three_commands_differ_only_in_layout(fitted) -> None:
    """A regression guard for the confusion this family invites.

    Same data and statistic; only the arrangement of panels differs.
    """

    graph = ws.irf_graph(fitted, statistic="oirf", periods=6)
    assert len(graph.axes) == 4
    plt.close("all")

    overlay = ws.irf_ograph(
        fitted, statistic="oirf", impulse="dinf", response="unem", periods=6
    )
    assert len(overlay.axes) == 1
    plt.close("all")

    combined = ws.irf_cgraph(fitted, statistic="oirf", periods=6)
    assert len(combined.axes) == 4
    plt.close("all")


# ---------------------------------------------------------------- saving (isaving/iname)
def test_save_writes_one_file(fitted, tmp_path) -> None:
    target = tmp_path / "figure.png"
    ws.irf_graph(fitted, statistic="oirf", periods=6, save=str(target))
    assert target.exists() and target.stat().st_size > 0


def test_save_without_a_suffix_defaults_to_png(fitted, tmp_path) -> None:
    ws.irf_graph(fitted, statistic="oirf", periods=6, save=str(tmp_path / "figure"))
    assert (tmp_path / "figure.png").exists()


def test_individual_plus_save_numbers_the_files(fitted, tmp_path) -> None:
    """Stata's `iname(stub)` / `isaving(stub)`: one file per panel, not one overwritten file."""

    ws.irf_graph(
        fitted, statistic="oirf", periods=6, individual=True, save=str(tmp_path / "panel.png")
    )
    produced = sorted(p.name for p in tmp_path.iterdir())
    assert produced == ["panel1.png", "panel2.png", "panel3.png", "panel4.png"]


def test_save_refuses_a_missing_directory(fitted, tmp_path) -> None:
    """Silently creating a path would hide a typo in the destination."""

    with pytest.raises(ConsistencyError, match="does not exist"):
        ws.irf_graph(
            fitted, statistic="oirf", periods=6, save=str(tmp_path / "absent" / "x.png")
        )
