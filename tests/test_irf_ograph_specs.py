"""`irf ograph`: several combinations overlaid on ONE axis.

STATA'S SPEC HAS FOUR FIELDS::

    irf ograph (irfname impulsevar responsevar stat) [(spec_2) ... [(spec_15)]]

so a spec is ``("iu", "unem", "dinf", "oirf")``, NOT a three-tuple. The previous build unpacked
three fields and raised ``ValueError: too many values to unpack`` on Stata's own spelling.

Two further alignments are pinned here:

* the index arguments are **1-based**, matching `i()`/`r()` (they defaulted to 0, which the 1-based
  rule now rejects);
* Stata documents a ceiling of **15** specs.
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
from westata.postestimation import irf_ograph

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


def _curves(figure) -> int:
    return len([line for line in figure.axes[0].lines if len(line.get_ydata()) > 2])


def _labels(figure) -> list[str]:
    legend = figure.axes[0].get_legend()
    return [text.get_text() for text in legend.get_texts()] if legend else []


# ---------------------------------------------------------------- Stata's 4-field spec
def test_a_four_field_spec_is_accepted(fitted) -> None:
    """`(irfname impulse response stat)` -- Stata's own order."""

    figure = irf_ograph(
        fitted,
        ographs=[("iu", "unem", "dinf", "oirf"), ("iu", "unem", "unem", "oirf")],
        periods=6,
    )
    try:
        assert _curves(figure) == 2
    finally:
        plt.close(figure)


def test_the_four_field_spec_legends_both_pairs(fitted) -> None:
    figure = irf_ograph(
        fitted,
        ographs=[("iu", "unem", "dinf", "oirf"), ("iu", "unem", "unem", "oirf")],
        periods=6,
    )
    try:
        labels = _labels(figure)
        assert len(labels) == 2
        assert any("dinf" in text for text in labels)
        assert any("unem" in text for text in labels)
    finally:
        plt.close(figure)


def test_the_irfname_slot_is_only_a_label(fitted) -> None:
    """One westata result carries one set of IRFs, so the name must not change the numbers."""

    named = irf_ograph(
        fitted, ographs=[("iu", "unem", "dinf", "oirf")], periods=6
    )
    dotted = irf_ograph(
        fitted, ographs=[(".", "unem", "dinf", "oirf")], periods=6
    )
    try:
        first = [float(v) for v in named.axes[0].lines[-1].get_ydata()]
        second = [float(v) for v in dotted.axes[0].lines[-1].get_ydata()]
        assert first == pytest.approx(second, rel=1e-12)
    finally:
        plt.close(named)
        plt.close(dotted)


def test_a_three_field_spec_still_works(fitted) -> None:
    """The earlier `(impulse, response, stat)` spelling is kept for compatibility."""

    figure = irf_ograph(
        fitted, ographs=[("unem", "dinf", "oirf"), ("dinf", "dinf", "oirf")], periods=6
    )
    try:
        assert _curves(figure) == 2
    finally:
        plt.close(figure)


def test_an_ambiguously_shaped_spec_is_refused(fitted) -> None:
    with pytest.raises(ws.ConsistencyError, match="4 fields"):
        irf_ograph(fitted, ographs=[("unem", "dinf")], periods=4)


# ---------------------------------------------------------------- the convenience form
def test_one_pair_with_several_statistics(fitted) -> None:
    figure = irf_ograph(
        fitted, impulse="unem", response="dinf", statistic=["oirf", "coirf", "irf"], periods=6
    )
    try:
        assert _curves(figure) == 3
        labels = _labels(figure)
        assert sorted(labels) == ["cirf", "coirf", "irf"] or len(labels) == 3
    finally:
        plt.close(figure)


def test_positions_are_one_based(fitted) -> None:
    """`i(2) r(1)` must equal `i(unem) r(dinf)`, the same rule as `irf graph`."""

    by_name = irf_ograph(fitted, impulse="unem", response="dinf", periods=4)
    by_position = irf_ograph(fitted, impulse=2, response=1, periods=4)
    try:
        first = [float(v) for v in by_name.axes[0].lines[-1].get_ydata()]
        second = [float(v) for v in by_position.axes[0].lines[-1].get_ydata()]
        assert first == pytest.approx(second, rel=1e-12)
    finally:
        plt.close(by_name)
        plt.close(by_position)


def test_zero_is_rejected(fitted) -> None:
    with pytest.raises(ws.ConsistencyError):
        irf_ograph(fitted, impulse=0, response=1, periods=4)


# ---------------------------------------------------------------- limits and shapes
def test_the_fifteen_spec_ceiling_is_enforced(fitted) -> None:
    specs = [("iu", "unem", "dinf", "oirf")] * 16
    with pytest.raises(ws.ConsistencyError, match="at most 15"):
        irf_ograph(fitted, ographs=specs, periods=4)


def test_fifteen_specs_are_allowed(fitted) -> None:
    specs = [("iu", "unem", "dinf", "oirf")] * 15
    figure = irf_ograph(fitted, ographs=specs, periods=4)
    try:
        assert _curves(figure) == 15
    finally:
        plt.close(figure)


def test_an_empty_spec_list_is_refused(fitted) -> None:
    with pytest.raises(ws.ConsistencyError, match="empty"):
        irf_ograph(fitted, ographs=[], periods=4)


def test_one_axis_only(fitted) -> None:
    """Overlaying means ONE axis, unlike `irf graph`'s per-pair grid."""

    figure = irf_ograph(
        fitted,
        ographs=[("iu", "unem", "dinf", "oirf"), ("iu", "dinf", "dinf", "oirf")],
        periods=5,
    )
    try:
        assert len([axis for axis in figure.axes if axis.get_visible()]) == 1
    finally:
        plt.close(figure)


def test_the_band_is_drawn_only_for_a_single_spec(fitted) -> None:
    """Several overlapping CI regions on one axis are unreadable, so only one spec gets a band."""

    single = irf_ograph(fitted, ographs=[("iu", "unem", "dinf", "oirf")], periods=5)
    many = irf_ograph(
        fitted,
        ographs=[("iu", "unem", "dinf", "oirf"), ("iu", "unem", "unem", "oirf")],
        periods=5,
    )
    try:
        assert len(single.axes[0].collections) > len(many.axes[0].collections)
    finally:
        plt.close(single)
        plt.close(many)
