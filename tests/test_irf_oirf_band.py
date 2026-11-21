"""The `oirf` band must be the JOINT (B, Sigma) delta method, as Stata's is.

THE BUG THIS PINS

`irf_graph(result, statistic="oirf")` drew bands about 6% too narrow, and exactly zero where Stata
reports a non-zero band. `_band` read `statsmodels`'s `stderr()`, which treats the Cholesky factor as
KNOWN.

THE EVIDENCE that Sigma carries sampling error is the step-0 row, where the oirf IS a Cholesky
element: for the structurally-zero entry `unem -> dinf` Stata reports se 0, while `dinf -> dinf`
reports .0007510342. A band built from B alone gives 0 for both.

THE FIX: Gaussian ML asymptotics make B and Sigma independent, so the variances add,

    Var(f)     = f_B' Var(vec B) f_B + f_S' Var(vec Sigma_ml) f_S
    Var(vec S) = (I + K)(Sigma (x) Sigma) / T

with K the commutation matrix. Both gradients are numeric.

The reference values come from `irf table oirf` after
`var dinf unem, lags(1/2)` on macro_swatson (N = 164).
"""

import pathlib
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.postestimation.irf_graph import _joint_stderr, _inner

DATA = r"D:\DeskBox\文件夹\计量经济学\python实现\data\macro_swatson.dta"

# Stata's `irf table oirf`, recovered as (upper - lower)/2 / 1.959963984540054.
STATA_SE = {
    ("dinf", "dinf"): [0.0007510342, 0.0010548663, 0.0011012957],
    ("dinf", "unem"): [0.0001926056, 0.0003553637, 0.0004714372],
    ("unem", "dinf"): [0.0, 0.0008844550, 0.0005186320],
    ("unem", "unem"): [0.0001359719, 0.0002691376, 0.0004262833],
}

# Stata's oirf point values, for the same rows.
STATA_OIRF = {
    ("dinf", "dinf"): [0.013603, -0.005984, -0.003259],
    ("dinf", "unem"): [-0.000188, 0.000107, -0.000035],
    ("unem", "dinf"): [0.0, -0.005521, -0.001434],
    ("unem", "unem"): [0.002462, 0.004033, 0.004782],
}

pytestmark = pytest.mark.skipif(
    not pathlib.Path(DATA).exists(), reason="macro_swatson.dta is not available"
)


@pytest.fixture(scope="module")
def fitted():
    frame = pd.read_stata(DATA).dropna(subset=["dinf", "unem"]).reset_index(drop=True)
    return ws.VAR(
        frame[["dinf", "unem"]].to_numpy(), lags=2, names=["dinf", "unem"]
    ).fit()


@pytest.fixture(scope="module")
def analysis(fitted):
    return _inner(fitted, 8)


# ---------------------------------------------------------------- point values
@pytest.mark.parametrize("pair", list(STATA_OIRF))
def test_the_oirf_point_values_match_stata(analysis, pair) -> None:
    impulse, response = pair
    index_impulse = ["dinf", "unem"].index(impulse)
    index_response = ["dinf", "unem"].index(response)
    mine = np.asarray(analysis.orth_irfs)[:3, index_response, index_impulse]
    assert mine == pytest.approx(STATA_OIRF[pair], abs=5e-6, rel=1e-4)


# ---------------------------------------------------------------- the band
@pytest.mark.parametrize("pair", list(STATA_SE))
def test_the_oirf_band_matches_stata(analysis, pair) -> None:
    impulse, response = pair
    index_impulse = ["dinf", "unem"].index(impulse)
    index_response = ["dinf", "unem"].index(response)
    stderr = _joint_stderr(analysis, "oirf", np.asarray(analysis.orth_irfs))
    assert stderr is not None
    mine = stderr[:3, index_response, index_impulse]
    # Stata prints 4 decimals of the bounds, so 1e-6 is the honest tolerance.
    assert mine == pytest.approx(STATA_SE[pair], abs=1e-6)


def test_the_structural_zero_entry_has_a_zero_band(analysis) -> None:
    """`unem -> dinf` at step 0 is structurally zero, and so is its band; a B-only band gives zero
    for the non-zero `dinf -> dinf` entry too, which is what made the defect visible."""

    stderr = _joint_stderr(analysis, "oirf", np.asarray(analysis.orth_irfs))
    assert stderr[0, 0, 1] == pytest.approx(0.0, abs=1e-14)
    assert stderr[0, 0, 0] == pytest.approx(0.0007510342, abs=1e-6)


def test_the_sigma_term_is_actually_contributing(analysis) -> None:
    """If the Sigma contribution were dropped, step 0 of `dinf -> dinf` would be 0."""

    stderr = _joint_stderr(analysis, "oirf", np.asarray(analysis.orth_irfs))
    assert stderr[0, 0, 0] > 1e-5


def test_the_band_is_positive_and_finite(analysis) -> None:
    stderr = _joint_stderr(analysis, "oirf", np.asarray(analysis.orth_irfs))
    assert np.all(np.isfinite(stderr))
    assert np.all(stderr >= 0.0)


def test_the_band_no_longer_matches_the_narrow_approximation(analysis) -> None:
    """The old path returned statsmodels' stderr; the joint one must differ measurably."""

    joint = _joint_stderr(analysis, "oirf", np.asarray(analysis.orth_irfs))
    approximation = np.asarray(analysis.stderr(), dtype=float)
    difference = np.abs(joint[1, 0, 1] - approximation[1, 0, 1])
    assert difference > 1e-5, "the two paths agreeing means the Sigma term vanished again"


# ---------------------------------------------------------------- the graph
def test_the_graph_bands_use_the_joint_standard_errors(fitted) -> None:
    from westata.postestimation import irf_series

    series = irf_series(fitted, statistic="oirf", periods=4)
    assert series.lower is not None
    # The first panel is dinf -> dinf; its step-0 band half-width must be ~.00075 * 1.96.
    half = float(series.upper[0, 0, 0] - series.values[0, 0, 0])
    assert half == pytest.approx(0.0007510342 * 1.959963984540054, rel=2e-3)


def test_the_graph_draws_four_panels(fitted) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from westata.postestimation import irf_graph

    plt.close("all")
    figure = irf_graph(fitted, statistic="oirf", periods=8)
    try:
        visible = [axis for axis in figure.axes if axis.get_visible()]
        assert len(visible) == 4
    finally:
        plt.close(figure)
