"""`irf create ..., order(varlist)` -- the Cholesky attribution order.

THE MATH

`order()` redo's the factorisation in the requested order:

    perm    = requested order as indices into the model's order
    Sigma_p = Sigma[perm][:, perm]
    A_lp    = A_l[perm][:, perm]
    P       = chol(Sigma_p)
    phi_0   = P,   phi_h = A_1p phi_{h-1} + A_2p phi_{h-2}
    report  phi_h mapped back by argsort(perm)

Permuting the impulse axis of an ALREADY orthogonalised array is NOT equivalent: the factorisation is
triangular, so the order changes the numbers themselves.

THE OBSERVABLE CONSEQUENCE, measured on `var dinf unem, lags(1/2)` from macro_swatson:

    natural            dinf -> unem step 0 = -.000188
    order(unem dinf)   dinf -> unem step 0 =  0

because dinf becomes the LAST shock, so it no longer moves unem contemporaneously. All 32 values
across both orders reproduce Stata to 4.8e-07.
"""

import pathlib
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.estimators.tseries import _reorder_orthogonal

DATA = r"D:\DeskBox\文件夹\计量经济学\python实现\data\macro_swatson.dta"

# Stata's oirf, steps 0..3, under each order.
STATA = {
    ("dinf", "unem"): {
        ("dinf", "dinf"): [0.013603, -0.005984, -0.003259, 0.004231],
        ("dinf", "unem"): [-0.000188, 0.000107, -0.000035, -0.000161],
        ("unem", "dinf"): [0.0, -0.005521, -0.001434, 0.000372],
        ("unem", "unem"): [0.002462, 0.004033, 0.004782, 0.005139],
    },
    ("unem", "dinf"): {
        ("dinf", "dinf"): [0.013564, -0.006388, -0.003359, 0.004247],
        ("dinf", "unem"): [0.0, 0.000414, 0.00033, 0.000231],
        ("unem", "dinf"): [-0.001038, -0.005048, -0.001181, 0.000048],
        ("unem", "unem"): [0.00247, 0.004014, 0.004771, 0.005137],
    },
}


@pytest.fixture(scope="module")
def fitted():
    if not pathlib.Path(DATA).exists():
        pytest.skip("macro_swatson.dta is not available")
    frame = pd.read_stata(DATA).dropna(subset=["dinf", "unem"]).reset_index(drop=True)
    return ws.VAR(
        frame[["dinf", "unem"]].to_numpy(), lags=2, names=["dinf", "unem"]
    ).fit()


# ---------------------------------------------------------------- the numbers
@pytest.mark.parametrize("order", [("dinf", "unem"), ("unem", "dinf")])
@pytest.mark.parametrize(
    "pair", [("dinf", "dinf"), ("dinf", "unem"), ("unem", "dinf"), ("unem", "unem")]
)
def test_every_pair_matches_stata(fitted, order, pair) -> None:
    names = ["dinf", "unem"]
    reordered = _reorder_orthogonal(fitted, list(order), 3)
    impulse = names.index(pair[0])
    response = names.index(pair[1])
    mine = [float(reordered[h][response, impulse]) for h in range(4)]
    # Stata prints 6 decimals, so 1e-6 is the honest tolerance.
    assert mine == pytest.approx(STATA[order][pair], abs=1e-6)


def test_the_natural_order_is_unchanged(fitted) -> None:
    """Passing the model's own order must reproduce the default analysis exactly."""

    baseline = np.asarray(fitted.irf(periods=3).orth_irfs, dtype=float)
    reordered = _reorder_orthogonal(fitted, ["dinf", "unem"], 3)
    np.testing.assert_allclose(reordered, baseline, rtol=1e-12, atol=1e-12)


def test_the_last_shock_has_no_contemporaneous_effect_on_the_first(fitted) -> None:
    """This is the whole point of order(): triangular attribution."""

    names = ["dinf", "unem"]
    reordered = _reorder_orthogonal(fitted, ["unem", "dinf"], 3)
    # dinf is now LAST, so its step-0 effect on unem (the first variable) is structurally zero.
    assert float(reordered[0][names.index("unem"), names.index("dinf")]) == pytest.approx(0.0, abs=1e-14)
    # Whereas the natural order has a real value there.
    natural = _reorder_orthogonal(fitted, ["dinf", "unem"], 3)
    assert abs(float(natural[0][names.index("unem"), names.index("dinf")])) > 1e-5


def test_only_the_leading_shocks_own_impact_is_unchanged(fitted) -> None:
    """The LEADING variable keeps its self-impact; a trailing one changes.

    MEASURED: the leading variable's self-impact is .002470 for unem and .013603 for dinf under the
    natural order, and stays put when it leads. A TRAILING variable's self-impact moves (.013603 ->
    .013564 for dinf once unem leads), because it is then computed conditional on unem -- which is
    precisely why the factorisation has to be redone rather than permuted afterwards.
    """

    names = ["dinf", "unem"]
    natural = _reorder_orthogonal(fitted, ["dinf", "unem"], 1)
    swapped = _reorder_orthogonal(fitted, ["unem", "dinf"], 1)

    leading_natural = float(natural[0][names.index("dinf"), names.index("dinf")])
    assert leading_natural == pytest.approx(0.013603, abs=1e-6)

    leading_swapped = float(swapped[0][names.index("unem"), names.index("unem")])
    assert leading_swapped == pytest.approx(0.002470, abs=1e-6)

    # A trailing variable's self-impact DOES move.
    trailing_swapped = float(swapped[0][names.index("dinf"), names.index("dinf")])
    assert trailing_swapped != pytest.approx(leading_natural, rel=1e-3)

def test_a_wrong_order_is_refused(fitted) -> None:
    for bad in (["unem"], ["unem", "dinf", "dinf"], ["unem", "mpg"]):
        with pytest.raises(ws.ConsistencyError, match="order must list"):
            _reorder_orthogonal(fitted, bad, 2)


# ---------------------------------------------------------------- the file export
def test_irf_create_honours_order(fitted, tmp_path) -> None:
    natural = fitted.irf_create(tmp_path / "n.irf", irfname="ui", step=3)
    reordered = fitted.irf_create(
        tmp_path / "r.irf", irfname="ui", step=3, order=["unem", "dinf"]
    )

    def cell(frame, impulse, response, step):
        subset = frame[(frame["impulse"] == impulse) & (frame["response"] == response)]
        return float(subset[subset["step"] == step]["oirf"].iloc[0])

    assert cell(natural, "dinf", "unem", 0) == pytest.approx(-0.000188, abs=1e-6)
    assert cell(reordered, "dinf", "unem", 0) == pytest.approx(0.0, abs=1e-14)
    assert cell(reordered, "unem", "dinf", 1) == pytest.approx(-0.005048, abs=1e-6)


def test_irf_create_without_order_is_unchanged(fitted, tmp_path) -> None:
    """The default path must be byte-for-byte the old behaviour."""

    plain = fitted.irf_create(tmp_path / "a.irf", irfname="ui", step=3)
    explicit = fitted.irf_create(
        tmp_path / "b.irf", irfname="ui", step=3, order=["dinf", "unem"]
    )
    pd.testing.assert_frame_equal(plain, explicit)


def test_order_does_not_touch_the_plain_irf_column(fitted, tmp_path) -> None:
    """`irf` is permutation-invariant: only the orthogonalised columns move."""

    natural = fitted.irf_create(tmp_path / "n2.irf", irfname="ui", step=2)
    reordered = fitted.irf_create(
        tmp_path / "r2.irf", irfname="ui", step=2, order=["unem", "dinf"]
    )
    pd.testing.assert_series_equal(natural["irf"], reordered["irf"])
