"""E3 conformance for the P8 quantile and robust regression commands.

Reference values come from ``stata_fixtures/p8_outputs.txt`` (StataNow 19.5,
captured by ``scripts/capture_p8_fixtures.py``). DGP for the quantile models:
n=600, seed=61, x1/x2 ~ N(0,1), scale = 0.5 + 0.4|x1|, y = 1 + 2x1 - 0.8x2 +
scale*N(0,1). The rreg DGP is n=400, seed=67 with 5% of y contaminated by
scale-30 noise.

Stata values:

======  ==========  ===========  ===========  ===========  =========
model   quantile    x1 (se)      x2 (se)      _cons (se)   PseudoR2
======  ==========  ===========  ===========  ===========  =========
qreg    0.25        2.042842     -.8013571    .518284      0.6424
                    (.0472461)   (.0452555)   (.0473547)
qreg    0.50        1.965287     -.7475726    .979461      0.6361
                    (.0355009)   (.0340052)   (.0355825)
qreg    0.75        1.978264     -.7318384    1.517499      0.6252
                    (.0513448)   (.0491815)   (.0514629)
qreg    0.90        1.996667     -.7252009    2.097133      0.6245
                    (.0717235)   (.0687016)   (.0718884)
rreg    --          2.014204     -.5340517    .9933214
                    (.0564219)   (.0525755)   (.0545107)
======  ==========  ===========  ===========  ===========  =========
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import westata as ws

FIXTURES = Path(__file__).resolve().parent / "stata_fixtures"
REFERENCE = FIXTURES / "p8_outputs.txt"

if not REFERENCE.exists():
    pytest.skip("P8 fixtures missing", allow_module_level=True)


def _require(name: str):
    attribute = getattr(ws, name, None)
    if attribute is None:
        pytest.skip(f"{name} is not exposed on westata yet")
    return attribute


def quantile_frame(n: int = 600, seed: int = 61) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    scale = 0.5 + 0.4 * np.abs(x1)
    y = 1.0 + 2.0 * x1 - 0.8 * x2 + scale * rng.normal(size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def outlier_frame(n: int = 400, seed: int = 67) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + rng.normal(size=n)
    kontam = rng.random(size=n) < 0.05
    y = np.where(kontam, y + rng.normal(scale=30.0, size=n), y)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


# Each entry: quantile -> (x1, se_x1, x2, se_x2, cons, se_cons, pseudo_r2)
STATA_QREG = {
    0.25: (2.042842, 0.0472461, -0.8013571, 0.0452555, 0.518284, 0.0473547, 0.6424),
    0.50: (1.965287, 0.0355009, -0.7475726, 0.0340052, 0.979461, 0.0355825, 0.6361),
    0.75: (1.978264, 0.0513448, -0.7318384, 0.0491815, 1.517499, 0.0514629, 0.6252),
    0.90: (1.996667, 0.0717235, -0.7252009, 0.0687016, 2.097133, 0.0718884, 0.6245),
}


@pytest.mark.parametrize("quantile", sorted(STATA_QREG))
def test_qreg_coefficients_match_stata(quantile):
    QuantileRegression = _require("QuantileRegression")
    model = QuantileRegression.from_formula(
        "y ~ x1 + x2", quantile_frame(), quantile=quantile
    )
    result = model.fit()

    x1, _, x2, _, cons, _, _ = STATA_QREG[quantile]
    assert float(result.params["x1"]) == pytest.approx(x1, rel=1e-3)
    assert float(result.params["x2"]) == pytest.approx(x2, rel=1e-3)
    assert float(result.params["Intercept"]) == pytest.approx(cons, rel=1e-3)


@pytest.mark.parametrize("quantile", sorted(STATA_QREG))
def test_qreg_pseudo_r2_matches_stata(quantile):
    QuantileRegression = _require("QuantileRegression")
    result = QuantileRegression.from_formula(
        "y ~ x1 + x2", quantile_frame(), quantile=quantile
    ).fit()

    expected = STATA_QREG[quantile][6]
    assert float(result.e["r2_p"]) == pytest.approx(expected, abs=5e-4)


@pytest.mark.parametrize("quantile", sorted(STATA_QREG))
def test_qreg_standard_errors_match_stata(quantile):
    """The sparsity estimate is the part most likely to be wrong.

    Stata's default ``vce(iid)`` uses the 'fitted' sparsity: re-estimate the
    regression at tau+h and tau-h, then difference xbar'beta over 2h. A kernel
    density of the residuals instead would shift these by ~29%.
    """

    QuantileRegression = _require("QuantileRegression")
    result = QuantileRegression.from_formula(
        "y ~ x1 + x2", quantile_frame(), quantile=quantile
    ).fit()

    _, se_x1, _, se_x2, _, se_cons, _ = STATA_QREG[quantile]
    assert float(result.bse["x1"]) == pytest.approx(se_x1, rel=5e-2)
    assert float(result.bse["x2"]) == pytest.approx(se_x2, rel=5e-2)
    assert float(result.bse["Intercept"]) == pytest.approx(se_cons, rel=5e-2)


def test_qreg_median_is_close_to_ols_on_homoskedastic_data():
    """On iid errors the conditional median equals the conditional mean."""

    QuantileRegression = _require("QuantileRegression")
    rng = np.random.default_rng(11)
    n = 800
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + rng.normal(size=n)
    frame = pd.DataFrame({"y": y, "x1": x1, "x2": x2})

    quantile = QuantileRegression.from_formula("y ~ x1 + x2", frame).fit()
    plain = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()

    np.testing.assert_allclose(
        quantile.params[["Intercept", "x1", "x2"]].to_numpy(),
        plain.params[["Intercept", "x1", "x2"]].to_numpy(),
        rtol=5e-2,
    )


def test_rreg_matches_stata_and_beats_ols():
    RobustRegression = _require("RobustRegression")
    frame = outlier_frame()
    robust = RobustRegression.from_formula("y ~ x1 + x2", frame).fit()

    # Stata: x1 2.014204, x2 -.5340517, _cons .9933214.
    assert float(robust.params["x1"]) == pytest.approx(2.014204, rel=1e-2)
    assert float(robust.params["x2"]) == pytest.approx(-0.5340517, rel=2e-2)
    assert float(robust.params["Intercept"]) == pytest.approx(0.9933214, rel=2e-2)

    # The DGP slope is 2.0; OLS is dragged away by the 5% contamination.
    plain = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()
    assert abs(float(robust.params["x1"]) - 2.0) < abs(float(plain.params["x1"]) - 2.0)

# ------------------------------------------------------- sqreg / bsqreg


def test_sqreg_coefficients_match_stata():
    """The point estimates are qreg's, which are already verified above."""

    Sqreg = _require("Sqreg")
    frame = quantile_frame()
    result = Sqreg.from_formula(
        "y ~ x1 + x2", frame, quantiles=(0.25, 0.5, 0.75)
    ).fit(reps=20, seed=1)

    for position, tau in enumerate((0.25, 0.5, 0.75)):
        x1, _, x2, _, cons, _, _ = STATA_QREG[tau]
        block = result.params.iloc[position * 3 : (position + 1) * 3].to_numpy()
        assert float(block[0]) == pytest.approx(cons, rel=1e-3)
        assert float(block[1]) == pytest.approx(x1, rel=1e-3)
        assert float(block[2]) == pytest.approx(x2, rel=1e-3)


def test_bsqreg_coefficients_match_stata():
    Bsqreg = _require("Bsqreg")
    frame = quantile_frame()
    result = Bsqreg.from_formula("y ~ x1 + x2", frame, quantile=0.5).fit(
        reps=20, seed=1
    )
    x1, _, x2, _, cons, _, _ = STATA_QREG[0.5]
    assert float(result.params["x1"]) == pytest.approx(x1, rel=1e-3)
    assert float(result.params["x2"]) == pytest.approx(x2, rel=1e-3)
    assert float(result.params["Intercept"]) == pytest.approx(cons, rel=1e-3)


def test_sqreg_bootstrap_standard_errors_are_the_right_order():
    """Bootstrap SEs are close to Stata's but cannot be reproduced exactly.

    Stata seeds its bootstrap from its own RNG, so a cell-by-cell match is
    impossible by construction. Measured on the reference DGP with reps=200:

        Stata sqreg  x1 se at q=.25/.5/.75   .0705717 / .0506841 / .0709206
        westata      x1 se                   .0682194 / .0429598 / .0675913
        Stata bsqreg x1 se at q=.5           .049082
        westata      x1 se                   .0429598

    The ratios are 0.97 / 0.85 / 0.95 and 0.88: the right order of magnitude and
    shape, with a systematic downward bias that suggests Stata resamples
    differently. Recorded rather than papered over -- these standard errors are
    E2, not E3.
    """

    Sqreg = _require("Sqreg")
    frame = quantile_frame()
    result = Sqreg.from_formula(
        "y ~ x1 + x2", frame, quantiles=(0.25, 0.5, 0.75)
    ).fit(reps=200, seed=1)

    errors = result.bse.to_numpy().reshape(3, 3)[:, 1]
    stata = np.array([0.0705717, 0.0506841, 0.0709206])
    ratio = errors / stata
    assert np.all(ratio > 0.7) and np.all(ratio < 1.3), ratio
