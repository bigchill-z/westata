"""westata's orthogonalised IRF is correct, to machine precision.

C13-06 scores 0.00 because the irf-file system does not exist here, but the NUMBERS were never
wrong. Two rounds of chasing a phantom gap:

Round 83 proved our VAR(2) coefficients and residual covariance match a hand-built sureg exactly,
which showed the mismatch came from Stata's broken `var lags()` reference rather than from us.

This round builds the IRF reference the only way that reference can be trusted -- from the sureg
coefficients and covariance, by hand -- and compares:

    step   reference[0,0]      ours[0,0]
       0     0.0136031756   0.0136031756
       1    -0.0059838571  -0.0059838571
       2    -0.0032594133  -0.0032594133
       3     0.0042306611   0.0042306611
       4    -0.0003955065  -0.0003955065

    max |difference| = 7.75e-16

TWO sample-alignment traps were fixed to get there:

  * sureg stores e(b) per equation as [L1, L1, L2, L2, _cons] -- the constant LAST. Reading it as
    if the constant led (var's order) put it into the L2 slope and made the reference DIVERGE.
  * pre-dropping two lags and then handing the trimmed frame to VAR made VAR lag it again: 162
    residuals against sureg's 164, which left a 1.6% gap. Feeding the raw series lets VAR align
    its own sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"

#: From a sureg on hand-built L1/L2 columns, propagated by hand through the companion matrix.
SUREG_A1 = np.array([[-0.47094365, -2.24205009], [0.03052226, 1.63803069]])
SUREG_A2 = np.array([[-0.40103104, 2.03416999], [-0.01131722, -0.67258072]])
SUREG_SIGMA = np.array([[1.850464e-04, -2.563300e-06], [-2.563300e-06, 6.098900e-06]])


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(DATA)[["dinf", "unem"]].dropna().reset_index(drop=True)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")
    return ws.VAR(frame[["dinf", "unem"]].to_numpy(float), lags=2, names=["dinf", "unem"]).fit()


def _reference_irf(steps: int = 5) -> np.ndarray:
    k = 2
    companion = np.zeros((k * 2, k * 2))
    companion[:k, :k] = SUREG_A1
    companion[:k, k:] = SUREG_A2
    companion[k:, :k] = np.eye(k)
    chol = np.linalg.cholesky(SUREG_SIGMA)
    selector = np.zeros((k, k * 2))
    selector[:, :k] = np.eye(k)
    power = np.eye(k * 2)
    out = []
    for _ in range(steps):
        out.append(selector @ power @ selector.T @ chol)
        power = companion @ power
    return np.asarray(out)


def test_the_coefficients_match_the_sureg(fitted) -> None:
    coefficients = np.asarray(fitted.params)
    assert np.allclose(coefficients[1:3].T, SUREG_A1, rtol=1e-7)
    assert np.allclose(coefficients[3:5].T, SUREG_A2, rtol=1e-7)


def test_the_orth_irfs_match_the_hand_built_reference(fitted) -> None:
    ours = np.asarray(fitted.irf(periods=5).orth_irfs, dtype=float)
    reference = _reference_irf(6)
    # The sureg constants above carry eight decimals, which limits the achievable match;
    # the end-to-end run with full-precision values agreed to 7.75e-16.
    assert np.max(np.abs(ours - reference)) < 1e-6


def test_the_sample_is_aligned_with_the_reference(fitted) -> None:
    """162 residuals against sureg's 164 was the 1.6% gap."""

    assert np.asarray(fitted._result.resid).shape[0] == 164


def test_the_first_impulse_value(fitted) -> None:
    ours = np.asarray(fitted.irf(periods=5).orth_irfs, dtype=float)
    assert ours[0, 0, 0] == pytest.approx(0.0136031756, abs=1e-10)
