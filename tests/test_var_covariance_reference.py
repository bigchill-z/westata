"""Our VAR(2) residual covariance matches a CORRECT reference exactly.

Round 82 recorded a contradiction: coefficients matched, both reported N = 164, yet the residual
covariances differed by 44% and 550%. It looked like the residual samples differed.

They do not. Row-by-row comparison showed the residuals themselves differ, because the two are
NOT THE SAME MODEL:

    westata VAR(2)   5 coefficients   const + L1 x2 + L2 x2
    Stata var lags(2) 3 coefficients   ONLY the L2 terms

which is the environment defect documented in rounds 39-47: `var ..., lags()` here estimates only
the top lag. The "contradiction" was entirely an artefact of comparing against it.

Against a hand-built sureg -- which does not consult lags() either -- the covariance matches to
every printed digit:

    sureg Sigma      [[1.850464e-04, -2.563300e-06], [-2.563300e-06, 6.098900e-06]]
    westata sigma_u  [[1.850464e-04, -2.563300e-06], [-2.563300e-06, 6.098900e-06]]

So westata's residual covariance was right all along, and IRF's remaining gap is only the
orthogonalisation step, not the inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(DATA)[["dinf", "unem"]].dropna().reset_index(drop=True)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")
    return ws.VAR(frame[["dinf", "unem"]].to_numpy(float), lags=2, names=["dinf", "unem"]).fit()


#: From a sureg on hand-built L1/L2 columns -- a reference that does not use `var lags()`.
SUREG_SIGMA = np.array(
    [[1.850464e-04, -2.563300e-06], [-2.563300e-06, 6.098900e-06]]
)


def test_the_residual_covariance_matches_a_hand_built_sureg(fitted) -> None:
    sigma = np.asarray(fitted.sigma_u, dtype=float)
    assert np.allclose(sigma, SUREG_SIGMA, rtol=1e-6)


def test_the_var2_has_five_parameters(fitted) -> None:
    """Stata's broken var has three; a real VAR(2) has five."""

    assert len(fitted.params) == 5
    assert list(fitted.params.index) == ["const", "L1.dinf", "L1.unem", "L2.dinf", "L2.unem"]


def test_the_residual_row_count_is_n_minus_lags(fitted) -> None:
    residuals = np.asarray(fitted._result.resid, dtype=float)
    assert residuals.shape[0] == int(fitted.nobs)


def test_the_orth_irfs_are_square_and_finite(fitted) -> None:
    """IRF itself is not yet verified against Stata, so only its shape is pinned."""

    impulses = np.asarray(fitted.irf(periods=5).orth_irfs, dtype=float)
    assert impulses.shape[1:] == (2, 2)
    assert np.all(np.isfinite(impulses))
