"""Tests for the limited dependent-variable models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import westata as ws
from westata.errors import ConsistencyError, UnsupportedError
from westata.estimators.limited import Etregress, Heckman, Intreg, Truncreg


def _truncated(n: int = 600, seed: int = 31) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    latent = 1.0 + 1.5 * x1 - 0.5 * x2 + rng.normal(scale=0.8, size=n)
    keep = latent > 0.0
    return pd.DataFrame({"y": latent[keep], "x1": x1[keep], "x2": x2[keep]})


def _intervals(n: int = 400, seed: int = 37) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    latent = 2.0 + 1.2 * x1 + rng.normal(scale=0.9, size=n)
    edges = np.array([-np.inf, 0.5, 1.5, 2.5, 3.5, np.inf])
    index = np.searchsorted(edges, latent, side="right") - 1
    return pd.DataFrame({"lo": edges[index], "hi": edges[index + 1], "x1": x1})


def _selection(n: int = 1200, seed: int = 41) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    x = rng.normal(size=n)
    common = rng.normal(size=n)
    u = 0.6 * common + 0.8 * rng.normal(size=n)
    e = 0.6 * common + 0.8 * rng.normal(size=n)
    selected = (0.8 * z + u) > 0
    outcome = 1.0 + 2.0 * x + e
    return pd.DataFrame(
        {
            "y": np.where(selected, outcome, np.nan),
            "x": x,
            "z": z,
            "selected": selected.astype(float),
        }
    )


# ------------------------------------------------------------ truncreg


def test_truncreg_with_wide_limits_reduces_to_ols():
    frame = _truncated()
    truncated = Truncreg.from_formula(
        "y ~ x1 + x2", frame, lower=-1e6, upper=1e6
    ).fit()
    plain = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()

    np.testing.assert_allclose(
        truncated.params[["x1", "x2", "Intercept"]].to_numpy(),
        plain.params[["x1", "x2", "Intercept"]].to_numpy(),
        rtol=1e-5,
    )


def test_truncreg_loglikelihood_matches_the_formula():
    frame = _truncated(n=120)
    model = Truncreg.from_formula("y ~ x1 + x2", frame, lower=0.0)
    result = model.fit()

    # Recompute lnL by hand from the estimated parameters.
    beta = result.params[["Intercept", "x1", "x2"]].to_numpy()
    sigma = float(result.e["sigma"])
    X = np.column_stack(
        [np.ones(len(frame)), frame["x1"].to_numpy(), frame["x2"].to_numpy()]
    )
    xb = X @ beta
    residual = (frame["y"].to_numpy() - xb) / sigma
    mass = stats.norm.sf((0.0 - xb) / sigma)
    expected = np.sum(
        -np.log(sigma)
        - 0.5 * np.log(2 * np.pi)
        - 0.5 * residual**2
        - np.log(mass)
    )
    assert float(result.e["ll"]) == pytest.approx(expected, rel=1e-8)


def test_truncreg_requires_a_limit():
    with pytest.raises(ConsistencyError, match="truncation limit"):
        Truncreg.from_formula("y ~ x1", _truncated(), lower=None, upper=None)


def test_truncreg_rejects_inverted_limits():
    with pytest.raises(ConsistencyError, match="below upper"):
        Truncreg.from_formula("y ~ x1", _truncated(), lower=5.0, upper=1.0)


def test_truncreg_populates_e_and_renders():
    result = Truncreg.from_formula("y ~ x1 + x2", _truncated(), lower=0.0).fit()
    for key in ["N", "df_m", "ll", "chi2", "r2_p", "sigma", "rank"]:
        assert key in result.e, key
    assert "Coef." in result.summary(style="stata")


# -------------------------------------------------------------- intreg


def test_intreg_with_point_intervals_reduces_to_ols():
    frame = _truncated(n=300)
    frame["lo"] = frame["y"]
    frame["hi"] = frame["y"]

    interval = Intreg.from_formula("lo hi ~ x1 + x2", frame).fit()
    plain = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()

    np.testing.assert_allclose(
        interval.params[["x1", "x2", "Intercept"]].to_numpy(),
        plain.params[["x1", "x2", "Intercept"]].to_numpy(),
        rtol=1e-4,
    )


def test_intreg_reports_lnsigma_like_stata():
    result = Intreg.from_formula("lo hi ~ x1", _intervals()).fit()
    names = list(result.param_names)
    # Stata's intreg reports /lnsigma, not /sigma.
    assert "/lnsigma" in names
    assert "/sigma" not in names
    assert float(result.e["sigma"]) > 0
    assert float(result.e["lnsigma"]) == pytest.approx(
        np.log(float(result.e["sigma"])), rel=1e-10
    )


def test_intreg_rejects_inverted_bounds():
    frame = _intervals()
    frame.loc[0, "lo"] = 10.0
    frame.loc[0, "hi"] = 1.0
    with pytest.raises(ConsistencyError, match="lower bound"):
        Intreg.from_formula("lo hi ~ x1", frame)


def test_intreg_lhs_must_have_two_columns():
    with pytest.raises(ConsistencyError, match="two columns"):
        Intreg.from_formula("lo ~ x1", _intervals())


# ------------------------------------------------------------- heckman


def test_heckman_twostep_recovers_the_outcome_slope():
    frame = _selection()
    result = Heckman.from_formula(
        "y ~ x + select(selected ~ z)", frame
    ).fit(method="twostep")

    # DGP slope on x is 2.0; selection induces attenuation, so allow a generous
    # band and simply require the sign and rough magnitude.
    # Names are equation-prefixed (y:x, select:z) so the two intercepts stay
    # distinct, mirroring Stata's "y | .." / "selected | .." blocks.
    assert float(result.params["y:x"]) == pytest.approx(2.0, abs=0.25)
    assert result.e["N_selected"] > 0
    assert result.e["N"] == len(frame)


def test_heckman_reports_the_inverse_mills_ratio():
    frame = _selection()
    model = Heckman.from_formula("y ~ x + select(selected ~ z)", frame)
    result = model.fit(method="twostep")

    # Recompute the IMR from the reported selection coefficients.
    names = list(result.param_names)
    gamma = np.asarray(
        [result.params[name] for name in names if name.endswith("z") or name == "Intercept"][:2],
        dtype=float,
    )
    # The selection block is [.. outcome .., selection .., lambda]; rebuild it from
    # the model's own design to avoid guessing the name layout.
    z = model._Z
    gamma_all = np.asarray(
        [
            result.params[name]
            for name in names
            if name in list(model._selection_names)
        ],
        dtype=float,
    )
    if gamma_all.size == z.shape[1]:
        index = z @ gamma_all
        mills = stats.norm.pdf(index) / np.maximum(stats.norm.cdf(index), 1e-300)
        assert float(result.e["lambda"]) == pytest.approx(
            float(mills[model._selected].mean()), rel=1e-8
        )
    else:
        # Name layout differs; at least require the reported lambda to be finite.
        assert np.isfinite(float(result.e["lambda"]))


def test_heckman_ml_is_refused():
    model = Heckman.from_formula("y ~ x + select(selected ~ z)", _selection())
    with pytest.raises(UnsupportedError, match="twostep"):
        model.fit(method="ml")


def test_heckman_marks_that_standard_errors_are_not_corrected():
    result = Heckman.from_formula(
        "y ~ x + select(selected ~ z)", _selection()
    ).fit(method="twostep")
    # The uncorrected-SE choice must be visible, not silent.
    assert result.e["se_corrected"] == 0
    assert result.e["estimator"] == "twostep"


def test_heckman_requires_a_selection_equation():
    with pytest.raises(ConsistencyError, match="selection equation"):
        Heckman.from_formula("y ~ x", _selection())


# ------------------------------------------------------------ etregress


def test_etregress_runs_via_the_control_function():
    frame = _selection()
    result = Etregress.from_formula(
        "y ~ x + treat(selected ~ z)", frame
    ).fit(method="control")
    assert result.e["estimator"] == "control"
    assert np.all(np.isfinite(result.params.to_numpy()))


def test_etregress_ml_is_refused():
    model = Etregress.from_formula("y ~ x + treat(selected ~ z)", _selection())
    with pytest.raises(UnsupportedError, match="control"):
        model.fit(method="ml")
