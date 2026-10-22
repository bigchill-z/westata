"""E3 conformance for the extended command set.

Fixtures come from ``scripts/capture_stata_fixtures_ext.py`` (StataNow 19.5 via
pystata) and are committed, so these tests need no licence.  Each tolerance
records the measured difference; where the gap is larger it is explained rather
than hidden.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.conformance.oracles.stata import StataCapture

FIXTURES = Path(__file__).resolve().parent / "stata_fixtures"


def _load(name: str) -> StataCapture:
    path = FIXTURES / f"{name}.json"
    if not path.exists():
        pytest.skip(f"{name}.json missing; run scripts/capture_stata_fixtures_ext.py")
    return StataCapture.load(path)


def _rel(actual, expected) -> float:
    a = np.asarray(actual, dtype=float).reshape(-1)
    e = np.asarray(expected, dtype=float).reshape(-1)
    if a.size != e.size:
        return float("inf")
    return float(np.max(np.abs(a - e) / np.maximum(np.abs(e), 1e-12)))


def _align(result, stata_names, translation=None) -> np.ndarray:
    translation = translation or {}

    def canon(name: str) -> str:
        if name in ("Intercept", "_cons", "const"):
            return "__c__"
        return translation.get(name, name)

    lookup = {canon(n): i for i, n in enumerate(result.param_names)}
    return np.asarray([lookup[canon(n)] for n in stata_names], dtype=int)


# ------------------------------------------------------------ panel data


def panel_frame(n_entities: int = 30, n_periods: int = 6, seed: int = 21):
    rng = np.random.default_rng(seed)
    entity = np.repeat(np.arange(n_entities), n_periods)
    time = np.tile(np.arange(1, n_periods + 1), n_entities)
    n = n_entities * n_periods
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    effect = rng.normal(size=n_entities)[entity]
    y = 1.0 + 1.5 * x1 - 0.5 * x2 + effect + rng.normal(scale=0.4, size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "firm": entity, "year": time})


def iv_frame(n: int = 500, seed: int = 11):
    rng = np.random.default_rng(seed)
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    v = rng.normal(size=n)
    x = 0.8 * z1 + 0.5 * z2 + 0.6 * v
    y = 1.0 + 2.0 * x + 0.5 * x2 + rng.normal(size=n)
    return pd.DataFrame({"y": y, "x": x, "x2": x2, "z1": z1, "z2": z2})


def nbreg_frame(n: int = 600, seed: int = 31):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    mu = np.exp(0.3 + 0.5 * x1)
    lam = rng.gamma(shape=1.0 / 0.4, scale=mu * 0.4)
    y = rng.poisson(lam).astype(float)
    return pd.DataFrame({"y": y, "x1": x1})


def ordered_frame(n: int = 700, seed: int = 7):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    latent = 0.8 * x1 - 0.4 * x2 + rng.normal(size=n)
    y = pd.cut(latent, bins=[-np.inf, -0.7, 0.4, np.inf], labels=[1, 2, 3]).astype(float)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def var_frame(n: int = 400, seed: int = 41):
    rng = np.random.default_rng(seed)
    a = np.array([[0.5, 0.1], [0.2, 0.4]])
    data = np.zeros((n, 2))
    shocks = rng.normal(size=(n, 2))
    for t in range(1, n):
        data[t] = a @ data[t - 1] + shocks[t]
    return pd.DataFrame({"y1": data[:, 0], "y2": data[:, 1], "t": np.arange(1, n + 1)})


# ------------------------------------------------------------- xtreg


def test_xtreg_fe_matches_stata():
    capture = _load("xtreg_fe")
    result = ws.PanelOLS.from_formula(
        "y ~ x1 + x2", panel_frame(), entity="firm", time="year"
    ).fit()
    perm = _align(result, capture.names)
    # Measured: 1.8e-16 parameters, 1.9e-16 standard errors.
    assert _rel(result.params.to_numpy()[perm], capture.b) < 1e-12
    assert _rel(result.bse.to_numpy()[perm], np.sqrt(np.diag(capture.V))) < 1e-12
    # Stata's xtreg, fe df_r = N - K - n_entities + 1.
    assert result.e["df_r"] == capture.scalars["df_r"]
    assert result.e["r2"] == pytest.approx(capture.scalars["r2"], rel=1e-12)


def test_xtreg_re_matches_stata():
    capture = _load("xtreg_re")
    result = ws.RandomEffects.from_formula(
        "y ~ x1 + x2", panel_frame(), entity="firm", time="year"
    ).fit()
    perm = _align(result, capture.names)
    # Measured: 6.8e-16 parameters, 1.0e-15 standard errors. Swamy-Arora matches.
    assert _rel(result.params.to_numpy()[perm], capture.b) < 1e-12
    assert _rel(result.bse.to_numpy()[perm], np.sqrt(np.diag(capture.V))) < 1e-12


def test_xtreg_be_matches_stata():
    capture = _load("xtreg_be")
    result = ws.BetweenOLS.from_formula(
        "y ~ x1 + x2", panel_frame(), entity="firm", time="year"
    ).fit()
    perm = _align(result, capture.names)
    # Measured: 3.6e-16.
    assert _rel(result.params.to_numpy()[perm], capture.b) < 1e-12
    assert _rel(result.bse.to_numpy()[perm], np.sqrt(np.diag(capture.V))) < 1e-12


def test_areg_matches_stata():
    capture = _load("areg")
    result = ws.FEOLS.from_formula("y ~ x1 + x2 | firm", panel_frame()).fit()

    # ``areg`` reports the absorbed effect's constant as ``_cons``, which an FE
    # model with an absorbed firm effect does not identify separately; westata
    # reports only the slopes for an absorbed model, so compare those.
    slope_names = [name for name in capture.names if name != "_cons"]
    positions = [capture.names.index(name) for name in slope_names]
    stata_b = np.asarray(capture.b, dtype=float)[positions]
    stata_se = np.sqrt(np.diag(capture.V))[positions]

    perm = _align(result, slope_names)
    # Measured: 4.6e-16 parameters, 1.8e-16 standard errors.
    assert _rel(result.params.to_numpy()[perm], stata_b) < 1e-12
    assert _rel(result.bse.to_numpy()[perm], stata_se) < 1e-12
    assert result.e["df_r"] == capture.scalars["df_r"]


# --------------------------------------------------------------- IV


def test_iv_liml_matches_stata():
    capture = _load("iv_liml")
    result = ws.IVLIML.from_formula(
        "y ~ x2 + [x ~ z1 + z2]", iv_frame()
    ).fit()
    perm = _align(result, capture.names)
    # Measured: 3.7e-16 on both parameters and standard errors.
    assert _rel(result.params.to_numpy()[perm], capture.b) < 1e-12
    assert _rel(result.bse.to_numpy()[perm], np.sqrt(np.diag(capture.V))) < 1e-12


def test_iv_gmm_coefficients_match_but_not_robust_standard_errors():
    capture = _load("iv_gmm")
    result = ws.IVGMM.from_formula("y ~ x2 + [x ~ z1 + z2]", iv_frame()).fit()
    perm = _align(result, capture.names)

    # Point estimates are exact (4.9e-16) ...
    assert _rel(result.params.to_numpy()[perm], capture.b) < 1e-12
    # ... but the robust standard errors differ by ~2.9e-2 (x2: 4.50e-2 vs
    # 4.38e-2). GMM's two-step weight matrix and the finite-sample scaling of
    # Stata's e(V) are not the same construction as linearmodels'. Recorded
    # rather than tuned: the point estimates agree, the inference does not.
    assert _rel(result.bse.to_numpy()[perm], np.sqrt(np.diag(capture.V))) < 5e-2


# ------------------------------------------------------------ count / ordered


def test_nbreg_matches_stata_including_lnalpha():
    capture = _load("nbreg")
    result = ws.NegativeBinomial.from_formula("y ~ x1", nbreg_frame()).fit()

    permutation = _align(result, capture.names, {"lnalpha": "/lnalpha"})
    # Measured: 1.7e-5 parameters (optimiser path), 9.1e-7 standard errors.
    assert _rel(result.params.to_numpy()[permutation], capture.b) < 1e-4
    assert _rel(result.bse.to_numpy()[permutation], np.sqrt(np.diag(capture.V))) < 1e-5
    assert result.e["ll"] == pytest.approx(capture.scalars["ll"], rel=1e-9)


def test_ologit_matches_stata_cut_points():
    capture = _load("ologit")
    result = ws.OrderedLogit.from_formula("y ~ x1 + x2", ordered_frame()).fit()

    # Stata names the cuts ``cut1``/``cut2``; westata uses ``/cut1``/``/cut2``.
    permutation = _align(result, capture.names, {"cut1": "/cut1", "cut2": "/cut2"})
    # Measured: 1.2e-5 slopes, 2.9e-5 cuts. An earlier version reported
    # statsmodels' log-increment parameterisation as if it were the cut point,
    # making cut2 wrong by 20%; the transform is what fixes it.
    assert _rel(result.params.to_numpy()[permutation], capture.b) < 1e-4
    assert result.e["ll"] == pytest.approx(capture.scalars["ll"], rel=1e-9)


def test_oprobit_matches_stata_cut_points():
    capture = _load("oprobit")
    result = ws.OrderedProbit.from_formula("y ~ x1 + x2", ordered_frame()).fit()
    permutation = _align(result, capture.names, {"cut1": "/cut1", "cut2": "/cut2"})
    # Measured: 6.9e-6 slopes, 1.5e-5 cuts.
    assert _rel(result.params.to_numpy()[permutation], capture.b) < 1e-4


# ----------------------------------------------------------------- var


def test_var_matches_stata():
    capture = _load("var")
    frame = var_frame()
    data = frame[["y1", "y2"]].to_numpy()
    result = ws.VAR(data, lags=1, names=["y1", "y2"]).fit()

    # Stata orders e(b) equation-by-equation with repeated names:
    # [L.y1, L.y2, _cons, L.y1, L.y2, _cons].
    lag_positions = [
        i for i, name in enumerate(capture.names) if name.startswith("L.")
    ]
    matrix = result.params.loc[["L1.y1", "L1.y2"]].to_numpy()
    reordered = np.concatenate([matrix[:, j] for j in range(matrix.shape[1])])

    # Measured: 2.9e-15. Both fit by OLS equation by equation, so this is exact.
    assert _rel(reordered, np.asarray(capture.b, dtype=float)[lag_positions]) < 1e-12
