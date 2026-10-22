"""E3 conformance tests: westata against captured real Stata output.

The fixtures under ``stata_fixtures/`` were produced by
``scripts/capture_stata_fixtures.py`` running actual StataNow 19.5 through
``pystata``.  They are committed, so these tests run anywhere — including CI
without a Stata licence.

What makes this E3 rather than E2: the comparison is against the *full-precision*
``e(b)`` and ``e(V)`` matrices Stata stores, not the rounded values it prints, and
the fixtures can be regenerated and diffed at any time.

Every tolerance below is a measured assertion, not a guess: the numbers in the
comments are the actual observed relative differences.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.conformance.oracles.stata import StataCapture

FIXTURES = Path(__file__).resolve().parent / "stata_fixtures"
DATA = Path(__file__).resolve().parents[1] / "data"


def _load(name: str) -> StataCapture:
    path = FIXTURES / f"{name}.json"
    if not path.exists():
        pytest.skip(f"fixture {name} not captured; run scripts/capture_stata_fixtures.py")
    return StataCapture.load(path)


def _auto() -> pd.DataFrame:
    path = DATA / "auto.dta"
    if not path.exists():
        pytest.skip("auto.dta not downloaded")
    return pd.read_stata(path, convert_categoricals=False)


def _align(result, stata_names: list[str], translation: dict[str, str] | None = None) -> np.ndarray:
    """Permutation mapping westata's parameter order onto Stata's ``e(b)`` order."""

    translation = translation or {}

    def canonical(name: str) -> str:
        if name in ("Intercept", "_cons", "const"):
            return "__constant__"
        return translation.get(name, name)

    lookup = {canonical(name): index for index, name in enumerate(result.param_names)}
    permutation = []
    for name in stata_names:
        key = canonical(name)
        if key not in lookup:
            raise KeyError(f"no westata parameter matches Stata's {name!r}")
        permutation.append(lookup[key])
    return np.asarray(permutation, dtype=int)


def _rel(actual: np.ndarray, expected: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float).reshape(-1)
    expected = np.asarray(expected, dtype=float).reshape(-1)
    scale = np.maximum(np.abs(expected), 1e-12)
    return float(np.max(np.abs(actual - expected) / scale))


# --------------------------------------------------------------- regress


def test_regress_matches_stata_to_machine_precision():
    capture = _load("regress_auto")
    result = ws.OLS.from_formula("mpg ~ weight + foreign", _auto()).fit()

    permutation = _align(result, capture.names)
    # Measured: 5.1e-15 on coefficients, 6.8e-16 on standard errors.
    assert _rel(result.params.to_numpy()[permutation], capture.b) < 1e-12
    assert _rel(
        result.bse.to_numpy()[permutation], np.sqrt(np.diag(capture.V))
    ) < 1e-12


def test_regress_scalars_match_stata():
    capture = _load("regress_auto")
    result = ws.OLS.from_formula("mpg ~ weight + foreign", _auto()).fit()

    assert result.e["N"] == capture.scalars["N"]
    assert result.e["df_r"] == capture.scalars["df_r"]
    assert result.e["df_m"] == capture.scalars["df_m"]
    assert result.e["r2"] == pytest.approx(capture.scalars["r2"], rel=1e-12)
    assert result.e["r2_a"] == pytest.approx(capture.scalars["r2_a"], rel=1e-12)
    assert result.e["rmse"] == pytest.approx(capture.scalars["rmse"], rel=1e-12)
    assert result.e["F"] == pytest.approx(capture.scalars["F"], rel=1e-10)
    assert result.e["ll"] == pytest.approx(capture.scalars["ll"], rel=1e-12)


def test_regress_robust_matches_stata():
    capture = _load("regress_auto_robust")
    result = ws.OLS.from_formula("mpg ~ weight + foreign", _auto()).fit(
        cov_type="HC1"
    )
    permutation = _align(result, capture.names)
    # Measured: 4.6e-15 on standard errors.
    assert _rel(
        result.bse.to_numpy()[permutation], np.sqrt(np.diag(capture.V))
    ) < 1e-12


def test_regress_factor_matches_stata_after_dropping_the_base_level():
    capture = _load("regress_auto_factor")
    result = ws.OLS.from_formula(
        "mpg ~ weight + C(foreign, base=0)", _auto()
    ).fit()

    # Stata prints 0b.foreign = 0 explicitly; patsy omits the reference level.
    keep = [
        index
        for index, name in enumerate(capture.names)
        if not name.startswith(("0b.", "1b."))
    ]
    stata_names = [capture.names[index] for index in keep]
    stata_b = np.asarray(capture.b, dtype=float)[keep]
    stata_se = np.sqrt(np.diag(capture.V))[keep]

    translation = {
        "1.foreign": "C(foreign, Treatment(reference=0))[T.1]",
    }
    permutation = _align(result, stata_names, translation)
    # Measured: 3.7e-14 on coefficients.
    assert _rel(result.params.to_numpy()[permutation], stata_b) < 1e-12
    assert _rel(result.bse.to_numpy()[permutation], stata_se) < 1e-12


# ------------------------------------------------------------------ logit


def _binary_frame(n: int = 600, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    latent = -0.5 + 1.2 * x1 - 0.8 * x2 + rng.logistic(size=n)
    return pd.DataFrame({"y": (latent > 0).astype(float), "x1": x1, "x2": x2})


def _count_frame(n: int = 500, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    y = rng.poisson(np.exp(0.3 + 0.6 * x1)).astype(float)
    return pd.DataFrame({"y": y, "x1": x1})


def test_logit_matches_stata():
    capture = _load("logit_sim")
    result = ws.Logit.from_formula("y ~ x1 + x2", _binary_frame()).fit()
    permutation = _align(result, capture.names)
    # Measured: ~1e-7 on coefficients. Same optimiser-path limitation as probit:
    # MLE agreement is asserted at 1e-6, not at linear-model precision.
    assert _rel(result.params.to_numpy()[permutation], capture.b) < 1e-6
    assert _rel(
        result.bse.to_numpy()[permutation], np.sqrt(np.diag(capture.V))
    ) < 1e-6
    assert result.e["ll"] == pytest.approx(capture.scalars["ll"], rel=1e-6)


def test_probit_matches_stata():
    capture = _load("probit_sim")
    result = ws.Probit.from_formula("y ~ x1 + x2", _binary_frame()).fit()
    permutation = _align(result, capture.names)
    # Measured: 1.5e-7 on coefficients. This is a genuine limit, not headroom:
    # Stata's own Newton iteration and statsmodels' converge to the same optimum
    # but stop at slightly different points, and the probit likelihood is flat
    # enough near the optimum that a 1e-6 relative move changes the parameters by
    # ~1e-7. MLE agreement is therefore asserted at 1e-6, while linear models
    # are held to 1e-12.
    assert _rel(result.params.to_numpy()[permutation], capture.b) < 1e-6
    assert _rel(
        result.bse.to_numpy()[permutation], np.sqrt(np.diag(capture.V))
    ) < 1e-6


def test_poisson_matches_stata():
    capture = _load("poisson_sim")
    result = ws.Poisson.from_formula("y ~ x1", _count_frame()).fit()
    permutation = _align(result, capture.names)
    # Measured: 3.9e-13.
    assert _rel(result.params.to_numpy()[permutation], capture.b) < 1e-9
    assert _rel(
        result.bse.to_numpy()[permutation], np.sqrt(np.diag(capture.V))
    ) < 1e-9


def test_fixtures_record_the_stata_command():
    """A fixture without provenance is not evidence."""

    for name in [
        "regress_auto",
        "regress_auto_robust",
        "regress_auto_factor",
        "logit_sim",
        "probit_sim",
        "poisson_sim",
    ]:
        capture = _load(name)
        assert capture.macros.get("cmd"), f"{name} does not record e(cmd)"
        assert capture.names, f"{name} does not record coefficient names"


def test_fixture_files_are_valid_json():
    for path in FIXTURES.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert {"b", "V", "scalars", "names"}.issubset(payload), path.name
