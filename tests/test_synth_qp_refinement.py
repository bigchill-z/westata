"""RED tests: adaptive interior-point QP for near-rank-1 (degenerate) V.

Measured motivation (2026-11-11, 8 live-Stata cases):
* When V is concentrated (Vmax > 0.98), H = Xco' V Xco is effectively rank 1,
  the QP optimum face is very high-dimensional, and Stata's Vanderbei/LOQO
  interior-point plugin returns an EARLY-STOPPED iterate: one dominant weight
  plus tiny (1e-3) weights spread across nearly all controls.  scipy SLSQP
  converges to a sparse vertex instead (dW 0.176 / 0.081 on the two cases).
* A primal-dual interior-point iteration stopped when the complementarity gap
  falls below a threshold reproduces that shape (dW 0.008 / 0.003), and the
  result is stable across two orders of magnitude of the threshold.
* For Vmax <= 0.98 SLSQP already matches Stata EXACTLY and must be kept.

These tests are written BEFORE the implementation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from westata.estimators.synth import synth

FEATURES = Path(__file__).resolve().parent / "conformance" / "stata_fixtures" / "synth_features.json"
SMOKING = Path(r"D:\myCodeProject\test_westata\data\synth_smoking.dta")


def _cases() -> dict:
    if not FEATURES.exists():  # pragma: no cover
        pytest.skip("synth_features.json not captured")
    raw = json.loads(FEATURES.read_text(encoding="utf-8"))["cases"]
    return {k.split("|", 1)[-1]: v for k, v in raw.items()}


def _data() -> pd.DataFrame:
    if not SMOKING.exists():  # pragma: no cover
        pytest.skip("smoking.dta is not available")
    df = pd.read_stata(SMOKING, convert_categoricals=False)
    for col in df.columns:
        if df[col].dtype.kind in "fi":
            df[col] = df[col].astype("float64")
    return df


def _run(predictors, **extra):
    return synth(_data(), unit="state", time="year", depvar="cigsale",
                 predictors=predictors, trunit=3, trperiod=1989, **extra)


def _dW(case_key, predictors, **extra) -> float:
    case = _cases()[case_key]
    r = _run(predictors, **extra)
    Ws = np.asarray(case["W_weights"], float)[:, 1]
    Wo = np.asarray(r.W_weights[:, 1], float)
    return float(np.max(np.abs(Ws - Wo)))


def test_degenerate_single_period_case_improves_to_under_0_02() -> None:
    """Was dW=0.176; the interior-point path must get under 0.02."""

    assert _dW("var_single_1975", ["lnincome", "beer", "cigsale(1975)"]) < 0.02


def test_degenerate_numlist_case_improves_to_under_0_02() -> None:
    """Was dW=0.081; the interior-point path must get under 0.02."""

    assert _dW("var_numlist_1980_1988", ["lnincome", "cigsale(1980(1)1988)"]) < 0.02


@pytest.mark.parametrize(
    "key,predictors",
    [
        ("var_amp_1980_1988", ["lnincome", "cigsale(1980&1988)", "retprice"]),
        ("unitnames_decode", ["lnincome", "beer", "retprice"]),
        ("customV_1234", ["lnincome", "age15to24", "beer", "retprice"]),
        ("mspe_xperiod_resultsperiod", ["lnincome", "beer", "retprice"]),
    ],
)
def test_non_degenerate_cases_still_match_exactly(key, predictors) -> None:
    """The adaptive rule must NOT disturb the cases SLSQP already nails."""

    extra = ({"customV": [1, 2, 3, 4]} if key == "customV_1234" else
             {"xperiod": list(range(1980, 1989)),
              "mspeperiod": list(range(1980, 1989)),
              "resultsperiod": list(range(1985, 1996))}
             if key == "mspe_xperiod_resultsperiod" else {})
    assert _dW(key, predictors, **extra) < 1e-3


def test_result_records_which_qp_solver_was_used() -> None:
    """Auditability: the caller can see which path produced W."""

    r = _run(["lnincome", "beer", "cigsale(1975)"])
    assert getattr(r, "qp_solver", None) == "interior-point"
    r2 = _run(["lnincome", "beer", "retprice"])
    assert getattr(r2, "qp_solver", None) == "SLSQP"


def test_interior_point_result_has_the_dense_tiny_weight_shape() -> None:
    """Stata's hallmark: many tiny weights, not a sparse vertex."""

    r = _run(["lnincome", "beer", "cigsale(1975)"])
    W = np.asarray(r.W_weights[:, 1], float)
    assert (W > 0).sum() >= 20, (W > 0).sum()
    assert W.max() > 0.5
    assert np.median(W[W > 0]) < 0.02
