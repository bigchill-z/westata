"""E2 conformance: ``ws.synth`` against six full-precision Stata captures.

The fixtures were captured from live StataNow 19.5 / ``synth.ado`` 0.0.8 by
``scripts/capture_synth_fixtures.py`` (P0).  They are committed, so this suite
runs WITHOUT a Stata licence.

GRADE: E2 -- bounded, measured differences.  E3 is explicitly excluded:
``synthopt.plugin`` is not bit-deterministic (two identical Stata runs differ
by 1.471e-11 in RMSPE), and the QP is rank-deficient (rank(H) <= K << J), so
the exact W on the optimal face is solver-dependent.  Every tolerance below
was MEASURED against these fixtures before being written here; see
``docs/superpowers/specs/2026-11-09-synth-b-design.md`` §5 and CHANGELOG.

Measured bounds across all six fixtures (worst case)::

    |dV|        <= 1e-10   (formula solved exactly; observed ~1e-13)
    |dW|        <= 1e-3    (e(W_weights) is rounded to 3 dp by roundmat)
    |dX_balance|<= 1e-3    (derived from the rounded W)
    |dY_*|      <= 1e-3    (derived from the rounded W)
    |dRMSPE|    <= 1e-2    (rank-deficient QP: the C++ plugin stops at
                            margin=0.005 off the optimal face; observed
                            worst 5.6e-4 on smoking_default)

The RMSPE bound is deliberately WIDER than the P1 design's provisional 1e-8:
measurement falsified that number, and the design requires "先测后写".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from westata.estimators.synth import synth

FIXTURES = Path(__file__).resolve().parent / "stata_fixtures"
REPO = Path(__file__).resolve().parents[2]
#: The fixtures were captured from ``_stata_pkgs/verify_data/smoking.dta``, an
#: untracked helper directory that is not shipped with the repo.  The SAME
#: Stata file (identical bytes / V / RMSPE) is available in the local reference
#: data tree; fall back to it, and skip -- never fabricate -- when neither is
#: present.  Basque was always external-only (recorded in the fixture metadata).
SMOKING = REPO / "_stata_pkgs" / "verify_data" / "smoking.dta"
SMOKING_FALLBACK = Path(r"D:\myCodeProject\test_westata\data\smoking.dta")
BASQUE = Path(r"D:\myCodeProject\test_westata\data\basque.dta")

MATRIX_NAMES = {
    "Y_treated", "Y_synthetic", "W_weights", "X_balance", "V_matrix", "RMSPE",
}

# Tolerances: measured, not guessed.  See module docstring.
ATOL_V = 1e-10
ATOL_W = 1e-3
ATOL_DERIVED = 1e-3
ATOL_RMSPE = 1e-2


def _fixture(name: str) -> dict:
    path = FIXTURES / f"{name}.json"
    if not path.exists():  # pragma: no cover
        pytest.skip(f"fixture {name} not captured")
    return json.loads(path.read_text(encoding="utf-8"))


def _smoking() -> pd.DataFrame:
    path = SMOKING if SMOKING.exists() else SMOKING_FALLBACK
    if not path.exists():  # pragma: no cover
        pytest.skip("smoking.dta is not available")
    return pd.read_stata(path, convert_categoricals=False)


def _basque() -> pd.DataFrame:
    if not BASQUE.exists():  # pragma: no cover
        pytest.skip("basque.dta is not available")
    return pd.read_stata(BASQUE, convert_categoricals=False)


# Exact Stata command lines from the fixture metadata, transcribed as kwargs.
SCENARIOS = [
    ("synth_smoking_default", "smoking",
     dict(depvar="cigsale",
          predictors=["lnincome", "age15to24", "beer", "retprice"],
          trunit=3, trperiod=1989)),
    ("synth_smoking_customV", "smoking",
     dict(depvar="cigsale",
          predictors=["lnincome", "age15to24", "beer", "retprice"],
          trunit=3, trperiod=1989, customV=[1.0, 1.0, 1.0, 1.0])),
    ("synth_smoking_mspe", "smoking",
     dict(depvar="cigsale",
          predictors=["lnincome", "age15to24", "beer", "retprice"],
          trunit=3, trperiod=1989, mspeperiod=list(range(1980, 1989)))),
    ("synth_smoking_xperiod", "smoking",
     dict(depvar="cigsale",
          predictors=["lnincome", "age15to24", "beer", "retprice"],
          trunit=3, trperiod=1989, xperiod=list(range(1980, 1989)))),
    ("synth_smoking_counit", "smoking",
     dict(depvar="cigsale",
          predictors=["lnincome", "age15to24", "beer", "retprice"],
          trunit=3, trperiod=1989, counit=list(range(1, 21)))),
    ("synth_basque_default", "basque",
     dict(depvar="gdpcap",
          predictors=["invest", "popdens", "sec_agriculture",
                      "school_illit", "school_med"],
          trunit=17, trperiod=1975)),
]


def _run(name: str, dataset: str, kwargs: dict):
    if dataset == "smoking":
        df = _smoking()
        unit, time = "state", "year"
    else:
        df = _basque()
        unit, time = "regionno", "year"
    return synth(df, unit=unit, time=time, **kwargs)


@pytest.mark.parametrize("name,dataset,kwargs", SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_all_six_matrices_are_present(name, dataset, kwargs) -> None:
    payload = _fixture(name)
    assert set(payload["matrices"]) == MATRIX_NAMES


@pytest.mark.parametrize("name,dataset,kwargs", SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_v_matrix_matches_stata(name, dataset, kwargs) -> None:
    """The solved regression-based V reproduces e(V_matrix) to ~1e-13."""

    expected = np.asarray(_fixture(name)["matrices"]["V_matrix"], dtype=float)
    got = np.asarray(_run(name, dataset, kwargs).V_matrix, dtype=float)
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, atol=ATOL_V, rtol=0)


@pytest.mark.parametrize("name,dataset,kwargs", SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_unit_weights_match_stata(name, dataset, kwargs) -> None:
    """e(W_weights) is rounded to 3 dp by roundmat(); SLSQP lands on it exactly."""

    expected = np.asarray(_fixture(name)["matrices"]["W_weights"], dtype=float)
    got = np.asarray(_run(name, dataset, kwargs).W_weights, dtype=float)
    assert got.shape == expected.shape
    # Column 0 is the unit NUMBER; column 1 is the weight.
    np.testing.assert_allclose(got[:, 0], expected[:, 0], atol=0, rtol=0)
    np.testing.assert_allclose(got[:, 1], expected[:, 1], atol=ATOL_W, rtol=0)


@pytest.mark.parametrize("name,dataset,kwargs", SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_balance_table_matches_stata(name, dataset, kwargs) -> None:
    """X_balance uses RAW X and the ROUNDED W (synth.ado:560-561, :589-606)."""

    expected = np.asarray(_fixture(name)["matrices"]["X_balance"], dtype=float)
    got = np.asarray(_run(name, dataset, kwargs).X_balance, dtype=float)
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, atol=ATOL_DERIVED, rtol=0)


@pytest.mark.parametrize("name,dataset,kwargs", SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_trajectories_match_stata(name, dataset, kwargs) -> None:
    """Y_treated / Y_synthetic / gap follow from the rounded W."""

    mats = _fixture(name)["matrices"]
    expected_t = np.asarray(mats["Y_treated"], dtype=float)
    expected_s = np.asarray(mats["Y_synthetic"], dtype=float)
    res = _run(name, dataset, kwargs)
    np.testing.assert_allclose(res.Y_treated, expected_t, atol=ATOL_DERIVED, rtol=0)
    np.testing.assert_allclose(res.Y_synthetic, expected_s, atol=ATOL_DERIVED, rtol=0)
    np.testing.assert_allclose(res.gap, expected_t - expected_s, atol=ATOL_DERIVED, rtol=0)


@pytest.mark.parametrize("name,dataset,kwargs", SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_rmspe_is_within_the_measured_e2_bound(name, dataset, kwargs) -> None:
    """RMSPE is the ONE quantity the rank-deficient QP leaves solver-dependent.

    Stata's C++ plugin uses a 0.005 margin and stops off the optimal face;
    scipy's SLSQP solves the QP to tighter optimality.  The resulting W agrees
    to 3 dp but RMSPE differs by up to 5.6e-4.  This test records the MEASURED
    bound rather than pretending the number is reproducible (design G3).
    """

    expected = float(np.asarray(_fixture(name)["matrices"]["RMSPE"]).ravel()[0])
    got = float(np.asarray(_run(name, dataset, kwargs).RMSPE).ravel()[0])
    assert abs(got - expected) <= ATOL_RMSPE, (
        f"{name}: RMSPE {got} vs Stata {expected}, diff {got - expected}"
    )


def test_the_rmspe_gap_is_disclosed_not_hidden() -> None:
    """Guard against a future 'fix' that silently tightens the claim to E3."""

    # The design and registry must state that E3 is excluded.
    spec = (REPO / "docs" / "superpowers" / "specs" /
            "2026-11-09-synth-b-design.md").read_text(encoding="utf-8")
    assert "E3" in spec and "排除" in spec
    registry = (REPO / "tests" / "conformance" / "registry.yml").read_text(
        encoding="utf-8"
    )
    assert "E3 excluded" in registry or "E3 排除" in registry


def test_mutation_guard_the_v_kernel_is_actually_used() -> None:
    """A regression test that would go red if regsval were replaced by equal weights.

    This is the mutation check required by docs/TESTING_SOP.md:186-199 (the
    mutation itself is performed by reverting the implementation in a scratch
    run; here we pin the observable consequence, i.e. that V is NOT uniform).
    """

    payload = _fixture("synth_smoking_default")
    expected = np.diag(np.asarray(payload["matrices"]["V_matrix"], dtype=float))
    got = np.diag(np.asarray(_run("synth_smoking_default", "smoking",
                                  SCENARIOS[0][2]).V_matrix, dtype=float))
    assert not np.allclose(expected, np.full_like(expected, 1 / len(expected)))
    np.testing.assert_allclose(got, expected, atol=ATOL_V, rtol=0)
