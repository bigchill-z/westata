"""Tests for the matrices-only Stata capture path and the synth fixtures.

These tests run WITHOUT Stata: the fixtures are committed evidence artifacts
captured by ``scripts/capture_synth_fixtures.py`` against real StataNow 19.5 /
``synth.ado`` 0.0.8.  Nothing here compares against a Python estimator and no
evidence/conformance level is claimed -- that belongs to a later phase.

Covered:

* ``StataCapture`` saves/loads a matrices-only capture (no Stata needed);
* OLD fixtures that predate the ``matrices`` field still load;
* every synth fixture carries all six ``e()`` matrices with the right shapes;
* the three established smoking ground-truth numbers match within 1e-10;
* ``V_matrix`` is diagonal, nonnegative, and its diagonal sums to 1.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from westata.conformance.oracles.stata import StataCapture, StataOracle

FIXTURES = Path(__file__).resolve().parent / "conformance" / "stata_fixtures"

#: The six matrices ``synth`` saves (verbatim Stata names).  There is NO plain
#: ``e(V)`` -- the diagonal predictor-weight matrix is ``e(V_matrix)``.
MATRIX_NAMES = frozenset(
    {"Y_treated", "Y_synthetic", "W_weights", "X_balance", "V_matrix", "RMSPE"}
)

#: Committing fixtures means the tests can enumerate what was actually captured.
SYNTH_FIXTURES = (
    "synth_smoking_default",
    "synth_smoking_customV",
    "synth_smoking_mspe",
    "synth_smoking_xperiod",
    "synth_smoking_counit",
    "synth_basque_default",
)

#: Ground truth from the Lead's live Stata runs on synth_smoking.dta,
#: ``synth cigsale lnincome age15to24 beer retprice, trunit(3) trperiod(1989)``.
#: Values are cross-checks only; they are NOT evidence-level claims.
GROUND_TRUTH = {
    "synth_smoking_default": {
        "v_diag": [
            0.02017586234387460,
            0.07051094571756153,
            0.25958848171523474,
            0.64972471022332912,
        ],
        "rmspe": 10.50086835465610413,
    },
    "synth_smoking_customV": {
        "v_diag": [0.25, 0.25, 0.25, 0.25],
        "rmspe": 7.89695568405810899,
    },
    "synth_smoking_mspe": {
        "v_diag": [
            0.00817867291892540,
            0.07983644228549265,
            0.22995854230012880,
            0.68202634249545313,
        ],
        "rmspe": 14.43602960078125363,
    },
}
ATOL = 1e-10


def _fixture(name: str) -> Path:
    return FIXTURES / f"{name}.json"


def _matrices(name: str) -> dict:
    return json.loads(_fixture(name).read_text(encoding="utf-8"))["matrices"]


# -------------------------------------------------------------------- task 1a
def test_matrices_only_capture_round_trips_without_stata(tmp_path: Path) -> None:
    """A matrices-only capture saves and loads with b/V as empty sentinels.

    Construct ``StataCapture`` directly: no Stata licence, no pystata, no
    embedded session.  This also pins the documented sentinel (empty b/V) so a
    future reader cannot mistake it for a coefficient capture.
    """

    matrices = {
        "V_matrix": [[0.25, 0.0], [0.0, 0.75]],
        "RMSPE": [[7.896955684058109]],
    }
    capture = StataCapture(
        b=np.asarray([], dtype=float),
        V=np.asarray([], dtype=float).reshape(0, 0),
        scalars={},
        macros={},
        names=[],
        version="19.5",
        matrices=matrices,
    )

    assert capture.b.size == 0, "b must be the empty sentinel"
    assert capture.V.size == 0, "V must be the empty sentinel"
    assert capture.matrices == matrices
    assert "matrices" in capture.to_dict()

    path = capture.save(tmp_path / "nested" / "matrices_only.json")
    loaded = StataCapture.load(path)

    assert loaded.matrices == matrices, "matrices did not round-trip"
    assert loaded.b.size == 0, "b sentinel lost on reload"
    assert loaded.V.size == 0, "V sentinel lost on reload"
    assert loaded.version == "19.5"
    assert loaded.scalars == {}
    assert loaded.macros == {}
    assert loaded.names == []
    # __getitem__ must expose matrices as well as scalars.
    assert loaded["RMSPE"] == [[7.896955684058109]]
    with pytest.raises(KeyError):
        loaded["does_not_exist"]

    # The saved JSON must be plain data (replayable without Stata).
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["b"] == []
    assert payload["V"] == []
    assert payload["matrices"] == matrices


def test_old_format_fixture_still_loads() -> None:
    """A pre-``matrices`` fixture loads with an empty ``matrices`` dict.

    ``regress_auto.json`` is one of the original coefficient fixtures.  Its
    ``b``/``V`` content must be preserved; adding the new optional field must
    not disturb it.
    """

    capture = StataCapture.load(FIXTURES / "regress_auto.json")

    assert capture.matrices == {}, "old fixtures must default matrices to {}"
    assert capture.b.size == 3, "coefficient vector changed"
    assert capture.b[0] == pytest.approx(-0.006587886375773953)
    assert capture.V.shape == (3, 3)
    assert capture.names, "coefficient names lost"
    # The old fixture has no 'matrices' key at all; load() must tolerate that.
    payload = json.loads((FIXTURES / "regress_auto.json").read_text(encoding="utf-8"))
    assert "matrices" not in payload, "test premise: fixture predates the field"
    # to_dict() now includes the field, but the original b/V values are intact.
    assert capture.to_dict()["b"] == payload["b"]
    assert capture.to_dict()["V"] == payload["V"]


def test_capture_matrices_method_exists_with_unchanged_capture_contract() -> None:
    """The new method exists; ``capture()`` still requires e(b)/e(V).

    Read the source-level contract rather than starting a Stata session: the
    guard in ``capture()`` must still be present so existing coefficient
    fixtures/tests keep their meaning.
    """

    import inspect

    assert hasattr(StataOracle, "capture_matrices")
    assert hasattr(StataOracle, "capture")
    capture_source = inspect.getsource(StataOracle.capture)
    assert "no e(b)/e(V) available" in capture_source
    assert "raise RuntimeError" in capture_source


# -------------------------------------------------------------------- task 3c
@pytest.mark.parametrize("name", SYNTH_FIXTURES)
def test_synth_fixture_has_all_six_matrices_with_correct_shapes(name: str) -> None:
    """Every committed fixture carries all six matrices in the documented shapes.

    * ``V_matrix``: K x K (K = number of predictors)
    * ``W_weights``: J x 2 (unit number, unit weight)
    * ``X_balance``: K x 2 (Treated, Synthetic)
    * ``RMSPE``: 1 x 1
    * ``Y_treated`` / ``Y_synthetic``: T x 1
    """

    path = _fixture(name)
    if not path.exists():
        pytest.skip(f"{path.name} not captured; run scripts/capture_synth_fixtures.py")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "matrices" in payload, f"{name}: no matrices block"

    matrices = payload["matrices"]
    assert set(matrices) == set(MATRIX_NAMES), f"{name}: missing/extra matrices"

    V = np.asarray(matrices["V_matrix"], dtype=float)
    K = V.shape[0]
    assert V.ndim == 2 and V.shape == (K, K), f"{name}: V_matrix {V.shape}"

    W = np.asarray(matrices["W_weights"], dtype=float)
    assert W.ndim == 2 and W.shape[1] == 2 and W.shape[0] > 0, (
        f"{name}: W_weights {W.shape}"
    )

    X = np.asarray(matrices["X_balance"], dtype=float)
    assert X.shape == (K, 2), f"{name}: X_balance {X.shape} != {(K, 2)}"

    R = np.asarray(matrices["RMSPE"], dtype=float)
    assert R.shape == (1, 1), f"{name}: RMSPE {R.shape}"

    Yt = np.asarray(matrices["Y_treated"], dtype=float)
    Ys = np.asarray(matrices["Y_synthetic"], dtype=float)
    T = Yt.shape[0]
    assert Yt.ndim == 2 and Yt.shape == (T, 1), f"{name}: Y_treated {Yt.shape}"
    assert Ys.shape == Yt.shape, f"{name}: Y_synthetic {Ys.shape} != {Yt.shape}"

    # Provenance that makes the capture replayable/auditable.
    assert payload.get("stata_version"), f"{name}: no Stata version"
    assert payload.get("ado"), f"{name}: no ado version"
    assert payload.get("dataset"), f"{name}: no dataset path"
    assert payload.get("command"), f"{name}: no command line"
    assert payload.get("captured"), f"{name}: no capture date"
    assert payload.get("n_obs", 0) > 0, f"{name}: no obs count"
    assert payload.get("n_predictors", 0) > 0, f"{name}: no predictor count"
    assert payload.get("n_units", 0) > 0, f"{name}: no unit count"

    # b/V are the documented empty sentinels: synth saves no e(b)/e(V).
    assert payload["b"] == [] and payload["V"] == [], (
        f"{name}: b/V must be empty sentinels"
    )


@pytest.mark.parametrize("name", SYNTH_FIXTURES)
def test_synth_fixture_loads_as_stata_capture(name: str) -> None:
    """Each fixture is a valid ``StataCapture`` payload (not just ad-hoc JSON)."""

    path = _fixture(name)
    if not path.exists():
        pytest.skip(f"{path.name} not captured; run scripts/capture_synth_fixtures.py")
    capture = StataCapture.load(path)
    assert set(capture.matrices) == set(MATRIX_NAMES)
    assert capture.b.size == 0 and capture.V.size == 0
    assert capture.version, "version should be recorded"


# -------------------------------------------------------------------- task 3d
@pytest.mark.parametrize("name", sorted(GROUND_TRUTH))
def test_known_smoking_ground_truth_matches(name: str) -> None:
    """The three established smoking reference numbers match within 1e-10.

    These are cross-checks of the CAPTURE against the Lead's live Stata runs.
    They do not assert agreement with any Python implementation.
    """

    path = _fixture(name)
    if not path.exists():
        pytest.skip(f"{path.name} not captured; run scripts/capture_synth_fixtures.py")
    matrices = _matrices(name)

    V = np.asarray(matrices["V_matrix"], dtype=float)
    diag = np.diag(V)
    expected = GROUND_TRUTH[name]["v_diag"]
    assert np.allclose(diag, expected, atol=ATOL, rtol=0.0), (
        f"{name}: V diagonal {diag.tolist()} != {expected}"
    )

    rmspe = float(np.asarray(matrices["RMSPE"], dtype=float).ravel()[0])
    assert abs(rmspe - GROUND_TRUTH[name]["rmspe"]) < ATOL, (
        f"{name}: RMSPE {rmspe} != {GROUND_TRUTH[name]['rmspe']}"
    )


# -------------------------------------------------------------------- task 3e
@pytest.mark.parametrize("name", SYNTH_FIXTURES)
def test_v_matrix_is_diagonal_nonnegative_and_sums_to_one(name: str) -> None:
    """``e(V_matrix)`` is a diagonal predictor-weight matrix.

    The diagonal entries are the optimised predictor weights: nonnegative and
    summing to 1 (up to floating-point tolerance).
    """

    path = _fixture(name)
    if not path.exists():
        pytest.skip(f"{path.name} not captured; run scripts/capture_synth_fixtures.py")
    V = np.asarray(_matrices(name)["V_matrix"], dtype=float)
    K = V.shape[0]

    assert V.shape == (K, K)
    assert np.allclose(V, np.diag(np.diag(V)), atol=1e-12), (
        f"{name}: V_matrix is not diagonal"
    )
    diag = np.diag(V)
    assert bool(np.all(diag >= -1e-12)), f"{name}: negative diagonal entries"
    assert abs(float(diag.sum()) - 1.0) < 1e-8, (
        f"{name}: diagonal sums to {diag.sum()}, expected 1"
    )


@pytest.mark.parametrize("name", SYNTH_FIXTURES)
def test_unit_weights_are_a_synthetic_control(name: str) -> None:
    """``W_weights`` columns are unit numbers and nonnegative unit weights.

    This is a structural sanity check on the evidence artifact, not a
    conformance claim about any estimator.
    """

    path = _fixture(name)
    if not path.exists():
        pytest.skip(f"{path.name} not captured; run scripts/capture_synth_fixtures.py")
    W = np.asarray(_matrices(name)["W_weights"], dtype=float)
    assert W.shape[1] == 2
    assert bool(np.all(W[:, 1] >= -1e-12)), f"{name}: negative unit weights"
    # `synth.ado:589-590` applies roundmat() (3 decimal places) to W BEFORE
    # saving e(W_weights); the rounding breaks the simplex normalisation.
    # Measured column sums across the six fixtures are 0.999 / 1.000 / 1.001,
    # i.e. <= 1e-3 drift.  This is Stata's real behaviour: the Python port must
    # copy it rather than "correcting" W back to sum 1, which would desynchronise
    # e(W_weights), e(X_balance) and e(Y_synthetic).
    assert abs(float(W[:, 1].sum()) - 1.0) < 2e-3, (
        f"{name}: unit weights sum to {W[:, 1].sum()}, expected ~1 "
        "(roundmat() 3-dp display rounding)"
    )
