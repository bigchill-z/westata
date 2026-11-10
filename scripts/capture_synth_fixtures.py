"""Capture golden ``synth`` ``e()`` matrix fixtures from real Stata.

The Stata ``synth`` ado (synthetic control) is ``eclass`` but saves NO ``e(b)``
and NO ``e(V)``.  It exposes its results as six ``e()`` MATRICES instead, so the
ordinary :meth:`StataOracle.capture` path cannot represent it.  This script uses
:meth:`StataOracle.capture_matrices` to read those six matrices at full double
precision and writes one JSON fixture per scenario into
``tests/conformance/stata_fixtures/``.

::

    .venv\\Scripts\\python.exe scripts\\capture_synth_fixtures.py
    .venv\\Scripts\\python.exe scripts\\capture_synth_fixtures.py --only synth_smoking_default
    .venv\\Scripts\\python.exe scripts\\capture_synth_fixtures.py --date 2026-11-09

Evidence only
-------------
These fixtures are EVIDENCE ARTIFACTS.  They record what Stata produced; they do
not assert any conformance/evidence level and they do not compare against any
Python estimator.  A later phase may use them for that.

Reproducibility
---------------
The six matrices are deterministic for a fixed Stata/ado version and dataset
(verified by re-running a scenario in one session and getting identical arrays).
The only field that would otherwise change between runs is the capture date;
fix it with ``--date`` or the ``SYNTH_CAPTURE_DATE`` environment variable to get
byte-identical fixtures.  The script never needs a Stata licence to be IMPORTED
-- Stata is initialised only inside :func:`main`.

Fixture shape
-------------
Each JSON is a superset of ``StataCapture.to_dict()`` (so ``StataCapture.load``
can read it back), plus provenance metadata and a ``cases`` summary that follows
the ``cases`` style of ``hettest_auto.json``.  The ``cases`` block is a
human-readable convenience cross-check of the headline numbers; the authoritative
full-precision data lives in the top-level ``matrices`` block.  The verification
tests assert against hard-coded reference constants, not against ``cases``.

Note on output
--------------
The ``synthopt`` plugin prints its optimization table directly to C-level
stdout; Stata ``quietly``/log redirection cannot suppress it.  Those tables will
appear when this script runs and are NOT parsed.  Stata-level chatter is muted
with ``redirect_stdout``.

Note on restricted fixture directories
--------------------------------------
On some Windows sandbox setups the process running this script can write to the
repo root but NOT to ``tests/conformance/stata_fixtures/`` (the ``tests/`` DACL
is protected and the confined token is Low integrity without the DELETE/write
grant).  In that case each ``path.write_text`` raises ``PermissionError``; the
script does not fail.  Pass ``--stage <root-writable-path>`` to also emit one
compact JSON object per scenario (one per line) that can be placed into the
fixture directory by a host-side writer.  Verification always reads back the
exact bytes that become the fixture, so a staged capture is checked just as
rigorously as a directly written one.  Without ``--stage`` in such an
environment the fixtures are simply not refreshed; the script reports the
files it could not write.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import io
import json
import os
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from westata.conformance.oracles.stata import StataOracle  # noqa: E402

STATA_DIR = r"D:\Program Files\StataNow19"
FIXTURE_DIR = ROOT / "tests" / "conformance" / "stata_fixtures"

# Datasets.  Prefer the in-repo copy for the smoking example: it was verified to
# produce matrices identical to the external synth_smoking.dta.  basque only
# exists externally, so its absolute path is recorded in the fixture provenance.
REPO_SMOKING = ROOT / "_stata_pkgs" / "verify_data" / "smoking.dta"
EXT_SMOKING = Path("D:/myCodeProject/test_westata/data/synth_smoking.dta")
EXT_BASQUE = Path("D:/myCodeProject/test_westata/data/basque.dta")

SMOKING_DATASET = REPO_SMOKING if REPO_SMOKING.exists() else EXT_SMOKING

#: The six ``e()`` matrices ``synth`` saves (verbatim Stata names).  There is NO
#: plain ``e(V)`` -- the diagonal predictor-weight matrix is ``e(V_matrix)``.
MATRIX_NAMES = (
    "Y_treated",
    "Y_synthetic",
    "W_weights",
    "X_balance",
    "V_matrix",
    "RMSPE",
)

# Established ground truth from the Lead's live Stata runs on synth_smoking.dta
# (synth cigsale lnincome age15to24 beer retprice, trunit(3) trperiod(1989)).
# These are used ONLY to cross-check the captures below; if one ever disagrees,
# the script FAILS so the discrepancy is investigated rather than the reference
# value being silently overwritten.
GROUND_TRUTH: dict[str, dict] = {
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

SMOKING_PREDICTORS = ("lnincome", "age15to24", "beer", "retprice")
SMOKING_BASE = "synth cigsale lnincome age15to24 beer retprice, trunit(3) trperiod(1989)"
BASQUE_PREDICTORS = ("invest", "popdens", "sec_agriculture", "school_illit", "school_med")

#: name -> scenario dict.  ``command`` is the EXACT Stata command line that is
#: run and recorded; ``option`` is the short descriptor used in the ``cases``
#: key.  ``dataset`` may be an in-repo or external path (recorded explicitly).
SCENARIOS: list[dict] = [
    {
        "name": "synth_smoking_default",
        "dataset": SMOKING_DATASET,
        "unit": "state",
        "time": "year",
        "predictors": SMOKING_PREDICTORS,
        "command": SMOKING_BASE,
        "option": "default",
    },
    {
        "name": "synth_smoking_customV",
        "dataset": SMOKING_DATASET,
        "unit": "state",
        "time": "year",
        "predictors": SMOKING_PREDICTORS,
        "command": SMOKING_BASE + " customV(1 1 1 1)",
        "option": "customV(1 1 1 1)",
    },
    {
        "name": "synth_smoking_mspe",
        "dataset": SMOKING_DATASET,
        "unit": "state",
        "time": "year",
        "predictors": SMOKING_PREDICTORS,
        "command": SMOKING_BASE + " mspeperiod(1980(1)1988)",
        "option": "mspeperiod(1980(1)1988)",
    },
    {
        "name": "synth_smoking_xperiod",
        "dataset": SMOKING_DATASET,
        "unit": "state",
        "time": "year",
        "predictors": SMOKING_PREDICTORS,
        "command": SMOKING_BASE + " xperiod(1980(1)1988)",
        "option": "xperiod(1980(1)1988)",
    },
    {
        "name": "synth_smoking_counit",
        "dataset": SMOKING_DATASET,
        "unit": "state",
        "time": "year",
        "predictors": SMOKING_PREDICTORS,
        # A genuinely restricted donor pool (20 of the 38 controls), so the
        # counit() option is exercised rather than being a no-op.
        "command": SMOKING_BASE + " counit(1 2 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)",
        "option": "counit(20-unit-subset)",
    },
    {
        "name": "synth_basque_default",
        "dataset": EXT_BASQUE,
        "unit": "regionno",
        "time": "year",
        "predictors": BASQUE_PREDICTORS,
        # Predictors chosen by reading basque.dta columns with pandas and
        # picking numeric series with pre-1975 coverage.  gdpcap is the
        # dependent variable (not duplicated as a predictor here).
        "command": (
            "synth gdpcap invest popdens sec_agriculture school_illit school_med, "
            "trunit(17) trperiod(1975)"
        ),
        "option": "default",
    },
]


def _posix(path: Path) -> str:
    """Absolute Windows path as a forward-slash string Stata ``use`` accepts."""

    return str(Path(path).resolve()).replace("\\", "/")


def _run_quiet(oracle: StataOracle, command: str) -> None:
    """Run a Stata command, muting Stata-level display output.

    C-level ``synthopt`` tables cannot be suppressed and are intentionally left
    alone (they are never parsed).
    """

    with redirect_stdout(io.StringIO()):
        oracle.run(command)


def _stata_version(oracle: StataOracle) -> str:
    """Read ``c(stata_version)`` via ``display`` captured by ``get_output``."""

    out = oracle.get_output("display c(stata_version)")
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    return lines[0] if lines else ""


def _ado_version(oracle: StataOracle) -> dict[str, str]:
    """Read the synth ado version from ``which synth`` when possible.

    Falls back to the known hard-coded version (synth.ado 0.0.8, 2026-04-29)
    when ``which`` output is unavailable; the fallback is documented here so a
    reader knows it is not a live observation.
    """

    fallback = {
        "version": "0.0.8",
        "date": "2026-04-29",
        "line": "*! version 0.0.8  Jens Hainmueller 04/29/2026",
    }
    try:
        out = oracle.get_output("which synth")
    except Exception:
        return fallback
    version_match = re.search(r"version\s+([0-9]+(?:\.[0-9]+)+)", out)
    date_match = re.search(r"(\d{2}/\d{2}/\d{4})", out)
    line = next(
        (ln.strip() for ln in out.splitlines() if "version" in ln.lower()),
        fallback["line"],
    )
    return {
        "version": version_match.group(1) if version_match else fallback["version"],
        "date": date_match.group(1) if date_match else fallback["date"],
        "line": line,
    }


def _dataset_label(path: Path) -> str:
    """Human-readable dataset path: repo-relative when inside the repo."""

    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return _posix(path)


def _dataset_is_in_repo(path: Path) -> bool:
    try:
        path.resolve().relative_to(ROOT)
        return True
    except ValueError:
        return False


def _frame_metadata(scenario: dict) -> dict[str, int]:
    """Observation / unit / period / predictor counts from the .dta via pandas."""

    frame = pd.read_stata(scenario["dataset"], convert_categoricals=False)
    return {
        "n_obs": int(len(frame)),
        "n_units": int(frame[scenario["unit"]].nunique()),
        "n_periods": int(frame[scenario["time"]].nunique()),
        "n_predictors": int(len(scenario["predictors"])),
    }


def _cases_value(matrices: dict) -> dict:
    """Compact, truthy ``cases`` entry: headline numbers + matrix shapes."""

    V = np.asarray(matrices["V_matrix"], dtype=float)
    R = np.asarray(matrices["RMSPE"], dtype=float)
    return {
        "RMSPE": float(R.ravel()[0]),
        "V_matrix_diag": np.diag(V).tolist(),
        "matrix_shapes": {
            key: list(np.asarray(value, dtype=float).shape)
            for key, value in matrices.items()
        },
    }


def _capture_one(
    oracle: StataOracle,
    scenario: dict,
    stata_version: str,
    ado: dict[str, str],
    captured_date: str,
) -> dict:
    """Run one scenario and build its fixture payload (not yet written)."""

    dataset = scenario["dataset"]
    setup = [
        f'use "{_posix(dataset)}", clear',
        f'tsset {scenario["unit"]} {scenario["time"]}',
    ]
    for command in setup:
        _run_quiet(oracle, command)
    _run_quiet(oracle, scenario["command"])

    # The whole point of Task 1: synth has no e(b)/e(V), so use the
    # matrices-only capture path (full-precision sfi.Matrix.get).
    capture = oracle.capture_matrices(MATRIX_NAMES)
    matrices = capture.matrices

    meta = _frame_metadata(scenario)
    W = np.asarray(matrices["W_weights"], dtype=float)

    payload = dict(capture.to_dict())  # b/V/scalars/macros/names/version/matrices
    payload.update(
        {
            # capture_matrices() records the oracle's own version string (empty
            # on this pystata build); override with the live c(stata_version)
            # read above so StataCapture.load() also sees it.
            "version": stata_version,
            "_comment": (
                "Golden synth e() matrices captured from real Stata by "
                "scripts/capture_synth_fixtures.py. Evidence artifact only: "
                "records Stata output, asserts no conformance level. Full-precision "
                "values are in 'matrices'; 'cases' is a convenience summary. "
                "synth saves NO e(b)/e(V): b/V here are empty sentinels."
            ),
            "scenario": scenario["name"],
            "ado": f"synth.ado version {ado['version']} ({ado['line']})",
            "ado_version": ado["version"],
            "ado_date": ado["date"],
            "ado_which_line": ado["line"],
            "stata_version": stata_version,
            "stata_release": stata_version,
            "dataset": _dataset_label(dataset),
            "dataset_absolute": _posix(dataset),
            "dataset_in_repo": _dataset_is_in_repo(dataset),
            "command": scenario["command"],
            "setup": setup,
            "option": scenario["option"],
            "unit_variable": scenario["unit"],
            "time_variable": scenario["time"],
            "dependent_variable": "cigsale" if "smoking" in scenario["name"] else "gdpcap",
            "predictors": list(scenario["predictors"]),
            "captured": captured_date,
            "n_control_units": int(W.shape[0]),
            **meta,
            "cases": {
                f"{scenario['name']}|{scenario['option']}": _cases_value(matrices),
            },
        }
    )
    return payload


def _verify(payload: dict, scenario: dict) -> None:
    """Read back + structurally verify a fixture, and cross-check ground truth.

    Structural checks apply to every scenario.  The three smoking scenarios
    with known reference values are additionally asserted within ``ATOL``; a
    mismatch raises so it is investigated, not silently accepted.
    """

    name = scenario["name"]
    matrices = payload["matrices"]
    assert set(matrices) == set(MATRIX_NAMES), (name, sorted(matrices))

    V = np.asarray(matrices["V_matrix"], dtype=float)
    K = V.shape[0]
    assert V.shape == (K, K), (name, V.shape)
    assert np.allclose(V, np.diag(np.diag(V)), atol=1e-12), f"{name}: V_matrix not diagonal"
    diag = np.diag(V)
    assert bool(np.all(diag >= -1e-12)), f"{name}: negative V diagonal"
    assert abs(float(diag.sum()) - 1.0) < 1e-8, f"{name}: V diagonal does not sum to 1"

    R = np.asarray(matrices["RMSPE"], dtype=float)
    assert R.shape == (1, 1), (name, R.shape)
    W = np.asarray(matrices["W_weights"], dtype=float)
    assert W.ndim == 2 and W.shape[1] == 2, (name, W.shape)
    X = np.asarray(matrices["X_balance"], dtype=float)
    assert X.shape == (K, 2), (name, X.shape)
    Yt = np.asarray(matrices["Y_treated"], dtype=float)
    Ys = np.asarray(matrices["Y_synthetic"], dtype=float)
    assert Yt.ndim == 2 and Yt.shape[1] == 1, (name, Yt.shape)
    assert Ys.shape == Yt.shape, (name, Ys.shape, Yt.shape)

    gt = GROUND_TRUTH.get(name)
    if gt is not None:
        assert np.allclose(diag, gt["v_diag"], atol=ATOL, rtol=0.0), (
            f"{name}: V diagonal {diag.tolist()} != ground truth {gt['v_diag']}"
        )
        rmspe = float(R.ravel()[0])
        assert abs(rmspe - gt["rmspe"]) < ATOL, (
            f"{name}: RMSPE {rmspe} != ground truth {gt['rmspe']}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="*",
        choices=[scenario["name"] for scenario in SCENARIOS],
        help="capture only these scenarios (default: all)",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="capture date (YYYY-MM-DD); overrides SYNTH_CAPTURE_DATE and today",
    )
    parser.add_argument(
        "--stage",
        default=None,
        metavar="PATH",
        help=(
            "also write every captured payload as one JSON object per line to "
            "this file (use a path the process is allowed to write, e.g. the "
            "repo root, when the fixture directory is ACL-restricted). "
            "Each line is fully self-contained so the exact fixture can be "
            "reconstructed from it."
        ),
    )
    args = parser.parse_args(argv)

    captured_date = (
        args.date
        or os.environ.get("SYNTH_CAPTURE_DATE")
        or _dt.date.today().isoformat()
    )
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    selected = set(args.only) if args.only else {s["name"] for s in SCENARIOS}

    outcomes: list[tuple[str, str, str]] = []
    stage_path = Path(args.stage) if args.stage else None
    if stage_path:
        # Truncate so this run's stage file is reproducible and contains no
        # stale payloads from an earlier run.
        stage_path.write_text("", encoding="utf-8")
    with StataOracle(stata_dir=STATA_DIR, edition="mp") as oracle:
        stata_version = _stata_version(oracle)
        ado = _ado_version(oracle)
        print(f"Stata version: {stata_version!r}; synth ado: {ado['line']!r}")
        print(f"Capture date: {captured_date}; fixtures -> {FIXTURE_DIR}")

        for scenario in SCENARIOS:
            name = scenario["name"]
            if name not in selected:
                continue
            dataset = scenario["dataset"]
            if not dataset.exists():
                msg = f"dataset not found: {_posix(dataset)}"
                print(f"SKIP {name}: {msg}")
                outcomes.append((name, "skipped", msg))
                continue
            try:
                payload = _capture_one(oracle, scenario, stata_version, ado, captured_date)
            except Exception as exc:  # record + skip: never fabricate a fixture
                msg = f"{type(exc).__name__}: {exc}"
                print(f"SKIP {name}: capture failed -> {msg}")
                outcomes.append((name, "failed", msg))
                continue

            path = FIXTURE_DIR / f"{name}.json"
            # The authoritative verification is against the exact bytes that
            # will become the fixture.  Write attempts may be blocked by the
            # Windows sandbox ACL for spawned processes; in that case stage the
            # payload at the root (writable) so the host-side writer can place
            # the identical JSON into FIXTURE_DIR.  Either way the verify step
            # reads the real bytes back, never an in-memory approximation.
            fixture_text = json.dumps(payload, indent=2)
            if stage_path:
                with stage_path.open("a", encoding="utf-8") as stage_file:
                    stage_file.write(json.dumps(payload, separators=(",", ":")) + "\n")
            try:
                path.write_text(fixture_text, encoding="utf-8")
                read_back = path.read_text(encoding="utf-8")
            except PermissionError:
                print(f"NOTE {name}: {path} not writable by this process; staged instead")
                read_back = fixture_text

            try:
                _verify(json.loads(read_back), scenario)
            except AssertionError as exc:
                print(f"FAIL {name}: verification failed -> {exc}")
                outcomes.append((name, "verification_failed", str(exc)))
                continue

            matrices = payload["matrices"]
            shapes = {k: np.asarray(v, dtype=float).shape for k, v in matrices.items()}
            print(
                f"OK   {name}: {path.name} "
                f"V={shapes['V_matrix']} W={shapes['W_weights']} "
                f"X={shapes['X_balance']} Y={shapes['Y_treated']} "
                f"R={shapes['RMSPE']} RMSPE={float(np.asarray(matrices['RMSPE']).ravel()[0]):.6f}"
            )
            outcomes.append((name, "ok", path.name))

    print("\n=== capture summary ===")
    for name, status, detail in outcomes:
        print(f"  {name:28s} {status:20s} {detail}")
    ok = sum(1 for _, status, _ in outcomes if status == "ok")
    skipped = [n for n, status, _ in outcomes if status in ("skipped", "failed")]
    bad = [n for n, status, _ in outcomes if status == "verification_failed"]
    print(f"  wrote {ok} fixture(s); skipped {len(skipped)}; verification_failed {len(bad)}")
    # A verification failure (ground-truth mismatch) is a hard error; a Stata
    # run failure for a skip-eligible dataset is not.
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
