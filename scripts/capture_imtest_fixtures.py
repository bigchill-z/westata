"""Capture golden Stata reference values for ``estat imtest``.

``imtest.ado`` (v2.1.5) computes White's test and the Cameron & Trivedi
decomposition.  Two details decide whether a reimplementation agrees with it,
and one of them silently changes the CONCLUSION rather than the last digits:

* **Degrees of freedom come from the auxiliary regression's RANK**, not from the
  number of columns built.  The ``rhs2`` block contains ``x_i^2`` for every
  regressor, and an indicator satisfies ``x^2 == x``, so its square is exactly
  collinear with the regressor -- ``_regress`` drops it and reports the reduced
  ``e(df_m)``.  On ``regress mpg weight foreign`` Stata's White test is
  ``df = 4``; counting columns gives 5, which inflates the p-value and can flip
  a rejection.  A fixture that only recorded the statistic would not catch this.
* **``imtest`` works in FLOAT by default** (``imtest.ado:35``: ``local type =
  cond("`double'" != "", "double", "float")``), so the default statistic differs
  from an exact computation in the 8th significant digit.  That is a property of
  Stata's storage type, not an error, and reproducing it bit-for-bit would mean
  emulating float32 accumulation.  Both the default and ``, double`` values are
  captured so the difference is documented instead of argued about; the tests
  compare against ``, double``, which is what a correct double-precision
  implementation should match.

Usage::

    .venv\\Scripts/python.exe scripts/capture_imtest_fixtures.py

The data is ``tests/data/auto.dta``, which ships with the repository.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STATA_DIR = Path(r"D:\Program Files\StataNow19")
STATA_EXE = STATA_DIR / "StataMP-64.exe"
DATASET = ROOT / "tests" / "data" / "auto.dta"
FIXTURE = ROOT / "tests" / "conformance" / "stata_fixtures" / "imtest_auto.json"

#: ``id -> (formula for westata, Stata regress tail)``.
#:
#: The indicator ``foreign`` is load-bearing: it is the variable whose squared
#: column is collinear with itself, so it is what makes the rank differ from the
#: column count.  ``M5`` uses THREE continuous regressors, where every squared
#: column is genuinely independent, as the control that the rank rule still
#: yields the nominal count.
MODELS: dict[str, dict[str, str]] = {
    "W1": {"formula": "mpg ~ weight + foreign", "stata": "mpg weight foreign"},
    "W2": {"formula": "mpg ~ weight + length + displacement",
           "stata": "mpg weight length displacement"},
    "W3": {"formula": "mpg ~ weight", "stata": "mpg weight"},
    "W4": {"formula": "price ~ weight + mpg + foreign",
           "stata": "price weight mpg foreign"},
    # Two indicators: TWO squared columns collapse, so the rank rule matters more.
    "W5": {"formula": "mpg ~ weight + foreign + C(rep78)",
           "stata": "mpg weight foreign i.rep78"},
}

#: Columns the westata formula needs pulled from the frame.
COLUMNS = ("mpg", "price", "weight", "length", "displacement", "foreign", "rep78")


def build_do() -> tuple[str, list[dict[str, Any]]]:
    """Return the do-file text and the case metadata that indexes it."""

    lines: list[str] = [
        "set more off",
        "set linesize 255",
        f'use "{DATASET.as_posix()}", clear',
        'di "#VER|" c(stata_version) "|" c(version)',
    ]
    cases: list[dict[str, Any]] = []

    def add(model: str, precision: str, stata_options: str, kind: str) -> None:
        """One ``estat imtest`` invocation, captured at full precision."""

        case_id = f"{model}|{precision}|{kind}"
        lines.append(f"capture quietly estat imtest{stata_options}")
        # A model with no covariates is refused (imtest.ado:49-52), so record
        # the return code rather than assuming every case produces numbers.
        lines.append(f'di "#RC|{case_id}|" _rc')
        lines.append(
            f'di "#IM|{case_id}|" %21.17f r(chi2) "|" %9.0f r(df) "|" '
            f'%21.17f r(chi2_h) "|" %9.0f r(df_h) "|" '
            f'%21.17f r(chi2_s) "|" %9.0f r(df_s) "|" '
            f'%21.17f r(chi2_k) "|" %9.0f r(df_k) "|" '
            f'%21.17f r(chi2_t) "|" %9.0f r(df_t)'
        )
        cases.append(
            {
                "id": case_id,
                "model": model,
                "precision": precision,
                "stata_command": f"estat imtest{stata_options}",
                "kind": kind,
            }
        )

    for model, spec in MODELS.items():
        lines.append(f"quietly regress {spec['stata']}")
        # Default float run, then the double run -- both are recorded.
        # The option must follow a COMMA: ``estat imtest double`` (no comma) makes
        # Stata read ``double`` as a varlist and fail with r(101), which is a
        # syntax accident rather than a documented refusal.
        for precision, comma_option in (("float", ""), ("double", ", double")):
            add(model, precision, f", white{comma_option[1:]}", "white")
            add(model, precision, comma_option, "decomposition")

    return "\n".join(lines) + "\n", cases


def render(payload: dict[str, Any]) -> str:
    """Serialise one case per line, so the fixture is reviewable."""

    def dump(value: Any) -> str:
        return json.dumps(value, sort_keys=True)

    lines = ["{", f'  "_comment": {dump(payload["_comment"])},']
    for key in ("ado", "dataset", "stata_version", "stata_release"):
        if key in payload:
            lines.append(f'  "{key}": {dump(payload[key])},')

    cases = payload["cases"]
    lines.append('  "cases": {')
    keys = sorted(cases)
    for index, key in enumerate(keys):
        comma = "," if index < len(keys) - 1 else ""
        lines.append(f"    {dump(key)}: {dump(cases[key])}{comma}")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def parse_log(text: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Turn marker lines into the fixture payload."""

    by_id = {case["id"]: case for case in cases}
    payload: dict[str, Any] = {
        "_comment": (
            "Golden values captured from real Stata by "
            "scripts/capture_imtest_fixtures.py. Do not edit by hand: re-run the "
            "script. Each case is 'model|precision|kind' -> "
            "{chi2, df, chi2_h, df_h, chi2_s, df_s, chi2_k, df_k, chi2_t, df_t, "
            "rc}. 'kind' is white (,white) or decomposition (no option). "
            "'precision' is Stata's own float (default) or double (,double)."
        ),
        "ado": "imtest.ado v2.1.5 (StataNow 19.5)",
        "dataset": "tests/data/auto.dta",
        "cases": {},
    }
    rc: dict[str, int] = {}

    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("#"):
            continue
        parts = line.split("|")
        tag = parts[0]
        if tag == "#VER":
            payload["stata_version"] = parts[1]
            payload["stata_release"] = parts[2]
        elif tag == "#RC" and len(parts) == 5:
            # The case id itself contains two pipes ("model|precision|kind"),
            # so the id occupies parts[1..3] and the value follows at [4].
            rc["|".join(parts[1:4])] = int(float(parts[4]))
        elif tag == "#IM" and len(parts) == 14:
            case_id = "|".join(parts[1:4])
            entry = dict(by_id[case_id])
            entry.pop("id", None)

            def number(text: str) -> float | None:
                # ``r(chi2)`` exists only when ``, white`` ran; the plain
                # decomposition leaves it missing, which Stata prints as ".".
                text = text.strip()
                return None if text in (".", "") else float(text)

            values = parts[4:]
            entry.update(
                {
                    "chi2": number(values[0]),
                    "df": number(values[1]),
                    "chi2_h": number(values[2]),
                    "df_h": number(values[3]),
                    "chi2_s": number(values[4]),
                    "df_s": number(values[5]),
                    "chi2_k": number(values[6]),
                    "df_k": number(values[7]),
                    "chi2_t": number(values[8]),
                    "df_t": number(values[9]),
                }
            )
            payload["cases"][case_id] = entry

    for case_id, code in rc.items():
        if case_id in payload["cases"]:
            payload["cases"][case_id]["rc"] = code
    return payload


def run_stata(do_text: str) -> str:
    """Run the do-file and return Stata's own log.

    The child's stdio goes to the null device rather than a pipe: piped stdio is
    unavailable under this host's sandbox (CreatePipe raises WinError 5), and the
    log file already carries everything needed.
    """

    with tempfile.TemporaryDirectory(prefix="imtest_capture_") as tmp:
        work = Path(tmp)
        do_file = work / "capture.do"
        do_file.write_text(do_text, encoding="utf-8")
        completed = subprocess.run(
            [str(STATA_EXE), "-e", "-q", "do", str(do_file)],
            cwd=str(work),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=600,
        )
        log_file = work / "capture.log"
        if not log_file.exists():
            raise RuntimeError(
                f"Stata produced no {log_file.name} (exit code {completed.returncode})"
            )
        return log_file.read_text(encoding="utf-8", errors="ignore")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None,
                        help="write the JSON here instead of the committed fixture")
    args = parser.parse_args()

    if not STATA_EXE.exists():
        print(f"Stata not found at {STATA_EXE}", file=sys.stderr)
        return 2

    do_text, cases = build_do()
    log_text = run_stata(do_text)
    payload = parse_log(log_text, cases)

    if not payload["cases"]:
        print("no cases parsed from the Stata log; the run failed", file=sys.stderr)
        print(log_text[-4000:], file=sys.stderr)
        return 1

    target = Path(args.out) if args.out else FIXTURE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render(payload), encoding="utf-8")
    print(f"wrote {target} ({len(payload['cases'])} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
