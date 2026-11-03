"""Capture golden Stata reference values for ``estat hettest``.

``hettest.ado`` (v3.4.4) is short enough to read, but reading it is not the same
as being right about it.  Two of its branches are easy to get backwards from the
source alone, and both were in fact implemented backwards here:

* **No varlist means the FITTED VALUES.**  ``hettest.ado:110-115`` replaces the
  empty varlist with a predicted ``z`` and regresses on that.  The package
  instead defaulted to *all* regressors, so ``result.estat_hettest()`` silently
  returned a two-regressor test where Stata returns a one-degree-of-freedom
  test -- 7.75 vs 2.53 on ``regress mpg weight foreign``.  Both numbers look
  plausible, which is what makes it dangerous.
* **Weights are carried into the auxiliary regression.**  ``hettest.ado:103-105,
  138`` passes ``[`e(wtype)' `e(wexp)']`` to ``_regress``, so every weighted
  fit's auxiliary regression is weighted too.  Ignoring them changes the
  statistic AND its degrees of freedom.

So the reference values below are captured from a real Stata, at full precision,
and committed as JSON.  The test suite compares against that JSON instead of
against a second transcription of the same ado -- a transcription can be wrong in
the same direction the implementation is wrong, and then the test passes while
the number is still wrong.

Usage::

    .venv\\Scripts\\python.exe scripts\\capture_hettest_fixtures.py
    .venv\\Scripts\\python.exe scripts\\capture_hettest_fixtures.py --only M1

The data is ``tests/data/auto.dta``, which ships with the repository, so the
capture is reproducible rather than describing a synthetic dataset that only the
capturing machine has (a defect ``tests/test_fixture_verification_scope.py``
documents for 18 older fixtures).
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
FIXTURE = ROOT / "tests" / "conformance" / "stata_fixtures" / "hettest_auto.json"

#: ``method -> (option text, r() scalar layout)``.  ``fstat`` reports F/df_m/df_r;
#: the other two report chi2/df.  Both blocks share the ``p`` scalar name.
METHOD_OPTION = {"normal": "", "iid": "iid", "fstat": "fstat"}

#: ``model id -> (westata formula, Stata regress tail, explicit varlists to test)``.
MODELS: dict[str, dict[str, Any]] = {
    "M1": {
        "formula": "mpg ~ weight + foreign",
        "stata": "mpg weight foreign",
        "explicit": [["weight"], ["weight", "foreign"]],
    },
    "M2": {
        "formula": "mpg ~ weight + length + displacement",
        "stata": "mpg weight length displacement",
        "explicit": [["weight"], ["weight", "length"]],
    },
    "M3": {
        "formula": "price ~ weight + mpg + foreign",
        "stata": "price weight mpg foreign",
        "explicit": [["weight"], ["weight", "mpg"]],
    },
}

#: ``rep78`` is the only count-like variable in ``auto``; it has 2 missing values,
#: so the weighted sample is 69 observations.  That is useful: it makes
#: ``e(N)`` differ from the row count and exposes a ``nobs``-vs-``sum(w)`` bug.
WEIGHTS: list[tuple[str, str] | None] = [
    None,
    ("aweight", "rep78"),
    ("fweight", "rep78"),
    ("iweight", "rep78"),
]

#: A factor-variable model, because the ``varlist`` names are SOURCE variable
#: names, not ``e(b)`` columns.  ``hettest.ado:138`` runs
#: ``_regress ui `varlist'``, and a varlist of ``rep78`` means the single raw
#: 1-5 column (df = 1) -- NOT the four treatment dummies that ``i.rep78``
#: produces in ``e(b)`` (df = 4).  Stata measures 12.76793787541800462 for the
#: raw column against 24.26516992264934203 for the dummies, so this is not a
#: distinction that can be hand-waved.  ``, rhs`` uses the ``e(b)`` names and
#: therefore DOES see all four dummies (df = 5 with the intercept dropped,
#: i.e. the four levels against the omitted base).
FACTOR_MODEL = {
    "F1": {
        "formula": "mpg ~ weight + C(rep78)",
        "stata": "mpg weight i.rep78",
        "varlists": [
            # raw source variable -> one column, df 1
            (["rep78"], ["rep78"]),
        ],
    },
    # A value-labelled 0/1 variable.  ``estat hettest foreign`` uses the raw
    # codes (one column, chi2(1) = 7.66936122444203594), whereas ``, rhs`` uses
    # the e(b) name ``foreign`` as a treatment dummy (chi2(2) = 7.7524933353).
    # Both give one column here, but by different routes, so the pair pins which
    # one the varlist took.
    "F2": {
        "formula": "mpg ~ weight + C(foreign)",
        "stata": "mpg weight i.foreign",
        "varlists": [
            (["foreign"], ["foreign"]),
        ],
    },
}


def _fmt_case(
    case_id: str,
    weight: tuple[str, str] | None,
    hettest_options: str,
    method: str,
) -> list[str]:
    """The two Stata lines that define one fixture case."""

    command = f"estat hettest{hettest_options}"
    if method == "fstat":
        return [
            f"quietly {command}",
            f'di "#HT|{case_id}|" %21.17f r(F) "|" %9.0f r(df_m) "|" '
            f'%21.17f r(p) "|" %9.0f r(df_r)',
        ]
    return [
        f"quietly {command}",
        f'di "#HT|{case_id}|" %21.17f r(chi2) "|" %9.0f r(df) "|" '
        f'%21.17f r(p) "|" .',
    ]


def _case_id(model: str, weight: tuple[str, str] | None, kind: str,
             variables: list[str], method: str) -> str:
    parts = [model]
    parts.append("unweighted" if weight is None else f"{weight[0]}_{weight[1]}")
    parts.append(kind)
    if kind == "varlist":
        parts.append("+".join(variables))
    parts.append(method)
    return "__".join(parts)


def build_do() -> tuple[str, list[dict[str, Any]]]:
    """Return the do-file text and the case metadata that indexes it."""

    lines: list[str] = [
        "set more off",
        "set linesize 255",
        f'use "{DATASET.as_posix()}", clear',
        'di "#VER|" c(stata_version) "|" c(version)',
    ]
    cases: list[dict[str, Any]] = []

    def add(model: str, weight: tuple[str, str] | None, kind: str,
            variables: list[str], method: str, hettest_options: str) -> None:
        case_id = _case_id(model, weight, kind, variables, method)
        lines.extend(_fmt_case(case_id, weight, hettest_options, method))
        cases.append(
            {
                "id": case_id,
                "model": model,
                "weight": None if weight is None else {"type": weight[0], "var": weight[1]},
                "kind": kind,
                "variables": variables,
                "method": method,
                "stata_command": f"estat hettest{hettest_options}",
            }
        )

    for model, spec in MODELS.items():
        for weight in WEIGHTS:
            # Weighted variants only for M1: the weight handling is a property of
            # the auxiliary regression, not of the model, so repeating it across
            # three models triples the runtime without adding evidence.
            if weight is not None and model != "M1":
                continue
            regress = f"quietly regress {spec['stata']}"
            if weight is not None:
                regress += f" [{weight[0]}={weight[1]}]"
            lines.append(regress)

            for method in METHOD_OPTION:
                suffix = METHOD_OPTION[method]
                # no varlist -> fitted values (the default Stata documents)
                text = "" if not suffix else f", {suffix}"
                add(model, weight, "fitted", [], method, text)
                # , rhs -> every e(b) name except _cons
                text = f", {suffix} rhs" if suffix else ", rhs"
                add(model, weight, "rhs", [], method, text)
                # explicit varlist
                for variables in spec["explicit"]:
                    listed = " ".join(variables)
                    text = f" {listed}, {suffix}" if suffix else f" {listed}"
                    add(model, weight, "varlist", variables, method, text)

    # --------------------------------------------------- factor-variable model
    # These pin the SOURCE-NAME vs DESIGN-COLUMN distinction that makes
    # ``variables=["rep78"]`` differ from ``rhs=True``.
    for model, spec in FACTOR_MODEL.items():
        lines.append(f"quietly regress {spec['stata']}")
        for method in METHOD_OPTION:
            suffix = METHOD_OPTION[method]
            text = "" if not suffix else f", {suffix}"
            add(model, None, "fitted", [], method, text)
            text = f", {suffix} rhs" if suffix else ", rhs"
            add(model, None, "rhs", [], method, text)
            for named, _columns in spec["varlists"]:
                listed = " ".join(named)
                text = f" {listed}, {suffix}" if suffix else f" {listed}"
                add(model, None, "varlist", named, method, text)

    # ---------------------------------------------------------------- mtest
    #
    # ``, rhs`` is required here, not optional: ``hettest.ado:126-128`` clears
    # ``mtmethod`` when the auxiliary regression has a single regressor, and the
    # no-varlist default IS a single regressor (the fitted values).  So
    # ``estat hettest, mtest(holm)`` reports the simultaneous test and returns NO
    # ``r(mtest)`` matrix at all -- verified on StataNow 19.5.  Only a multi-column
    # auxiliary regression produces the family.
    for model in ("M1", "M2"):
        lines.append(f"quietly regress {MODELS[model]['stata']}")
        lines.append("quietly estat hettest, mtest(noadjust) rhs")
        lines.append("tempname M")
        lines.append("matrix `M' = r(mtest)")
        lines.append(
            f'di "#MTB|{model}|" %9.0f rowsof(`M\') "|" %9.0f colsof(`M\')'
        )
        lines.append("local nv = rowsof(`M')")
        # ``local x : rowname M i`` is not extended-macro syntax (r(101) "varlist
        # not allowed"); the row names come back as one space-separated list.
        lines.append("local rnames : rownames `M'")
        lines.append("forvalues i = 1/`nv' {")
        lines.append("    local rn : word `i' of `rnames'")
        lines.append(
            '    di "#MT|' + model + '|`rn\'|" %21.17f el(`M\',`i\',1) "|" '
            '%9.0f el(`M\',`i\',2) "|" %21.17f el(`M\',`i\',3)'
        )
        lines.append("}")
        # The simultaneous statistic is reported alongside the family.
        lines.append(f'di "#HTS|{model}|" %21.17f r(chi2) "|" %9.0f r(df)')

    # The single-regressor case is a distinct behaviour worth pinning: mtest is
    # silently dropped and no matrix is returned.
    lines.extend(
        [
            "quietly regress mpg weight foreign",
            "quietly estat hettest, mtest(holm)",
            'di "#MTD|M1|" r(mtmethod) "|" %21.17f r(chi2) "|" %9.0f r(df)',
            "capture confirm matrix r(mtest)",
            'di "#MTX|M1|" _rc',
        ]
    )

    # ------------------------------------------------------------- refusals
    lines.extend(
        [
            "quietly regress mpg weight foreign, noconstant",
            "capture estat hettest",
            'di "#RC|noconstant|" _rc',
            "quietly regress mpg weight foreign",
            "quietly regress mpg weight foreign, vce(robust)",
            "capture estat hettest",
            'di "#RC|robust|" _rc',
            "quietly regress mpg weight foreign [pweight=rep78]",
            "capture estat hettest",
            'di "#RC|pweight|" _rc',
        ]
    )
    return "\n".join(lines) + "\n", cases


def parse_log(text: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Turn the marker lines into the fixture payload.

    The payload is a compact mapping rather than a list of verbose records: the
    fixture is committed and read by humans as well as tests, so one line per
    case beats forty.  Keys are ``model|weight|kind|variables|method`` with
    ``-`` standing for "no weight" and ``fitted``/``rhs`` describing the
    auxiliary regressors.
    """

    by_id = {case["id"]: case for case in cases}
    payload: dict[str, Any] = {
        "_comment": (
            "Golden values captured from real Stata by "
            "scripts/capture_hettest_fixtures.py. Do not edit by hand: re-run "
            "the script. Each case is "
            "'model|weight|kind|statistic|df|pvalue|df_r'; kind is fitted (no "
            "varlist), rhs (, rhs) or varlist:<names> (explicit varlist)."
        ),
        "ado": "hettest.ado v3.4.4 (StataNow 19.5)",
        "dataset": "tests/data/auto.dta",
        "cases": {},
        "mtest": {},
        "refusals": {},
    }

    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("#"):
            continue
        parts = line.split("|")
        tag = parts[0]
        if tag == "#VER":
            payload["stata_version"] = parts[1]
            payload["stata_release"] = parts[2]
        elif tag == "#HT" and len(parts) == 6:
            case = by_id[parts[1]]
            weight = case["weight"]
            weight_key = "-" if weight is None else f"{weight['type']}:{weight['var']}"
            kind = case["kind"]
            if kind == "varlist":
                kind = "varlist:" + ",".join(case["variables"])
            key = "|".join(
                [case["model"], weight_key, kind, case["method"]]
            )
            payload["cases"][key] = [
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
                None if parts[5].strip() == "." else float(parts[5]),
            ]
        elif tag == "#MTB" and len(parts) == 4:
            payload["mtest"].setdefault(parts[1], {"rows": int(float(parts[2])),
                                                   "cols": int(float(parts[3])),
                                                   "variables": []})
        elif tag == "#MT" and len(parts) == 6:
            payload["mtest"].setdefault(parts[1], {"variables": []})["variables"].append(
                {
                    "name": parts[2],
                    "chi2": float(parts[3]),
                    "df": float(parts[4]),
                    "pvalue": float(parts[5]),
                }
            )
        elif tag == "#HTS" and len(parts) == 4:
            payload["mtest"].setdefault(parts[1], {})["simultaneous"] = {
                "chi2": float(parts[2]),
                "df": float(parts[3]),
            }
        elif tag == "#MTD" and len(parts) == 5:
            # mtest() with a single auxiliary regressor: no family, no matrix.
            # Layout is ``#MTD|model|mtmethod|chi2|df``.
            payload["mtest_single_regressor"] = {
                "mtmethod": parts[2].strip(),
                "chi2": float(parts[3]),
                "df": float(parts[4]),
            }
        elif tag == "#MTX" and len(parts) == 3:
            payload.setdefault("mtest_single_regressor", {})[
                "r_mtest_missing"
            ] = int(float(parts[2])) != 0
        elif tag == "#RC" and len(parts) == 3:
            payload["refusals"][parts[1]] = int(float(parts[2]))

    return payload


def run_stata(do_text: str) -> str:
    with tempfile.TemporaryDirectory(prefix="hettest_capture_") as tmp:
        work = Path(tmp)
        do_file = work / "capture.do"
        do_file.write_text(do_text, encoding="utf-8")
        # Stata writes ``capture.log`` itself; the child's stdio is sent to the
        # null device rather than through a pipe.  Piped stdio is unavailable
        # under this host's sandbox (CreatePipe raises WinError 5), and the log
        # file already carries everything this script needs.
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


def render(payload: dict[str, Any]) -> str:
    """Serialise one case per line.

    ``json.dumps(indent=2)`` puts every list element on its own line, turning a
    72-case fixture into 500 lines nobody will read.  A committed reference file
    is read in review, so its layout is part of its value: one row per Stata run,
    aligned with the do-file that produced it.
    """

    def dump(value: Any) -> str:
        return json.dumps(value, sort_keys=True)

    def section(name: str, rows: list[str], last: bool) -> list[str]:
        out = [f'  "{name}": {{']
        out.extend(rows)
        out.append("  }" if last else "  },")
        return out

    lines = ["{", f'  "_comment": {dump(payload["_comment"])},']
    for key in ("ado", "dataset", "stata_version", "stata_release"):
        if key in payload:
            lines.append(f'  "{key}": {dump(payload[key])},')

    cases = sorted(payload["cases"].items())
    rows = [
        f"    {dump(key)}: {dump(list(value))}"
        + ("," if position < len(cases) - 1 else "")
        for position, (key, value) in enumerate(cases)
    ]
    lines.extend(section("cases", rows, last=False))

    models = sorted(payload["mtest"].items())
    rows = [
        f"    {dump(model)}: {dump(entry)}"
        + ("," if position < len(models) - 1 else "")
        for position, (model, entry) in enumerate(models)
    ]
    lines.extend(section("mtest", rows, last=False))

    lines.append(f'  "refusals": {dump(payload["refusals"])},')
    extras = [
        f'  "{key}": {dump(value)}'
        for key, value in sorted(payload.items())
        if not key.startswith("_")
        and key
        not in (
            "ado",
            "dataset",
            "stata_version",
            "stata_release",
            "cases",
            "mtest",
            "refusals",
        )
    ]
    lines.extend(line + "," for line in extras[:-1])
    if extras:
        lines.append(extras[-1])
    else:
        lines[-1] = lines[-1].rstrip(",")
    lines.append("}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default=None,
                        help="capture a single model id (M1/M2/M3) for a quick check")
    parser.add_argument("--out", default=None,
                        help="write the JSON here instead of the committed fixture; "
                             "useful when the fixture directory is not writable from "
                             "this process")
    args = parser.parse_args()

    if not STATA_EXE.exists():
        print(f"Stata not found at {STATA_EXE}", file=sys.stderr)
        return 2

    do_text, cases = build_do()
    if args.only:
        keep = [c for c in cases if c["model"] == args.only]
        if not keep:
            print(f"no cases for model {args.only!r}", file=sys.stderr)
            return 2
        allowed = " ".join(f'"{c["id"]}"' for c in keep)
        cases = keep
        # Rebuild the do-file around the selected cases only.
        lines = ["set more off", "set linesize 255",
                 f'use "{DATASET.as_posix()}", clear',
                 'di "#VER|" c(stata_version) "|" c(version)']
        current = None
        for line in do_text.splitlines():
            if line.startswith("quietly regress "):
                current = line
                continue
            if line.startswith("#HT|"):
                case_id = line.split("|")[1]
                if case_id in {c["id"] for c in keep}:
                    lines.append(current)
                    lines.append(line)
        do_text = "\n".join(lines) + "\n"

    log_text = run_stata(do_text)
    payload = parse_log(log_text, cases)

    if not payload["cases"]:
        print("no cases parsed from the Stata log; the run failed", file=sys.stderr)
        print(log_text[-4000:], file=sys.stderr)
        return 1

    target = Path(args.out) if args.out else FIXTURE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render(payload), encoding="utf-8")
    print(f"wrote {target} ({len(payload['cases'])} cases, "
          f"{len(payload['mtest'])} mtest models, "
          f"{len(payload['refusals'])} refusals)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
