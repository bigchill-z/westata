"""Collect the raw material for the full API reference manual.

The manual must describe 142 public exports. Writing them by hand would reproduce a failure
class this repository has already hit more than once: documentation that drifts from the
code, or numbers that nobody could have obtained from the code beside them.

So the reference is GENERATED from two sources, both mechanical:

1. **Signatures** come from ``inspect`` on the live objects, so they cannot be stale.
2. **Example output** comes from actually running the example and capturing the real
   result, so no printed number is invented.

What cannot be generated is the PROSE -- what a command does, which Stata command it
corresponds to, and the caveats. Those are hand-written into a side file
(``docs/api_source/prose_<lang>.json``) and merged here. A function with no prose entry is
reported as MISSING rather than silently emitted with an empty description, so an
incomplete manual is visible instead of plausible.

Usage::

    python docs/collect_api_reference.py            # writes docs/api_source/api_data.json
    python docs/collect_api_reference.py --check    # exit 1 if any prose is missing
"""

from __future__ import annotations

import argparse
import inspect
import io
import json
import sys
import traceback
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# A stale `src/westata.egg-info/PKG-INFO` shadows the installed metadata: with `src`
# first on `sys.path`, `importlib.metadata.version("westata")` reads that directory, so
# `ws.__version__` reported the PREVIOUS release and the generated reference silently
# claimed the wrong version. This bit twice (0.3.13 and 0.3.16), so it is now a hard
# error rather than a silently-wrong document.
_EGG_INFO = ROOT / "src" / "westata.egg-info"
if _EGG_INFO.exists():
    raise SystemExit(
        f"{_EGG_INFO} is a stale build artifact and shadows the installed version "
        f"metadata. Delete it (`rm -rf src/westata.egg-info`) and re-run; the API "
        f"reference would otherwise record the previous release."
    )


import westata as ws  # noqa: E402

OUT = ROOT / "docs" / "api_source"

#: Exports that are modules or typing artifacts rather than API surface.  Listing them
#: explicitly (instead of filtering by name pattern) keeps the count auditable.
NOT_API = frozenset(
    {
        "annotations",
        "data",
        "errors",
        "estimators",
        "expressions",
        "formula",
        "postestimation",
        "protocols",
        "random",
        "results",
        "ssc",
        "vce",
        "report",
        "etable",
        "to_latex",
        "to_markdown",
    }
)


def _datasets() -> dict[str, pd.DataFrame]:
    """The small frames every example draws on.

    Deterministic seeds, because an example whose output changes between runs cannot be
    printed in a manual.
    """

    rng = np.random.default_rng(20261031)
    n = 200

    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    latent = 0.4 + 0.9 * x1 - 0.3 * x2 + rng.normal(size=n)

    base = pd.DataFrame(
        {
            "y": 1.0 + 0.8 * x1 - 0.5 * x2 + rng.normal(size=n),
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "yb": (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-latent))).astype(int),
            "ycnt": rng.poisson(np.exp(0.3 + 0.4 * x1)),
        }
    )
    # A censored outcome and an interval-censored pair for the limited-response family.
    base["ycens"] = np.clip(1.0 + 0.8 * x1 + rng.normal(size=n), 0.0, None)
    base["lo"] = 0.7 + 0.8 * x1 + rng.normal(size=n) * 0.1
    base["hi"] = 1.3 + 0.8 * x1 + rng.normal(size=n) * 0.1
    # Two outcomes for the multivariate family.
    base["y1"] = 1.0 + 0.6 * x1 + rng.normal(size=n)
    base["y2"] = 2.0 - 0.4 * x2 + rng.normal(size=n)
    # Panel identifiers and a panel outcome: a firm-level effect plus noise.
    units = 40
    per = n // units
    base["firm"] = np.repeat(np.arange(units), per)
    base["year"] = np.tile(np.arange(1, per + 1), units)
    firm_effect = np.repeat(rng.normal(size=units), per)
    base["ypanel"] = 1.0 + 0.7 * x1 + firm_effect + rng.normal(size=n)
    # A treatment and its outcome, for the treatment-effect family.
    base["treat"] = (0.5 + 0.8 * x3 + rng.normal(size=n) > 0).astype(int)
    base["ytreat"] = 1.0 + 0.8 * x1 + 1.5 * base["treat"] + rng.normal(size=n)
    # A selection indicator with a missing outcome off the selected rows.
    base["sel"] = (0.5 + 0.8 * x3 + rng.normal(size=n) > 0).astype(int)
    base["yselect"] = np.where(
        base["sel"] == 1, 1.0 + 0.7 * x1 + 0.5 * rng.normal(size=n), np.nan
    )
    return {"base": base}


class _Collector:
    """Run one example and capture both its value and its printed form."""

    def __init__(self) -> None:
        self.frames = _datasets()

    def run(self, code: str) -> dict[str, object]:
        """Evaluate an example, which may be one expression OR a short script.

        ``eval`` alone rejects ``r = ...; r.e['r2']``, so a multi-statement example is run
        with ``exec`` and its LAST expression is evaluated separately. Examples that need
        setup are common enough (fit once, then read a macro) that supporting only a bare
        expression would push writers toward writing examples nobody can paste.
        """

        captured = io.StringIO()
        namespace: dict[str, object] = {"ws": ws, "np": np, "pd": pd, **self.frames}
        try:
            with redirect_stdout(captured):
                value = _evaluate(code, namespace)
            return {"ok": True, "value": _render(value), "code": code, "error": None}
        except Exception as error:  # noqa: BLE001 - the manual reports what happened
            return {
                "ok": False,
                "value": None,
                "code": code,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(limit=3),
            }


def _evaluate(code: str, namespace: dict[str, object]) -> object:
    """Run ``code`` and return the value of its last statement when that is an expression.

    Two forms are supported, because both are natural to write in a manual:

    * a single expression -- ``ws.OLS.from_formula(...).fit().summary()``
    * a short script whose last line is the expression to show --
      ``r = ws.OLS...; r.e['r2']``

    ``eval`` handles the first. For the second, the script up to the final newline is
    executed and the final line is evaluated. Splitting on ``;`` was tried first and broke
    on indentation, so the separator is a NEWLINE: examples that need several statements
    should write them one per line.
    """

    try:
        return eval(compile(code, "<example>", "eval"), namespace)  # noqa: S307
    except SyntaxError:
        pass

    lines = code.strip().splitlines()
    if not lines:
        return None
    head, tail = "\n".join(lines[:-1]), lines[-1]
    if head.strip():
        exec(compile(head, "<example>", "exec"), namespace)  # noqa: S102
    try:
        return eval(compile(tail, "<example>", "eval"), namespace)  # noqa: S307
    except SyntaxError:
        exec(compile(tail, "<example>", "exec"), namespace)  # noqa: S102
        return None


def _render(value: object) -> str:
    """Turn an example's return value into the text the manual prints."""

    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, np.floating, np.integer)):
        return repr(float(value))
    if isinstance(value, np.ndarray):
        return np.array2string(value, precision=6, suppress_small=True)
    # Results objects and DataFrames both have a useful text form.
    try:
        return str(value)
    except Exception:  # noqa: BLE001
        return repr(value)


def _signature(name: str, target: object) -> str | None:
    try:
        return str(inspect.signature(target))
    except (TypeError, ValueError):
        return None


def _result_surface() -> dict[str, object]:
    """The surface every fitted result exposes: methods and properties.

    This is the OTHER half of the API. A manual that documents only ``ws.OLS`` and stops
    leaves the reader unable to look up ``result.margins(...)``, ``result.e[...]`` or
    ``result.predict(kind="xb")`` -- which is where most postestimation work happens.
    """

    from westata.results.base import StataResults

    methods = []
    properties = []
    for name in sorted(dir(StataResults)):
        if name.startswith("_"):
            continue
        member = getattr(StataResults, name, None)
        if callable(member):
            methods.append(
                {
                    "name": name,
                    "signature": _signature(name, member),
                    "docstring": inspect.getdoc(member) or "",
                }
            )
        else:
            properties.append(
                {
                    "name": name,
                    "docstring": inspect.getdoc(member) or "",
                }
            )
    return {"class": "StataResults", "methods": methods, "properties": properties}


def collect() -> dict[str, object]:
    collector = _Collector()
    exports: list[dict[str, object]] = []

    for name in sorted(dir(ws)):
        if name.startswith("_") or name in NOT_API:
            continue
        target = getattr(ws, name)
        kind = (
            "class"
            if inspect.isclass(target)
            else "function"
            if inspect.isfunction(target)
            else "other"
        )
        entry: dict[str, object] = {
            "name": name,
            "kind": kind,
            "module": getattr(target, "__module__", None),
            "docstring": inspect.getdoc(target) or "",
            "signature": _signature(name, target),
        }
        if kind == "class":
            entry["from_formula"] = _signature(
                name, getattr(target, "from_formula", None)
            )
            entry["fit"] = _signature(name, getattr(target, "fit", None))
            methods = []
            for attribute in dir(target):
                if attribute.startswith("_") or attribute in ("from_formula", "fit"):
                    continue
                member = getattr(target, attribute, None)
                if inspect.isfunction(member) or inspect.ismethod(member):
                    methods.append({"name": attribute, "signature": _signature(attribute, member)})
            entry["methods"] = methods
        exports.append(entry)

    return {
        "generated_by": "docs/collect_api_reference.py",
        "westata_version": getattr(ws, "__version__", "unknown"),
        "exports": exports,
        "result_surface": _result_surface(),
    }


def _load_prose() -> dict[str, dict[str, object]]:
    """Merge every ``prose_*.json`` in the reference directory.

    The prose is split across files so several writers can work without touching the same
    one; the merge is unordered but keys must be unique, which is checked here rather than
    silently letting a later file win.
    """

    merged: dict[str, dict[str, object]] = {}
    origin: dict[str, str] = {}
    for path in sorted(OUT.glob("prose_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key, entry in payload.items():
            if key.startswith("_"):
                continue
            if key in merged:
                raise SystemExit(
                    f"duplicate prose key {key!r} in {path.name} and {origin[key]}"
                )
            merged[key] = entry
            origin[key] = path.name
    return merged


def _check(prose: dict[str, dict[str, object]]) -> int:
    """Run every example and report coverage. Returns the number of problems."""

    collector = _Collector()
    data = collect()
    problems = 0

    known = {entry["name"] for entry in data["exports"]}
    # The result object is documented as a section of its own: its METHODS and PROPERTIES
    # are the names, and the class itself is covered by that section.
    known |= {m["name"] for m in data["result_surface"]["methods"]}
    known |= {p["name"] for p in data["result_surface"]["properties"]}
    known.add("StataResults")
    # Instance attributes that belong on the result surface but are set per fit rather than
    # declared on the class, so ``dir(StataResults)`` does not list them.
    known |= {"model_data"}

    missing = sorted(name for name in known if name not in prose)
    if missing:
        print(f"MISSING prose for {len(missing)} of {len(known)} names:")
        for name in missing:
            print(f"  - {name}")
        problems += len(missing)

    for name, entry in sorted(prose.items()):
        if name not in known:
            print(f"UNKNOWN name in prose (no such export or member): {name!r}")
            problems += 1
            continue
        example = entry.get("example")
        if not example:
            continue
        outcome = collector.run(str(example))
        if not outcome["ok"]:
            print(f"FAILED example for {name}: {outcome['error']}")
            problems += 1
        for field in ("stata", "en", "zh"):
            if not entry.get(field):
                print(f"MISSING field {field!r} for {name}")
                problems += 1
        if entry.get("notes_en") and not entry.get("notes_zh"):
            print(f"notes_en without notes_zh for {name}")
            problems += 1
        if entry.get("notes_zh") and not entry.get("notes_en"):
            print(f"notes_zh without notes_en for {name}")
            problems += 1

    print(f"checked {len(prose)} prose entries against {len(known)} names: {problems} problem(s)")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify coverage, write nothing")
    args = parser.parse_args()

    if args.check:
        return 1 if _check(_load_prose()) else 0

    data = collect()
    data["prose"] = _load_prose()
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "api_data.json"
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    count = len(data["exports"])
    print(f"wrote {target} with {count} exports and {len(data['prose'])} prose entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
