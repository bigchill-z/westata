"""Check the westata *rename contract* against docs/NAMING_CONVENTIONS.md.

The source tree has NOT been renamed yet, so a checker that only asked "does the
code follow the new names?" would report everything as pending and tell us nothing
new.  The valuable mechanical question is whether the CONTRACT in
``scripts/naming_map.py`` is self-consistent with the rules -- this is where real
mistakes happened during the discussion (mis-reading the estat_* rule, a
``ContrastResult`` collision, flipping singular/plural twice).

Rules R-N1..R-N6 validate the contract; R-N7 reports source migration progress
(informational, never blocks).  Every finding carries the responsible contract
entry so it is actionable.

Usage::

    python scripts/check_naming.py              # human report, exit 0
    python scripts/check_naming.py --strict     # exit 1 if any contract rule fires
    python scripts/check_naming.py --rule R-N5  # one rule
    python scripts/check_naming.py --json       # machine-readable
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import sys
from collections import Counter
from dataclasses import dataclass, field

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
SRC = SRC_ROOT / "westata"

sys.path.insert(0, str(ROOT / "scripts"))
import naming_map as nm  # noqa: E402


def all_renames(defs: dict[str, list[str]]) -> list[nm.Rename]:
    """Static contract renames + result renames derived from the source AST.

    Result classes are generated from rules (prefix fix + 's'), so a new result
    class automatically becomes a pending rename without editing the contract.
    """
    static = list(nm.ALL_RENAMES)
    # Result renames: derive from source where possible, but ALSO from the frozen compat table
    # so a completed migration does not shrink the contract. The table is the durable record of
    # every old singular name that existed before the migration.
    try:
        from westata._compat import RESULT_RENAMED, RENAMED as FROZEN_RENAMED
    except ImportError:  # pragma: no cover - checker run outside the package
        RESULT_RENAMED, FROZEN_RENAMED = (), ()
    seen_old = {r.old for r in static}
    for name in defs:
        if (name.endswith("Result") and not name.endswith("Results")
                and name not in nm.RESULT_COLLISION_KEEP and name not in seen_old):
            static.append(nm.result_rename(name))
            seen_old.add(name)
    for old_name, new_name in RESULT_RENAMED:
        if old_name not in seen_old:
            static.append(nm.Rename(old_name, new_name, "result"))
            seen_old.add(old_name)
    if "IrFTable" in defs and "IrFTable" not in seen_old:
        static.append(nm.Rename("IrFTable", "IrfTable", "result"))
    return static


@dataclass
class Finding:
    rule: str
    where: str
    message: str


@dataclass
class Report:
    findings: dict[str, list[Finding]] = field(default_factory=dict)
    info: dict[str, object] = field(default_factory=dict)

    def add(self, finding: Finding) -> None:
        self.findings.setdefault(finding.rule, []).append(finding)

    def blocking(self) -> list[Finding]:
        return [f for group in self.findings.values() for f in group]


# --------------------------------------------------------------------------- #
# source helpers
# --------------------------------------------------------------------------- #
def _source_defs() -> dict[str, list[str]]:
    """name -> list of 'path:line' for every top-level class/function def."""
    out: dict[str, list[str]] = {}
    for p in sorted(SRC.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        tree = ast.parse(p.read_text(encoding="utf-8-sig"), filename=str(p))
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                rel = p.relative_to(ROOT).as_posix()
                out.setdefault(node.name, []).append(f"{rel}:{node.lineno}")
    return out


# --------------------------------------------------------------------------- #
# R-N1: cryptic prefixes expanded; familiar acronyms preserved
# --------------------------------------------------------------------------- #
def check_rn1(report: Report) -> None:
    for r in nm.CLASS_RENAMES:
        for prefix, expansion in nm.PREFIX_EXPANSIONS.items():
            if r.new.startswith(prefix) and expansion != prefix:
                report.add(Finding(
                    "R-N1", r.new,
                    f"new class name still carries un-expanded prefix '{prefix}' "
                    f"(expected '{expansion}...')",
                ))
    banned = {
        "InstrumentalVariables": "IV",
        "DifferenceInDifferences": "DID",
        "VectorAutoregression": "VAR",
    }
    for r in nm.CLASS_RENAMES:
        for text, acronym in banned.items():
            if text in r.new:
                report.add(Finding(
                    "R-N1", r.new,
                    f"new name expands familiar acronym {acronym} into '{text}'",
                ))


# --------------------------------------------------------------------------- #
# R-N2: functions renamed too; every rename is a real change
# --------------------------------------------------------------------------- #
def check_rn2(report: Report, defs: dict[str, list[str]]) -> None:
    """A function rename is valid when OLD or NEW exists (migration may be done or pending).

    Requiring the old name unconditionally made every completed migration a finding, which
    hides the real defect: a rename where NEITHER spelling is present.
    """

    for r in nm.FUNCTION_RENAMES:
        if r.old == r.new:
            report.add(Finding("R-N2", r.old, "function 'rename' is a no-op"))
        if r.old not in defs and r.new not in defs:
            report.add(Finding(
                "R-N2", r.old,
                f"neither old nor new function name ('{r.new}') found in source tree",
            ))


# --------------------------------------------------------------------------- #
# R-N3: result classes uniformly plural (with registered collision exceptions)
# --------------------------------------------------------------------------- #
def check_rn3(report: Report, defs: dict[str, list[str]]) -> None:
    singular = {n for n in defs if n.endswith("Result") and not n.endswith("Results")}
    # Every singular result gets a derived rename; only the two collision names are exempt.
    for name in sorted(singular - nm.RESULT_COLLISION_KEEP):
        derived = nm.result_rename(name)
        if derived.new == name or not derived.new.endswith("Results"):
            report.add(Finding(
                "R-N3", name,
                f"derived result rename '{derived.new}' is not a plural *Results target",
            ))
    # spelling normalisation for table/row objects (not *Result) must be registered too
    if "IrFTable" in defs and nm.RESULT_NAME_FIXES.get("IrFTable") != "IrfTable":
        report.add(Finding("R-N3", "IrFTable", "table spelling fix IrFTable->IrfTable not registered"))


# --------------------------------------------------------------------------- #
# R-N4: spelling hard fixes present
# --------------------------------------------------------------------------- #
def check_rn4(report: Report) -> None:
    required = {"TefectsRaResult", "SucestResult", "IrFTable", "Rdrobust", "Varbasic"}
    covered = {r.old for r in nm.ALL_RENAMES} | set(nm.RESULT_NAME_FIXES)
    for name in sorted(required - covered):
        report.add(Finding("R-N4", name, "known spelling defect has no rename entry"))


# --------------------------------------------------------------------------- #
# R-N5: estat_ prefix KEPT, suffix expands; every estat_* handled
# --------------------------------------------------------------------------- #
def check_rn5(report: Report, defs: dict[str, list[str]]) -> None:
    src_estat = {n for n in defs if n.startswith("estat_")}
    renamed = {r.old: r.new for r in nm.FUNCTION_RENAMES if r.old.startswith("estat_")}
    for old, new in renamed.items():
        if not new.startswith("estat_"):
            report.add(Finding("R-N5", old, f"new name '{new}' drops the estat_ prefix"))
        suffix = old[len("estat_"):]
        expected_suffix = nm.ESTAT_SUFFIX_EXPANSIONS.get(suffix)
        if expected_suffix is None:
            report.add(Finding(
                "R-N5", old,
                f"suffix '{suffix}' is not in ESTAT_SUFFIX_EXPANSIONS (nor a registered keep)",
            ))
        else:
            expected = "estat_" + expected_suffix
            if new != expected:
                report.add(Finding("R-N5", old, f"new name '{new}' != expected '{expected}'"))
    # A migrated estat_* function has its NEW name in the source and no old name; a pending
    # one has the old name. Neither is a finding -- only a name that is in neither state is.
    new_estat = set(renamed.values())
    unhandled = sorted(
        name for name in src_estat - set(renamed) - nm.ESTAT_KEEP
        if name not in new_estat
    )
    for name in unhandled:
        report.add(Finding("R-N5", name, "estat_* function has no rename and is not a registered keep"))
    for old, new in renamed.items():
        if old not in defs and new not in defs:
            report.add(Finding(
                "R-N5", old,
                f"neither old nor new estat name ('{new}') found in source tree",
            ))
    for name in nm.ESTAT_KEEP:
        suffix = name[len("estat_"):] if name.startswith("estat_") else ""
        if suffix and suffix not in nm.ESTAT_SUFFIX_KEEP:
            report.add(Finding("R-N5", name, f"kept estat suffix '{suffix}' is not registered as keep"))


# --------------------------------------------------------------------------- #
# R-N6: one concept one name — no collisions in the mapping
# --------------------------------------------------------------------------- #
def check_rn6(report: Report, defs: dict[str, list[str]]) -> None:
    renames = all_renames(defs)
    old_counts = Counter(r.old for r in renames)
    for old, c in old_counts.items():
        if c > 1:
            report.add(Finding("R-N6", old, f"old name mapped {c} times"))
    new_counts = Counter(r.new for r in renames)
    for new, c in new_counts.items():
        if c > 1:
            owners = sorted(r.old for r in renames if r.new == new)
            report.add(Finding("R-N6", new, f"collision: {owners} all map to this name"))
    # After a completed migration every new name exists in source -- that is the goal, not
    # a collision. Keep the check for a new name that exists but is NOT a contract target at
    # all (someone introduced a conflicting name outside the migration).
    for new in new_counts:
        owners = sorted(r.old for r in renames if r.new == new)
        if new in defs and not any(old in old_counts for old in owners):
            report.add(Finding(
                "R-N6", new,
                f"name exists at {', '.join(defs[new])} but no contract entry owns it",
            ))
    for name in nm.RESULT_COLLISION_KEEP:
        if name not in defs:
            report.add(Finding("R-N6", name, "registered collision-exception name not found in source"))


# --------------------------------------------------------------------------- #
# R-N7 (informational): source migration progress
# --------------------------------------------------------------------------- #
def check_rn7(report: Report, defs: dict[str, list[str]]) -> None:
    renames = all_renames(defs)
    pending = [r for r in renames if r.old in defs and r.new not in defs]
    done = [r for r in renames if r.new in defs]
    missing = [r for r in renames if r.old not in defs and r.new not in defs]
    report.info["migration"] = {
        "contract_total": len(renames),
        "pending": len(pending),
        "done": len(done),
        "stale_old_missing": len(missing),
    }
    for r in missing:
        report.add(Finding(
            "R-N7", r.old,
            f"contract is stale: neither old nor new name ('{r.new}') exists in source",
        ))


CHECKS = {
    "R-N1": lambda rep, d: check_rn1(rep),
    "R-N2": check_rn2,
    "R-N3": check_rn3,
    "R-N4": lambda rep, d: check_rn4(rep),
    "R-N5": check_rn5,
    "R-N6": check_rn6,
    "R-N7": check_rn7,
}


def collect(only: str | None = None) -> Report:
    report = Report()
    defs = _source_defs()
    for check in CHECKS.values():
        check(report, defs)
    if only:
        wanted = only.upper()
        report.findings = {
            rule: group
            for rule, group in report.findings.items()
            if rule.upper().startswith(wanted)
        }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--strict", action="store_true", help="exit 1 if any contract rule fires")
    parser.add_argument("--rule", default=None, help="only this rule, e.g. R-N5")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    report = collect(args.rule)
    if args.json:
        payload = {
            "findings": {
                rule: [{"where": f.where, "message": f.message} for f in group]
                for rule, group in sorted(report.findings.items())
            },
            "info": report.info,
        }
        print(json.dumps(payload, indent=1))
    else:
        print("westata naming-convention contract (docs/NAMING_CONVENTIONS.md)\n")
        total = 0
        for rule, group in sorted(report.findings.items()):
            print(f"{rule}  {len(group)} finding(s)")
            for f in group:
                total += 1
                print(f"  {'INFO' if rule == 'R-N7' else 'FAIL'}  {f.where}: {f.message}")
            print()
        if report.info:
            print("migration progress:", report.info["migration"])
        if not report.findings:
            print("no findings")
        print(f"\n{total} finding(s)")
    if args.strict and report.blocking():
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
