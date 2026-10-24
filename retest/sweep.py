"""One entry point for the independent conformance sweep.

The sweep was written as seven per-domain scripts plus an aggregator, which means running it meant
knowing which files to invoke in which order. That is enough friction that a check gets skipped, and
a check that gets skipped is worth nothing -- the whole reason this sweep exists is that the
library's own test suite passed while 24 defects sat undetected.

This module makes the sweep a single command:

    python -m retest.sweep                 # run everything, then compare with the registry
    python -m retest.sweep --domains tseries panel
    python -m retest.sweep --no-run        # compare only, using the last saved results

It does three things the individual scripts do not:

  1. runs the domain scripts in sequence and reports each one's exit status and duration;
  2. regenerates ``retest/aggregate.json``;
  3. diffs the measured tiers against ``tests/conformance/registry.yml`` and exits non-zero when a
     registry level is BETTER than the sweep measured. That direction is the only one that matters:
     a level that is too low is conservative, while a level that is too high is a false claim.

Exit codes: 0 agreement, 1 regression found, 2 the sweep could not run (no Stata, no results).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
DOMAINS = ROOT / "domains"
AGGREGATE = ROOT / "aggregate.json"
REGISTRY = REPO / "tests" / "conformance" / "registry.yml"

TIER_RANK = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
LEVEL_RANK = {"E3": 0, "E2": 1, "E1": 2}
#: The worst tier each registry level is allowed to measure at.
LEVEL_MAX_TIER = {"E3": "A", "E2": "C", "E1": "E"}


def domain_script(name: str) -> Path:
    return DOMAINS / name / "run.py"


def run_domain(name: str) -> tuple[bool, float, str]:
    """Run one domain script. Returns (completed, seconds, tail of output).

    ``completed`` means the script REACHED THE END -- produced its results -- not that it found no
    defects. A domain script exits non-zero precisely when it measures a failing comparison, so
    treating a non-zero exit as a crash made every real finding look like an infrastructure
    problem. What distinguishes the two is whether ``results.jsonl`` was rewritten during the run:
    a script that dies early leaves the previous file untouched (or absent), while a script that
    finishes writes a fresh one regardless of what it found.
    """

    script = domain_script(name)
    if not script.exists():
        return False, 0.0, f"missing {script}"

    results = DOMAINS / name / "results.jsonl"
    before = results.stat().st_mtime if results.exists() else 0.0

    started = time.monotonic()
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.monotonic() - started

    wrote_results = results.exists() and results.stat().st_mtime > before
    if not wrote_results:
        tail = "\n".join((completed.stdout or "").strip().splitlines()[-3:])
        tail += "\n" + "\n".join((completed.stderr or "").strip().splitlines()[-5:])
        return False, elapsed, f"no results written\n{tail}"

    tail = "\n".join((completed.stdout or "").strip().splitlines()[-1:])
    return True, elapsed, tail


def regenerate_aggregate() -> bool:
    script = ROOT / "aggregate.py"
    if not script.exists():
        return False
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.returncode == 0


def load_registry_levels() -> dict[str, str]:
    sys.path.insert(0, str(REPO / "src"))
    from westata.conformance.registry import load_registry

    return {entry.command: entry.level for entry in load_registry(REGISTRY)}


def measured_tiers() -> dict[str, list[str]]:
    """Worst-to-best tiers per registry-ish command, from aggregate.json."""

    payload = json.loads(AGGREGATE.read_text(encoding="utf-8"))
    return payload


def compare(registry_levels: dict[str, str], aggregate: dict) -> list[tuple[str, str, str]]:
    """Registry entries whose level is better than the sweep measured.

    Deliberately narrow: this reports only the dangerous direction. It does not try to regrade, and
    it does not touch files -- a check that rewrites the thing it checks cannot be trusted.
    """

    sys.path.insert(0, str(ROOT / "lib"))
    from tiering import entry_rows  # local helper, shared with regrade tooling

    findings: list[tuple[str, str, str]] = []
    for entry_command, level in registry_levels.items():
        rows = entry_rows(aggregate, entry_command)
        graded = [(row, tier) for row, tier in rows if tier]
        if not graded:
            continue
        worst = max((tier for _, tier in graded), key=lambda tier: TIER_RANK[tier])
        allowed = LEVEL_MAX_TIER.get(level, "E")
        if TIER_RANK[worst] > TIER_RANK[allowed]:
            findings.append((entry_command, level, worst))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domains",
        nargs="*",
        default=None,
        help="subset of domains to run (default: all)",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="skip execution, compare whatever results are already saved",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="only print failures and the final verdict",
    )
    args = parser.parse_args(argv)

    names = args.domains or sorted(
        path.name for path in DOMAINS.iterdir() if path.is_dir() and domain_script(path.name).exists()
    )
    if not names:
        print("no domain scripts found", file=sys.stderr)
        return 2

    if not args.no_run:
        print(f"running {len(names)} domain(s)\n")
        failed: list[str] = []
        for name in names:
            ok, elapsed, tail = run_domain(name)
            status = "ok  " if ok else "DEAD"
            print(f"  [{status}] {name:20s} {elapsed:7.1f}s")
            if not ok:
                failed.append(name)
                print(f"         {tail}")
        if failed:
            print(f"\n{len(failed)} domain(s) did not complete: {', '.join(failed)}")
            # A crashed domain leaves stale results behind; comparing them would be worse than
            # stopping, because a stale file can look like a pass.
            return 2
        if not regenerate_aggregate():
            print("aggregate.py failed", file=sys.stderr)
            return 2

    if not AGGREGATE.exists():
        print(f"no aggregate at {AGGREGATE}; run without --no-run first", file=sys.stderr)
        return 2

    aggregate = json.loads(AGGREGATE.read_text(encoding="utf-8"))
    registry_levels = load_registry_levels()
    findings = compare(registry_levels, aggregate)

    total = aggregate.get("total")
    tiers = aggregate.get("tier_counts", {})
    print(f"\nsweep: {total} comparison points, tiers {tiers}")

    if findings:
        print(f"\n{len(findings)} registry level(s) are better than measured:")
        for command, level, worst in sorted(findings):
            print(f"  {command:44s} declared {level}, measured worst {worst}")
        print(
            "\nEach of these is a false claim: the registry says the command is reliable at a level "
            "the measurement does not support. Fix the entry or fix the code -- do not widen the "
            "level to hide it."
        )
        return 1

    print("\nevery registry level is supported by the measurement")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
