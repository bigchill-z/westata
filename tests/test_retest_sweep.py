"""Tests for the conformance sweep and its tier helpers.

A conformance check that itself goes wrong is worse than no check, because it produces confidence
without evidence. These tests pin the two rules that took the most effort to get right, and one
end-to-end property: the sweep must actually fail when the registry overstates a command.

They run without Stata. The sweep's execution step needs a live session, but its COMPARISON step is
pure bookkeeping over ``aggregate.json``, and that is the part that can silently rot.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "retest" / "lib"))

from tiering import (  # noqa: E402
    TIER_RANK,
    entry_rows,
    is_measurement,
    row_tier,
)

AGGREGATE = ROOT / "retest" / "aggregate.json"
if not AGGREGATE.exists():
    pytest.skip("aggregate.json missing", allow_module_level=True)

AGGREGATE_DATA = json.loads(AGGREGATE.read_text(encoding="utf-8"))


# --------------------------------------------------------------- row filtering


def test_a_diagnostic_probe_is_not_a_measurement():
    """Probes grade an internal mechanism, not the command's output.

    Counting them demoted `var` to E1 while its actual output measured at 1e-14 throughout.
    """

    probe = {"name": "var: VAR(1) standard errors: internal DOF factor (diagnostic)"}
    assert not is_measurement(probe)


def test_a_root_cause_row_is_not_a_measurement():
    assert not is_measurement({"name": "tobit /sigma SE root cause (diagnostic)"})


def test_an_alternative_mapping_row_is_not_a_measurement():
    """These compare against a mapping the repair round corrected.

    Their tier is the evidence FOR the fix, so counting them penalises code for having been
    repaired -- which is what happened to `ivregress` and `areg`.
    """

    alternative = {
        "name": "10d ivregress 2sls vce(robust) vs westata HC1（文档映射）"
    }
    assert not is_measurement(alternative)


def test_a_plain_command_row_is_a_measurement():
    assert is_measurement({"name": "VAR(1) coefficients"})


# ------------------------------------------------------------- tier formation


def test_the_row_tier_is_the_worse_of_the_two():
    """A command is only as good as its weakest second-order quantity.

    `teffects nnmatch` has exact coefficients (tier A) and 5-16% standard errors (tier E). Grading
    on coefficients alone would promote it.
    """

    assert row_tier({"param_tier": "A", "se_tier": "E"}) == "E"
    assert row_tier({"param_tier": "C", "se_tier": "A"}) == "C"
    assert row_tier({"param_tier": "A", "se_tier": "A"}) == "A"


def test_a_row_with_no_measured_tier_returns_none():
    assert row_tier({"param_tier": "NA", "se_tier": "NA"}) is None


# ------------------------------------------------------------- attribution


def test_nnmatch_does_not_absorb_the_other_teffects_estimators():
    """ra, ipw, aipw and nnmatch share one leading word.

    Matching on `teffects` alone charged nnmatch's 5-16% standard-error failures to the other
    three, which measure at 1e-15.
    """

    ra_rows = entry_rows(AGGREGATE_DATA, "teffects ra")
    nnmatch_rows = entry_rows(AGGREGATE_DATA, "teffects nnmatch")
    ra_names = {row.get("name") for row, _ in ra_rows}
    nnmatch_names = {row.get("name") for row, _ in nnmatch_rows}
    assert ra_names.isdisjoint(nnmatch_names)
    # nnmatch carries a failing tier; ra does not.
    assert any(tier in ("D", "E") for _, tier in nnmatch_rows if tier)
    assert not any(tier in ("D", "E") for _, tier in ra_rows if tier)


def test_regress_does_not_absorb_jackknife_rows():
    """`jackknife: regress` invokes jackknife. It says nothing about regress."""

    names = {row.get("name") for row, _ in entry_rows(AGGREGATE_DATA, "regress")}
    assert not any("jackknife" in (name or "") for name in names)


def test_an_unmapped_command_yields_no_rows():
    assert entry_rows(AGGREGATE_DATA, "a command the sweep never tried") == []


# ---------------------------------------------------------- end-to-end behaviour


def test_the_registry_never_claims_better_than_measured():
    """The property the whole sweep exists to enforce."""

    sys.path.insert(0, str(ROOT / "src"))
    from westata.conformance.registry import load_registry

    registry = load_registry(ROOT / "tests" / "conformance" / "registry.yml")
    allowed = {"E3": "A", "E2": "C", "E1": "E"}

    violations = []
    for entry in registry:
        graded = [(row, tier) for row, tier in entry_rows(AGGREGATE_DATA, entry.command) if tier]
        if not graded:
            continue
        worst = max((tier for _, tier in graded), key=lambda tier: TIER_RANK[tier])
        if TIER_RANK[worst] > TIER_RANK[allowed[entry.level]]:
            violations.append((entry.command, entry.level, worst))

    assert not violations, (
        "registry levels exceed what the sweep measured; fix the entry or the code, do not "
        f"widen the level: {violations}"
    )

# ------------------------------------------------------------- crash detection


def test_a_missing_domain_script_is_reported_not_ignored():
    """A domain listed in the sweep but absent on disk must fail loudly.

    Silently skipping it would leave the previous results.jsonl in place, and stale results
    compare as a pass.
    """

    sys.path.insert(0, str(ROOT / "retest"))
    from sweep import run_domain

    ok, _elapsed, tail = run_domain("a_domain_that_does_not_exist")
    assert not ok
    assert "missing" in tail


def test_a_domain_is_judged_by_writing_results_not_by_exit_code():
    """Domain scripts exit non-zero when they FIND defects.

    tseries reports arima and var failures and exits 1. The first version of the sweep read that
    exit code as a crash and printed "ok" only for runs that happened to pass everything -- which
    would have hidden the findings entirely. Completion is judged by whether results.jsonl was
    rewritten.
    """

    import inspect

    sys.path.insert(0, str(ROOT / "retest"))
    from sweep import run_domain

    source = inspect.getsource(run_domain)
    assert "results.jsonl" in source, "completion must be judged by the results file"
    assert "wrote_results" in source
