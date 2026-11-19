"""The naming-contract checker must be self-consistent AND every rule must fire.

Two failure modes matter here, and neither is "the contract reports nothing":

1. **A dead rule.** A rule that can never fire reports zero forever and looks healthy.
   ``test_every_rule_can_fire`` injects a deliberate violation into each rule's
   inputs and asserts the rule reports it -- the same discipline as
   ``test_every_rule_can_fire_on_a_synthetic_module`` in test_api_conventions.py.

2. **A contract that silently drifts from the decisions.** The discussion flipped
   the result suffix twice and mis-read the ``estat_*`` rule once. The semantic
   anchor tests pin those decisions in the machine-readable contract so a future
   edit cannot quietly undo them.

The contract itself is expected to be clean: no findings, all renames pending
(the source has not been renamed yet).  ``--strict`` must therefore pass today.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_naming as checker  # noqa: E402
import naming_map as nm  # noqa: E402


def test_contract_is_clean_and_everything_is_pending() -> None:
    report = checker.collect()
    assert report.blocking() == [], [f.where for f in report.blocking()]
    info = report.info["migration"]
    assert info["contract_total"] > 100, "the contract is suspiciously small"
    assert info["pending"] == info["contract_total"], "source should not be renamed yet"
    assert info["done"] == 0
    assert info["stale_old_missing"] == 0, "no contract entry may reference a nonexistent name"


def test_every_rule_can_fire(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject one violation per rule; each check must report that rule."""

    defs = checker._source_defs()

    # R-N1: a new name expands a familiar acronym.
    monkeypatch.setattr(nm, "CLASS_RENAMES", nm.CLASS_RENAMES + [
        nm.Rename("Foo", "InstrumentalVariablesRegress", "class"),
    ])
    rep = checker.Report()
    checker.check_rn1(rep)
    assert "R-N1" in rep.findings
    monkeypatch.undo()

    # R-N2: a no-op rename and a rename whose old name is absent from source.
    monkeypatch.setattr(nm, "FUNCTION_RENAMES", nm.FUNCTION_RENAMES + [
        nm.Rename("noop", "noop", "function"),
        nm.Rename("ghost_function", "real_function", "function"),
    ])
    rep = checker.Report()
    checker.check_rn2(rep, defs)
    assert "R-N2" in rep.findings
    monkeypatch.undo()

    # R-N3: a broken result derivation (target is not plural).
    monkeypatch.setattr(nm, "result_rename", lambda old: nm.Rename(old, old, "result"))
    rep = checker.Report()
    checker.check_rn3(rep, defs)
    assert "R-N3" in rep.findings
    monkeypatch.undo()

    # R-N4: remove all spelling-fix registrations -> defects become uncovered.
    monkeypatch.setattr(nm, "RESULT_NAME_FIXES", {})
    monkeypatch.setattr(nm, "ALL_RENAMES", [r for r in nm.ALL_RENAMES if r.new != "RDRobust"])
    rep = checker.Report()
    checker.check_rn4(rep)
    assert "R-N4" in rep.findings
    monkeypatch.undo()

    # R-N5: an estat_* rename that drops the prefix.
    monkeypatch.setattr(nm, "FUNCTION_RENAMES", nm.FUNCTION_RENAMES + [
        nm.Rename("estat_esize", "estimate_effect_size", "function"),
    ])
    rep = checker.Report()
    checker.check_rn5(rep, defs)
    assert "R-N5" in rep.findings
    monkeypatch.undo()

    # R-N6: two old names mapping to the same new name.
    monkeypatch.setattr(nm, "ALL_RENAMES", nm.ALL_RENAMES + [
        nm.Rename("extra_old", "Regress", "class"),
    ])
    rep = checker.Report()
    checker.check_rn6(rep, defs)
    assert "R-N6" in rep.findings
    monkeypatch.undo()

    # R-N7: a stale contract entry (neither old nor new exists in source).
    monkeypatch.setattr(nm, "ALL_RENAMES", nm.ALL_RENAMES + [
        nm.Rename("gone_old", "gone_new", "class"),
    ])
    rep = checker.Report()
    checker.check_rn7(rep, defs)
    assert "R-N7" in rep.findings
    monkeypatch.undo()


# ----- semantic anchors: pin the decisions that were flip-flopped during discussion -----

def test_estat_prefix_is_kept_and_only_suffix_expands() -> None:
    """The user's correction: estat_ stays; the suffix expands. Not estimate_*."""
    estat = {r.old: r.new for r in nm.FUNCTION_RENAMES if r.old.startswith("estat_")}
    assert estat["estat_esize"] == "estat_effect_size"
    assert estat["estat_firststage"] == "estat_first_stage"
    assert estat["estat_weakrobust"] == "estat_weak_instrument_robust"
    assert all(new.startswith("estat_") for new in estat.values())
    assert not any(new.startswith("estimate_") for new in estat.values())


def test_estat_person_names_and_familiar_acronyms_are_kept() -> None:
    for name in ("estat_bgodfrey", "estat_moran", "estat_dwatson", "estat_vce", "estat_vif"):
        assert name in nm.ESTAT_KEEP
        assert name not in {r.old for r in nm.FUNCTION_RENAMES}


def test_results_are_uniformly_plural_with_collision_exception() -> None:
    """The final decision: *Result -> *Results, except the two semantic opposites."""
    assert nm.RESULT_COLLISION_KEEP == {"ContrastResult", "ContrastResults"}
    assert nm.result_rename("AucResult").new == "AucResults"
    assert nm.result_rename("XtSerialResult").new == "PanelSerialResults"
    assert nm.result_rename("TefectsRaResult").new == "TreatmentEffectsRaResults"


def test_familiar_acronyms_are_not_expanded() -> None:
    new_names = {r.new for r in nm.CLASS_RENAMES}
    assert "InstrumentalVariablesRegress" not in new_names
    assert "DifferenceInDifferencesRegress" not in new_names
    assert "Regress" in new_names  # OLS -> Regress is the explicit special case
    assert "RDRobust" in new_names  # RD kept as a familiar acronym
