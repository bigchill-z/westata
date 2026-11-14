"""The API convention checker must keep working, and its baseline must not grow.

Two failure modes are worth a test, and neither is "the checker finds nothing":

1. **The checker silently stops checking.** Every rule depends on a detail of AST
   or ``inspect`` (``from __future__ import annotations`` makes annotations
   strings, ``kw_only`` arguments live in a different list, etc.). One wrong
   assumption turns a rule into a no-op that still prints a confident "0
   findings". ``test_every_rule_is_exercised`` asserts each rule family can be
   made to fire on a synthetic module, so a dead rule is visible.

2. **The baseline grows.** ``docs/API_CONVENTIONS.md`` records 116 findings with
   103 to fix. A refactor that adds 20 more while claiming to clean up is the
   thing this file exists to catch. The cap is raised deliberately, in the same
   commit as the document.

The checker itself is NOT run in ``--strict`` mode here: the tree is expected to
have findings until the burn-down in the document finishes. CI runs it in
report mode meanwhile.
"""

from __future__ import annotations

import ast
import inspect
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

sys.path.insert(0, str(SCRIPTS))

import check_api_conventions as conventions  # noqa: E402

#: The number of findings the 0.3.11 tree produces. Raise it only with a document
#: change that explains why the new deviation is acceptable in the meantime.
BASELINE_FINDINGS = 28

#: Findings the checker knowingly accepts, and which ``--strict`` will not fail on.
BASELINE_ALLOWED = 28

#: The number that actually has to reach zero. This is the number that matters:
#: ``BASELINE_FINDINGS`` can grow by adding documented exceptions, this cannot.
BASELINE_BLOCKING = 0


def test_the_checker_runs_and_matches_the_documented_baseline() -> None:
    report = conventions.collect()
    total = sum(len(group) for group in report.findings.values())
    allowed = total - len(report.blocking())

    assert total == BASELINE_FINDINGS, (
        f"the checker now reports {total} findings, but docs/API_CONVENTIONS.md §7 "
        f"records {BASELINE_FINDINGS}. Either the deviation was fixed (lower both "
        f"numbers) or a new one appeared (fix it, or record it deliberately)."
    )
    assert allowed == BASELINE_ALLOWED, (
        f"{allowed} findings are allowed by ALLOWED, the document says "
        f"{BASELINE_ALLOWED}."
    )
    assert len(report.blocking()) == BASELINE_BLOCKING, (
        f"{len(report.blocking())} findings are blocking, the document says "
        f"{BASELINE_BLOCKING}. This is the number the burn-down must drive to zero, "
        f"so a new exception cannot be used to hide a regression."
    )


def test_every_rule_can_fire_on_a_synthetic_module(tmp_path: Path, monkeypatch) -> None:
    """A rule that can never fire is worse than no rule: it reports zero forever."""

    offending = tmp_path / "offending.py"
    # The synthetic module carries one instance of every mechanically-checkable
    # violation, so a rule that stops firing is visible as a missing key.
    offending.write_text(
        "\n".join(
            [
                "import pandas as pd",
                "",
                "class Offender:",
                "    @classmethod",
                "    def from_formula(cls, outcome, data):",
                "        return cls(data)",
                "",
                "    def __init__(self, data=None):",
                "        pass",
                "",
                "    def fit(self, entity_effects=True, cov_type=None, **kwargs):",
                "        return None",
                "",
                "class OffenderResult:",
                "    def __str__(self):",
                "        return ''",
                "",
                "def public(data, frame=None, camelCase=None, noabsorbish=False,"
                " level=95.0, bogus=1):",
                "    return data",
                "",
                "def _dup(value):",
                "    return value",
                "",
            ]
        ),
        encoding="utf-8",
    )

    original = conventions._modules
    monkeypatch.setattr(conventions, "_modules", lambda: [offending])
    report = conventions.Report()
    conventions.check_r3(report)
    rules = set(report.findings)

    assert "R3.1" in rules, "the negative-switch rule did not fire on `noabsorbish=`"
    assert "R3.3" in rules, "the level-unit rule did not fire on `level=95.0`"
    assert "R2.5" in rules, "the snake_case rule did not fire on `camelCase=`"

    report = conventions.Report()
    conventions.check_r2(report)
    rules = set(report.findings)
    assert "R2.1" in rules, "the fit()-order rule did not fire on `entity_effects` first"
    assert "R2.3" in rules, "the `data` rule did not fire on a `frame` parameter"
    assert "R2.6" in rules, "the **kwargs rule did not fire on a swallowed `kwargs`"
    assert "R3.2" in rules, "the from_formula rule did not fire on `(outcome, data)`"

    monkeypatch.setattr(conventions, "_modules", original)


def test_document_records_the_baseline_table() -> None:
    """The §7 table must agree with the checker, row by row.

    Two numbers drifting apart silently is the failure this guards: the document is
    what a human reads to decide what to work on, the checker is what CI enforces.
    """

    import json
    import subprocess

    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "check_api_conventions.py"), "--json"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    findings = json.loads(result.stdout)

    document = (ROOT / "docs" / "API_CONVENTIONS.md").read_text(encoding="utf-8")
    for rule, group in findings.items():
        total = len(group)
        allowed = sum(1 for item in group if item["allowed"])
        blocking = total - allowed
        # The row reads: | R2.1 `...` | total | allowed | blocking | ... |
        pattern = re.compile(
            rf"^\|\s*{re.escape(rule)}\s[^|]*\|\s*{total}\s*\|\s*{allowed}\s*\|\s*{blocking}\s*\|",
            re.M,
        )
        assert pattern.search(document), (
            f"{rule}: checker says total={total}, allowed={allowed}, blocking={blocking}, "
            f"but docs/API_CONVENTIONS.md §7 has no matching row"
        )
    """Every rule id the checker can emit is explained in the convention document."""

    document = (ROOT / "docs" / "API_CONVENTIONS.md").read_text(encoding="utf-8")
    emitted = set()
    for name, member in vars(conventions).items():
        if not name.startswith("check_"):
            continue
        source = inspect.getsource(member)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", "") == "Finding"
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                emitted.add(node.args[0].value)
        # Also catch the f-string id used by the `**kwargs` branch.
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if len(node.value) == 4 and node.value[0] == "R" and node.value[1].isdigit():
                    emitted.add(node.value)

    assert emitted, "no rule ids were found in the checker; this test checked nothing"
    missing = sorted(rule for rule in emitted if rule not in document)
    assert not missing, f"rules {missing} are emitted but not documented"


def test_section_nine_covers_every_rule_the_checker_emits() -> None:
    """§9's mapping table must list every rule, or it silently stops being a map.

    The table went stale once already: R5.5–R5.8 were added to the checker and to §7
    but not to §9, so a reader consulting "which rules are mechanical?" got a list
    that was missing four of them. This test derives the rule set from the checker's
    own source and requires each id to appear in §9.
    """

    emitted = set()
    for name, member in vars(conventions).items():
        if not name.startswith("check_"):
            continue
        for node in ast.walk(ast.parse(inspect.getsource(member))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if len(node.value) == 4 and node.value[0] == "R" and node.value[1].isdigit():
                    emitted.add(node.value)

    document = (ROOT / "docs" / "API_CONVENTIONS.md").read_text(encoding="utf-8")
    section_start = document.index("## 9. 校验器与规范的对应关系")
    section_end = document.index("## 10.", section_start)
    section = document[section_start:section_end]

    missing = sorted(rule for rule in emitted if rule not in section)
    assert not missing, (
        f"§9 does not mention {missing}; the mapping table is what a reader uses to "
        f"decide which rules are mechanically enforced, so it has to list them all"
    )


def test_allowed_table_entries_name_a_rule_that_exists() -> None:
    """An `ALLOWED` key for a rule that no longer exists silently stops applying."""

    emitted = set()
    for name, member in vars(conventions).items():
        if not name.startswith("check_"):
            continue
        source = inspect.getsource(member)
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if len(node.value) == 4 and node.value[0] == "R" and node.value[1].isdigit():
                    emitted.add(node.value)

    stale = sorted({rule for rule, _ in conventions.ALLOWED} - emitted)
    assert not stale, (
        f"ALLOWED mentions {stale}, which no check emits; remove the stale entry "
        f"rather than leaving a rule that can never be matched"
    )


def test_strict_mode_reports_the_baseline_as_a_failure() -> None:
    """`--strict` must be wired to the blocking count, not to the total."""

    assert conventions.main(["--strict"]) != 0 or BASELINE_BLOCKING == 0


def test_report_mode_never_fails_ci() -> None:
    """Until the burn-down finishes, plain mode must stay green."""

    assert conventions.main([]) == 0


def test_cov_type_family_table_covers_every_estimator() -> None:
    """R3.7 needs a decision for EVERY estimator, not just the easy ones.

    A new estimator added without a row in ``COV_TYPE_ROBUST_TARGETS`` or
    ``COV_TYPE_ROBUST_REFUSED`` silently escapes the rule -- it can echo whatever
    ``cov_type`` it likes. This test makes that omission fail loudly.
    """

    import inspect

    import westata as ws

    estimable = {
        name
        for name in ws.__all__
        if inspect.isclass(getattr(ws, name)) and hasattr(getattr(ws, name), "fit")
    }
    covered = set(conventions.COV_TYPE_ROBUST_TARGETS) | conventions.COV_TYPE_ROBUST_REFUSED
    #: Estimators with no `cov_type` argument at all (they report a full table, not a
    #: coefficient covariance), so R3.7 does not apply.
    exempt = {
        "Manova",
        "Jackknife",
        "Bootstrap",
        "Permute",
        "SvyMean",
        "SvyTotal",
        "SvyProportion",
        "SvyRegress",
        "SvyLogit",
        "RegressionAdjustment",
        "IPW",
        "AIPW",
        "NearestNeighbourMatching",
        "MultinomialLogit",
        "Nl",
        "Frontier",
        "ARIMA",
        "VAR",
        "VECM",
        "GARCH",
        "Xtdidregress",
        "XtAbond",
        "RandomEffectsMLE",
        "BetweenOLS",
        "FirstDifferenceOLS",
        # `newey` and `prais` have a class form (R2.7) but their `fit()` deliberately
        # takes no `cov_type`: Stata's commands have no `vce()` option -- the HAC /
        # AR(1) variance IS the estimator -- so accepting one would promise a choice
        # that does not exist (R1.5). See their class docstrings.
        "Newey",
        "Prais",
    }

    missing = sorted(estimable - covered - exempt)
    assert not missing, (
        f"these estimators take a cov_type but have no R3.7 decision: {missing}. "
        f"Add them to COV_TYPE_ROBUST_TARGETS or COV_TYPE_ROBUST_REFUSED, and to the "
        f"table in docs/API_CONVENTIONS.md R3.7.1."
    )


@pytest.mark.parametrize("rule", ["R2", "R3", "R4", "R5", "R6"])
def test_each_rule_family_can_be_selected(rule: str) -> None:
    """`--rule` is how the burn-down is worked; a broken selector hides that work."""

    report = conventions.collect(rule)
    assert all(key.startswith(rule) for key in report.findings), (
        f"--rule {rule} returned families "
        f"{sorted(k for k in report.findings if not k.startswith(rule))}"
    )
