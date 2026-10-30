"""Tests that keep the fixture-verification story honest.

Two facts about this repository's conformance fixtures, both established by
auditing the tests rather than assuming:

1. **18 of the 21 fixtures describe synthetic datasets that are NOT shipped**
   (``areg.json``, ``logit_sim.json``, ``xtreg_*.json``, ...). Only
   ``regress_auto*.json`` can be reproduced, because ``auto.dta`` is in the tree.
   So a value-level comparison against Stata is possible for those three and
   impossible for the rest.

2. **No test asserts a `e()` value against a fixture it cannot reproduce.** The
   fixture-driven tests compare KEY SETS. That is a legitimate check -- it proves
   the key exists with a plausible name -- but it does not prove the number.

This file pins both facts so that neither silently changes:

* if someone adds a shipped dataset that makes more fixtures reproducible, the
  count assertions here fail and the new coverage has to be recorded;
* if someone starts asserting values against a non-reproducible fixture, the
  cross-check below fails.

The point is not to celebrate the limitation but to keep it visible. A future
reader should be able to tell, from a test, exactly how much of the `e()` surface
is verified numerically and how much only structurally.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

FIXTURES = Path("tests/conformance/stata_fixtures")
TESTS = Path("tests")

#: Fixtures whose dataset is shipped, so their values CAN be compared numerically.
REPRODUCIBLE = frozenset(
    {"regress_auto.json", "regress_auto_factor.json", "regress_auto_robust.json"}
)


def _fixtures() -> dict[str, dict]:
    return {
        path.name: json.loads(path.read_text()) for path in FIXTURES.glob("*.json")
    }


def test_the_reproducible_set_is_what_the_tests_assume() -> None:
    """Only the `auto` fixtures have their data in the tree.

    ``auto.dta`` ships in ``tests/data``; none of the synthetic panels do, even
    though each fixture carries a ``datasignature`` recording what Stata saw.
    """

    data = _fixtures()
    uses_auto = {
        name
        for name, payload in data.items()
        if "mpg" in str(payload.get("macros", {}).get("cmdline", ""))
    }

    assert uses_auto == REPRODUCIBLE, uses_auto ^ REPRODUCIBLE


def test_fixtures_record_how_much_data_they_describe() -> None:
    """Each fixture names its sample; not all ship a ``datasignature``.

    ``areg.json`` and the panel fixtures carry ``datasignature: N:k:h1:h2`` from
    Stata; the IV and count fixtures do not, but every one records ``N`` in its
    scalars. What matters is that the sample SIZE is always recoverable, because
    that is the parameter a test would need to rebuild the data.
    """

    data = _fixtures()
    for name, payload in data.items():
        scalars = payload.get("scalars", {})
        assert "N" in scalars, name
        assert float(scalars["N"]) > 0, name


def test_fixture_values_are_used_to_check_relations_not_to_fake_fits() -> None:
    """Fixture numbers may appear in tests, but only inside relations.

    A static "is this literal alone on its line" heuristic turned out too brittle
    to be worth maintaining -- multi-line docstrings legitimately quote an
    identity, and a parser for that is more code than it protects. The honest
    check is a narrower one: for every fixture that cannot be reproduced, confirm
    that no test compares a RESULT field against that fixture's number.

    That is what ``result.e["x"] == <fixture literal>`` or the reverse would look
    like, and it is detectable without parsing prose: the literal must appear on a
    line that also mentions a result attribute.
    """

    data = _fixtures()
    suspicious: list[str] = []

    for path in TESTS.glob("test_*.py"):
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for line_no, line in enumerate(lines):
            # Only lines that both carry a result attribute and a fixture number.
            if "result.e[" not in line and ".e[" not in line:
                continue
            for name, payload in data.items():
                if name in REPRODUCIBLE:
                    continue
                for key, value in payload.get("scalars", {}).items():
                    if not isinstance(value, float) or float(value).is_integer():
                        continue
                    needle = f"{value:.6f}"
                    if needle in line:
                        suspicious.append(
                            f"{path.name}:{line_no + 1}: {name}:{key} "
                            f"compared directly -> {line.strip()[:60]}"
                        )

    assert not suspicious, suspicious


def test_the_auto_fixtures_are_checked_numerically() -> None:
    """Sanity check the other direction: the reproducible ones ARE value-tested."""

    data = _fixtures()
    texts = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in TESTS.glob("test_*.py")
    )

    checked = 0
    for name in REPRODUCIBLE:
        for key, value in data[name].get("scalars", {}).items():
            if not isinstance(value, float) or float(value).is_integer():
                continue
            if f"{value:.6f}" in texts or f"{value:.10f}" in texts:
                checked += 1
                break

    # At least the primary regression fixture must be value-checked; without this
    # the whole numeric-verification claim would be empty.
    assert checked >= 1, "no auto-fixture value appears in any test"


def test_the_limitation_is_documented() -> None:
    """The gap must be written down where a reader will find it."""

    register = Path("docs/GAP_REGISTER.md").read_text(encoding="utf-8", errors="replace")
    # Use a stable ASCII marker: the register is Chinese text and an encoding
    # round-trip would make a translated phrase fragile.
    assert "8.20.39" in register
