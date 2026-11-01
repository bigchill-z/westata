"""Tests for the AJR (2001) replication.

This replication's baseline is **Stata itself**, not a published table: the deposited package
carries no log and the AER article is paywalled.  The tests therefore assert agreement between
two implementations on the authors' own specifications, and they say so in each assertion
message so a future reader does not mistake the claim for a table match.

The joint Wald test gets its own assertion.  It is not a function of any single coefficient
but of the whole covariance matrix, so matching it demonstrates more than matching the
diagonal of ``e(V)`` would.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent / "papers" / "ajr2001"
OUT = PAPER / "out"

sys.path.insert(0, str(PAPER))

pytestmark = pytest.mark.skipif(
    not (OUT / "stata_results.json").exists(),
    reason="run replication/papers/ajr2001/run_all.py first; Stata is required",
)


def _load(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def stata() -> dict:
    return _load("stata_results.json")["specs"]


@pytest.fixture(scope="module")
def westata() -> dict:
    return _load("westata_results.json")["specs"]


SPECS = ("T3_c1", "T3_c2", "T3_c3", "T3_c4", "T4_c1")


@pytest.mark.parametrize("spec", SPECS)
def test_westata_matches_stata_on_every_parameter(stata, westata, spec):
    """Coefficients and clustered standard errors, paired by name.

    This asserts agreement between two implementations, **not** that the paper's table was
    reproduced -- the package has no log and the article is paywalled.
    """

    a, b = stata[spec], westata[spec]
    names = [str(n) for n in a["names"]]
    shared = [n for n in (str(x) for x in b["names"]) if n in names]
    assert shared, f"{spec}: no shared parameter names"

    a_index = [names.index(n) for n in shared]
    b_index = [[str(x) for x in b["names"]].index(n) for n in shared]

    for position, name in enumerate(shared):
        for label, a_value, b_value in (
            ("coefficient", a["b"][a_index[position]], b["b"][b_index[position]]),
            ("SE", a["se"][a_index[position]], b["se"][b_index[position]]),
        ):
            assert abs(b_value - a_value) <= 1e-10 + 1e-9 * abs(a_value), (
                f"{spec}/{name} {label}: stata={a_value!r} westata={b_value!r}"
            )


@pytest.mark.parametrize("spec", SPECS)
def test_the_joint_wald_test_matches(stata, westata, spec):
    """The authors' joint test must agree, which exercises the whole covariance matrix.

    The paper's conclusion lives in this statistic rather than in any single coefficient, so
    a replication that matched the coefficients but not the test would be missing the point.
    """

    a_f = stata[spec]["scalars"].get("joint_F")
    b_f = westata[spec].get("joint_F")
    assert a_f is not None and b_f is not None, f"{spec}: joint test not recorded"
    assert b_f == pytest.approx(a_f, rel=1e-9), f"{spec}: F {b_f} against Stata's {a_f}"


@pytest.mark.parametrize("spec", SPECS)
def test_the_two_sides_use_the_same_sample(stata, westata, spec):
    """A sample restriction applied differently shows up here before it shows up in a slope."""

    assert westata[spec]["nobs"] == stata[spec]["nobs"], (
        f"{spec}: westata N={westata[spec]['nobs']}, stata N={stata[spec]['nobs']}"
    )


def test_the_replication_records_that_there_is_no_published_baseline():
    """The honest-labelling requirement, asserted rather than left to prose.

    If someone later finds the paper's tables and adds them, this test should be updated
    deliberately -- not silently pass while the report still claims a self-consistency run.
    """

    spec = _load("differences.json")
    meta = spec["meta"]
    assert "self-consistency" in meta["baseline_type"].lower(), meta["baseline_type"]
    assert "no author log" in meta["baseline_type"].lower() or "paywalled" in meta["baseline_type"].lower()


def test_panelols_still_rejects_weights():
    """The gap: the documented ``xtreg`` equivalent cannot take the paper's weight.

    Four of the five replicated columns are weighted.  If ``PanelOLS`` ever gains a
    ``weights`` parameter, this test fails and ``ajr2001_gaps.md`` section 1 must become a
    "fixed" entry.
    """

    import inspect

    import westata as ws

    signature = inspect.signature(ws.PanelOLS.from_formula)
    assert "weights" not in signature.parameters, (
        "PanelOLS now accepts weights; ajr2001_gaps.md section 1 needs updating"
    )

    frame = __import__("pandas").read_stata(
        PAPER / "data" / "20100816_replication_dataset.dta", convert_categoricals=False
    )
    with pytest.raises(TypeError):
        ws.PanelOLS.from_formula(
            "urbrate ~ fpresence1750", frame, entity="id", time="year",
            weights=frame["totalpop1750"],
        )


def test_data_provenance_is_pinned():
    provenance = PAPER / "data" / "provenance.json"
    if not provenance.exists():
        pytest.skip("run ajr2001_acquire.py first")
    record = json.loads(provenance.read_text(encoding="utf-8"))
    for name, item in record.items():
        assert len(item["sha256"]) == 64, name
        assert item["bytes"] > 0, name
