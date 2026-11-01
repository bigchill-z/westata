"""Tests for the ADHM (2020) replication.

What these tests are for
------------------------
The coefficients are easy -- five numbers against the authors' log.  The assertions that
carry information are about the **standard errors**, because that is the part of this paper
that required work:

* ``ivreg2`` two-way clustered standard errors must reproduce the authors' log exactly.
  This is the replication claim, and it needs the user-written package to be installed.
* ``westata`` single-way clustering must match ``ivregress`` to machine precision.
* ``westata`` must still **reject** multiway clustering -- if it ever starts working, the
  report is stale and a reader would be told the wrong thing.
* ``westata``'s ``FEOLS`` must still **accept** it -- that asymmetry is the internal
  inconsistency the gaps document describes.
* The CGM identity must hold on this data, because it is the evidence that the gap is a
  combination step rather than a missing method.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent / "papers" / "adhm2020"
OUT = PAPER / "out"

sys.path.insert(0, str(PAPER))

pytestmark = pytest.mark.skipif(
    not (OUT / "stata_results.json").exists(),
    reason="run replication/papers/adhm2020/run_all.py first; Stata and ivreg2 are required",
)


def _load(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def stata() -> dict:
    return _load("stata_results.json")


@pytest.fixture(scope="module")
def westata() -> dict:
    return _load("westata_results.json")


def _coef(record: dict, name: str) -> tuple[float, float]:
    index = record["names"].index(name)
    return float(record["b"][index]), float(record["se"][index])


ENDOG = "d_imp_usch_pd"
CLUSTERS = ("czone", "congressionaldistrict")

#: ``(spec, coefficient, two-way standard error)`` from the authors' log.
PUBLISHED = (
    ("T3_A1", 15.83783, 9.236939),
    ("T3_A2", 28.69324, 21.29582),
    ("T3_A3", 33.30375, 21.12325),
    ("T3_A4", 38.96939, 21.22885),
    ("T3_A5", 37.23199, 21.04598),
)


@pytest.mark.parametrize("spec,paper_b,paper_se", PUBLISHED)
def test_ivreg2_reproduces_the_authors_log_including_two_way_se(
    stata, spec, paper_b, paper_se
):
    """The replication claim: both the coefficient and the two-way clustered SE come back.

    This is the test that would have been impossible before ``ivreg2`` was installed, and
    its absence was the flaw in the first version of this replication -- the paper's own
    standard errors were quoted from the log rather than reproduced.
    """

    b, se = _coef(stata["ivreg2"][spec], ENDOG)
    assert b == pytest.approx(paper_b, abs=1e-4), f"{spec}: coefficient {b}"
    assert se == pytest.approx(paper_se, rel=1e-6), f"{spec}: two-way SE {se}"


def test_ivreg2_actually_ran_with_two_way_clustering(stata):
    """Guard against the test passing for the wrong reason.

    If someone replaced ``ivreg2`` with ``ivregress`` and a single cluster, the loop above
    would still pass on the coefficient but the standard errors would drift by up to 11
    percent.  Asserting the command string and the package status pins what actually ran.
    """

    for spec, _, _ in PUBLISHED:
        command = stata["ivreg2"][spec]["command"]
        assert command.startswith("ivreg2"), command
        assert "cluster(czone congressionaldistrict)" in command, command
    packages = stata["meta"]["packages"]
    assert packages.get("ivreg2") and "MISSING" not in packages["ivreg2"]


@pytest.mark.parametrize("spec,paper_b,paper_se", PUBLISHED)
def test_westata_matches_the_coefficient(stata, westata, spec, paper_b, paper_se):
    """westata's coefficient is the same whatever clustering it can manage."""

    for cluster in CLUSTERS:
        b, _ = _coef(westata["specs"][spec][cluster], ENDOG)
        assert b == pytest.approx(paper_b, abs=1e-4), f"{spec} {cluster}: {b}"


def test_single_way_standard_errors_match_stata(westata, stata):
    """Where the capability exists, it is exact -- on all ten regressions."""

    worst = 0.0
    for spec, _, _ in PUBLISHED:
        for cluster in CLUSTERS:
            _, sse = _coef(stata["ivregress"][spec][cluster], ENDOG)
            _, wse = _coef(westata["specs"][spec][cluster], ENDOG)
            rel = abs(wse - sse) / abs(sse)
            worst = max(worst, rel)
            assert rel < 1e-9, f"{spec} {cluster}: {rel:.2e}"
    assert worst < 1e-9


def test_two_way_clustering_is_still_unsupported_and_says_so(westata):
    """The limitation must persist and must be an explicit refusal, not a wrong number."""

    for spec, _, _ in PUBLISHED:
        record = westata["twoway"][spec]
        assert record["supported"] is False, f"{spec}: two-way clustering now works"
        assert record["error"] == "UnsupportedError", (
            f"{spec}: rejected with {record['error']}; a different exception type suggests "
            "a bug rather than an unimplemented feature"
        )


def test_the_single_way_fallback_understates_the_two_way_standard_error(westata):
    """The cost of the limitation has a consistent sign, which is why it matters.

    Both single-way standard errors sit below the two-way one, so substituting one is not
    "a slightly different number" but a systematic understatement of uncertainty.
    """

    payload = _load("differences.json")
    for rec in payload["specs"]:
        assert rec["understatement_czone"] > 0, rec["spec"]
        assert rec["understatement_cdist"] > 0, rec["spec"]
        # The largest case should stay material -- if the gap collapsed to nothing the
        # report's emphasis would be wrong.
    assert max(r["understatement_czone"] for r in payload["specs"]) > 0.05


def test_the_probe_reports_no_gaps_and_keeps_the_comparisons():
    """Guard ``adhm2020_gaps.md``: both gaps are fixed, the parity note stands.

    This test used to run the other way -- it asserted that two-way clustering was
    REFUSED for IV and for OLS, because the document recorded those as open gaps. They
    were fixed by implementing the Cameron-Gelbach-Miller decomposition, so the guard
    inverts with the facts: a regression that brings either refusal back should fail
    here rather than quietly contradicting the document.

    The parity note is still asserted: stock ``ivregress`` genuinely rejects multiway
    clusters, so westata matching ``ivreg2`` is not the same claim as stock Stata being
    able to do it. That distinction survives the fix.
    """

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "adhm2020_probe_gaps", PAPER / "adhm2020_probe_gaps.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["adhm2020_probe_gaps"] = module
    spec.loader.exec_module(module)

    statuses = {p.name: p.run() for p in (*module.PROBES, *module.CONTROLS)}

    assert not [n for n, (s, _) in statuses.items() if s == "ERROR"], statuses

    assert statuses["multiway cluster (IV)"][0] == "OK", (
        "two-way IV clustering regressed; adhm2020_gaps.md §2 says it produces the "
        f"paper's 9.236939. Probe said: {statuses['multiway cluster (IV)'][1]}"
    )
    assert statuses["vs stock Stata (IV)"][0] == "NOTE", (
        "the stock-Stata parity note no longer applies"
    )
    assert statuses["multiway cluster (OLS)"][0] == "OK", (
        "two-way OLS clustering regressed; adhm2020_gaps.md §3 says it works"
    )
    assert statuses["FEOLS two-way cluster"][0] == "OK", (
        "FEOLS stopped accepting two-way groups; adhm2020_gaps.md §4's claim that all "
        "three estimators agree no longer holds"
    )
    assert statuses["CGM decomposition"][0] == "OK", (
        "the CGM identity no longer holds on this data, so the implementation does not "
        "match what the gaps document describes"
    )


def test_data_provenance_is_pinned():
    provenance = PAPER / "data" / "provenance.json"
    if not provenance.exists():
        pytest.skip("run adhm2020_acquire.py first")
    record = json.loads(provenance.read_text(encoding="utf-8"))
    for name, item in record.items():
        assert len(item["sha256"]) == 64, name
        assert item["bytes"] > 0, name
