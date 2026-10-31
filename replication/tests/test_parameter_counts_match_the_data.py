"""The parameter counts a replication document quotes must match its measured data.

Three numbers were once wrong across these documents: ``adh2013`` claimed 161 paired
parameters where the data contain 178 (106 structural + 72 first-stage), ``ad2013``
claimed 545 where its own REPORT.md already said 546, and a cross-paper total claimed 732
where the data give 750. 161 was plausible -- it even decomposed into "89 structural + 72
first-stage", and 72 IS the real first-stage count -- so it survived review in six places
across four files. A number that looks internally consistent is harder to catch than one
that does not.

WHAT CHANGED AND WHY THE TESTS SHRANK: the documents no longer carry cross-paper material
(see ``test_gaps_docs_are_single_paper``), so two checks were deleted rather than left in:

* ``test_the_cross_paper_table_is_consistent_across_all_six_documents`` -- it looked for
  three-column comparison rows, which no longer exist. It matched nothing and its
  assertion could never fire. A test that cannot fail is not a guard; it is a comment
  that looks like one.
* ``test_the_three_paper_total_is_the_sum`` -- it asserted 750, a total no document
  states any more. Keeping it would assert a number nothing claims.

Each paper's own count is still checked, and so is each paper's own precision figure,
against the data the report was generated from.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest

PAPERS = pathlib.Path(__file__).resolve().parents[1] / "papers"

DOC_FILES = {
    "adh2013": ("REPORT.md", "WESTATA_GAPS.md"),
    "ck1994": ("REPORT.md", "WESTATA_GAPS.md"),
    "ad2013": ("REPORT.md", "WESTATA_GAPS.md"),
}

#: Each paper's own paired-parameter count, measured from ``out/stata_results.json``.
EXPECTED_TOTALS = {"adh2013": 178, "ck1994": 26, "ad2013": 546}

#: Each paper's own worst relative coefficient difference, measured from
#: ``out/differences.json``, to two significant figures as the reports quote it.
EXPECTED_PRECISION = {"adh2013": 1.6e-11, "ck1994": 2.1e-13, "ad2013": 2.9e-11}


def _measured_counts(paper: str) -> dict[str, int]:
    """Paired parameters, counted from the comparison data rather than from prose."""

    path = PAPERS / paper / "out" / "stata_results.json"
    if not path.exists():  # pragma: no cover - committed artefact
        pytest.skip(f"{path} not present")

    data = json.loads(path.read_text(encoding="utf-8"))
    structural = sum(len(spec["b"]) for spec in data["specs"].values())
    # ``first_stage`` carries a ``*_robust`` twin per specification; count each once.
    first_stage = sum(
        len(spec["b"])
        for name, spec in data.get("first_stage", {}).items()
        if not name.endswith("_robust")
    )
    return {
        "structural": structural,
        "first_stage": first_stage,
        "total": structural + first_stage,
    }


def _worst_relative_difference(paper: str) -> float:
    """Largest relative coefficient difference the comparison data record."""

    path = PAPERS / paper / "out" / "differences.json"
    if not path.exists():  # pragma: no cover - committed artefact
        pytest.skip(f"{path} not present")
    data = json.loads(path.read_text(encoding="utf-8"))
    return max(spec["max_rel_b"] for spec in data["specs"])


# --------------------------------------------------- the counts, per paper
@pytest.mark.parametrize("paper", sorted(EXPECTED_TOTALS))
def test_the_count_is_what_the_data_contain(paper: str) -> None:
    """The measured total must equal the number the documents are written against."""

    measured = _measured_counts(paper)

    assert measured["total"] == EXPECTED_TOTALS[paper], (
        f"{paper}: measured {measured['total']} paired parameters "
        f"({measured['structural']} structural + {measured['first_stage']} first-stage), "
        f"but the documents say {EXPECTED_TOTALS[paper]}"
    )


@pytest.mark.parametrize("paper", sorted(DOC_FILES))
def test_no_document_quotes_a_retired_count(paper: str) -> None:
    """161 / 545 / 732 were all wrong; none should reappear in this paper's files."""

    retired = ("161", "545", "732")
    offenders = []
    for filename in DOC_FILES[paper]:
        path = PAPERS / paper / filename
        if not path.exists():  # pragma: no cover
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for token in retired:
                # Whole-token match only: decimals and longer numbers are legitimate.
                if re.search(rf"(?<![0-9.]){token}(?![0-9])", line):
                    offenders.append(f"{filename}:{number}: {line.strip()[:90]}")

    assert offenders == [], (
        "documents still quote a retired parameter count:\n" + "\n".join(offenders)
    )


# ------------------------------------------------- the precision, per paper
@pytest.mark.parametrize("paper", sorted(EXPECTED_PRECISION))
def test_the_headline_precision_is_the_measured_one(paper: str) -> None:
    """Where a document quotes a worst case, it must match the data."""

    actual = _worst_relative_difference(paper)
    quoted = EXPECTED_PRECISION[paper]

    assert actual == pytest.approx(quoted, rel=0.1), (
        f"{paper}: documents quote {quoted:.1e} but the data give {actual:.3e}"
    )


@pytest.mark.parametrize("paper", sorted(DOC_FILES))
def test_no_document_claims_precision_better_than_it_has(paper: str) -> None:
    """A report must not round its own error DOWN.

    ``adh2013``'s worst difference is 1.56e-11 and ``ad2013``'s is 2.87e-11. Both
    documents once said "1e-11 以内" ("within 1e-11"), which is false for both -- the
    numbers are ABOVE 1e-11. It is a small flattery, but this repository exists to not do
    that, and a reader who checks the table against the sentence finds them contradicting
    each other.

    The rule: where a document states a bound, the measured worst case must satisfy it.
    "量级" (order of magnitude) is allowed because it claims no exact bound; "以内"
    (within) is not, unless it holds.
    """

    worst = _worst_relative_difference(paper)
    problems = []

    for filename in DOC_FILES[paper]:
        path = PAPERS / paper / filename
        if not path.exists():  # pragma: no cover
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "1e-11 以内" in line or "1e-11以内" in line:
                if worst > 1e-11:
                    problems.append(
                        f"{filename}:{number} claims 'within 1e-11' but the worst "
                        f"measured difference is {worst:.3e}"
                    )

    assert problems == [], "\n".join(problems)