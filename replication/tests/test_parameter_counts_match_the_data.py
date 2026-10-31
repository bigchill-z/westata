"""The parameter counts the replication documents quote must match the measured data.

Three numbers were wrong across the papers' documents: ``adh2013`` claimed 161 paired
parameters where the comparison data contain 178 (106 structural + 72 first-stage),
``ad2013`` claimed 545 where its own REPORT.md already said 546, and the three-paper total
claimed 732 where the data give 750.

The failure mode is worth naming, because it is not a typo someone would spot: 161 was
plausible (it even decomposed into "89 structural + 72 first-stage", and 72 is the real
first-stage count), so it survived review in six places across four files. A number that
looks internally consistent is harder to catch than one that does not.

These tests recompute from ``out/stata_results.json`` -- the artefact the reports are
generated from -- and compare against what the prose claims. That way the prose cannot
drift from the data without a failure, which is the same guarantee the probes give for
capability claims.
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


def test_the_counts_are_what_the_documents_claim() -> None:
    """The three totals, as recomputed from the data."""

    expected = {"adh2013": 178, "ck1994": 26, "ad2013": 546}

    for paper, total in expected.items():
        measured = _measured_counts(paper)
        assert measured["total"] == total, (
            f"{paper}: measured {measured['total']} paired parameters "
            f"({measured['structural']} structural + {measured['first_stage']} "
            f"first-stage), but the documents say {total}"
        )


def test_the_three_paper_total_is_the_sum() -> None:
    total = sum(_measured_counts(paper)["total"] for paper in ("adh2013", "ck1994", "ad2013"))

    assert total == 750


# ------------------------------------------------- the prose agrees with the data
@pytest.mark.parametrize("paper", sorted(DOC_FILES))
def test_no_document_quotes_a_retired_count(paper: str) -> None:
    """161 / 545 / 732 were all wrong; none should reappear."""

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

    assert offenders == [], "documents still quote a retired parameter count:\n" + "\n".join(offenders)


def test_the_cross_paper_table_is_consistent_across_all_six_documents() -> None:
    """The comparison table repeats adh2013 / ck1994 / ad2013 in that column order.

    Every document carries the same three numbers, so they must agree with each other and
    with the data. This is the check that would have caught "545" in one file while a
    sibling said "546": the two could not both be right.
    """

    expected = [_measured_counts(p)["total"] for p in ("adh2013", "ck1994", "ad2013")]
    problems = []

    for paper, filenames in DOC_FILES.items():
        for filename in filenames:
            path = PAPERS / paper / filename
            if not path.exists():  # pragma: no cover
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not re.search(r"\| *(?:参数数|成对参数) *\|", line):
                    continue
                cells = [cell.strip().strip("*") for cell in line.strip().strip("|").split("|")]
                numbers = [cell for cell in cells[1:] if cell.isdigit()]
                if len(numbers) != 3:
                    continue
                stated = [int(value) for value in numbers]
                if stated != expected:
                    problems.append(
                        f"{paper}/{filename}:{number} states {stated}, measured {expected}"
                    )

    assert problems == [], "\n".join(problems)