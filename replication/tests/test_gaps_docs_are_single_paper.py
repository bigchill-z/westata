"""Each paper's gap list must describe only that paper's replication.

The three replication docs originally cross-referenced each other: "why this list differs
from the first one", a three-paper cumulative table, "与前两篇的关系", a closing
"附：三篇对照". That material belongs to the series as a whole, not to any one paper's
findings, and it made each document harder to read on its own -- a reader of the AD 2013
list had to hold two other papers' conclusions in mind to understand it.

The rule enforced here: a gap list talks about ITS paper. Cross-paper totals,
comparisons and links are out.

Two things are NOT violations and are excluded deliberately:

* academic citations -- adh2013/REPORT.md mentions three methodology papers that use
  shift-share IV as a target, which has nothing to do with this replication series;
* a paper naming itself (``ad2013`` appearing in ``ad2013/REPORT.md``).

Also checked: the section numbers the guard tests cite still exist, because deleting §0
and §6 shifted the numbering and a dangling "see section 4" would be a new defect.
"""

from __future__ import annotations

import pathlib
import re

import pytest

PAPERS = pathlib.Path(__file__).resolve().parents[1] / "papers"

DOCS = [
    ("adh2013", "WESTATA_GAPS.md"),
    ("adh2013", "REPORT.md"),
    ("ck1994", "WESTATA_GAPS.md"),
    ("ck1994", "REPORT.md"),
    ("ad2013", "WESTATA_GAPS.md"),
    ("ad2013", "REPORT.md"),
    ("adhm2020", "WESTATA_GAPS.md"),
    ("adhm2020", "REPORT.md"),
]

#: Phrases that only make sense when talking about ANOTHER paper in this series.
#: ``三篇`` is excluded because adh2013/REPORT.md legitimately uses it for three ACADEMIC
#: methodology papers (Goldsmith-Pinkham et al., Borusyak et al., Adão et al.) that use
#: shift-share IV as a target -- nothing to do with this replication series. The
#: replication-series uses ("三篇复现", "三篇的累计") are caught by the more specific
#: phrases below.
CROSS_PAPER = (
    "前两篇",
    "第一篇",
    "第二篇",
    "前一篇",
    "与前两篇的关系",
    "合起来的结论",
    "累计视图",
    "三篇复现",
    "三篇的累计",
    "三篇对照",
    "三篇一致",
)

#: Table headers naming the other papers as comparison columns.
COMPARISON_TABLE = (
    "| ADH 2013 |",
    "| CK 1994 |",
    "| AD 2013 |",
)


def _paper_text(paper: str, filename: str) -> str:
    path = PAPERS / paper / filename
    if not path.exists():  # pragma: no cover - committed files
        pytest.skip(f"{path} not present")
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("paper,filename", DOCS)
def test_a_document_does_not_discuss_other_papers(paper: str, filename: str) -> None:
    """The main rule, stated as plainly as possible."""

    text = _paper_text(paper, filename)
    offenders = []
    for number, line in enumerate(text.splitlines(), 1):
        for phrase in CROSS_PAPER:
            if phrase in line:
                offenders.append(f"{paper}/{filename}:{number}: {line.strip()[:90]}")

    assert offenders == [], (
        "these lines discuss another replication rather than this one:\n"
        + "\n".join(offenders)
    )


@pytest.mark.parametrize("paper,filename", DOCS)
def test_a_document_does_not_carry_a_comparison_table(paper: str, filename: str) -> None:
    """A three-column table of the series is a series artefact, not a paper finding."""

    text = _paper_text(paper, filename)
    offenders = [
        f"{paper}/{filename}:{number}: {line.strip()[:90]}"
        for number, line in enumerate(text.splitlines(), 1)
        if any(marker in line for marker in COMPARISON_TABLE)
    ]

    assert offenders == [], "comparison table found:\n" + "\n".join(offenders)


@pytest.mark.parametrize("paper,filename", DOCS)
def test_a_document_does_not_link_to_another_paper(paper: str, filename: str) -> None:
    text = _paper_text(paper, filename)
    offenders = []
    for number, line in enumerate(text.splitlines(), 1):
        if re.search(r"\.\./(?:adh2013|ck1994|ad2013)/", line):
            offenders.append(f"{paper}/{filename}:{number}: {line.strip()[:90]}")

    assert offenders == [], "cross-document link found:\n" + "\n".join(offenders)


# ------------------------------------- the section numbers the guards cite
@pytest.mark.parametrize(
    "paper,filename,cited",
    [
        ("ad2013", "WESTATA_GAPS.md", [1, 2, 4]),
        ("ck1994", "WESTATA_GAPS.md", [2, 3]),
    ],
)
def test_the_sections_the_guard_tests_cite_still_exist(
    paper: str, filename: str, cited: list[int]
) -> None:
    """The guard tests say "see section N"; removing sections must not dangle those."""

    text = _paper_text(paper, filename)
    headings = {int(match.group(1)) for match in re.finditer(r"^## (\d+)\.", text, re.M)}

    missing = [number for number in cited if number not in headings]
    assert missing == [], (
        f"{paper}/{filename} no longer has section(s) {missing}; headings present: "
        f"{sorted(headings)}"
    )