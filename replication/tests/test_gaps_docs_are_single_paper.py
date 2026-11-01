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

Location
--------
Every gap list lives in ``replication/gaps/<paper>_gaps.md`` rather than beside its paper.
Keeping them together is the point -- a reader asking "what does westata still need" should
find one directory, not five.  ``test_every_gap_list_is_in_the_gaps_directory`` enforces the
layout, so a future paper cannot quietly reintroduce a stray file.
"""

from __future__ import annotations

import pathlib
import re

import pytest

REPLICATION = pathlib.Path(__file__).resolve().parents[1]
PAPERS = REPLICATION / "papers"
#: Every gap list now lives in one directory instead of beside its paper.
GAPS = REPLICATION / "gaps"

#: ``(paper, filename, directory)``.  Gap lists live in ``replication/gaps`` with a
#: ``<paper>_gaps.md`` name; reports stay beside their code.
DOCS = [
    ("adh2013", "adh2013_gaps.md", GAPS),
    ("adh2013", "REPORT.md", PAPERS / "adh2013"),
    ("ck1994", "ck1994_gaps.md", GAPS),
    ("ck1994", "REPORT.md", PAPERS / "ck1994"),
    ("ad2013", "ad2013_gaps.md", GAPS),
    ("ad2013", "REPORT.md", PAPERS / "ad2013"),
    ("nq2011", "nq2011_gaps.md", GAPS),
    ("nq2011", "REPORT.md", PAPERS / "nq2011"),
    ("adhm2020", "adhm2020_gaps.md", GAPS),
    ("adhm2020", "REPORT.md", PAPERS / "adhm2020"),
    ("ajr2001", "ajr2001_gaps.md", GAPS),
    ("ajr2001", "REPORT.md", PAPERS / "ajr2001"),
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
    """Read one document, looking in the gaps directory first and then beside the paper.

    Gap lists moved to ``replication/gaps/<paper>_gaps.md`` so that every finding sits in
    one place; reports stayed next to their code.  Looking in both keeps the guard working
    whichever convention a given file follows, and ``test_every_gap_list_is_in_the_gaps_dir``
    below pins the convention itself.
    """

    for candidate in (GAPS / filename, PAPERS / paper / filename):
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    pytest.skip(f"{paper}/{filename} not present")


@pytest.mark.parametrize("paper,filename,directory", DOCS)
def test_a_document_does_not_discuss_other_papers(paper: str, filename: str, directory) -> None:
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


@pytest.mark.parametrize("paper,filename,directory", DOCS)
def test_a_document_does_not_carry_a_comparison_table(paper: str, filename: str, directory) -> None:
    """A three-column table of the series is a series artefact, not a paper finding."""

    text = _paper_text(paper, filename)
    offenders = [
        f"{paper}/{filename}:{number}: {line.strip()[:90]}"
        for number, line in enumerate(text.splitlines(), 1)
        if any(marker in line for marker in COMPARISON_TABLE)
    ]

    assert offenders == [], "comparison table found:\n" + "\n".join(offenders)


@pytest.mark.parametrize("paper,filename,directory", DOCS)
def test_a_document_does_not_link_to_another_paper(paper: str, filename: str, directory) -> None:
    text = _paper_text(paper, filename)
    offenders = []
    for number, line in enumerate(text.splitlines(), 1):
        if re.search(r"\.\./(?:adh2013|ck1994|ad2013|nq2011|adhm2020)/", line):
            offenders.append(f"{paper}/{filename}:{number}: {line.strip()[:90]}")

    assert offenders == [], "cross-document link found:\n" + "\n".join(offenders)


# ------------------------------------- the section numbers the guards cite
@pytest.mark.parametrize(
    "paper,filename,cited",
    [
        ("ad2013", "ad2013_gaps.md", [1, 2, 4]),
        ("ck1994", "ck1994_gaps.md", [2, 3]),
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


def test_every_gap_list_is_in_the_gaps_directory():
    """All gap lists live in one directory -- and none is left behind beside a paper.

    The instruction was explicit: keep every findings document together.  A file left in
    ``papers/<paper>/`` would still pass the content guards above while quietly defeating
    the point, so the location itself is asserted.
    """

    expected = {
        "adh2013_gaps.md", "ck1994_gaps.md", "ad2013_gaps.md",
        "nq2011_gaps.md", "adhm2020_gaps.md", "ajr2001_gaps.md",
    }
    present = {path.name for path in GAPS.glob("*_gaps.md")}
    assert present == expected, (
        f"gaps directory holds {sorted(present)}, expected {sorted(expected)}"
    )

    strays = sorted(
        str(path.relative_to(REPLICATION))
        for path in PAPERS.glob("*/WESTATA_GAPS.md")
    )
    assert strays == [], (
        "these gap lists were not moved into replication/gaps: " + ", ".join(strays)
    )
