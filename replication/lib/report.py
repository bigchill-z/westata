"""Render comparison records as markdown, in the two shapes a replication needs.

Why two tables and not one
--------------------------
A replication answers two different questions, and mixing them makes both hard
to read:

* **Does the paper's published number reappear?**  That is the replication
  table: for each specification, the headline estimate and its standard error as
  the paper prints them, as StataNow reproduces them, and as westata reproduces
  them.  A reader can see at a glance which of the three legs is off without
  reading a single relative difference.
* **How far apart are the implementations?**  That is the difference table: the
  maximum relative difference over coefficients and over standard errors, plus
  the verdict.  It is the table a regression-test log should carry, and it is
  deliberately numeric rather than pass/fail so a slow drift is visible before
  it crosses a tolerance.

The paper's numbers come from the published ``.xls`` of tables, not from
re-running the authors' do-file, so they are supplied as :class:`Published`
entries.  Nothing here pretends the paper's column was recomputed.

Every table is written to a path as well as returned as a string: a replication
log that only exists in scrollback cannot be reviewed later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .compare import Record

__all__ = [
    "Published",
    "difference_table",
    "fmt_number",
    "fmt_rel",
    "replication_table",
    "write_markdown",
]

#: Relative-difference thresholds, quoted in the report so a reader knows what
#: the stored verdicts meant without opening the comparison module.
THRESHOLDS: tuple[tuple[str, float], ...] = (
    ("IDENTICAL", 1e-9),
    ("NUMERICALLY_EQUAL", 1e-6),
    ("MINOR_DIFF", 1e-3),
)


def fmt_number(value: float | None, digits: int = 4) -> str:
    """Format a coefficient the way a journal table would, or ``n/a``.

    Published estimates span ``1e-4`` to ``1e2``, so fixed decimals read better
    than scientific notation; non-finite and missing values become ``n/a`` so an
    absent number can never be mistaken for zero.
    """
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    if number == 0:
        return "0"
    if abs(number) < 1e-3 or abs(number) >= 1e5:
        return f"{number:.{max(digits - 1, 1)}e}"
    return f"{number:.{digits}f}"


def fmt_rel(value: float, digits: int = 3) -> str:
    """Format a relative difference, keeping ``NaN``/``inf`` visible as words."""
    if value is None or value != value:
        return "n/a"
    if math.isinf(value):
        return "inf"
    return f"{value:.{digits}e}"


@dataclass
class Published:
    """The paper's own reported numbers for one specification.

    ``estimate``/``se`` are the coefficient and standard error the paper prints
    for its exposure variable; ``n`` and ``r2`` are its reported sample size and
    R-squared.  Any may be ``None`` when the table does not print it.

    ``row`` names the coefficient those numbers refer to, in the *canonical*
    vocabulary the comparison layer uses (``d_traderes_pw_lag``, ``x=2`` for a
    factor level).  It defaults to the specification's first coefficient, which
    is where every table in this replication prints its exposure variable; when
    the paper prints a different row, name it rather than relying on order.
    """

    name: str
    estimate: float | None = None
    se: float | None = None
    n: int | None = None
    r2: float | None = None
    row: str = ""
    note: str = ""


def _published_row(record: Record, published: Published) -> str:
    """Resolve which coefficient row the published numbers describe."""
    if published.row:
        try:
            record.row(published.row)
        except KeyError as exc:
            raise ValueError(
                f"published row {published.row!r} is not a coefficient of record "
                f"{record.name!r}; it has: {', '.join(row.name for row in record.rows)}"
            ) from exc
        return published.row
    if not record.rows:
        raise ValueError(f"record {record.name!r} has no coefficients to publish")
    return record.rows[0].name


def replication_table(
    records: Sequence[Record],
    published: Sequence[Published] = (),
    *,
    title: str = "Replication table",
) -> str:
    """Markdown table: estimate, SE, N and R2 by paper / Stata / westata.

    The paper column reads ``n/a`` for a specification with no
    :class:`Published` entry.  That is the honest rendering: such a row is a
    harness check, not a replication of a number the paper prints.
    """
    published_by_name = {item.name: item for item in published}
    lines = [
        f"### {title}",
        "",
        "| spec | parameter | paper est. | Stata est. | westata est. | paper SE "
        "| Stata SE | westata SE | paper N | Stata N | westata N "
        "| paper R2 | Stata R2 | westata R2 |",
        "|" + "---|" * 14,
    ]
    for record in records:
        pub = published_by_name.get(record.name) or Published(name=record.name)
        name = _published_row(record, pub)
        row = record.row(name)
        stata_r2 = record.scalar_side("r2", "stata")
        westata_r2 = record.scalar_side("r2", "westata")
        cells = [
            record.name,
            f"`{name}`",
            fmt_number(pub.estimate),
            fmt_number(row.stata_b),
            fmt_number(row.westata_b),
            fmt_number(pub.se),
            fmt_number(row.stata_se),
            fmt_number(row.westata_se),
            str(pub.n) if pub.n is not None else "n/a",
            str(record.stata_n),
            str(record.westata_n),
            fmt_number(pub.r2),
            fmt_number(stata_r2),
            fmt_number(westata_r2),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    unquoted = [record.name for record in records if record.name not in published_by_name]
    if published_by_name and unquoted:
        lines.append("_No published numbers available for: " + ", ".join(unquoted) + "._")
        lines.append("")
    return "\n".join(lines)


def difference_table(
    records: Sequence[Record],
    *,
    title: str = "Difference table",
) -> str:
    """Markdown table: max relative difference in coefficients and in SEs.

    The threshold legend is printed with the table rather than assumed, so a
    reader does not have to open the comparison module to know what
    ``IDENTICAL`` meant for this run.
    """
    legend = ", ".join(f"{label} <= {value:g}" for label, value in THRESHOLDS)
    lines = [
        f"### {title}",
        "",
        f"_Verdict thresholds: {legend}, MATERIAL_DIFF above; NOT_COMPARABLE when a "
        "side is missing. Standard errors are compared with the same thresholds._",
        "",
        "| spec | params | max coef rel | max SE rel | N (Stata/westata) | verdict | note |",
        "|" + "---|" * 7,
    ]
    for record in records:
        n_cell = f"{record.stata_n}/{record.westata_n}"
        if not record.n_agree:
            n_cell = f"**{n_cell}**"
        verdict = record.verdict if record.verdict == "IDENTICAL" else f"**{record.verdict}**"
        note = record.notes[0].replace("|", "/") if record.notes else ""
        cells = [
            record.name,
            str(record.n_params),
            fmt_rel(record.max_b_rel),
            fmt_rel(record.max_se_rel),
            n_cell,
            verdict,
            note,
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    counts: dict[str, int] = {}
    for record in records:
        counts[record.verdict] = counts.get(record.verdict, 0) + 1
    summary = ", ".join(f"{key}: {value}" for key, value in sorted(counts.items()))
    lines.append(f"_{len(records)} specification(s); {summary or 'none'}._")
    lines.append("")
    return "\n".join(lines)


def write_markdown(
    path: str | Path,
    records: Sequence[Record],
    published: Sequence[Published] = (),
    *,
    title: str = "Replication report",
) -> Path:
    """Write both tables to ``path`` and return the resolved path.

    The parent directory is created because a replication run is often the first
    thing to write into ``replication/out``; failing there would lose the run's
    only artifact.
    """
    if not records:
        raise ValueError("refusing to write a report with no records")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    body = [
        f"# {title}",
        "",
        replication_table(records, published, title="Estimates"),
        difference_table(records, title="Implementation differences"),
    ]
    target.write_text("\n".join(body), encoding="utf-8")
    return target.resolve()
