"""Compare westata's rendered table against Stata's, character for character.

The previous round fixed the OLS-style table and left every other family unchecked,
because the check was manual: render one model, print both, look at them. That does not
scale to 60-odd commands and it is how a 78-vs-80 width survived review in the first
place.

This script does the comparison mechanically. For each family it

  1. builds the SAME data the capture script used (imported, not retyped),
  2. extracts Stata's block from the committed capture,
  3. renders the westata equivalent,
  4. diffs them and reports the width of every line on both sides.

Only the header and column geometry are compared -- the numbers come out different
because the capture was taken against a Stata run whose convergence path is not
reproduced bit for bit. What must match is where things SIT: total width, the position of
the ``|`` and ``+``, and the column headers.

Run: ``.venv/Scripts/python.exe scripts/check_layout.py``
"""

from __future__ import annotations

import difflib
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

FIXTURES = REPO / "tests" / "conformance" / "stata_fixtures"


def capture_block(name: str, start: str, stop: str | None = None,
                  fixture: str = "p6p7_outputs.txt") -> list[str]:
    """Stata's captured output between two markers."""

    text = (FIXTURES / fixture).read_text(encoding="utf-8")
    begin = text.find(f"=== {start} ===")
    if begin < 0:
        raise SystemExit(f"no {start!r} section in the capture")
    begin = text.find("\n", begin) + 1
    end = text.find(f"=== {stop} ===") if stop else len(text)
    if end < 0:
        end = len(text)
    lines = [line.rstrip() for line in text[begin:end].splitlines()]
    # Drop the capture file's own section markers; they are scaffolding, not output.
    return [line for line in lines if not line.startswith("=== ")]


def shape(lines: list[str]) -> dict[str, object]:
    """The geometry facts a layout check cares about."""

    widths = sorted({len(line) for line in lines if line.strip()})
    rules = [line for line in lines if set(line) <= {"-", "+"} and len(line) > 10]
    piped = [line for line in lines if "|" in line]
    return {
        "widths": widths,
        "rule_widths": sorted({len(line) for line in rules}),
        "pipe_at": sorted({line.index("|") for line in piped}),
        "headers": [
            line for line in piped if "Coefficient" in line or "Std. err." in line
        ],
    }


def report(name: str, stata: list[str], ours: list[str]) -> bool:
    """Print the differences; return True when the geometry agrees."""

    a, b = shape(stata), shape(ours)
    ok = True
    print(f"\n=== {name} ===")
    for key in ("widths", "rule_widths", "pipe_at"):
        if a[key] != b[key]:
            ok = False
            print(f"  {key:12} Stata={a[key]}  westata={b[key]}")
    if a["headers"] != b["headers"]:
        ok = False
        for line in a["headers"]:
            print(f"  STATA  |{line}|")
        for line in b["headers"]:
            print(f"  OURS   |{line}|")
    if ok:
        print("  geometry matches")
    else:
        diff = list(
            difflib.unified_diff(a["headers"], b["headers"], lineterm="", n=0)
        )
        print(f"  ({len(diff)} header diff lines)")
    return ok


def main() -> int:
    import pandas as pd

    import westata as ws
    from capture_p6p7_fixtures import (
        count_frame,
        interval_frame,
        overdispersed_frame,
        truncated_frame,
    )

    checks: list[tuple[str, list[str], object]] = []

    truncated = truncated_frame()
    checks.append(
        (
            "truncreg",
            capture_block("truncreg", "truncreg", "intreg"),
            ws.Truncreg.from_formula("y ~ x1 + x2", truncated, lower=0.0).fit(),
        )
    )

    # ---- quantile family (p8) ----
    from capture_p8_fixtures import quantile_frame

    qframe = quantile_frame()
    checks.append(
        (
            "qreg",
            capture_block(
                "qreg_0.5", "qreg_0.5", "qreg_0.75", fixture="p8_outputs.txt"
            ),
            ws.QuantileRegression.from_formula(
                "y ~ x1 + x2", qframe, quantile=0.5
            ).fit(),
        )
    )

    # ---- binary family (p6p7 has zip/zinb; logit lives in a later section) ----
    from capture_p6p7_fixtures import count_frame

    counts = count_frame()
    checks.append(
        (
            "zip",
            capture_block("zip", "zip", "zinb"),
            # The capture ran `zip y x1, inflate(x1)`. Passing no inflate= would fit the
            # plain Poisson and compare two different models -- which is exactly the
            # mistake this harness made on its first run.
            ws.ZeroInflatedPoisson.from_formula("y ~ x1", counts, inflate="x1").fit(),
        )
    )

    failed = []
    for name, stata_lines, result in checks:
        ours = str(result.summary(style="stata")).splitlines()
        # Stata's capture has the iteration log above the table; keep only from the first
        # line that looks like the table itself.
        first = next(
            (i for i, line in enumerate(stata_lines) if "Coefficient" in line), None
        )
        if first is None:
            print(f"\n=== {name} ===\n  no table in the capture")
            continue
        start = first
        while start > 0 and stata_lines[start - 1].strip():
            start -= 1
        stata_table = stata_lines[start:]
        ours_first = next(
            (i for i, line in enumerate(ours) if "Coefficient" in line), None
        )
        if ours_first is None:
            print(f"\n=== {name} ===\n  westata rendered no coefficient table")
            failed.append(name)
            continue
        ours_start = ours_first
        while ours_start > 0 and ours[ours_start - 1].strip():
            ours_start -= 1
        if not report(name, stata_table, ours[ours_start:]):
            failed.append(name)

    print()
    if failed:
        print(f"layout differs: {', '.join(failed)}")
        return 1
    print("all checked families match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
