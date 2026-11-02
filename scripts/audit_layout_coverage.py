"""Report which layout each command renders, and how much of the surface is verified.

The package implements far more commands than can be checked against a Stata capture in
one sitting, so this script makes the gap countable instead of a matter of memory. It
groups the commands by the header layout they are registered under, and lists the ones
whose rendering has been diffed against a committed capture.

Run it after adding a command or fixing a layout::

    ./.venv/Scripts/python.exe scripts/audit_layout_coverage.py

The "verified" list is maintained BY HAND and must only contain commands actually diffed
against a capture. A command is not verified because its layout branch exists -- the
branch is shared, and two commands on one branch can still print different headers (that
is exactly how ``xtgls`` came to be wrong while sharing ``xtpcse``'s block).
"""

from __future__ import annotations

import sys
import warnings
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

warnings.filterwarnings("ignore")

from westata.results.summary import _HEADER_BLOCK  # noqa: E402

#: svy: proportion is deliberately absent: its point estimate and standard error
#: match, but its CONFIDENCE INTERVAL does not (.4283092 vs Stata's .4287728 on the p12
#: design) and the transform Stata uses is not yet established.
#:
#: Commands whose rendered table has been diffed line by line against a Stata capture
#: and found to match (allowing for the numbers themselves differing when the sample is
#: regenerated). Each was checked with a with/without test, not by eye.
VERIFIED = {
    "regress": "OLS table, p10_outputs.txt + live regress",
    "newey": "HAC header, live newey y s, lag(3)",
    "qreg": "quantile table, p8_outputs.txt",
    "truncreg": "censored ML header, p6p7_outputs.txt",
    "zip": "zero-inflated header + equation groups, p6p7_outputs.txt",
    "xtreg, fe": "panel header + sigma/rho block, live xtreg y x1, fe",
    "svy: regress": "survey design block, live svy: regress",
    "didregress": "DID treatment/time block + ATET group, p11_did.txt",
    "xtpcse": "PCSE design block, p9_outputs.txt:495-510",
    "xtgls": "FGLS design block, p9_outputs.txt:468-488",
    "arch": "ARCH header + OPG banner (coefficient NAMES still differ)",
    "svy: mean": "62-wide estimate table, 12/12 lines character-exact vs p12",
    "svy: total": "same block, 4/4 lines exact (Total label)",
}


def main() -> int:
    by_layout = Counter(_HEADER_BLOCK.values())
    order = sorted(by_layout, key=lambda name: (-by_layout[name], name))

    print(f"{'layout':<10} {'commands':>9} {'verified':>9}")
    print("-" * 32)
    total_verified = 0
    for layout in order:
        commands = [c for c, v in _HEADER_BLOCK.items() if v == layout]
        verified = [c for c in commands if c in VERIFIED]
        total_verified += len(verified)
        print(f"{layout:<10} {len(commands):>9} {len(verified):>9}")
    print("-" * 32)
    print(f"{'TOTAL':<10} {len(_HEADER_BLOCK):>9} {total_verified:>9}")

    print("\nUNVERIFIED commands, by layout:")
    for layout in order:
        pending = sorted(
            c for c, v in _HEADER_BLOCK.items() if v == layout and c not in VERIFIED
        )
        if not pending:
            continue
        print(f"\n  [{layout}] {len(pending)} pending")
        for i in range(0, len(pending), 7):
            print("    " + ", ".join(pending[i : i + 7]))

    stale = sorted(set(VERIFIED) - set(_HEADER_BLOCK))
    if stale:
        print("\nWARNING: verified entries with no registered layout:")
        for name in stale:
            print(f"  {name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
