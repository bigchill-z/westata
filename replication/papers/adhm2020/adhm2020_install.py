"""Install the user-written Stata packages this replication needs.

Why this module exists
----------------------
ADHM cluster on two dimensions (``cluster(czone congressionaldistrict)``).  Stock Stata
supports that only for ``regress``; ``ivregress`` refuses with *multiway cluster VCE not
allowed*.  The authors used ``ivreg2``, a user-written command, which does support it.

An earlier version of this replication deliberately avoided ``ssc install`` so the script
would run offline, and compared only single-way clusters.  That was the wrong trade: it
left the paper's actual standard errors unreproduced, which is the one number the two-way
clustering debate is about.  The packages are now installed once, idempotently, and the
verification step fails loudly if they are missing.

The packages
------------
* ``ivreg2``    -- the estimator the authors used; supports multiway clustering for IV
* ``ranktest``  -- required by ``ivreg2`` for the underidentification statistics
* ``ftools``    -- required by both

They install into the user's ado directory (``~/ado/plus``), not into the project, so this
module verifies availability rather than vendoring them.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "retest" / "lib"))

#: ``(package, why)`` -- installed in this order because ``ivreg2`` depends on the others.
PACKAGES = (
    ("ftools", "matrix utilities required by ivreg2 and ranktest"),
    ("ranktest", "underidentification and weak-instrument statistics"),
    ("ivreg2", "the estimator the authors used; multiway clustering for IV"),
)


def verify(stata) -> dict[str, bool]:
    """Return which packages Stata can actually see.  No installation, no network."""

    available = {}
    for package, _ in PACKAGES:
        try:
            stata.quiet(f"which {package}")
            available[package] = True
        except Exception:  # noqa: BLE001 - "which" raises when the command is missing
            available[package] = False
    return available


def ensure(stata) -> dict[str, str]:
    """Install any missing package, then re-verify.  Returns a status per package.

    A failed install is not fatal here -- the caller decides.  ``adhm2020_run_stata`` needs
    ``ivreg2`` and will fail on its own terms if it is still missing, with a clearer message
    than a network error would give.
    """

    status = {}
    present = verify(stata)
    for package, reason in PACKAGES:
        if present.get(package):
            status[package] = "already installed"
            continue
        try:
            # ``ssc install`` reaches the network; keep its output out of the report.
            stata.quiet(f"ssc install {package}, replace")
        except Exception as exc:  # noqa: BLE001
            status[package] = f"install failed: {type(exc).__name__}"
            continue
        status[package] = "installed"

    final = verify(stata)
    for package, _ in PACKAGES:
        if not final.get(package):
            status[package] = status.get(package, "") + " | STILL MISSING"
    return status


def main() -> int:
    from stata_probe import Stata

    with Stata() as stata:
        status = ensure(stata)
    for package, note in status.items():
        print(f"{package:10s} {note}")
    return 0 if all("MISSING" not in note for note in status.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
