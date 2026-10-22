"""Install estout (esttab/estout/eststo) from the author's GitHub source.

Same reason as reghdfe: SSC is unreachable, raw.githubusercontent.com is not.
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from westata.conformance.oracles.stata import StataOracle  # noqa: E402

BASE = "https://raw.githubusercontent.com/benjann/estout/master/"
FILES = [
    "_eststo.ado", "_eststo.hlp",
    "estadd.ado", "estadd.hlp",
    "estout.ado", "estout.hlp",
    "estpost.ado", "estpost.hlp",
    "eststo.ado", "eststo.hlp",
    "esttab.ado", "esttab.hlp",
    "estout_mystyle.def",
]


def plus_dir() -> Path:
    with StataOracle() as oracle:
        output = oracle.get_output("sysdir")
    for line in output.splitlines():
        if line.strip().startswith("PLUS:"):
            return Path(line.split(":", 1)[1].strip().rstrip("/\\"))
    raise SystemExit("could not parse PLUS from sysdir")


def main() -> int:
    target = plus_dir()
    print(f"installing into {target}")
    target.mkdir(parents=True, exist_ok=True)

    failures = 0
    for name in FILES:
        try:
            with urllib.request.urlopen(BASE + name, timeout=60) as response:
                payload = response.read()
        except (urllib.error.URLError, OSError) as exc:
            print(f"  FAILED {name}: {exc}")
            failures += 1
            continue
        (target / name).write_bytes(payload)
        print(f"  {name:26s} {len(payload):>7d} bytes")

    print(f"\n{failures} failure(s)")
    with StataOracle() as oracle:
        for command in ["discard", "which esttab", "which estout", "which eststo"]:
            try:
                output = oracle.get_output(command)
                if output.strip():
                    print(f"$ {command}\n  {output.strip().splitlines()[0][:140]}")
            except Exception as exc:  # noqa: BLE001
                print(f"$ {command}\n  FAILED: {str(exc)[:160]}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
