"""Generate golden Stata fixtures for the conformance suite.

Runs real Stata through ``pystata`` and writes each capture to
``tests/conformance/stata_fixtures/<name>.json``.  The fixtures are committed so
that the comparison can be replayed on any machine, including ones without
Stata.

Usage::

    .venv\\Scripts\\python.exe scripts\\capture_stata_fixtures.py
    .venv\\Scripts\\python.exe scripts\\capture_stata_fixtures.py --only regress
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from westata.conformance.oracles.stata import StataOracle  # noqa: E402

STATA_DIR = r"D:\Program Files\StataNow19"
FIXTURE_DIR = ROOT / "tests" / "conformance" / "stata_fixtures"


def _binary_frame(n: int = 600, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    latent = -0.5 + 1.2 * x1 - 0.8 * x2 + rng.logistic(size=n)
    return pd.DataFrame({"y": (latent > 0).astype(float), "x1": x1, "x2": x2})


def _count_frame(n: int = 500, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    y = rng.poisson(np.exp(0.3 + 0.6 * x1)).astype(float)
    return pd.DataFrame({"y": y, "x1": x1})


#: name -> (pandas data or None, list of Stata commands).
SCENARIOS: dict[str, tuple[pd.DataFrame | None, list[str]]] = {
    "regress_auto": (None, ["sysuse auto, clear", "regress mpg weight foreign"]),
    "regress_auto_robust": (
        None,
        ["sysuse auto, clear", "regress mpg weight foreign, vce(robust)"],
    ),
    "regress_auto_factor": (
        None,
        ["sysuse auto, clear", "regress mpg weight i.foreign"],
    ),
    "logit_sim": (_binary_frame(), ["logit y x1 x2"]),
    "probit_sim": (_binary_frame(), ["probit y x1 x2"]),
    "poisson_sim": (_count_frame(), ["poisson y x1"]),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", choices=sorted(SCENARIOS))
    args = parser.parse_args(argv)

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    selected = args.only or sorted(SCENARIOS)

    with StataOracle(stata_dir=STATA_DIR, edition="mp") as oracle:
        for name in selected:
            data, commands = SCENARIOS[name]
            if data is not None:
                oracle.send_data(data)
            for command in commands:
                oracle.run(command)
            capture = oracle.capture()
            path = capture.save(FIXTURE_DIR / f"{name}.json")
            print(f"{name}: {path.name}  b={capture.b.size}  names={capture.names}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
