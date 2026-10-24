"""Capture the Stata AIPW reference values for ``tests/test_teffects.py``'s DGP.

Run:  .venv\\Scripts\\python.exe retest/fix/systems/capture_aipw_tests.py

The test module's own ``treatment_frame`` is n=1200, seed=211 with a different
propensity model from the retest sample, so its ATE/ATET numbers differ from the
ones in ``repro_S3.py``.  This script prints the numbers the tests assert.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _common import Stata, banner


def treatment_frame(n: int = 1200, seed: int = 211) -> pd.DataFrame:
    """Verbatim copy of ``tests/test_teffects.py::treatment_frame``."""

    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    propensity = 1.0 / (1.0 + np.exp(-(0.2 + 0.8 * x1 - 0.4 * x2)))
    treated = (rng.random(size=n) < propensity).astype(float)
    y = 1.0 + 2.0 * treated + 1.5 * x1 - 0.5 * x2 + rng.normal(size=n)
    return pd.DataFrame({"y": y, "treated": treated, "x1": x1, "x2": x2})


def main() -> int:
    frame = treatment_frame()
    with Stata() as st:
        st.send(frame)
        for command in ("teffects aipw (y x1 x2) (treated x1 x2)",
                        "teffects aipw (y x1 x2) (treated x1 x2), atet"):
            cap = st.estimate(command)
            banner(command)
            print(f"  e(b) names {cap.names}")
            for name, value, se in zip(cap.names, cap.b, np.sqrt(np.diag(cap.V))):
                print(f"    {name:<12} {float(value)!r:>24}   se {float(se)!r}")
            if "r1vs0.treated" in cap.names:
                i0 = cap.names.index("r1vs0.treated")
                i1 = cap.names.index("0.treated")
                total = float(cap.b[i0] + cap.b[i1])
                var = float(cap.V[i0, i0] + cap.V[i1, i1] + 2 * cap.V[i0, i1])
                print(f"    POmean(1)    {total!r:>24}   se {np.sqrt(var)!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
