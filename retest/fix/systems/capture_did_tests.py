"""Capture the Stata values the DID tests need after S5 (covariates=None -> none).

Run:  .venv\\Scripts\\python.exe retest/fix/systems/capture_did_tests.py

Produces, on the exact fixture the tests build (``tests/test_teffects_did.py::_did``
and ``tests/test_did.py::did_frame``), the numbers the updated tests assert.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _common import Stata, banner


def did_p11(n_units: int = 200, n_periods: int = 8, seed: int = 223) -> pd.DataFrame:
    """Verbatim copy of ``tests/test_teffects_did.py::_did``."""

    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(n_units):
        effect = rng.normal(scale=0.6)
        arm = unit % 2 == 0
        for period in range(1, n_periods + 1):
            treated_now = 1.0 if (arm and period >= 4) else 0.0
            x1 = rng.normal()
            y = (1.0 + 0.5 * x1 + effect + 0.3 * period + 1.2 * treated_now
                 + rng.normal(scale=0.5))
            rows.append((unit, period, y, x1, treated_now, float(arm)))
    return pd.DataFrame(rows, columns=["unit", "period", "y", "x1", "treated", "group"])


def did_frame(n_units: int = 60, n_periods: int = 4) -> pd.DataFrame:
    """The shape ``tests/test_did.py`` builds (a slow/treated two-period-style DGP).

    The test module's own builder is imported from it instead, so this stays in
    sync; see ``main``.
    """

    raise NotImplementedError


def main() -> int:
    import sys

    sys.path.insert(0, "tests")
    import test_did as td

    panel_tests = td.did_frame()
    print("test_did panel columns", list(panel_tests.columns), len(panel_tests))
    panel_p11 = did_p11()

    with Stata() as st:
        banner("P11 fixture: didregress (y) (treated)  -- no covariate")
        st.send(panel_p11)
        for command in (
            "didregress (y) (treated), group(unit) time(period)",
            "didregress (y x1) (treated), group(unit) time(period)",
        ):
            cap = st.estimate(command)
            i = cap.names.index("r1vs0.treated")
            print(f"  {command}")
            print(f"    ATET {float(cap.b[i])!r}   se {float(np.sqrt(cap.V[i, i]))!r}"
                  f"   N {cap.scalars.get('N')!r}   N_clust {cap.scalars.get('N_clust')!r}"
                  f"   df_r {cap.scalars.get('df_r')!r}")
        st.run("xtset unit period")
        command = "xtdidregress (y) (treated), group(unit) time(period)"
        cap = st.estimate(command)
        i = cap.names.index("r1vs0.treated")
        print(f"  {command}")
        print(f"    ATET {float(cap.b[i])!r}   se {float(np.sqrt(cap.V[i, i]))!r}"
              f"   N {cap.scalars.get('N')!r}   N_clust {cap.scalars.get('N_clust')!r}"
              f"   df_r {cap.scalars.get('df_r')!r}")

        banner("test_did fixture: didregress (y) (treated)  -- no covariate")
        st.send(panel_tests)
        for command in (
            "didregress (y) (treated), group(unit) time(period)",
            "didregress (y x1) (treated), group(unit) time(period)",
        ):
            cap = st.estimate(command)
            i = cap.names.index("r1vs0.treated")
            print(f"  {command}")
            print(f"    ATET {float(cap.b[i])!r}   se {float(np.sqrt(cap.V[i, i]))!r}"
                  f"   N {cap.scalars.get('N')!r}   df_r {cap.scalars.get('df_r')!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
