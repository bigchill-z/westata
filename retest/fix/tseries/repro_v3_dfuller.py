"""V3: ``dfuller``'s printed critical values come from a hard-coded table.

Stata does NOT use MacKinnon's response surface for the 1%/5%/10% critical
values it prints: ``ado/base/d/dfuller.ado``'s ``GetCrit`` hard-codes a six-point
table (N = 25/50/100/250/500/infinity) per deterministic case and interpolates
LINEARLY in N.  ``pperron.ado``'s ``GetCrit`` uses the same Z(t) rows.

The statistic and the MacKinnon p-value were already A-grade (1e-13) and are
untouched; only the printed critical values change.

Run::

    cd D:\\myCodeProject\\statapy
    .venv\\Scripts\\python.exe retest/fix/tseries/repro_v3_dfuller.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import _repro_common as C
from stata_probe import Stata

import westata as ws
from westata.estimators.tseries import _DFULLER_TABLE, _dfuller_critical


#: The table as it appears in dfuller.ado, kept here to prove the source module
#: really copies it (an edited table fails this comparison).
ADO_TABLE = {
    "n": (
        (-2.66, -2.62, -2.60, -2.58, -2.58, -2.58),
        (-1.95, -1.95, -1.95, -1.95, -1.95, -1.95),
        (-1.60, -1.61, -1.61, -1.62, -1.62, -1.62),
    ),
    "c": (
        (-3.75, -3.58, -3.51, -3.46, -3.44, -3.43),
        (-3.00, -2.93, -2.89, -2.88, -2.87, -2.86),
        (-2.63, -2.60, -2.58, -2.57, -2.57, -2.57),
    ),
    "ct": (
        (-4.38, -4.15, -4.04, -3.99, -3.98, -3.96),
        (-3.60, -3.50, -3.45, -3.43, -3.42, -3.41),
        (-3.24, -3.18, -3.15, -3.13, -3.13, -3.12),
    ),
}

CASES = [
    ("dfuller y_rw, lags(2)", C.Y_RW, dict(regression="c", maxlag=2, autolag=None), "c"),
    ("dfuller y_rw, lags(2) trend", C.Y_RW, dict(regression="ct", maxlag=2, autolag=None), "ct"),
    ("dfuller y_rw, lags(2) noconstant", C.Y_RW, dict(regression="n", maxlag=2, autolag=None), "n"),
    ("dfuller y_ar, lags(4)", C.Y_AR, dict(regression="c", maxlag=4, autolag=None), "c"),
    ("dfuller y_rw, lags(0)", C.Y_RW, dict(regression="c", maxlag=0, autolag=None), "c"),
    ("dfuller y_ar, lags(4) trend", C.Y_AR, dict(regression="ct", maxlag=4, autolag=None), "ct"),
]


def macKinnon(series, **kwargs):
    """statsmodels' response-surface critical values (the old behaviour)."""

    from statsmodels.tsa.stattools import adfuller

    outcome = adfuller(series, autolag=kwargs.get("autolag"), **{
        k: v for k, v in kwargs.items() if k != "autolag"
    }, result_object=True)
    return np.array([
        outcome.critical_values["1%"],
        outcome.critical_values["5%"],
        outcome.critical_values["10%"],
    ])


def main() -> int:
    print("=" * 116)
    print("V3  dfuller printed 1%/5%/10% critical values  (StataNow 19.5)")
    print("=" * 116)

    # 1) the module's table must be byte-identical to dfuller.ado's
    for case, rows in ADO_TABLE.items():
        for expected, actual in zip(rows, _DFULLER_TABLE[case]):
            assert tuple(expected) == tuple(actual), (case, expected, actual)
    print("dfuller.ado table reproduced verbatim in tseries.py: OK "
          f"({sum(len(r) for rows in ADO_TABLE.values() for r in rows)} values)")

    with Stata() as st:
        st.send(C.pd.DataFrame({
            "t": np.arange(1, C.N + 1),
            "y_rw": C.Y_RW,
            "y_ar": C.Y_AR,
        }))
        st.run("tsset t")

        print(f"\n{'command':<36s} {'N':>5s} | {'quantity':<24s} {'before':>11s} "
              f"{'after':>11s} | Stata")
        worst_before = worst_after = 0.0
        for command, series, kwargs, case in CASES:
            st.quiet(command)
            nobs = float(st.scalar("r(N)"))
            stata_cv = np.array([
                float(st.scalar("r(cv_1)")),
                float(st.scalar("r(cv_5)")),
                float(st.scalar("r(cv_10)")),
            ])
            result = ws.unitroot(series, test="adf", **kwargs)
            before = macKinnon(series, **kwargs)
            after = np.array([
                result.critical["1%"], result.critical["5%"], result.critical["10%"]
            ])
            rel_before = C.rel(before, stata_cv)
            rel_after = C.rel(after, stata_cv)
            worst_before = max(worst_before, rel_before)
            worst_after = max(worst_after, rel_after)
            print(f"{command:<36s} {nobs:5.0f} | {'critical values 1/5/10%':<24s} "
                  f"{rel_before:11.3e} {rel_after:11.3e} | "
                  f"{np.array2string(stata_cv, precision=5)}")
            print(f"{'':<36s} {'':5s} | {'statistic (must not move)':<24s} "
                  f"{'':>11s} {abs(result.statistic - float(st.scalar('r(Zt)'))):11.3e}")
            # the attribution: reproduce Stata's numbers from the intercepted table
            reproduced = _dfuller_critical(case, nobs)
            reproduced_arr = np.array(
                [reproduced["1%"], reproduced["5%"], reproduced["10%"]]
            )
            assert C.rel(reproduced_arr, stata_cv) <= 1e-12, (command, reproduced_arr, stata_cv)

    print(f"\nworst relative difference -- before {worst_before:.3e}, "
          f"after {worst_after:.3e}  (target 1e-9; measured floor ~1.3e-16)")

    # 2) sanity: pperron uses the same Z(t) table (separate call path)
    with Stata() as st:
        st.send(C.pd.DataFrame({
            "t": np.arange(1, C.N + 1),
            "y_rw": C.Y_RW,
        }))
        st.run("tsset t")
        st.quiet("pperron y_rw")
        stata_cv = np.asarray(st.matrix("r(cvalues)"), dtype=float)[1, 1:]
        result = ws.unitroot(C.Y_RW, test="pp")
        after = np.array([result.critical["1%"], result.critical["5%"], result.critical["10%"]])
        print(f"\npperron y_rw: Z(t) table reused -> after {C.rel(after, stata_cv):.3e} "
              f"(Stata {stata_cv.tolist()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
