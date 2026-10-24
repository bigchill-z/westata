"""V5: ``arima``'s ``e(N)`` on an integrated model must count the estimation sample.

Stata: ``e(N) = tmax - tmin + 1 - d - D*s``; with ``d=1`` on a 300-observation
``tsset`` sample that is 299, while westata reported the length of the series
handed to statsmodels (300).  The gap is exactly ``d + D*s``.

The value is not cosmetic: Stata's own ``estat ic`` uses that N for the BIC
penalty, so any downstream per-observation quantity changes too.  ``ll``,
``aic`` and ``bic`` themselves stay correct because they are evaluated on the
same differenced innovations.

Run::

    cd D:\\myCodeProject\\statapy
    .venv\\Scripts\\python.exe retest/fix/tseries/repro_v5_arima_n.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import _repro_common as C
from stata_probe import Stata

import westata as ws

CASES = [
    ("arima y_ar, arima(1,0,0)", "y_ar", (1, 0, 0), (0, 0, 0, 0), "c", C.Y_AR, 0),
    ("arima y_rw, arima(0,1,1)", "y_rw", (0, 1, 1), (0, 0, 0, 0), "t", C.Y_RW, 1),
    ("arima y_111, arima(1,1,1)", "y_111", (1, 1, 1), (0, 0, 0, 0), "t", C.Y_111, 1),
    ("arima y_rw, arima(1,2,1) noconstant", "y_rw", (1, 2, 1), (0, 0, 0, 0), "n", C.Y_RW, 2),
    ("arima y_rw, arima(0,2,1) noconstant", "y_rw", (0, 2, 1), (0, 0, 0, 0), "n", C.Y_RW, 2),
    ("arima y_s, arima(1,0,0) sarima(0,1,1,4)", "y_s", (1, 0, 0), (0, 1, 1, 4), "t", C.Y_S, 4),
]


def main() -> int:
    print("=" * 112)
    print("V5  arima e(N) on integrated models  (StataNow 19.5)")
    print("=" * 112)

    with Stata() as st:
        st.send(C.pd.DataFrame({
            "t": np.arange(1, C.N + 1),
            "y_ar": C.Y_AR, "y_rw": C.Y_RW, "y_111": C.Y_111, "y_s": C.Y_S,
        }))
        st.run("tsset t")

        print(f"\n{'command':<44s} {'T':>4s} {'d+D*s':>6s} {'Stata e(N)':>11s} "
              f"{'westata N':>10s} {'before':>8s}")
        for command, _, order, seasonal, trend, series, differencing in CASES:
            st.estimate(command)
            stata_n = int(st.scalar("e(N)"))
            stata_ll = float(st.scalar("e(ll)"))
            st.quiet("estat ic")
            ic = np.asarray(st.matrix("r(S)"), dtype=float).reshape(-1)
            names = [n.lower() for n in st.matrix_names("r(S)")]
            stata_aic = float(ic[names.index("aic")])
            stata_bic = float(ic[names.index("bic")])
            k = int(np.asarray(st.matrix("e(b)")).size)

            result = ws.ARIMA(series, order=order, seasonal_order=seasonal, trend=trend).fit()
            west_n = int(result.e["N"])
            before = len(series)
            print(f"{command:<44s} {len(series):4d} {differencing:6d} {stata_n:11d} "
                  f"{west_n:10d} {before:8d}")
            assert west_n == stata_n, (command, west_n, stata_n)
            assert int(result.nobs) == len(series), "StataResults.nobs stays the residual length"

            # the BIC penalty uses Stata's N -- confirm both sides agree
            bic_from_parts = -2.0 * stata_ll + k * np.log(stata_n)
            print(f"{'':44s} ll rel {abs(result.e['ll'] - stata_ll) / abs(stata_ll):.2e} | "
                  f"BIC(Stata parts based on e(N)) {bic_from_parts:.6f} vs "
                  f"r(S) {stata_bic:.6f} -> rel "
                  f"{abs(bic_from_parts - stata_bic) / abs(stata_bic):.2e} | "
                  f"westata bic rel {abs(result.e['bic'] - stata_bic) / abs(stata_bic):.2e}")
            print(f"{'':44s} AIC rel {abs(result.e['aic'] - stata_aic) / abs(stata_aic):.2e}")

    print("\nAll six cases: westata e(N) == Stata e(N); 'before' was T (the series "
          "length), i.e. off by d + D*s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
