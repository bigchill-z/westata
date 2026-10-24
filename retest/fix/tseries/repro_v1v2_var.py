"""V1 + V2: VAR's per-equation ML degrees of freedom and vargranger's chi2.

Reproduces, against a live StataNow 19.5 session:

* V1 -- ``var``'s ``e(V)``/``e(Sigma)`` divide each equation's RSS by ``T`` (ML),
  while statsmodels divides by ``T - k_eq`` (unbiased).  The ratio is
  ``sqrt((T - k_eq)/T)`` = 0.9949706303122452 for a bivariate VAR(1) on 299
  usable observations with ``k_eq = 3``.
* V2 -- ``granger_causality`` used to return statsmodels' F statistic; Stata's
  ``vargranger`` prints a Wald chi2, and that chi2 additionally carries
  ``T/(T - k_eq)`` because the Wald form uses the ML residual covariance.

Run::

    cd D:\\myCodeProject\\statapy
    .venv\\Scripts\\python.exe retest/fix/tseries/repro_v1v2_var.py

Prints a before/after/Stata table per lag order.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import _repro_common as C
from stata_probe import Stata

import westata as ws
from statsmodels.tsa.vector_ar.var_model import VAR as _SMVAR


def stata_param_order(names: list[str], lags: int, neqs: int) -> list[int]:
    """Map Stata's equation-by-equation ``e(b)`` onto statsmodels' ``vec(B)``."""

    per_equation = lags * neqs + 1
    order: list[int] = []
    for equation in range(neqs):
        for position in range(per_equation):
            name = names[equation * per_equation + position]
            if name == "_cons":
                order.append(equation)
            else:
                match = re.fullmatch(r"L(\d*)\.y(\d+)", name)
                assert match, name
                lag = int(match.group(1) or 1)
                variable = int(match.group(2)) - 1
                order.append((1 + (lag - 1) * neqs + variable) * neqs + equation)
    return order


def main() -> int:
    print("=" * 108)
    print("V1/V2  VAR(1) and VAR(2): SE, e(Sigma), vargranger  (StataNow 19.5, "
          "SEED=20261024, N=300)")
    print("=" * 108)

    with Stata() as st:
        st.send(C.pd.DataFrame({
            "t": np.arange(1, C.N + 1),
            "y1": C.Y_VAR[:, 0],
            "y2": C.Y_VAR[:, 1],
        }))
        st.run("tsset t")
        assert st.nobs() == C.N

        for lags, spec in ((1, "lags(1)"), (2, "lags(1/2)")):
            command = f"var y1 y2, {spec}"
            st.estimate(command)
            names = st.matrix_names("e(b)")
            stata_b = np.asarray(st.matrix("e(b)"), dtype=float).reshape(-1)
            stata_v = np.asarray(st.matrix("e(V)"), dtype=float)
            stata_se = np.sqrt(np.diag(stata_v))
            stata_sigma = np.asarray(st.matrix("e(Sigma)"), dtype=float)
            T = float(st.scalar("e(N)"))
            k_eq = lags * 2 + 1
            ml = (T - k_eq) / T

            st.quiet("vargranger")
            g = np.asarray(st.matrix("r(gstats)"), dtype=float)
            stata_chi2 = np.array([g[0, 0], g[2, 0]])
            stata_df = np.array([g[0, 1], g[2, 1]])
            stata_p = np.array([g[0, 2], g[2, 2]])

            print(f"\n--- {command}   (Stata e(N) = {T:.0f}, k_eq = {k_eq}, "
                  f"(T-k_eq)/T = {ml:.17f}, sqrt = {np.sqrt(ml):.16f})")

            # ---- statsmodels raw quantities (the pre-fix behaviour) --------
            raw = _SMVAR(C.Y_VAR).fit(lags, trend="c", ic=None)
            flat = np.asarray(raw.params, dtype=float).ravel()
            raw_se = np.sqrt(np.diag(np.asarray(raw.cov_params(), dtype=float)))
            order = stata_param_order(names, lags, 2)

            # ---- westata, after the fix -----------------------------------
            fixed = ws.VAR(C.Y_VAR, lags=lags, trend="c").fit()

            rows = []
            rows.append((
                "coefficients e(b)",
                C.rel(flat[order], stata_b),
                C.rel(fixed.stata_params().to_numpy(), stata_b),
            ))
            rows.append((
                "standard errors (raw statsmodels)",
                C.rel(raw_se[order], stata_se),
                C.rel(fixed.bse.to_numpy(), stata_se),
            ))
            rows.append((
                "e(Sigma) as reported",
                C.rel(raw.sigma_u.ravel(), stata_sigma.ravel()),
                C.rel(fixed.sigma_u.ravel(), stata_sigma.ravel()),
            ))
            rows.append((
                "e(V) full matrix",
                C.rel(np.asarray(raw.cov_params(), dtype=float)[np.ix_(order, order)].ravel(),
                      stata_v.ravel()),
                C.rel(fixed.stata_cov_params.ravel(), stata_v.ravel()),
            ))

            f_test = raw.test_causality("y1", ["y2"], kind="f")
            raw_wald = np.array([
                float(raw.test_causality("y1", ["y2"], kind="wald").test_statistic),
                float(raw.test_causality("y2", ["y1"], kind="wald").test_statistic),
            ])
            fixed_granger = [
                fixed.granger_causality("y1", ["y2"]),
                fixed.granger_causality("y2", ["y1"]),
            ]
            fixed_chi2 = np.array([float(d.test_statistic) for d in fixed_granger])
            fixed_p = np.array([float(d.pvalue) for d in fixed_granger])
            fixed_df = np.array([float(d.df) for d in fixed_granger])

            rows.append((
                "vargranger chi2",
                C.rel(raw_wald, stata_chi2),
                C.rel(fixed_chi2, stata_chi2),
            ))
            rows.append((
                "vargranger p-value",
                C.rel(np.array([float(raw.test_causality("y1", ["y2"], kind="wald").pvalue),
                                float(raw.test_causality("y2", ["y1"], kind="wald").pvalue)]),
                      stata_p),
                C.rel(fixed_p, stata_p),
            ))

            print(f"  {'quantity':<34s} {'before':>12s} {'after':>12s}   grade")
            for label, before, after in rows:
                grade = "A" if after <= 1e-9 else ("B" if after <= 1e-6 else
                                                   "C" if after <= 1e-3 else "D/E")
                print(f"  {label:<34s} {before:12.3e} {after:12.3e}   {after and grade}")
            print(f"  {'df':<34s} {str(tuple(f_test.df)):>12s} "
                  f"{str(tuple(int(x) for x in fixed_df)):>12s}   stata "
                  f"{tuple(int(x) for x in stata_df)}")
            print(f"  default result class: {type(fixed_granger[0]).__name__}, "
                  f"method={fixed_granger[0].method!r}")
            print(f"  Stata chi2  = {stata_chi2.tolist()}")
            print(f"  westata chi2= {fixed_chi2.tolist()}")
            print(f"  Stata p     = {stata_p.tolist()}")
            print(f"  westata p   = {fixed_p.tolist()}")

            # ml=False must reproduce Stata's `var ..., dfk` (divide by T-k_eq).
            st.estimate(f"{command} dfk")
            stata_dfk_b = np.asarray(st.matrix("e(b)"), dtype=float).reshape(-1)
            stata_dfk_se = np.sqrt(np.diag(np.asarray(st.matrix("e(V)"), dtype=float)))
            stata_dfk_sigma = np.asarray(st.matrix("e(Sigma)"), dtype=float)
            dfk = ws.VAR(C.Y_VAR, lags=lags, trend="c").fit(ml=False)
            print(f"  dfk check (ml=False): se rel "
                  f"{C.rel(dfk.bse.to_numpy(), stata_dfk_se):.3e}, "
                  f"e(Sigma) rel {C.rel(dfk.sigma_u.ravel(), stata_dfk_sigma.ravel()):.3e}, "
                  f"dfk b rel {C.rel(dfk.stata_params().to_numpy(), stata_dfk_b):.3e}")

    print("\nNote: 'before' for the chi2 row is statsmodels' Wald statistic WITHOUT "
          "the T/(T-k_eq) factor,\nwhich is what the old kind='f' default compared "
          "against Stata's chi2 (an F vs a chi2 is\neven worse: F = 20.6804 vs "
          "chi2 = 20.8900 on VAR(1)).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
