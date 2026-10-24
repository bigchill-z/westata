"""Repro/verification for the `limited` group, T1-T3: tobit.

Run:  .venv\\Scripts\\python.exe retest/fix/limited/repro_tobit.py

T1  `/sigma` SE was the SE of ln(sigma)  (missing Jacobian d sigma / d ln sigma)
T2  `e(df_r)` was N - k instead of Stata's N - df_m   (df_m = rank - 1 - hascons)
T3  coefficient SEs came from a finite-difference Hessian (step=1e-5)

Every number is captured live from StataNow 19.5 in this process.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "retest/lib")

import numpy as np
import pandas as pd

import westata as ws
from stata_probe import Stata, canon_stata, canon_westata

# --------------------------------------------------------------------- data
rng = np.random.default_rng(2024)
n = 500
x1 = rng.normal(size=n)
x2 = rng.normal(size=n)
ystar = 1.0 + 0.8 * x1 - 0.5 * x2 + rng.normal(size=n)
tobit_df = pd.DataFrame({"x1": x1, "x2": x2})
tobit_df["y"] = np.clip(ystar, 0.0, None)          # ll(0)
tobit_df["yu"] = np.minimum(ystar, 2.5)            # ul(2.5)
tobit_df["yb"] = np.clip(ystar, 0.0, 2.0)          # ll(0) ul(2)

# A second, differently shaped dataset (n=200, k=2) for the df_r rule.
rng2 = np.random.default_rng(7)
x = rng2.normal(size=200)
small = pd.DataFrame({"x": x, "y": np.clip(1.2 + 0.6 * x + rng2.normal(size=200),
                                           0.0, None)})

CASES = [
    ("tobit y x1 x2, ll(0)", "y ~ x1 + x2", tobit_df, dict(lower=0.0)),
    ("tobit yu x1 x2, ul(2.5)", "yu ~ x1 + x2", tobit_df, dict(lower=None, upper=2.5)),
    ("tobit yb x1 x2, ll(0) ul(2)", "yb ~ x1 + x2", tobit_df,
     dict(lower=0.0, upper=2.0)),
    ("tobit y x, ll(0)", "y ~ x", small, dict(lower=0.0)),
]


def rel(a, e):
    a = np.asarray(a, dtype=float)
    e = np.asarray(e, dtype=float)
    return float(np.max(np.abs(a - e) / np.maximum(np.abs(e), 1e-12)))


worst = {"param": 0.0, "coef_se": 0.0, "sigma_se": 0.0, "stdp": 0.0, "df_r": 0.0}

with Stata() as st:
    print(f"Stata {st.version}\n")
    print(f"{'case':30s} {'quantity':18s} {'westata':>16s} {'Stata':>16s} {'rel':>10s}")
    print("-" * 98)
    for command, formula, frame, kw in CASES:
        st.send(frame)
        cap = st.estimate(command)
        res = ws.Tobit.from_formula(formula, frame, **kw).fit()

        stata_names = list(cap.names)
        k = len(stata_names) - 1                      # last row is var(e.y)
        stata_b = np.asarray(cap.b, dtype=float).copy()
        stata_se = np.sqrt(np.diag(np.asarray(cap.V, dtype=float))).copy()
        # Stata stores var(e.y) = sigma^2; map it and its SE to the sigma scale
        # by the delta method (d sigma / d var = 1/(2 sigma)).
        stata_b[k] = np.sqrt(stata_b[k])
        stata_se[k] = stata_se[k] / (2.0 * stata_b[k])

        # Align by name: Stata order is (x1, x2, _cons, var(e.y)); westata's is
        # (Intercept, x1, x2, /sigma).
        index = {}
        for i, nm in enumerate(stata_names[:-1]):
            index[canon_stata(nm)] = i
        index["__scale__"] = k
        order = []
        for nm in res.param_names:
            order.append(index["__scale__"] if nm == "/sigma" else index[canon_westata(nm)])
        order = np.asarray(order, dtype=int)

        ours_b = np.asarray(res.params, dtype=float)
        ours_se = np.asarray(res.bse, dtype=float)
        param_rel = rel(ours_b, stata_b[order])
        coef_mask = np.array([nm != "/sigma" for nm in res.param_names])
        coef_se_rel = rel(ours_se[coef_mask], stata_se[order][coef_mask])
        sigma_se_rel = rel(ours_se[-1], stata_se[k])
        worst["param"] = max(worst["param"], param_rel)
        worst["coef_se"] = max(worst["coef_se"], coef_se_rel)
        worst["sigma_se"] = max(worst["sigma_se"], sigma_se_rel)

        print(f"{command:30s} {'b (all, aligned)':18s} {'':>16s} {'':>16s} "
              f"{param_rel:10.2e}")
        for pos, nm in enumerate(res.param_names):
            print(f"{'':30s} {('se: ' + nm)[:18]:18s} {ours_se[pos]:16.10g} "
                  f"{stata_se[order][pos]:16.10g} "
                  f"{abs(ours_se[pos]-stata_se[order][pos])/stata_se[order][pos]:10.2e}")

        # T2: residual degrees of freedom
        stata_df_r = float(cap.scalars["df_r"])
        stata_df_m = float(cap.scalars["df_m"])
        print(f"{'':30s} {'e(df_r)':18s} {res.e['df_r']:16.10g} "
              f"{stata_df_r:16.10g} "
              f"{abs(res.e['df_r']-stata_df_r)/stata_df_r:10.2e}")
        print(f"{'':30s} {'e(df_m)':18s} {res.e['df_m']:16.10g} "
              f"{stata_df_m:16.10g} "
              f"{abs(res.e['df_m']-stata_df_m)/stata_df_m:10.2e}")
        print(f"{'':30s} {'e(sigma)':18s} {res.e['sigma']:16.10g} "
              f"{stata_b[k]:16.10g} "
              f"{abs(res.e['sigma']-stata_b[k])/stata_b[k]:10.2e}")
        worst["df_r"] = max(worst["df_r"], abs(res.e['df_r'] - stata_df_r) / stata_df_r)

        # T3: stdp = sqrt(x'Vx) at every observation, against Stata's predict.
        st.quiet("predict double _stdp, stdp")
        stata_stdp = st.var("_stdp")
        beta_index = [i for i, nm in enumerate(res.param_names) if nm != "/sigma"]
        columns = []
        for nm in np.asarray(res.param_names)[beta_index]:
            columns.append(frame[nm].to_numpy(dtype=float)
                           if nm in frame.columns else np.ones(len(frame)))
        X = np.column_stack(columns)
        cov_all = np.asarray(res.cov_params(), dtype=float)
        ours_stdp = np.sqrt(np.einsum("ij,jk,ik->i", X,
                                      cov_all[np.ix_(beta_index, beta_index)], X))
        stdp_rel = rel(ours_stdp, stata_stdp)
        print(f"{'':30s} {'predict stdp':18s} {'':>16s} {'':>16s} {stdp_rel:10.2e}")
        worst["stdp"] = max(worst["stdp"], stdp_rel)
        print()

print("=" * 98)
print(f"worst coefficient b   rel = {worst['param']:.3e}")
print(f"worst coefficient se  rel = {worst['coef_se']:.3e}")
print(f"worst se(/sigma)      rel = {worst['sigma_se']:.3e}   (T1 target <= 1e-6)")
print(f"worst e(df_r)         rel = {worst['df_r']:.3e}   (T2 target exact)")
print(f"worst predict stdp    rel = {worst['stdp']:.3e}   (T3 target <= 1e-9)")
