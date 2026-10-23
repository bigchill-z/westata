"""AUDIT probe 21 — systems_causal: aipw ATET branch and didregress silent
covariate inference.

Both are "dangerous default / wrong code path" claims; we confirm the mechanism
rather than just the number.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, "retest/domains/verify")
sys.path.insert(0, "src")

from my_stata import S, rel

import westata as ws

print("=" * 80)
print("### A. teffects aipw atet: is ATE exact and ATET broken?")
rng = np.random.default_rng(941)
m = 400
x1 = rng.normal(size=m)
x2 = rng.normal(size=m)
t = (rng.random(m) < 0.4).astype(float)
y = 2.0 + 0.9 * x1 - 0.6 * x2 + 2.5 * t + rng.normal(size=m)
D = pd.DataFrame(dict(y=y, t=t, x1=x1, x2=x2))
with S() as s:
    s.send(D)
    for est, cmdtxt in (("ate", "teffects aipw (y x1 x2) (t x1 x2)"),
                        ("atet", "teffects aipw (y x1 x2) (t x1 x2), atet")):
        s.run("quietly " + cmdtxt)
        b, V = s.mat("e(b)").reshape(-1), s.mat("e(V)")
        nmb = s.cnames("e(b)")
        print(f"  Stata {est:4s}: e(b) names={nmb} b={np.round(b,10)} se={np.round(np.sqrt(np.diag(V)),10)}")
for est in ("ate", "atet"):
    r = ws.AIPW.from_formula("y", "t", D, covariates=["x1", "x2"], estimand=est).fit()
    print(f"  westata {est:4s}: estimate={r.estimate:.12f} se={r.std_err:.12f}")

print()
print("=" * 80)
print("### B. didregress silently infers covariates when covariates=None")
rng2 = np.random.default_rng(515)
U = 60
rows = []
for i in range(U):
    fe = rng2.normal(0, 1)
    for period in (1, 2):
        d = 1 if (i >= 30 and period == 2) else 0
        rows.append(dict(unit=i + 1, period=period, treated=d,
                         y=fe + 0.5 * (period - 1) + 1.2 * d + rng2.normal(0, 0.7),
                         x1=rng2.normal()))
Dd = pd.DataFrame(rows)
with S() as s:
    s.send(Dd)
    s.run("quietly didregress (y) (treated), group(unit) time(period)")
    b_narrow, Vn = s.mat("e(b)").reshape(-1), s.mat("e(V)")
    nmn = s.cnames("e(b)")
    print(f"  Stata names {nmn}")
    print(f"  Stata (y)             b={np.round(b_narrow,10)} se={np.round(np.sqrt(np.diag(Vn)),10)}")
    s.run("quietly didregress (y x1) (treated), group(unit) time(period)")
    b_wide, Vw = s.mat("e(b)").reshape(-1), s.mat("e(V)")
    print(f"  Stata (y x1)          b={np.round(b_wide,10)} se={np.round(np.sqrt(np.diag(Vw)),10)}")

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    r_auto = ws.Didregress.from_formula("y", "treated", Dd, group="unit", time="period").fit()
    msgs = [str(w.message) for w in caught]
print("  westata covariates=None with an extra time-varying numeric column:")
print(f"    param_names={list(r_auto.param_names)}")
print(f"    params={np.round(np.asarray(r_auto.params),10)}")
print(f"    bse={np.round(np.asarray(r_auto.bse),10)}")
print(f"    warnings: {msgs}")
narrow = Dd[["unit", "period", "treated", "y"]]
r_narrow = ws.Didregress.from_formula("y", "treated", narrow, group="unit", time="period").fit()
print(f"  westata on the narrow table:")
print(f"    param_names={list(r_narrow.param_names)}")
print(f"    params={np.round(np.asarray(r_narrow.params),10)}")
print(f"    bse={np.round(np.asarray(r_narrow.bse),10)}")
i_n = [j for j, k in enumerate(nmn) if "treated" in k]
print(f"  Stata ATET row index {i_n}  b={b_narrow[i_n]}")
kat = float(np.asarray(r_narrow.params)[list(r_narrow.param_names).index("treated")])
kat_auto = float(np.asarray(r_auto.params)[list(r_auto.param_names).index("treated")])
print(f"    rel(westata narrow, Stata (y))    = {rel([kat], b_narrow[i_n]):.4e}")
print(f"    rel(westata auto,   Stata (y))    = {rel([kat_auto], b_narrow[i_n]):.4e}")
try:
    ws.Didregress.from_formula("y", "treated", Dd, group="unit", time="period",
                               covariates=[]).fit()
    print("  covariates=[] accepted")
except Exception as exc:
    print(f"  covariates=[] -> {type(exc).__name__}: {str(exc)[:100]}")
