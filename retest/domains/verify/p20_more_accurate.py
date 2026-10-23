"""AUDIT probe 20 — two "westata is MORE accurate than Stata" claims.

These are the riskiest kind of upstream conclusion: if the adjudication is
wrong, the report tells the maintainer to fix correct code. We re-derive both
from scratch with exact arithmetic.

  (1) systems_causal §7: jackknife SE — Stata stores replicates in float32,
      westata in float64; westata agrees with the exact rational truth.
  (2) linear_iv §5: OLS e(F) is hard-coded to the classical R^2 form and does
      not follow cov_type.
"""

from __future__ import annotations

import sys
from fractions import Fraction

import numpy as np
import pandas as pd

sys.path.insert(0, "retest/domains/verify")
sys.path.insert(0, "src")

from my_stata import S, rel

import westata as ws

auto = pd.read_stata("tests/data/auto.dta", convert_categoricals=False)
mpg = auto["mpg"].to_numpy()
wt = auto["weight"].to_numpy()
n = len(mpg)

print("=" * 80)
print("### (1) jackknife SE: exact rational arithmetic as the arbiter")
# leave-one-out OLS on integer data -> exact Fractions
X = [[Fraction(1), Fraction(int(w))] for w in wt]
Y = [Fraction(int(v)) for v in mpg]


def fit_excl(idx):
    XtX = [[Fraction(0)] * 2 for _ in range(2)]
    Xty = [Fraction(0)] * 2
    for i in range(n):
        if i == idx:
            continue
        for a in range(2):
            for b in range(2):
                XtX[a][b] += X[i][a] * X[i][b]
            Xty[a] += X[i][a] * Y[i]
    det = XtX[0][0] * XtX[1][1] - XtX[0][1] * XtX[1][0]
    b0 = (XtX[1][1] * Xty[0] - XtX[0][1] * Xty[1]) / det
    b1 = (XtX[0][0] * Xty[1] - XtX[1][0] * Xty[0]) / det
    return b0, b1


reps = [fit_excl(i) for i in range(n)]
b0s = np.array([float(r[0]) for r in reps])
b1s = np.array([float(r[1]) for r in reps])
# jackknife variance: (n-1)/n * sum (theta_i - mean)^2
def jk_se(v):
    m = v.mean()
    return float(np.sqrt((n - 1) / n * np.sum((v - m) ** 2)))


se_true = np.array([jk_se(b0s), jk_se(b1s)])
print(f"  exact jackknife SE (const, weight) = {se_true}")

with S() as s:
    s.send(auto[["mpg", "weight"]])
    s.run("quietly jackknife: regress mpg weight")
    b, V = s.mat("e(b)").reshape(-1), s.mat("e(V)")
    print(f"  Stata jackknife e(b) = {b}")
    print(f"  Stata jackknife se   = {np.sqrt(np.diag(V))}")
se_stata = np.sqrt(np.diag(V))[[1, 0]]
print(f"  rel(Stata, exact)   = {rel(se_stata, se_true):.4e}")

jk = ws.Jackknife.from_formula("mpg ~ weight", auto[["mpg", "weight"]]).fit()
pn = list(jk.param_names)
print(f"  westata param_names = {pn}")
se_w = np.array([float(np.asarray(jk.bse)[pn.index(k)]) for k in ("Intercept", "weight")])
print(f"  westata jackknife se = {se_w}")
print(f"  rel(westata, exact) = {rel(se_w, se_true):.4e}")
print(f"  rel(westata, Stata) = {rel(se_w, se_stata):.4e}")
# float32 round-trip test
s32 = np.array(b0s, dtype=np.float32).astype(np.float64)
s32b = np.array(b1s, dtype=np.float32).astype(np.float64)
se_f32 = np.array([jk_se(s32), jk_se(s32b)])
print(f"  exact reps rounded through float32 -> SE = {se_f32}")
print(f"  rel(float32-simulated, Stata)      = {rel(se_f32, se_stata):.4e}")
print(f"  rel(float64-exact,     Stata)      = {rel(se_true, se_stata):.4e}")
print("  VERDICT:", "westata closer to truth" if rel(se_w, se_true) < rel(se_stata, se_true)
      else "Stata closer / tie")

print()
print("=" * 80)
print("### (2) OLS e(F): is it frozen at the classical R^2 form?")
rng = np.random.default_rng(20261024)
G, T = 14, 10
N = G * T
gid = np.repeat(np.arange(G), T)
x1 = rng.normal(size=N); x2 = rng.normal(size=N)
u = np.repeat(rng.normal(0, 1, G), T)
P = pd.DataFrame(dict(gid=gid, x1=x1, x2=x2, y=1.0 + 0.8 * x1 - 0.5 * x2 + u + rng.normal(size=N)))
with S() as s:
    s.send(P)
    for tag, cmd in (("default", "regress y x1 x2"),
                     ("robust", "regress y x1 x2, vce(robust)"),
                     ("cluster", "regress y x1 x2, vce(cluster gid)")):
        b, V, nm = s.est(cmd)
        F_s = s.e("F")
        ii = [i for i, k in enumerate(nm) if k != "_cons"]
        bs, Vs = b[ii], V[np.ix_(ii, ii)]
        F_wald = float(bs @ np.linalg.solve(Vs, bs) / len(ii))
        print(f"  [{tag:8s}] Stata e(F)={F_s:.12f}   b'V^-1b/k on slopes={F_wald:.12f}"
              f"   rel={rel([F_wald],[F_s]):.3e}")
        for cov in ({"default": None, "robust": "HC1", "cluster": "cluster"}[tag],):
            kw = {}
            if cov == "cluster":
                kw = {"cov_kwds": {"groups": P["gid"].to_numpy()}}
            m = ws.OLS.from_formula("y ~ x1 + x2", P).fit(cov_type=cov, **kw) if cov else \
                ws.OLS.from_formula("y ~ x1 + x2", P).fit()
            print(f"             westata e(F)={m.e.get('F')}  rel vs Stata={rel([m.e.get('F')],[F_s]):.3e}"
                  f"   results.F={getattr(m,'fvalue',None)}")
