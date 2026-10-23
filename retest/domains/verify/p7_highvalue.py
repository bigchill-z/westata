"""AUDIT probe 7 — the three highest-value fail claims outside discrete/tobit:
  (1) systems_causal `nl` quadratic: pinv rcond silently drops a direction
  (2) systems_causal `teffects nnmatch` ATET SE == ATE SE (identical)
  (3) panel `xtpcse` pairwise == casewise (silently same numbers)
  (4) postest `margins` logit AME SE via fixed h=1e-6 central difference
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "retest/domains/verify")
sys.path.insert(0, "src")

from my_stata import S, rel

import westata as ws

auto = pd.read_stata("tests/data/auto.dta", convert_categoricals=False)
A = pd.DataFrame({c: auto[c].to_numpy(float) for c in
                  ("mpg", "weight", "price", "foreign", "length")})

print("=" * 78)
print("### 1. nl quadratic: is the SE collapse pinv's rcond?")
with S() as s:
    s.send(A)
    b, V, nm = s.est("nl (mpg = {b0} + {b1}*weight + {b2}*weight^2)")
    print("  Stata names:", nm)
    print("  Stata b  =", b)
    print("  Stata se =", np.sqrt(np.diag(V)))
nl = ws.Nl.from_formula("mpg = {b0} + {b1}*weight + {b2}*weight**2", A).fit()
print("  westata names:", list(nl.param_names))
print("  westata b  =", np.asarray(nl.params))
print("  westata se =", np.asarray(nl.bse))
print(f"  b rel  = {rel(np.asarray(nl.params), b):.4e}")
print(f"  se rel = {rel(np.asarray(nl.bse), np.sqrt(np.diag(V))):.4e}")

# independent: build J from the analytic design and test inv vs pinv
w = A["weight"].to_numpy()
J = np.column_stack([np.ones_like(w), w, w**2])
JtJ = J.T @ J
print(f"\n  J'J diag = {np.diag(JtJ)}")
print(f"  cond(J'J) = {np.linalg.cond(JtJ):.4e}")
sv = np.linalg.svd(JtJ, compute_uv=False)
print(f"  singular values = {sv}")
print(f"  s_min/s_max = {sv[-1]/sv[0]:.4e}   numpy default rcond = {JtJ.shape[0]*np.finfo(float).eps:.4e}")
resid = A["mpg"].to_numpy() - J @ np.asarray(nl.params)
n = len(w)
k = 3
sigma2 = float(resid @ resid) / (n - k)
for lbl, Vc in (("inv", np.linalg.inv(JtJ)),
                ("pinv default", np.linalg.pinv(JtJ)),
                ("pinv rcond=0", np.linalg.pinv(JtJ, rcond=0.0))):
    se = np.sqrt(np.diag(sigma2 * Vc))
    print(f"  {lbl:14s} se = {se}   rel vs Stata = {rel(se, np.sqrt(np.diag(V))):.4e}")
print(f"  westata se                     = {np.asarray(nl.bse)}")

print("\n" + "=" * 78)
print("### 2. teffects nnmatch: is ATET's SE literally identical to ATE's?")
rng = np.random.default_rng(20261024)
m = 400
x1 = rng.normal(size=m); x2 = rng.normal(size=m)
t = (rng.random(m) < 0.4).astype(float)
y = 1.0 + 0.8 * x1 - 0.5 * x2 + 2.0 * t + rng.normal(size=m)
D = pd.DataFrame(dict(y=y, t=t, x1=x1, x2=x2))
with S() as s:
    s.send(D)
    for est in ("", ", atet"):
        s.run(f"quietly teffects nnmatch (y x1 x2) (t){est}")
        print(f"  Stata nnmatch{est or ' (ate)'}: ATE={s.e('ate')} SE={s.e('se')}")
        print(f"      e(b)={s.mat('e(b)')}, se={np.sqrt(np.diag(s.mat('e(V)')))}")
    s.run("quietly teffects nnmatch (y x1 x2) (t)")
    b_ate, V_ate = s.mat("e(b)").reshape(-1), s.mat("e(V)")
    s.run("quietly teffects nnmatch (y x1 x2) (t), atet")
    b_atet, V_atet = s.mat("e(b)").reshape(-1), s.mat("e(V)")
    print(f"  Stata ATE se={np.sqrt(np.diag(V_ate))}  ATET se={np.sqrt(np.diag(V_atet))}"
          f"  ratio={np.sqrt(np.diag(V_atet))/np.sqrt(np.diag(V_ate))}")
mm_ate = ws.NearestNeighbourMatching.from_formula("y", "t", D, covariates=["x1", "x2"],
                                                 estimand="ate", nneighbor=1).fit()
mm_atet = ws.NearestNeighbourMatching.from_formula("y", "t", D, covariates=["x1", "x2"],
                                                  estimand="atet", nneighbor=1).fit()
se_a = float(mm_ate.std_err); se_t = float(mm_atet.std_err)
print(f"  westata ATE  estimate={mm_ate.estimate:.12f} se={se_a:.17f}")
print(f"  westata ATET estimate={mm_atet.estimate:.12f} se={se_t:.17f}")
print(f"  -> ATET se == ATE se bitwise? {se_a == se_t}   diff = {se_a - se_t:.3e}")
print(f"  -> westata ATET/ATE se ratio = {se_t/se_a:.10f}")
print(f"  -> Stata    ATET/ATE se ratio = {np.sqrt(np.diag(V_atet))[0]/np.sqrt(np.diag(V_ate))[0]:.10f}")
print(f"  rel(westata ATE  se, Stata) = {rel([se_a],[np.sqrt(np.diag(V_ate))[0]]):.4e}")
print(f"  rel(westata ATET se, Stata) = {rel([se_t],[np.sqrt(np.diag(V_atet))[0]]):.4e}")
print(f"  rel(westata ATE  est, Stata) = {rel([mm_ate.estimate],[b_ate[0]]):.4e}")
print(f"  rel(westata ATET est, Stata) = {rel([mm_atet.estimate],[b_atet[0]]):.4e}")

print("\n" + "=" * 78)
print("### 3. xtpcse: pairwise vs casewise on an UNBALANCED panel")
rng2 = np.random.default_rng(2026)
G, T = 20, 6
rows = []
for g in range(G):
    ui = rng2.normal()
    for tt in range(T):
        if tt >= 3 and rng2.random() < 0.35:
            continue
        x1v = rng2.normal(); x2v = rng2.normal()
        rows.append((g, tt, 1.0 + 0.8 * x1v - 0.5 * x2v + ui + rng2.normal(0, 0.5),
                     x1v, x2v))
P = pd.DataFrame(rows, columns=["id", "t", "y", "x1", "x2"])
with S() as s:
    s.send(P)
    s.run("xtset id t")
    s.run("quietly xtpcse y x1 x2")
    b_cw, V_cw = s.mat("e(b)").reshape(-1), s.mat("e(V)")
    s.run("quietly xtpcse y x1 x2, pairwise")
    b_pw, V_pw = s.mat("e(b)").reshape(-1), s.mat("e(V)")
    print(f"  Stata casewise se = {np.sqrt(np.diag(V_cw))}")
    print(f"  Stata pairwise se = {np.sqrt(np.diag(V_pw))}")
    print(f"  Stata casewise==pairwise? {np.allclose(V_cw, V_pw)}"
          f"  ratio={np.sqrt(np.diag(V_pw))/np.sqrt(np.diag(V_cw))}")
m_cw = ws.XtPcse.from_formula("y ~ x1 + x2", P, entity="id", time="t",
                              selection="casewise").fit()
m_pw = ws.XtPcse.from_formula("y ~ x1 + x2", P, entity="id", time="t",
                              selection="pairwise").fit()
print(f"  westata casewise se = {np.asarray(m_cw.bse)}")
print(f"  westata pairwise se = {np.asarray(m_pw.bse)}")
print(f"  -> westata pairwise == casewise bitwise? "
      f"{np.array_equal(np.asarray(m_cw.bse), np.asarray(m_pw.bse))}")
print(f"  rel(westata casewise, Stata casewise) = "
      f"{rel(np.asarray(m_cw.bse), np.sqrt(np.diag(V_cw))):.4e}")
print(f"  rel(westata pairwise, Stata pairwise) = "
      f"{rel(np.asarray(m_pw.bse), np.sqrt(np.diag(V_pw))):.4e}")

print("\n" + "=" * 78)
print("### 4. margins logit AME SE: analytic delta method as arbiter")
with S() as s:
    s.send(A)
    s.run("quietly logit foreign weight mpg")
    s.run("margins, dydx(weight)")
    amb = s.mat("r(b)").reshape(-1)
    amV = s.mat("r(V)")
    print("  Stata AME  =", amb, " se =", np.sqrt(np.diag(amV)))
    lg = ws.Logit.from_formula("foreign ~ weight + mpg", A).fit()
    w_ = A["weight"].to_numpy(); mp_ = A["mpg"].to_numpy()
    X = np.column_stack([np.ones(len(A)), w_, mp_])
    bb = np.array([np.asarray(lg.params)[list(lg.param_names).index(k)]
                   for k in ("Intercept", "weight", "mpg")])
    Vb = np.asarray(lg.cov_params())
    idxw = [list(lg.param_names).index(k) for k in ("Intercept", "weight", "mpg")]
    Vb = Vb[np.ix_(idxw, idxw)]
    p = 1 / (1 + np.exp(-(X @ bb)))
    sp = p * (1 - p)
    # AME for weight: AME = b_w * mean(sp); gradient wrt b via chain rule
    n_ = len(A)
    gg = np.zeros(3)
    j = 1  # weight position
    dsp = sp[:, None] * (1 - 2 * p)[:, None] * X        # (n,3)
    for kk in range(3):
        gg[kk] = np.mean(dsp[:, kk]) * bb[j]
    gg[j] += np.mean(sp)
    se_analytic = float(np.sqrt(gg @ Vb @ gg))
    print(f"  analytic AME se (using westata's b/V) = {se_analytic:.12f}")
    print(f"  Stata                                 = {float(np.sqrt(np.diag(amV))[0]):.12f}")
    print(f"  rel(analytic, Stata) = {rel([se_analytic], [float(np.sqrt(np.diag(amV))[0])]):.3e}")
    mg = lg.margins(dydx="weight")
    wse = np.atleast_1d(np.asarray(mg.std_err, dtype=float))
    west = np.atleast_1d(np.asarray(mg.estimate, dtype=float))
    print(f"  westata margins: estimate={west}  std_err={wse}")
    print(f"  rel(westata AME est, Stata)  = {rel(west, amb):.4e}")
    print(f"  rel(westata AME se, Stata)   = "
          f"{rel(wse, np.atleast_1d(np.sqrt(np.diag(amV)))):.4e}")
    print(f"  rel(westata AME se, analytic)= {rel(wse, [se_analytic]):.4e}")
    # step-size sweep to prove the finite-difference cancellation story
    import westata.postestimation.margins as MG
    orig = MG._statistic_and_gradient if hasattr(MG, "_statistic_and_gradient") else None
    print(f"  (finite-difference step and _statistic_and_gradient present: {orig is not None})")
