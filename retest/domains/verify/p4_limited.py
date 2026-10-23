"""AUDIT probe 4 — limited_quantile: independent re-derivation of the fail claims.

Claims under test:
  L1 qreg vce(robust) at tau != .5 : westata meat = sum(psi^2 xx'), Stata = tau(1-tau)X'X
  L2 rreg SE: Stata = e(rmse)*sqrt(diag((X'X)^-1))  (NOT pseudovalue)
  L3 heckman twostep: e(sigma)=sqrt(RSS/m + b_lambda^2*mean(lam(lam+w)))
  L4 etregress twostep: e(rho) sign
  L5 frontier: numeric-Hessian step artefact
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "retest/domains/verify")
sys.path.insert(0, "src")

from my_stata import S, rel

import westata as ws

# ------------------------------------------------------------------ qreg
rng = np.random.default_rng(4242)
n = 400
x1 = rng.normal(size=n)
x2 = rng.normal(size=n)
sig = np.exp(0.2 + 0.4 * x1)
y = 1.0 + 0.75 * x1 - 0.45 * x2 + sig * rng.normal(size=n)
df = pd.DataFrame(dict(x1=x1, x2=x2, y=y))
X = np.column_stack([x1, x2, np.ones(n)])

print("=" * 78)
print("### L1. qreg vce(robust) at tau=.25 : which meat does each side use?")
with S() as s:
    s.send(df)
    b, V, nm = s.est("qreg y x1 x2, quantile(.25) vce(robust)")
    _b0, _V0, _ = s.est("qreg y x1 x2, quantile(.25)")
    print("  [check] e(b)/e(V) with vs without vce(robust) differ?",
          np.max(np.abs(_b0 - b)), np.max(np.abs(_V0 - V)))
    print("  coefficients (robust == default):", np.round(b, 10))
    print("  Stata names:", nm)
    print("  Stata robust se =", np.round(np.sqrt(np.diag(V)), 10))
    q = ws.QuantileRegression.from_formula("y ~ x1 + x2", df, quantile=0.25).fit(cov_type="robust")
    print("  westata robust se =", np.round(np.asarray(q.bse), 10),
          " names", list(q.param_names))
    # reorder westata -> stata
    pn = list(q.param_names)
    order = [pn.index(k) for k in ("x1", "x2", "Intercept")]
    print(f"  se rel (name-aligned) = {rel(np.asarray(q.bse)[order], np.sqrt(np.diag(V))):.4e}")

    # manual sandwich: J from Stata's qreg density, meat = tau(1-tau) X'X
    beta = b  # Stata's estimate
    u = y - X @ beta
    tau = 0.25
    # Hall-Sheather bandwidth as Stata reports it
    bw = s.e("bwidth")
    sparsity = s.e("sparsity")
    if sparsity is None:
        s.run("quietly qreg y x1 x2, quantile(.25)")
        sparsity = s.e("sparsity")
        bw = s.e("bwidth")
        s.run("quietly qreg y x1 x2, quantile(.25) vce(robust)")
    print(f"  Stata bwidth={bw} sparsity={sparsity} sum_adev={s.e('sum_adev')}")
    # density per obs via the qreg (Koenker) rule
    # fit at tau+-h to get f_i (Stata's own recipe uses sparsity = n*h/sum(u in [-h,h]))
    h = 0.5 * n ** (-1 / 3) * (
        __import__("scipy").stats.norm.ppf(0.75) - __import__("scipy").stats.norm.ppf(0.25)
    ) ** (2 / 3) if False else None
    # Use the reported sparsity: J = (n*h/sparsity) * X'X / n ... simpler:
    # Stata: V = tau(1-tau) * sparsity * (X'X)^-1 X'X (X'X)^-1 = tau(1-tau)*sparsity*(X'X)^-1
    XtX_inv = np.linalg.inv(X.T @ X)
    V_stata_form = tau * (1 - tau) * sparsity * XtX_inv
    print(f"  tau(1-tau)*sparsity*(X'X)^-1 se = {np.round(np.sqrt(np.diag(V_stata_form)),10)}"
          f"   rel vs Stata = {rel(np.sqrt(np.diag(V_stata_form)), np.sqrt(np.diag(V))):.4e}")

    # westata's implied meat: recover M = J V J with J = X'X / (n*h) style; simpler:
    # check whether westata's V equals J^-1 [sum psi^2 xx'] J^-1 with psi=tau-1{u<0}
    psi = tau - (u < 0).astype(float)
    M2 = (X * psi[:, None] ** 2).T @ X
    # J = sum f_i x_i x_i' ; use Stata's sparsity-based scalar density proxy
    # Stata's J is (n*h/sparsity)*mean? Use: J = (1/(n*h))*sum 1{|u_i|<=h} x x'
    hh = bw
    w = (np.abs(u) <= hh).astype(float)
    J = (X * w[:, None]).T @ X / (2 * n * hh)
    V_psi = np.linalg.inv(J) @ M2 @ np.linalg.inv(J)
    V_const = np.linalg.inv(J) @ (tau * (1 - tau) * X.T @ X) @ np.linalg.inv(J)
    print(f"  my sandwich with psi^2 meat       se = {np.round(np.sqrt(np.diag(V_psi)),10)}")
    print(f"  my sandwich with tau(1-tau)X'X    se = {np.round(np.sqrt(np.diag(V_const)),10)}")
    print(f"    rel(psi^2 meat,        Stata) = {rel(np.sqrt(np.diag(V_psi)), np.sqrt(np.diag(V))):.4e}")
    print(f"    rel(tau(1-tau)X'X meat,Stata) = {rel(np.sqrt(np.diag(V_const)), np.sqrt(np.diag(V))):.4e}")
    print(f"    rel(psi^2 meat,   westata) = {rel(np.sqrt(np.diag(V_psi)), np.asarray(q.bse)[order]):.4e}")
    print(f"    ratio Stata/westata se = {np.sqrt(np.diag(V))/np.asarray(q.bse)[order]}")

print("\n" + "=" * 78)
print("### L2. rreg SE: e(rmse)*sqrt(diag((X'X)^-1)) ?")
rng2 = np.random.default_rng(7)
n2 = 300
a1 = rng2.normal(size=n2)
a2 = rng2.normal(size=n2)
yy = 1.0 + 0.6 * a1 - 0.3 * a2 + rng2.normal(size=n2)
d2 = pd.DataFrame(dict(x1=a1, x2=a2, y=yy))
X2 = np.column_stack([a1, a2, np.ones(n2)])
with S() as s:
    s.send(d2)
    b, V, nm = s.est("rreg y x1 x2")
    rmse = s.e("rmse")
    rss = s.e("rss")
    print("  Stata rreg: se =", np.round(np.sqrt(np.diag(V)), 10))
    print(f"  Stata e(rmse)={rmse}  e(rss)={rss}  e(df_r)={s.e('df_r')}")
    se_law = rmse * np.sqrt(np.diag(np.linalg.inv(X2.T @ X2)))
    print(f"  rmse*sqrt(diag((X'X)^-1))            = {np.round(se_law,10)}"
          f"   rel = {rel(se_law, np.sqrt(np.diag(V))):.4e}")
    print(f"  cross-check: rmse^2*(n-k) == rss ?  {rmse**2*(n2-3):.10f} vs {rss:.10f}")
    rr = ws.RobustRegression.from_formula("y ~ x1 + x2", d2).fit()
    pn = list(rr.param_names)
    order = [pn.index(k) for k in ("x1", "x2", "Intercept")]
    print(f"  westata rreg se = {np.round(np.asarray(rr.bse)[order],10)}")
    print(f"  -> rel(Stata, westata) = {rel(np.sqrt(np.diag(V)), np.asarray(rr.bse)[order]):.4e}")
    print(f"  -> rel(Stata law, westata) = {rel(se_law, np.asarray(rr.bse)[order]):.4e}")

print("\n" + "=" * 78)
print("### L3/L4. heckman twostep e(sigma)/e(rho); etregress e(rho) sign")
rng3 = np.random.default_rng(2024)
m = 600
zz1 = rng3.normal(size=m)
zz2 = rng3.normal(size=m)
xx1 = rng3.normal(size=m)
rho_true = 0.55
w = 0.6 * zz1 - 0.4 * zz2
v = rho_true * w + np.sqrt(1 - rho_true**2) * rng3.normal(size=m)
sel = (w + rng3.normal(size=m) * 1.0) > 0
yout = np.where(sel, 1.0 + 0.8 * xx1 + 0.6 * v, np.nan)
d3 = pd.DataFrame(dict(y=yout, s=sel.astype(float), x1=xx1, z1=zz1, z2=zz2))
with S() as s:
    s.send(d3.fillna(-9.0))  # Stata needs the missing as .
    s.run("replace y = . if s == 0")
    b, V, nm = s.est("heckman y x1, select(s = z1 z2) twostep")
    print("  heckman twostep Stata names:", nm)
    print("  Stata b :", np.round(b, 10))
    print("  Stata se:", np.round(np.sqrt(np.diag(V)), 10))
    for k in ("sigma", "rho", "lambda", "selambda", "ll", "ll_p", "N"):
        print(f"     e({k}) = {s.e(k)}")
    # independent reconstruction of Stata's sigma
    # probit on selection
    import statsmodels.api as sm
    Zw = np.column_stack([zz1, zz2, np.ones(m)])
    pr = sm.Probit(sel.astype(float), Zw).fit(disp=0)
    wlin = Zw @ pr.params
    delta = pr.model.predict(pr.params)
    lam = np.exp(-0.5 * wlin**2) / np.sqrt(2 * np.pi) / delta
    sub = sel
    Xs = np.column_stack([xx1[sub], np.ones(sub.sum())])
    ys = yout[sub]
    lam_s = lam[sub]
    coef = np.linalg.lstsq(np.column_stack([Xs, lam_s]), ys, rcond=None)[0]
    resid = ys - np.column_stack([Xs, lam_s]) @ coef
    mm = sub.sum()
    b_lam = coef[-1]
    sig_form = np.sqrt(resid @ resid / mm + b_lam**2 * np.mean(lam_s * (lam_s + wlin[sub])))
    print(f"\n  independent sigma reconstruction = {sig_form:.10f}")
    print(f"  Stata e(sigma)                   = {s.e('sigma'):.10f}"
          f"   rel={rel([sig_form],[s.e('sigma')]):.4e}")
    hk = ws.Heckman.from_formula("y ~ x1", d3.dropna(), selection="s ~ z1 + z2").fit()
    print(f"  westata e(sigma) = {hk.e.get('sigma')}   e(rho) = {hk.e.get('rho')}")
    print(f"  Stata    e(sigma) = {s.e('sigma')}   e(rho) = {s.e('rho')}")
    print(f"  rel(sigma) = {rel([hk.e.get('sigma')],[s.e('sigma')]):.4e}"
          f"   rel(rho) = {rel([hk.e.get('rho')],[s.e('rho')]):.4e}")
    print(f"  westata params = {list(hk.param_names)}")
    print(f"  lambda coef: Stata {b[nm.index('lambda')]}  westata "
          f"{np.asarray(hk.params)[list(hk.param_names).index('lambda')] if 'lambda' in hk.param_names else 'n/a'}")

print("\n  --- etregress twostep ---")
rng4 = np.random.default_rng(31)
mm = 600
u1 = rng4.normal(size=mm)
u2 = rng4.normal(size=mm)
zz = 0.8 * u1 + 0.6 * rng4.normal(size=mm)
tr = (zz + u1 > 0).astype(float)
rho_e = 0.45
eps = rho_e * zz + np.sqrt(1 - rho_e**2) * u2
yy2 = 1.0 + 0.7 * rng4.normal(size=mm) + 1.5 * tr + eps
d4 = pd.DataFrame(dict(y=yy2, t=tr, z1=zz, z2=rng4.normal(size=mm)))
with S() as s:
    s.send(d4)
    b, V, nm = s.est("etregress y z2, treat(t = z1 z2) twostep")
    print("  Stata etregress names:", nm)
    print("  Stata b :", np.round(b, 10))
    print("  Stata se:", np.round(np.sqrt(np.diag(V)), 10))
    for k in ("rho", "sigma", "lambda", "selambda", "ll", "ll_p", "N"):
        print(f"     e({k}) = {s.e(k)}")
    print(f"  e(lambda)/e(sigma) = {s.e('lambda')/s.e('sigma'):.10f}  vs e(rho) = {s.e('rho')}")
    et = ws.Etregress.from_formula("y ~ z2 + t", d4, treat="t ~ z1 + z2").fit()
    print(f"  westata params = {list(et.param_names)}")
    print(f"  westata b = {np.round(np.asarray(et.params),10)}")
    print(f"  westata e = {{k: et.e.get(k) for k in ('rho','sigma','lambda')}}".replace("{k: et.e.get(k) for k in ('rho','sigma','lambda')}", str({k: et.e.get(k) for k in ('rho','sigma','lambda')})))
