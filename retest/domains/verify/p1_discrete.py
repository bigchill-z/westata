"""AUDIT probe 1 — discrete: ologit/oprobit e(V), e(df_m), poisson HC1, zip e(p), cluster df.

Independent re-derivation: we never call upstream run.py; we re-compute the
correct delta-method Jacobian ourselves from first principles.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, "retest/domains/verify")
sys.path.insert(0, "src")

from my_stata import S, rel

import westata as ws

rng = np.random.default_rng(20261024)
n = 600
x1 = rng.normal(size=n)
x2 = rng.normal(size=n)
eta = 0.5 + 0.8 * x1 - 0.4 * x2 + rng.logistic(size=n)
ycat = np.digitize(eta, [-1.0, 0.3, 1.6])  # 4 categories
yb = (eta > 0).astype(int)
y_c = rng.poisson(np.exp(0.4 + 0.5 * x1 - 0.3 * x2))
grp = rng.integers(0, 6, size=n)
df = pd.DataFrame(dict(x1=x1, x2=x2, ycat=ycat, yb=yb, y_c=y_c, grp=grp))

print("=" * 78)
print("A. ologit / oprobit e(V): is westata's Jacobian the transpose?")
with S() as s:
    s.send(df)
    print("  nobs(Stata) =", s.nobs(), " n =", n)
    for cmd, cls in (("ologit ycat x1 x2", ws.OrderedLogit), ("oprobit ycat x1 x2", ws.OrderedProbit)):
        b, V, names = s.est(cmd)
        print(f"\n  [{cmd}] Stata e(b) names = {names}")
        print(f"  Stata b = {b}")
        print(f"  Stata se= {np.sqrt(np.diag(V))}")
        res = cls.from_formula("ycat ~ x1 + x2", df).fit()
        print(f"  westata names = {list(res.param_names)}")
        print(f"  westata b  = {np.asarray(res.params)}")
        print(f"  westata se = {np.asarray(res.bse)}")
        # Stata order: x1 x2 /cut1 /cut2 /cut3
        wb = np.asarray(res.params, dtype=float)
        wse = np.asarray(res.bse, dtype=float)
        # map: find index of each name
        pn = list(res.param_names)
        idx = {k: pn.index(k) for k in pn}
        order = ["x1", "x2", "/cut1", "/cut2", "/cut3"]
        order = [k for k in order if k in idx]
        wp = np.array([wb[idx[k]] for k in order])
        ws_ = np.array([wse[idx[k]] for k in order])
        print(f"  -> westata in Stata order {order}: b {wp}")
        print(f"     point-est rel   = {rel(wp, b):.3e}")
        print(f"     se rel (as-is)  = {rel(ws_, np.sqrt(np.diag(V))):.3e}")

        # INDEPENDENT recomputation of the correct Jacobian.
        # statsmodels ordered model uses theta = [cut1, log(cut2-cut1), log(cut3-cut2)]
        # with cut_k = theta0 + sum_{j<=k} exp(theta_j) for k>=1 (0-based: cut[0]=theta0)
        th = np.asarray(res.model.transform_threshold_params(np.asarray(res.params)[None, :].T)) if False else None
        # Rebuild the cut points ourselves from the fitted model and check monotone spacing
        cuts = np.array([wb[idx[k]] for k in ["/cut1", "/cut2", "/cut3"]])
        print(f"     fitted cut points = {cuts}")
        # correct J for (cut1, cut2, cut3) wrt (theta0, theta1, theta2)
        J_ok = np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, (cuts[1] - cuts[0]), 0.0],
                [1.0, (cuts[1] - cuts[0]), (cuts[2] - cuts[1])],
            ]
        )
        # westata's stated (buggy) J is the transpose
        J_bad = J_ok.T
        Voc = V[: len(order), : len(order)]
        # V_theta = J^-1 V_cut (J^-1)'  ->  V_cut = J V_theta J'
        # we need V_theta: take westata's cov in its own parameterisation
        full = np.asarray(res.cov_params(), dtype=float)
        # full is wrt (x1,x2,theta0,theta1,theta2) reordered to param_names
        tnames = [k for k in pn if k not in ("x1", "x2")]
        Vi = np.array([idx[k] for k in tnames])
        Vtheta = full[np.ix_(Vi, Vi)]
        for label, J in (("correct J", J_ok), ("J transposed", J_bad)):
            Vcut = J @ Vtheta @ J.T
            print(f"     {label:14s}: cut se = {np.sqrt(np.diag(Vcut))}  rel vs Stata = "
                  f"{rel(np.sqrt(np.diag(Vcut)), np.sqrt(np.diag(V))[2:]):.3e}")

print()
print("=" * 78)
print("B. e(df_m) for nbreg / ologit ; e(p) for zip")
rng2 = np.random.default_rng(777)
n2 = 600
xa = rng2.normal(size=n2)
xb = rng2.normal(size=n2)
mu = np.exp(0.4 + 0.5 * xa - 0.3 * xb)
alpha = 0.5
lam = rng2.gamma(1.0 / alpha, alpha * mu)
y_nb = rng2.poisson(lam)
y_zi = np.where(rng2.random(n2) < 0.30, 0, rng2.poisson(np.exp(0.5 + 0.4 * xa)))
d2 = pd.DataFrame(dict(x1=xa, x2=xb, y_nb=y_nb, y_zi=y_zi))
with S() as s:
    s.send(d2)
    for cmd in ("nbreg y_nb x1 x2", "ologit ycat x1 x2", "zip y_zi x1 x2, inflate(x1 x2)"):
        try:
            s.run("quietly " + cmd)
        except Exception as exc:
            print("  ", cmd, "-> Stata error", exc)
            continue
        print(f"\n  [{cmd}]")
        for k in ("df_m", "df_r", "rank", "k_aux", "chi2", "p", "chi2type"):
            v = s.e(k)
            print(f"     e({k}) = {v}")
    # ologit on ycat (from earlier df) needs re-send; do it explicitly
    s.send(df)
    s.run("quietly ologit ycat x1 x2")
    print("\n  [ologit ycat x1 x2] df_m/rank/k_aux:", s.e("df_m"), s.e("rank"), s.e("k_aux"))
    s.send(d2)
    s.run("quietly zip y_zi x1 x2, inflate(x1 x2)")
    chi2, rank_e, df_m, pval = s.e("chi2"), s.e("rank"), s.e("df_m"), s.e("p")
    print(f"\n  [zip] chi2={chi2} rank={rank_e} df_m={df_m} e(p)={pval}")
    print(f"     sf(chi2, df_m)   = {sps.chi2.sf(chi2, df_m):.6e}")
    print(f"     sf(chi2, rank-1) = {sps.chi2.sf(chi2, rank_e - 1):.6e}")

print()
print("=" * 78)
print("C. Poisson HC1 vs HC0 in westata; and Stata's N/(N-1) factor")
with S() as s:
    s.send(df)
    s.est("poisson y_c x1 x2")
    p_st = s.mat("e(V)")
    s.est("poisson y_c x1 x2, vce(robust)")
    p_rob = s.mat("e(V)")
    s.est("poisson y_c x1 x2")
    print("  Stata: V_robust/V_ml elementwise ratio (should be sandwich):")
    try:
        ratio = p_rob / p_st
        print("   ", np.round(ratio.reshape(-1), 6))
    except Exception:
        pass
    po = ws.Poisson.from_formula("y_c ~ x1 + x2", df)
    r0 = po.fit(cov_type="HC0")
    r1 = po.fit(cov_type="HC1")
    print("  westata HC0 vs HC1 identical?", np.array_equal(np.asarray(r0.cov_params()), np.asarray(r1.cov_params())))
    print("  westata HC0 se =", np.asarray(r0.bse))
    print("  westata HC1 se =", np.asarray(r1.bse))
    # Manually build the sandwich with the Stata factor N/(N-1)
    import statsmodels.api as sm
    X = sm.add_constant(df[["x1", "x2"]].to_numpy())
    m = sm.Poisson(df["y_c"].to_numpy(), X).fit(disp=0, maxiter=200)
    sc_ = m.score_obs if hasattr(m, "score_obs") else None
    bread = np.linalg.inv(-m.model.hessian(m.params))
    meat0 = (sc_ * m.model.predict(m.params)[:, None] * X).T @ (sc_ * m.model.predict(m.params)[:, None] * X) if False else None
    # Poisson score_i = (y - mu) x_i ; use that directly
    mu_hat = m.model.predict(m.params)
    psi = (df["y_c"].to_numpy() - mu_hat)[:, None] * X
    meat = psi.T @ psi
    V0 = bread @ meat @ bread
    N = n
    V_N1 = V0 * N / (N - 1)
    print("  sandwich(no factor) se  =", np.sqrt(np.diag(V0)))
    print("  sandwich*N/(N-1) se     =", np.sqrt(np.diag(V_N1)))
    print(f"  -> Stata robust se        = {np.sqrt(np.diag(p_rob))}")
    print(f"  rel(no-factor, Stata)     = {rel(np.sqrt(np.diag(V0)), np.sqrt(np.diag(p_rob))):.3e}")
    print(f"  rel(*N/(N-1), Stata)      = {rel(np.sqrt(np.diag(V_N1)), np.sqrt(np.diag(p_rob))):.3e}")
    print(f"  rel(westata HC0, Stata)   = {rel(np.asarray(r0.bse), np.sqrt(np.diag(p_rob))):.3e}")

print()
print("=" * 78)
print("D. logit cluster: does Stata use only G/(G-1)?")
with S() as s:
    s.send(df)
    b, V, names = s.est("logit yb x1 x2")
    V_ml = V
    b2, V2, _ = s.est("logit yb x1 x2, vce(cluster grp)")
    G = len(np.unique(grp))
    N = n
    K = 3
    print(f"  G = {G}, N = {N}, K = {K}")
    print(f"  Stata cluster se = {np.sqrt(np.diag(V2))}")
    lg = ws.Logit.from_formula("yb ~ x1 + x2", df)
    rc = lg.fit(cov_type="cluster", cov_kwds={"groups": grp})
    print(f"  westata cluster se = {np.asarray(rc.bse)}")
    ratio = np.sqrt(np.diag(V2)) / np.asarray(rc.bse)
    print(f"  Stata/westata se ratio = {ratio}")
    print(f"  sqrt((G-1)/G)          = {np.sqrt((G-1)/G):.12f}")
    print(f"  sqrt(G/(G-1)*(N-1)/(N-K))^-1 = {1/np.sqrt(G/(G-1)*(N-1)/(N-K)):.12f}")
    print(f"  predicted ratio = sqrt((N-K)/(N-1)*(G-1)/G) = {np.sqrt((N-K)/(N-1)*(G-1)/G):.12f}")
    print(f"  westata e(chi2) = {rc.e.get('chi2')}  Stata e(chi2) = {s.e('chi2')}  Stata chi2type = {s.e('chi2type')}")
