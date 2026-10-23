"""AUDIT probe 15 — identify Stata's qreg vce(robust) formula on the upstream DGP.

Upstream's stated root cause is:
    westata meat = sum psi_i^2 x x'  ;  Stata meat = tau(1-tau) X'X
    "swapping it reproduces Stata to 9.0e-9 at tau=.25"
That attribution is testable. At tau=.5 psi_i^2 == 1/4 identically, so the two
candidate meats COINCIDE; if Stata's tau=.5 answer still disagrees on some
datasets, the meat story cannot be the whole (or even the right) explanation.

We test a catalogue of Stata-documented variants against the numbers.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "retest/domains/verify")
sys.path.insert(0, "src")

from my_stata import S, rel

import westata as ws


def candidates(X, y, beta, tau, sp, bw, bu, bl):
    n, k = X.shape
    resid = y - X @ beta
    psi = tau - (resid < 0).astype(float)
    XtX = X.T @ X
    XtXi = np.linalg.inv(XtX)
    f_i = (2.0 * bw) / (X @ (bu - bl))        # observation-specific density
    out = {}
    # A. classic sandwich with obs-specific bread, psi^2 meat
    J = (X.T * f_i) @ X
    M = (X.T * psi**2) @ X
    out["A J_obs^-1 psi^2 J_obs^-1"] = np.linalg.inv(J) @ M @ np.linalg.inv(J)
    # B. same but constant meat
    Mc = tau * (1 - tau) * XtX
    out["B J_obs^-1 tau(1-t)s2.. X'X J_obs^-1"] = np.linalg.inv(J) @ Mc @ np.linalg.inv(J)
    # C. bread-less: (1/(n-k)) sum (psi_i/f_i)^2 x x'   (Stata [R] qreg form)
    w = (psi / f_i) ** 2
    out["C (1/(n-k)) sum (psi/f)^2 xx'"] = (X.T * w) @ X / (n - k)
    out["C2 (1/n) sum (psi/f)^2 xx'"] = (X.T * w) @ X / n
    # D. scalar sparsity bread, psi^2 meat
    Js = sp * XtX
    out["D J_scalar^-1 psi^2 J_scalar^-1"] = np.linalg.inv(Js) @ M @ np.linalg.inv(Js)
    # E. tau(1-tau) * sparsity^2 * (X'X)^-1 /n   (iid form)
    out["E tau(1-t)s^2 (XtX)^-1 / n"] = tau * (1 - tau) * sp**2 * XtXi / n
    # F. Stata doc: V = (n/(n-k)) * (1/n) * s^2 * tau(1-tau) * (X'X/n)^-1 ... scaled
    out["F n/(n-k) * iid"] = (n / (n - k)) * tau * (1 - tau) * sp**2 * XtXi / n
    # G. bread-less with scalar density
    w2 = (psi / sp) ** 2
    out["G (1/(n-k)) sum (psi/s)^2 xx'"] = (X.T * w2) @ X / (n - k)
    # H. obs-specific density, bread = sum f_i x x' / n, meat sum psi^2 x x' / n
    out["H (1/n)-scaled sandwich"] = (
        np.linalg.inv(J / n) @ (M / n) @ np.linalg.inv(J / n) / n
    )
    return out


def run(tag, df, taus):
    print("=" * 76)
    print(f"### {tag}")
    for tau in taus:
        r = ws.QuantileRegression.from_formula("y ~ x1 + x2", df, quantile=tau)
        ri = r.fit(cov_type="iid")
        sp = r._sparsity
        X = r._X
        pn = list(ri.param_names)
        order = [pn.index(k) for k in ("x1", "x2", "Intercept")]
        beta = np.asarray(ri.params)[order]
        with S() as s:
            s.send(df)
            b, Vr, nm = s.est(f"qreg y x1 x2, quantile({tau}) vce(robust)")
            _b, Vi, _ = s.est(f"qreg y x1 x2, quantile({tau})")
            sps = s.e("sparsity")
            s.run(f"quietly qreg y x1 x2, quantile({tau})")
            sps = s.e("sparsity"); bws = s.e("bwidth")
        ii = [nm.index(k) for k in ("x1", "x2", "_cons")]
        Vr = Vr[np.ix_(ii, ii)]
        se_s = np.sqrt(np.diag(Vr))
        cands = candidates(X, df["y"].to_numpy(), beta, tau, sps, bws,
                           sp["beta_upper"], sp["beta_lower"])
        print(f"\n  tau={tau}  Stata se = {np.round(se_s, 10)}")
        for lbl, V in cands.items():
            se = np.sqrt(np.abs(np.diag(V)))
            print(f"    {lbl:38s} rel={rel(se, se_s):.3e}")


rng = np.random.default_rng(31)
n = 400
x1 = rng.normal(size=n); x2 = rng.normal(size=n)
y = 1.0 + 0.8 * x1 - 0.4 * x2 + rng.normal(size=n) * (1 + 0.4 * np.abs(x1))
run("UPSTREAM exact DGP (seed 31)", pd.DataFrame(dict(x1=x1, x2=x2, y=y)), (0.25, 0.5))

rng = np.random.default_rng(4242)
a1 = rng.normal(size=n); a2 = rng.normal(size=n)
sg = np.exp(0.2 + 0.4 * a1)
yh = 1.0 + 0.75 * a1 - 0.45 * a2 + sg * rng.normal(size=n)
run("heteroskedastic (seed 4242)", pd.DataFrame(dict(x1=a1, x2=a2, y=yh)), (0.25, 0.5))
