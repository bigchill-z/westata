"""AUDIT probe 5c — etregress: the exact sign relation.

Stata:  e(lambda) = b_ctrl (the control-function coefficient)
        e(sigma)  = sqrt(RSS/n + b_ctrl^2 * mean(c^2 + w c))   (c = generalized residual)
        e(rho)    = e(lambda)/e(sigma)           <- verified to 0.0
westata: e(rho) = ?
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "retest/domains/verify")
sys.path.insert(0, "src")

from my_stata import S, rel

import westata as ws
import statsmodels.api as sm


def make_dgp(seed, rho_t):
    rng = np.random.default_rng(seed)
    n = 800
    u1, u2 = rng.normal(size=n), rng.normal(size=n)
    idx = 0.7 * u1 + 0.5 * rng.normal(size=n)
    tr = (idx + 0.9 * u1 > 0).astype(float)
    denom = np.sqrt(1 + 0.81)
    eps = rho_t * (idx + 0.9 * u1) / denom + np.sqrt(1 - rho_t**2) * u2
    x = rng.normal(size=n)
    y = 1.0 + 0.6 * x + 1.4 * tr + eps
    z1 = idx
    z2 = rng.normal(size=n)
    return pd.DataFrame(dict(y=y, t=tr, x1=x, z1=z1, z2=z2))


for rho_t in (0.45, -0.55):
    for seed in (777, 1234):
        d = make_dgp(seed, rho_t)
        print("=" * 76)
        print(f"### etregress twostep  DGP rho_true={rho_t}  seed={seed}")
        with S() as s:
            s.send(d)
            b, V, nm = s.est("etregress y x1, treat(t = z1 z2) twostep")
            e_sig, e_lam, e_rho = s.e("sigma"), s.e("lambda"), s.e("rho")
            print("  Stata names:", nm)
            print("  Stata b :", np.round(b, 10))
            print(f"  e(sigma)={e_sig:.12f} e(lambda)={e_lam:.12f} e(rho)={e_rho:.12f}")
            print(f"  e(lambda)/e(sigma) = {e_lam/e_sig:.12f}"
                  f"   == e(rho)? {abs(e_rho - e_lam/e_sig):.2e}")

            # independent reconstruction of Stata's sigma
            Zw = np.column_stack([d["z1"], d["z2"], np.ones(len(d))])
            t = d["t"].to_numpy()
            pr = sm.Probit(t, Zw).fit(disp=0)
            g = Zw @ pr.params
            Phi = pr.model.predict(pr.params)
            phi = np.exp(-0.5 * g**2) / np.sqrt(2 * np.pi)
            gen = phi / Phi
            X = np.column_stack([d["x1"], t, np.ones(len(d)), gen])
            yy = d["y"].to_numpy()
            coef = np.linalg.lstsq(X, yy, rcond=None)[0]
            resid = yy - X @ coef
            b_ctrl = coef[-1]
            n = len(d)
            sig_rec = np.sqrt(resid @ resid / n + b_ctrl**2 * np.mean(gen**2 + g * gen))
            print(f"  reconstruct sigma = sqrt(RSS/n + b_ctrl^2 mean(c^2+wg)) = {sig_rec:.12f}"
                  f"  rel={rel([sig_rec],[e_sig]):.3e}")
            alt = np.sqrt(resid @ resid / n)
            print(f"  alt sqrt(RSS/n) = {alt:.12f}  rel={rel([alt],[e_sig]):.3e}")
            alt2 = np.sqrt(resid @ resid / (n - 4))
            print(f"  alt sqrt(RSS/(n-4)) = {alt2:.12f}  rel={rel([alt2],[e_sig]):.3e}")
            print(f"  Stata b_ctrl (lambda row) = {b[nm.index('lambda')]:.12f}")

        et = ws.Etregress.from_formula("y ~ x1 + treat(t ~ z1 + z2)", d).fit()
        pn = list(et.param_names)
        wl = np.asarray(et.params)[pn.index("lambda")]
        wse = float(np.asarray(et.bse)[pn.index("lambda")])
        print(f"\n  westata params {pn}")
        print(f"  westata b = {np.round(np.asarray(et.params),10)}")
        print(f"  westata b_ctrl = {wl:.12f}   Stata = {b[nm.index('lambda')]:.12f}"
              f"  rel={rel([wl],[b[nm.index('lambda')]]):.3e}")
        print(f"  westata e(sigma)={et.e.get('sigma')!r}  Stata={e_sig!r}"
              f"  rel={rel([et.e.get('sigma')],[e_sig]):.3e}")
        print(f"  westata e(rho)={et.e.get('rho')!r}  Stata={e_rho!r}")
        print(f"  westata b_ctrl/sigma  = {wl/et.e.get('sigma'):.12f}")
        print(f"  westata -b_ctrl/sigma = {-wl/et.e.get('sigma'):.12f}")
        print(f"  -> westata rho == -b_ctrl/sigma ? "
              f"{abs(et.e.get('rho') - (-wl/et.e.get('sigma'))) < 1e-9}")
        print(f"  -> westata rho == +b_ctrl/sigma ? "
              f"{abs(et.e.get('rho') - (wl/et.e.get('sigma'))) < 1e-9}")
        print(f"  SIGN VERDICT: Stata rho={e_rho:+.8f}   westata rho={et.e.get('rho'):+.8f}"
              f"   opposite sign? {e_rho*et.e.get('rho') < 0}")
