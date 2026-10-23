"""AUDIT probe 2 — tobit: /sigma <-> var(e.y), df_r, predict stdf.

High-risk item from the task: "tobit 的 /sigma vs var(e.y) 换算". We check:
  (a) point estimate: is sigma = sqrt(var(e.y))?
  (b) SE: is se(sigma) the sigma-scale or the log-sigma-scale SE?  We arbitrate
      with an INDEPENDENT numerical Hessian in both parameterisations.
  (c) df_r: Stata's df_m counts only slopes?
  (d) predict stdf: sigma^2 vs rss/N.
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

auto = pd.read_stata("tests/data/auto.dta", convert_categoricals=False)
d = pd.DataFrame(dict(mpg=auto["mpg"].to_numpy(float),
                      weight=auto["weight"].to_numpy(float),
                      foreign=auto["foreign"].to_numpy(float)))
N = len(d)
names = ["weight", "foreign", "_cons"]
X = np.column_stack([d["weight"], d["foreign"], np.ones(N)])
y = d["mpg"].to_numpy()
lo = 17.0


def tobit_ll(theta, y, X, lo):
    """Stata's tobit ll, parameterised as (b, log sigma)."""
    b = theta[: X.shape[1]]
    ls = theta[-1]
    s = np.exp(ls)
    xb = X @ b
    unc = y > lo
    ll = 0.0
    ll += np.sum(-ls - 0.5 * np.log(2 * np.pi) - 0.5 * ((y[unc] - xb[unc]) / s) ** 2)
    ll += np.sum(sps.norm.logcdf(np.maximum((lo - xb[~unc]) / s, -1e3)))
    return ll


def num_hessian(f, theta, h=1e-4):
    k = theta.size
    H = np.zeros((k, k))
    f0 = f(theta)
    for i in range(k):
        for j in range(k):
            tp = theta.copy(); tp[i] += h; tp[j] += h
            tm = theta.copy(); tm[i] -= h; tm[j] -= h
            tpp = theta.copy(); tpp[i] += h; tpp[j] -= h
            tmm = theta.copy(); tmm[i] -= h; tmm[j] += h
            H[i, j] = (f(tp) - f(tpp) - f(tmm) + f(tm)) / (4 * h * h)
    return H


print("=" * 78)
print("### tobit mpg weight foreign, ll(17)")
with S() as s:
    s.send(d)
    boot = s.est("tobit mpg weight foreign, ll(17)")
    b_s, V_s, nm = boot
    print("  Stata e(b) names:", nm)
    print("  Stata b  =", b_s)
    print("  Stata se =", np.sqrt(np.diag(V_s)))
    var_stata = float(b_s[nm.index("var(e.mpg)")])
    print("  Stata var(e.mpg) =", var_stata, " (from e(b); not an e() scalar)")
    for k in ("N", "df_m", "df_r", "rank", "ll", "ll_0", "chi2", "p", "r2_p", "rmse"):
        print(f"     e({k}) = {s.e(k)}")

    tb = ws.Tobit.from_formula("mpg ~ weight + foreign", d, lower=17.0).fit()
    print("\n  westata params =", list(tb.param_names), np.asarray(tb.params))
    print("  westata bse    =", np.asarray(tb.bse))
    print("  westata e: ", {k: tb.e.get(k) for k in ("N", "df_m", "rank", "ll", "df_r")})
    print("  westata df_resid =", tb.df_resid)
    print("  westata cov_type =", tb.cov_type)

    # (a) sigma point estimate
    sig_s = np.sqrt(var_stata)
    sig_w = float(np.asarray(tb.params)[-1])
    print(f"\n  (a) sigma: Stata sqrt(var(e.y)) = {sig_s:.12f}   westata /sigma = {sig_w:.12f}"
          f"  rel={rel([sig_w], [sig_s]):.3e}")

    # (b) SE scale arbitration: independent numerical Hessian
    # align westata's params [Intercept, weight, foreign, /sigma] to X's column order
    pn = list(tb.param_names)
    pw = np.asarray(tb.params)
    b_hat = np.array([pw[pn.index("weight")], pw[pn.index("foreign")], pw[pn.index("Intercept")]])
    theta_hat = np.concatenate([b_hat, [np.log(sig_w)]])
    H = num_hessian(lambda t: tobit_ll(t, y, X, lo), theta_hat)
    V_log = np.linalg.inv(-H)                    # covariance in (b, log sigma)
    se_log = np.sqrt(np.diag(V_log))
    se_sig_indep = sig_w * se_log[-1]            # delta method -> sigma scale
    se_stata_sig = np.sqrt(np.diag(V_s))[-1] if nm[-1] != "var(e.mpg)" else None
    # Stata reports var(e.mpg) with its own SE
    se_var_stata = np.sqrt(np.diag(V_s))[-1]
    se_sigma_stata = se_var_stata / (2 * np.sqrt(var_stata))
    se_linsigma_stata = se_var_stata / (2 * var_stata)
    print(f"\n  (b) SE arbitration (independent central-difference Hessian in (b,ln sigma)):")
    print(f"      se(ln sigma) from my Hessian = {se_log[-1]:.10f}")
    print(f"      se(ln sigma) implied by Stata's var(e.y) SE = {se_linsigma_stata:.10f}")
    print(f"      rel = {rel([se_log[-1]], [se_linsigma_stata]):.3e}")
    print(f"      se(sigma)   from my Hessian = sigma*se(ln sigma) = {se_sig_indep:.10f}")
    print(f"      se(sigma)   from Stata's var(e.y) SE (delta)     = {se_sigma_stata:.10f}")
    print(f"      rel = {rel([se_sig_indep], [se_sigma_stata]):.3e}")
    print(f"      westata reports bse[/sigma] = {np.asarray(tb.bse)[-1]:.10f}")
    print(f"      -> equals se(ln sigma)?  rel = {rel([np.asarray(tb.bse)[-1]], [se_log[-1]]):.3e}")
    print(f"      -> equals se(sigma)?     rel = {rel([np.asarray(tb.bse)[-1]], [se_sig_indep]):.3e}")
    print(f"      ratio westata/Stata se(sigma) = {np.asarray(tb.bse)[-1]/se_sigma_stata:.9f}"
          f"   1/sigma = {1/sig_w:.9f}")
    # also check the slope block against my independent Hessian
    print(f"\n      slope se: mine {np.round(se_log[:3],10)}")
    print(f"                Stata {np.round(np.sqrt(np.diag(V_s))[:3],10)}"
          f"  rel={rel(se_log[:3], np.sqrt(np.diag(V_s))[:3]):.3e}")

    # (c) df_r
    print(f"\n  (c) df_r: Stata e(df_m)={s.e('df_m')} e(df_r)={s.e('df_r')}, N={s.e('N')}")
    print(f"      westata df_resid = {tb.df_resid}   params={len(tb.params)}")
    print(f"      N - df_m(stata) - 0 = {N - int(s.e('df_m'))}")

    # (d) predict stdf
    s.run("predict double xb_t, xb")
    s.run("predict double stdp_t, stdp")
    s.run("predict double stdf_t, stdf")
    xb_s = s.var("xb_t"); stdp_s = s.var("stdp_t"); stdf_s = s.var("stdf_t")
    print(f"\n  (d) predict stdf: Stata stdf^2-stdp^2 mean = {np.nanmean(stdf_s**2 - stdp_s**2):.10f}")
    print(f"      sigma^2 (Stata var(e.y))                = {var_stata:.10f}")
    print(f"      rss/N (westata fallback)                = ?")
    resid = tb.resid
    print(f"      westata rss/N = {np.sum(resid**2)/N:.10f}   (westata resid sum^2 = {np.sum(resid**2):.8f})")
    try:
        w_stdf = tb.predict(kind="stdf")
        w_stdp = tb.predict(kind="stdp")
        print(f"      westata stdf^2-stdp^2 mean = {np.nanmean(np.asarray(w_stdf)**2 - np.asarray(w_stdp)**2):.10f}")
        print(f"      rel(westata stdf, Stata stdf) = {rel(np.asarray(w_stdf), stdf_s):.4e}")
    except Exception as exc:
        print("      westata predict stdf failed:", type(exc).__name__, exc)
    print(f"      rel(westata stdp, Stata stdp) = "
          f"{rel(np.asarray(tb.predict(kind='stdp')), stdp_s):.4e}")
    # independent Stata stdp check
    Vb = V_s[:3, :3]
    stdp_indep = np.sqrt(np.einsum("ij,jk,ik->i", X, Vb, X))
    print(f"      rel(sqrt(x'V_stata x), Stata stdp) = {rel(stdp_indep, stdp_s):.4e}")
