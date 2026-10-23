"""AUDIT probe 4f — qreg robust: what bread does Stata use?

At tau=.5, psi_i^2 == 0.25 identically, so the meat is EXACTLY 0.25*X'X for both
implementations. Any residual difference there must be the BREAD. We solve for
the bread Stata implies and compare it with candidate scalings of (X'X)^-1.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular

sys.path.insert(0, "retest/domains/verify")
sys.path.insert(0, "src")

from my_stata import S, rel

import westata as ws


def bread_from(V, M):
    """One B with V = B M B' (Cholesky-based; unique up to orthogonal factor)."""
    Lm = np.linalg.cholesky(M)
    Lv = np.linalg.cholesky(V)
    return Lv @ np.linalg.inv(Lm)


def run(tag, x1, x2, y, tau):
    df = pd.DataFrame(dict(x1=x1, x2=x2, y=y))
    n = len(df)
    X = np.column_stack([x1, x2, np.ones(n)])
    XtX = X.T @ X
    XtXi = np.linalg.inv(XtX)
    with S() as s:
        s.send(df)
        b, V, nm = s.est(f"qreg y x1 x2, quantile({tau}) vce(robust)")
        _b, Vi, _ = s.est(f"qreg y x1 x2, quantile({tau})")
        sp, bw = s.e("sparsity"), s.e("bwidth")
    ii = [nm.index(k) for k in ("x1", "x2", "_cons")]
    V = V[np.ix_(ii, ii)]
    M = tau * (1 - tau) * XtX
    B = bread_from(V, M)
    print(f"\n  --- {tag} tau={tau}  (Stata sparsity={sp:.6g}, bwidth={bw:.6g}) ---")
    # candidates for the bread (all symmetric, so B = scalar * XtXi * scale)
    cands = {
        "sparsity*(X'X)^-1": sp * XtXi,
        "(1/sparsity)*(X'X)^-1": (1 / sp) * XtXi,
        "(X'X)^-1": XtXi,
    }
    for lbl, C in cands.items():
        print(f"    {lbl:26s} rel(BCB' implied V, Stata V) = "
              f"{rel((C @ M @ C.T).reshape(-1), V.reshape(-1)):.4e}")
    # is B a scalar multiple of XtXi?
    ratio = B / XtXi
    print(f"    B = c*(X'X)^-1 with constant c? spread={np.ptp(ratio[np.isfinite(ratio)]):.4e}"
          f"  c_est={np.median(ratio[np.isfinite(ratio)]):.8f}")
    c = np.median(ratio[np.isfinite(ratio)])
    print(f"    implied c = {c:.10f}   = sqrt(n/(n-k))*sparsity? "
          f"{np.sqrt(n/(n-3))*sp:.10f}   = sparsity? {sp:.10f}")


rng = np.random.default_rng(4242)
n = 400
a1 = rng.normal(size=n); a2 = rng.normal(size=n)
sig = np.exp(0.2 + 0.4 * a1)
yh = 1.0 + 0.75 * a1 - 0.45 * a2 + sig * rng.normal(size=n)
rng = np.random.default_rng(99)
b1 = rng.normal(size=n); b2 = rng.normal(size=n)
yhom = 1.0 + 0.75 * b1 - 0.45 * b2 + rng.normal(size=n)
print("=" * 74)
print("### tau = .5 (meat is exactly tau(1-tau)X'X on BOTH sides)")
for tag, xx in (("hetero", (a1, a2, yh)), ("homo", (b1, b2, yhom))):
    run(tag, xx[0], xx[1], xx[2], 0.5)
print("\n### tau = .25 / .75 for contrast")
for tau in (0.25, 0.75):
    run("hetero", a1, a2, yh, tau)
