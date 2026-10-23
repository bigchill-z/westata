"""AUDIT probe 4d — what IS Stata's qreg robust VCE? Elementwise ratios."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "retest/domains/verify"); sys.path.insert(0, "src")
from my_stata import S, rel
import westata as ws

rng = np.random.default_rng(4242); n = 400
x1 = rng.normal(size=n); x2 = rng.normal(size=n)
sig = np.exp(0.2 + 0.4 * x1)
y = 1.0 + 0.75 * x1 - 0.45 * x2 + sig * rng.normal(size=n)
df = pd.DataFrame(dict(x1=x1, x2=x2, y=y))
X = np.column_stack([x1, x2, np.ones(n)])

for tau in (0.25, 0.5, 0.75):
    with S() as s:
        s.send(df)
        b, Vr, nm = s.est(f"qreg y x1 x2, quantile({tau}) vce(robust)")
        _b, Vi, _ = s.est(f"qreg y x1 x2, quantile({tau})")
        sp = s.e("sparsity"); bw = s.e("bwidth")
    ii = [nm.index(k) for k in ("x1","x2","_cons")]
    Vr = Vr[np.ix_(ii,ii)]; Vi = Vi[np.ix_(ii,ii)]
    print("="*72); print(f"tau={tau}  Stata sparsity={sp} bwidth={bw}")
    print("  Stata iid    se =", np.round(np.sqrt(np.diag(Vi)),10))
    print("  Stata robust se =", np.round(np.sqrt(np.diag(Vr)),10))
    print("  V_rob/V_iid elementwise:\n", np.array2string(Vr/Vi, precision=6))
    # candidate: V = (n/(n-k)) * Ji (sum psi^2 x x') Ji with Ji = sparsity*(X'X)^-1
    psi = tau - ((y - X @ b) < 0).astype(float)
    XtX_inv = np.linalg.inv(X.T @ X)
    Ji = sp * XtX_inv
    Mpsi = (X.T * psi**2) @ X
    for lbl, Vc in (("s^2 (X'X)^-1 M (X'X)^-1", Ji @ Mpsi @ Ji),
                    ("* n/(n-k)", Ji @ Mpsi @ Ji * n/(n-3)),
                    ("with obs-specific density", None)):
        if Vc is None: continue
        print(f"  {lbl:30s} rel={rel(np.sqrt(np.diag(Vc)), np.sqrt(np.diag(Vr))):.4e}"
              f"  se={np.round(np.sqrt(np.diag(Vc)),10)}")
    q = ws.QuantileRegression.from_formula("y ~ x1 + x2", df, quantile=tau)
    ri = q.fit(cov_type="iid")
    spw = q._sparsity; Xw = q._X
    order=[list(ri.param_names).index(k) for k in ("x1","x2","Intercept")]
    print("  westata sparsity =", spw["sparsity"], " iid se rel =",
          rel(np.asarray(ri.bse)[order], np.sqrt(np.diag(Vi))))
    print("  rel(Vr/Vi constant?)", np.ptp(Vr/Vi))
