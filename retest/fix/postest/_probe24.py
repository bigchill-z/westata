import sys
sys.path.insert(0, "retest/lib")
import numpy as np, pandas as pd
from stata_probe import Stata

with Stata() as st:
    st.run("sysuse auto, clear")
    st.quiet("ologit rep78 weight mpg")
    b = st.matrix("e(b)").ravel()
    eta = b[0]*st.var("weight") + b[1]*st.var("mpg")
    cuts = b[2:]
    K = len(cuts) + 1
    def F(z): return 1/(1+np.exp(-np.clip(z-eta, -700, 700)))
    def pk(k):
        lo = np.full_like(eta, -np.inf) if k == 1 else np.full_like(eta, cuts[k-2])
        hi = np.full_like(eta, np.inf) if k == K else np.full_like(eta, cuts[k-1])
        return F(hi) - F(lo)
    y = st.var("rep78")
    keep = np.isfinite(y)
    yk = y[keep].astype(int)
    for opt, name in (("", "d"), (" outcome(1)", "o1"), (" outcome(3)", "o3"),
                      (" outcome(#1)", "n1"), (" outcome(#3)", "n3")):
        try:
            var = f"pr_{name}"
            st.quiet(f"predict double {var}, pr{opt}")
            v = st.var(var)[keep]
            groups = {f"P(y={k})": float(np.max(np.abs(v - pk(k)[keep]))) for k in range(1, K+1)}
            pobs = np.array([pk(k)[i] for i, k in enumerate(yk)])
            groups["P(y=observed)"] = float(np.max(np.abs(v - pobs)))
            best = min(groups.items(), key=lambda kv: kv[1])
            print(f"  pr{opt or ' (default)'}: best {best[0]} {best[1]:.2e} | " +
                  " ".join(f"{k}={d:.1e}" for k, d in groups.items()))
        except Exception as exc:
            print(f"  pr{opt or ' (default)'} -> {str(exc).strip()[:80]}")
    # oprobit
    st.quiet("oprobit rep78 weight mpg")
    b2 = st.matrix("e(b)").ravel()
    eta2 = b2[0]*st.var("weight") + b2[1]*st.var("mpg")
    cuts2 = b2[2:]
    from scipy import stats
    def G(z): return stats.norm.cdf(z-eta2)
    def qk(k):
        lo = np.full_like(eta2, -np.inf) if k == 1 else np.full_like(eta2, cuts2[k-2])
        hi = np.full_like(eta2, np.inf) if k == len(cuts2)+1 else np.full_like(eta2, cuts2[k-1])
        return G(hi) - G(lo)
    st.quiet("predict double q_pr, pr")
    v = st.var("q_pr")[keep]
    groups = {f"P(y={k})": float(np.max(np.abs(v - qk(k)[keep]))) for k in range(1, len(cuts2)+2)}
    print("  oprobit pr default: best", min(groups.items(), key=lambda kv: kv[1]),
          "| all", " ".join(f"{k}={d:.1e}" for k, d in groups.items()))
