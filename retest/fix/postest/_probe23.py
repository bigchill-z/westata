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
    def pk(k):
        with np.errstate(over="ignore"):
            lo = np.full_like(eta, -np.inf) if k == 1 else np.full_like(eta, cuts[k-2])
            hi = np.full_like(eta, np.inf) if k == K else np.full_like(eta, cuts[k-1])
            F = lambda z: 1/(1+np.exp(-np.clip(z-eta, -700, 700)))
            return F(hi) - F(lo)
    y = st.var("rep78")
    for opt, name in (("", "d"), (" outcome(1)", "o1"), (" outcome(3)", "o3"),
                      (" outcome(#1)", "n1"), (" outcome(#3)", "n3")):
        try:
            var = f"pr_{name}"
            st.quiet(f"predict double {var}, pr{opt}")
            v = st.var(var)
            groups = {f"P(y={k})": float(np.nanmax(np.abs(v - pk(k)))) for k in range(1, K+1)}
            pobs = np.array([pk(int(yi))[i] for i, yi in enumerate(y)])
            groups["P(y=observed)"] = float(np.nanmax(np.abs(v - pobs)))
            best = min(groups.items(), key=lambda kv: kv[1])
            print(f"  pr{opt or ' (default)'}: best {best[0]} {best[1]:.2e} | all " +
                  " ".join(f"{k}={d:.1e}" for k, d in groups.items()))
        except Exception as exc:
            print(f"  pr{opt or ' (default)'} -> {str(exc).strip()[:80]}")
