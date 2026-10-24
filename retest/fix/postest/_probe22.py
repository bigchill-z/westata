import sys
sys.path.insert(0, "retest/lib")
import numpy as np, pandas as pd
from stata_probe import Stata
with Stata() as st:
    st.run("sysuse auto, clear")
    st.quiet("ologit rep78 weight mpg")
    b = st.matrix("e(b)").ravel(); print("names", st.matrix_names("e(b)"))
    eta = b[0]*st.var("weight") + b[1]*st.var("mpg")
    cuts = b[2:]
    def p(k):
        lo = -np.inf if k == 1 else cuts[k-2]
        hi = np.inf if k == len(cuts)+1 else cuts[k-1]
        return 1/(1+np.exp(-(hi-eta))) - 1/(1+np.exp(-(lo-eta)))
    y = st.var("rep78")
    print("y unique", np.unique(y))
    for opt in ("", " outcome(1)", " outcome(2)", " outcome(observed)"):
        try:
            st.run(f"drop o_pr{opt.split('(')[0].strip() if opt else ''}") if False else None
        except Exception: pass
        name = "o_pr" + (opt.split("(")[1].rstrip(")") if opt else "def")
        try:
            st.quiet(f"predict double {name}, pr{opt}")
            v = st.var(name)
            match = {f"P(y={k})": float(np.max(np.abs(v-p(k)))) for k in range(1, len(cuts)+2)}
            obs = np.array([p(int(yi))(i) if False else 0 for i, yi in enumerate(y[:1])])
            pobs = np.array([p(k)[i] for i, k in enumerate(y.astype(int))])
            match["P(y=observed)"] = float(np.max(np.abs(v - pobs)))
            best = min(match.items(), key=lambda kv: kv[1])
            print(f"  pr{opt or ' (default)'}: best={best[0]} diff={best[1]:.3e} all={ {k: f'{d:.1e}' for k,d in match.items()} }")
        except Exception as exc:
            print(f"  pr{opt or ' (default)'} -> {str(exc).strip()[:90]}")
