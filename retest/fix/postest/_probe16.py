"""Tobit margins numeric path: westata vs Stata (both b/se and the gradient)."""
import sys
sys.path.insert(0, "retest/lib")
import numpy as np, pandas as pd
from stata_probe import Stata
import westata as ws

auto = pd.read_stata("tests/data/auto.dta", convert_categoricals=False)

def rel(a, e): return abs(a-e)/max(abs(e),1e-12)

with Stata() as st:
    st.run("sysuse auto, clear")
    st.quiet("tobit mpg weight foreign, ll(17)")
    b = st.matrix("e(b)").ravel(); V = st.matrix("e(V)")
    names = st.matrix_names("e(b)")
    for cmd in ("margins, dydx(weight) predict(e(17,.))",
                "margins, dydx(weight) predict(ystar(17,.))",
                "margins, dydx(weight) predict(xb)",
                "margins, dydx(foreign) predict(ystar(17,.))"):
        st.quiet(cmd)
        t = st.matrix("r(table)"); jac = np.asarray(st.matrix("r(Jacobian)")).ravel()
        print(f"{cmd}")
        print("   stata b=%.15g se=%.15g  sqrt(jac'Vjac)=%.15g" % (
            t[0,0], t[1,0], float(np.sqrt(jac @ V @ jac))))
        print("   jac:", np.array2string(jac, precision=6))

    tb = ws.Tobit.from_formula("mpg ~ weight + foreign", auto, lower=17).fit()
    print("westata tobit names", tb.param_names)
    for kind, lbl in (("e","e"), ("ystar","ystar"), ("xb","xb")):
        r = tb.margins(dydx="weight", predict_kind=kind)
        print(f"   ws dydx(weight) predict({lbl}): b=%.15g se=%.15g" % (r.estimate, r.std_err))
    rf = tb.margins(dydx="foreign", predict_kind="ystar")
    rf = rf[0] if isinstance(rf, list) else rf
    print("   ws dydx(foreign) predict(ystar): b=%.15g se=%.15g" % (rf.estimate, rf.std_err))
    Vw = tb.cov_params(); bw = np.asarray(tb.params, float)
    print("   ws b:", np.array2string(bw, precision=10), " stata b:", np.array2string(b, precision=10))
    print("   ws V diag:", np.array2string(np.diag(Vw), precision=8), " stata V diag:", np.array2string(np.diag(V), precision=8))
    # align order: stata [weight, foreign, _cons, var_e]; ws [Intercept, weight, foreign, /sigma]
    perm = [1, 2, 0, 3]
    print("   V rel:", float(np.max(np.abs(Vw[np.ix_(perm,perm)] - V)/np.maximum(np.abs(V),1e-12))))
