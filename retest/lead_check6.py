"""Captain spot-check #6: the nl pinv truncation claim (ill-conditioned design)."""
import sys; sys.path.insert(0, "retest/lib")
import numpy as np, pandas as pd, westata as ws
from stata_probe import Stata

auto = pd.read_stata("tests/data/auto.dta", convert_categoricals=False)
with Stata() as st:
    st.run("sysuse auto, clear")
    cap = st.estimate("nl (mpg = {b0} + {b1}*weight + {b2}*weight^2)", quiet=False)
    print("stata b:", cap.b)
    print("stata se:", np.sqrt(np.diag(cap.V)))
    m = ws.Nl.from_formula("mpg = {b0} + {b1}*weight + {b2}*weight**2", auto,
                           start={"b0": cap.b[0], "b1": cap.b[1], "b2": cap.b[2]})
    res = m.fit()
    print("west b:", np.asarray(res.params))
    print("west se:", np.asarray(res.bse))
    j = res.model_data
    print("extra attrs:", [a for a in dir(res) if 'jac' in a.lower() or 'model' in a.lower()][:6])
