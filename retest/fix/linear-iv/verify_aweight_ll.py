"""Verify every claim in the FIXES.md aweight ll/ll_0 paragraph (Lead audit aid)."""
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, "retest/lib")
sys.path.insert(0, "src")

import westata as ws  # noqa: E402
from stata_probe import Stata  # noqa: E402

warnings.simplefilter("ignore")

auto = pd.read_stata("tests/data/auto.dta")[["mpg", "weight", "foreign"]].copy()
auto["foreign"] = (auto["foreign"] == "Foreign").astype(float)
n = len(auto)
auto["w"] = np.array([((i + 1) % 4) + 1 for i in range(n)], dtype=float)
auto["gid"] = np.array([(i + 1) % 12 for i in range(n)], dtype=float)

CASES = [
    ("nonrobust", "regress mpg weight foreign", {}, None),
    ("robust", "regress mpg weight foreign, vce(robust)", {"cov_type": "HC1"}, None),
    ("cluster", "regress mpg weight foreign, vce(cluster gid)",
     {"cov_type": "cluster", "cov_kwds": {"groups": auto["gid"]}}, None),
    ("pweight", "regress mpg weight foreign [pweight=w]", {}, "pweight"),
    ("aweight", "regress mpg weight foreign [aweight=w]", {}, "aweight"),
    ("fweight", "regress mpg weight foreign [fweight=w]", {}, "fweight"),
    ("iweight", "regress mpg weight foreign [iweight=w]", {}, "iweight"),
]

with Stata() as st:
    st.send(auto)
    for label, cmd, kw, wt in CASES:
        kw = dict(kw)
        cap = st.estimate(cmd)
        res = (
            ws.OLS.from_formula("mpg ~ weight + foreign", auto).fit(**kw)
            if wt is None
            else ws.OLS.from_formula(
                "mpg ~ weight + foreign", auto, weights=auto["w"], weight_type=wt
            ).fit(**kw)
        )
        ll_s, l0_s = cap.scalars["ll"], cap.scalars["ll_0"]
        ll_w, l0_w = res.e["ll"], res.e["ll_0"]
        d_ll = ll_w - ll_s
        d_l0 = l0_w - l0_s
        ratio = np.exp(2 * d_ll / n)
        print(f"{label:10s} ll: stata={ll_s:.12f} westata={ll_w:.12f} "
              f"|diff|={abs(d_ll):.10f}")
        print(f"{'':10s} ll_0: |diff|={abs(d_l0):.10f}   "
              f"sigma2 ratio implied = {ratio:.13f}   "
              f"(ll-ll_0) delta = {(ll_w-l0_w)-(ll_s-l0_s):.3e}")
