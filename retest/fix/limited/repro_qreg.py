"""T4 verification: qreg ``vce(robust)`` at tau = .25/.5/.75, both DGPs.

Run:  .venv\\Scripts\\python.exe retest/fix/limited/repro_qreg.py

root cause (twice wrong before):
  1. the meat used ``sum psi_i^2 x x'``.  Stata's is ``tau(1-tau) X'X``.  The
     two coincide only at tau=.5 (where psi^2 == 0.25 for every i), which is
     why a median-only check could not see it;
  2. the bread does carry the per-observation density ``fi_i`` -- replacing the
     meat alone still left 15%-147% on heteroskedastic data.

Stata's own ``qreg.ado`` (VCE_fitted + GetVCE) is
``V = tau(1-tau) H (X'X) H`` with ``H = inv(sum fi_i x_i x_i')`` and
``fi_i = 2h / (xb_i(tau+h) - xb_i(tau-h))``, zeroed below sqrt(epsdouble).
"""

from __future__ import annotations

import sys

sys.path.insert(0, "retest/lib")

import numpy as np
import pandas as pd

import westata as ws
from stata_probe import Stata

n = 400
rng = np.random.default_rng(31)
x1 = rng.normal(size=n)
x2 = rng.normal(size=n)
hetero = pd.DataFrame({
    "x1": x1, "x2": x2,
    "y": 1.0 + 0.8 * x1 - 0.4 * x2 + rng.normal(size=n) * (1.0 + 0.4 * np.abs(x1)),
})

rng2 = np.random.default_rng(77)
h1 = rng2.normal(size=n)
h2 = rng2.normal(size=n)
homo = pd.DataFrame({
    "x1": h1, "x2": h2, "y": 1.0 + 0.8 * h1 - 0.4 * h2 + rng2.normal(size=n),
})

worst = {"b": 0.0, "se_iid": 0.0, "se_robust": 0.0}

print(f"{'dataset':9s} {'tau':5s} {'quantity':10s} {'westata':>14s} {'Stata':>14s} {'rel':>10s}")
print("-" * 78)

with Stata() as st:
    print(f"Stata {st.version}\n")
    for label, df in (("hetero", hetero), ("homo", homo)):
        st.send(df)
        for tau in (0.25, 0.5, 0.75):
            for cov_type, command in (
                ("iid", f"qreg y x1 x2, quantile({tau})"),
                ("robust", f"qreg y x1 x2, quantile({tau}) vce(robust)"),
            ):
                cap = st.estimate(command)
                pos = {name: i for i, name in enumerate(cap.names)}
                order = [pos["_cons"], pos["x1"], pos["x2"]]
                stata_b = np.asarray(cap.b, dtype=float)[order]
                stata_se = np.sqrt(np.diag(np.asarray(cap.V, dtype=float)))[order]

                res = ws.QuantileRegression.from_formula(
                    "y ~ x1 + x2", df, quantile=tau).fit(cov_type=cov_type)
                ours_b = np.asarray(res.params, dtype=float)[[0, 1, 2]]
                ours_se = np.asarray(res.bse, dtype=float)[[0, 1, 2]]

                b_rel = float(np.max(np.abs(ours_b - stata_b) / np.abs(stata_b)))
                se_rel = float(np.max(np.abs(ours_se - stata_se) / stata_se))
                worst["b"] = max(worst["b"], b_rel)
                worst[f"se_{cov_type}"] = max(worst[f"se_{cov_type}"], se_rel)
                print(f"{label:9s} {tau:<5} {'b':10s} {'':>14s} {'':>14s} {b_rel:10.2e}")
                for j, nm in enumerate(("_cons", "x1", "x2")):
                    print(f"{'':9s} {'':5s} {('se ' + nm + ' ' + cov_type)[:10]:10s} "
                          f"{ours_se[j]:14.9g} {stata_se[j]:14.9g} "
                          f"{abs(ours_se[j]-stata_se[j])/stata_se[j]:10.2e}")
                print(f"{'':9s} {'':5s} {'se (max)':10s} {'':>14s} {'':>14s} "
                      f"{se_rel:10.2e}")

print("\n" + "=" * 78)
print(f"worst coefficient b       rel = {worst['b']:.3e}")
print(f"worst se under vce(iid)   rel = {worst['se_iid']:.3e}")
print(f"worst se under vce(robust) rel = {worst['se_robust']:.3e}  (target <= 1e-6)")
