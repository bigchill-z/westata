"""T8 status check: westata's CURRENT frontier SE against live Stata.

``src/westata/estimators/nonlinear.py`` belongs to the ``systems`` group; this
group only measures.  Two measurements are reported:

1. what the shipped ``Frontier`` lands on now, and
2. a step scan of the log-likelihood Hessian, which is what this group was
   originally handed (SE 4e-3 at the old ``step=1e-5`` default).

The second is kept because it is the evidence behind the recommendation in
``FIXES.md``; the first is what matters for the T8 verdict.
"""

from __future__ import annotations

import math
import sys

sys.path.insert(0, "retest/lib")

import numpy as np
import pandas as pd
from scipy import stats

import westata as ws
from stata_probe import Stata

# DGP used throughout the T8 discussion (n=400, seed=99).
rng = np.random.default_rng(99)
n = 400
x1 = rng.normal(size=n)
x2 = rng.normal(size=n)
v = rng.normal(size=n)
u = np.abs(rng.normal(scale=0.7, size=n))
frame = pd.DataFrame({"x1": x1, "x2": x2,
                      "y": 1.0 + 0.8 * x1 - 0.4 * x2 + v - u})
X = np.column_stack([np.ones(n), x1, x2])


def loglike(theta):
    """``sum [ln2 - ln sigma + ln phi(z) + ln Phi(-z lam)]``, z=(y-x'b)/sigma."""

    k = X.shape[1]
    s2v, s2u = math.exp(float(theta[k])), math.exp(float(theta[k + 1]))
    sigma = math.sqrt(s2v + s2u)
    lam = math.exp(0.5 * (float(theta[k + 1]) - float(theta[k])))
    z = (frame["y"].to_numpy(dtype=float) - X @ theta[:k]) / sigma
    return float(np.sum(math.log(2.0) - math.log(sigma) - 0.5 * z**2
                        - 0.5 * math.log(2.0 * math.pi)
                        + stats.norm.logcdf(-z * lam)))


def hessian_of_loglike(point, step):
    """Central-difference Hessian of the objective at `step`."""

    m = point.size
    H = np.zeros((m, m))
    h = step * np.maximum(np.abs(point), 1.0)
    f0 = loglike(point)
    for i in range(m):
        for j in range(i, m):
            ei = np.zeros(m)
            ej = np.zeros(m)
            ei[i] = h[i]
            ej[j] = h[j]
            if i == j:
                value = (loglike(point + ei) - 2.0 * f0 + loglike(point - ei)) / h[i] ** 2
            else:
                value = (loglike(point + ei + ej) - loglike(point + ei - ej)
                         - loglike(point - ei + ej) + loglike(point - ei - ej)) / (
                             4.0 * h[i] * h[j])
            H[i, j] = value
            H[j, i] = value
    return H


with Stata() as st:
    print(f"Stata {st.version}")
    st.send(frame)
    cap = st.estimate("frontier y x1 x2")
    # Stata's e(b) order is (x1, x2, _cons, /lnsig2v, /lnsig2u); westata reports
    # (Intercept, x1, x2, /lnsig2v, /lnsig2u).
    stata_b = np.asarray(cap.b, dtype=float)
    stata_se = np.sqrt(np.diag(np.asarray(cap.V, dtype=float)))
    order = [2, 0, 1, 3, 4]
    expected_b = stata_b[order]
    expected_se = stata_se[order]
    theta = np.array([expected_b[0], expected_b[1], expected_b[2],
                      expected_b[3], expected_b[4]])

    res = ws.Frontier.from_formula("y ~ x1 + x2", frame).fit()
    ours_b = np.asarray(res.params, dtype=float)
    ours_se = np.asarray(res.bse, dtype=float)

    print(f"\n--- 1. what the shipped Frontier does now")
    print(f"  e(ll) Stata {cap.scalars['ll']:.15g}   westata {res.e['ll']:.15g}")
    print(f"  {'parameter':12s} {'westata b':>18s} {'Stata b':>18s} {'rel':>10s}")
    for name, a, e in zip(res.param_names, ours_b, expected_b):
        print(f"  {name:12s} {a:18.11g} {e:18.11g} {abs(a - e) / abs(e):10.2e}")
    print(f"  {'parameter':12s} {'westata se':>18s} {'Stata se':>18s} {'rel':>10s}")
    worst_se = 0.0
    for name, a, e in zip(res.param_names, ours_se, expected_se):
        rel = abs(a - e) / abs(e)
        worst_se = max(worst_se, rel)
        print(f"  {name:12s} {a:18.11g} {e:18.11g} {rel:10.2e}")
    worst_b = float(np.max(np.abs(ours_b - expected_b) / np.abs(expected_b)))
    print(f"\n  worst parameter rel = {worst_b:.3e}")
    print(f"  worst SE rel        = {worst_se:.3e}   (T8 target <= 1e-6)")

    print(f"\n--- 2. log-likelihood Hessian step scan (the original defect)")
    print(f"  {'step':>8s} {'max rel SE vs Stata':>22s}")
    for step in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        H = hessian_of_loglike(theta, step)
        try:
            se = np.sqrt(np.diag(np.linalg.inv(-H)))
        except np.linalg.LinAlgError:
            print(f"  {step:8.0e} {'not positive definite':>22s}")
            continue
        print(f"  {step:8.0e} "
              f"{float(np.max(np.abs(se - expected_se) / expected_se)):22.3e}")
    print("\n  The scan is not monotone: large steps truncate, small steps cancel,")
    print("  and the old default 1e-5 sat in the trough (4.0e-4 - 4.0e-3).")
