"""T5/T6: heckman / etregress derived scalars.

Stata's two-step stores (heckman.ado:744-746, etregress.ado:491-492):

    sigma = sqrt(rss/m + b_lambda^2 * mean(lambda*(lambda + w)))
    rho   = b_lambda / sigma
    lambda (e(lambda)) = rho*sigma = b_lambda

where ``w`` is the probit index, ``m = e(N)`` of the *second-stage* regression
(selected rows for heckman, all rows for etregress), and ``rss`` its residual
sum of squares.  westata used ``sqrt(rss/(m-k))`` for sigma and, for etregress,
the wrong sign on rho.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "retest/lib")

import numpy as np
import pandas as pd
from scipy import stats

import westata as ws
from stata_probe import Stata

# ------------------------------------------------------------------- DGP 1
rng = np.random.default_rng(7)
n = 600
z1 = rng.normal(size=n)
z2 = rng.normal(size=n)
sx1 = rng.normal(size=n)
u = rng.normal(size=n)
v = 0.6 * u + np.sqrt(1 - 0.36) * rng.normal(size=n)
s = (0.5 + 0.7 * z1 - 0.4 * z2 + u > 0).astype(float)
y = 1.0 + 0.9 * sx1 + v
dgp1 = pd.DataFrame({"x1": sx1, "z1": z1, "z2": z2, "s": s, "y": y})
dgp1["y_sel"] = np.where(s == 1, y, np.nan)

# ------------------------------------------------------------------- DGP 2
rng2 = np.random.default_rng(2024)
n2 = 800
w1 = rng2.normal(size=n2)
w2 = rng2.normal(size=n2)
t1 = rng2.normal(size=n2)
u2 = rng2.normal(size=n2)
rho_true = -0.55
e2 = rho_true * u2 + np.sqrt(1 - rho_true**2) * rng2.normal(size=n2)
treat = (0.3 + 0.8 * w1 + 0.5 * w2 + u2 > 0).astype(float)
y2 = 2.0 + 1.4 * t1 + 1.1 * treat + e2
dgp2 = pd.DataFrame({"x1": t1, "z1": w1, "z2": w2, "t": treat, "y": y2})


def mills(index):
    return np.exp(stats.norm.logpdf(index) - stats.norm.logcdf(index))


worst = {"heck_sigma": 0.0, "heck_rho": 0.0}
print(f"{'case':34s} {'quantity':10s} {'westata':>16s} {'Stata':>16s} {'rel':>10s}")
print("-" * 92)

with Stata() as st:
    print(f"Stata {st.version}\n")

    # ============================================================== heckman
    for tag, frame, outcome in (("dgp1", dgp1, "y_sel"),):
        st.send(frame)
        cap = st.estimate(f"heckman {outcome} x1, select(s = z1 z2) twostep")
        res = ws.Heckman.from_formula("y ~ x1 + select(s ~ z1 + z2)", frame).fit()

        stata = {k: cap.scalars.get(k) for k in ("rho", "sigma", "lambda", "selambda")}
        print(f"heckman {tag}: Stata e(rho)={stata['rho']} e(sigma)={stata['sigma']} "
              f"e(lambda)={stata['lambda']}")
        print(f"{'':16s} westata e(rho)={res.e['rho']} e(sigma)={res.e['sigma']} "
              f"e(lambda)={res.e['lambda']}")

        sel = np.asarray(frame["s"], dtype=float) > 0.5
        m = int(sel.sum())
        probit = ws.Probit.from_formula("s ~ z1 + z2", frame).fit()
        gamma = np.asarray(probit.params, dtype=float)
        index = np.column_stack([np.ones(len(frame)), frame["z1"], frame["z2"]]) @ gamma
        lam = mills(index)
        X = np.column_stack([np.ones(len(frame)), frame["x1"]])
        aug = np.column_stack([X, lam])
        b, *_ = np.linalg.lstsq(aug[sel], np.asarray(frame[outcome], dtype=float)[sel],
                                rcond=None)
        r = np.asarray(frame[outcome], dtype=float)[sel] - aug[sel] @ b
        rss = float(r @ r)
        delta = float(np.mean((lam * (lam + index))[sel]))
        sigma_chk = float(np.sqrt(rss / m + b[-1] ** 2 * delta))
        print(f"{'':16s} reconstruction: sqrt(rss/m + b_lam^2 * mean(lam(lam+w))) = "
              f"{sigma_chk:.12g}")
        print(f"{'':16s}   vs sqrt(rss/(m-3)) = {np.sqrt(rss/(m-3)):.12g}  <-- old westata")
        for key, ours, expected in (("e(sigma)", res.e["sigma"], stata["sigma"]),
                                    ("e(rho)", res.e["rho"], stata["rho"])):
            rel = abs(ours - expected) / abs(expected)
            label = key[2:-1]
            worst[f"heck_{label}"] = max(worst[f"heck_{label}"], rel)
            print(f"{'':16s} {key:10s} {ours:16.10g} {expected:16.10g} {rel:10.2e}")
        print()

    # ============================================================ etregress
    for tag, frame in (("dgp1", dgp1), ("dgp2", dgp2)):
        st.send(frame)
        cap = st.estimate("etregress y x1, treat(s = z1 z2) twostep") if tag == "dgp1" \
            else st.estimate("etregress y x1, treat(t = z1 z2) twostep")
        treat_name = "s" if tag == "dgp1" else "t"
        res = ws.Etregress.from_formula(
            f"y ~ x1 + treat({treat_name} ~ z1 + z2)", frame).fit()
        stata = {k: cap.scalars.get(k) for k in ("rho", "sigma", "lambda")}
        print(f"etregress {tag}: Stata e(rho)={stata['rho']} e(sigma)={stata['sigma']} "
              f"e(lambda)={stata['lambda']}")
        print(f"{'':16s} westata e(rho)={res.e['rho']} e(sigma)={res.e['sigma']}")

        tt = np.asarray(frame[treat_name], dtype=float)
        probit = ws.Probit.from_formula(f"{treat_name} ~ z1 + z2", frame).fit()
        gamma = np.asarray(probit.params, dtype=float)
        index = np.column_stack([np.ones(len(frame)), frame["z1"], frame["z2"]]) @ gamma
        lam = np.where(tt > 0.5, mills(index),
                       -np.exp(stats.norm.logpdf(index) - stats.norm.logsf(index)))
        X = np.column_stack([np.ones(len(frame)), frame["x1"], tt])
        aug = np.column_stack([X, lam])
        b, *_ = np.linalg.lstsq(aug, np.asarray(frame["y"], dtype=float), rcond=None)
        r = np.asarray(frame["y"], dtype=float) - aug @ b
        rss = float(r @ r)
        m = len(frame)
        delta = float(np.mean(lam * (lam + index)))
        sigma_chk = float(np.sqrt(rss / m + b[-1] ** 2 * delta))
        print(f"{'':16s} reconstruction sigma = {sigma_chk:.12g}  "
              f"(rho = {b[-1]/sigma_chk:.12g}, b_ctrl = {b[-1]:.12g})")
        for key, ours, expected in (("e(sigma)", res.e["sigma"], stata["sigma"]),
                                    ("e(rho)", res.e["rho"], stata["rho"])):
            rel = abs(ours - expected) / abs(expected)
            print(f"{'':16s} {key:10s} {ours:16.10g} {expected:16.10g} {rel:10.2e}")
        print()

print("=" * 92)
print(f"heckman e(sigma) worst rel = {worst['heck_sigma']:.3e}")
print(f"heckman e(rho)   worst rel = {worst['heck_rho']:.3e}")
