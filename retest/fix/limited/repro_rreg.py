"""T7 verification: rreg SE law + the e(rss)/mse/rmse/r2/F block."""

from __future__ import annotations

import sys

sys.path.insert(0, "retest/lib")

import numpy as np
import pandas as pd

from stata_probe import Stata

# ``westata.estimators.quantile`` is imported directly, without executing
# ``westata/__init__.py``: this group's repro scripts must stay runnable while
# another group has a work-in-progress file elsewhere in the package.
import importlib
_pkg = __import__("types").ModuleType("westata")
_pkg.__path__ = ["src/westata"]
import sys as _sys
_sys.modules.setdefault("westata", _pkg)
_sub = __import__("types").ModuleType("westata.estimators")
_sub.__path__ = ["src/westata/estimators"]
_sys.modules.setdefault("westata.estimators", _sub)
ws = importlib.import_module("westata.estimators.quantile")

# ------------------------------------------------------------------ datasets
rng = np.random.default_rng(67)
n = 400
x1 = rng.normal(size=n)
x2 = rng.normal(size=n)
y = 1.0 + 2.0 * x1 - 0.5 * x2 + rng.normal(size=n)
contam = rng.random(size=n) < 0.05
outlier = pd.DataFrame({"y": np.where(contam, y + rng.normal(scale=30.0, size=n), y),
                        "x1": x1, "x2": x2})

rng2 = np.random.default_rng(31)
m = 400
q1 = rng2.normal(size=m)
q2 = rng2.normal(size=m)
hetero = pd.DataFrame({"x1": q1, "x2": q2,
                       "y": 1.0 + 0.8 * q1 - 0.4 * q2
                            + rng2.normal(size=m) * (1 + 0.4 * np.abs(q1))})

rng3 = np.random.default_rng(5)
c1 = rng3.normal(size=300)
c2 = rng3.normal(size=300)
clean = pd.DataFrame({"x1": c1, "x2": c2,
                      "y": 1 + 0.5 * c1 - 0.3 * c2 + rng3.normal(size=300)})

auto = pd.read_stata("tests/data/auto.dta", convert_categoricals=False)

CASES = [
    ("rreg y x1 x2 (5% contamination)", outlier, "y ~ x1 + x2", "rreg y x1 x2"),
    ("rreg y x1 x2 (heteroskedastic)", hetero, "y ~ x1 + x2", "rreg y x1 x2"),
    ("rreg y x1 x2 (clean normal)", clean, "y ~ x1 + x2", "rreg y x1 x2"),
    ("rreg mpg weight length (auto)", auto, "mpg ~ weight + length",
     "rreg mpg weight length"),
]

worst = {"b": 0.0, "se": 0.0, "rss": 0.0, "mse": 0.0, "rmse": 0.0,
         "r2": 0.0, "F": 0.0}

print(f"{'case':36s} {'quantity':8s} {'westata':>16s} {'Stata':>16s} {'rel':>10s}")
print("-" * 92)

with Stata() as st:
    print(f"Stata {st.version}\n")
    for label, frame, formula, command in CASES:
        st.send(frame)
        cap = st.estimate(command)
        names = list(cap.names)
        pos = {name: i for i, name in enumerate(names)}
        res = ws.RobustRegression.from_formula(formula, frame).fit()

        stata_order = [pos["_cons"] if nm == "Intercept" else pos[nm]
                       for nm in res.param_names]
        stata_b = np.asarray(cap.b, dtype=float)[stata_order]
        stata_se = np.sqrt(np.diag(np.asarray(cap.V, dtype=float)))[stata_order]
        ours_b = np.asarray(res.params, dtype=float)
        ours_se = np.asarray(res.bse, dtype=float)

        b_rel = float(np.max(np.abs(ours_b - stata_b) / np.abs(stata_b)))
        se_rel = float(np.max(np.abs(ours_se - stata_se) / np.abs(stata_se)))
        worst["b"] = max(worst["b"], b_rel)
        worst["se"] = max(worst["se"], se_rel)
        print(f"{command:36s} {'b':8s} {'':>16s} {'':>16s} {b_rel:10.2e}")
        for j, nm in enumerate(res.param_names):
            print(f"{'':36s} {('se ' + nm)[:8]:8s} {ours_se[j]:16.10g} "
                  f"{stata_se[j]:16.10g} "
                  f"{abs(ours_se[j] - stata_se[j]) / stata_se[j]:10.2e}")

        for key, scala in (("rss", "rss"), ("mse", None), ("rmse", "rmse"),
                           ("r2", "r2"), ("F", "F")):
            expected = (cap.scalars.get(scala) if scala
                        else cap.scalars["rss"] / cap.scalars["df_r"])
            actual = float(res.e[key])
            rel = abs(actual - expected) / abs(expected)
            worst[key] = max(worst[key], rel)
            print(f"{'':36s} {('e(' + key + ')'):8s} {actual:16.10g} "
                  f"{expected:16.10g} {rel:10.2e}")

        # The documented self-consistency constraint: rmse^2*(n-k) == rss.
        assert abs(res.e["rmse"] ** 2 * res.e["df_r"] - res.e["rss"]) <= 1e-12 * res.e["rss"]
        # And Stata's SE law, computed here from e(rmse) and the raw design.
        design = []
        for nm in res.param_names:
            design.append(frame[nm].to_numpy(dtype=float)
                          if nm in frame.columns else np.ones(len(frame)))
        X = np.column_stack(design)
        law = np.sqrt(res.e["rmse"] ** 2 * np.diag(np.linalg.pinv(X.T @ X)))
        print(f"{'':36s} {'SE law':8s} {'':>16s} {'':>16s} "
              f"{np.max(np.abs(law - stata_se) / stata_se):10.2e}")
        print()

print("=" * 92)
print(f"worst coefficient b  rel = {worst['b']:.3e}")
print(f"worst SE             rel = {worst['se']:.3e}   (was 1.5e-1 - 3.0e-1)")
for key in ("rss", "mse", "rmse", "r2", "F"):
    print(f"worst e({key:4s})        rel = {worst[key]:.3e}")
