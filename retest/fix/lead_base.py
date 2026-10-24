"""Lead's independent before/after harness for the repair round.

Each check is self-contained: it starts its own Stata session, builds its own
data, and compares Stata's full-precision e(b)/e(V) with westata's current
source.  Nothing here imports a domain fix script.
"""
import sys
sys.path.insert(0, "retest/lib")
import numpy as np, pandas as pd
import westata as ws
from stata_probe import Stata

def rel(a, e):
    a = np.asarray(a, float).reshape(-1); e = np.asarray(e, float).reshape(-1)
    if a.size != e.size: return float("inf")
    return float(np.max(np.abs(a-e)/np.maximum(np.abs(e), 1e-12)))

def panel_data(seed=2026, n_id=30, T=6):
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(n_id), T); tt = np.tile(np.arange(T), n_id)
    x1 = rng.normal(size=n_id*T); x2 = rng.normal(size=n_id*T)
    u = np.repeat(rng.normal(size=n_id), T)
    y = 1 + 0.8*x1 - 0.5*x2 + u + rng.normal(scale=0.5, size=n_id*T)
    return pd.DataFrame({"id": ids, "t": tt, "x1": x1, "x2": x2, "y": y})
