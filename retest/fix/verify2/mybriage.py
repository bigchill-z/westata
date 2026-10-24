"""Independent minimal Stata bridge for the verify2 audit.

Deliberately written from scratch: this module does NOT import
``retest/lib/stata_probe.py`` and does not import any domain ``run.py``.
Nothing here is shared with the fix groups, so a bug in their harness cannot
propagate into these numbers.

Only two jobs: (1) start one embedded StataNow session, (2) hand back
``e(b)``/``e(V)``/``e()``/``r()`` and the data as plain numpy.
"""

from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd

STATA_HOME = r"D:\Program Files\StataNow19"
SENTINEL = 8.988e307


class S:
    def __init__(self):
        self.dirty = False

    def start(self):
        sys.path.insert(0, STATA_HOME + r"\utilities")
        from pystata import config

        config.init("mp", splash=False)
        from pystata import stata as _st

        self.st = _st
        self.version = str(self.st.sfi.Macro.getGlobal("c(stata_version)"))
        return self

    # -- commands ------------------------------------------------------
    def run(self, cmd):
        self.st.run(cmd, echo=False)

    def q(self, cmd):
        self.run(cmd if cmd.startswith("quietly") else "quietly " + cmd)

    def text(self, cmd):
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            self.st.run(cmd, echo=False)
        return buf.getvalue()

    # -- ereturn -------------------------------------------------------
    def mats(self, name="e(b)"):
        try:
            return np.asarray(self.st.sfi.Matrix.get(name), dtype=float)
        except Exception:
            return None

    def mat_names(self, name="e(b)"):
        try:
            return [str(x) for x in self.st.sfi.Matrix.getColNames(name)]
        except Exception:
            return []

    def sc(self, name):
        try:
            v = self.st.sfi.Scalar.getValue(name)
        except Exception:
            return None
        if v is None:
            return None
        v = float(v)
        if math.isnan(v) or v >= SENTINEL * 0.5:
            return None
        return v

    def macro(self, name):
        try:
            return str(self.st.sfi.Macro.getGlobal(name))
        except Exception:
            return ""

    # -- data ----------------------------------------------------------
    def send(self, df):
        self.st.pdataframe_to_data(df, force=True)

    def getvar(self, name):
        a = np.asarray(self.st.sfi.Data.get(name), dtype=float).reshape(-1).copy()
        a[a >= SENTINEL * 0.5] = np.nan
        return a

    def nobs(self):
        return int(self.st.sfi.Data.getObsTotal())

    # -- estimation ----------------------------------------------------
    def est(self, cmd, quiet=True):
        self.run(("quietly " + cmd) if quiet else cmd)
        b = self.mats("e(b)")
        if b is None:
            raise RuntimeError("no e(b) from: " + cmd)
        b = b.reshape(-1)
        V = self.mats("e(V)")
        names = self.mat_names("e(b)")
        if V is None or V.size != b.size * b.size:
            V = np.full((b.size, b.size), np.nan)
        return E(cmd, b, V, names, self.ereturn())

    def ereturn(self):
        """All e() scalars and macros, keyed without the ``e()`` wrapper."""
        out = {}
        try:
            raw = self.st.get_ereturn()
        except Exception:
            return out
        for k, v in raw.items():
            key = k[2:-1] if (isinstance(k, str) and k.startswith("e(") and k.endswith(")")) else k
            if isinstance(v, str):
                out[key] = v
            elif isinstance(v, (int, float, np.floating, np.integer)):
                out[key] = float(v)
        return out

    def rvals(self):
        out = {}
        for n in (
            "chi2", "df", "p", "F", "df_r", "df_m", "r2", "b", "se", "estimate",
            "mean", "sd", "N", "sum_w",
        ):
            v = self.sc(f"r({n})")
            if v is not None:
                out[n] = v
        return out


class E:
    """One captured Stata estimation."""

    def __init__(self, cmd, b, V, names, eret):
        self.cmd = cmd
        self.b = b
        self.V = V
        self.names = names
        self.e = eret

    def se(self):
        return np.sqrt(np.diag(self.V))

    def idx(self, name):
        try:
            return self.names.index(name)
        except ValueError:
            return None

    def get(self, name):
        i = self.idx(name)
        return None if i is None else float(self.b[i])

    def getse(self, name):
        i = self.idx(name)
        return None if i is None else float(np.sqrt(self.V[i, i]))

    def sub(self, keep):
        """Sub-matrix for the coefficients whose names are in ``keep``."""
        ii = [self.names.index(k) for k in keep if k in self.names]
        return self.b[ii], self.V[np.ix_(ii, ii)], [self.names[i] for i in ii]


def rel(a, b):
    """Max relative difference, matching BRIEF §2.2."""
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12)))


def tier(x):
    if x != x:
        return "NA"
    for cut, lab in ((1e-9, "A"), (1e-6, "B"), (1e-3, "C"), (1e-2, "D")):
        if x <= cut:
            return lab
    return "E"


def fmt(x, d=3):
    if x is None or x != x:
        return "n/a"
    return f"{x:.{d}e}"


def frame(n, seed, cols=()):
    rng = np.random.default_rng(seed)
    out = {"id": np.arange(1, n + 1)}
    for c in cols:
        out[c] = rng.standard_normal(n)
    return pd.DataFrame(out)
