"""Independent minimal Stata bridge for the audit.

Deliberately does NOT import ``retest/lib/stata_probe.py`` nor any domain
``run.py``: the point of the audit is to re-derive every number with code that
shares nothing with the thing being audited.
"""

from __future__ import annotations

import io
import math
import sys
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

STATA_HOME = r"D:\Program Files\StataNow19"
MISS = 8.98846567431158e307


class S:
    def __init__(self) -> None:
        self.st = None

    def __enter__(self):
        util = STATA_HOME + r"\utilities"
        if util not in sys.path:
            sys.path.insert(0, util)
        from pystata import config

        config.init("mp", splash=False)
        from pystata import stata

        self.st = stata
        return self

    def __exit__(self, *a):
        self.st = None

    # -- run ---------------------------------------------------------------
    def run(self, cmd: str) -> None:
        self.st.run(cmd, echo=False)

    def out(self, cmd: str) -> str:
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.st.run(cmd, echo=False)
        return buf.getvalue()

    def quiet(self, cmd: str) -> None:
        self.run(cmd if cmd.startswith("quietly") else "quietly " + cmd)

    # -- scalars -----------------------------------------------------------
    def sc(self, name: str):
        try:
            v = self.st.sfi.Scalar.getValue(name)
        except Exception:
            return None
        if v is None:
            return None
        v = float(v)
        if math.isnan(v) or v >= MISS * 0.5:
            return None
        return v

    def e(self, key: str):
        return self.sc("e(" + key + ")")

    def r(self, key: str):
        return self.sc("r(" + key + ")")

    # -- matrices ----------------------------------------------------------
    def mat(self, name: str):
        try:
            return np.asarray(self.st.sfi.Matrix.get(name), dtype=float)
        except Exception:
            return None

    def cnames(self, name: str):
        try:
            return [str(x) for x in self.st.sfi.Matrix.getColNames(name)]
        except Exception:
            return []

    def est(self, cmd: str, *, quiet: bool = True):
        """Run an estimation command, return (b, V, names)."""
        self.run(("quietly " + cmd) if quiet else cmd)
        b = self.mat("e(b)")
        if b is None:
            raise RuntimeError(f"{cmd!r} left no e(b)")
        names = self.cnames("e(b)")
        V = self.mat("e(V)")
        b = b.reshape(-1)
        if len(names) != b.size:
            names = [f"b{i}" for i in range(b.size)]
        if V is not None and V.size != b.size * b.size:
            V = None
        return b, V, names

    # -- data --------------------------------------------------------------
    def send(self, df: pd.DataFrame) -> None:
        self.st.pdataframe_to_data(df, force=True)

    def var(self, name: str) -> np.ndarray:
        a = np.asarray(self.st.sfi.Data.get(name), dtype=float).reshape(-1).copy()
        a[a >= MISS * 0.5] = np.nan
        return a

    def nobs(self) -> int:
        return int(self.st.sfi.Data.getObsTotal())


def rel(a, b) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.size != b.size:
        return float("inf")
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12)))


def show(tag: str, a, b) -> None:
    print(f"  {tag:52s} rel={rel(a, b):.6e}", flush=True)
