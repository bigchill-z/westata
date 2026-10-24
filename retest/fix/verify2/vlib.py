"""Shared helpers for the verify2 audit (no imports from retest/lib or domains)."""

from __future__ import annotations

import numpy as np

from mybriage import fmt, rel


def wb(res, name):
    try:
        return float(res.params[name])
    except Exception:
        return None


def wse(res, name):
    try:
        return float(res.bse[name])
    except Exception:
        return None


def we(res, key, default=None):
    try:
        v = res.e[key]
    except Exception:
        return default
    return v


def pairs_ok(E, res, pairs):
    """Drop pairs whose Stata or westata side is missing."""
    out = []
    for s, w in pairs:
        sv, wv = E.get(s), wb(res, w)
        if sv is None or wv is None:
            continue
        out.append((s, w))
    return out


def cmp_b(E, res, pairs):
    pairs = pairs_ok(E, res, pairs)
    a = [wb(res, w) for _, w in pairs]
    b = [E.get(s) for s, _ in pairs]
    return rel(a, b), len(pairs)


def cmp_se(E, res, pairs):
    pairs = pairs_ok(E, res, pairs)
    a = [wse(res, w) for _, w in pairs]
    b = [E.getse(s) for s, _ in pairs]
    return rel(a, b), len(pairs)


def cmp_mat(E, res, pairs):
    pairs = pairs_ok(E, res, pairs)
    si = [E.names.index(s) for s, _ in pairs]
    V = np.asarray(res.cov_params(), dtype=float)
    wi = [list(res.param_names).index(w) for _, w in pairs]
    W = V[np.ix_(wi, wi)]
    Sv = E.V[np.ix_(si, si)]
    return rel(W.reshape(-1), Sv.reshape(-1)), len(pairs)


def line(tag, b, se, tier_b=None, tier_se=None):
    tb = f"({tier_b})" if tier_b else ""
    ts = f"({tier_se})" if tier_se else ""
    b_txt = "n/a" if b is None or b != b else f"{b:.3e}"
    s_txt = "n/a" if se is None or se != se else f"{se:.3e}"
    print(f"  {tag:<58s} b={b_txt:<10s}{tb:<4s} se={s_txt:<10s}{ts}", flush=True)
    return b, se


def report(tag, b, se, bmax, semax):
    print(f"  {tag:<44s} b={fmt(b)} ({bmax})  se={fmt(se)} ({semax})", flush=True)
