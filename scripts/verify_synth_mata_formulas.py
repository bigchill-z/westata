"""P0 evidence: verify the identified ``normalize`` / ``regsval`` formulas.

This is NOT an implementation of ``synth``.  It is a read-only oracle check that
pins the two Stata Mata kernels (which live only in compiled
``lsynth_mata_subr.mlib``) to a closed-form Python equivalent, so the B-phase
estimator can be validated against it without guessing.

Established 2026-11-09 against live StataNow 19.5 (MP) on ``synth_smoking.dta``::

    normalize:  s[k]   = std( [Xtr[k] | Xco[k, 1..J]], ddof=1 )   # treated INCLUDED
                xtrmat = Xtr / s ;  xcomat = Xco / s[:, None]

    regsval:    Xdm = Xnorm - rowmean(Xnorm)                       # across units
                Zdm = Z    - rowmean(Z)                             # across units, per period
                B   = solve(Xdm @ Xdm.T, Xdm @ Zdm.T)               # K x T0
                s_k = sum_t B[k, t]**2
                V   = diag( s_k / sum(s_k) )

Two contract details that are easy to get wrong (both measured, not inferred):

* ``synth_smoking.dta`` stores numeric variables as **float32** and ``state`` as
  a labelled int32 category.  Read with ``convert_categoricals=False`` and cast
  numeric columns to float64; otherwise the per-unit mean accumulates in float32
  and V drifts by ~3e-7 while the formula itself is exact to ~8e-14.
* The predictor order is the order typed in ``synth``'s varlist.  Stata's
  ``e(V_matrix)`` diagonal follows that same order.

Run::

    .venv/Scripts/python.exe scripts/verify_synth_mata_formulas.py

Exit code 0 means every case reproduced Stata's full-precision value within
``1e-10`` (relative).  It needs no Stata licence: the reference numbers below
were captured from live Stata and are committed here.
"""

from __future__ import annotations

import sys

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    print("pandas is required", file=sys.stderr)
    raise

DATA = r"D:\myCodeProject\test_westata\data\synth_smoking.dta"
PREDICTORS = ["lnincome", "age15to24", "beer", "retprice"]
DEPVAR = "cigsale"
TREATED_UNIT = 3
TREATMENT_PERIOD = 1989
NUMERIC = [DEPVAR, *PREDICTORS, "year"]

# Full-precision captures from StataNow 19.5 (MP), `format(%21.17f)`.
# Key = (xperiod, mspeperiod) description; value = e(V_matrix) diagonal.
GROUND_TRUTH: list[tuple[str, list[int], list[int], np.ndarray]] = [
    (
        "default: xperiod=mspeperiod=1970-1988",
        list(range(1970, 1989)),
        list(range(1970, 1989)),
        np.array([
            0.02017586234387460, 0.07051094571756153,
            0.25958848171523474, 0.64972471022332912,
        ]),
    ),
    (
        "mspeperiod(1980(1)1988)",
        list(range(1970, 1989)),
        list(range(1980, 1989)),
        np.array([
            0.00817867291892540, 0.07983644228549265,
            0.22995854230012880, 0.68202634249545313,
        ]),
    ),
    (
        "xperiod(1980(1)1988)",
        list(range(1980, 1989)),
        list(range(1970, 1989)),
        np.array([
            0.07437375857344768, 0.02038158458137459,
            0.25173467560147089, 0.65350998124370685,
        ]),
    ),
]

# Stata's `xtrmat` (normalized X for the treated unit), default xperiod.
XTRMAT_REFERENCE = np.array([
    76.45067575342753, 24.39699081669801,
    5.43448494758249, 13.03343723782903,
])

RTOL = 1e-10


def _load() -> pd.DataFrame:
    df = pd.read_stata(DATA, convert_categoricals=False)
    for col in NUMERIC:
        df[col] = df[col].astype("float64")
    return df


def _units(df: pd.DataFrame) -> list[int]:
    """Treated unit first, then controls in ascending Stata integer code."""
    codes = sorted(int(u) for u in df["state"].unique())
    return [TREATED_UNIT, *(u for u in codes if u != TREATED_UNIT)]


def build_X(df: pd.DataFrame, units: list[int], xperiod: list[int]) -> np.ndarray:
    """K x N predictor means; missing periods are ignored (Stata `tabstat, s(mean)`)."""
    sub = df[df["year"].isin(xperiod)]
    X = np.empty((len(PREDICTORS), len(units)), dtype=np.float64)
    for k, col in enumerate(PREDICTORS):
        for j, unit in enumerate(units):
            vals = sub.loc[sub["state"] == unit, col].dropna().to_numpy(dtype=np.float64)
            X[k, j] = vals.mean()
    return X


def build_Z(df: pd.DataFrame, units: list[int], mspeperiod: list[int]) -> np.ndarray:
    """T0 x N pre-treatment outcome paths; rows = time, columns = units."""
    sub = df[df["year"].isin(mspeperiod)]
    Z = np.empty((len(mspeperiod), len(units)), dtype=np.float64)
    for i, period in enumerate(mspeperiod):
        for j, unit in enumerate(units):
            Z[i, j] = sub.loc[
                (sub["year"] == period) & (sub["state"] == unit), DEPVAR
            ].mean()
    return Z


def normalize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (X / s, s) where s[k] is the SAMPLE sd across all N units."""
    s = X.std(axis=1, ddof=1)
    s = np.where(s == 0, 1.0, s)  # constant-row guard; exact Stata behaviour TBD
    return X / s[:, None], s


def regsval(Xnorm: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """The regression-based diagonal V weights, as a length-K vector."""
    Xdm = Xnorm - Xnorm.mean(axis=1, keepdims=True)
    Zdm = Z - Z.mean(axis=1, keepdims=True)
    B = np.linalg.solve(Xdm @ Xdm.T, Xdm @ Zdm.T)
    squared = (B**2).sum(axis=1)
    return squared / squared.sum()


def main() -> int:
    df = _load()
    units = _units(df)
    print(f"dataset={DATA}  units={len(units)}  predictors={len(PREDICTORS)}")

    X = build_X(df, units, list(range(1970, TREATMENT_PERIOD)))
    Xnorm, s = normalize(X)
    xtr_err = float(np.max(np.abs(Xnorm[:, 0] - XTRMAT_REFERENCE)))
    print(f"normalize: max|xtrmat - Stata| = {xtr_err:.3e}  "
          f"{'PASS' if xtr_err < RTOL else 'FAIL'}")

    ok = xtr_err < RTOL
    for label, xperiod, mspeperiod, truth in GROUND_TRUTH:
        Xn, _ = normalize(build_X(df, units, xperiod))
        got = regsval(Xn, build_Z(df, units, mspeperiod))
        err = float(np.max(np.abs(got - truth)))
        passed = err < RTOL
        ok &= passed
        print(f"regsval [{label}]: max|V - Stata| = {err:.3e}  "
              f"{'PASS' if passed else 'FAIL'}")
        if not passed:
            print(f"  got   = {np.array2string(got, precision=17)}")
            print(f"  truth = {np.array2string(truth, precision=17)}")

    print("P0 FORMULA CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
