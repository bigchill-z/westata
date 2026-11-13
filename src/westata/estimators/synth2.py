"""``synth2``: Stata's synthetic-control wrapper, NON-GRAPHICS feature set.

``synth2`` (v2.1.0, Yan & Chen) is a WRAPPER around ``synth``: it contains no
optimizer of its own.  All estimation is delegated to :func:`westata.synth`
(``synth2.ado:157/269/417/516``).  What it adds is post-processing,
permutation inference and robustness analysis:

* fit statistics ``e(rmse)/e(r2)/e(att)`` (``synth2.ado:636-672, 753``)
* a 6-column covariate balance table ``e(bal)`` (``:174-183``)
* non-zero unit weights ``e(U_wt)`` and the V diagonal ``e(V_wt)``
* in-space placebo (fake treatment units) -> ``e(mspe)`` + pointwise
  two/right/left p-values ``e(pval)`` (``:409-459``, Mata ``:935-960``)
* in-time placebo (fake treatment times) -> re-estimated trajectories,
  NO inference (``:502-552``)
* leave-one-out robustness over positive-weight donors -> per-period
  min/max envelopes (``:261-299``)

Deliberately NOT ported (graphics, per the approved design):
``figure``/``nofigure``/``savegraph()``/``show()``/``graphlist``/``e(graph)``,
and the Stata ``frame()``/``symbol()`` result-dataset mechanism (all results
are exposed as attributes of :class:`Synth2Result` instead).

Evidence grade: E2 by quantity.  The underlying ``synth`` is itself E2
(rank-deficient QP), and placebo/LOO re-run it many times, so differences
accumulate.  The FORMULAS below are transcribed verbatim from readable ado
source; each quantity's measured gap is recorded in
``tests/conformance/registry.yml``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from westata.errors import ConsistencyError, UnsupportedError
from westata.estimators.synth import SynthResult, synth
from westata.results.serialisation import ResultSerialisation

__all__ = ["Synth2Result", "synth2"]

#: Stata stores an absent ``e()`` matrix as this 1x1 missing sentinel.
_STATA_MISSING = 8.98846567431158e307


# --------------------------------------------------------------------------- #
# Result object -- a frozen dataclass, NOT StataResults (no e(b)/e(V))
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Synth2Result(ResultSerialisation):
    """Outcome of :func:`synth2`.

    Field names mirror Stata's ``e()`` verbatim.  Note ``synth2`` calls
    ``ereturn clear`` (``synth2.ado:330``) and therefore does NOT expose
    ``synth``'s ``e(Y_synthetic)``/``e(V_matrix)`` etc. -- only its own
    surface below, plus ``synth_result`` for callers who need the raw fit.
    """

    # e() scalars (synth2.ado:350-358)
    N: int
    T: int
    T0: int
    T1: int
    K: int
    J: int
    rmse: float
    r2: float
    att: float

    # e() matrices
    bal: np.ndarray                     # K x 6
    U_wt: np.ndarray                    # (n_pos) x 1
    V_wt: np.ndarray                    # K x K diagonal
    mspe: np.ndarray | None = None      # (1+P) x 4, in-space placebo only
    pval: np.ndarray | None = None      # T1 x 4, in-space placebo only

    # trajectories
    Y_treated: np.ndarray | None = None
    Y_synthetic: np.ndarray | None = None
    time_index: np.ndarray | None = None

    # placebo
    placebo_probability: float | None = None
    placebo_removed: tuple[Any, ...] = ()
    mspe_names: tuple[str, ...] = ()
    time_placebo: dict[float, dict[str, np.ndarray]] = field(default_factory=dict)

    # leave-one-out
    loo_min: np.ndarray | None = None
    loo_max: np.ndarray | None = None
    loo_effect_min: np.ndarray | None = None
    loo_effect_max: np.ndarray | None = None
    loo_n_runs: int = 0
    loo_units: tuple[Any, ...] = ()

    # audit
    synth_result: SynthResult | None = None
    method: str = "regression"
    numerically_inactive: tuple[str, ...] = ()

    @property
    def gap(self) -> np.ndarray:
        """Treatment effect path ``Y_treated - Y_synthetic``."""
        if self.Y_treated is None or self.Y_synthetic is None:
            raise AttributeError("trajectories not available")
        return self.Y_treated - self.Y_synthetic

    def summary(self, style: str = "stata") -> str:
        """A compact Stata-style summary (no fabricated inference)."""
        lines = [
            "synth2 -- Synthetic Control with Robustness Tests",
            "-" * 60,
            f"{'Treated Unit':<28}{self.synth_result.treated_unit if self.synth_result else ''}",
            f"{'Treatment Period':<28}{self.synth_result.treatment_period if self.synth_result else ''}",
            f"{'Number of Control Units':<28}{self.J}",
            f"{'Number of Covariates':<28}{self.K}",
            f"{'Root Mean Squared Error':<28}{self.rmse:.7f}",
            f"{'R-squared':<28}{self.r2:.7f}",
            f"{'ATT (post-period mean effect)':<32}{self.att:.7f}",
        ]
        if self.mspe is not None:
            lines += ["-" * 60, "In-space placebo (MSPE ratio table):",
                      f"  {'Unit':<20}{'PreMSPE':>12}{'PostMSPE':>12}"
                      f"{'Post/Pre':>12}{'Fake/Treated':>14}"]
            names = self.mspe_names or tuple(str(i) for i in range(len(self.mspe)))
            for name, row in zip(names, self.mspe):
                lines.append(
                    f"  {str(name):<20}{row[0]:>12.4f}{row[1]:>12.4f}"
                    f"{row[2]:>12.4f}{row[3]:>14.4f}"
                )
            if self.placebo_probability is not None:
                lines.append(
                    f"  Probability of a ratio >= treated's: "
                    f"{self.placebo_probability:.4f}"
                )
        if self.loo_min is not None:
            lines += ["-" * 60,
                      f"Leave-one-out: {self.loo_n_runs} re-estimations "
                      "(positive-weight donors only)"]
        lines.append("-" * 60)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


# --------------------------------------------------------------------------- #
# placebo option parsing (synth2_placebo syntax: syn2.ado:378-390)
# --------------------------------------------------------------------------- #
def _parse_placebo(spec: str, trperiod: float, all_periods: list[float],
                   control_units: list[Any]) -> dict[str, Any]:
    """Parse a ``placebo(...)`` string into its sub-options.

    Accepted: ``unit`` / ``unit(.)`` / ``unit(1 2 3)``, ``period(1985)``,
    ``cut(2)``/``cutoff(2)``, and combinations.  ``show``/``nofigure`` are
    graphics-only and refused here (synth2.ado:476-479, 460).
    """

    spec = spec.strip()
    if not spec:
        raise ValueError("synth2: placebo string is empty")
    out: dict[str, Any] = {
        "unit": None, "period": None, "cutoff": None, "show": None,
    }
    for token in ("unit", "period", "cut", "cutoff", "show"):
        # match `name(...)` or bare `name`
        m = re.search(rf"\b{token}\s*\(([^)]*)\)", spec)
        bare = re.search(rf"(?<!\(){token}(?!\s*\()", spec)
        if m:
            raw = m.group(1).strip()
        elif bare and token in ("unit",):
            raw = "."           # bare `unit` -> `unit(.)` (synth2.ado:252)
        elif bare:
            raise ValueError(
                f"synth2: placebo({token}) requires a value, e.g. "
                f"placebo({token}(...))"
            )
        else:
            continue
        if token == "show":
            raise UnsupportedError(
                "synth2: placebo(show()) is graphics-only and out of scope"
            )
        if token in ("cut", "cutoff"):
            try:
                out["cutoff"] = float(raw)
            except ValueError as exc:
                raise ValueError(
                    f"synth2: placebo cutoff must be a number, got {raw!r}"
                ) from exc
            if out["cutoff"] < 1:
                raise ValueError("synth2: placebo cutoff must be >= 1")
        elif token == "unit":
            if raw == "." or raw == "":
                out["unit"] = list(control_units)   # all controls
            else:
                try:
                    out["unit"] = [float(x) for x in raw.split()]
                except ValueError as exc:
                    raise ValueError(
                        f"synth2: placebo unit list invalid: {raw!r}"
                    ) from exc
                unknown = [u for u in out["unit"] if u not in control_units]
                if unknown:
                    raise ValueError(
                        f"synth2: placebo unit(s) not in the donor pool: {unknown}"
                    )
        elif token == "period":
            try:
                vals = [float(x) for x in raw.split()]
            except ValueError as exc:
                raise ValueError(
                    f"synth2: placebo period list invalid: {raw!r}"
                ) from exc
            for v in vals:
                if v >= trperiod:
                    raise ValueError(
                        f"synth2: placebo period {v:g} must be < trperiod "
                        f"({trperiod:g}) (synth2.ado:389,505-510)"
                    )
                if v not in all_periods:
                    raise ValueError(
                        f"synth2: placebo period {v:g} not in the time variable"
                    )
            out["period"] = vals
    # Reject unknown keys so a typo cannot silently disable inference.
    known = {"unit", "period", "cut", "cutoff"}
    stripped = re.sub(r"\b(unit|period|cut|cutoff|show)\s*\([^)]*\)", "", spec)
    stripped = re.sub(r"\bunit\b", "", stripped).strip(" ,")
    leftovers = [w for w in re.split(r"[\s,]+", stripped) if w]
    if leftovers:
        raise ValueError(
            f"synth2: unknown placebo sub-option(s): {leftovers}"
        )
    if out["unit"] is None and out["period"] is None:
        raise ValueError(
            "synth2: placebo() must request unit and/or period, e.g. "
            "placebo('unit') or placebo('period(1985)')"
        )
    return out


# --------------------------------------------------------------------------- #
# Public estimator
# --------------------------------------------------------------------------- #
def synth2(
    data: Any,
    *,
    unit: str,
    time: str,
    depvar: str,
    predictors: Sequence[str],
    trunit: Any,
    trperiod: float,
    counit: Sequence[Any] | None = None,
    ctrlunit: Sequence[Any] | None = None,
    preperiod: Sequence[float] | None = None,
    postperiod: Sequence[float] | None = None,
    xperiod: Sequence[float] | None = None,
    mspeperiod: Sequence[float] | None = None,
    customV: Sequence[float] | None = None,
    nested: bool = False,
    allopt: bool = False,
    margin: float = 0.005,
    maxiter: int = 1000,
    sigf: int = 12,
    bound: float = 10,
    placebo: str | None = None,
    loo: bool = False,
    # Graphics-only options: accepted as keywords so the error is explicit.
    figure: bool = False,
    nofigure: bool = False,
    savegraph: str | None = None,
    symbol: int = 1,
    frame: str | None = None,
    show: int | None = None,
    **unsupported: Any,
) -> Synth2Result:
    """Fit Stata ``synth2``'s NON-GRAPHICS feature set.

    All optimisation is delegated to :func:`westata.synth`.  See the module
    docstring and ``docs/superpowers/specs/2026-11-11-synth2-design.md`` for
    the per-quantity evidence grading and the exact ado formulas.
    """

    # --- refuse graphics / frame / symbol (explicit, not silent) ----------
    # NOTE: compare with `is True` / explicit values.  `True == 1` in Python,
    # so a naive `value not in (False, None, 1)` check silently lets
    # `figure=True` through -- a real bug caught by the RED test.
    if figure is True or nofigure is True:
        raise UnsupportedError(
            "synth2: figure/nofigure are graphics-only and out of scope "
            "(see the approved design spec). Results are exposed as "
            "Synth2Result attributes instead."
        )
    for flag, value in (
        ("savegraph", savegraph), ("frame", frame), ("show", show),
    ):
        if value is not None:
            raise UnsupportedError(
                f"synth2: option '{flag}' is graphics/frame-only and out of "
                "scope (see the approved design spec)."
            )
    if symbol != 1:
        raise UnsupportedError(
            "synth2: symbol() is a Stata frame naming option and out of scope."
        )
    if unsupported:
        raise UnsupportedError(
            f"synth2: unsupported option(s): {sorted(unsupported)}"
        )

    if not isinstance(data, pd.DataFrame):
        raise TypeError("synth2: data must be a pandas DataFrame")
    for col in (unit, time, depvar):
        if col not in data.columns:
            raise ValueError(f"synth2: column '{col}' not found in data")

    # --- Stata's row-deletion semantics (synth2.ado:79-120) --------------
    frame_df = data.copy()
    trperiod = float(trperiod)
    all_periods = sorted(float(t) for t in frame_df[time].dropna().unique())
    if trperiod not in all_periods:
        raise ValueError(
            f"synth2: trperiod={trperiod:g} is not present in the time variable"
        )
    # Stata's ``levelsof`` returns ascending ids (synth.ado:53); a year-major
    # long panel's first-appearance order is different and would mis-align the
    # placebo mspe rows.  Sort explicitly, exactly as ws.synth does.
    all_units = sorted(frame_df[unit].dropna().unique())
    if trunit not in all_units:
        raise ValueError(f"synth2: trunit={trunit!r} is not present in '{unit}'")

    if ctrlunit is not None and counit is not None:
        raise ValueError(
            "synth2: ctrlunit() and counit() cannot be specified together "
            "(synth2.ado:39-48)"
        )
    effective_counit = counit if counit is not None else ctrlunit
    if effective_counit is not None:
        unknown = [u for u in effective_counit if u not in all_units]
        if unknown:
            raise ValueError(f"synth2: counit contains unknown units: {unknown}")
        if trunit in effective_counit:
            raise ValueError("synth2: counit must not contain trunit")
        keep_units = [trunit, *effective_counit]
        frame_df = frame_df[frame_df[unit].isin(keep_units)].copy()
    control_units = [u for u in all_units if u != trunit]
    if effective_counit is not None:
        control_units = list(effective_counit)

    time_pre = [t for t in all_periods if t < trperiod]
    time_post = [t for t in all_periods if t >= trperiod]
    if preperiod is not None:
        req = [float(t) for t in preperiod]
        bad = [t for t in req if t >= trperiod or t not in time_pre]
        if bad:
            raise ValueError(
                f"synth2: preperiod must be < trperiod and present in the "
                f"data; invalid: {bad}"
            )
        frame_df = frame_df[frame_df[time].isin(req + time_post)].copy()
        time_pre = req
    if postperiod is not None:
        req = [float(t) for t in postperiod]
        if not req or req[0] != trperiod:
            raise ValueError(
                "synth2: postperiod's first element must equal trperiod "
                "(synth2.ado:107-111)"
            )
        bad = [t for t in req if t < trperiod or t not in time_post]
        if bad:
            raise ValueError(
                f"synth2: postperiod must be >= trperiod and present in the "
                f"data; invalid: {bad}"
            )
        frame_df = frame_df[frame_df[time].isin(time_pre + req)].copy()
        time_post = req
    time_all = [*time_pre, *time_post]
    if not time_pre or not time_post:
        raise ConsistencyError(
            "synth2: row filtering left no pre- or post-treatment periods"
        )

    # --- parse placebo ----------------------------------------------------
    placebo_opts = (
        _parse_placebo(placebo, trperiod, all_periods, control_units)
        if placebo else None
    )

    # --- shared kwargs for every ws.synth call ----------------------------
    synth_kw = dict(
        unit=unit, time=time, depvar=depvar, predictors=list(predictors),
        trunit=trunit, trperiod=trperiod, counit=effective_counit,
        xperiod=xperiod, mspeperiod=mspeperiod, customV=customV,
        nested=nested, allopt=allopt, margin=margin, maxiter=maxiter,
        sigf=sigf, bound=bound,
    )

    # --- main fit (synth2.ado:157) ----------------------------------------
    main = synth(frame_df, **synth_kw)
    Y_treated = main.Y_treated.copy()
    Y_synthetic = main.Y_synthetic.copy()
    effect = (Y_treated - Y_synthetic).ravel()
    tidx = main.time_index
    pre = tidx < trperiod
    post = ~pre

    # --- fit statistics (synth2.ado:640-647, 753) ------------------------
    rmse = float(main.RMSPE)
    mspe_pre = rmse ** 2
    mspe_post = float(np.mean(effect[post] ** 2))
    denom = float(np.sum((Y_synthetic.ravel()[pre]
                          - Y_synthetic.ravel()[pre].mean()) ** 2))
    r2 = 1.0 - float(np.sum(effect[pre] ** 2)) / denom if denom > 0 else float("nan")
    att = float(np.mean(effect[post]))

    # --- balance table (synth2.ado:174-183) -------------------------------
    Xbal = np.asarray(main.X_balance, dtype=float)
    treated_col = Xbal[:, 0]
    synth_col = Xbal[:, 1]
    avg_col = _average_control_means(
        frame_df, unit, time, depvar, predictors, trunit, xperiod,
        time_pre, all_periods,
    )
    bal = np.column_stack([
        np.diag(np.asarray(main.V_matrix, dtype=float)),   # Vweight
        treated_col,                                      # Treated
        synth_col,                                        # Synthetic value
        (synth_col / treated_col - 1) * 100,               # BiasSyntheticControl
        avg_col,                                          # Average Control value
        (avg_col / treated_col - 1) * 100,                 # BiasAverageControl
    ])

    # --- U_wt / V_wt (synth2.ado:673-690, 163) ---------------------------
    W = np.asarray(main.W_weights[:, 1], dtype=float)
    W_labels = list(main.W_weights[:, 0])
    pos = [(w, lab) for w, lab in zip(W, W_labels) if w > 0]
    pos.sort(key=lambda t: -t[0])
    U_wt = np.asarray([[w] for w, _ in pos], dtype=float)
    V_wt = np.asarray(main.V_matrix, dtype=float).copy()

    N = int(len(frame_df))
    T, T0, T1 = len(time_all), len(time_pre), len(time_post)
    K = len(predictors)
    J = len(control_units) + 1        # e(J) counts ALL units (ado:355)

    mspe = None
    pval = None
    mspe_names: tuple[str, ...] = ()
    probability = None
    removed: tuple[Any, ...] = ()
    time_placebo: dict[float, dict[str, np.ndarray]] = {}

    # --- in-space placebo (synth2.ado:409-459, Mata :935-960) -----------
    if placebo_opts is not None and placebo_opts["unit"] is not None:
        mspe, pval, mspe_names, probability, removed = _in_space_placebo(
            frame_df, synth_kw, placebo_opts, main,
            Y_treated, Y_synthetic, effect, tidx, pre, post, trperiod,
            mspe_pre, mspe_post, trunit,
        )

    # --- in-time placebo (synth2.ado:502-552), NO inference --------------
    if placebo_opts is not None and placebo_opts["period"] is not None:
        for fake_t in placebo_opts["period"]:
            kw = dict(synth_kw)
            kw["trperiod"] = fake_t
            # Stata passes the SAME xperiod/mspeperiod (synth2.ado:516).
            r = synth(frame_df, **kw)
            time_placebo[fake_t] = {
                "Y_treated": r.Y_treated.copy(),
                "Y_synthetic": r.Y_synthetic.copy(),
                "gap": (r.Y_treated - r.Y_synthetic).copy(),
                "time_index": r.time_index.copy(),
                "RMSPE": float(r.RMSPE),
            }

    # --- leave-one-out (synth2.ado:261-299) -------------------------------
    loo_min = loo_max = loo_eff_min = loo_eff_max = None
    loo_n_runs = 0
    loo_units: tuple[Any, ...] = ()
    if loo:
        loo_min, loo_max, loo_eff_min, loo_eff_max, loo_n_runs, loo_units = \
            _leave_one_out(frame_df, synth_kw, main, U_wt, control_units)

    inactive = tuple(
        k for k, given in (
            ("sigf", sigf != 12), ("margin", margin != 0.005),
            ("bound", bound != 10),
        ) if given
    )

    return Synth2Result(
        N=N, T=T, T0=T0, T1=T1, K=K, J=J, rmse=rmse, r2=r2, att=att,
        bal=bal, U_wt=U_wt, V_wt=V_wt, mspe=mspe, pval=pval,
        Y_treated=Y_treated, Y_synthetic=Y_synthetic, time_index=tidx,
        placebo_probability=probability, placebo_removed=removed,
        mspe_names=mspe_names, time_placebo=time_placebo,
        loo_min=loo_min, loo_max=loo_max,
        loo_effect_min=loo_eff_min, loo_effect_max=loo_eff_max,
        loo_n_runs=loo_n_runs, loo_units=loo_units,
        synth_result=main, method=main.method,
        numerically_inactive=inactive,
    )


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _average_control_means(
    data: pd.DataFrame, unit: str, time: str, depvar: str,
    predictors: Sequence[str], trunit: Any,
    xperiod: Sequence[float] | None, time_pre: list[float],
    all_periods: list[float],
) -> np.ndarray:
    """``synth2_balance`` (synth2.ado:558-579): simple mean over CONTROL units.

    For each predictor, average over the same periods used for X (``xperiod``
    if given, else the pretreatment period), excluding the treated unit.
    Missing observations are ignored, matching Stata's ``sum``.
    """
    from westata.estimators.synth import _parse_predictors, _clean

    window = list(xperiod) if xperiod is not None else time_pre
    cols, windows, _ = _parse_predictors(list(predictors), window, all_periods)
    out = np.empty(len(cols), dtype=float)
    # Stata's synth2_balance (ado:574-575) uses EACH predictor's OWN period
    # window when it carries a `(period)` suffix; it is NOT restricted to the
    # global xperiod.  Take the mean from the FULL data filtered by that
    # predictor's window, excluding the treated unit.
    controls_all = data[data[unit] != trunit]
    for k, (col, win) in enumerate(zip(cols, windows)):
        vals = _clean(controls_all.loc[controls_all[time].isin(win), col]
                      .to_numpy(dtype=np.float64))
        finite = vals[np.isfinite(vals)]
        out[k] = float(finite.mean()) if finite.size else float("nan")
    return out


def _in_space_placebo(
    data: pd.DataFrame, synth_kw: dict, opts: dict, main: SynthResult,
    Y_treated: np.ndarray, Y_synthetic: np.ndarray, effect: np.ndarray,
    tidx: np.ndarray, pre: np.ndarray, post: np.ndarray, trperiod: float,
    mspe_pre: float, mspe_post: float, trunit: Any,
):
    """Re-run synth for every fake treated unit; build mspe + pointwise pval."""

    cutoff = opts["cutoff"]
    unit_col = synth_kw["unit"]
    time_col = synth_kw["time"]
    depvar = synth_kw["depvar"]

    # Treated row (synth2.ado:660).
    rows = [[mspe_pre, mspe_post, mspe_post / mspe_pre, 1.0]]
    cut_rows = [rows[0][:]]
    names: list[str] = [str(trunit)]
    removed: list[Any] = []
    # Effects matrix: treated first, then each selected fake unit's post effects
    # (synth2_placebo Mata :939-944).
    eff_cols = [effect[post]]

    for fake in opts["unit"]:
        kw = dict(synth_kw)
        kw["trunit"] = fake
        # NOTE: synth2.ado:253-256 does NOT forward customV to placebo runs.
        kw.pop("customV", None)
        try:
            r = synth(data, **kw)
        except Exception as exc:
            raise ConsistencyError(
                f"synth2: in-space placebo re-fit failed for fake unit "
                f"{fake!r}: {exc}"
            ) from exc
        eff = (r.Y_treated - r.Y_synthetic).ravel()
        fake_pre_mspe = float(r.RMSPE) ** 2
        fake_post_mspe = float(np.mean(eff[post] ** 2))
        ratio_pp = fake_post_mspe / fake_pre_mspe
        ratio_tc = fake_pre_mspe / mspe_pre
        row = [fake_pre_mspe, fake_post_mspe, ratio_pp, ratio_tc]
        rows.append(row)
        names.append(str(fake))
        keep = cutoff is None or ratio_tc <= cutoff
        if keep:
            cut_rows.append(row)
            eff_cols.append(eff[post])
        else:
            removed.append(fake)

    mspe = np.asarray(rows, dtype=float)
    mspe_cut = np.asarray(cut_rows, dtype=float)

    # Aggregate probability = mean(ratio_treated <= ratio_all), INCLUDING the
    # treated row (synth2.ado:441/443/448).
    prob = float(np.mean(mspe_cut[0, 2] <= mspe_cut[:, 2]))

    # Pointwise p-values (synth2.ado:947-949), >= comparison, INCLUDING the
    # treated effect itself in the denominator (:940).
    E = np.column_stack(eff_cols)       # T1 x (1 + n_selected)
    tr_eff = E[:, 0]
    two = np.mean(np.abs(tr_eff[:, None]) <= np.abs(E), axis=1)
    right = np.mean(tr_eff[:, None] <= E, axis=1)
    left = np.mean(tr_eff[:, None] >= E, axis=1)
    pval = np.column_stack([tr_eff, two, right, left])

    return mspe, pval, tuple(names), prob, tuple(removed)


def _leave_one_out(
    data: pd.DataFrame, synth_kw: dict, main: SynthResult,
    U_wt: np.ndarray, control_units: list[Any],
):
    """Re-run synth without each positive-weight donor; return min/max envelopes."""

    # Positive-weight donors, descending (synth2.ado:676-682).  U_wt carries
    # only weights, so recover the unit labels from W_weights by matching.
    W_labels = list(main.W_weights[:, 0])
    W_vals = np.asarray(main.W_weights[:, 1], dtype=float)
    donors = [
        (float(w), lab) for w, lab in zip(W_vals, W_labels) if w > 0
    ]
    donors.sort(key=lambda t: -t[0])

    all_donors = list(control_units)
    ys: list[np.ndarray] = []
    used: list[Any] = []
    for _, lab in donors:
        # Match the label back to the underlying unit code.
        code = None
        for u in all_donors:
            if str(u) == str(lab) or u == lab:
                code = u
                break
        if code is None:
            # unitnames may have replaced codes with labels; fall back to order.
            idx = len(used)
            code = all_donors[idx] if idx < len(all_donors) else lab
        remaining = [u for u in all_donors if u != code]
        kw = dict(synth_kw)
        kw["counit"] = remaining
        try:
            r = synth(data, **kw)
        except Exception as exc:
            raise ConsistencyError(
                f"synth2: leave-one-out re-fit failed excluding {code!r}: {exc}"
            ) from exc
        ys.append(r.Y_synthetic.ravel())
        used.append(code)

    if not ys:
        return None, None, None, None, 0, ()
    M = np.column_stack(ys)
    loo_min = M.min(axis=1).reshape(-1, 1)
    loo_max = M.max(axis=1).reshape(-1, 1)
    yt = main.Y_treated.ravel()
    eff = yt[:, None] - M
    return (
        loo_min, loo_max,
        eff.min(axis=1).reshape(-1, 1),
        eff.max(axis=1).reshape(-1, 1),
        len(ys), tuple(used),
    )
