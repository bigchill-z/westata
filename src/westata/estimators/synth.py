"""Synthetic control estimation (Stata ``synth``), B-grade / evidence E2.

``synth`` constructs a weighted combination of control units that matches the
treated unit on pre-intervention predictors, then uses it as the counterfactual.
This module implements the DEFAULT (regression-based V) path.  The published
Abadie-Diamond-Hainmueller algorithm and the two Stata Mata kernels that were
thought to be opaque were recovered exactly in P0 (2026-11-09) by black-box
identification against live StataNow 19.5; see
``scripts/verify_synth_mata_formulas.py`` and
``docs/superpowers/specs/2026-11-09-synth-b-design.md``.

What IS reproduced
------------------
* Data preparation: predictor means over ``xperiod`` ignoring missing values;
  outcome paths over ``mspeperiod`` / ``resultsperiod``.
* ``normalize()`` == sample SD (ddof=1) of ``[treated | controls]``, per predictor.
* ``regsval()`` == diag of the sum of SQUARED per-period multivariate
  cross-sectional OLS coefficients (with a per-period intercept), normalised.
* The inner constrained QP ``min 0.5 W'HW + c'W`` with ``sum(W)=1``, ``0<=W<=1``.
* All six Stata ``e()`` matrices, including the 3-decimal ``roundmat()`` of W.

What is NOT claimed (E2, by design)
-----------------------------------
``synthopt.plugin`` is a C++ interior-point QP solver with a 0.005 constraint
margin; ``H = Xco' V Xco`` has rank <= K << J, so the QP optimum is NOT unique
and the plugin stops off the optimal face.  scipy's SLSQP solves it to tighter
optimality: the reported W agrees with Stata to 3 decimals in every measured
case, but RMSPE (computed from the exact W) differs by up to 5.6e-4.  E3 /
bit-for-bit reproduction is therefore excluded; the measured bound is recorded
in ``tests/conformance/registry.yml``.

Deliberately refused (``UnsupportedError``): ``nested`` / ``allopt`` (Stata's
closed-source ``ml`` optimiser), ``synth2``, figures, ``keep()``, placebo and
any inferential quantity (Stata's synth has no ``e(V)``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from westata.errors import ConsistencyError, UnsupportedError

__all__ = ["SynthResult", "synth"]

#: Stata stores a missing numeric value as this sentinel, not as NaN.
_STATA_MISSING = 8.98846567431158e307

#: Defaults from ``synth.ado`` (version 0.0.8, 2026-04-29), NOT from the
#: outdated ``synth.sthlp`` (2014), which claims margin=0.05 / sigf=7.
_MARGIN = 0.005
_MAXITER = 1000
_SIGF = 12
_BOUND = 10

#: ``ml`` options that Stata accepts via ``*`` but that cannot change our
#: numbers (they switch/steer Stata's closed-source ml optimiser, which we do
#: not reproduce).  Recorded on the result so a caller is never misled.
NUMERICALLY_INACTIVE_ML = frozenset({
    "technique", "nrtolerance", "nonrtolerance", "difficult", "search", "trace",
})

#: QP-plugin settings whose Stata effect comes from the C++ interior-point
#: convergence behaviour; our SLSQP/Powell solvers have no counterpart.
NUMERICALLY_INACTIVE_QP = frozenset({"sigf", "margin", "bound"})


# --------------------------------------------------------------------------- #
# Result object -- a frozen dataclass, NOT StataResults (there is no e(b)/e(V))
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SynthResult:
    """Outcome of :func:`synth`.

    Field names mirror Stata's ``e()`` matrices verbatim.  Note ``V_matrix``
    (there is no plain ``e(V)``).  ``synth`` has no coefficient vector, no
    covariance matrix and no degrees of freedom; those statsmodels-style
    attributes are deliberately ABSENT rather than fabricated.
    """

    V_matrix: np.ndarray          # K x K diagonal predictor weights
    W_weights: np.ndarray         # J x 2: (unit number, unit weight)
    X_balance: np.ndarray         # K x 2: (Treated, Synthetic), RAW scale
    RMSPE: float                  # sqrt(SSE / T0), from the exact W
    Y_treated: np.ndarray         # T x 1
    Y_synthetic: np.ndarray       # T x 1, from the ROUNDED W
    unit_names: tuple[Any, ...]   # control unit identifiers, in W row order
    predictor_names: tuple[str, ...]
    time_index: np.ndarray        # T periods backing the Y_* rows
    treated_unit: Any
    treatment_period: float
    T0: int                       # number of pre-treatment (MSPE) periods
    #: The exact (unrounded) QP solution.  Kept for diagnostics; the six e()
    #: matrices use the rounded weights where Stata does.
    W_exact: np.ndarray
    #: Warnings Stata would print for predictor observations ignored when
    #: averaging (``missingchecker``).  Empty when nothing was ignored.
    missing_warnings: tuple[str, ...] = ()
    #: Which V path produced this result: "regression", "customV",
    #: "nested" or "allopt".
    method: str = "regression"
    #: Exposed optimiser settings actually used (diagnostics / auditability).
    optimizer_settings: dict[str, float | int] | None = None
    #: Which QP solver produced the weights: "SLSQP" or "interior-point".
    #: Recorded so a caller can audit which measured E2 bound applies.
    qp_solver: str = "SLSQP"
    #: Display labels for the control units, used by ``keep()`` for the
    #: ``_Co_Number`` column.  Stata's ``keep()`` stores VALUE LABELS (state
    #: names) there when the panel variable has a label set; ``W_weights``
    #: itself keeps the underlying codes so the numeric contract is unchanged.
    unit_labels: tuple[str, ...] = ()
    #: Underlying numeric unit codes (parallel to ``unit_labels``).  ``keep()``
    #: writes these as the numeric ``_Co_Number`` and attaches ``unit_labels``
    #: as Stata value labels.
    unit_codes: tuple[Any, ...] = ()
    #: Options that are ACCEPTED and RECORDED but do NOT change any number.
    #: Stata's ``sigf``/``margin``/``bound`` shift RMSPE by altering its C++
    #: interior-point plugin's convergence (measured: sigf(7) -2.1e-3,
    #: margin(0.05) +4.6e-8, bound(1) +5.3e-8); ``technique`` switches
    #: Stata's closed-source ml optimiser.  Our SLSQP/Powell solvers have no
    #: counterpart, so we refuse to fake an effect.  Empty by default.
    numerically_inactive: tuple[str, ...] = ()
    #: Stata's verbatim ``e(V_matrix)`` row/col labels, e.g.
    #: ``cigsale(1980(1)1988)`` / ``cigsale(1980&1988)``.  Verified against
    #: live Stata captures (synth_features.json); the ado's ``var(`xtimelab'``
    #: looks malformed but the assembled label is well-formed.
    stata_predictor_names: tuple[str, ...] = ()

    @property
    def gap(self) -> np.ndarray:
        """Treatment effect path ``Y_treated - Y_synthetic``."""

        return self.Y_treated - self.Y_synthetic

    @property
    def e(self) -> dict[str, np.ndarray | float]:
        """The six Stata ``e()`` matrices, keyed by their verbatim names."""

        return {
            "V_matrix": self.V_matrix,
            "W_weights": self.W_weights,
            "X_balance": self.X_balance,
            "RMSPE": self.RMSPE,
            "Y_treated": self.Y_treated,
            "Y_synthetic": self.Y_synthetic,
        }

    def __str__(self) -> str:
        """A compact Stata-style report (no fabricated inference)."""

        lines = [
            "Synthetic Control Method for Comparative Case Studies",
            "-" * 60,
            f"{'Predictors':<28}{', '.join(self.predictor_names)}",
            f"{'Treated Unit':<28}{self.treated_unit}",
            f"{'Treatment Period':<28}{self.treatment_period:g}",
            f"{'Pre-intervention periods':<28}{self.T0}",
            f"{'Root Mean Squared Prediction Error':<40}{self.RMSPE:.7f}",
            "-" * 60,
            "Predictor Balance:",
            f"  {'Predictor':<24}{'Treated':>14}{'Synthetic':>14}",
        ]
        for name, row in zip(self.predictor_names, self.X_balance):
            lines.append(f"  {str(name):<24}{row[0]:>14.5f}{row[1]:>14.5f}")
        lines += ["-" * 60, "Unit Weights (nonzero):",
                  f"  {'Unit':<24}{'W_Weight':>14}"]
        for unit, w in self.W_weights:
            if w > 0:
                lines.append(f"  {str(unit):<24}{w:>14.3f}")
        lines.append("-" * 60)
        return "\n".join(lines)

    def summary(self, style: str = "stata") -> str:
        """Render the report; ``style`` is accepted for statsmodels parity."""

        return str(self)


# --------------------------------------------------------------------------- #
# Solved Mata kernels (P0: verified <= 9.1e-15 on constructed data and
# <= 8.6e-14 on real Stata captures)
# --------------------------------------------------------------------------- #
def _clean(values: np.ndarray) -> np.ndarray:
    """Replace Stata's missing sentinel with NaN, then reject non-finite input."""

    out = np.asarray(values, dtype=np.float64).copy()
    out[np.abs(out) >= 1e300] = np.nan
    return out


def _normalize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``normalize()``: row scaling by the SAMPLE sd, treated unit included.

    ``X`` is K x N with column 0 = treated, columns 1..J = controls.  Mata's
    ``quadvariance(v)`` is the sample variance (ddof=1).  A constant row is a
    no-op (Stata would emit its missing sentinel; we refuse that upstream).
    """

    X = _clean(X)
    s = X.std(axis=1, ddof=1)
    s = np.where(s == 0.0, 1.0, s)
    return X / s[:, None], s


def _regsval(Xnorm: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """``regsval()``: normalised sum of squared per-period OLS coefficients.

    ``Xnorm`` K x N (normalized, treated in column 0); ``Z`` T0 x N with rows
    = pre-treatment periods in the SAME unit order.  Both are demeaned ACROSS
    UNITS (a per-period intercept).  Returns the length-K diagonal of V.

    Stata uses ``invsym(Sxx)`` for rank-deficient designs; that generalised
    inverse's sweep order was NOT reproduced.  We refuse with
    :class:`ConsistencyError` instead of inventing a number.
    """

    Xdm = Xnorm - Xnorm.mean(axis=1, keepdims=True)
    Zdm = Z - Z.mean(axis=1, keepdims=True)
    Sxx = Xdm @ Xdm.T
    K = Xnorm.shape[0]
    rank = int(np.linalg.matrix_rank(Sxx))
    if rank < K:
        raise ConsistencyError(
            f"synth: Sxx is rank-deficient (rank {rank} < K={K}); Stata's "
            "invsym() generalised inverse is not reproduced. Refusing to "
            "guess a V matrix."
        )
    B = np.linalg.solve(Sxx, Xdm @ Zdm.T)      # K x T0
    raw = (B**2).sum(axis=1)
    total = raw.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise ConsistencyError(
            "synth: regression-based V has zero/NaN total; all predictors "
            "are uninformative for the pre-period outcome."
        )
    return raw / total


def _custom_v(values: Sequence[float], K: int) -> np.ndarray:
    """``normweights()`` + ``getabs()``: normalise |b| to sum 1."""

    b = np.abs(np.asarray(values, dtype=np.float64))
    if b.size != K:
        raise ValueError(f"customV must have exactly {K} weights, got {b.size}")
    if not np.all(np.isfinite(b)):
        raise ValueError("customV contains non-finite values")
    total = b.sum()
    if total <= 0.0:
        raise ValueError("customV must contain at least one positive weight")
    return b / total


# --------------------------------------------------------------------------- #
# Inner constrained QP (``synth.ado:546-557`` / ``synth_ll.ado:13-27``)
# --------------------------------------------------------------------------- #
def _solve_w(H: np.ndarray, c: np.ndarray, J: int,
             maxiter: int = _MAXITER) -> np.ndarray:
    """Solve ``min 0.5 w'Hw + c'w`` s.t. ``sum(w)=1``, ``0<=w<=1``.

    SLSQP was chosen empirically: on all six captured fixtures it reproduces
    Stata's 3-decimal ``e(W_weights)`` exactly.  The problem is convex but
    rank-deficient, so the exact optimum is not unique; ``W_exact`` is kept
    for diagnostics while all reported matrices use the rounded weights.
    """

    x0 = np.full(J, 1.0 / J)
    # Stata's `maxiter()` is the C++ interior-point plugin's iteration cap;
    # live measurement shows 10 and 1000 give IDENTICAL results because the
    # plugin converges quickly.  SLSQP needs more iterations for a
    # rank-deficient H, so a small user maxiter must NOT silently truncate the
    # solve -- that would return a plausible-but-wrong W (measured: maxiter=10
    # gave RMSPE 16.5 instead of 19.7).  Give SLSQP a safe floor and treat
    # `maxiter` as an upper bound only; then VERIFY convergence explicitly.
    slsqp_maxiter = max(200, min(maxiter, 2000))
    result = minimize(
        lambda w: 0.5 * float(w @ H @ w) + float(c @ w),
        x0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * J,
        constraints=[{"type": "eq", "fun": lambda w: float(w.sum() - 1.0)}],
        options={"ftol": 1e-14, "maxiter": slsqp_maxiter},
    )
    w = np.asarray(result.x, dtype=np.float64)
    if not np.all(np.isfinite(w)):
        raise ConsistencyError("synth: QP returned non-finite unit weights")
    # SLSQP on a severely rank-deficient H can stop with a small equality
    # residual (e.g. 4e-4) while still being on the optimal face.  Clipping
    # and renormalising REPAIRS that numerical error; it is not a wrong
    # solution.  (A 1e-4 pre-normalisation rejection broke in-space placebo
    # re-fits, where Stata's plugin likewise returns a usable point.)
    w = np.clip(w, 0.0, 1.0)
    total = float(w.sum())
    if not np.isfinite(total) or total <= 0:
        raise ConsistencyError(
            f"synth: QP returned an infeasible point (sum(w)={total}); "
            "refusing to return a wrong W"
        )
    w = w / total
    return w


def _solve_w_interior(
    H: np.ndarray, c: np.ndarray, J: int, margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Primal-dual interior-point QP, stopped on the complementarity gap.

    Stata's ``synthopt.plugin`` is a Vanderbei/LOQO interior-point solver
    (``synth.sthlp:264-266``).  When ``H = Xco' V Xco`` is effectively rank 1
    (V concentrated on one predictor) the QP optimum face is very
    high-dimensional: Stata returns an EARLY-STOPPED interior iterate -- one
    dominant weight plus tiny (1e-3) weights spread across nearly all controls
    -- whereas SLSQP converges to a sparse vertex.  Reproducing Stata therefore
    requires reproducing the interior-point TRAJECTORY, not just the optimum.

    We run the standard log-barrier Newton system on
    ``min .5 w'Hw + c'w  s.t. sum(w)=1, w>0`` and stop when the complementarity
    gap ``sum(w * max(Hw+c+lambda, 0))`` falls below a threshold derived from
    Stata's public ``margin``.  Measured on the two degenerate captures this
    lands on the same plateau as Stata (dW 0.008 / 0.003 vs SLSQP's 0.176 /
    0.081), and the result is unchanged across two orders of magnitude of the
    gap threshold -- i.e. a stable trajectory feature, not a tuned coefficient.

    The barrier parameter is derived from ``margin`` rather than hardcoded:
    a smaller margin means Stata converges closer to the face, which maps to a
    smaller barrier mu.
    """

    mu = max(margin**3, 1e-12)          # 0.005^3 = 1.25e-7 (measured plateau)
    gap_target = max(margin**2, 1e-8)  # 0.005^2 = 2.5e-5 (stable 1e-6..1e-2)
    reg = 1e-12
    Hr = H + reg * np.eye(J)
    w = np.full(J, 1.0 / J)
    lam = 0.0
    ones = np.ones((J, 1))
    zero = np.zeros((1, 1))
    for _ in range(max(60, int(10 / max(margin, 1e-6)))):
        g = Hr @ w + c + lam - mu / w
        r = float(w.sum() - 1.0)
        D = mu / (w * w)
        K = np.block([[Hr + np.diag(D), ones], [ones.T, zero]])
        try:
            sol = np.linalg.solve(K, -np.concatenate([g, [r]]))
        except np.linalg.LinAlgError:  # pragma: no cover - pathological H
            break
        dw, dl = sol[:J], float(sol[J])
        alpha = 1.0
        for _ in range(50):
            if np.all(w + alpha * dw > 0):
                break
            alpha *= 0.5
        else:  # pragma: no cover
            break
        w = w + alpha * dw
        lam = lam + alpha * dl
        gap = float(np.sum(w * np.maximum(H @ w + c + lam, 0.0)))
        if gap < gap_target or np.max(np.abs(dw)) < 1e-13:
            break
    w = np.clip(w, 0.0, 1.0)
    # NOTE: deliberately NO renormalisation.  Stata's e(W_weights) sums to
    # 1.001 / 0.999 on these cases because roundmat() acts on the raw iterate.
    # Return both the exact iterate (for RMSPE) and the Stata-style rounded W.
    return w, np.round(w, 3)


def _roundmat(w: np.ndarray) -> np.ndarray:
    """``roundmat()``: 3 decimal places, applied BEFORE the four e() matrices."""

    return np.round(w, 3)


def _outer_loss(
    v_diag: np.ndarray, G: np.ndarray, d: np.ndarray,
    Zpre: np.ndarray, Zco: np.ndarray, J: int,
) -> float:
    """``synth_ll.ado:27``: unnormalised pre-period SSE for a candidate V.

    ``H = Xcon' V Xcon = sum_k v_k G[k]`` and ``c = -sum_k v_k d[k]`` where
    ``G[k] = Xcon[k] outer Xcon[k]`` and ``d[k] = Xtrn[k] * Xcon[k]``.  These
    are precomputed once by ``_nested_v`` so each objective evaluation is O(J^2)
    instead of rebuilding two matrices.
    """

    b = np.abs(np.asarray(v_diag, dtype=np.float64))
    total = b.sum()
    if not np.isfinite(total) or total <= 0:
        return float("inf")
    v = b / total
    H = np.tensordot(v, G, axes=(0, 0))
    c = -(v @ d)
    # The outer search only needs a moderately accurate inner QP; 1e-8 ftol
    # is ~5x cheaper than the 1e-14 used for the final reported fit and leaves
    # the selected V unchanged to the precision the outer optimiser can see.
    x0 = np.full(J, 1.0 / J)
    res = minimize(
        lambda w: 0.5 * float(w @ H @ w) + float(c @ w),
        x0, method="SLSQP", bounds=[(0.0, 1.0)] * J,
        constraints=[{"type": "eq", "fun": lambda w: float(w.sum() - 1.0)}],
        options={"ftol": 1e-8, "maxiter": 60},
    )
    w = np.clip(np.asarray(res.x, float), 0.0, 1.0)
    if w.sum() > 0:
        w = w / w.sum()
    err = Zpre[:, 0] - Zco @ w
    return float(err @ err)


def _nested_v(
    Xtrn: np.ndarray, Xcon: np.ndarray, Zpre: np.ndarray, J: int,
    initial: np.ndarray, *, allopt: bool, maxiter: int,
) -> np.ndarray:
    """Algorithm-equivalent outer V search for ``nested`` / ``allopt``.

    Stata uses its closed-source ``ml model d0`` (NR/DFP/BFGS) with two restarts
    and, for ``allopt``, three starting points (regression V, equal V, and
    ``ml search``).  We use the SAME objective (``_outer_loss``) and the same
    starting-point STRUCTURE, but scipy's optimiser instead of Stata's ``ml``.
    The convergence point therefore cannot be bit-identical to Stata; this is
    an E2 quantity by construction, not an E3 one.
    """

    K = Xtrn.shape[0]
    # Precompute the rank-1 pieces of H and c once (see _outer_loss).
    G = np.einsum("ki,kj->kij", Xcon, Xcon)          # K x J x J
    d = Xtrn.ravel()[:, None] * Xcon                  # K x J
    Zco = Zpre[:, 1:]
    starts = [np.asarray(initial, dtype=float)]
    if allopt:
        starts.append(np.full(K, 1.0 / K))       # equal V
        # Third Stata start is ``ml search``; emulate with a coarse random
        # perturbation of the regression V (a deterministic, checkable proxy).
        rng = np.random.default_rng(20261110)
        starts.append(initial * rng.uniform(0.5, 2.0, size=K))
    best_b, best_loss = None, float("inf")
    for start in starts:
        start = np.abs(np.asarray(start, dtype=float))
        if start.sum() <= 0:
            start = np.full(K, 1.0 / K)
        start = start / start.sum()
        # Powell needs far fewer objective evaluations than Nelder-Mead for
        # this tiny (K <= 5) problem, and each evaluation solves a JxJ QP.
        result = minimize(
            lambda b: _outer_loss(b, G, d, Zpre, Zco, J),
            start,
            method="Powell",
            options={"maxiter": min(maxiter, 20), "xtol": 1e-6, "ftol": 1e-8},
        )
        loss = _outer_loss(np.asarray(result.x, float), G, d, Zpre, Zco, J)
        if loss < best_loss:
            b = np.abs(np.asarray(result.x, dtype=float))
            if b.sum() > 0 and np.all(np.isfinite(b)):
                best_loss, best_b = loss, b / b.sum()
    if best_b is None:                            # pragma: no cover
        raise ConsistencyError("synth: nested V optimisation failed to converge")
    return best_b


# --------------------------------------------------------------------------- #
# ``var(period)`` predictor syntax (synth.ado:217-313)
# --------------------------------------------------------------------------- #
def _expand_numlist(token: str) -> list[float]:
    """Stata ``numlist`` subset used by ``var(period)``.

    Supports ``a``, ``a&b&...`` and ``a(b)c`` (``start(step)end``).
    ``synth.ado:255-264`` strips ``&``/parentheses then runs ``numlist, sort``;
    the parentheses form ``2001(1)2003`` means 2001, 2002, 2003.
    """

    out: list[float] = []
    for raw in token.replace("&", " ").split():
        part = raw.strip()
        if not part:
            continue
        if "(" in part:
            head, _, rest = part.partition("(")
            step_str, sep, end_str = rest.partition(")")
            try:
                start = float(head)
                step = float(step_str) if step_str.strip() else 1.0
                end = float(end_str) if sep and end_str.strip() else start
                if step <= 0 or end < start:
                    raise ValueError
                n = 0
                value = start
                while value <= end + 1e-9 and n < 10000:
                    out.append(value)
                    value += step
                    n += 1
                continue
            except ValueError as exc:
                raise ValueError(f"invalid numlist '{part}'") from exc
        try:
            out.append(float(part.rstrip(")")))
        except ValueError as exc:
            raise ValueError(f"invalid period '{part}'") from exc
    return sorted(set(out))


def _parse_predictors(
    predictors: Sequence[str], xperiod: list[float], all_periods: list[float],
) -> tuple[list[str], list[list[float]], tuple[str, ...]]:
    """Split each ``var(period)`` token into (column, periods, label).

    Returns the column names, the per-predictor period lists, and the Stata-style
    labels used for ``e(V_matrix)`` row/column names (``var(1980(1)1988`` etc.).
    """

    cols: list[str] = []
    windows: list[list[float]] = []
    labels: list[str] = []
    for spec in predictors:
        spec = str(spec)
        if "(" not in spec:
            if ")" in spec:
                raise ValueError(f"malformed predictor '{spec}'")
            cols.append(spec)
            windows.append(list(xperiod))
            labels.append(spec)
            continue
        if not spec.endswith(")"):
            raise ValueError(f"malformed predictor '{spec}' (unclosed parenthesis)")
        name, _, inside = spec[:-1].partition("(")
        name = name.strip()
        if not name:
            raise ValueError(f"malformed predictor '{spec}' (empty variable name)")
        periods = _expand_numlist(inside)
        if not periods:
            raise ValueError(f"malformed predictor '{spec}' (no periods)")
        unknown = [p for p in periods if p not in all_periods]
        if unknown:
            raise ValueError(
                f"predictor '{name}' specifies periods absent from the time "
                f"variable: {unknown}"
            )
        cols.append(name)
        windows.append(periods)
        # Stata's own e(V_matrix) row label is ``var(1980(1)1988`` -- note the
        # MISSING closing paren (synth.ado:305).  We keep that verbatim for the
        # Stata-facing labels and use a well-formed string for the public name.
        labels.append(f"{name}({inside.strip()})")
    return cols, windows, tuple(labels)


# --------------------------------------------------------------------------- #
# ``keep()`` results dataset (synth.ado:620-657) -- NOT a graph
# --------------------------------------------------------------------------- #
def _write_keep(
    path: str, result: "SynthResult", replace: bool,
) -> None:
    """Write Stata's SIDE-BY-SIDE 5-column results dataset.

    ``synth.ado:620-650`` uses ``svmat`` to append the weight columns (J rows)
    and the outcome columns (T rows) to the SAME dataset, starting at row 0.
    The result therefore has ``max(J, T)`` rows, NOT ``J + T``: the weight
    columns are populated on the first J rows and the outcome columns on the
    first T rows, with the shorter block leaving its columns missing at the
    end.  ``:650`` drops only rows where BOTH are missing.

    Verified against live StataNow 19.5: for J=38, T=31 the file has 38 rows,
    ``_Co_Number``/``_W_Weight`` non-null 38 times and ``_Y_*``/``_time`` 31
    times.  ``_Co_Number`` carries the unit value labels (state names) when a
    label set / ``unitnames()`` is in play; we preserve string labels as such.
    """

    from pathlib import Path

    target = Path(path)
    if target.exists() and not replace:
        raise FileExistsError(
            f"synth: keep() target '{path}' already exists; pass replace=True "
            "(Stata's `replace` option)"
        )
    J = result.W_weights.shape[0]
    T = result.Y_treated.shape[0]
    n = max(J, T)
    out = pd.DataFrame(
        {
            "_Co_Number": pd.Series([np.nan] * n, dtype="object"),
            "_W_Weight": np.full(n, np.nan),
            "_Y_treated": np.full(n, np.nan),
            "_Y_synthetic": np.full(n, np.nan),
            "_time": np.full(n, np.nan),
        }
    )
    # Both blocks start at row 0 (svmat appends COLUMNS, not rows).
    # Stata's _Co_Number is a NUMERIC column carrying VALUE LABELS, so we write
    # the numeric codes and attach the labels via to_stata(value_labels=...).
    # That keeps all J rows non-null (a string column would turn the trailing
    # rows into empty strings) and reproduces the labelled display exactly.
    raw_codes = (
        list(result.unit_codes[:J])
        if result.unit_codes and len(result.unit_codes) >= J
        else list(result.W_weights[:J, 0])
    )
    codes = pd.to_numeric(
        pd.Series(raw_codes, dtype="object"), errors="coerce"
    ).to_numpy(dtype=float)
    out.loc[: J - 1, "_Co_Number"] = codes
    out.loc[: J - 1, "_W_Weight"] = np.asarray(
        result.W_weights[:J, 1], dtype=float
    )
    out.loc[: T - 1, "_Y_treated"] = result.Y_treated.ravel()
    out.loc[: T - 1, "_Y_synthetic"] = result.Y_synthetic.ravel()
    out.loc[: T - 1, "_time"] = result.time_index
    out["_Co_Number"] = pd.to_numeric(out["_Co_Number"], errors="coerce")

    # Attach value labels only when they are MEANINGFUL (not just the string
    # form of the numeric code).  Stata does the same when no label set exists.
    value_labels: dict[float, str] | None = None
    if result.unit_labels and len(result.unit_labels) >= J:
        pairs = {
            float(code): str(lab)
            for code, lab in zip(codes, result.unit_labels[:J])
            if np.isfinite(code) and str(lab) != str(code)
        }
        if pairs:
            value_labels = {"_Co_Number": pairs}
    out.to_stata(target, write_index=False, value_labels=value_labels)


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
def _as_float64(df: Any, columns: Sequence[str]) -> Any:
    """Cast numeric columns to float64 BEFORE averaging (P0: float32 drifts V)."""

    out = df.copy()
    for col in columns:
        if col in out.columns and out[col].dtype.kind in "fiu":
            out[col] = out[col].astype("float64")
    return out


def _periods(data: Any, time: str, trperiod: float, which: str,
             supplied: Sequence[float] | None) -> list[float]:
    all_periods = sorted(float(t) for t in data[time].dropna().unique())
    if supplied is not None:
        requested = [float(t) for t in supplied]
        unknown = [t for t in requested if t not in all_periods]
        if unknown:
            raise ValueError(
                f"{which} contains periods absent from the time variable: {unknown}"
            )
        return requested
    if which in ("xperiod", "mspeperiod"):
        return [t for t in all_periods if t < trperiod]
    return all_periods                      # resultsperiod = ALL periods


def _build_matrices(
    data: Any, unit: str, time: str, depvar: str, predictor_cols: Sequence[str],
    predictor_windows: Sequence[Sequence[float]], predictor_labels: Sequence[str],
    units: list[Any], mspeperiod: list[float], resultsperiod: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return (X, Zpre, Zall, warnings).

    X is K x N (N = J+1, treated first); Zpre is T0 x N; Zall is T x N.
    Each predictor may have its OWN averaging window (``var(period)``).
    Missing predictor values are ignored in the mean, but an ALL-missing
    predictor for a unit, or any missing outcome in a required period, is an
    error (Stata ``matmissing()`` semantics).  Ignored observations produce the
    same ``-ignored for averaging`` warnings Stata's ``missingchecker`` prints.
    """

    K, N = len(predictor_cols), len(units)
    X = np.empty((K, N), dtype=np.float64)
    warnings: list[str] = []
    for k, (col, window, label) in enumerate(
        zip(predictor_cols, predictor_windows, predictor_labels)
    ):
        sub_x = data[data[time].isin(window)]
        for j, u in enumerate(units):
            raw = sub_x.loc[sub_x[unit] == u, col].to_numpy(dtype=np.float64)
            # Stata's missing sentinel must be REJECTED, not silently averaged
            # away: a single 8.99e307 cell would otherwise be dropped by the
            # nan-mean and produce a plausible-but-wrong V (rc=0 in Stata too).
            if np.any(np.abs(raw) >= 1e300):
                raise ConsistencyError(
                    f"synth: predictor '{col}' contains a Stata missing "
                    f"sentinel for unit {u}; refusing to average around it"
                )
            vals = _clean(raw)
            finite = vals[np.isfinite(vals)]
            if finite.size == 0:
                raise ConsistencyError(
                    f"synth: predictor '{col}' is missing for ALL specified "
                    f"period observations of unit {u}"
                )
            if finite.size < raw.size:
                suffix = f"({label.split('(', 1)[1]}" if "(" in label else ""
                warnings.append(
                    f"control units: missing obs for predictor {col}{suffix} "
                    f"-ignored for averaging"
                )
            X[k, j] = finite.mean()

    def outcome_matrix(periods: list[float]) -> np.ndarray:
        sub = data[data[time].isin(periods)]
        Z = np.empty((len(periods), N), dtype=np.float64)
        for i, t in enumerate(periods):
            for j, u in enumerate(units):
                raw = (
                    sub.loc[(sub[time] == t) & (sub[unit] == u), depvar]
                    .to_numpy(dtype=np.float64)
                )
                if np.any(np.abs(raw) >= 1e300):
                    raise ConsistencyError(
                        f"synth: outcome '{depvar}' contains a Stata missing "
                        f"sentinel for unit {u} in period {t}"
                    )
                vals = _clean(raw)
                if vals.size != 1 or not np.isfinite(vals[0]):
                    raise ConsistencyError(
                        f"synth: outcome '{depvar}' is missing/duplicated for "
                        f"unit {u} in period {t}"
                    )
                Z[i, j] = vals[0]
        return Z

    Zpre = outcome_matrix(mspeperiod)
    Zall = outcome_matrix(resultsperiod)
    return X, Zpre, Zall, warnings


# --------------------------------------------------------------------------- #
# Public estimator
# --------------------------------------------------------------------------- #
def synth(
    data: Any,
    *,
    unit: str,
    time: str,
    depvar: str,
    predictors: Sequence[str],
    trunit: Any,
    trperiod: float,
    counit: Sequence[Any] | None = None,
    xperiod: Sequence[float] | None = None,
    mspeperiod: Sequence[float] | None = None,
    resultsperiod: Sequence[float] | None = None,
    customV: Sequence[float] | None = None,
    unitnames: str | None = None,
    # Optimiser settings (synth.ado:35-38; the 2014 help file's 0.05/7 is stale)
    margin: float = _MARGIN,
    maxiter: int = _MAXITER,
    sigf: int = _SIGF,
    bound: float = _BOUND,
    # Outer V search (algorithm-equivalent, NOT bit-identical to Stata's ml)
    nested: bool = False,
    allopt: bool = False,
    # `ml maximize` options accepted by Stata via `*`/mlopts (synth.ado:41,434).
    # They are validated and RECORDED.  iterate() maps faithfully to the outer
    # optimiser's iteration budget; technique() is recorded but cannot change
    # our single Powell-based outer algorithm (documented, not faked).
    technique: str | None = None,
    iterate: int | None = None,
    nrtolerance: float | None = None,
    nonrtolerance: bool = False,
    difficult: bool = False,
    search: str | None = None,
    trace: bool = False,
    # Results dataset (NOT a graph)
    keep: str | None = None,
    replace: bool = False,
    # Deliberately refused: graphics and synth2's placebo machinery
    figure: bool = False,
    placebo: bool = False,
    **unsupported: Any,
) -> SynthResult:
    """Fit a synthetic control (Stata ``synth``), extended feature set.

    Implements Stata's default path plus ``unitnames()``, per-predictor
    ``var(period)`` windows, ``keep()``, the four optimiser settings and an
    algorithm-equivalent ``nested`` / ``allopt`` outer V search.  Graphics and
    ``synth2``'s placebo/LOO remain out of scope.

    Evidence grading is per quantity and MEASURED, never assumed: see
    ``tests/conformance/registry.yml``.  ``nested``/``allopt`` and RMSPE cannot
    be bit-identical to Stata (closed-source ``ml`` optimiser + rank-deficient
    QP + non-deterministic C++ plugin); they are E2 by construction.
    """

    if figure:
        raise UnsupportedError(
            "synth: figure/graph output is deliberately out of scope "
            "(GAP_REGISTER.md §5); use matplotlib on the result object."
        )
    if placebo:
        raise UnsupportedError(
            "synth: placebo/permutation tests are synth2's in-space/in-time "
            "machinery, not part of `synth`; out of scope."
        )
    if unsupported:
        raise UnsupportedError(
            f"synth: unsupported option(s): {sorted(unsupported)}"
        )

    # --- validate Stata's `ml` options (synth.ado:41,434) -----------------
    # Stata forwards these via `*` to `mlopts`, then `wrapml` passes them to
    # `ml model d0 ... maximize`.  We validate the same shapes and record them.
    ml_given = any(
        x is not None and x is not False
        for x in (technique, iterate, nrtolerance, nonrtolerance,
                  difficult, search, trace)
    )
    if ml_given and not (nested or allopt):
        raise ValueError(
            "synth: ml options (technique/iterate/nrtolerance/...) only apply "
            "to the nested/allopt outer search; pass nested=True or allopt=True"
        )
    if technique is not None and not isinstance(technique, str):
        raise TypeError("synth: technique must be a string, e.g. 'bfgs'")
    if iterate is not None and (not isinstance(iterate, int) or iterate <= 0):
        raise ValueError("synth: iterate must be a positive integer")
    if nrtolerance is not None and (
        not np.isfinite(nrtolerance) or nrtolerance <= 0
    ):
        raise ValueError("synth: nrtolerance must be a positive finite number")
    if search is not None and not isinstance(search, str):
        raise TypeError("synth: search must be a string, e.g. 'on'/'off'")
    ml_settings: dict[str, Any] = {}
    for key, value in (
        ("technique", technique), ("iterate", iterate),
        ("nrtolerance", nrtolerance), ("nonrtolerance", nonrtolerance),
        ("difficult", difficult), ("search", search), ("trace", trace),
    ):
        if value is not None and value is not False:
            ml_settings[key] = value

    # --- validate the exposed optimiser settings --------------------------
    if not np.isfinite(margin) or margin <= 0:
        raise ValueError("synth: margin must be a positive finite number")
    if not isinstance(maxiter, int) or maxiter <= 0:
        raise ValueError("synth: maxiter must be a positive integer")
    if not isinstance(sigf, int) or sigf <= 0:
        raise ValueError("synth: sigf must be a positive integer")
    if not np.isfinite(bound) or bound <= 0:
        raise ValueError("synth: bound must be a positive finite number")

    if not isinstance(data, pd.DataFrame):
        raise TypeError("synth: data must be a pandas DataFrame")

    trperiod = float(trperiod)
    all_periods = [float(t) for t in data[time].dropna().unique()]
    if trperiod not in all_periods:
        raise ValueError(
            f"synth: trperiod={trperiod} is not present in the time variable"
        )

    xp = _periods(data, time, trperiod, "xperiod", xperiod)
    mp = _periods(data, time, trperiod, "mspeperiod", mspeperiod)
    rp = _periods(data, time, trperiod, "resultsperiod", resultsperiod)
    if not xp or not mp or not rp:
        raise ConsistencyError("synth: one of the period lists is empty")

    # --- parse `var(period)` predictors BEFORE checking columns -----------
    pcols, pwindows, plabels = _parse_predictors(predictors, xp, all_periods)
    for col in (unit, time, depvar, *pcols):
        if col not in data.columns:
            raise ValueError(f"synth: column '{col}' not found in data")
    K = len(pcols)
    if K == 0:
        raise ValueError("synth: at least one predictor is required")

    # --- unitnames must be a STRING column, constant within unit ----------
    label_by_unit: dict[Any, str] | None = None
    if unitnames is not None:
        if unitnames not in data.columns:
            raise ValueError(
                f"synth: unitnames column '{unitnames}' not found in data"
            )
        if not pd.api.types.is_string_dtype(data[unitnames]):
            raise ValueError(
                f"synth: unitnames must name a STRING variable, "
                f"'{unitnames}' is {data[unitnames].dtype}"
            )
        label_by_unit = {}
        for u, grp in data.groupby(unit, sort=False):
            names = grp[unitnames].dropna().unique()
            if names.size != 1:
                raise ValueError(
                    f"synth: unitnames '{unitnames}' varies within unit {u}"
                )
            label_by_unit[u] = str(names[0])

    data = _as_float64(data, [unit, time, depvar, *pcols])
    # Stata's ``levelsof pvar`` (synth.ado:53) returns the panel ids in ASCENDING
    # order, and ``counit`` defaults to that list minus the treated unit
    # (:62-64).  A long panel stored year-by-year has a completely different
    # first-appearance order, so using ``unique()`` order would permute the
    # rows of e(W_weights) even when every number is correct.  Sort explicitly.
    all_units = sorted(data[unit].dropna().unique(), key=lambda x: (str(type(x)), x))
    if trunit not in all_units:
        raise ValueError(f"synth: trunit={trunit!r} is not present in '{unit}'")

    if counit is None:
        control_units = [u for u in all_units if u != trunit]
    else:
        control_units = sorted(
            (u for u in counit if u != trunit),
            key=lambda x: (str(type(x)), x),
        )
        unknown = [u for u in control_units if u not in all_units]
        if unknown:
            raise ValueError(f"synth: counit contains unknown units: {unknown}")
    J = len(control_units)
    if J < 2:
        raise ConsistencyError(
            f"synth: need at least 2 control units, got J={J}; Stata itself "
            "errors on J=1 (r(3200))."
        )

    units = [trunit, *control_units]
    X, Zpre, Zall, warnings = _build_matrices(
        data, unit, time, depvar, pcols, pwindows, plabels, units, mp, rp
    )

    # --- V: regression-based (solved), user supplied, or nested ------------
    Xnorm, _ = _normalize(X)
    Xtrn, Xcon = Xnorm[:, 0:1], Xnorm[:, 1:]
    if customV is not None:
        v_diag = _custom_v(customV, K)
        method = "customV"
    else:
        v_diag = _regsval(Xnorm, Zpre)
        method = "regression"
    if nested or allopt:
        # Stata's `iterate(#)` caps the ml optimiser; map it faithfully to the
        # outer search's iteration budget (synth.ado:441-446, wrapml:846-895).
        outer_maxiter = iterate if iterate is not None else maxiter
        v_diag = _nested_v(
            Xtrn, Xcon, Zpre, J, v_diag, allopt=allopt, maxiter=outer_maxiter
        )
        method = "allopt" if allopt else "nested"
    V = np.diag(v_diag)

    # --- inner QP on the NORMALISED X -------------------------------------
    H = Xcon.T @ V @ Xcon
    c = (-(Xtrn.T @ V @ Xcon)).ravel()
    # Adaptive dual solver.  Measured discriminant across 8 live-Stata cases:
    # Vmax > 0.98 means V is concentrated on ~one predictor, H is effectively
    # rank 1, and Stata's interior-point plugin returns an early-stopped dense
    # iterate that SLSQP cannot reproduce.  Otherwise SLSQP matches Stata
    # EXACTLY and must be kept.  The threshold is a property of the DATA (V),
    # not a per-case switch.
    vmax = float(np.max(v_diag))
    if vmax > 0.98:
        W_exact, W_rounded = _solve_w_interior(H, c, J, margin)
        qp_solver = "interior-point"
    else:
        W_exact = _solve_w(H, c, J, maxiter=maxiter)
        W_rounded = _roundmat(W_exact)
        qp_solver = "SLSQP"

    # --- results: RMSPE uses the EXACT W, everything else the ROUNDED W ---
    err = Zpre[:, 0] - Zpre[:, 1:] @ W_exact
    rmspe = float(np.sqrt(float(err @ err) / Zpre.shape[0]))

    Y_treated = Zall[:, 0:1].copy()
    Y_synthetic = (Zall[:, 1:] @ W_rounded).reshape(-1, 1)
    X_balance = np.column_stack([X[:, 0], X[:, 1:] @ W_rounded])
    # unitnames changes ONLY the W_weights row labels, never the numbers.
    w_labels: list[Any] = [
        label_by_unit[u] if label_by_unit is not None else u
        for u in control_units
    ]
    W_weights = np.column_stack(
        [np.asarray(w_labels, dtype=object), W_rounded]
    )
    # keep()'s _Co_Number shows VALUE LABELS.  Stata derives them from the
    # panel variable's label set even when unitnames() is absent (synth.ado:81
    # `local clab: value label pvar`), but that label set lives in the .dta
    # file, not in a DataFrame we receive.  So: honour unitnames() when given,
    # and otherwise fall back to the code as text (what Stata shows when no
    # label set exists).  Documented in USING_WITH_STATA.md.
    if label_by_unit is not None:
        keep_labels = tuple(label_by_unit[u] for u in control_units)
    else:
        keep_labels = tuple(str(u) for u in control_units)

    result = SynthResult(
        V_matrix=V,
        W_weights=W_weights,
        X_balance=X_balance,
        RMSPE=rmspe,
        Y_treated=Y_treated,
        Y_synthetic=Y_synthetic,
        unit_names=tuple(w_labels),
        predictor_names=plabels,
        time_index=np.asarray(rp, dtype=np.float64),
        treated_unit=trunit,
        treatment_period=trperiod,
        T0=Zpre.shape[0],
        W_exact=W_exact,
        missing_warnings=tuple(warnings),
        method=method,
        optimizer_settings={
            "margin": margin, "maxiter": maxiter, "sigf": sigf, "bound": bound,
            **ml_settings,
        },
        qp_solver=qp_solver,
        unit_labels=keep_labels,
        unit_codes=tuple(control_units),
        numerically_inactive=tuple(
            sorted(
                {k for k in ml_settings if k in NUMERICALLY_INACTIVE_ML}
                | {
                    k for k in NUMERICALLY_INACTIVE_QP
                    if (k == "sigf" and sigf != _SIGF)
                    or (k == "margin" and margin != _MARGIN)
                    or (k == "bound" and bound != _BOUND)
                }
            )
        ),
        stata_predictor_names=tuple(plabels),
    )

    if keep is not None:
        _write_keep(keep, result, replace)
    return result
