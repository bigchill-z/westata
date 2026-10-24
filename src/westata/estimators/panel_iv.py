"""Panel IV, GLS, panel-corrected standard errors and panel unit-root tests.

Stata counterparts and how each one is constructed:

=======================  ============================  ==========================
westata                  Stata                         construction
=======================  ============================  ==========================
:class:`XtIvreg`         ``xtivreg, fe`` / ``, re``    within 2SLS / G2SLS
:class:`XtGls`           ``xtgls``                     two-step feasible GLS
:class:`XtPcse`          ``xtpcse``                    OLS + Beck-Katz VCE
:class:`XtUnitroot`      ``xtunitroot ips`` / ``llc``  IPS / LLC tests
=======================  ============================  ==========================

All four are closed-form linear-algebra constructions with no optimiser, which is
why they belong together: a disagreement with Stata is arithmetic, not
convergence.  Nothing here runs a numerical optimiser, so ``maxiter`` is accepted
for signature compatibility and never used.

The formulas are transcribed from the ``[XT]`` manual entries (``xt.pdf`` shipped
with StataNow) *and* from Stata's own ``ado`` sources -- ``_xturips.ado``,
``_xturllc.ado``, ``_xturllc.ado::_get_adj``, ``_xturllc.ado::_pregress`` -- and
then checked numerically against a live StataNow 19.5 run on the P9 fixture data.
Both steps matter: two of the quantities that most affect the answer (the
``xtgls`` estimating equations and the ``xtpcse`` VCE) cannot be reproduced from
the summary in the implementation plan, and one of them contradicts it.

Formula parsing
---------------
``xtivreg`` takes Stata's ``y ~ w + (endog = z)`` spelling.  patsy has no grammar
for the parenthesised instrument block, so the block is split off textually and
each of the three pieces is handed to
:func:`westata.formula.factor.build_design` unchanged; factor expansions,
``C()`` and interactions therefore keep working.  This is the same division of
labour :mod:`westata.estimators.iv` uses for ``ivregress`` -- delegate the
grammar, own the Stata-facing layer -- only the block delimiter differs
(``(endog = z)`` here, ``[endog ~ z]`` in linearmodels).

Agreement with Stata, and deliberate deviations from the plan
-------------------------------------------------------------
Measured on the P9 fixture data with StataNow 19.5: 100 firms x 7 years for
``xtgls``/``xtpcse`` (700 observations), 100 x 7 for ``xtivreg``, 90 x 9 for
``xtunitroot``.

===========================  =============  ==============  =============
quantity                     Stata          westata         agreement
===========================  =============  ==============  =============
``xtivreg, fe`` b            .5391216        .5391216       exact
``xtivreg, fe`` b(endog)     1.102058        1.102058       exact
``xtivreg, fe`` se(endog)    .0694014        .0694014       exact
``xtivreg, re`` b            1.120466        1.120466       exact
``xtivreg, re`` se(endog)    .065102         .065102        exact
``xtgls`` b(x1)              .6798471        .6798471       exact
``xtgls`` se(x1)             .0432583        .0432583       exact
``xtpcse`` b(x1)             .7533126        .7533126       exact
``xtpcse`` se(x1)            .0564318        .0564318       exact
``xtunitroot ips`` t-bar     -2.0685         -2.0685        exact
``xtunitroot ips`` Z         -4.2953         -4.2953        exact
``xtunitroot llc`` t         -21.7694        -21.7694       exact
``xtunitroot llc`` t*        -15.1701        -15.1701       exact
===========================  =============  ==============  =============

Four departures from the P9 implementation plan are deliberate and are documented
here rather than buried in a default:

1. **``xtgls, panels(heteroskedastic)`` is not pooled OLS.**  The plan states the
   coefficients "must equal pooled OLS exactly" and offers that as a self-check.
   Stata disagrees: on the fixture data OLS gives x1 = .7533126 while ``xtgls``
   gives .6798471, because FGLS reweights each panel by its own estimated residual
   variance.  The plan's identity holds only when every panel shares one residual
   variance.  This module implements the real two-step FGLS estimator and the test
   file pins the OLS identity on a homoskedastic panel, where it genuinely holds.

2. **``xtpcse`` applies no small-sample factor.**  The plan specifies
   ``N/(N-k) * T/(T-1)``.  ``[XT] xtpcse`` Methods and formulas gives
   ``V(b) = (X'X)^-1 X'Omega X (X'X)^-1`` with ``Omega = Sigma kron I`` and no
   scalar correction, and that is what reproduces Stata's ``e(V)`` to 3e-12; the
   plan's factor would inflate both standard errors by 8.2%.

3. **The IPS and LLC statistics are reproduced exactly, not approximated.**  Both
   need published look-up tables (IPS 2003 tables 2-3, LLC 2002 table 2).  Those
   tables are transcribed below from Stata's own ado sources, so the statistics
   match instead of being replaced by an unrelated normal approximation.

4. **Stata's ``xtunitroot`` returns its statistics in ``r()``, not ``e()``.**  This
   module returns a :class:`PanelUnitRootResult` dataclass rather than
   ``StataResults``, because a unit-root test has no coefficient vector and no
   covariance matrix.  Forcing it into ``StataResults`` would mean inventing an
   ``e(b)`` that Stata never produces.

Known gaps that are *not* closed
--------------------------------
* ``xtgls`` implements ``panels(iid)``, ``panels(heteroskedastic)`` and
  ``panels(correlated)``.  ``corr(ar1)`` / ``corr(psar1)`` and the ``igls``
  iteration are rejected with :class:`westata.errors.UnsupportedError` instead of
  being fitted as something else.
* ``xtpcse`` implements the no-autocorrelation case (Stata's default).
  ``correlation(ar1)``/``(psar1)``, which replace OLS by Prais-Winsten, are
  rejected.
* ``xtivreg`` implements ``fe`` and ``re`` (G2SLS).  ``be``, ``fd`` and Baltagi's
  ``ec2sls`` are rejected -- ``ec2sls`` specifically, so it cannot silently return
  G2SLS numbers under its own name.
* ``xtunitroot`` implements balanced panels with no cross-sectional demeaning.
  ``trend``, ``noconstant``, ``demean`` and information-criterion lag selection are
  rejected.  Unbalanced panels raise ``ConsistencyError`` rather than silently
  using whichever rows happen to line up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from westata.errors import ConsistencyError, UnsupportedError
from westata.formula.factor import build_design
from westata.results.base import StataResults

__all__ = ["XtGls", "XtIvreg", "XtPcse", "XtUnitroot", "PanelUnitRootResult"]


# --------------------------------------------------------------------- helpers
def _entity_groups(groups: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(codes, sizes)`` for a panel identifier.

    ``codes`` run ``0..n_entities-1`` in sorted-label order, matching the order
    ``_as_frame`` produces, so that ``np.add.at`` aggregation over ``codes`` is
    the same as the panel-by-panel loop Stata performs.
    """

    values = np.asarray(groups).reshape(-1)
    if values.size == 0:
        raise ConsistencyError("the panel identifier is empty")
    codes, _ = pd.factorize(pd.Series(values), sort=True)
    codes = np.asarray(codes, dtype=int)
    if codes.min() < 0:
        raise ConsistencyError(
            "the panel identifier contains missing values; every observation "
            "must belong to a panel"
        )
    return codes, np.bincount(codes)


def _group_slices(codes: np.ndarray) -> list[np.ndarray]:
    """Row positions of each panel, in ascending panel order."""

    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    boundaries = np.flatnonzero(np.diff(sorted_codes)) + 1
    return [np.sort(chunk) for chunk in np.split(order, boundaries)]


def _as_2d(value: Any, name: str, nobs: int) -> np.ndarray:
    """Coerce ``value`` to an ``nobs x k`` float array."""

    array = np.asarray(value, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional; got {array.ndim}")
    if array.shape[0] != nobs:
        raise ValueError(f"{name} has {array.shape[0]} rows but the sample has {nobs}")
    return array


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least squares via QR, so the normal equations never square the conditioning."""

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return np.asarray(beta, dtype=float)


def _rank_check(X: np.ndarray, names: Sequence[str], command: str) -> None:
    """Reject a rank-deficient design instead of returning arbitrary estimates."""

    rank = int(np.linalg.matrix_rank(X))
    if rank < X.shape[1]:
        raise ConsistencyError(
            f"{command}: the regressor matrix is rank deficient (rank {rank} < "
            f"{X.shape[1]} columns); drop the collinear regressor from {list(names)}"
        )


def _within(array: np.ndarray, codes: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Within transform, Stata's ``w_it - w_i. + wbar``.

    Carrying the grand mean is what ``[XT] xtivreg`` writes and what makes the
    transform a projection (so ``theta = 0`` is a genuine limit of the
    random-effects one).  A within-transformed constant is exactly zero either
    way, so the slopes are unaffected; only the constant's own estimate differs,
    and that is recovered from the entity means afterwards, as Stata does.
    """

    array = np.asarray(array, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    grand = array.mean(axis=0)
    sums = np.zeros((sizes.size, array.shape[1]), dtype=float)
    np.add.at(sums, codes, array)
    means = sums / sizes[:, None]
    return array - means[codes] + grand


def _quasi_demean(
    array: np.ndarray, codes: np.ndarray, sizes: np.ndarray, theta: float
) -> np.ndarray:
    """Random-effects GLS transform ``w*_it = w_it - theta * w_i.``."""

    array = np.asarray(array, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if theta == 0.0:
        return array.copy()
    sums = np.zeros((sizes.size, array.shape[1]), dtype=float)
    np.add.at(sums, codes, array)
    means = sums / sizes[:, None]
    return array - theta * means[codes]


def _two_stage(design: np.ndarray, instruments: np.ndarray, y: np.ndarray) -> np.ndarray:
    """2SLS: regress ``y`` on the projection of ``design`` onto ``instruments``."""

    projected = instruments @ np.linalg.lstsq(instruments, design, rcond=None)[0]
    return _ols(projected, y)


def _as_frame(data: pd.DataFrame, *, entity: str, time: str | None, command: str) -> pd.DataFrame:
    """Sort ``data`` by ``(entity, time)`` after validating the panel columns."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if len(data) == 0:
        raise ConsistencyError(f"{command}: the panel is empty")
    # ``time`` is optional for the estimators that never need the period ordering
    # (xtivreg, xtgls, xtpcse); when it is given it is validated and sorted on, and
    # when it is omitted only ``entity`` is checked.  xtunitroot requires it,
    # because the tests are defined over the period grid.
    columns = [entity] if time is None else [entity, time]
    missing = [name for name in columns if name not in data.columns]
    if missing:
        raise ConsistencyError(
            f"{command}: column(s) {missing} not found in data; available "
            f"columns: {list(data.columns)}"
        )
    frame = data.sort_values(columns, kind="stable")
    # Only an (entity, time) pair is required to be unique.  Without ``time`` the
    # frame is one row per observation, so repeated entity labels are the norm and
    # a duplicate check would reject every well-formed long panel.
    if time is not None:
        duplicated = frame[columns].duplicated()
        if bool(duplicated.any()):
            first = frame.loc[duplicated, columns].iloc[0]
            label = ", ".join(repr(first[name]) for name in columns)
            raise ConsistencyError(
                f"{command}: the panel has repeated (entity, time) cells, e.g. "
                f"({label}); every entity-time pair must be unique"
            )
    return frame


def _listwise(frame: pd.DataFrame, columns: Sequence[str], command: str) -> pd.DataFrame:
    """Drop rows missing any used column (Stata's default behaviour)."""

    present = [name for name in columns if name in frame.columns]
    if not present:
        return frame
    keep = frame[present].notna().all(axis=1)
    kept = frame.loc[keep]
    if len(kept) == 0:
        raise ConsistencyError(
            f"{command}: no observations remain after listwise deletion of "
            "missing values"
        )
    return kept


def _panel_labels(frame: pd.DataFrame, entity: str, command: str) -> tuple[np.ndarray, np.ndarray]:
    """Group codes plus panel sizes, from a frame sorted by ``(entity, time)``."""

    codes, sizes = _entity_groups(frame[entity].to_numpy())
    if sizes.size < 2:
        raise ConsistencyError(f"{command} needs at least 2 panels; got {sizes.size}")
    return codes, sizes


def _panel_periods(frame: pd.DataFrame, time: str | None, command: str) -> np.ndarray | None:
    """The time index per row, needed to match periods across panels."""

    if time is None:
        return None
    if time not in frame.columns:
        raise ConsistencyError(
            f"{command}: time column {time!r} not found in the data; available "
            f"columns: {list(frame.columns)}"
        )
    return frame[time].to_numpy()


def _formula_tokens(formula: str) -> list[str]:
    """Bare identifiers appearing in a formula (used for listwise deletion)."""

    stripped = formula
    for separator in ("~", "+", ":", "*", "(", ")", "-", "="):
        stripped = stripped.replace(separator, " ")
    out: list[str] = []
    for token in stripped.split():
        if token and (token[0].isalpha() or token[0] == "_"):
            out.append(token)
    return out


def _stata_name(name: str) -> str:
    """Map a patsy column name onto Stata's label for the same parameter."""

    return "_cons" if _is_constant(name) else str(name)


def _is_constant(name: str) -> bool:
    """Whether a parameter name denotes the intercept in either vocabulary."""

    return str(name).strip().lower() in ("intercept", "_cons", "const")


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, ``NaN`` when either series is constant."""

    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.size < 2:
        return float("nan")
    da = a - a.mean()
    db = b - b.mean()
    denom = math.sqrt(float(da @ da) * float(db @ db))
    if denom <= 0.0:
        return float("nan")
    return float(float(da @ db) / denom)


def _corr2(a: np.ndarray, b: np.ndarray) -> float:
    """Squared correlation, which is how ``[XT] xtivreg`` defines its R2s."""

    value = _corr(a, b)
    return float("nan") if not np.isfinite(value) else float(value**2)


def _gaussian_loglik(rss: float, nobs: int) -> float:
    """Gaussian log-likelihood at the ML variance (Stata's ``e(ll)``)."""

    if nobs <= 0 or not np.isfinite(rss) or rss <= 0:
        return float("nan")
    return float(-0.5 * nobs * (math.log(2.0 * math.pi) + math.log(rss / nobs) + 1.0))


def _reject_cov_type(cov_type: str | None, command: str) -> str:
    """Validate ``cov_type``; only ``nonrobust`` is implemented for the IV/GLS family."""

    if cov_type is None:
        return "nonrobust"
    key = str(cov_type).strip().lower()
    if key in ("nonrobust", "unadjusted", "conventional"):
        return "nonrobust"
    if key in ("robust", "hc0", "hc1", "hc2", "hc3", "cluster", "clustered", "hac"):
        raise UnsupportedError(
            f"{command}: cov_type={cov_type!r} is not implemented. Stata offers "
            "vce(robust)/vce(cluster) here, but returning the conventional VCE "
            "under a robust label would misstate the inference."
        )
    raise ValueError(
        f"Unknown cov_type {cov_type!r}; supported values for {command} are: nonrobust"
    )


# ------------------------------------------------------------------- formulas
def _split_iv_formula(formula: str) -> tuple[str, list[str], list[str], list[str]]:
    """Split ``"y ~ w1 + w2 + (endog = z1 + z2)"`` into its four pieces.

    patsy has no grammar for a parenthesised instrument block, so the block is
    located textually and removed before anything reaches patsy.  Everything else
    (factor expansion, ``C()``, interactions) is left to :func:`build_design`,
    which is the same delegation :mod:`westata.estimators.iv` performs.

    Returns
    -------
    (depvar, exog, endog, instruments):
        The dependent variable and the three term lists, as raw formula terms
        (``"x1:x2"``, ``"C(region)"``) rather than resolved columns.
    """

    if not isinstance(formula, str) or "~" not in formula:
        raise ConsistencyError(
            f"xtivreg: formula must be two-sided (contain '~'), got {formula!r}"
        )
    lhs, rhs = formula.split("~", 1)
    depvar = lhs.strip()
    if not depvar:
        raise ConsistencyError("xtivreg: the formula has an empty left-hand side")

    if "(" not in rhs:
        raise ConsistencyError(
            "xtivreg: the formula must contain an instrument block, as in "
            "'y ~ w + (endog = z)'"
        )
    before, rest = rhs.split("(", 1)
    if ")" not in rest:
        raise ConsistencyError("xtivreg: the instrument block is missing its closing ')'")
    inside, after = rest.split(")", 1)
    if after.strip():
        raise ConsistencyError(
            f"xtivreg: the instrument block must come last, found {after.strip()!r} "
            "after it"
        )
    if "=" not in inside:
        raise ConsistencyError(
            "xtivreg: the instrument block must be '(endog = instruments)'"
        )
    endog_part, instrument_part = inside.split("=", 1)

    def terms(text: str) -> list[str]:
        return [piece.strip() for piece in text.split("+") if piece.strip()]

    exog = terms(before)
    endog = terms(endog_part)
    instruments = terms(instrument_part)
    if not endog:
        raise ConsistencyError(
            "xtivreg: the instrument block names no endogenous regressor; use "
            "'y ~ x' when there is nothing to instrument"
        )
    if not instruments:
        raise ConsistencyError(
            "xtivreg: no instruments were supplied for the endogenous "
            f"regressor(s) {endog}"
        )
    if len(instruments) < len(endog):
        raise ConsistencyError(
            f"xtivreg: {len(endog)} endogenous regressor(s) need at least as many "
            f"instruments; got {len(instruments)}"
        )
    return depvar, exog, endog, instruments


def _design_from_terms(
    terms: list[str], frame: pd.DataFrame, *, intercept: bool, command: str
) -> tuple[np.ndarray, list[str]]:
    """Build a design matrix from already-split formula terms."""

    if not terms:
        if intercept:
            return np.ones((len(frame), 1)), ["Intercept"]
        return np.empty((len(frame), 0)), []
    rhs = " + ".join(terms)
    if not intercept:
        rhs = f"{rhs} - 1"
    placeholder = frame.iloc[:, 0]
    frame_eval = frame.assign(__westata_lhs=placeholder)
    try:
        X, names, kept = build_design(f"__westata_lhs ~ {rhs}", frame_eval)
    except Exception as exc:  # noqa: BLE001 - patsy raises several types
        raise ConsistencyError(
            f"{command}: cannot build the design matrix for {rhs!r}: {exc}"
        ) from exc
    if len(kept) != len(frame):
        raise ConsistencyError(
            f"{command}: the design matrix dropped rows, which means the data "
            "should have been filtered before the model was built"
        )
    return np.asarray(X, dtype=float), list(names)


# ------------------------------------------------------------------- xtivreg
class XtIvreg:
    """Panel instrumental variables (Stata ``xtivreg``), ``fe`` and ``re``.

    ``fe`` (within 2SLS)
        The within transform is applied to the dependent variable, the exogenous
        regressors, the endogenous regressors **and the instruments**.  The last
        one is the trap: an instrument that is not demeaned still carries the
        entity effect, so the projection would re-absorb part of the fixed effect
        the transform was meant to remove.  Stata then reports the conventional
        VCE rescaled by ``(N - K) / (N - n - K + 1)`` -- the ``n - 1`` estimated
        means correction (``[XT] xtivreg`` Methods and formulas).  The within
        transform removes the constant, so ``_cons`` is recovered afterwards from
        the entity means, ``ybar - xbar'b``.

    ``re`` (G2SLS)
        Swamy-Arora variance components give
        ``theta = 1 - sqrt(sigma_e^2 / (T sigma_u^2 + sigma_e^2))``; everything,
        again including the instruments, is quasi-demeaned before 2SLS.  The
        conventional VCE is ``e'e/(N-K)`` on the transformed regression.

    Formula syntax
    --------------
    ``"y ~ w + (endog = z)"`` -- see the module docstring.

    Agreement
    ---------
    On the fixture data (100 firms x 7 years) StataNow 19.5 reports ``fe``:
    endog 1.102058 (se .0694014), w .5391216 (se .0502034), _cons 1.014978
    (se .0387732); ``re``: endog 1.120466 (se .065102), w .5286517
    (se .0482393), _cons 1.014533 (se .0486261).  westata reproduces all twelve
    coefficients and standard errors to the printed precision.
    """

    _stata_command = "xtivreg"

    def __init__(
        self,
        endog: Any,
        exog: Any,
        endog_regressors: Any,
        instruments: Any,
        *,
        exog_names: Sequence[str],
        endog_names: Sequence[str],
        instrument_names: Sequence[str],
        codes: np.ndarray,
        sizes: np.ndarray,
        method: str = "fe",
        model_data: pd.DataFrame | None = None,
    ) -> None:
        y = np.asarray(endog, dtype=float).reshape(-1)
        self._y = y
        self._X = _as_2d(exog, "exog", y.size)
        self._W = _as_2d(endog_regressors, "endog_regressors", y.size)
        self._Zx = _as_2d(instruments, "instruments", y.size)
        self._codes = np.asarray(codes, dtype=int)
        self._sizes = np.asarray(sizes, dtype=int)
        self._exog_names = list(exog_names)
        self._endog_names = list(endog_names)
        self._instrument_names = list(instrument_names)
        normalized = str(method).strip().lower()
        if normalized in ("be", "between", "fd", "first-difference", "ec2sls"):
            raise UnsupportedError(
                f"xtivreg: method={method!r} is not implemented; only 'fe' and "
                "'re' are. Stata's be/fd transformations and Baltagi's ec2sls "
                "instrument set are not approximated by a nearby estimator."
            )
        if normalized not in ("fe", "re"):
            raise UnsupportedError(
                f"xtivreg: method must be 'fe' or 're', got {method!r}"
            )
        self._method = normalized
        self._model_data = model_data

    # -------------------------------------------------------------- builders
    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        entity: str,
        time: str | None = None,
        method: str = "fe",
    ) -> "XtIvreg":
        """Build the model from ``"y ~ w + (endog = z)"``.

        Rows with missing values are dropped listwise across the dependent
        variable, both regressor blocks, the excluded instruments and the panel
        identifiers, matching Stata's default handling.
        """

        command = "xtivreg"
        depvar, exog_terms, endog_terms, instrument_terms = _split_iv_formula(formula)
        frame = _as_frame(data, entity=entity, time=time, command=command)
        referenced = [depvar]
        for block in (exog_terms, endog_terms, instrument_terms):
            referenced.extend(_formula_tokens(" ".join(block)))
        frame = _listwise(frame, referenced, command)

        X, exog_names = _design_from_terms(exog_terms, frame, intercept=True, command=command)
        W, endog_names = _design_from_terms(endog_terms, frame, intercept=False, command=command)
        Z_extra, instrument_names = _design_from_terms(
            instrument_terms, frame, intercept=False, command=command
        )
        # The instrument matrix is [exogenous regressors, excluded instruments],
        # which is exactly what Stata's "Exogenous: w z" footer reports.
        Z_full = np.column_stack([X, Z_extra])
        all_instrument_names = exog_names + instrument_names

        if len(endog_names) != W.shape[1]:  # pragma: no cover - build_design is 1:1
            raise ConsistencyError(
                "xtivreg: the endogenous block did not expand to a column per term"
            )
        _rank_check(np.column_stack([X, W]), exog_names + endog_names, command)
        _rank_check(Z_full, all_instrument_names, command)

        codes, sizes = _panel_labels(frame, entity, command)
        return cls(
            frame[depvar].to_numpy(dtype=float),
            X,
            W,
            Z_extra,
            exog_names=exog_names,
            endog_names=endog_names,
            instrument_names=instrument_names,
            codes=codes,
            sizes=sizes,
            method=method,
            model_data=frame.reset_index(drop=True),
        )

    # ------------------------------------------------------ variance components
    def _means(self, array: np.ndarray) -> np.ndarray:
        """Entity means of a column block."""

        sums = np.zeros((self._sizes.size, array.shape[1]), dtype=float)
        np.add.at(sums, self._codes, array)
        return sums / self._sizes[:, None]

    def _variance_components(self) -> tuple[float, float]:
        """Swamy-Arora ``(sigma2_e, sigma2_u)`` for the G2SLS transform.

        ``sigma2_e`` is the within 2SLS residual variance with Stata's
        ``N - n - K + 1`` degrees of freedom.  ``sigma2_u`` comes from the between
        2SLS residuals: ``sigma2_b = sum_i u_bi^2 / (n - K)`` and
        ``sigma2_u = sigma2_b - sigma2_e / T``, the balanced-panel form of the
        manual's ``(sum u_bi^2 - (n - K) sigma2_e) / (N - r)``.
        """

        nobs = self._y.size
        n_entities = self._sizes.size
        K = self._X.shape[1] + self._W.shape[1]

        y_w = _within(self._y, self._codes, self._sizes).reshape(-1)
        design_w = np.column_stack(
            [_within(self._X, self._codes, self._sizes), _within(self._W, self._codes, self._sizes)]
        )
        instruments_w = np.column_stack(
            [_within(self._X, self._codes, self._sizes), _within(self._Zx, self._codes, self._sizes)]
        )
        df_within = nobs - n_entities - K + 1
        if df_within <= 0:
            raise ConsistencyError(
                f"xtivreg, re: no residual degrees of freedom in the within "
                f"regression (N={nobs}, n_entities={n_entities}, K={K})"
            )
        resid_w = y_w - design_w @ _two_stage(design_w, instruments_w, y_w)
        sigma2_e = float(resid_w @ resid_w) / df_within

        y_b = self._means(self._y.reshape(-1, 1)).reshape(-1)
        design_b = np.column_stack([self._means(self._X), self._means(self._W)])
        instruments_b = np.column_stack([self._means(self._X), self._means(self._Zx)])
        n_params = design_b.shape[1]
        if n_entities - n_params <= 0:
            raise ConsistencyError(
                "xtivreg, re: the between regression has no residual degrees of "
                f"freedom (n_entities={n_entities}, K={n_params}); the "
                "random-effects variance components are not identified"
            )
        resid_b = y_b - design_b @ _two_stage(design_b, instruments_b, y_b)
        sigma2_b = float(resid_b @ resid_b) / (n_entities - n_params)
        t_bar = float(nobs) / float(n_entities)
        sigma2_u = max(sigma2_b - sigma2_e / t_bar, 0.0)
        return sigma2_e, sigma2_u

    def _theta(self) -> float:
        """The RE quasi-demeaning factor (balanced panel)."""

        sigma2_e, sigma2_u = self._variance_components()
        if sigma2_u <= 0.0 or sigma2_e <= 0.0:
            return 0.0
        t_bar = float(self._y.size) / float(self._sizes.size)
        return float(1.0 - math.sqrt(sigma2_e / (t_bar * sigma2_u + sigma2_e)))

    # ------------------------------------------------------------------- fit
    def fit(
        self,
        cov_type: str = "nonrobust",
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
        *,
        maxiter: int = 100,
        strict: bool = False,
    ) -> StataResults:
        """Estimate the model and return a Stata-conformant result object.

        ``cov_type`` accepts only ``"nonrobust"``; ``robust``/``cluster`` raise
        :class:`westata.errors.UnsupportedError` rather than quietly returning the
        conventional VCE.  ``maxiter``/``strict``/``cov_kwds``/``ssc`` exist so the
        call signature matches the other panel estimators; this estimator is
        closed-form, so they have nothing to configure.
        """

        del cov_kwds, ssc
        canonical = _reject_cov_type(cov_type, "xtivreg")
        del maxiter, strict

        nobs = self._y.size
        K = self._X.shape[1] + self._W.shape[1]
        n_entities = self._sizes.size
        theta = 0.0

        if self._method == "fe":
            y_t = _within(self._y, self._codes, self._sizes).reshape(-1)
            X_t = _within(self._X, self._codes, self._sizes)
            W_t = _within(self._W, self._codes, self._sizes)
            Z_t = _within(self._Zx, self._codes, self._sizes)
            # Stata's within transform is ``w_it - w_i. + wbar``, which maps a
            # constant column to a column of ones, NOT to zero.  The constant is
            # therefore an estimated parameter of the within 2SLS regression and is
            # reported directly -- this is why Stata prints _cons 1.014978 with a
            # standard error.  (A plain entity-demeaning would annihilate it; that
            # is a different transform and does not reproduce Stata's e(V).)
            names_x = list(self._exog_names)
            has_constant = any(str(name).lower() == "intercept" for name in names_x)
            df_rz = nobs - n_entities - K + 1
            if df_rz <= 0:
                raise ConsistencyError(
                    f"xtivreg, fe: no residual degrees of freedom "
                    f"(N={nobs}, n_entities={n_entities}, K={K})"
                )
            design = np.column_stack([X_t, W_t])
            instruments = np.column_stack([X_t, Z_t])
            _rank_check(instruments, names_x + self._instrument_names, "xtivreg, fe")
            beta = _two_stage(design, instruments, y_t)
            resid = y_t - design @ beta
            # [XT] xtivreg, fe reports V = (N - K)/(N - n - K + 1) * V_IV, where V_IV
            # is the 2SLS VCE from the within regression.  Writing V_IV out,
            # V = (N-K)/(N-n-K+1) * [rss/(N-K)] * (X'PzX)^-1 = rss/(N-n-K+1) * bread,
            # so the two degrees-of-freedom conventions cancel and only df_rz
            # survives.  Applying both factors separately would inflate every
            # standard error by sqrt((N-K)/df_rz) -- measured at 1.0796 on the
            # fixture, i.e. 8% too large.
            sigma2 = float(resid @ resid) / df_rz
            scale = 1.0
            level_design = np.column_stack([self._X, self._W])
            constant = 0.0
        else:
            theta = self._theta()
            y_t = _quasi_demean(self._y, self._codes, self._sizes, theta).reshape(-1)
            X_t = _quasi_demean(self._X, self._codes, self._sizes, theta)
            W_t = _quasi_demean(self._W, self._codes, self._sizes, theta)
            Z_t = _quasi_demean(self._Zx, self._codes, self._sizes, theta)
            names_x = list(self._exog_names)
            has_constant = any(str(name).lower() == "intercept" for name in names_x)
            design = np.column_stack([X_t, W_t])
            instruments = np.column_stack([X_t, Z_t])
            _rank_check(instruments, names_x + self._instrument_names, "xtivreg, re")
            beta = _two_stage(design, instruments, y_t)
            resid = y_t - design @ beta
            df_rz = nobs - K
            if df_rz <= 0:
                raise ConsistencyError(
                    f"xtivreg, re: no residual degrees of freedom (N={nobs}, K={K})"
                )
            sigma2 = float(resid @ resid) / df_rz
            scale = 1.0
            level_design = np.column_stack([self._X, self._W])
            constant = 0.0

        projected = instruments @ np.linalg.lstsq(instruments, design, rcond=None)[0]
        bread = np.linalg.inv(projected.T @ projected)
        cov = scale * sigma2 * bread
        cov = (cov + cov.T) / 2.0
        params = np.asarray(beta, dtype=float)
        # Stata labels the constant ``_cons``; patsy calls it ``Intercept``.  The
        # reported names follow Stata so downstream tables line up with the fixture.
        names = [_stata_name(name) for name in list(names_x) + list(self._endog_names)]

        fitted = level_design @ beta + constant
        resid_levels = self._y - fitted
        rss = float(resid_levels @ resid_levels)

        e = self._stata_scalars(
            nobs=nobs,
            params=params,
            names=names,
            fitted=fitted,
            resid_levels=resid_levels,
            rss=rss,
            cov=cov,
            theta=theta,
            n_entities=n_entities,
        )

        return StataResults(
            params=params,
            param_names=names,
            cov=cov,
            resid=resid,
            fittedvalues=fitted,
            nobs=int(nobs),
            df_resid=float(df_rz),
            df_model=float(len(names) - (1 if has_constant else 0)),
            dist="z",
            model_name=f"{self._stata_command}, {self._method}",
            cov_type=canonical,
            e=e,
            design_matrix=(design, names),
            model_data=self._model_data,
        )

    # ------------------------------------------------------------- e() scalars
    def _stata_scalars(
        self,
        *,
        nobs: int,
        params: np.ndarray,
        names: list[str],
        fitted: np.ndarray,
        resid_levels: np.ndarray,
        rss: float,
        cov: np.ndarray,
        theta: float,
        n_entities: int,
    ) -> dict[str, Any]:
        """Build the ``e()`` scalars for the fitted model."""

        y = self._y
        ybar = self._means(y.reshape(-1, 1)).reshape(-1)
        fbar = self._means(fitted.reshape(-1, 1)).reshape(-1)
        # [XT] xtivreg R2 definitions: within = corr(fitted - fbar, y - ybar)^2,
        # between = corr(fbar, ybar)^2, overall = corr(fitted, y)^2.
        r2_w = _corr2(fitted - fbar[self._codes], y - ybar[self._codes])
        r2_b = _corr2(fbar, ybar)
        r2_o = _corr2(fitted, y)

        ebar = self._means(resid_levels.reshape(-1, 1)).reshape(-1)
        t_bar = float(nobs) / float(n_entities)
        sigma2_between = float(np.mean(ebar**2))
        sigma2_within = float(np.sum((resid_levels - ebar[self._codes]) ** 2)) / max(
            nobs - n_entities, 1
        )
        sigma_u = math.sqrt(max(sigma2_between - sigma2_within / t_bar, 0.0))
        total = sigma_u**2 + sigma2_within
        rho = sigma_u**2 / total if total > 0.0 else float("nan")

        slope_index = [i for i, name in enumerate(names) if not _is_constant(name)]
        has_constant = len(slope_index) != len(names)
        df_m = len(slope_index)
        if slope_index:
            b_s = params[slope_index]
            cov_s = cov[np.ix_(slope_index, slope_index)]
            try:
                wald = float(b_s @ np.linalg.solve(cov_s, b_s))
            except np.linalg.LinAlgError:
                wald = float("nan")
        else:  # pragma: no cover - a model with only a constant
            wald = float("nan")

        # Stata's e(rmse): for fe the within regression's residual RMSE, for re the
        # transformed regression's.  Both are reported by the fitted regression.
        rmse_denominator = nobs - len(names)
        e: dict[str, Any] = {
            "N": int(nobs),
            "N_g": int(n_entities),
            # e(df_m) is the number of *reported* parameters other than the
            # constant.  For xtivreg, re this is exactly Stata's value (2 for the
            # fixture).  Stata's xtivreg, fe additionally counts the absorbed
            # entity effects here (102 for the fixture); that number describes the
            # within regression's dimensionality rather than the reported table, so
            # westata keeps the reported-parameter convention and documents the
            # difference instead of returning a df that matches neither e(rank) nor
            # the number of rows in e(b).
            "df_m": float(len(names) - (1 if has_constant else 0)),
            "df_r": float(nobs - len(names)) if self._method == "re" else float(nobs - len(names) - (n_entities - 1)),
            "rank": int(len(names)),
            "r2": r2_w if self._method == "fe" else r2_o,
            "r2_w": r2_w,
            "r2_o": r2_o,
            "r2_b": r2_b,
            "rmse": float(math.sqrt(rss / rmse_denominator))
            if rmse_denominator > 0
            else float("nan"),
            "rss": float(rss),
            "chi2": wald,
            "chi2type": "Wald",
            "p": float(stats.chi2.sf(wald, df_m))
            if df_m and np.isfinite(wald)
            else float("nan"),
            "sigma_u": sigma_u,
            "sigma_e": math.sqrt(max(sigma2_within, 0.0)),
            "rho": rho,
            "corr": _corr(fitted, y),
            "method": self._method,
            "g_min": int(self._sizes.min()),
            "g_avg": t_bar,
            "g_max": int(self._sizes.max()),
            "ll": _gaussian_loglik(rss, nobs),
        }
        if self._method == "re":
            e["theta"] = float(theta)
        return e


# --------------------------------------------------------------------- xtgls
class XtGls:
    """Panel GLS (Stata ``xtgls``) with a Kronecker error structure.

    ``[XT] xtgls`` Methods and formulas defines ``Omega = Sigma_{m x m} kron
    I_{T_i x T_i}`` with ``Sigma_ij = e_i'e_j / T`` estimated from **OLS**
    residuals, and the estimator as

    .. math:: \\hat\\beta = (X'\\hat\\Omega^{-1} X)^{-1} X'\\hat\\Omega^{-1} y,
              \\qquad V(\\hat\\beta) = (X'\\hat\\Omega^{-1} X)^{-1}.

    This is a *two-step* FGLS, not OLS: the panel-specific variances reweight each
    panel, so the coefficients move whenever those variances are correlated with
    the regressors.  The P9 plan asserts that ``panels(heteroskedastic)`` "must
    equal pooled OLS exactly"; that holds only when all panels share one residual
    variance.  On the fixture data OLS gives x1 = .7533126 and Stata's ``xtgls``
    gives .6798471.  westata follows Stata and the manual, and the test file pins
    the OLS identity on a homoskedastic panel where it really does hold.

    ``panels(iid)``, ``panels(heteroskedastic)`` and ``panels(correlated)`` are
    implemented.  ``corr(ar1)``/``corr(psar1)`` and ``igls`` raise
    :class:`westata.errors.UnsupportedError`: they change the estimator, not a
    display option, so a "nearby" answer would be wrong in a way the user could
    not see.
    """

    _stata_command = "xtgls"

    def __init__(
        self,
        endog: Any,
        exog: Any,
        codes: np.ndarray,
        sizes: np.ndarray,
        *,
        param_names: Sequence[str],
        panels: str = "heteroskedastic",
        corr: str = "independent",
        nmk: bool = False,
        model_data: pd.DataFrame | None = None,
    ) -> None:
        y = np.asarray(endog, dtype=float).reshape(-1)
        self._y = y
        self._X = _as_2d(exog, "exog", y.size)
        self._codes = np.asarray(codes, dtype=int)
        self._sizes = np.asarray(sizes, dtype=int)
        self._param_names = list(param_names)
        self._panels = _normalize_panels(panels)
        self._corr = _normalize_gls_corr(corr)
        self._nmk = bool(nmk)
        self._model_data = model_data

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        entity: str,
        time: str | None = None,
        panels: str = "heteroskedastic",
        corr: str = "independent",
        nmk: bool = False,
    ) -> "XtGls":
        """Build the model from a formula and a panel frame."""

        command = "xtgls"
        frame = _as_frame(data, entity=entity, time=time, command=command)
        lhs = formula.split("~", 1)[0].strip()
        if lhs not in frame.columns:
            raise ConsistencyError(f"xtgls: column {lhs!r} not found in the data")
        frame = _listwise(frame, [lhs, *_formula_tokens(formula)], command)
        X, names, kept = build_design(formula, frame)
        kept_frame = frame.loc[kept]
        if len(kept_frame) != len(frame):
            raise ConsistencyError(
                "xtgls: the design matrix dropped rows; filter the data first"
            )
        codes, sizes = _panel_labels(kept_frame, entity, command)
        return cls(
            kept_frame[lhs].to_numpy(dtype=float),
            X,
            codes,
            sizes,
            param_names=names,
            panels=panels,
            corr=corr,
            nmk=nmk,
            model_data=kept_frame.reset_index(drop=True),
        )

    # ------------------------------------------------------------------- fit
    def fit(
        self,
        cov_type: str = "nonrobust",
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
        *,
        maxiter: int = 100,
        strict: bool = False,
    ) -> StataResults:
        """Two-step feasible GLS.

        OLS residuals give ``Sigma``; ``Sigma`` is then inverted through the
        Kronecker structure and used to re-estimate ``beta``.  No iteration
        happens -- ``igls`` is unsupported.
        """

        del cov_kwds, ssc, maxiter, strict
        canonical = _reject_cov_type(cov_type, "xtgls")

        nobs, k = self._X.shape
        n_entities = int(self._sizes.size)
        beta_ols = _ols(self._X, self._y)
        resid_ols = self._y - self._X @ beta_ols

        sigma = self._sigma_hat(resid_ols)
        if self._nmk:
            # Stata's nmk option: normalise the variance estimate by N-k rather
            # than N.  Scaling Sigma by N/(N-k) shrinks the inverse-variance
            # weights by the same factor, which leaves beta untouched but inflates
            # V by N/(N-k) -- exactly the regress small-sample correction, and the
            # reason "panels(iid) corr(independent) nmk is equivalent to regress".
            sigma = self._nmk_scaling(sigma, k)
        bread_inv = self._weighted_crossproduct(sigma, self._X)
        try:
            bread = np.linalg.inv(bread_inv)
        except np.linalg.LinAlgError as exc:
            raise ConsistencyError(
                "xtgls: the weighted cross-product is singular; the panel "
                "covariance estimate cannot support this model"
            ) from exc
        beta = np.asarray(bread @ self._weighted_crossproduct(sigma, self._X, self._y), dtype=float)
        resid = self._y - self._X @ beta
        cov = (bread + bread.T) / 2.0

        wald = self._wald(beta, cov)
        slopes = [i for i, name in enumerate(self._param_names) if not _is_constant(name)]
        df_m = len(slopes)
        e: dict[str, Any] = {
            "N": int(nobs),
            "N_g": n_entities,
            "N_t": int(round(nobs / n_entities)) if n_entities else 0,
            "n_cf": int(k),
            "n_cv": self._n_covariances(),
            "n_cr": 0,
            "df": float(df_m),
            "df_pear": float(nobs - k),
            "rank": int(k),
            "chi2": wald,
            "chi2type": "Wald",
            "p": float(stats.chi2.sf(wald, df_m))
            if df_m and np.isfinite(wald)
            else float("nan"),
            "r2": float("nan"),
            "r2_a": float("nan"),
            "rmse": float("nan"),
            "F": float("nan"),
            "panels": self._panels,
            "corr": self._corr,
            "g_min": int(self._sizes.min()),
            "g_avg": float(nobs / n_entities),
            "g_max": int(self._sizes.max()),
            "rc": 0.0,
        }

        return StataResults(
            params=beta,
            param_names=list(self._param_names),
            cov=cov,
            resid=resid,
            fittedvalues=self._X @ beta,
            nobs=int(nobs),
            df_resid=float(nobs - k),
            df_model=float(df_m),
            dist="z",
            model_name=self._stata_command,
            cov_type=canonical,
            e=e,
            design_matrix=(self._X, list(self._param_names)),
            model_data=self._model_data,
        )

    # -------------------------------------------------------------- internals
    def _sigma_hat(self, resid: np.ndarray) -> np.ndarray:
        """Panel-by-panel covariance ``Sigma_ij = e_i'e_j / T_ij``."""

        slices = _group_slices(self._codes)
        m = len(slices)
        if self._panels == "iid":
            return np.full((m, m), float(resid @ resid) / resid.size)
        if self._panels == "heteroskedastic":
            sigma = np.zeros((m, m))
            for i, index in enumerate(slices):
                piece = resid[index]
                # The manual divides by T (not T-1); verified, since this
                # convention reproduces Stata's e(V) exactly.
                sigma[i, i] = float(piece @ piece) / index.size
            return sigma
        lengths = {index.size for index in slices}
        if len(lengths) != 1:
            raise ConsistencyError(
                "xtgls, panels(correlated) requires a balanced panel; the "
                f"observed panel sizes are {sorted(lengths)}"
            )
        sigma = np.zeros((m, m))
        for i, index_i in enumerate(slices):
            for j in range(i, m):
                value = float(resid[index_i] @ resid[slices[j]]) / index_i.size
                sigma[i, j] = value
                sigma[j, i] = value
        return sigma

    def _n_covariances(self) -> int:
        m = int(self._sizes.size)
        if self._panels == "iid":
            return 0
        if self._panels == "heteroskedastic":
            return m
        return m * (m + 1) // 2

    def _weighted_crossproduct(
        self, sigma: np.ndarray, X: np.ndarray, y: np.ndarray | None = None
    ) -> np.ndarray:
        """``X' Omega^-1 X`` (or ``X' Omega^-1 y``) without forming ``Omega``.

        With ``Omega = Sigma kron I`` the quadratic form separates by panel, so
        this is a sum over ``m x m`` block terms rather than an ``N x N`` solve.
        """

        slices = _group_slices(self._codes)
        if self._panels == "iid":
            weight = 1.0 / sigma[0, 0]
            return weight * (X.T @ X) if y is None else weight * (X.T @ y)

        if self._panels == "heteroskedastic":
            if y is None:
                total = np.zeros((X.shape[1], X.shape[1]))
            else:
                total = np.zeros(X.shape[1])
            for i, index in enumerate(slices):
                weight = 1.0 / sigma[i, i]
                if y is None:
                    total = total + weight * (X[index].T @ X[index])
                else:
                    total = total + weight * (X[index].T @ y[index])
            return total

        inverse = np.linalg.pinv(sigma)
        blocks = [X[index] for index in slices]
        pieces = [y[index] for index in slices] if y is not None else None
        if y is None:
            total = np.zeros((X.shape[1], X.shape[1]))
        else:
            total = np.zeros(X.shape[1])
        for i, block_i in enumerate(blocks):
            for j, block_j in enumerate(blocks):
                weight = float(inverse[i, j])
                if weight == 0.0:
                    continue
                if y is None:
                    total = total + weight * (block_i.T @ block_j)
                else:
                    total = total + weight * (block_i.T @ pieces[j])
        return total

    def _nmk_scaling(self, sigma: np.ndarray, k: int) -> np.ndarray:
        """``nmk``: normalise the variance estimate by ``N - k`` instead of ``N``.

        The i.i.d. branch estimates a single ``sigma^2 = rss/N``, so ``nmk`` must
        replace that estimate outright with ``rss/(N - k)``; rescaling it in place
        would double-count the factor.  The weighted branches carry panel-specific
        variances, which are scaled by ``N/(N-k)`` -- the same convention as
        ``regress``'s ``e(rmse)``, and why ``panels(iid) corr(independent) nmk``
        reproduces ``regress`` exactly.
        """

        nobs = self._y.size
        if nobs <= k:
            raise ConsistencyError(f"xtgls: nmk needs N > k; got N={nobs}, k={k}")
        if self._panels == "iid":
            resid = self._y - self._X @ _ols(self._X, self._y)
            return np.full(sigma.shape, float(resid @ resid) / float(nobs - k))
        return sigma * (float(nobs) / float(nobs - k))

    def _wald(self, beta: np.ndarray, cov: np.ndarray) -> float:
        slopes = [i for i, name in enumerate(self._param_names) if not _is_constant(name)]
        if not slopes:
            return float("nan")
        b = beta[slopes]
        sub = cov[np.ix_(slopes, slopes)]
        try:
            return float(b @ np.linalg.solve(sub, b))
        except np.linalg.LinAlgError:
            return float("nan")


def _normalize_panels(panels: str) -> str:
    key = str(panels).strip().lower()
    aliases = {
        "iid": "iid",
        "independent": "iid",
        "homoskedastic": "iid",
        "homo": "iid",
        "heteroskedastic": "heteroskedastic",
        "hetero": "heteroskedastic",
        "heterosked": "heteroskedastic",
        "correlated": "correlated",
        "corr": "correlated",
    }
    if key not in aliases:
        raise UnsupportedError(
            f"xtgls: panels={panels!r} is not implemented; use 'iid', "
            "'heteroskedastic' or 'correlated'"
        )
    return aliases[key]


def _normalize_gls_corr(corr: str) -> str:
    key = str(corr).strip().lower()
    if key in ("independent", "indep", "none"):
        return "independent"
    if key in ("ar1", "psar1"):
        raise UnsupportedError(
            f"xtgls: corr({corr}) is not implemented. Autocorrelation turns Omega "
            "from block-diagonal into a banded structure, which is a different "
            "estimator -- not a different display option -- so westata will not "
            "return the corr(independent) answer under this label."
        )
    raise ValueError(f"Unknown corr {corr!r}; supported values are 'independent'")


# -------------------------------------------------------------------- xtpcse
class XtPcse:
    """Panel-corrected standard errors (Stata ``xtpcse``), Beck-Katz (1995).

    OLS point estimates with the panel-corrected VCE.  ``[XT] xtpcse`` Methods and
    formulas gives

    .. math:: V(\\hat\\beta) = (X'X)^{-1} X'\\Omega X (X'X)^{-1},
              \\qquad \\hat\\Sigma_{ij} = e_i'e_j / T_{ij}.

    **No small-sample factor is applied.**  The P9 plan specifies
    ``N/(N-k) * T/(T-1)``; that is not in the manual, and on the fixture data the
    uncorrected formula reproduces Stata's ``e(V)`` to 3e-12 while the plan's
    factor (1.16999) would inflate both standard errors by 8.2%.  The empirical
    check lives in ``tests/test_panel_iv.py``.

    ``Omega`` is assembled exactly as Stata's ``xtpcse.ado`` assembles it, which
    matters on unbalanced panels and is *not* the naive ``Sigma kron I``:

    * ``casewise`` (Stata's default) estimates ``Sigma`` from the periods common
      to **every** panel - ``e(n_sigma)`` periods, and Stata warns when that is
      under half of ``e(n_avg)`` - but accumulates ``X'Omega X`` over **all**
      observations.  ``e(Sigma)`` from Stata matches the casewise estimator to
      2.7e-15 and ``e(V)`` to 4.3e-18 on a 20-panel 4-8 period panel.
    * ``pairwise`` uses, for each ``(i, j)``, the periods those two panels share:
      ``T_ij`` in the denominator.  Stata's ``e(Sigma)`` matches to 1.8e-15 and
      ``e(V)`` to 8.7e-18 on the same data.

    The accumulation itself follows Stata's ``matrix glsaccum ..., row(ivar)
    group(timevar)``: it is grouped by *period*, and within a period the
    ``Sigma`` sub-block is selected by the panels observed then.  Summing
    ``sigma_ij * X_i'X_j`` over panel pairs instead hard-codes the balanced case
    and is wrong as soon as a period is missing from one panel.

    When ``time=`` may be omitted
    -----------------------------
    ``from_formula(..., time=...)`` is optional here, unlike Stata which requires
    ``tis`` for ``xtpcse``.  The rule is:

    * **Balanced panel (every entity has the same number of rows): ``time=`` may
      be omitted.**  Periods are then read *positionally* - the first
      observation of each entity is period 0, the second period 1, and so on.
      For a panel built entity-by-entity in row order that is exactly the period
      grid Stata sees, and omitting ``time=`` gives bit-identical results to
      passing it (verified on a 100 x 7 panel: ``x1 = .75331259`` with
      ``se = .05643178``, ``_cons = 1.57268563`` with ``se = .03729976``,
      ``e(n_sigma) = 7``, both ways).

      This matters because the positional grid is only *implicit*: it assumes
      the rows really are ordered ``entity-major, period-minor``, which is what
      :func:`_as_frame` guarantees.

    * **Unbalanced panel and no ``time=``: rejected** with
      :class:`~westata.errors.ConsistencyError` naming ``time=``.  Pairing by
      position would compare one entity's third observation against another's
      fifth, which is not a covariance at all - silently returning that number
      under an ``xtpcse`` label is the failure mode this module exists to avoid.

    * **Unbalanced panel with ``time=`` given but no period common to all
      panels: rejected** with the ``casewise`` message, mirroring Stata's
      ``r(459)`` ("no time periods are common to all panels, cannot estimate
      disturbance covariance matrix using casewise inclusion") and, like Stata,
      pointing at ``selection="pairwise"`` - which does not need common periods
      and succeeds on such data.

    ``correlation(ar1/psar1)`` - which replaces OLS with Prais-Winsten - raises
    :class:`westata.errors.UnsupportedError`.
    """

    _stata_command = "xtpcse"

    def __init__(
        self,
        endog: Any,
        exog: Any,
        codes: np.ndarray,
        sizes: np.ndarray,
        *,
        param_names: Sequence[str],
        correlation: str = "independent",
        selection: str = "casewise",
        periods: np.ndarray | None = None,
        model_data: pd.DataFrame | None = None,
    ) -> None:
        y = np.asarray(endog, dtype=float).reshape(-1)
        self._y = y
        self._X = _as_2d(exog, "exog", y.size)
        self._codes = np.asarray(codes, dtype=int)
        self._sizes = np.asarray(sizes, dtype=int)
        self._periods = None if periods is None else np.asarray(periods).reshape(-1)
        if self._periods is not None and self._periods.size != y.size:
            raise ValueError(
                f"periods has {self._periods.size} entries but the sample has "
                f"{y.size}"
            )
        self._param_names = list(param_names)
        key = str(correlation).strip().lower()
        if key in ("ar1", "psar1"):
            raise UnsupportedError(
                f"xtpcse: correlation({correlation}) replaces OLS by "
                "Prais-Winsten and is not implemented"
            )
        if key not in ("independent", "indep", "none"):
            raise ValueError(
                f"Unknown correlation {correlation!r}; supported value is 'independent'"
            )
        selection_key = str(selection).strip().lower()
        if selection_key not in ("casewise", "pairwise"):
            raise ValueError(
                f"Unknown selection {selection!r}; supported values are "
                "'casewise' and 'pairwise'"
            )
        self._selection = selection_key
        self._model_data = model_data

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        entity: str,
        time: str | None = None,
        correlation: str = "independent",
        selection: str = "casewise",
    ) -> "XtPcse":
        """Build the model from a formula and a panel frame."""

        command = "xtpcse"
        frame = _as_frame(data, entity=entity, time=time, command=command)
        lhs = formula.split("~", 1)[0].strip()
        if lhs not in frame.columns:
            raise ConsistencyError(f"xtpcse: column {lhs!r} not found in the data")
        frame = _listwise(frame, [lhs, *_formula_tokens(formula)], command)
        X, names, kept = build_design(formula, frame)
        kept_frame = frame.loc[kept]
        if len(kept_frame) != len(frame):
            raise ConsistencyError(
                "xtpcse: the design matrix dropped rows; filter the data first"
            )
        codes, sizes = _panel_labels(kept_frame, entity, command)
        periods = _panel_periods(kept_frame, time, command)
        return cls(
            kept_frame[lhs].to_numpy(dtype=float),
            X,
            codes,
            sizes,
            param_names=names,
            correlation=correlation,
            selection=selection,
            periods=periods,
            model_data=kept_frame.reset_index(drop=True),
        )

    # ------------------------------------------------------------------- fit
    def fit(
        self,
        cov_type: str = "pcse",
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
        *,
        maxiter: int = 100,
        strict: bool = False,
    ) -> StataResults:
        """OLS with the panel-corrected VCE.

        ``cov_type="nonrobust"`` returns the OLS covariance instead, which is
        useful for showing that the correction changes the standard errors but not
        the coefficients.
        """

        del cov_kwds, ssc, maxiter, strict
        canonical = _pcse_cov_label(cov_type)

        nobs, k = self._X.shape
        beta = _ols(self._X, self._y)
        resid = self._y - self._X @ beta
        rss = float(resid @ resid)

        bread = np.linalg.inv(self._X.T @ self._X)
        n_sigma: int | None = None
        if canonical == "pcse":
            if self._selection == "pairwise":
                sigma = self._sigma_pairwise(resid)
            else:
                sigma, n_sigma = self._sigma_casewise(resid)
            meat = self._omega_meat(sigma)
            cov = bread @ meat @ bread
        else:
            cov = (rss / (nobs - k)) * bread
        cov = (cov + cov.T) / 2.0

        tss = float(((self._y - self._y.mean()) ** 2).sum())
        r2 = 1.0 - rss / tss if tss > 0.0 else float("nan")
        df_resid = nobs - k
        slopes = [i for i, name in enumerate(self._param_names) if not _is_constant(name)]
        df_m = len(slopes)
        if slopes:
            b_s = beta[slopes]
            try:
                wald = float(b_s @ np.linalg.solve(cov[np.ix_(slopes, slopes)], b_s))
            except np.linalg.LinAlgError:
                wald = float("nan")
        else:  # pragma: no cover
            wald = float("nan")

        e: dict[str, Any] = {
            "N": int(nobs),
            "N_g": int(self._sizes.size),
            "df_m": float(df_m),
            "df_r": float(df_resid),
            "rank": int(k),
            "df": float(df_m),
            "r2": r2,
            "r2_a": float(1.0 - (1.0 - r2) * (nobs - 1) / df_resid)
            if df_resid > 0 and np.isfinite(r2)
            else float("nan"),
            "rmse": float(math.sqrt(rss / df_resid)) if df_resid > 0 else float("nan"),
            "rss": rss,
            "mss": tss - rss,
            "chi2": wald,
            "chi2type": "Wald",
            "p": float(stats.chi2.sf(wald, df_m))
            if df_m and np.isfinite(wald)
            else float("nan"),
            "n_cf": int(k),
            "n_cv": self._n_covariances(),
            "n_cr": 0,
            "g_min": int(self._sizes.min()),
            "g_avg": float(nobs / self._sizes.size),
            "g_max": int(self._sizes.max()),
            "panels": "correlated",
            "corr": "independent",
            "selection": self._selection,
            "rc": 0.0,
            "sum_w": float(nobs),
        }
        if n_sigma is not None:
            # Stata stores the casewise period count and prints it in the header;
            # ``e(missmeth)``/``e(n_sigma)`` are what its own advice to switch to
            # ``pairwise`` is based on.
            e["n_sigma"] = float(n_sigma)
            e["missmeth"] = "casewise"

        return StataResults(
            params=beta,
            param_names=list(self._param_names),
            cov=cov,
            resid=resid,
            fittedvalues=self._X @ beta,
            nobs=int(nobs),
            df_resid=float(df_resid),
            df_model=float(df_m),
            dist="z",
            model_name=self._stata_command,
            cov_type=canonical,
            e=e,
            design_matrix=(self._X, list(self._param_names)),
            model_data=self._model_data,
        )

    # -------------------------------------------------------------- internals
    def _panel_periods(self) -> np.ndarray | None:
        return self._periods

    def _period_values(self) -> np.ndarray:
        """The period identifier per row, inferred from row order when needed.

        ``time`` is optional in this adapter (``xtpcse`` itself needs ``tis`` in
        Stata, but a westata caller who only declared the entity should not have
        to repeat it for a balanced panel).  When it is omitted, a balanced panel
        is read positionally - the first observation of every entity is period
        ``0``, the second period ``1``, and so on, which is exactly what the
        row-ordered panel contains.  An unbalanced panel cannot be matched that
        way, so it is rejected with an explicit error instead of silently
        pairing the wrong observations.
        """

        if self._periods is not None:
            return self._periods
        slices = _group_slices(self._codes)
        lengths = {int(rows.size) for rows in slices}
        if len(lengths) != 1:
            raise ConsistencyError(
                "xtpcse: the panel is unbalanced but no ``time`` column was "
                "given, so the periods cannot be matched across panels; pass "
                "time=<column> (Stata requires tis in this case too)"
            )
        out = np.empty(self._y.size, dtype=int)
        for rows in slices:
            out[rows] = np.arange(rows.size)
        return out

    def _common_periods(self) -> list[Any]:
        """Periods observed by *every* panel (Stata's ``casewise`` set)."""

        periods = self._period_values()
        groups: dict[int, set[Any]] = {}
        for code, period in zip(self._codes, periods):
            groups.setdefault(int(code), set()).add(period)
        common = set.intersection(*groups.values()) if groups else set()
        return sorted(common)

    def _sigma_casewise(self, resid: np.ndarray) -> tuple[np.ndarray, int]:
        """``e(Sigma)`` with Stata's ``casewise`` selection.

        ``xtpcse.ado::getSigma`` with ``casewise`` keeps only the periods common
        to every panel and divides the cross-products by their count
        ``n_sigma``.  That count is what Stata reports as ``e(n_sigma)`` and uses
        to warn when it falls below half of ``e(g_avg)``.
        """

        common = self._common_periods()
        if not common:
            raise ConsistencyError(
                "xtpcse: no time periods are common to all panels, so the "
                "disturbance covariance matrix cannot be estimated using "
                "casewise inclusion (Stata's r(459)); use selection='pairwise'"
            )
        panel_periods = self._ordered_panel_periods()
        m = int(self._sizes.size)
        sigma = np.zeros((m, m))
        for i in range(m):
            rows_i = panel_periods[i]
            for j in range(i, m):
                rows_j = panel_periods[j]
                total = 0.0
                for period in common:
                    total += float(
                        resid[rows_i[period]] * resid[rows_j[period]]
                    )
                sigma[i, j] = sigma[j, i] = total / float(len(common))
        return sigma, len(common)

    def _sigma_pairwise(self, resid: np.ndarray) -> np.ndarray:
        """``e(Sigma)`` with Stata's ``pairwise`` selection.

        Each ``(i, j)`` element averages over exactly the periods those two
        panels share, so ``T_ij`` varies by pair; when a pair shares no period
        Stata assumes the covariance is zero (and warns once).
        """

        panel_periods = self._ordered_panel_periods()
        m = int(self._sizes.size)
        sigma = np.zeros((m, m))
        for i in range(m):
            for j in range(i, m):
                shared = sorted(set(panel_periods[i]) & set(panel_periods[j]))
                if not shared:
                    sigma[i, j] = sigma[j, i] = 0.0
                    continue
                total = sum(
                    float(resid[panel_periods[i][p]] * resid[panel_periods[j][p]])
                    for p in shared
                )
                sigma[i, j] = sigma[j, i] = total / float(len(shared))
        return sigma

    def _ordered_panel_periods(self) -> list[dict[Any, int]]:
        """``[{period: row}]`` per panel, in ascending panel-code order."""

        periods = self._period_values()
        mapping: list[dict[Any, int]] = [dict() for _ in range(int(self._codes.max()) + 1)]
        for row, (code, period) in enumerate(zip(self._codes, periods)):
            mapping[int(code)][period] = int(row)
        return mapping

    def _omega_meat(self, sigma: np.ndarray) -> np.ndarray:
        """``X' Omega X`` accumulated the way Stata's ``glsaccum`` does.

        ``xtpcse.ado`` calls ``matrix glsaccum XomegaX = varlist,
        glsmat(Sigma) row(ivar) group(timevar)``, i.e. the design is grouped by
        *period*::

            X'Omega X = sum_t  X_t'  Sigma[rows present at t, rows present at t] X_t

        which equals ``sum_{i,j} sigma_ij X_i'X_j`` only when the panel is
        balanced.  Writing it by period is what keeps unbalanced panels right.
        """

        periods = self._period_values()
        k = self._X.shape[1]
        total = np.zeros((k, k))
        for period in sorted(set(periods.tolist())):
            rows = np.flatnonzero(periods == period)
            codes = self._codes[rows]
            block = sigma[np.ix_(codes, codes)]
            total += self._X[rows].T @ block @ self._X[rows]
        return total

    def _n_covariances(self) -> int:
        m = int(self._sizes.size)
        return m * (m + 1) // 2


def _pcse_cov_label(cov_type: str | None) -> str:
    if cov_type is None:
        return "pcse"
    key = str(cov_type).strip().lower()
    if key in ("pcse", "panel", "panel-corrected"):
        return "pcse"
    if key in ("nonrobust", "unadjusted", "conventional", "ols"):
        return "nonrobust"
    raise ValueError(
        f"xtpcse: unknown cov_type {cov_type!r}; use 'pcse' (the default) or "
        "'nonrobust' for the OLS covariance"
    )


# ---------------------------------------------------------------- xtunitroot
#: IPS (2003) table 3 moment pairs ``(mean, variance)`` of the unadjusted ``t_i``,
#: indexed by panel length ``T``.  Transcribed from ``_xturips.ado::_ipstab1``,
#: which is the table Stata interpolates.
_IPS_MOMENTS: dict[int, tuple[float, float]] = {
    6: (-1.520, 1.745),
    7: (-1.514, 1.414),
    8: (-1.501, 1.228),
    9: (-1.501, 1.132),
    10: (-1.504, 1.069),
    15: (-1.514, 0.923),
    20: (-1.522, 0.851),
    25: (-1.520, 0.809),
    30: (-1.526, 0.789),
    40: (-1.523, 0.770),
    50: (-1.527, 0.760),
    100: (-1.532, 0.735),
    500: (-1.531, 0.715),
    1000: (-1.529, 0.707),
}

#: IPS (2003) table 3 moment pairs for the ``t-tilde`` statistic (no lag
#: augmentation, panel means included).  Same source as :data:`_IPS_MOMENTS`.
_IPS_TILDE_MOMENTS: dict[int, tuple[float, float]] = {
    6: (-1.125, 0.497),
    7: (-1.178, 0.506),
    8: (-1.214, 0.506),
    9: (-1.244, 0.527),
    10: (-1.274, 0.521),
    15: (-1.349, 0.565),
    20: (-1.395, 0.592),
    25: (-1.423, 0.609),
    30: (-1.439, 0.623),
    40: (-1.463, 0.639),
    50: (-1.477, 0.656),
    100: (-1.504, 0.683),
    500: (-1.526, 0.704),
    1000: (-1.526, 0.702),
}

#: LLC (2002) table 2 finite-sample adjustment pairs ``(mu, sigma)`` for model 2
#: (constant, no trend), indexed by ``T``.  From ``_xturllc.ado::_get_adj``.
_LLC_ADJUST: dict[int, tuple[float, float]] = {
    25: (-0.554, 0.919),
    30: (-0.546, 0.889),
    35: (-0.541, 0.867),
    40: (-0.537, 0.850),
    45: (-0.533, 0.837),
    50: (-0.531, 0.826),
    60: (-0.527, 0.810),
    70: (-0.524, 0.798),
    80: (-0.521, 0.789),
    90: (-0.520, 0.782),
    100: (-0.518, 0.776),
    250: (-0.509, 0.742),
}


def _interpolate(table: dict[int, tuple[float, float]], key: float) -> tuple[float, float]:
    """Linear interpolation on the table key, clamped at both ends.

    ``_ipstab1`` interpolates linearly between grid points and takes the nearest
    row outside them; the clamping here reproduces that outside the grid.
    """

    keys = sorted(table)
    if key <= keys[0]:
        return table[keys[0]]
    if key >= keys[-1]:
        return table[keys[-1]]
    for lower, upper in zip(keys, keys[1:]):
        if lower <= key <= upper:
            weight = (key - lower) / (upper - lower)
            lo = table[lower]
            hi = table[upper]
            return (
                float(lo[0] + weight * (hi[0] - lo[0])),
                float(lo[1] + weight * (hi[1] - lo[1])),
            )
    return table[keys[-1]]  # pragma: no cover - unreachable


@dataclass(frozen=True)
class PanelUnitRootResult:
    """A panel unit-root statistic together with its p-value.

    Stata's ``xtunitroot`` stores these in ``r()``, not ``e()``: the command is an
    r-class test, not an estimation.  Returning a ``StataResults`` would mean
    inventing an ``e(b)`` and an ``e(V)`` that Stata never produces, so this small
    dataclass is the honest container.

    Attributes
    ----------
    statistic:
        The statistic carrying the p-value (``Z-t-tilde-bar`` for IPS,
        ``Adjusted t*`` for LLC).
    pvalue:
        Lower-tail standard-normal p-value of ``statistic``.
    test:
        ``"ips"`` or ``"llc"``.
    name:
        Stata's label for the reported statistic.
    details:
        The other statistics Stata prints in the same table, so a caller can
        reproduce the whole table rather than one line of it.
    """

    statistic: float
    pvalue: float
    test: str
    name: str = ""
    details: Mapping[str, float] | None = None

    def __str__(self) -> str:
        label = self.name or self.test
        return f"{label}: statistic {self.statistic:.4f}, p {self.pvalue:.4f}"


class XtUnitroot:
    """Panel unit-root tests (Stata ``xtunitroot ips`` / ``xtunitroot llc``).

    Both are panel-wide Dickey-Fuller procedures, and both match Stata on the
    fixture data because the look-up tables are transcribed rather than re-derived.

    ``ips`` (Im-Pesaran-Shin; balanced, no lags by default)
        Per panel, ``Delta y_it`` is regressed on ``y_i,t-1`` and a panel constant,
        giving ``t_i``.  Stata reports three statistics:

        * ``t-bar = mean(t_i)`` -- exact finite-``N`` tabulated critical values, and
          no p-value;
        * ``t-tilde-bar = mean(t_i * s_i / sigma(y_i))``, where ``s_i`` is the ADF
          residual RMSE -- Stata's small-``T`` variant;
        * ``Z-t-tilde-bar = sqrt(N)(t-tilde-bar - E)/sqrt(V)`` with ``E`` and ``V``
          interpolated from IPS (2003) table 3 at ``T`` -- the one with a p-value.

        On the fixture (90 panels x 9 periods) westata reproduces ``t-bar``
        -2.0685, ``t-tilde-bar`` -1.5727 and ``Z`` -4.2953.

    ``llc`` (Levin-Lin-Chu; balanced, one ADF lag by default)
        Stata's default is ``lags(1)`` with a Bartlett kernel whose bandwidth is
        ``int(3.21 * T^(1/3))`` -- 6 at ``T = 9``, which is why the fixture header
        prints "6.00 lags average".  The adjusted statistic is
        ``t* = (t - N*Tbar*sbar/Var_ep * se(t) * mu) / sigma`` with ``(mu, sigma)``
        interpolated from LLC (2002) table 2.  On the fixture westata reproduces
        ``Unadjusted t = -21.7694`` and ``Adjusted t* = -15.1701``.

    Both tests require a strongly balanced panel with no gaps -- Stata's
    requirement too (``_xtstrbal``).  An unbalanced panel raises
    :class:`westata.errors.ConsistencyError` rather than silently using whichever
    rows happen to line up.  ``trend``, ``noconstant``, ``demean`` and
    information-criterion lag selection raise
    :class:`westata.errors.UnsupportedError`.
    """

    _stata_command = "xtunitroot"

    def __init__(
        self,
        values: Any,
        codes: np.ndarray,
        sizes: np.ndarray,
        *,
        test: str = "ips",
        lags: int | None = None,
        trend: bool = False,
        noconstant: bool = False,
        demean: bool = False,
        variable_name: str = "y",
    ) -> None:
        self._values = np.asarray(values, dtype=float).reshape(-1)
        self._codes = np.asarray(codes, dtype=int)
        self._sizes = np.asarray(sizes, dtype=int)
        key = str(test).strip().lower()
        if key not in ("ips", "llc"):
            raise UnsupportedError(
                f"xtunitroot: test={test!r} is not implemented; only 'ips' "
                "(Im-Pesaran-Shin) and 'llc' (Levin-Lin-Chu) are"
            )
        if trend:
            raise UnsupportedError(
                "xtunitroot: trend is not implemented; the deterministic component "
                "changes both the Dickey-Fuller regression and the tabulated moments"
            )
        if noconstant:
            raise UnsupportedError(
                "xtunitroot: noconstant is not implemented; the look-up tables "
                "differ by deterministic specification"
            )
        if demean:
            raise UnsupportedError(
                "xtunitroot: demean (subtracting cross-sectional means) is not "
                "implemented"
            )
        if lags is not None and (
            not isinstance(lags, int) or isinstance(lags, bool) or lags < 0
        ):
            raise UnsupportedError(
                f"xtunitroot: lags={lags!r} is not implemented; pass a "
                "non-negative integer, or None for Stata's documented default "
                "(0 for ips, 1 for llc)"
            )
        self._test = key
        self._lags = lags
        self._name = variable_name

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        entity: str,
        time: str | None = None,
        test: str = "ips",
        lags: int | None = None,
        trend: bool = False,
        noconstant: bool = False,
        demean: bool = False,
    ) -> "XtUnitroot":
        """Build the test from a formula and a panel frame.

        Only the left-hand side matters; ``"y ~ 1"`` and ``"y"`` both name ``y``.
        The panel must be strongly balanced.

        Unlike the other three estimators, ``time`` is **required** here: both the
        IPS and LLC statistics are defined over the period grid (the moment tables
        are indexed by ``T`` and the tests require no gaps), so a call without it
        cannot be interpreted.
        """

        command = "xtunitroot"
        if time is None:
            raise ConsistencyError(
                "xtunitroot: time is required; the panel unit-root statistics are "
                "defined over the period grid and the panel must be strongly "
                "balanced, so the time variable cannot be inferred"
            )
        lhs = formula.split("~", 1)[0].strip()
        if not lhs:
            raise ConsistencyError("xtunitroot: the formula has an empty left-hand side")
        frame = _as_frame(data, entity=entity, time=time, command=command)
        if lhs not in frame.columns:
            raise ConsistencyError(f"xtunitroot: column {lhs!r} not found in the data")
        frame = _listwise(frame, [lhs], command)
        codes, sizes = _panel_labels(frame, entity, command)
        if int(sizes.min()) != int(sizes.max()):
            raise ConsistencyError(
                "xtunitroot: the panel must be strongly balanced; observed panel "
                f"sizes range from {int(sizes.min())} to {int(sizes.max())}"
            )
        periods = frame.groupby(entity, sort=True)[time].apply(
            lambda values: np.sort(np.asarray(values))
        )
        reference = np.asarray(periods.iloc[0])
        for other in periods.iloc[1:]:
            if not np.array_equal(reference, np.asarray(other)):
                raise ConsistencyError(
                    "xtunitroot: the panel must be strongly balanced with no gaps; "
                    "at least two panels observe different periods"
                )
        return cls(
            frame[lhs].to_numpy(dtype=float),
            codes,
            sizes,
            test=test,
            lags=lags,
            trend=trend,
            noconstant=noconstant,
            demean=demean,
            variable_name=lhs,
        )

    # ------------------------------------------------------------------- fit
    def fit(self, **_ignored: Any) -> PanelUnitRootResult:
        """Run the requested test and return its statistics."""

        if self._test == "ips":
            return self._ips()
        return self._llc()

    # -------------------------------------------------------------- internals
    def _adf_t(self, series: np.ndarray, lags: int) -> tuple[float, float]:
        """Per-panel ADF ``(t on the lagged level, residual RMSE)``.

        The regression is ``Delta y_t`` on ``y_{t-1}``, ``lags`` lagged differences
        and a constant -- Stata's ``_xturips`` specification.  ``s^2 =
        e'e/(n - k)`` is the degrees-of-freedom-corrected residual variance, which
        is what ``regress``'s standard error uses.
        """

        difference = np.diff(series)
        level = series[:-1]
        if difference.size <= lags + 2:
            raise ConsistencyError(
                f"xtunitroot: a panel has only {series.size} observations, too few "
                f"for {lags} lag(s) plus a constant; the series are too short"
            )
        target = difference[lags:]
        columns = [level[lags:]]
        for lag in range(1, lags + 1):
            columns.append(difference[lags - lag : difference.size - lag])
        columns.append(np.ones(target.size))
        X = np.column_stack(columns)
        if np.linalg.matrix_rank(X) < X.shape[1]:
            raise ConsistencyError(
                "xtunitroot: a panel's ADF regressor matrix is rank deficient"
            )
        beta = _ols(X, target)
        resid = target - X @ beta
        df_resid = target.size - X.shape[1]
        if df_resid <= 0:  # pragma: no cover - guarded above
            raise ConsistencyError(
                "xtunitroot: a panel has no residual degrees of freedom in its ADF "
                "regression"
            )
        sigma2 = float(resid @ resid) / df_resid
        standard_error = math.sqrt(float(sigma2 * np.linalg.inv(X.T @ X)[0, 0]))
        if standard_error <= 0.0 or not np.isfinite(standard_error):
            raise ConsistencyError(
                "xtunitroot: a panel produced a zero ADF standard error; the lagged "
                "level is collinear with the other regressors"
            )
        return float(beta[0] / standard_error), math.sqrt(sigma2)

    def _ips(self) -> PanelUnitRootResult:
        """Im-Pesaran-Shin test with Stata's default zero ADF lags."""

        lags = 0 if self._lags is None else int(self._lags)
        t_values: list[float] = []
        t_tilde_values: list[float] = []
        for index in _group_slices(self._codes):
            series = self._values[index]
            t_stat, rmse = self._adf_t(series, lags)
            spread = float(np.std(np.diff(series), ddof=1))
            if spread <= 0.0:
                raise ConsistencyError(
                    "xtunitroot: a panel's differenced series is constant, so the "
                    "t-tilde rescaling is undefined"
                )
            t_values.append(t_stat)
            t_tilde_values.append(t_stat * rmse / spread)

        t_array = np.asarray(t_values, dtype=float)
        tilde_array = np.asarray(t_tilde_values, dtype=float)
        n_panels = t_array.size
        # Both moment tables are looked up at the panel length, not at the number of
        # ADF rows: on T=9 the t-tilde row is the T=9 one (verified -- using T=8
        # gives Z = -4.7836 where Stata prints -4.2953).
        length = float(self._sizes[0])
        mu, var = _interpolate(_IPS_MOMENTS, length)
        mu_tilde, var_tilde = _interpolate(_IPS_TILDE_MOMENTS, length)
        if var_tilde <= 0.0 or var <= 0.0:  # pragma: no cover - table is positive
            raise ConsistencyError(
                "xtunitroot: the IPS variance table returned a non-positive "
                "variance, so the standardised statistic is undefined"
            )
        t_bar = float(t_array.mean())
        t_tilde_bar = float(tilde_array.mean())
        z_tilde = math.sqrt(n_panels) * (t_tilde_bar - mu_tilde) / math.sqrt(var_tilde)
        return PanelUnitRootResult(
            statistic=float(z_tilde),
            pvalue=float(stats.norm.cdf(z_tilde)),
            test="ips",
            name="Z-t-tilde-bar",
            details={
                "t-bar": t_bar,
                "t-tilde-bar": t_tilde_bar,
                "Z-t-bar": float(
                    math.sqrt(n_panels) * (t_bar - mu) / math.sqrt(var)
                ),
                "N_t": length,
                "lags": float(lags),
            },
        )

    def _llc(self) -> PanelUnitRootResult:
        """Levin-Lin-Chu test with Stata's default one ADF lag."""

        lags = 1 if self._lags is None else int(self._lags)
        e_parts: list[np.ndarray] = []
        nu_parts: list[np.ndarray] = []
        s_parts: list[float] = []
        nobs_parts: list[float] = []

        for index in _group_slices(self._codes):
            series = self._values[index]
            length = float(series.size)
            difference = np.diff(series)
            level = series[:-1]
            if difference.size <= lags + 2:
                raise ConsistencyError(
                    f"xtunitroot: a panel has only {series.size} observations, too "
                    f"few for {lags} lag(s) plus a constant; the series are too short"
                )
            target = difference[lags:]
            lag_block = np.column_stack(
                [difference[lags - lag : difference.size - lag] for lag in range(1, lags + 1)]
                + [np.ones(target.size)]
            )
            if np.linalg.matrix_rank(lag_block) < lag_block.shape[1]:
                raise ConsistencyError("xtunitroot: a panel's LLC lag block is rank deficient")

            # (1) ADF regression of Delta y on y_{t-1}, lagged differences and a
            #     panel constant; delta is the coefficient on the lagged level.
            design = np.column_stack([level[lags:], lag_block])
            if np.linalg.matrix_rank(design) < design.shape[1]:
                raise ConsistencyError(
                    "xtunitroot: a panel's LLC ADF regressor matrix is rank deficient"
                )
            delta = float(_ols(design, target)[0])

            # (2) Orthogonalised residuals: project both Delta y and y_{t-1} on the
            #     lagged differences *and the constant*, then e = y - delta*nu.
            #     Including the constant here is what reproduces Stata's
            #     r(delta) = -0.8388 on the fixture; without it, -0.2947.
            residual_e = target - lag_block @ _ols(lag_block, target)
            residual_nu = level[lags:] - lag_block @ _ols(lag_block, level[lags:])
            combined = residual_e - delta * residual_nu
            # _pregress divides the pooled rss by capT - p - 1 with capT the panel
            # length (one observation is lost to differencing).
            df_panel = length - lags - 1.0
            if df_panel <= 0.0:
                raise ConsistencyError(
                    f"xtunitroot: a panel has only {series.size} observations, so "
                    "the LLC residual variance has no degrees of freedom"
                )
            sigma2_panel = float(combined @ combined) / df_panel
            if sigma2_panel <= 0.0:
                raise ConsistencyError("xtunitroot: a panel's LLC residual variance is zero")
            scale = math.sqrt(sigma2_panel)
            e_parts.append(residual_e / scale)
            nu_parts.append(residual_nu / scale)

            # (3) Long-run/short-run variance ratio, Bartlett kernel, bandwidth
            #     int(3.21 * T^(1/3)) -- the LLC rule.
            w = difference - difference.mean()
            bandwidth = int(3.21 * w.size ** (1.0 / 3.0))
            acc = float(w @ w)
            for lag in range(1, bandwidth + 1):
                acc += 2.0 * (1.0 - lag / (bandwidth + 1.0)) * float(w[lag:] @ w[:-lag])
            long_run = acc / w.size
            s_parts.append(math.sqrt(long_run / sigma2_panel))
            nobs_parts.append(float(target.size))

        e = np.concatenate(e_parts)
        nu = np.concatenate(nu_parts)
        s_array = np.asarray(s_parts, dtype=float)
        nobs_array = np.asarray(nobs_parts, dtype=float)

        nu_nu = float(nu @ nu)
        if nu_nu <= 0.0:
            raise ConsistencyError("xtunitroot: the LLC normalised regressor has zero variance")
        delta = float(nu @ e) / nu_nu
        resid = e - delta * nu
        var_ep = float(resid @ resid) / e.size
        s_bar = float(s_array.mean())
        n_t = float(nobs_array.sum())
        # Stata's _xturllc applies no (N-1)/N rescaling to this standard error:
        # se = sqrt(Var_ep / nu'nu).  Adding the rescaling moves t from -21.7694 to
        # -21.7867, so the plain form is the one Stata uses.
        standard_error = math.sqrt(var_ep / nu_nu)
        t_delta = delta / standard_error

        mu, sigma_adj = _interpolate(_LLC_ADJUST, float(self._sizes[0]))
        t_star = (t_delta - n_t * s_bar / var_ep * standard_error * mu) / sigma_adj

        return PanelUnitRootResult(
            statistic=float(t_star),
            pvalue=float(stats.norm.cdf(t_star)),
            test="llc",
            name="Adjusted t*",
            details={
                "Unadjusted t": t_delta,
                "delta": delta,
                "se": standard_error,
                "Var_ep": var_ep,
                "sbar": s_bar,
                "mu_adj": mu,
                "sig_adj": sigma_adj,
                "N_t": float(self._sizes[0]),
                "lags": float(lags),
                "hac_lags": float(int(3.21 * (self._sizes[0] - 1) ** (1.0 / 3.0))),
            },
        )
