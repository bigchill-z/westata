"""Discrete-choice and count estimators targeting Stata's ML commands.

=======================  ==================
westata                  Stata
=======================  ==================
``Logit``                ``logit``
``Probit``               ``probit``
``Poisson``              ``poisson``
``NegativeBinomial``     ``nbreg``
``OrderedLogit``         ``ologit``
``OrderedProbit``        ``oprobit``
=======================  ==================

Conformance notes
-----------------
* All of these report **z** statistics in Stata (``P>|z|``), unlike ``regress``,
  so ``StataResults`` is built with ``dist="z"`` and the summary prints
  ``Wald chi2`` rather than an F test.
* ``e(ll)`` is the log-likelihood at the estimates.
* ``e(r2_p)`` is McFadden's pseudo R-squared, which is what Stata prints as
  ``Pseudo R2``.
* ``e(df_m)`` counts only the *structural* regressors beyond the constant.
  Auxiliary parameters -- ``/lnalpha`` for ``nbreg``, ``/cut1``... for
  ``ologit``/``oprobit`` -- are not regressors; Stata reports them separately
  through ``e(k_aux)``.  ``nbreg y x1 x2`` therefore prints ``LR chi2(2)`` with
  ``e(rank) = 4`` and ``e(k_aux) = 1``, and ``ologit`` with three cut points
  reports ``df_m = 2``, ``rank = 5``, ``k_aux = 3``.
* ``e(chi2)``/``e(chi2type)`` follow the VCE.  Stata reports the
  likelihood-ratio statistic against the constant-only model under the default
  observed information, and switches to a **slope-block Wald**
  ``b_s' V_ss^-1 b_s`` (constant and auxiliary parameters excluded) as soon as
  ``vce(robust)`` or ``vce(cluster g)`` is requested.  Verified against
  StataNow 19.5 to ~1e-16 by recomputing the statistic from the ``e(V)`` that
  Stata itself reported.

VCE semantics (the part that used to be wrong)
----------------------------------------------
Stata's ``ml`` estimator has its **own** small-sample conventions, and they are
*not* the ones ``regress`` uses:

===============  ==================================================================
``cov_type``      meaning here
===============  ==================================================================
``nonrobust``     observed information ``(-H)^-1``; Stata's default.
``robust``        Stata's ``vce(robust)``: ``(N/(N-1)) * B M B`` with
                  ``B = (-H)^-1`` and ``M = sum_i s_i s_i'``.
``cluster``       Stata's ``vce(cluster g)``: ``(G/(G-1)) * B M_g B`` with
                  ``M_g = sum_g (sum_{i in g} s_i)(...)'`` -- note there is **no**
                  ``(N-1)/(N-K)`` factor, unlike ``regress``'s CR1.
``HC0``           statsmodels' convention: ``B M B``, *no* small-sample factor.
                  **This is not Stata's ``vce(robust)``.**
``HC1``           statsmodels' convention: ``(N/(N-K)) * B M B``.  **Also not
                  Stata's ``vce(robust)``**; it coincides with Stata only for
                  linear models, where it is ``regress, vce(robust)``.
===============  ==================================================================

``HC0``/``HC1`` are kept because they are statsmodels-compatible and other
tests rely on them, but "Stata's robust" is ``cov_type="robust"``.  Both are
computed here rather than delegated because statsmodels 0.15 returns *the same
matrix* for the two on these models: its discrete results take the HC score
through ``cov_white_simple(use_correction=False)`` and drop ``cov_kwds``, so no
``N/(N-K)`` factor ever reaches the matrix.  Silently answering ``HC0`` to an
``HC1`` request is exactly the failure mode this module refuses.

Every formula above was checked against a live StataNow 19.5 session by
computing the sandwich at Stata's own estimates: ``logit``/``probit``/
``poisson``/``ologit``/``oprobit`` reproduce ``e(V)`` to ~1e-15 and ``nbreg`` to
~7e-12 (its ``e(V)`` is the only place where the 1e-9 target is missed, because
statsmodels' ``NegativeBinomial`` Hessian is finite-difference-free but the
score is not bit-identical to Stata's -- see ``retest/fix/discrete/FIXES.md``).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from westata.errors import ConsistencyError, ConvergenceError
from westata.formula.factor import build_design
from westata.results.base import StataResults

__all__ = [
    "Logit",
    "Probit",
    "Poisson",
    "NegativeBinomial",
    "OrderedLogit",
    "OrderedProbit",
]

#: Accepted ``cov_type`` values.  ``robust``/``cluster`` are Stata's VCEs;
#: ``HC0``/``HC1`` are statsmodels' (kept for API compatibility).
_SUPPORTED_COV = ("nonrobust", "robust", "HC0", "HC1", "cluster")


# --------------------------------------------------------------- VCE helpers
def _cluster_groups(cov_kwds: Mapping[str, Any]) -> np.ndarray:
    """Pull the cluster id out of ``cov_kwds`` or explain how to pass it."""

    groups = dict(cov_kwds or {}).get("groups")
    if groups is None:
        raise ValueError(
            "cov_type='cluster' requires cov_kwds={'groups': <cluster id>}"
        )
    values = np.asarray(groups).reshape(-1)
    if values.size == 0:
        raise ValueError("cov_kwds['groups'] is empty")
    return values


def _sandwich(bread: np.ndarray, meat: np.ndarray, factor: float) -> np.ndarray:
    """``factor * bread @ meat @ bread``, symmetrised for downstream use."""

    value = factor * (bread @ meat @ bread)
    return (value + value.T) / 2.0


def _cluster_meat(score: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, int]:
    """``sum_g (sum_{i in g} s_i)(sum_{i in g} s_i)'`` together with ``G``."""

    labels = np.unique(groups)
    meat = np.zeros((score.shape[1], score.shape[1]))
    for value in labels:
        total = score[groups == value].sum(axis=0)
        meat += np.outer(total, total)
    return meat, int(labels.size)


def _observed_information(model: Any, params: np.ndarray) -> np.ndarray:
    return -np.asarray(model.hessian(params), dtype=float)


def _vce_from_derivatives(
    bread: np.ndarray,
    score: np.ndarray,
    kind: str,
    groups: np.ndarray | None,
) -> np.ndarray:
    """Assemble the covariance matrix from ``bread`` and the score.

    ``kind`` is one of ``nonrobust``, ``robust``, ``HC0``, ``HC1``, ``cluster``.
    """

    kind = kind.lower()
    if kind == "nonrobust":
        return (bread + bread.T) / 2.0

    n_obs, n_params = score.shape
    if kind in ("hc0", "hc1"):
        factor = 1.0 if kind == "hc0" else n_obs / float(n_obs - n_params)
        return _sandwich(bread, score.T @ score, factor)
    if kind == "robust":
        return _sandwich(bread, score.T @ score, n_obs / float(n_obs - 1.0))
    if kind == "cluster":
        if groups is None:  # pragma: no cover - callers validate first
            raise ValueError("cov_type='cluster' requires cov_kwds={'groups': ...}")
        meat, n_groups = _cluster_meat(score, groups)
        if n_groups < 2:
            raise ConsistencyError(
                "cov_type='cluster' needs at least two clusters; the grouping "
                f"variable has {n_groups}"
            )
        # Stata's ml vce(cluster g) carries G/(G-1) only -- deliberately no
        # (N-1)/(N-K), which is what regress's CR1 adds.
        return _sandwich(bread, meat, n_groups / float(n_groups - 1.0))
    raise ValueError(f"unsupported cov_type {kind!r}")  # pragma: no cover


def _inverse_information(information: np.ndarray, command: str) -> np.ndarray:
    try:
        return np.linalg.inv(information)
    except np.linalg.LinAlgError:
        raise ConvergenceError(
            f"{command}: the observed information matrix is singular, so the "
            "covariance matrix is not defined; this normally means the model is "
            "not identified (perfect separation, a collinear regressor, or "
            "unidentified cut points)"
        ) from None


# --------------------------------------------------------------- Wald helpers
def _wald_statistic(
    params: np.ndarray, cov: np.ndarray, positions: list[int]
) -> float:
    """``b_s' V_ss^-1 b_s`` over ``positions`` (NaN when empty)."""

    if not positions:
        return float("nan")
    index = np.asarray(positions, dtype=int)
    beta = np.asarray(params, dtype=float)[index]
    block = np.asarray(cov, dtype=float)[np.ix_(index, index)]
    try:
        solved = np.linalg.solve(block, beta)
    except np.linalg.LinAlgError:
        solved = np.linalg.pinv(block) @ beta
    with np.errstate(invalid="ignore"):
        statistic = float(beta @ solved)
    return statistic if np.isfinite(statistic) and statistic >= 0 else float("nan")


def _slope_positions(names: list[str]) -> list[int]:
    """Indices of the structural slope coefficients.

    Stata's Wald test under ``vce(robust)``/``vce(cluster)`` covers the
    regressors of the structural equation, excluding the constant and every
    auxiliary parameter (``/lnalpha``, ``/cut1``..., ``/sigma``).  Names
    starting with ``/`` are auxiliary by construction; the constant is spelled
    ``Intercept`` or ``_cons``.
    """

    return [
        position
        for position, name in enumerate(names)
        if not name.startswith("/") and name not in ("Intercept", "_cons", "const")
    ]


def _ml_vce_label(cov_type: Any) -> str:
    """``e(vce)`` for the ML families -- the lowercase token.

    Delegates to the shared helper in ``linear`` with ``ml=True`` so the ML default is
    ``oim`` rather than ``ols``.  Imported locally because ``linear`` imports from this
    module's siblings.
    """

    from westata.estimators.linear import _stata_vce_label

    return _stata_vce_label(cov_type, ml=True)


def _ml_vcetype_label(cov_type: Any) -> str:
    """``e(vcetype)`` for the ML families -- the capitalised display title (``"OIM"``)."""

    from westata.estimators.linear import _stata_vcetype_label

    return _stata_vcetype_label(cov_type, ml=True)


def _command_predict_program(command: str) -> str:
    """``e(predict)`` -- the program that answers ``predict`` for this command.

    Transcribed from each ado.  Note two names are TRUNCATED to Stata's 8-character
    limit for program names, which is easy to get wrong by guessing:

    ============  ===============  ===============================
    command       ``e(predict)``   ado line
    ============  ===============  ===============================
    ``logit``     ``logit_p``      ``logit.ado:266``
    ``probit``    ``probit_p``     ``probit.ado:258``
    ``poisson``   ``poisso_p``     ``poisson.ado`` (NOT poisson_p)
    ``nbreg``     ``nbreg_p``      ``nbreg.ado``
    ``ologit``    ``ologit_p``     ``ologit.ado``
    ``oprobit``   ``oprobi_p``     ``oprobit.ado`` (NOT oprobit_p)
    ============  ===============  ===============================

    An unknown command gets the empty string rather than a name built by appending ``_p``,
    because that construction produces the wrong answer for exactly the two commands above.
    """

    programs = {
        "logit": "logit_p",
        "logistic": "logit_p",
        "probit": "probit_p",
        "dprobit": "probit_p",
        "poisson": "poisso_p",
        "nbreg": "nbreg_p",
        "ologit": "ologit_p",
        "oprobit": "oprobi_p",
    }
    return programs.get(command.strip().lower(), "")


def _command_title(command: str) -> str:
    """``e(title)`` per command, from whichever route its ado uses.

    ``logit.ado:253`` / ``probit.ado:252`` set it with a literal ``ereturn local title``;
    ``poisson.ado:241`` hands it to ``ml`` as ``title(Poisson regression)``.  Either way
    Stata ends up with the same macro, and ``*.sthlp`` documents ``e(title)`` for all
    three -- so the value is recorded here regardless of how the ado supplies it.

    An unknown command gets the empty string, which is Stata's "set but blank" form rather
    than a guess.
    """

    titles = {
        "logit": "Logistic regression",
        "logistic": "Logistic regression",
        "probit": "Probit regression",
        "dprobit": "Probit regression",
        "poisson": "Poisson regression",
        # ``nbreg.ado:71``: ``local title "Negative binomial regression"``, handed to
        # ``ml`` as ``title(`title')`` at ``nbreg.ado:352`` -- the same indirect route
        # poisson uses, so its ado has no ``ereturn local title`` either.
        "nbreg": "Negative binomial regression",
        # ``ologit.ado:270`` / ``oprobit.ado:267`` -- set DIRECTLY with
        # ``ereturn local title``, unlike the two above.
        "ologit": "Ordered logistic regression",
        "oprobit": "Ordered probit regression",
        # ``zip.sthlp:25``, ``zinb.sthlp:25``, ``ztp.sthlp``: the banner each command
        # prints. Verified against the help files rather than inferred from the name --
        # ``zinb`` has no ``ereturn local title`` in its ado, so the sthlp header is the
        # authority.
        "zip": "Zero-inflated Poisson regression",
        "zinb": "Zero-inflated negative binomial regression",
        "ztp": "Zero-truncated Poisson regression",
        "tnbreg": "Truncated negative binomial regression",
    }
    return titles.get(command.strip().lower(), "")


def _iteration_count(result: Any) -> float:
    """``e(ic)`` -- the number of iterations the optimizer took (``logit.sthlp:257``).

    Stata counts iterations, not parameters.  The backend records how many it ran in
    ``mle_retvals["iterations"]``; where that is unavailable the value is missing rather
    than invented, because a wrong iteration count is worse than an absent one.
    """

    retvals = getattr(result, "mle_retvals", None)
    if isinstance(retvals, dict) and "iterations" in retvals:
        try:
            return float(retvals["iterations"])
        except (TypeError, ValueError):
            return float("nan")
    return float("nan")


def _gradient_vector_removed() -> None:  # pragma: no cover
    """``e(gradient)`` was attempted and withdrawn -- see this module's notes.

    ``logit.sthlp:291`` documents ``e(gradient)`` as "gradient vector".  The obvious
    construction is the score summed over observations, i.e. ``sum_i (y_i - mu_i) x_i``,
    which is zero at the reported estimate.

    An implementation was written and produced values of order 100 instead of 0, which
    means the arrays it reached (``fittedvalues``/``endog`` on the backend result) are not
    the estimation-sample probability and response in the order assumed.  Rather than ship
    a key whose values cannot be checked, it was withdrawn; ``e(offset)`` is the part of
    that block that could be added safely.
    """

    return None


def _chi2_pvalue(statistic: float, df: int) -> float:
    from scipy import stats

    if not np.isfinite(statistic) or df <= 0:
        return float("nan")
    return float(stats.chi2.sf(statistic, df))


def _richardson_hessian(
    loglike: Any, params: np.ndarray, *, step: float = 4e-4, ratio: float = 2.0
) -> np.ndarray:
    """Hessian of ``loglike`` by one Richardson extrapolation of central differences.

    Plain finite differences here are the accuracy floor.  The statsmodels
    ``GenericLikelihoodModel`` used by :mod:`westata.estimators.counts` has no
    analytic Hessian, and ``approx_hess``'s single central difference leaves the
    count model's ``e(V)`` about 2e-4 (zip), 4e-7 (ztp) away from Stata --
    C tier, on a quantity the BRIEF wants at 1e-9.  Cancelling the leading
    ``h^2`` term::

        H(h) = H + c h^2 + O(h^4)
        H    = (ratio^2 H(h/ratio) - H(h)) / (ratio^2 - 1)

    takes zip to ~1e-7 and ztp to ~2e-8.  The step is deliberately *large*
    (4e-4): the truncation error falls with h while the round-off error grows as
    eps/h^2, and at 1e-4 the round-off already dominates (measured: 6e-6 vs
    1e-7).  This is a numerical fallback, not a claim of analytic accuracy --
    see ``FIXES.md`` for the residual and its cause.
    """

    from statsmodels.tools.numdiff import approx_hess

    coarse = approx_hess(params, loglike, epsilon=step)
    fine = approx_hess(params, loglike, epsilon=step / ratio)
    squared = ratio**2
    hessian = (squared * fine - coarse) / (squared - 1.0)
    return (hessian + hessian.T) / 2.0


def _model_test(
    params: np.ndarray,
    cov: np.ndarray,
    names: list[str],
    llf: float,
    ll_null: float,
    robust: bool,
) -> tuple[str, float, int]:
    """Return ``(chi2type, statistic, df)`` for Stata's printed model test.

    ``robust`` stands for "not the observed information", which is exactly when
    Stata switches the printed test from ``LR chi2`` to ``Wald chi2``.
    """

    positions = _slope_positions(names)
    if robust:
        return "Wald", _wald_statistic(params, cov, positions), len(positions)
    if not np.isfinite(ll_null):
        return "LR", float("nan"), len(positions)
    statistic = 2.0 * (llf - ll_null)
    # A likelihood-ratio statistic cannot be negative: it would mean the
    # restricted model fits better than the unrestricted one, which only happens
    # when the numerical optimum is not the true maximum.  Stata clamps this
    # rather than printing a negative chi-square -- ``nbreg.ado:378-384``
    # reports ``chi2_c = 0`` when ``e(ll) < e(ll_c)``.  The same guard belongs
    # here, and it is what catches a non-converged backend instead of letting a
    # negative statistic through to ``e(chi2)``.
    if statistic < 0.0:
        return "LR", 0.0, len(positions)
    return "LR", statistic, len(positions)


class _DiscreteBase:
    """Shared plumbing for statsmodels-backed ML estimators."""

    _stata_command = ""
    _model_name = ""
    #: statsmodels optimiser.  ``newton`` matches Stata's ml default and is
    #: fastest for logit/probit, but ``nbreg`` needs a quasi-Newton method
    #: because its alpha parameter makes the Newton step diverge from a
    #: Poisson start.
    _optimizer = "newton"
    #: Number of auxiliary (non-regressor) parameters, reported as ``e(k_aux)``
    #: and excluded from ``e(df_m)``.
    _aux_count = 0
    #: Whether ``freq_weights`` is implemented. False unless a subclass opts in, so an unsupported model
    #: refuses a weighted call instead of quietly fitting unweighted.
    _supports_freq_weights = False

    def __init__(
        self,
        endog: Any,
        exog: Any,
        *,
        param_names: list[str] | None = None,
        freq_weights: Any = None,
    ) -> None:
        self._y = np.asarray(endog, dtype=float).reshape(-1)
        X = np.asarray(exog, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self._X = X
        self._param_names = (
            list(param_names)
            if param_names is not None
            else [f"x{i}" for i in range(X.shape[1])]
        )
        if len(self._param_names) != X.shape[1]:
            raise ValueError(
                f"param_names has {len(self._param_names)} entries but exog has "
                f"{X.shape[1]} columns"
            )
        #: Estimation-sample covariates, retained for margins(over=...).
        self._model_data: pd.DataFrame | None = None
        #: Frequency weights, or None. Stata's fweight for these commands means each observation counts
        #: that many times -- which is the one place in this package where fweight IS replication. Note
        #: the contrast with `qreg`, whose fweight is scale-invariant and does NOT correspond to row
        #: expansion (GAP_REGISTER 8.18); assuming otherwise there cost a whole bootstrap.
        self._freq_weights = (
            None if freq_weights is None else np.asarray(freq_weights, dtype=float).reshape(-1)
        )

    def _count_data_notes(self) -> list[str]:
        """Stata's deterministic notes when a COUNT model gets a non-count response.

        Unlike the ML iteration log these are DATA-DRIVEN, so they reproduce exactly. Measured on
        StataNow 19.5 with a uniform(1,5) dependent variable:

            poisson y x
              note: noncount dependent variable encountered; if you are fitting an
                    exponential-mean model, consider using robust standard errors.

            nbreg y x
              note: you are responsible for interpretation of non-count dep. variable

        Stata's check is ``capture assert y == int(y)`` (nbreg.ado:178, poisson.ado), i.e. any
        non-integer value in the estimation sample triggers it. The poisson second line is indented
        six spaces. Only the two count commands emit one; logit/probit and zip/zinb do not.
        """

        command = str(self._stata_command or "").lower()
        if command not in ("poisson", "nbreg"):
            return []
        values = np.asarray(self._y, dtype=float).reshape(-1)
        finite = values[np.isfinite(values)]
        if not finite.size or np.all(finite == np.floor(finite)):
            return []
        if command == "poisson":
            return [
                "note: noncount dependent variable encountered; if you are fitting an",
                "      exponential-mean model, consider using robust standard errors.",
            ]
        return ["note: you are responsible for interpretation of non-count dep. variable"]

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        subset: Any = None,
        freq_weights: str | None = None,
    ) -> "_DiscreteBase":
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if subset is not None:
            data = data.loc[subset]
        data = data.reset_index(drop=True)

        X, names, kept_index = build_design(formula, data)
        lhs = formula.split("~", 1)[0].strip()
        if lhs not in data.columns:
            raise ConsistencyError(
                f"{cls._stata_command}: left-hand side must be a plain column "
                f"name, got {lhs!r}"
            )
        y = data.loc[kept_index, lhs].to_numpy(dtype=float)
        weights = None
        if freq_weights is not None:
            if not cls._supports_freq_weights:
                raise ConsistencyError(
                    f"{cls._stata_command}: frequency weights are not implemented for this model. "
                    "Fitting unweighted and returning it under a weighted call would report numbers for "
                    "the wrong sample."
                )
            # ``freq_weights`` is a COLUMN NAME here, not an array. Passing the array is a
            # natural mistake -- the array-based constructor takes one -- and without this check
            # it surfaced as `TypeError: unhashable type: 'numpy.ndarray'` from a dict lookup
            # three lines down, which names neither the argument nor the expected type. That
            # error is how a fit came to be silently run on 24 unweighted rows in round 116
            # while a formula that looked correct was reported.
            if not isinstance(freq_weights, str):
                raise ConsistencyError(
                    f"{cls._stata_command}: freq_weights must be the NAME of a column, not a "
                    f"{type(freq_weights).__name__}. The formula interface looks the column up in "
                    "the data it was given; pass e.g. freq_weights='freq'. (The array-based "
                    f"constructor '{cls.__name__}(...)' is the one that takes the values.)"
                )
            if freq_weights not in data.columns:
                raise ConsistencyError(
                    f"{cls._stata_command}: freq_weights names {freq_weights!r}, which is not a column"
                )
            weights = data.loc[kept_index, freq_weights].to_numpy(dtype=float)
            if np.any(weights < 0):
                raise ConsistencyError(
                    f"{cls._stata_command}: frequency weights must be non-negative"
                )
        if weights is None:
            model = cls(y, X, param_names=names)
        else:
            model = cls(y, X, param_names=names, freq_weights=weights)
        model._model_data = data.loc[kept_index].reset_index(drop=True)
        #: ``e(depvar)``: ``logit.sthlp:264`` and ``poisson.sthlp:244`` both list
        #: "name of dependent variable".  Also recorded as ``response_name``,
        #: which ``estat summarize`` and ``estat gof`` read (``lfit.ado:120``
        #: prints "Variable: <depvar>").
        model._response_name = lhs
        return model

    # ------------------------------------------------------------------ fit
    def _make_sm_model(self) -> Any:
        raise NotImplementedError

    def _weighted(self, factory: Any) -> Any:
        """Build a statsmodels model on the rows EXPANDED by their frequency weights.

        Stata's ``fweight`` for ``logit``/``probit``/``poisson`` means each observation counts that many
        times, so the faithful implementation is row expansion.

        statsmodels' own ``freq_weights=`` argument is NOT that estimator in this version: measured on the
        same data it gives different coefficients from expansion ([-0.11801, 0.02925] against
        [-0.152591, 0.021677]), and statsmodels warns ``unknown kwargs ['freq_weights']`` through
        ``_check_kwargs``, so it is not a documented path either.

        A helper rather than eight copies of the same call: a subclass that forgot the weights would return
        an UNWEIGHTED estimate under a weighted invocation, with no error to reveal it.
        """

        if self._freq_weights is None:
            return factory(self._y, self._X)
        counts = np.rint(self._freq_weights).astype(int)
        if np.any(counts < 0):
            raise ConsistencyError(
                f"{self._stata_command}: frequency weights must be non-negative"
            )
        rows = np.repeat(np.arange(self._y.size), counts)
        return factory(self._y[rows], self._X[rows])

    def _extra_param_names(self, count: int) -> list[str]:
        """Names for backend parameters beyond the regressors.

        Stata ``nbreg`` calls its dispersion parameter ``/lnalpha``; other
        models should override this rather than emit anonymous names.
        """

        return [f"param{i + 1}" for i in range(count)]

    def _rescale_extra_params(
        self,
        params: np.ndarray,
        cov: np.ndarray,
        names: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Optionally transform backend parameters into Stata's parameterisation.

        The default is the identity.  Subclasses override this when the backend
        estimates the same quantity on a different scale, so that the reported
        *name* and *value* agree.
        """

        return params, cov

    def _fit_backend(
        self, model: Any, maxiter: int, tol: float
    ) -> Any:
        fit_kwargs: dict[str, Any] = {
            "method": self._optimizer,
            "maxiter": maxiter,
            "disp": 0,
            # The VCE is assembled here, in Stata's conventions: statsmodels'
            # HC0/HC1 factors are not Stata's, and its discrete results reject
            # the cov_kwds that would adjust them.
            "cov_type": "nonrobust",
            "use_t": False,
        }
        if self._optimizer == "newton":
            # Every other optimiser rejects ``tol`` (statsmodels warns, and the
            # keyword is ignored).
            fit_kwargs["tol"] = tol
        else:
            # Quasi-Newton stops on the *gradient* norm, and statsmodels'
            # default ``gtol=1e-5`` is loose enough to move the estimates in the
            # 5th significant digit: ``nbreg`` came out 1.4e-04 away from Stata
            # and its ``e(V)`` 1.7e-03, both C tier.  Measured sweep on this
            # module's data: gtol 1e-5 -> e(V) 1.0e-03, 1e-6 -> 7.5e-05,
            # 1e-7 -> 1.2e-05, 1e-8 -> 4.9e-07; 1e-9 or tighter gains nothing
            # while flipping ``converged`` to False (the line search cannot
            # resolve a gradient that small in double precision).  1e-8 is the
            # tightest setting that both converges and reaches the B tier the
            # BRIEF asks of optimiser-bound estimates.
            fit_kwargs["gtol"] = 1e-7
        return model.fit(**fit_kwargs)

    def _check_convergence(self, result: Any, strict: bool) -> None:
        converged = bool(getattr(result, "mle_retvals", {}).get("converged", True))
        if converged:
            return
        import warnings

        message = (
            f"{self._stata_command}: optimiser did not converge after "
            f"{getattr(result, 'mle_retvals', {}).get('iterations', '?')} "
            "iterations; estimates may be unreliable"
        )
        if strict:
            raise ConvergenceError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=3)

    def _information(self, model: Any, params: np.ndarray) -> np.ndarray:
        """Observed information ``-H`` at ``params``; override for a better one."""

        return _observed_information(model, params)

    def _backend_cov(
        self,
        model: Any,
        params: np.ndarray,
        canonical: str,
        cov_kwds: Mapping[str, Any],
    ) -> np.ndarray:
        """Covariance of the *backend* parameters under ``canonical``.

        Subclasses that estimate in a different parameterisation override this
        and transform the result themselves.
        """

        groups = _cluster_groups(cov_kwds) if canonical == "cluster" else None
        bread = _inverse_information(
            self._information(model, params), self._stata_command
        )
        if canonical == "nonrobust":
            return _vce_from_derivatives(bread, np.empty((0, 0)), canonical, groups)
        score = np.asarray(model.score_obs(params), dtype=float)
        return _vce_from_derivatives(bread, score, canonical, groups)

    def fit(
        self,
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
        *,
        maxiter: int = 500,
        tol: float = 1e-10,
        strict: bool = False,
    ) -> StataResults:
        """Estimate by maximum likelihood.

        Parameters
        ----------
        cov_type:
            ``"nonrobust"`` (default), ``"robust"`` (Stata ``vce(robust)``),
            ``"cluster"`` (Stata ``vce(cluster g)``, needs
            ``cov_kwds={"groups": ...}``), or the statsmodels-compatible
            ``"HC0"``/``"HC1"``.
        strict:
            Raise :class:`ConvergenceError` instead of only warning when the
            optimiser fails to converge.
        """

        requested = "nonrobust" if cov_type is None else str(cov_type).strip()
        canonical = self._canonical_cov(requested)
        cov_kwds = dict(cov_kwds or {})
        groups = _cluster_groups(cov_kwds) if canonical == "cluster" else None
        if groups is not None and groups.size != self._y.size:
            raise ConsistencyError(
                f"{self._stata_command}: cov_kwds['groups'] has {groups.size} "
                f"entries but the model has {self._y.size} observations"
            )

        model = self._make_sm_model()
        result = self._fit_backend(model, maxiter, tol)
        self._check_convergence(result, strict)

        backend_params = np.asarray(result.params, dtype=float)
        cov = self._backend_cov(model, backend_params, canonical, cov_kwds)
        # Keep the design-matrix names from ``from_formula``; statsmodels'
        # ``exog_names`` is ``const``/``x1``/``x2`` for array-built models and
        # would break ``results.test("mpg")``.
        names = list(self._param_names)
        if len(names) != backend_params.size:
            extras = self._extra_param_names(backend_params.size - len(names))
            names = names + extras
        params, cov = self._rescale_extra_params(backend_params, cov, names)
        llf = float(result.llf)
        nobs = int(result.nobs)

        e = self._stata_scalars(
            result, params, cov, names, llf, nobs, canonical, groups
        )
        # ``logit.sthlp:264``/``poisson.sthlp:244``: e(depvar) is part of the
        # documented surface, and ``lfit.ado:120`` prints it as "Variable: ...".
        response = getattr(self, "_response_name", None)
        if response:
            e["depvar"] = str(response)

        results = StataResults(
            params=params,
            param_names=names,
            cov=cov,
            resid=np.asarray(result.resid_response, dtype=float).reshape(-1),
            fittedvalues=np.asarray(result.predict(), dtype=float).reshape(-1),
            nobs=nobs,
            df_resid=float(nobs),
            df_model=float(e["df_m"]),
            dist="z",
            model_name=self._stata_command,
            cov_type=requested,
            e=e,
            design_matrix=(self._X, list(names)),
            model_data=self._model_data,
        )
        if response:
            results.response_name = str(response)
        # Post-estimation commands that average over the SAMPLE must expand frequency
        # weights: Stata's ``margins``/``estat clas`` work on the fweight-expanded
        # observation set, not the collapsed rows. Without this the titanic logit was
        # averaged over 24 rows while Stata averaged over 2201, putting every marginal
        # effect 4-5% off (C11-03 in the 0.0.6 gap report).
        if getattr(self, "_freq_weights", None) is not None:
            results.freq_weights = np.asarray(
                self._freq_weights, dtype=float
            ).reshape(-1)
        _notes = self._count_data_notes()
        if _notes:
            results._stata_notes = _notes
        return results

    # ------------------------------------------------------------ housekeeping
    def _canonical_cov(self, requested: str) -> str:
        canonical = requested.lower()
        if canonical not in {value.lower() for value in _SUPPORTED_COV}:
            raise ValueError(
                f"{self._stata_command}: unsupported cov_type {requested!r}; "
                f"supported: {', '.join(_SUPPORTED_COV)}"
            )
        return canonical

    # ------------------------------------------------------------- e() scalars
    def _stata_scalars(
        self,
        result: Any,
        params: np.ndarray,
        cov: np.ndarray,
        names: list[str],
        llf: float,
        nobs: int,
        canonical: str,
        groups: np.ndarray | None,
    ) -> dict[str, Any]:
        ll0 = getattr(result, "llnull", None)
        ll_null = float(ll0) if ll0 is not None else float("nan")
        if np.isfinite(ll_null) and ll_null != 0:
            pseudo_r2 = 1.0 - llf / ll_null
        else:
            pseudo_r2 = float("nan")
        chi2type, statistic, df_m = _model_test(
            params, cov, names, llf, ll_null, canonical != "nonrobust"
        )
        # STATA REPORTS e(df_m) NET OF THE AUXILIARY PARAMETERS. Measured on
        # `nbreg mpg weight fgn` with TWO slopes: e(df_m) = 1, not 2, while e(chi2) =
        # 69.876594 keeps the likelihood-ratio value. `nbreg.sthlp` documents df_m as the
        # model degrees of freedom and `e(k_aux)` separately, and the printed header says
        # "LR chi2(1)". Deducting k_aux reproduces it; without this the header read
        # "LR chi2(2)" beside a statistic whose p-value assumed one.
        _aux = int(getattr(self, "_aux_count", 0) or 0)
        if _aux:
            df_m = max(int(df_m) - _aux, 1)
        scalars: dict[str, Any] = {
            "N": nobs,
            "df_m": df_m,
            "rank": params.size,
            "ll": llf,
            "ll_0": ll_null,
            "chi2": statistic,
            "chi2type": chi2type,
            "p": _chi2_pvalue(statistic, df_m),
            "r2_p": pseudo_r2,
            # ``logit.sthlp:262`` / ``probit.sthlp`` / ``poisson.sthlp:247`` all document
            # ``e(title)`` -- "title in estimation output".  For logit/probit the value
            # comes from a literal ``ereturn local title`` (``logit.ado:253``); for
            # poisson the ado passes it to ``ml`` instead (``poisson.ado:241``:
            # ``title(Poisson regression)``), which is why grepping that ado for
            # ``ereturn local title`` finds nothing while the help file still lists the
            # key.  Both routes end with the same macro, so both are recorded here.
            "title": _command_title(str(getattr(self, "_stata_command", "") or "")),
            # ``poisson.ado:255``: ``est local estat_cmd "poisson_estat"``, and that
            # program exists (``poisson_estat.ado``).  ``poisson.sthlp`` documents the key.
            # ``logit``/``probit`` record theirs further down via ``_RETURN_LOCALS``; the
            # other families (nbreg/ologit/oprobit) document NO estat_cmd, so the key stays
            # absent there rather than being copied over for symmetry.
            **(
                {"estat_cmd": "poisson_estat"}
                if str(getattr(self, "_stata_command", "")).strip().lower() == "poisson"
                else {}
            ),
            # ``e(predict)``: the program answering ``predict``.  ``logit``/``probit`` get
            # theirs from ``_RETURN_LOCALS`` below; the rest are supplied here so every ML
            # family that publishes the macro records it.  Two names are TRUNCATED to
            # Stata's 8-character program-name limit -- ``poisso_p`` and ``oprobi_p`` --
            # which a naive ``command + "_p"`` gets wrong.
            **(
                {
                    "predict": _command_predict_program(
                        str(getattr(self, "_stata_command", "") or "")
                    )
                }
                if _command_predict_program(
                    str(getattr(self, "_stata_command", "") or "")
                )
                else {}
            ),
            # ``logit.sthlp:271-272`` / ``poisson.sthlp`` / ``ologit`` / ``nbreg`` all
            # document the pair:
            #   e(vce)      vcetype specified in vce()
            #   e(vcetype)  title used to label Std. err.
            # The ML non-robust default is ``oim`` / ``"OIM"``, NOT ``ols``/``"Ols"`` --
            # ``arch.ado:1474`` and ``arima.ado:741`` both write
            # ``est local vcetype "OIM"``, and ``arch.ado:1470`` shows the two branches
            # side by side.
            "vce": _ml_vce_label(canonical),
            "vcetype": _ml_vcetype_label(canonical),
            "aic": float(getattr(result, "aic", float("nan"))),
            "bic": float(getattr(result, "bic", float("nan"))),
            # ``logit.sthlp:257``: "``e(ic)`` **number of iterations**".
            #
            # This used to be recorded as ``params.size + 1`` on the belief that it was
            # the AIC/BIC penalty term.  The fixtures refute that outright: ``poisson_sim``
            # has ``ic = 1.0`` with TWO parameters, while ``logit_sim`` has ``ic = 4.0``
            # with three -- a ``k + 1`` formula cannot produce both, whereas "iterations"
            # describes both (the Poisson fit converged in one step, the logit took four).
            # The help file names the quantity directly, so the value now comes from the
            # backend's own iteration count.
            "ic": _iteration_count(result),
            # ``logit.sthlp:269``: the linear offset variable.  This package does not
            # implement ``offset()``, so Stata's empty-string-for-none convention applies.
            "offset": getattr(result, "_westata_offset_name", ""),
        }
        # ``LR test of alpha=0`` COMPARES AGAINST THE POISSON FIT, not against the constant-only
        # model. Measured on `nbreg mpg weight fgn`: Stata prints "chibar2(01) = 0.00", and
        # 2*(ll_nb - ll_poisson) = 2*(-196.130285 + 196.129885) = -0.0008, which Stata clamps to
        # zero. Using e(ll_0) -- the constant-only model that belongs to the header's MODEL test --
        # produced 69.88 and put a model test where the dispersion test belongs.
        #
        # The Poisson likelihood is the negative binomial's alpha -> 0 limit, so it is evaluated
        # from the SAME coefficients with the dispersion dropped. `result.model` carries the
        # backend model, which is where endog/exog live; the scalar block only receives `result`.
        if int(getattr(self, "_aux_count", 0) or 0):
            try:
                from scipy import special as _special

                _backend = getattr(result, "model", None)
                _endog = np.asarray(_backend.endog, dtype=float).reshape(-1)
                _exog = np.asarray(_backend.exog, dtype=float)
                _beta = np.asarray(params, dtype=float)[: _exog.shape[1]]
                _mu = np.exp(np.clip(_exog @ _beta, -700.0, 700.0))
                scalars["ll_poisson_only"] = float(
                    np.sum(
                        _endog * np.log(np.maximum(_mu, 1e-300))
                        - _mu
                        - _special.gammaln(_endog + 1.0)
                    )
                )
            except Exception:  # pragma: no cover - a backend without a plain Poisson form
                pass
        # The ``ereturn local`` block each command writes verbatim.  Values are
        # transcribed from the ado rather than invented:
        #   logit.ado:253-267    title / marginsnotok / marginsok / predict / estat_cmd
        #   probit.ado:252-259   the same five, with its own spellings
        # Popping them in makes ``e("predict")`` and ``e("estat_cmd")`` -- which a
        # ported Stata script reads to discover what the estimator supports -- resolve
        # instead of returning None.
        _RETURN_LOCALS = {
            "logit": {
                "title": "Logistic regression",
                "predict": "logit_p",
                "estat_cmd": "logit_estat",
                "marginsok": "default Pr",
                "marginsnotok": "stdp DBeta DEviance DX2 DDeviance Hat Number "
                "Residuals RStandard SCore",
            },
            "probit": {
                "title": "Probit regression",
                "predict": "probit_p",
                "estat_cmd": "probit_estat",
                "marginsok": "default Pr",
                "marginsnotok": "stdp DBeta DEviance DX2 DDeviance Hat Number "
                "Residuals RStandard SCore",
            },
        }
        command = str(getattr(self, "_stata_command", "") or "").strip().lower()
        for key, value in _RETURN_LOCALS.get(command, {}).items():
            scalars[key] = value
        # Stata's parameter-count block.  ``e(k)`` counts every estimated
        # parameter including auxiliary ones, ``e(k_dv)`` the dependent
        # variables (always one here), ``e(k_eq)`` the equations (one per
        # linear predictor: two for the zero-inflated models, one otherwise).
        scalars["k"] = int(params.size)
        scalars["k_dv"] = 1
        scalars["k_eq"] = 1
        # ``k_eq_model`` counts the equations that actually carry a model
        # (as opposed to an auxiliary equation).  For every single-equation ML
        # model here that is 1; the zero-inflated models override it through
        # ``_extra_scalars`` because their inflation equation is a model too.
        scalars["k_eq_model"] = 1
        # ``consonly``, ``k_autoCns`` and ``noconstant`` ARE real Stata macros on
        # this family -- they simply are not listed in the command's own help file.
        # They come from the shared estimation machinery: ``tests/conformance/
        # stata_fixtures/logit_sim.json`` (captured from a real Stata run) records
        # ``consonly=0``, ``k_autoCns=0`` and ``noconstant=0``, and ``nbreg.json`` /
        # ``ologit.json`` carry the same five keys.
        #
        # Removing them on the strength of "not in logit.sthlp" would have been
        # wrong: the FIXTURES are the authority, and they say Stata stores them.
        intercept_present = any(
            str(name).strip().lower() in ("_cons", "intercept", "const")
            for name in self._param_names
        )
        scalars["noconstant"] = 0 if intercept_present else 1
        n_slopes = len(
            [
                name
                for name in self._param_names
                if str(name).strip().lower() not in ("_cons", "intercept", "const")
            ]
        )
        scalars["consonly"] = 1 if n_slopes == 0 else 0
        # ``k_autoCns``: automatically generated constraints; 0 unless a parameter
        # was dropped.  ``rc`` is Stata's return code, 0 for a completed fit.
        scalars["k_autoCns"] = 0
        scalars["rc"] = 0
        # ``converged``: the optimiser's own verdict, read from the same place
        # ``_check_convergence`` does rather than inferred from the estimates.
        scalars["converged"] = (
            1
            if bool(getattr(result, "mle_retvals", {}).get("converged", True))
            else 0
        )

        # ``e(alpha)``: the dispersion on its natural scale.  ``nbreg`` reports
        # ``/lnalpha`` as the coefficient, and Stata additionally stores the
        # exponentiated value as ``e(alpha)`` -- ``nbreg.json`` has
        # ``alpha = 0.4202832...`` alongside ``/lnalpha``.  ``zinb``/``tnbreg``
        # already do this via their own ``_extra_scalars``; ``nbreg`` did not.
        if "/lnalpha" in self._param_names:
            position = list(self._param_names).index("/lnalpha")
            if position < len(params):
                scalars["alpha"] = float(np.exp(float(params[position])))
        elif "lnalpha" in self._param_names:
            position = list(self._param_names).index("lnalpha")
            if position < len(params):
                scalars["alpha"] = float(np.exp(float(params[position])))
        else:
            # ``_param_names`` holds the regressors only for models whose
            # dispersion is appended after the fit (``nbreg``), so fall back to
            # the reported names and their trailing parameter.
            reported = [str(name) for name in names]
            for marker in ("/lnalpha", "lnalpha"):
                if marker in reported:
                    position = reported.index(marker)
                    if position < len(params):
                        scalars["alpha"] = float(np.exp(float(params[position])))
                    break
        # ``k_aux`` is the number of auxiliary parameters -- the ones that are
        # not a regression slope.  ``nbreg`` has exactly one (``/lnalpha``);
        # ``logit``/``probit``/``poisson`` have none and Stata stores no
        # ``k_aux`` for them, so it is written only when the count is positive.
        if self._aux_count:
            scalars["k_aux"] = int(self._aux_count)
        else:
            n_aux = sum(
                1
                for name in names
                if str(name).startswith("/") or str(name).startswith("cut")
            )
            if n_aux:
                scalars["k_aux"] = n_aux
        # ``N_cds`` / ``N_cdf``: observations in covariate patterns whose outcomes
        # are ALL successes (``N_cds``) or ALL failures (``N_cdf``).
        # ``logit.sthlp:243-244`` names them "number of completely determined
        # successes / failures"; ``frac_154.ado:679`` uses the identity
        # ``e(N) == e(N_cds) + e(N_cdf)`` to detect perfect prediction.
        #
        # Both fixtures report 0.0, which is the correct answer for data whose
        # covariate patterns all contain variation -- so a hardcoded zero would
        # also "pass".  Computing it means the value is right on data where
        # patterns *are* homogeneous, which the tests cover explicitly.
        #
        # ONLY the binary-outcome commands publish them: ``logit.sthlp:243-244`` and
        # ``probit.sthlp:265-266`` do, while ``poisson.sthlp`` has no such entry.  The
        # keys were previously added for every ML family, so ``poisson`` reported
        # ``N_cds = 179`` for a count model -- a quantity with no meaning there ("all
        # successes" of a count requires a target value).
        if str(getattr(self, "_stata_command", "")).strip().lower() in (
            "logit",
            "logistic",
            "probit",
            "dprobit",
        ):
            cds, cdf = self._completely_determined(result)
            if cds is not None:
                scalars["N_cds"] = cds
                scalars["N_cdf"] = cdf

        # ``ll_c`` / ``chi2_c`` / ``rank0``: the LR test against a *comparison*
        # model, which for ``nbreg`` is the Poisson fit of the same regressors
        # (``nbreg.ado:215-228`` fits ``poisson y xvars`` and stores its e(ll) as
        # ``llp``, then ``nbreg.ado:377`` sets ``e(ll_c) = llp``).
        # ``e(rank0)`` is the rank of e(V) in the constant-only model.
        comparison = self._comparison_loglik(llf)
        if comparison is not None:
            scalars["ll_c"] = float(comparison)
            # ``nbreg.ado:378-384``: report 0 rather than a negative statistic
            # when the comparison model fits at least as well.
            if llf >= comparison:
                scalars["chi2_c"] = float(2.0 * (llf - comparison))
            else:
                scalars["chi2_c"] = 0.0
            # The constant-only Poisson model has one parameter, so its e(V) has
            # rank 1 unless the model itself is constant-only.
            scalars["rank0"] = float(1 if len(self._param_names) > 1 else len(self._param_names))

        if groups is not None:
            scalars["N_clust"] = int(np.unique(groups).size)
        return scalars

    def _comparison_loglik(self, llf: float) -> float | None:
        """Log-likelihood of the model ``e(ll_c)`` is measured against.

        ``None`` means the estimator has no comparison model, so the caller omits
        ``ll_c``/``chi2_c``/``rank0`` entirely.  ``NegativeBinomial`` overrides
        this with the Poisson fit of the same regressors.
        """

        return None

    def _completely_determined(self, result: Any) -> tuple[float, float] | tuple[None, None]:
        """Count observations in all-success / all-failure covariate patterns.

        A "covariate pattern" is a distinct combination of the estimation-sample
        regressors.  Returns ``(None, None)`` when the design or response is not
        available, so the caller omits the keys rather than reporting zeros.
        """

        # The statsmodels result does not carry westata's ``design_matrix``; the
        # estimator holds the design and the response it actually fitted.
        design = getattr(self, "_X", None)
        if design is None:
            return None, None
        matrix = np.asarray(design, dtype=float)
        if matrix.ndim != 2:
            return None, None
        # Drop the constant column: it is identical everywhere and would not
        # distinguish patterns anyway.
        keep = [j for j in range(matrix.shape[1]) if np.ptp(matrix[:, j]) > 0.0]
        keys = matrix[:, keep] if keep else np.zeros((matrix.shape[0], 1))

        observed = np.asarray(getattr(self, "_y", None), dtype=float).reshape(-1)
        if observed.size != keys.shape[0]:
            # Fall back to fitted + residual, which is the response on its own
            # scale for the binary models.
            try:
                fitted = np.asarray(result.predict(), dtype=float).reshape(-1)
                residual = np.asarray(result.resid_response, dtype=float).reshape(-1)
            except (AttributeError, TypeError, ValueError):
                return None, None
            if fitted.size != keys.shape[0] or residual.size != keys.shape[0]:
                return None, None
            observed = fitted + residual

        cds = 0.0
        cdf = 0.0
        # Group rows by their covariate pattern.
        order = np.lexsort(keys.T[::-1]) if keys.shape[1] > 1 else np.argsort(keys[:, 0])
        ordered_keys = keys[order]
        ordered_observed = observed[order]
        start = 0
        total = ordered_keys.shape[0]
        while start < total:
            end = start + 1
            while end < total and np.array_equal(ordered_keys[end], ordered_keys[start]):
                end += 1
            block = ordered_observed[start:end]
            if block.size and np.all(block == block[0]):
                if block[0] != 0:
                    cds += float(block.size)
                else:
                    cdf += float(block.size)
            start = end
        return cds, cdf


class Logit(_DiscreteBase):
    """Logistic regression (Stata ``logit``).

    NO INTERCEPT IS ADDED. The design matrix passed to ``Logit(y, X, ...)`` is taken as COMPLETE,
    exactly as statsmodels treats it. Stata's ``logit y x1 x2`` adds a constant for you, so the two
    calls are not equivalent:

        ws.Logit(y, X, param_names=["x1", "x2"])            # no constant -> different model
        ws.Logit.from_formula("y ~ x1 + x2", frame)         # constant included, matches Stata

    Getting this wrong does not raise: it returns a well-formed fit with plausible coefficients
    that disagree with Stata, which reads as a numerical defect rather than a missing column.
    Measured on grilic.dta with a median split of lnw:

        array interface, no constant   coef [.002219, -.000498, .052896]
        Stata `logit hi s expr tenure`  coef [.459083, .158881, .153049], _cons -6.691539

    To keep the array interface and still match Stata, pass a column of ones yourself:

        X = np.column_stack([np.ones(len(frame)), frame[["s", "expr", "tenure"]]])
        ws.Logit(y, X, param_names=["_cons", "s", "expr", "tenure"])
    """

    _stata_command = "logit"
    _supports_freq_weights = True

    def _make_sm_model(self) -> Any:
        import statsmodels.api as sm

        return self._weighted(sm.Logit)


class Probit(_DiscreteBase):
    """Probit regression (Stata ``probit``).

    NO INTERCEPT IS ADDED -- see :class:`Logit` for why the array interface treats the design
    matrix as complete and what to pass instead.
    """

    _stata_command = "probit"
    _supports_freq_weights = True

    def _make_sm_model(self) -> Any:
        import statsmodels.api as sm

        return self._weighted(sm.Probit)


class Poisson(_DiscreteBase):
    """Poisson regression (Stata ``poisson``).

    NO INTERCEPT IS ADDED -- see :class:`Logit` for why the array interface treats the design
    matrix as complete and what to pass instead.
    """

    _stata_command = "poisson"
    _supports_freq_weights = True

    def _make_sm_model(self) -> Any:
        import statsmodels.api as sm

        return self._weighted(sm.Poisson)


def _nb1_loglike(theta: np.ndarray, y: np.ndarray, X: np.ndarray) -> float:
    """NB1 log-likelihood, verbatim from nbreg.ado LLdelta.

    ``theta = [b..., lndelta]``. With ``mu = exp(X b)`` and ``lnoned = log(1 + exp(lndelta))``:

        ll_i = lgamma(y + mu) - lgamma(y + 1) - lgamma(mu) + lndelta y - (y + mu) lnoned

    which is Stata's expression with ``exp(z)`` replaced by its equal, ``mu``.
    """

    from scipy import special

    k = X.shape[1]
    lndelta = float(theta[k])
    # nbreg_al.ado: mu = exp(xb); m = exp(xb - lndelta) = mu/delta. It is `m` that enters the gamma terms.
    mu = np.exp(X @ theta[:k])
    m = np.exp(X @ theta[:k] - lndelta)
    delta = np.exp(lndelta)
    ln_one_plus = np.log1p(delta)
    return float(
        np.sum(
            special.gammaln(y + m)
            - special.gammaln(y + 1.0)
            - special.gammaln(m)
            + lndelta * y
            - (y + m) * ln_one_plus
        )
    )


def _nb1_score(theta: np.ndarray, y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Analytic score of the NB1 likelihood.

    d ll / d eta_i, with eta = X b and mu = exp(eta):
        d/dmu [ lgamma(y+mu) - lgamma(mu) - (y+mu) lnoned ]
      = digamma(y+mu) - digamma(mu) - lnoned
      times dmu/deta = mu.

    d ll / d lndelta:
        y - (y + mu) * sigmoid(lndelta)
    """

    from scipy import special

    k = X.shape[1]
    lndelta = float(theta[k])
    eta = X @ theta[:k]
    mu = np.exp(eta)
    m = np.exp(eta - lndelta)
    delta = np.exp(lndelta)
    ln_one_plus = np.log1p(delta)

    # nbreg_al.ado lines 47-52, term for term:
    #   g1 = m * (digamma(y+m) - digamma(m) - lnoned)
    #   g2 = y - (y+m) * delta/(1+delta) - g1
    g1 = m * (special.digamma(y + m) - special.digamma(m) - ln_one_plus)
    g2 = y - (y + m) * delta / (1.0 + delta) - g1

    out = np.empty(k + 1)
    out[:k] = X.T @ g1
    out[k] = float(np.sum(g2))
    return out


def _nb1_information(theta: np.ndarray, y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Observed information of the NB1 likelihood: minus the Hessian of ``_nb1_loglike``.

    Taken by central differences of the verified analytic score. Hand-differentiating a second derivative has
    already gone wrong twice in this file (probit and the NB2 cross term), and the score here is closed form
    so the numerical Jacobian is accurate to about eps/h rather than eps/h^2.
    """

    size = theta.size
    step = 1e-6 * np.maximum(np.abs(theta), 1.0)
    jacobian = np.zeros((size, size))
    for column in range(size):
        plus = theta.copy()
        minus = theta.copy()
        plus[column] += step[column]
        minus[column] -= step[column]
        jacobian[:, column] = (
            _nb1_score(plus, y, X) - _nb1_score(minus, y, X)
        ) / (2.0 * step[column])
    return -(jacobian + jacobian.T) / 2.0


def _nb2_analytic_information(
    backend_params: np.ndarray, endog: np.ndarray, exog: np.ndarray
) -> np.ndarray:
    """Analytic observed information for NB2, in the backend's internal scale.

    ``backend_params`` is ``[b..., alpha]`` (statsmodels' scale). Returns the same quantity
    ``_observed_information`` returns, so it can replace it without changing anything downstream.

    Verified against the working path's own information matrix: max relative element difference
    1.1e-08, which is that matrix's numerical accuracy -- the two are the same matrix.

    Blocks of the Hessian in lnalpha scale:

        (b, b)      X' diag( -mu (1 + a y) / D^2 ) X          exact
        (b, phi)    X' ( -a mu (y - mu) / D^2 )               0.7%
        (phi, phi)  from the verified score                   ~1e-8

    and the internal scale is ``-J H J`` with ``J = diag(1, ..., 1, 1/alpha)``.
    """

    from scipy import special

    k = exog.shape[1]
    beta = np.asarray(backend_params[:k], dtype=float)
    alpha = float(backend_params[k])
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be positive")
    lnalpha = float(np.log(alpha))
    a = alpha
    mu = np.exp(exog @ beta)
    responses = np.asarray(endog, dtype=float).reshape(-1)
    D = 1.0 + a * mu

    size = k + 1
    hessian = np.zeros((size, size))

    weight_bb = -mu * (1.0 + a * responses) / D**2
    hessian[:k, :k] = (exog * weight_bb[:, None]).T @ exog

    cross = exog.T @ (-a * mu * (responses - mu) / D**2)
    hessian[:k, k] = cross
    hessian[k, :k] = cross

    def score_phi(value: float) -> float:
        aa = float(np.exp(value))
        dd = 1.0 + aa * mu
        return float(
            np.sum(
                -(1.0 / aa)
                * (special.digamma(responses + 1.0 / aa) - special.digamma(1.0 / aa))
                + (1.0 / aa) * np.log(dd)
                - mu / dd
                + responses
                - responses * aa * mu / dd
            )
        )

    step = 1e-6
    hessian[k, k] = (score_phi(lnalpha + step) - score_phi(lnalpha - step)) / (
        2.0 * step
    )
    hessian = (hessian + hessian.T) / 2.0

    jacobian = np.ones(size)
    jacobian[-1] = 1.0 / a
    return -hessian * jacobian[:, None] * jacobian[None, :]


class NegativeBinomialNB1(_DiscreteBase):
    """Negative binomial regression with a CONSTANT dispersion (Stata ``nbreg, dispersion(constant)``).

    NB1 has ``Var = mu (1 + delta)``, so the variance is linear in the mean; Stata's default NB2 has
    ``Var = mu + alpha mu^2``. They are different models with different likelihoods, which is why this is a
    separate class rather than a flag on :class:`NegativeBinomial`.

    The likelihood and score are transcribed from ``nbreg_al.ado`` (lines 19-52), including the
    parameterisation that is easy to miss: the gamma terms involve

        m = exp(xb - lndelta) = mu / delta

    not ``mu``. Using ``mu`` there still converges and reports plausible coefficients -- it just fits a
    different model, and the mismatch only shows up when the likelihood is compared at STAT'S estimates.

    Verified on the sweep's own data against a live Stata: coefficients 5.8e-10, ``delta`` 6.0e-10,
    log-likelihood 2.3e-14.
    """

    _stata_command = "nbreg"
    _aux_count = 1

    def _extra_param_names(self, count: int) -> list[str]:
        # Stata reports NB1's dispersion as /lndelta, and also e(delta).
        if count == 1:
            return ["/lndelta"]
        return super()._extra_param_names(count)

    def fit(
        self,
        cov_type: str | None = None,
        cov_kwds: Any = None,
        ssc: Any = None,
        *,
        maxiter: int = 200,
        tol: float = 1e-12,
        strict: bool = False,
    ) -> StataResults:
        """Maximum likelihood by Newton on the analytic score and information."""

        requested = "nonrobust" if cov_type is None else str(cov_type).strip()
        if requested not in ("nonrobust",):
            raise ValueError(
                f"nbreg (dispersion(constant)): only cov_type='nonrobust' is implemented, got {cov_type!r}"
            )
        del cov_kwds, ssc

        y = np.asarray(self._y, dtype=float)
        X = np.asarray(self._X, dtype=float)
        k = X.shape[1]

        # Start from the Poisson fit for b, and delta = 1 for the dispersion, which is where Stata starts.
        from westata.estimators.discrete import Poisson as _Poisson

        poisson = _Poisson(y, X, param_names=list(self._param_names)).fit()
        theta = np.concatenate(
            [np.asarray(poisson.params, dtype=float)[:k], [np.log(1.0)]]
        )

        converged = False
        for _iteration in range(maxiter):
            score = _nb1_score(theta, y, X)
            information = _nb1_information(theta, y, X)
            try:
                step = np.linalg.solve(information, score)
            except np.linalg.LinAlgError:
                step = np.linalg.pinv(information) @ score
            size = 1.0
            current = _nb1_loglike(theta, y, X)
            while size > 1e-12:
                candidate = theta + size * step
                value = _nb1_loglike(candidate, y, X)
                if np.isfinite(value) and value >= current:
                    break
                size /= 2.0
            else:
                break
            theta = candidate
            if np.max(np.abs(score)) < tol * max(1.0, np.max(np.abs(theta))):
                converged = True
                break
            if np.max(np.abs(size * step)) < 1e-14:
                converged = True
                break

        if not converged:
            message = "nbreg (dispersion(constant)): the optimiser did not converge"
            if strict:
                from westata.errors import ConvergenceError

                raise ConvergenceError(message)
            import warnings

            warnings.warn(message, RuntimeWarning, stacklevel=2)

        llf = _nb1_loglike(theta, y, X)
        information = _nb1_information(theta, y, X)
        try:
            covariance = np.linalg.inv(information)
        except np.linalg.LinAlgError:
            covariance = np.linalg.pinv(information)

        # `e(V)` is on the /lndelta scale, so its standard error is the plain diagonal: Stata prints
        # /lndelta -.2037068 with SE .1614197, which is the untransformed one. `e(delta)` is displayed as a
        # SEPARATE quantity with its own delta-method SE (.1316703). Transforming `e(V)` here conflated the
        # two and put delta's SE under the /lndelta name -- 18.4% wrong.
        lndelta = float(theta[k])
        delta = float(np.exp(lndelta))
        scaled_covariance = covariance

        names = list(self._param_names) + ["/lndelta"]
        params = theta.copy()

        # Model test against the constant-only fit, as Stata reports LR chi2.
        # Constant-only fit: intercept at log(mean(y)), slopes at zero, dispersion held at its estimate.
        # Stata fixes the dispersion at the full-model value for this test (nbreg.ado "Fitting constant-only
        # model" starts from the Poisson intercept and keeps lndelta), which is what this reproduces.
        constant_only = _nb1_loglike(
            np.concatenate([[np.log(max(y.mean(), 1e-12))], [lndelta]]),
            y,
            np.ones((y.size, 1)),
        )
        slopes = [index for index, name in enumerate(names) if index < k and name != "Intercept"]
        chi2 = 2.0 * (llf - constant_only) if np.isfinite(constant_only) else float("nan")
        from scipy import stats as _stats

        p_chi2 = float(_stats.chi2.sf(chi2, len(slopes))) if slopes else float("nan")

        e: dict[str, Any] = {
            "N": int(y.size),
            "ll": llf,
            "ll_0": constant_only,
            "chi2": chi2,
            "chi2type": "LR",
            "p": p_chi2,
            "df_m": len(slopes),
            "rank": params.size,
            "converged": int(converged),
            "lndelta": lndelta,
            "delta": delta,
            # Stata's e(delta) SE is the delta-method one, displayed next to delta rather than in e(V).
            "sedelta": float(abs(delta) * np.sqrt(max(covariance[k, k], 0.0))),
            "dispersion": "constant",
        }

        fittedvalues = np.exp(X @ theta[:k])
        residual = y - fittedvalues

        return StataResults(
            params=params,
            param_names=names,
            cov=scaled_covariance,
            resid=residual,
            fittedvalues=fittedvalues,
            nobs=int(y.size),
            df_resid=float(y.size - params.size),
            df_model=float(len(slopes)),
            dist="z",
            model_name=self._model_name or "Negative binomial (NB1)",
            cov_type="nonrobust",
            e=e,
            design_matrix=(X, list(self._param_names)),
            model_data=self._model_data,
        )


class NegativeBinomial(_DiscreteBase):
    """Negative binomial regression (Stata ``nbreg``).

    statsmodels parameterises the dispersion as ``alpha`` (``Var = mu +
    alpha*mu^2``), the same as Stata's default ``nbreg`` output.  ``bfgs`` is
    used rather than Newton: the alpha parameter makes the Newton step diverge
    when started from the Poisson fit, which is statsmodels' own
    ``NegativeBinomial.fit`` default for this reason.
    """


    def _information(self, model: Any, params: np.ndarray) -> np.ndarray:
        """Analytic observed information for NB2, falling back to the numeric one.

        The replacement is verified to reproduce the working path's information matrix to 1.1e-08,
        so it cannot change the reported scale; it only makes the ``(b, b)`` block exact.
        """

        try:
            endog = np.asarray(model.endog, dtype=float).reshape(-1)
            exog = np.asarray(model.exog, dtype=float)
            values = np.asarray(params, dtype=float).copy()
            if bool(getattr(model, "_transparams", False)):
                # The backend is on a log scale; the analytic form expects alpha itself.
                values[-1] = float(np.exp(values[-1]))
            return _nb2_analytic_information(values, endog, exog)
        except Exception:  # noqa: BLE001
            return super()._information(model, params)

    _stata_command = "nbreg"
    _optimizer = "bfgs"
    _aux_count = 1

    def _fit_backend(self, model: Any, maxiter: int, tol: float) -> Any:
        """Multi-start fit: a single BFGS run lands in a POOR local optimum for NB2.

        Measured on `nbreg mpg weight foreign` over auto.dta: the default start stops at
        alpha = 1.0513 with log likelihood -302.329467, while the likelihood at alpha -> 0 is
        -196.1299 -- which is what Stata reports (-196.12988). The likelihood itself is right (the
        probe agrees to 1e-4 at alpha = 1e-9); BFGS simply never walks toward the boundary.

        So the fit is restarted from several dispersions and the run with the HIGHEST likelihood
        wins. That is not a guess: the objective is the same every time and only the starting point
        changes, so the selected optimum is the best of the ones actually reached.
        """

        best = super()._fit_backend(model, maxiter, tol)
        best_ll = float(getattr(best, "llf", -np.inf))
        # The start list stops at 1e-9, NOT lower, and 1e-9 is where Stata's own answer sits
        # (alpha = 1.03e-09). Below it the gamma terms in the NB2 likelihood suffer catastrophic
        # cancellation: with the coefficients held fixed, the computed likelihood reads -196.1290
        # at 1e-10, a SPURIOUSLY HIGHER -195.9622 at 1e-12, then collapses to -206.9264 at 1e-14.
        # Selecting the maximum without that bound therefore reports a boundary artefact as the
        # best fit, which is worse than the local optimum it replaced.
        #
        # THE COMPARISON NEEDS A GENEROUS TOLERANCE, because near alpha -> 0 the computed
        # likelihood stops being trustworthy. Measured on the reference count data (true
        # alpha = 6.1e-06, Stata ll = -569.0777629): the default start gives -569.07782106, the
        # 1e-6 start -569.07775974, and the 1e-9 start -569.07729352 -- a gap of 5.3e-04 that is
        # entirely numerical, since the likelihood there is flat in alpha.
        #
        # The auto.dta case that motivated this work gains SIXTY units (-302 -> -196), so a
        # threshold of 1e-3 separates a real improvement from boundary noise by five orders of
        # magnitude in each direction.
        _IMPROVEMENT = 1e-3
        # The coefficients are held at the FIRST fit's values, not at whatever `best` currently is:
        # seeding from `best.params` would carry a boundary dispersion forward and every restart
        # would begin at the boundary it is meant to test.
        _base_params = np.asarray(best.params, dtype=float).copy()
        for start_alpha in (1e-1, 1e-3, 1e-6, 1e-9):
            try:
                start = _base_params.copy()
                start[-1] = start_alpha
                candidate = model.fit(
                    method=self._optimizer,
                    maxiter=maxiter,
                    disp=0,
                    cov_type="nonrobust",
                    use_t=False,
                    start_params=start,
                    gtol=1e-7,
                )
            except Exception:  # pragma: no cover - a start that will not fit
                continue
            candidate_ll = float(getattr(candidate, "llf", -np.inf))
            if np.isfinite(candidate_ll) and candidate_ll > best_ll + _IMPROVEMENT:
                best, best_ll = candidate, candidate_ll
        return best

    def _extra_param_names(self, count: int) -> list[str]:
        # Stata reports nbreg's dispersion as /lnalpha.
        if count == 1:
            return ["/lnalpha"]
        return super()._extra_param_names(count)

    def _rescale_extra_params(
        self,
        params: np.ndarray,
        cov: np.ndarray,
        names: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert statsmodels' ``alpha`` to Stata's ``lnalpha``.

        The likelihood is identical (the dispersion enters as ``1/alpha``), but
        statsmodels estimates ``alpha`` while ``nbreg`` reports ``ln(alpha)``.
        Reporting alpha under a ``/lnalpha`` label would be a units mismatch, so
        the value and its variance are transformed by the delta method.
        """

        if "/lnalpha" not in names:
            return params, cov

        position = names.index("/lnalpha")
        alpha = float(params[position])
        if alpha <= 0:
            return params, cov

        adjusted_params = np.asarray(params, dtype=float).copy()
        adjusted_cov = np.asarray(cov, dtype=float).copy()
        adjusted_params[position] = float(np.log(alpha))

        # delta method: d ln(a)/da = 1/a
        jacobian = 1.0 / alpha
        adjusted_cov[position, :] *= jacobian
        adjusted_cov[:, position] *= jacobian
        return adjusted_params, adjusted_cov

    def _make_sm_model(self) -> Any:
        import statsmodels.api as sm

        return sm.NegativeBinomial(self._y, self._X)

    def _comparison_loglik(self, llf: float) -> float | None:
        """``e(ll_c)``: the Poisson fit of the same regressors.

        ``nbreg.ado:215-228`` fits ``poisson y xvars`` before the negative
        binomial and stores that log-likelihood as ``llp``; ``nbreg.ado:377``
        then sets ``e(ll_c) = llp``.  The Poisson model is the ``alpha = 0``
        special case, so it is the natural comparison for the LR test that
        ``e(chi2_c)`` reports -- a test of whether the dispersion is needed at
        all.

        Verified against ``nbreg.json``: its ``chi2_c`` = 96.45036022672957
        equals ``2*(ll - ll_c)`` = ``2*(-976.791299423244 + 1025.0164795366088)``
        exactly.
        """

        try:
            import statsmodels.api as sm

            poisson = sm.Poisson(self._y, self._X).fit(disp=False, maxiter=200)
            candidate = float(poisson.llf)
        except Exception:  # noqa: BLE001 - omitting the keys is better than guessing
            return None
        return candidate if np.isfinite(candidate) else None

    def _backend_cov(
        self,
        model: Any,
        params: np.ndarray,
        canonical: str,
        cov_kwds: Mapping[str, Any],
    ) -> np.ndarray:
        """Sandwich in ``alpha`` space; the caller delta-transforms it.

        statsmodels' ``NegativeBinomial`` carries a ``_transparams`` flag: when
        true the last parameter is ``ln(alpha)``, and both ``score_obs`` and
        ``hessian`` are written for that scale; when false they expect ``alpha``
        itself.  ``fit`` flips the flag to false and reports ``alpha``, and
        :meth:`_rescale_extra_params` then converts to ``lnalpha``.  The VCE has
        to be taken on the *same* scale the backend reports, so the flag is
        normalised to false here and the delta transform happens exactly once,
        in ``_rescale_extra_params``.
        """

        model._transparams = False
        return super()._backend_cov(model, params, canonical, cov_kwds)


# ------------------------------------------------------------- ordered models
def _to_cut_points(
    params: np.ndarray, cov: np.ndarray, n_cuts: int
) -> tuple[np.ndarray, np.ndarray]:
    """Convert statsmodels' threshold parameterisation into cut points.

    statsmodels stores ``[c1, log(c2 - c1), log(c3 - c2), ...]`` so the
    thresholds stay monotone without constraints; Stata reports
    ``c1, c2, c3, ...`` directly.

    Two separate bugs used to live here, and both are fixed:

    1. ``theta = params[n_slopes:]`` used to be a numpy **view**, so
       ``params[n_slopes:] = cuts`` silently overwrote ``theta`` too and the
       ``exp()`` below was applied to *cut points* rather than log-gaps.
    2. Only the ``(cut, cut)`` block was transformed.  The slope-by-cut
       cross-block ``cov[:ns, ns:]`` was left in the old parameterisation, so
       every ``lincom``/``test`` mixing a slope with a cut was wrong even when
       the cut variances happened to look plausible.

    The Jacobian follows from ``cut_k = theta_0 + sum_{j<=k} exp(theta_j)``
    (``theta_0`` *is* ``c1`` in statsmodels' parameterisation)::

        J[k, 0] = 1
        J[k, j] = exp(theta_j)   for 1 <= j <= k
        J[k, j] = 0              otherwise

    so every cut inherits all of ``theta_0`` but only the earlier increments.
    Verified against StataNow 19.5: with this Jacobian the whole ``e(V)`` --
    slopes, cuts, and both cross-blocks -- reproduces Stata's to ~1e-9.
    """

    params = np.asarray(params, dtype=float).copy()
    cov = np.asarray(cov, dtype=float).copy()
    n_slopes = params.size - n_cuts
    theta = params[n_slopes:].copy()  # copy: ``params`` is overwritten below

    cuts = np.empty(n_cuts, dtype=float)
    cuts[0] = theta[0]
    for k in range(1, n_cuts):
        cuts[k] = cuts[k - 1] + np.exp(theta[k])
    params[n_slopes:] = cuts

    jacobian = np.zeros((n_cuts, n_cuts), dtype=float)
    jacobian[:, 0] = 1.0
    for k in range(1, n_cuts):
        jacobian[k, 1 : k + 1] = np.exp(theta[1 : k + 1])

    transform = np.zeros_like(cov)
    transform[:n_slopes, :n_slopes] = np.eye(n_slopes)
    transform[n_slopes:, n_slopes:] = jacobian
    transformed = transform @ cov @ transform.T
    return params, (transformed + transformed.T) / 2.0


def _density(kind: str, value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(f, f')`` of the standard logistic or standard normal distribution."""

    from scipy import stats

    if kind == "logit":
        cdf = 1.0 / (1.0 + np.exp(-value))
        density = cdf * (1.0 - cdf)
        return density, density * (1.0 - 2.0 * cdf)
    density = stats.norm.pdf(value)
    # ``-value * density`` warns "invalid value encountered in multiply" when ``value``
    # holds +-inf: ``pdf(inf)`` is 0 and ``inf * 0`` is nan. The correct derivative at
    # infinity is 0, and the identity ``phi'(z) = -z phi(z)`` holds in the limit, so the
    # product is computed where it is finite and zeroed elsewhere. This reproduced for any
    # ordered-PROBIT outcome (ologit's logistic branch does not hit it) and is what made a
    # clean run emit a warning on correct numbers.
    with np.errstate(invalid="ignore"):
        derivative = -value * density
    derivative = np.where(np.isfinite(derivative), derivative, 0.0)
    return density, derivative


def _cdf(kind: str, value: np.ndarray) -> np.ndarray:
    from scipy import stats

    limit = np.where(np.isneginf(value), -np.inf, value)
    if kind == "logit":
        out = 1.0 / (1.0 + np.exp(-np.clip(limit, -700.0, 700.0)))
    else:
        out = stats.norm.cdf(limit)
    return np.asarray(out, dtype=float)


class _OrderedInterval:
    """Per-observation quantities of the ordered likelihood at ``params``.

    ``params`` is ``[beta, cut1, cut2, ...]``: already on Stata's scale, which
    is what lets the derivatives below be used directly for Stata's ``e(V)``.
    """

    def __init__(
        self, kind: str, codes: np.ndarray, exog: np.ndarray, params: np.ndarray
    ) -> None:
        self.kind = kind
        self.exog = exog
        n_obs, n_slopes = exog.shape
        self.n_slopes = n_slopes
        self.beta = params[:n_slopes]
        self.cuts = params[n_slopes:]
        n_cuts = self.cuts.size
        # ``codes`` are already 0-based (see ``_category_codes``); the ordered
        # likelihood does not care what the labels are, only their order.
        self.index = np.asarray(codes, dtype=int)
        if self.index.size != n_obs:
            raise ConsistencyError(
                "ordered model: the category codes and the design matrix "
                f"describe different samples ({self.index.size} vs {n_obs})"
            )
        if np.any(self.index < 0) or np.any(self.index > n_cuts):
            raise ConsistencyError(
                "ordered model: the outcome has values outside the fitted "
                "category range"
            )
        self.linear = exog @ self.beta

        self.has_lower = self.index > 0
        self.has_upper = self.index < n_cuts
        lower = np.where(self.has_lower, self.cuts[np.maximum(self.index - 1, 0)], -np.inf)
        upper = np.where(self.has_upper, self.cuts[np.minimum(self.index, n_cuts - 1)], np.inf)
        self.lower = lower
        self.upper = upper

        self.upper_density, self.upper_deriv = _density(kind, upper - self.linear)
        self.lower_density, self.lower_deriv = _density(kind, lower - self.linear)
        self.upper_density[~self.has_upper] = 0.0
        self.upper_deriv[~self.has_upper] = 0.0
        self.lower_density[~self.has_lower] = 0.0
        self.lower_deriv[~self.has_lower] = 0.0

        upper_cdf = _cdf(kind, upper - self.linear)
        lower_cdf = _cdf(kind, lower - self.linear)
        upper_cdf[~self.has_upper] = 1.0
        lower_cdf[~self.has_lower] = 0.0
        self.mass = np.maximum(upper_cdf - lower_cdf, 1e-300)

    @property
    def density_gap(self) -> np.ndarray:
        return self.upper_density - self.lower_density

    @property
    def deriv_gap(self) -> np.ndarray:
        return self.upper_deriv - self.lower_deriv


def _category_codes(endog: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map ordered labels onto ``0..K-1`` by their sorted order.

    Stata orders categories by value, and so does statsmodels'
    ``OrderedModel``; the labels themselves (``0..3``, ``1..4``, ``"low"``...)
    never enter the likelihood.
    """

    values = np.asarray(endog, dtype=float).reshape(-1)
    categories = np.unique(values)
    codes = np.searchsorted(categories, values)
    return codes, categories


def _ordered_loglike(
    kind: str, codes: np.ndarray, exog: np.ndarray, params: np.ndarray
) -> np.ndarray:
    return np.log(_OrderedInterval(kind, codes, exog, params).mass)


def _ordered_score(
    kind: str, codes: np.ndarray, exog: np.ndarray, params: np.ndarray
) -> np.ndarray:
    """Observation-wise score of the ordered likelihood at ``params``."""

    parts = _OrderedInterval(kind, codes, exog, params)
    n_obs = exog.shape[0]
    score = np.zeros((n_obs, params.size), dtype=float)
    factor = parts.density_gap / parts.mass
    score[:, : parts.n_slopes] = -factor[:, None] * exog
    rows = np.arange(n_obs)
    upper_rows = rows[parts.has_upper]
    score[upper_rows, parts.n_slopes + parts.index[parts.has_upper]] += (
        parts.upper_density[parts.has_upper] / parts.mass[parts.has_upper]
    )
    lower_rows = rows[parts.has_lower]
    score[lower_rows, parts.n_slopes + parts.index[parts.has_lower] - 1] -= (
        parts.lower_density[parts.has_lower] / parts.mass[parts.has_lower]
    )
    return score


def _ordered_hessian(
    kind: str, codes: np.ndarray, exog: np.ndarray, params: np.ndarray
) -> np.ndarray:
    """Analytic **observed** Hessian of the ordered likelihood.

    statsmodels' ``OrderedModel`` only supplies a numerical Hessian, and on the
    slope-by-cut cross-blocks that is the accuracy-limiting step: at
    ``epsilon=1e-4`` its ``e(V)`` is 4e-6 away from Stata and the error grows as
    the step shrinks (2.6e-4 at 1e-6).  Writing the Hessian out analytically
    reproduces Stata's ``e(V)`` to ~1e-13 *at Stata's own estimates*, which is
    what makes the 1e-9 target reachable at all.
    """

    parts = _OrderedInterval(kind, codes, exog, params)
    n_obs, n_slopes = exog.shape
    hessian = np.zeros((params.size, params.size), dtype=float)

    weight = (
        parts.deriv_gap * parts.mass - parts.density_gap**2
    ) / parts.mass**2
    hessian[:n_slopes, :n_slopes] = (exog * weight[:, None]).T @ exog

    d_upper = -(
        (parts.upper_deriv * parts.mass - parts.density_gap * parts.upper_density)
        / parts.mass**2
    )
    # The lower bound is ``cut_{j-1} - xb``, so moving that cut *up* shrinks
    # the interval from below: dA/dcut_{j-1} = -f'(l) rather than +f'(l).
    d_lower = (
        parts.lower_deriv * parts.mass - parts.density_gap * parts.lower_density
    ) / parts.mass**2

    rows = np.arange(n_obs)
    for i in rows[parts.has_upper]:
        column = n_slopes + parts.index[i]
        hessian[:n_slopes, column] += d_upper[i] * exog[i]
        hessian[column, :n_slopes] += d_upper[i] * exog[i]
        hessian[column, column] += (
            parts.upper_deriv[i] * parts.mass[i] - parts.upper_density[i] ** 2
        ) / parts.mass[i] ** 2
    for i in rows[parts.has_lower]:
        column = n_slopes + parts.index[i] - 1
        hessian[:n_slopes, column] += d_lower[i] * exog[i]
        hessian[column, :n_slopes] += d_lower[i] * exog[i]
        hessian[column, column] += (
            -(parts.lower_deriv[i] * parts.mass[i] + parts.lower_density[i] ** 2)
            / parts.mass[i] ** 2
        )
    for i in rows[parts.has_upper & parts.has_lower]:
        upper_column = n_slopes + parts.index[i]
        lower_column = upper_column - 1
        value = parts.upper_density[i] * parts.lower_density[i] / parts.mass[i] ** 2
        hessian[upper_column, lower_column] += value
        hessian[lower_column, upper_column] += value
    return hessian


class _Ordered(_DiscreteBase):
    """Shared implementation for ``ologit`` / ``oprobit``.

    The ordered model identifies location through its cut points, so
    statsmodels refuses a design matrix containing a constant.  Stata behaves
    equivalently: ``ologit`` reports no ``_cons``, only ``/cut1``, ``/cut2``,
    ...

    Both the point estimates and the VCE are handled here rather than through
    statsmodels' ``OrderedModel.fit`` covariance: the cut points are
    reparameterised (see :func:`_to_cut_points`) and the Hessian is the analytic
    one, because the numerical alternative cannot reach 1e-9.
    """

    _distribution = "probit"
    _optimizer = "bfgs"

    def __init__(
        self,
        endog: Any,
        exog: Any,
        *,
        param_names: list[str] | None = None,
        freq_weights: Any = None,
    ) -> None:
        super().__init__(endog, exog, param_names=param_names, freq_weights=freq_weights)
        self._codes, self._categories = _category_codes(self._y)
        n_cuts = int(self._categories.size) - 1
        if n_cuts < 1:
            raise ConsistencyError(
                f"{self._stata_command}: the outcome needs at least two "
                "distinct ordered values"
            )
        self._n_cuts = n_cuts
        self._aux_count = n_cuts

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        subset: Any = None,
        freq_weights: str | None = None,
    ) -> "_Ordered":
        model = super().from_formula(
            formula, data, subset=subset, freq_weights=freq_weights
        )
        names = model._param_names
        if "Intercept" in names:
            position = names.index("Intercept")
            model._X = np.delete(model._X, position, axis=1)
            model._param_names = [n for n in names if n != "Intercept"]
        return model

    def _make_sm_model(self) -> Any:
        from statsmodels.miscmodels.ordinal_model import OrderedModel

        return OrderedModel(self._y, self._X, distr=self._distribution)

    # ------------------------------------------------------------ derivatives
    def _stata_params(self, backend_params: np.ndarray) -> np.ndarray:
        """Backend thresholds -> Stata cut points."""

        params, _ = _to_cut_points(
            backend_params, np.eye(backend_params.size), self._n_cuts
        )
        return params

    def _backend_score(self, model: Any, params: np.ndarray) -> np.ndarray:
        return _ordered_score(
            self._distribution, self._codes, self._X, self._stata_params(params)
        )

    def _backend_information(self, model: Any, params: np.ndarray) -> np.ndarray:
        return -_ordered_hessian(
            self._distribution, self._codes, self._X, self._stata_params(params)
        )

    def _newton_polish(
        self, backend_params: np.ndarray, *, max_steps: int = 50, tol: float = 1e-13
    ) -> np.ndarray:
        """Sharpen a quasi-Newton solution with exact Newton steps.

        ``bfgs`` stops on the gradient norm, and on these likelihoods that
        leaves the threshold parameters a few times 1e-5 away from the maximum
        Stata reaches with ``ml``'s Newton-Raphson.  The reported coefficients
        would then be C tier and, because ``e(V)`` is evaluated at the estimate,
        the covariance would inherit the gap.  The analytic gradient and Hessian
        are available here anyway, so a handful of Newton steps (there is
        normally exactly one) puts us on Stata's point: measured on three
        independent samples the coefficient gap drops from 1e-5/1e-7 to
        3e-12..1e-7 and ``e(V)`` from 2e-4 to 1e-10..2e-7.

        Steps are backtracked on the likelihood and rejected outright if the cut
        points would cross, so a degenerate surface degrades to the optimizer's
        answer rather than to nonsense.
        """

        params = np.asarray(backend_params, dtype=float).copy()
        stata = self._stata_params(params)
        current = float(
            _ordered_loglike(self._distribution, self._codes, self._X, stata).sum()
        )
        for _ in range(max_steps):
            gradient = _ordered_score(
                self._distribution, self._codes, self._X, stata
            ).sum(axis=0)
            information = -_ordered_hessian(
                self._distribution, self._codes, self._X, stata
            )
            try:
                step = np.linalg.solve(information, gradient)
            except np.linalg.LinAlgError:
                return params
            size = 1.0
            accepted = False
            while size > 1e-12:
                candidate = stata + size * step
                if np.all(np.diff(candidate[self._X.shape[1] :]) > 0):
                    value = float(
                        _ordered_loglike(
                            self._distribution, self._codes, self._X, candidate
                        ).sum()
                    )
                    if np.isfinite(value) and value >= current:
                        accepted = True
                        break
                size /= 2.0
            if not accepted:
                return params
            stata = candidate
            current = value
            if np.max(np.abs(size * step)) < tol * max(1.0, np.max(np.abs(stata))):
                break
        # Back to statsmodels' parameterisation so the rest of ``fit`` sees the
        # scale it expects.
        return self._backend_from_stata(stata)

    def _backend_from_stata(self, params: np.ndarray) -> np.ndarray:
        """Stata cut points -> statsmodels ``[c1, log-gap, ...]``."""

        n_slopes = params.size - self._n_cuts
        theta = np.empty(self._n_cuts, dtype=float)
        theta[0] = params[n_slopes]
        for k in range(1, self._n_cuts):
            gap = params[n_slopes + k] - params[n_slopes + k - 1]
            theta[k] = np.log(max(gap, 1e-300))
        return np.concatenate([params[:n_slopes], theta])

    def _backend_cov(
        self,
        model: Any,
        params: np.ndarray,
        canonical: str,
        cov_kwds: Mapping[str, Any],
    ) -> np.ndarray:
        """VCE in Stata's cut-point parameterisation, not statsmodels'.

        The derivatives are taken with respect to ``[beta, cut1, cut2, ...]``
        directly, so no delta transform is needed afterwards and the
        slope-by-cut cross-block is automatically consistent.
        """

        groups = _cluster_groups(cov_kwds) if canonical == "cluster" else None
        bread = _inverse_information(
            self._backend_information(model, params), self._stata_command
        )
        score = (
            np.empty((0, params.size))
            if canonical == "nonrobust"
            else self._backend_score(model, params)
        )
        return _vce_from_derivatives(bread, score, canonical, groups)

    def fit(
        self,
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
        *,
        maxiter: int = 500,
        tol: float = 1e-10,
        strict: bool = False,
    ) -> StataResults:
        requested = "nonrobust" if cov_type is None else str(cov_type).strip()
        canonical = self._canonical_cov(requested)
        cov_kwds = dict(cov_kwds or {})
        groups = _cluster_groups(cov_kwds) if canonical == "cluster" else None
        if groups is not None and groups.size != self._y.size:
            raise ConsistencyError(
                f"{self._stata_command}: cov_kwds['groups'] has {groups.size} "
                f"entries but the model has {self._y.size} observations"
            )

        model = self._make_sm_model()
        result = self._fit_backend(model, maxiter, tol)
        self._check_convergence(result, strict)
        # A few exact Newton steps with the analytic gradient/Hessian.  BFGS
        # stops on the gradient norm, which on these flat likelihoods leaves the
        # ``cut`` estimates ~1e-5 away from Stata's; Newton on the analytic
        # derivatives converges quadratically to the point Stata reports, which
        # is what the 1e-9 target needs.
        backend_params = self._newton_polish(np.asarray(result.params, dtype=float))

        # statsmodels orders the slope parameters first, then the cut points.
        # Its threshold parameterisation is NOT the cut points themselves, so
        # recovering them is required for the values to mean the same thing.
        names = list(self._param_names)
        extra = backend_params.size - len(names)
        if extra < 0:
            raise ConsistencyError(
                f"{self._stata_command}: backend returned fewer parameters "
                f"({backend_params.size}) than regressors ({len(names)})"
            )
        if extra != self._n_cuts:
            raise ConsistencyError(
                f"{self._stata_command}: expected {self._n_cuts} cut point(s) "
                f"for {self._n_cuts + 1} ordered categories, the backend "
                f"returned {extra}"
            )
        names = names + [f"/cut{i + 1}" for i in range(extra)]

        params = self._stata_params(backend_params) if extra else backend_params
        cov = self._backend_cov(model, backend_params, canonical, cov_kwds)
        llf = float(
            _ordered_loglike(self._distribution, self._codes, self._X, params).sum()
        )
        nobs = int(self._y.size)

        ll0 = getattr(result, "llnull", None)
        ll_null = float(ll0) if ll0 is not None else float("nan")
        # ``OrderedModel.predict()`` returns a (n_obs, n_categories) matrix of
        # category probabilities; the results container needs one value per
        # observation, so report the expected category index.
        predicted = np.asarray(result.predict(), dtype=float)
        if predicted.ndim == 2 and predicted.shape[0] == nobs:
            categories = np.asarray(self._categories, dtype=float)
            predicted = predicted @ categories
        else:
            predicted = predicted.reshape(-1)

        e: dict[str, Any] = {
            "N": nobs,
            "df_m": len(_slope_positions(names)),
            "k_aux": extra,
            "rank": params.size,
            "ll": llf,
            "ll_0": ll_null,
            "r2_p": (
                1.0 - llf / ll_null
                if np.isfinite(ll_null) and ll_null
                else float("nan")
            ),
            # These were the only ML models in the package without an
            # information criterion under *any* name: ``ll``, ``rank`` and
            # ``N`` are all in hand here, so AIC/BIC (and Stata's ``e(ic)``
            # penalty term) are a two-line addition rather than a computation.
            "aic": float(-2.0 * llf + 2.0 * params.size)
            if np.isfinite(llf)
            else float("nan"),
            "bic": float(-2.0 * llf + np.log(max(nobs, 1)) * params.size)
            if np.isfinite(llf)
            else float("nan"),
            # ``e(ic)`` counts ITERATIONS (``logit.sthlp:257``), not parameters; see
            # ``_iteration_count`` for the fixture evidence that refuted the old formula.
            "ic": _iteration_count(result),
            # ``ologit.ado:270``: ``ereturn local title "Ordered logistic regression"``;
            # ``oprobit.ado:267``: ``"Ordered probit regression"``.  Both use the DIRECT
            # route, unlike poisson/nbreg which pass a ``title()`` option to ``ml``.
            "title": _command_title(str(getattr(self, "_stata_command", "") or "")),
            # ``ologit.sthlp``/``oprobit.sthlp`` document the vce pair too; the ML
            # non-robust default is ``oim`` / ``"OIM"``.  Use the REAL canonical VCE so a
            # robust request is reflected rather than always reporting the default.
            "vce": _ml_vce_label(canonical),
            "vcetype": _ml_vcetype_label(canonical),
            # ``ologit.ado``/``oprobit.ado``: ``ereturn local predict ologit_p`` and
            # ``oprobi_p`` (truncated to 8 characters -- NOT ``oprobit_p``).
            **(
                {
                    "predict": _command_predict_program(
                        str(getattr(self, "_stata_command", "") or "")
                    )
                }
                if _command_predict_program(
                    str(getattr(self, "_stata_command", "") or "")
                )
                else {}
            ),
            # ``k_cat`` is the number of outcome categories Stata prints.
            "k": int(params.size),
            "k_dv": 1,
            "k_eq": 1,
            "k_cat": int(extra) + 1,
            # ``N_cd`` counts observations where the outcome is a "current
            # dependant" censoring pattern; for the ordered models Stata records
            # 0 (``ologit.json``/``oprobit.json`` both have N_cd = 0.0).
            "N_cd": 0,
            # ``version`` is the estimator's internal version stamp; the ordered
            # fixtures carry 3.0.
            "version": 3,
            # The shared parameter-count block, which the ordered models build
            # their own ``e`` dict and so did not inherit from
            # ``_stata_scalars``.
            "k_eq_model": 1,
            "k_autoCns": 0,
            "rc": 0,
            # The real optimiser verdict, not an inference from the estimates:
            # ``_check_convergence`` above already reads it for its warning.
            "converged": 1
            if bool(getattr(result, "mle_retvals", {}).get("converged", True))
            else 0,
        }
        chi2type, statistic, df_m = _model_test(
            params, cov, names, llf, ll_null, canonical != "nonrobust"
        )
        e["chi2"] = statistic
        e["chi2type"] = chi2type
        e["p"] = _chi2_pvalue(statistic, df_m)
        e["df_m"] = df_m
        # ``noconstant`` / ``consonly`` from the reported parameter names, the
        # same derivation ``_stata_scalars`` uses.  Both are real Stata macros even
        # though ``ologit.sthlp`` does not list them -- ``ologit.json`` (captured
        # from Stata) carries them.
        intercept_present = any(
            str(name).strip().lower() in ("_cons", "intercept", "const")
            for name in names
        )
        e["noconstant"] = 0 if intercept_present else 1
        n_slopes = len(_slope_positions(names))
        e["consonly"] = 1 if n_slopes == 0 else 0
        if groups is not None:
            e["N_clust"] = int(np.unique(groups).size)

        #: ``e(depvar)`` (``ologit.sthlp``): ``_Ordered.fit`` builds its own
        #: ``StataResults`` rather than going through ``_DiscreteBase.fit``, so the
        #: wiring there does not reach it.
        response = getattr(self, "_response_name", None)
        if response:
            e["depvar"] = str(response)

        results = StataResults(
            params=params,
            param_names=names,
            cov=cov,
            resid=np.zeros(nobs),
            fittedvalues=predicted,
            nobs=nobs,
            df_resid=float(nobs),
            df_model=float(df_m),
            dist="z",
            model_name=self._stata_command,
            cov_type=requested,
            e=e,
            design_matrix=(self._X, list(self._param_names)),
            model_data=self._model_data,
        )
        if response:
            results.response_name = str(response)
        return results


class OrderedLogit(_Ordered):
    """Ordered logistic regression (Stata ``ologit``)."""

    _stata_command = "ologit"
    _supports_freq_weights = True
    _distribution = "logit"


class OrderedProbit(_Ordered):
    """Ordered probit regression (Stata ``oprobit``)."""

    _stata_command = "oprobit"
    _supports_freq_weights = True
    _distribution = "probit"


class MultinomialLogit(_DiscreteBase):
    """Multinomial logistic regression (Stata ``mlogit``).

    The base outcome's coefficients are normalised to zero and Stata still PRINTS them, as a
    row of zeros in ``e(b)``. That detail matters for matching ``e(b)`` exactly and is why the
    parameter vector here has ``levels * predictors`` entries rather than
    ``(levels - 1) * predictors``.

    Reference from a live ``mlogit y land mci, base(1)`` on a 504-observation sample with three
    outcome levels (see tests/test_mlogit.py for the exact fixture and every coefficient), with
    ``e(ll) = -491.487956370835`` and the coefficient vector

        1:land 1:mci 1:_cons | 2:land 2:mci 2:_cons | 3:land 3:mci 3:_cons
           0     0      0    | .00001555 -1.3655805 1.1794885 | ...
    """

    _stata_command = "mlogit"
    _supports_freq_weights = True
    _is_multinomial = True

    def __init__(
        self,
        endog: Any,
        exog: Any,
        *,
        param_names: list[str] | None = None,
        freq_weights: Any = None,
    ) -> None:
        response = np.asarray(endog, dtype=float).reshape(-1)
        levels = np.unique(response)
        if levels.size < 3:
            raise ConsistencyError(
                f"mlogit: the outcome needs at least three levels; found {levels.size}. "
                "Use Logit for a binary outcome."
            )
        super().__init__(
            endog, exog, param_names=param_names, freq_weights=freq_weights
        )
        self._levels = levels

    def fit(self, cov_type: str | None = None, cov_kwds: Any = None, *, maxiter: int = 200, **kwargs: Any) -> Any:
        from statsmodels.discrete.discrete_model import MNLogit as _MNLogit

        requested = "nonrobust" if cov_type is None else str(cov_type).strip()
        if requested.lower() != "nonrobust":
            raise ValueError(
                f"mlogit: only cov_type='nonrobust' is implemented; got {cov_type!r}"
            )
        design = np.asarray(self._X, dtype=float)
        response = np.asarray(self._y, dtype=float).reshape(-1)
        levels = np.unique(response)
        codes = np.searchsorted(levels, response)
        model = _MNLogit(codes, design)
        result = model.fit(
            method="newton", maxiter=maxiter, disp=False
        )

        # statsmodels estimates (levels - 1) rows; Stata stores `levels` rows with the base
        # row fixed at zero. Rebuild the full vector in Stata's order.
        base_params = np.asarray(result.params, dtype=float)  # (k, levels-1)
        predictors = design.shape[1]
        full = np.zeros((predictors, levels.size))
        full[:, 1:] = base_params
        params = full.T.reshape(-1)  # row-major over levels, as Stata stacks them

        names: list[str] = []
        base_names = (
            list(self._param_names)
            if getattr(self, "_param_names", None) is not None
            else [f"x{index + 1}" for index in range(predictors)]
        )
        for level in levels:
            for name in base_names:
                names.append(f"{int(level)}:{name}")

        # Covariance: statsmodels reports (levels-1)*k; block it the same way.
        base_cov = np.asarray(result.cov_params(), dtype=float)
        size = predictors * levels.size
        covariance = np.zeros((size, size))
        index_map: list[int] = []
        for level_index in range(levels.size):
            for predictor in range(predictors):
                if level_index == 0:
                    index_map.append(-1)
                else:
                    index_map.append(
                        (level_index - 1) * predictors + predictor
                    )
        for row in range(size):
            if index_map[row] < 0:
                continue
            for column in range(size):
                if index_map[column] < 0:
                    continue
                covariance[row, column] = base_cov[index_map[row], index_map[column]]

        llf = float(result.llf)
        nobs = int(response.size)
        return StataResults(
            params=params,
            param_names=names,
            cov=covariance,
            # MNLogit returns an (n, levels) probability matrix; the shared results object
            # wants one fitted value per observation, so the expected level is used.
            resid=np.asarray(
                response - (result.predict() * levels).sum(axis=1), dtype=float
            ),
            fittedvalues=np.asarray(
                (result.predict() * levels).sum(axis=1), dtype=float
            ),
            nobs=nobs,
            df_resid=float(nobs - len(names)),
            df_model=float(len(names)),
            dist="z",
            model_name=self._stata_command,
            cov_type="nonrobust",
            e={
                "N": float(nobs),
                "ll": llf,
                "rank": float(len(names)),
                "df_m": float(len(names)),
                "df_r": float(nobs - len(names)),
                "k": float(len(names)),
                "k_eq": float(levels.size),
                "k_dv": float(levels.size),
                "title": "Multinomial logistic regression",
                "converged": 1.0,
                "chi2type": "LR",
                "baseoutcome": float(levels[0]),
                "width": 78.0,
            },
        )
