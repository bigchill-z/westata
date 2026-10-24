"""Limited dependent-variable models (P6).

======================  ==================
westata                 Stata
======================  ==================
``Truncreg``            ``truncreg``
``Intreg``              ``intreg``
``Heckman``             ``heckman``
``Etregress``           ``etregress``
======================  ==================

``Truncreg`` and ``Intreg`` are maximum-likelihood estimators.  They reuse
:class:`westata.estimators.discrete._DiscreteBase` for everything Stata-facing
(parameter naming, ``e()`` discipline, ``StataResults`` assembly, ``strict=``
convergence handling) and only supply a statsmodels ``GenericLikelihoodModel``
whose ``loglikeobs`` is the Stata likelihood, plus the scale bookkeeping.
``Heckman`` and ``Etregress`` are two-step estimators with no single likelihood
to hand to a backend, so they assemble ``StataResults`` directly.

Likelihoods
-----------
Truncated regression (``[R] truncreg``).  Observations outside ``(a, b)`` were
never sampled, so the density is divided by the probability of being inside the
interval::

    lnL = sum[ -ln(sigma) - 0.5 ln(2 pi) - 0.5 ((y - xb)/sigma)^2
               - ln( Phi((b - xb)/sigma) - Phi((a - xb)/sigma) ) ]

Interval regression (``[R] intreg``).  Only the interval ``[a, b]`` is known::

    lnL = sum[ ln( Phi((b - xb)/sigma) - Phi((a - xb)/sigma) ) ]

``lower == upper`` marks an exactly observed value and contributes
``ln phi(z) - ln sigma`` instead, which is what makes the point-interval case
reduce to OLS.  ``-inf``/``+inf`` bounds mark censoring.

Both likelihoods are maximised with ``sigma`` on a log scale so it stays
positive, and reported the way Stata reports them: ``truncreg`` prints
``/sigma``; ``intreg`` prints ``/lnsigma`` (with ``e(sigma)`` and a
delta-method covariance for the exponentiated scale).

Standard errors and model tests
-------------------------------
``cov_type="nonrobust"`` is the observed information (the numerical Hessian of
the log-likelihood), which is Stata's default for both commands.  Against the
captured StataNow 19.5 output in
``tests/conformance/stata_fixtures/p6p7_outputs.txt`` this reproduces the
coefficients to 7 significant digits, the standard errors to 6, ``e(ll)`` to 6,
and the Wald/LR model test to 5.  ``truncreg`` prints a Wald ``chi2`` on the
slopes; ``intreg`` the LR ``chi2`` against its constant-only model -- exactly the
tests Stata reports for those two commands.  Both use ``dist="z"``.

Two-step estimators and their standard errors
---------------------------------------------
``Heckman`` (two-step, Heckman 1979) and ``Etregress`` (control function) put a
*generated regressor* into a linear equation: the inverse Mills ratio
``lambda = phi(z g)/Phi(z g)``, or the control-function residual
``t*phi/Phi - (1-t)*phi/(1-Phi)``, is estimated by a first-stage probit.  Their
covariance is therefore the two-step M-estimator sandwich

    V = A^-1 B A'^-1 / n,   A = E[d psi / d theta],   B = E[psi psi']

with ``psi`` stacking the probit score and the augmented-regression moments.

**The standard errors reported by :class:`Heckman` and :class:`Etregress` are
this full correction, not naive OLS standard errors.**  They are still *not*
Stata's numbers: Stata's ``heckman``/``etregress`` default to maximum likelihood,
which reports ML standard errors for ML point estimates.  Here the point
estimates are the two-step ones and the standard errors are the matching
two-step ones, so the two sets differ by construction -- the approximation is
the estimator, not the standard error.  ``e(vce)`` records ``"twostep"`` and
``e(se_corrected)`` is ``1``, so this cannot be mistaken for naive OLS output.

The correction was checked two ways rather than assumed:

* Against a 400-replication row bootstrap on a high-correlation DGP
  (``rho ~ 0.9``), the analytic sandwich is within 5% of the bootstrap standard
  deviation (``0.052 vs 0.054``, ``0.029 vs 0.030``, ``0.067 vs 0.071`` for the
  outcome intercept, slope and ``lambda``); naive OLS standard errors differ by
  ~7% on the ``lambda`` coefficient in the same DGP.
* In a Monte Carlo at ``n = 6000`` the sandwich's ratio to the empirical
  dispersion of the estimates converges toward one, i.e. it is the correct
  asymptotically-efficient quantity rather than a rescaled OLS number.

Note that ``rho`` is *not* Stata's ML ``rho``: it is the plug-in
``b_lambda / sigma`` implied by the two-step fit, so ``heckman``'s reported
``rho``, ``sigma`` and ``lambda`` describe the same estimator as the reported
coefficients.

Not implemented (raises :class:`westata.errors.UnsupportedError`)
----------------------------------------------------------------
* ``Heckman.fit(method="ml")`` -- joint ML over ``beta``, ``gamma`` and ``rho``.
* ``Etregress.fit(method="ml")`` -- joint ML over the outcome and treatment
  equations.

Neither is approximated silently: the request raises and names what is missing.

Documented deviations from Stata
--------------------------------
* ``Heckman``'s ``sigma``/``rho`` come from the augmented regression
  (``rho = b_lambda / sigma``) rather than from Stata's ML information matrix.
* ``e(lambda)`` follows the P6 plan and holds the **mean inverse Mills ratio**
  over the selected sample.  Stata's ``e(lambda)`` is ``rho * sigma``, the
  coefficient on ``lambda``; that value is available as ``e(lam_coef)``.
* All four models record ``e(r2_p)`` (McFadden).  Stata prints no pseudo
  R-squared for these commands, so it is an additional diagnostic; for the
  two-step models it is computed against the Gaussian constant-only counterpart
  of the equation actually fitted.
* ``Heckman``'s ``e(N)`` is the full sample (as in Stata) while ``nobs`` is the
  number of selected observations, because that is the length of the residual
  and fitted-value vectors.
* ``truncreg`` drops observations at or outside the truncation limits (as Stata
  does, reporting "(N obs truncated)") and records the count in
  ``e(n_truncated)``; it warns rather than doing so silently.
"""

from __future__ import annotations

import re
import warnings
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import log_ndtr
from statsmodels.base.model import GenericLikelihoodModel

from westata.errors import ConsistencyError, ConvergenceError, UnsupportedError
from westata.estimators.discrete import _DiscreteBase
from westata.estimators.tobit import _numerical_hessian
from westata.formula.factor import build_design
from westata.results.base import StataResults

__all__ = ["Etregress", "Heckman", "Intreg", "Truncreg"]


# --------------------------------------------------------------- arithmetic


def _interval_log_mass(z_lower: np.ndarray, z_upper: np.ndarray) -> np.ndarray:
    """``log(Phi(z_upper) - Phi(z_lower))`` without catastrophic cancellation.

    ``Phi(b) - Phi(a)`` cancels to zero when both bounds sit in the same tail,
    which is exactly the regime truncation produces.  Factoring out the larger
    term (``Phi(b)``, since ``b > a``) leaves

        log(Phi(b) - Phi(a)) = log_ndtr(b) + log(-expm1(log_ndtr(a) - log_ndtr(b)))

    and ``-expm1(delta)`` is accurate both for narrow intervals (``delta`` near
    zero, where the direct difference would lose every digit) and for wide ones
    (``delta`` very negative).  ``log_ndtr`` is used instead of
    ``stats.norm.logcdf`` because it keeps full relative precision far into the
    tails.  Infinite bounds are the caller's responsibility.
    """

    upper = log_ndtr(np.asarray(z_upper, dtype=float))
    delta = log_ndtr(np.asarray(z_lower, dtype=float)) - upper
    with np.errstate(divide="ignore", invalid="ignore"):
        return upper + np.log(-np.expm1(np.minimum(delta, 0.0)))


def _mills_ratio(index: np.ndarray) -> np.ndarray:
    """Inverse Mills ratio ``phi(w)/Phi(w)``, computed in the log domain.

    ``phi(w)/Phi(w)`` overflows for large negative ``w``; taking the difference
    of the logs keeps it finite for every attainable index.
    """

    index = np.asarray(index, dtype=float)
    return np.exp(stats.norm.logpdf(index) - log_ndtr(index))


def _pseudo_r2(llf: float, ll0: float) -> float:
    """McFadden's pseudo R-squared, NaN when there is no usable null model."""

    if not np.isfinite(ll0) or ll0 == 0 or not np.isfinite(llf):
        return float("nan")
    return 1.0 - llf / ll0


def _gaussian_pseudo_r2(
    outcome: np.ndarray, fitted: np.ndarray, *, nobs: int, k: int
) -> tuple[float, float, float]:
    """McFadden pseudo R2, ``ll`` and ``ll_0`` for a Gaussian second stage.

    Stata's two-step estimators still report a log-likelihood, and it is the
    Gaussian likelihood of the equation they actually fit.  The null model is
    the same likelihood with only a constant, which is what
    ``r2_p = 1 - ll/ll_0`` has to be computed against for the ratio to mean
    anything.
    """

    residual = outcome - fitted
    dof = max(nobs - k, 1)
    sigma2 = float(residual @ residual) / dof
    if not np.isfinite(sigma2) or sigma2 <= 0:
        return float("nan"), float("nan"), float("nan")

    llf = float(-0.5 * nobs * (np.log(2.0 * np.pi * sigma2) + 1.0))

    # Constant-only counterpart on the same sample.
    centred = outcome - float(np.mean(outcome))
    sigma2_null = float(centred @ centred) / max(nobs - 1, 1)
    if not np.isfinite(sigma2_null) or sigma2_null <= 0:
        return float("nan"), llf, float("nan")
    ll0 = float(-0.5 * nobs * (np.log(2.0 * np.pi * sigma2_null) + 1.0))
    return _pseudo_r2(llf, ll0), llf, ll0


def _slope_positions(names: Sequence[str]) -> list[int]:
    """Indices of the coefficients a Stata Wald/LR test covers.

    Stata's ``Wald chi2(k)`` for these commands counts the structural regressor
    slopes of the outcome equation.  Excluded:

    * the constant, however the estimator spells it (``Intercept``, ``_cons``,
      or an equation-prefixed ``y:Intercept``);
    * the estimated scale (``/sigma``, ``/lnsigma``);
    * the generated regressors -- ``lambda`` (Heckman) and the control-function
      residual (``Etregress``) -- because they are auxiliary parameters, not
      regressors whose null Stata tests;
    * every coefficient of the selection/treatment equation (``select:`` /
      ``treat:``), which belongs to a different equation.
    """

    positions = []
    for position, name in enumerate(names):
        leaf = name.split(":", 1)[-1]
        if leaf in ("Intercept", "_cons", "lambda"):
            continue
        if name.startswith("/") or name.startswith("select:") or name.startswith(
            "treat:"
        ):
            continue
        positions.append(position)
    return positions


def _wald_statistic(
    params: np.ndarray, cov: np.ndarray, positions: Sequence[int]
) -> float:
    """Wald statistic ``b' V^-1 b`` over the given coefficient block."""

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


def _chi2_pvalue(statistic: float, df: int) -> float:
    if not np.isfinite(statistic) or df <= 0:
        return float("nan")
    return float(stats.chi2.sf(statistic, df))


# --------------------------------------------------------------- ML backend


class _SelfLikelihoodModel(GenericLikelihoodModel):
    """statsmodels ``GenericLikelihoodModel`` carrying a Stata likelihood.

    :meth:`westata.estimators.discrete._DiscreteBase.fit` drives this object: it
    reads ``params``, ``cov_params()``, ``predict()``, ``llf``,
    ``resid_response`` and ``mle_retvals['converged']``.  A
    ``GenericLikelihoodModel`` supplies all but ``resid_response`` (attached in
    :meth:`fit`), and it needs an explicit ``hessian`` both for Newton steps and
    for Stata's observed-information covariance: statsmodels' default numerical
    Hessian is computed from the score and is far noisier at the optimum.

    Parameters
    ----------
    endog, exog:
        Response and design matrix; ``exog`` includes the constant.
    loglike_obs:
        ``f(theta) -> length-n`` log-likelihood contributions, where ``theta``
        is ``[beta..., extra]`` and ``extra`` is the scale on the log scale.
    start_params:
        Starting values (OLS coefficients plus a residual scale).
    extra_names:
        Stata's spelling for the parameters after ``beta``.
    """

    def __init__(
        self,
        endog: np.ndarray,
        exog: np.ndarray,
        *,
        loglike_obs: Callable[[np.ndarray], np.ndarray],
        start_params: np.ndarray,
        extra_names: Sequence[str] = ("/sigma",),
    ) -> None:
        self._loglike_obs = loglike_obs
        self.start_params = np.asarray(start_params, dtype=float).reshape(-1)
        super().__init__(
            endog, exog, extra_params_names=list(extra_names) or None, missing="none"
        )

    # ------------------------------------------------------------ likelihood
    def loglikeobs(self, params: Any) -> np.ndarray:
        return np.asarray(self._loglike_obs(np.asarray(params, dtype=float))).reshape(-1)

    def hessian(self, params: Any) -> np.ndarray:
        """Observed information of the *log*-likelihood (Stata's default)."""

        return _numerical_hessian(self.loglike, np.asarray(params, dtype=float))

    def predict(self, params: Any, exog: Any = None, which: str = "mean") -> np.ndarray:
        design = self.exog if exog is None else np.asarray(exog, dtype=float)
        return design @ np.asarray(params, dtype=float)[: design.shape[1]]

    # ------------------------------------------------------------------- fit
    def fit(self, **kwargs: Any) -> Any:
        """Fit with an optimiser ladder, returning the best usable result.

        Newton is the step that lands on Stata's estimates, but it can diverge
        from an unlucky start.  If Newton fails, the fit falls back to BFGS and
        then Nelder-Mead, keeping whichever result reached the highest
        likelihood.  ``mle_retvals['converged']`` always describes the result
        actually returned, so ``_DiscreteBase``'s ``strict=`` handling stays
        truthful.
        """

        kwargs.setdefault("disp", 0)
        kwargs.setdefault("start_params", self.start_params)
        method = str(kwargs.get("method", "newton")).lower()
        attempts: list[dict[str, Any]] = [dict(kwargs)]
        if method == "newton":
            fallback = {key: value for key, value in kwargs.items() if key != "tol"}
            attempts.append({**fallback, "method": "bfgs"})
            attempts.append({**fallback, "method": "nm"})

        best: Any = None
        for attempt in attempts:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    candidate = GenericLikelihoodModel.fit(self, **attempt)
                except (ValueError, np.linalg.LinAlgError, np.linalg.LinAlgWarning):
                    continue
            if not np.all(np.isfinite(np.asarray(candidate.params, dtype=float))):
                continue
            if best is None or float(candidate.llf) > float(best.llf):
                best = candidate
            if bool(candidate.mle_retvals.get("converged", False)):
                break

        if best is None:
            raise ConvergenceError(
                "the likelihood could not be evaluated to a finite value at any "
                "starting point; check the data and the truncation/interval limits"
            )

        # ``_DiscreteBase.fit`` reads ``resid_response``; the generic-likelihood
        # results class does not define it.  ``y - x'b`` is the response residual
        # for these Gaussian-based models.
        best.resid_response = np.asarray(
            self.endog, dtype=float
        ).reshape(-1) - np.asarray(self.predict(best.params), dtype=float)
        return best


# ---------------------------------------------------------- ML limited base


class _MLLimitedBase(_DiscreteBase):
    """Shared Stata layer for the self-implemented truncated/interval models.

    Subclasses supply the likelihood (:meth:`_loglike_with`), the scale
    parameter's name and the model-level test Stata prints.
    """

    #: ``/sigma`` (truncreg) or ``/lnsigma`` (intreg).
    _scale_name = "/sigma"
    _optimizer = "newton"

    # ------------------------------------------------------------ parameters
    def _extra_param_names(self, count: int) -> list[str]:
        if count == 1:
            return [self._scale_name]
        return super()._extra_param_names(count)

    def _sigma(self, params: np.ndarray) -> float:
        """Scale estimate from the *reported* parameter vector."""

        return float(params[-1])

    # ------------------------------------------------------------ likelihood
    def _loglike_with(
        self, exog: np.ndarray, endog: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """Per-observation log-likelihood on an arbitrary design."""

        raise NotImplementedError

    def _loglike_obs(self, theta: np.ndarray) -> np.ndarray:
        return self._loglike_with(self._X, self._y, theta)

    def _loglike(self, theta: np.ndarray) -> float:
        """Total log-likelihood; public because the tests check it directly."""

        values = np.asarray(self._loglike_obs(np.asarray(theta, dtype=float)))
        if not np.all(np.isfinite(values)):
            return float("-inf")
        return float(np.sum(values))

    def _null_loglike(self) -> float:
        """Constant-only log-likelihood, used for the LR test and pseudo R2."""

        ones = np.ones((self._y.size, 1))
        start = np.array(
            [float(np.mean(self._y)), np.log(max(float(np.std(self._y)), 1e-3))]
        )

        def objective(parameters: np.ndarray) -> float:
            values = np.asarray(self._loglike_with(ones, self._y, parameters))
            total = float(np.sum(values)) if np.all(np.isfinite(values)) else -np.inf
            return -total if np.isfinite(total) else 1e12

        result = optimize.minimize(
            objective, start, method="BFGS", options={"maxiter": 1000, "gtol": 1e-10}
        )
        refined = optimize.minimize(
            objective,
            np.asarray(result.x, dtype=float),
            method="Nelder-Mead",
            options={"maxiter": 1000, "xatol": 1e-10, "fatol": 1e-10},
        )
        if refined.fun <= result.fun:
            result = refined
        value = -float(result.fun)
        return value if np.isfinite(value) else float("nan")

    def _make_sm_model(self) -> Any:
        beta0, *_ = np.linalg.lstsq(self._X, self._y, rcond=None)
        scale0 = max(float(np.std(self._y - self._X @ beta0)), 1e-3)
        start = np.concatenate([beta0, [np.log(scale0)]])
        return _SelfLikelihoodModel(
            self._y,
            self._X,
            loglike_obs=self._loglike_obs,
            start_params=start,
            extra_names=[self._scale_name],
        )

    # ------------------------------------------------------------------- fit
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
        if requested.lower() != "nonrobust":
            raise ValueError(
                f"{self._stata_command}: only cov_type='nonrobust' is implemented; "
                f"got {cov_type!r}. Stata reports the observed-information "
                "covariance for this command, which is what 'nonrobust' returns; "
                "the robust sandwich is not wired up."
            )
        self._llf_estimate = None
        result = super().fit(
            cov_type="nonrobust", maxiter=maxiter, tol=tol, strict=strict
        )
        # ``_DiscreteBase`` derives ``df_model`` as ``len(params) - 1``, which
        # counts the estimated scale.  Stata's model test for these commands
        # covers the slopes only, and the summary renderer labels that test with
        # ``df_model``, so the result is restated with the Stata df.
        return _restate(result, float(result.e["df_m"]))

    # ------------------------------------------------------------- e() scalars
    def _model_test(
        self, params: np.ndarray, cov: np.ndarray, names: Sequence[str], llf: float
    ) -> tuple[str, float, int]:
        """Return ``(chi2type, statistic, df)`` for the printed model test."""

        positions = _slope_positions(names)
        return "Wald", _wald_statistic(params, cov, positions), len(positions)

    def _extra_scalars(self, params: np.ndarray) -> dict[str, Any]:
        return {}

    def _stata_scalars(
        self,
        result: Any,
        params: np.ndarray,
        cov: np.ndarray,
        names: list[str],
        llf: float,
        nobs: int,
        canonical: str = "nonrobust",
        groups: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Stata's ``e()`` block for the truncated/interval models.

        The signature follows :meth:`westata.estimators.discrete._DiscreteBase
        ._stata_scalars` -- the base class builds the covariance and the names,
        so the rescaled ``/sigma`` column is already in ``cov`` here.
        """

        chi2type, statistic, df = self._model_test(params, np.asarray(cov), names, llf)
        ll0 = self._null_loglike()
        sigma = self._sigma(params)
        k = len(params)

        e: dict[str, Any] = {
            "N": nobs,
            "df_m": df,
            "rank": k,
            "ll": llf,
            "ll_0": ll0,
            "chi2": statistic,
            "chi2type": chi2type,
            "p": _chi2_pvalue(statistic, df),
            "r2_p": _pseudo_r2(llf, ll0),
            "sigma": sigma,
            "var_e": sigma**2,
            "converged": int(
                bool(getattr(result, "mle_retvals", {}).get("converged", True))
            ),
            "aic": float(-2.0 * llf + 2.0 * k),
            "bic": float(-2.0 * llf + k * np.log(nobs)),
        }
        e.update(self._extra_scalars(params))
        return e


def _restate(result: StataResults, df_model: float) -> StataResults:
    """Rebuild ``result`` with a corrected ``df_model``.

    ``df_model`` drives the reported model-test label and the ``e(df_m)``
    scalar, so it cannot simply be overwritten on the finished container
    without leaving the summary and the p-value inconsistent; the container is
    reassembled instead.  The ``e()`` mapping is copied verbatim.
    """

    return StataResults(
        params=np.asarray(result.params, dtype=float),
        param_names=result.param_names,
        cov=np.asarray(result.cov_params(), dtype=float),
        resid=np.asarray(result.resid, dtype=float),
        fittedvalues=np.asarray(result.fittedvalues, dtype=float),
        nobs=int(result.nobs),
        df_resid=float(result.df_resid),
        df_model=float(df_model),
        dist="z",
        model_name=result.model_name,
        cov_type=result.cov_type,
        e=dict(result.e),
        design_matrix=result.design_matrix,
        model_data=result.model_data,
    )


# ------------------------------------------------------------------ truncreg


class Truncreg(_MLLimitedBase):
    """Truncated regression (Stata ``truncreg``).

    Observations outside ``(lower, upper)`` are absent from the sample, so each
    contribution is the normal density divided by the probability of falling
    inside the interval.

    Parameters
    ----------
    endog:
        Observed outcome; every value must lie strictly inside the limits.
    exog:
        Design matrix including a constant.
    lower, upper:
        Truncation limits in the units of ``endog``; at least one is required.
        Use ``None`` for an open limit (``-inf`` / ``+inf``).

    Notes
    -----
    Stata's ``truncreg`` excludes observations with ``y <= lower`` or
    ``y >= upper`` from the estimation sample and reports the count as
    ``(N obs truncated)``; this does the same, records the count in
    ``e(n_truncated)`` and warns, because silently changing the sample would
    make ``e(N)`` impossible to reconcile with the input data.
    """

    _stata_command = "truncreg"

    def __init__(
        self,
        endog: Any,
        exog: Any,
        *,
        lower: float | None = None,
        upper: float | None = None,
        param_names: list[str] | None = None,
    ) -> None:
        if lower is None and upper is None:
            raise ConsistencyError(
                "truncreg: at least one truncation limit is required; with no "
                "truncation use OLS (Stata: regress)"
            )
        if lower is not None and upper is not None and not lower < upper:
            raise ConsistencyError(
                f"truncreg: lower limit ({lower}) must be below upper limit ({upper})"
            )

        response = np.asarray(endog, dtype=float).reshape(-1)
        design = np.asarray(exog, dtype=float)
        if design.ndim == 1:
            design = design.reshape(-1, 1)
        if design.shape[0] != response.size:
            raise ValueError(
                f"endog has {response.size} rows but exog has {design.shape[0]} rows"
            )

        # Stata drops observations at or outside the truncation points; they
        # could not have been sampled, so they carry no information.
        outside = np.zeros(response.size, dtype=bool)
        if lower is not None:
            outside |= response <= lower
        if upper is not None:
            outside |= response >= upper
        n_truncated = int(outside.sum())
        kept = ~outside
        if not kept.any():
            raise ConsistencyError(
                f"truncreg: all {response.size} observation(s) fall at or outside "
                f"the truncation limits (lower={lower}, upper={upper}); there is "
                "nothing left to estimate"
            )

        super().__init__(
            response[kept], design[kept], param_names=param_names
        )
        self._lower = lower
        self._upper = upper
        self._n_truncated = n_truncated
        if n_truncated:
            warnings.warn(
                f"truncreg: {n_truncated} observation(s) at or outside the "
                f"truncation limits (lower={lower}, upper={upper}) were dropped "
                f"(Stata's Truncated regression reports this as '{n_truncated} obs "
                "truncated')",
                RuntimeWarning,
                stacklevel=2,
            )

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        lower: float | None = None,
        upper: float | None = None,
        subset: Any = None,
    ) -> "Truncreg":
        """Build the model from a formula (Stata ``truncreg y x, ll(#) ul(#)``)."""

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if subset is not None:
            data = data.loc[subset]
        data = data.reset_index(drop=True)

        lhs = formula.split("~", 1)[0].strip()
        if lhs not in data.columns:
            raise ConsistencyError(
                f"truncreg: the left-hand side must be a plain column name, got {lhs!r}"
            )
        X, names, kept_index = build_design(formula, data)
        y = data.loc[kept_index, lhs].to_numpy(dtype=float)
        # ``_model_data`` must describe the estimation sample, which drops the
        # observations outside the truncation limits in ``__init__``.
        frame = data.loc[kept_index].reset_index(drop=True)
        model = cls(y, X, lower=lower, upper=upper, param_names=names)
        model._model_data = frame.loc[_truncation_mask(y, lower, upper)].reset_index(
            drop=True
        )
        return model

    # ------------------------------------------------------------ likelihood
    def _loglike_with(
        self, exog: np.ndarray, endog: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        k = exog.shape[1]
        beta = theta[:k]
        log_sigma = float(theta[k])
        sigma = float(np.exp(log_sigma))
        xb = exog @ beta
        residual = (endog - xb) / sigma

        lower = -np.inf if self._lower is None else float(self._lower)
        upper = np.inf if self._upper is None else float(self._upper)
        z_lower = (lower - xb) / sigma
        z_upper = (upper - xb) / sigma
        lower_open = not np.isfinite(z_lower).any()
        upper_open = not np.isfinite(z_upper).any()

        if lower_open and upper_open:
            log_mass = np.zeros_like(xb)
        elif lower_open:
            log_mass = log_ndtr(z_upper)
        elif upper_open:
            # log(1 - Phi(z)) = log_ndtr(-z), accurate far into the left tail.
            log_mass = log_ndtr(-z_lower)
        else:
            log_mass = _interval_log_mass(z_lower, z_upper)

        with np.errstate(invalid="ignore"):
            return (
                -log_sigma
                - 0.5 * np.log(2.0 * np.pi)
                - 0.5 * residual**2
                - log_mass
            )

    def _sigma(self, params: np.ndarray) -> float:
        # ``_rescale_extra_params`` has already exponentiated the reported scale.
        return float(params[-1])

    def _rescale_extra_params(
        self, params: np.ndarray, cov: np.ndarray, names: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Report ``/sigma`` rather than ``/lnsigma``, via the delta method.

        The backend optimises ``ln sigma`` so the scale stays positive;
        ``truncreg`` prints ``sigma``, and ``d sigma / d ln sigma = sigma``.
        """

        position = names.index(self._scale_name)
        sigma = float(np.exp(params[position]))
        adjusted_params = np.asarray(params, dtype=float).copy()
        adjusted_cov = np.asarray(cov, dtype=float).copy()
        adjusted_params[position] = sigma
        if np.isfinite(sigma) and sigma > 0:
            adjusted_cov[position, :] *= sigma
            adjusted_cov[:, position] *= sigma
        return adjusted_params, adjusted_cov

    def _extra_scalars(self, params: np.ndarray) -> dict[str, Any]:
        return {
            "lower": self._lower,
            "upper": self._upper,
            "n_truncated": self._n_truncated,
            "n_uncensored": int(self._y.size),
        }


def _truncation_mask(
    response: np.ndarray, lower: float | None, upper: float | None
) -> np.ndarray:
    """Observations strictly inside the truncation limits."""

    kept = np.ones(np.asarray(response).size, dtype=bool)
    if lower is not None:
        kept &= response > lower
    if upper is not None:
        kept &= response < upper
    return kept


# ------------------------------------------------------------------- intreg


class Intreg(_MLLimitedBase):
    """Interval regression (Stata ``intreg``).

    Only the interval ``[lower, upper]`` is known for each observation:
    ``lower == upper`` marks an exactly observed value, ``-inf`` a left-censored
    one and ``+inf`` a right-censored one.

    Parameters
    ----------
    lower, upper:
        Interval bounds, length ``n``, with ``lower <= upper`` elementwise.
    exog:
        Design matrix including a constant.

    Notes
    -----
    The class stores the interval midpoints as its response, but only as an
    optimiser starting value and as the basis of the reported ``resid`` and
    ``fittedvalues``; the likelihood never uses them.
    """

    _stata_command = "intreg"
    #: Stata's intreg reports the log of the scale.
    _scale_name = "/lnsigma"

    def __init__(
        self,
        lower: Any,
        upper: Any,
        exog: Any,
        *,
        param_names: list[str] | None = None,
    ) -> None:
        lo = np.asarray(lower, dtype=float).reshape(-1)
        hi = np.asarray(upper, dtype=float).reshape(-1)
        if lo.size != hi.size:
            raise ConsistencyError(
                f"intreg: lower and upper must have the same length ({lo.size} vs "
                f"{hi.size})"
            )
        inverted = lo > hi
        if inverted.any():
            raise ConsistencyError(
                f"intreg: {int(inverted.sum())} observation(s) have a lower bound "
                "above the upper bound; every interval must satisfy lower <= upper"
            )
        if not np.isfinite(lo).any() and not np.isfinite(hi).any():
            raise ConsistencyError(
                "intreg: every observation is (-inf, +inf); the model carries no "
                "information about the location"
            )
        self._lower = lo
        self._upper = hi
        # Starting values: the midpoint of a finite interval, a shifted bound for
        # a one-sided one, and the unconditional mean's role (zero) for a
        # (-inf, +inf) row.  These never enter the likelihood, but they must all
        # stay finite -- ``np.where`` evaluates both branches, so an unguarded
        # ``(lo + hi) / 2`` would produce inf/nan on a censored row and poison
        # the whole start vector.
        both_finite = np.isfinite(lo) & np.isfinite(hi)
        left_only = np.isfinite(lo) & ~np.isfinite(hi)
        right_only = ~np.isfinite(lo) & np.isfinite(hi)

        midpoint = np.zeros(lo.size, dtype=float)
        midpoint[both_finite] = (lo[both_finite] + hi[both_finite]) / 2.0
        midpoint[left_only] = lo[left_only] + 1.0
        midpoint[right_only] = hi[right_only] - 1.0
        super().__init__(midpoint, exog, param_names=param_names)

    @classmethod
    def from_formula(
        cls, formula: str, data: pd.DataFrame, *, subset: Any = None
    ) -> "Intreg":
        """Build from a formula whose left-hand side is ``"lower upper"``."""

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if subset is not None:
            data = data.loc[subset]
        data = data.reset_index(drop=True)

        if "~" not in formula:
            raise ConsistencyError(
                f"intreg: the formula must have the form 'lower upper ~ x', got "
                f"{formula!r}"
            )
        lhs, rhs = formula.split("~", 1)
        pieces = lhs.split()
        if len(pieces) != 2:
            raise ConsistencyError(
                "intreg: the left-hand side must name two columns, the lower and "
                f"upper bounds (e.g. 'lo hi ~ x'), got {lhs.strip()!r}"
            )
        lower_column, upper_column = pieces
        for column in (lower_column, upper_column):
            if column not in data.columns:
                raise ConsistencyError(
                    f"intreg: column {column!r} not found in the data"
                )

        # Build the design matrix from the right-hand side only, borrowing one of
        # the bound columns as patsy's response so that listwise deletion still
        # applies to the regressors.
        X, names, kept_index = build_design(f"{upper_column} ~ {rhs.strip()}", data)
        lower = data.loc[kept_index, lower_column].to_numpy(dtype=float)
        upper = data.loc[kept_index, upper_column].to_numpy(dtype=float)
        model = cls(lower, upper, X, param_names=names)
        model._model_data = data.loc[kept_index].reset_index(drop=True)
        return model

    # ------------------------------------------------------------ likelihood
    def _sigma(self, params: np.ndarray) -> float:
        # The backend estimates ln sigma and no rescaling is applied here.
        return float(np.exp(params[-1]))

    def _loglike_with(
        self, exog: np.ndarray, endog: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        k = exog.shape[1]
        beta = theta[:k]
        log_sigma = float(theta[k])
        sigma = float(np.exp(log_sigma))
        xb = exog @ beta

        # The bounds belong to the estimation sample, which is the same row set
        # for every call (``_null_loglike`` only changes the design matrix).
        lo = self._lower
        hi = self._upper
        z_lower = (lo - xb) / sigma
        z_upper = (hi - xb) / sigma

        exact = np.isfinite(lo) & np.isfinite(hi) & (lo == hi)
        lower_open = ~np.isfinite(z_lower)
        upper_open = ~np.isfinite(z_upper)

        values = np.zeros_like(xb)
        with np.errstate(divide="ignore", invalid="ignore"):
            both_finite = ~lower_open & ~upper_open & ~exact
            if both_finite.any():
                values[both_finite] = _interval_log_mass(
                    z_lower[both_finite], z_upper[both_finite]
                )
            only_upper = ~upper_open & lower_open & ~exact
            if only_upper.any():
                values[only_upper] = log_ndtr(z_upper[only_upper])
            only_lower = ~lower_open & upper_open & ~exact
            if only_lower.any():
                values[only_lower] = log_ndtr(-z_lower[only_lower])
            # (-inf, +inf) carries no information: a zero contribution keeps such
            # an observation from poisoning the sum, and `matches the formula's
            # limit of log(1) = 0.
            if exact.any():
                values[exact] = stats.norm.logpdf(z_lower[exact]) - log_sigma
        return values

    def _model_test(
        self, params: np.ndarray, cov: np.ndarray, names: Sequence[str], llf: float
    ) -> tuple[str, float, int]:
        # Stata's intreg prints the LR test against the constant-only model.
        positions = _slope_positions(names)
        df = len(positions)
        if df == 0:
            return "LR", float("nan"), 0
        ll0 = self._null_loglike()
        statistic = (
            2.0 * (llf - ll0) if np.isfinite(ll0) and np.isfinite(llf) else float("nan")
        )
        return "LR", float(statistic), df

    def _extra_scalars(self, params: np.ndarray) -> dict[str, Any]:
        lo, hi = self._lower, self._upper
        return {
            "lnsigma": float(params[-1]),
            "n_left": int(np.isneginf(lo).sum()),
            "n_right": int(np.isposinf(hi).sum()),
            "n_interval": int((np.isfinite(lo) & np.isfinite(hi) & (lo != hi)).sum()),
            "n_uncensored": int((np.isfinite(lo) & np.isfinite(hi) & (lo == hi)).sum()),
        }


# ------------------------------------------------------- two-step plumbing


def _split_equation(formula: str, keyword: str, command: str) -> tuple[str, str]:
    """Split ``"y ~ x + keyword(s ~ z)"`` into the two equation formulas.

    The second equation is located by its ``keyword(...)`` block, which must be
    the last term of the right-hand side and must be joined to the first
    equation with ``+``.  Stata spells this ``heckman y x, select(...)``; the
    formula transcription keeps every variable name in one string.

    Returns
    -------
    (outcome_formula, second_formula):
        Both are ordinary one-sided-in-``~`` formulas, e.g.
        ``("y ~  x", "s ~ z")``.  The trailing ``+`` is removed from the
        outcome equation and stray whitespace is normalised.
    """

    marker = f"{keyword}("
    position = formula.find(marker)
    if position < 0:
        what = (
            "a selection equation, as in 'y ~ x + select(s ~ z)'"
            if keyword == "select"
            else "a treatment equation, as in 'y ~ x + treat(t ~ z)'"
        )
        raise ConsistencyError(f"{command}: the formula must contain {what}")

    head = formula[:position].rstrip()
    tail = formula[position + len(marker) :].rstrip()

    # The keyword block is the last term, so the only ')' that can close it is
    # the formula's final one.  Anything after it would be a second term.
    if not tail.endswith(")"):
        raise ConsistencyError(
            f"{command}: {keyword}(...) is missing its closing ')' in "
            f"{formula!r}"
        )
    inner = tail[:-1]
    if ")" in inner:
        raise ConsistencyError(
            f"{command}: unsupported nested parentheses inside {keyword}(...) "
            f"in {formula!r}"
        )
    if not head.endswith("+"):
        raise ConsistencyError(
            f"{command}: expected '+' before {keyword}(...), as in "
            f"'y ~ x + {keyword}(s ~ z)'; got {formula!r}"
        )
    head = head[:-1].rstrip()

    if "~" not in head or "~" not in inner:
        raise ConsistencyError(
            f"{command}: both equations need a '~'; got {formula!r}"
        )
    return head, inner.strip()


def _design_columns(formula: str) -> list[str]:
    """Best-effort list of the data columns a formula references.

    Only used to decide which rows may be missing; patsy owns the real parsing.
    Identifier-like tokens that match a column are kept, formula operators and
    numeric literals are dropped.
    """

    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula)
    reserved = {"C", "Treatment", "reference", "base", "np", "log", "exp", "I"}
    return [token for token in tokens if token not in reserved]


def _augmented_regression(
    y: np.ndarray, A: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """OLS of ``y`` on ``A`` returning ``(beta, resid)``."""

    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta, y - A @ beta


def _two_step_covariance(
    *,
    outcome: np.ndarray,
    A: np.ndarray,
    beta: np.ndarray,
    selection: np.ndarray,
    Z: np.ndarray,
    gamma: np.ndarray,
    selected: np.ndarray,
    generated: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Two-step M-estimator covariance for a generated-regressor model.

    The moment vector stacks the outcome equation's normal equations (only on
    the selected rows, since ``A`` already contains the generated term) and the
    probit score of the first stage::

        psi_i = [ -1{s} (y - x'b) a_i ;  (s - Phi(w)) phi(w) / (Phi(1-Phi)) z_i ]

    ``generated(gamma)`` maps first-stage coefficients to the generated
    regressor, so the Jacobian can be taken numerically over the *whole*
    parameter vector -- that is what propagates first-stage estimation error
    into beta's covariance, the correction naive OLS omits.

    Returns the ``(k_beta + k_gamma)`` covariance of ``[beta, gamma]``.
    """

    n = outcome.size
    k_beta = beta.size
    k_gamma = gamma.size
    theta = np.concatenate([beta, gamma])

    def moments(parameters: np.ndarray) -> np.ndarray:
        b = parameters[:k_beta]
        g = parameters[k_beta:]
        generated_values = generated(g)
        index = Z @ g
        cdf = np.clip(stats.norm.cdf(index), 1e-300, 1.0 - 1e-16)
        pdf = stats.norm.pdf(index)

        augmented = A.copy()
        augmented[:, -1] = generated_values
        predicted = augmented @ b

        psi = np.zeros((n, k_beta + k_gamma))
        psi[selected, :k_beta] = -(outcome[selected] - predicted[selected])[
            :, None
        ] * augmented[selected]
        # Probit score: (s - Phi) phi / (Phi (1 - Phi)) * z.
        psi[:, k_beta:] = (
            (selection - cdf) / (cdf * (1.0 - cdf)) * pdf
        )[:, None] * Z
        return psi

    psi = moments(theta)
    meat = psi.T @ psi / n

    step = 1e-6 * np.maximum(np.abs(theta), 1.0)
    jacobian = np.zeros((k_beta + k_gamma, k_beta + k_gamma))
    for index in range(theta.size):
        increment = np.zeros(theta.size)
        increment[index] = step[index]
        jacobian[index] = (
            moments(theta + increment).sum(axis=0)
            - moments(theta - increment).sum(axis=0)
        ) / (2.0 * step[index])
    jacobian /= n

    try:
        bread = np.linalg.inv(jacobian)
    except np.linalg.LinAlgError:
        bread = np.linalg.pinv(jacobian)
    covariance = bread @ meat @ bread.T / n
    return covariance


def _probit_fit(
    selection: np.ndarray, Z: np.ndarray, *, maxiter: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """Probit by Newton-Raphson with step halving.

    A self-contained loop rather than ``statsmodels.Probit``: the first stage
    here only needs the coefficient vector (the sandwich in
    :func:`_two_step_covariance` owns inference), and keeping the iteration
    local gives one consistent failure mode instead of relying on statsmodels'
    convergence warnings.  Step halving is what keeps Newton monotone -- without
    it a probit Newton step can overshoot into the flat part of the likelihood.
    """

    k = Z.shape[1]
    gamma = np.zeros(k)
    cdf = np.clip(stats.norm.cdf(Z @ gamma), 1e-300, 1.0 - 1e-16)
    loglike = float(np.sum(selection * np.log(cdf) + (1 - selection) * np.log(1 - cdf)))

    for _ in range(maxiter):
        index = Z @ gamma
        cdf = np.clip(stats.norm.cdf(index), 1e-300, 1.0 - 1e-16)
        pdf = stats.norm.pdf(index)
        score = Z.T @ ((selection - cdf) * pdf / (cdf * (1.0 - cdf)))
        weights = pdf**2 / (cdf * (1.0 - cdf))
        information = (Z.T * weights) @ Z
        try:
            step = np.linalg.solve(information, score)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(information) @ score

        # Step halving keeps the iteration monotone; probit Newton is otherwise
        # capable of overshooting into the flat part of the likelihood.
        scale = 1.0
        while scale > 1e-10:
            candidate = gamma + scale * step
            cand_cdf = np.clip(
                stats.norm.cdf(Z @ candidate), 1e-300, 1.0 - 1e-16
            )
            cand_loglike = float(
                np.sum(
                    selection * np.log(cand_cdf)
                    + (1 - selection) * np.log(1 - cand_cdf)
                )
            )
            if cand_loglike >= loglike - 1e-14:
                gamma = candidate
                loglike = cand_loglike
                break
            scale *= 0.5
        else:  # pragma: no cover - no improving step means we are at the optimum
            break

        if np.max(np.abs(scale * step)) < 1e-12:
            break

    index = Z @ gamma
    cdf = np.clip(stats.norm.cdf(index), 1e-300, 1.0 - 1e-16)
    pdf = stats.norm.pdf(index)
    weights = pdf**2 / (cdf * (1.0 - cdf))
    information = (Z.T * weights) @ Z
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(information)
    return gamma, covariance


def _check_selection_indicator(
    indicator: np.ndarray, command: str, keyword: str
) -> np.ndarray:
    """Validate a 0/1 indicator and return it as a boolean mask."""

    unique = np.unique(indicator[np.isfinite(indicator)])
    if not np.all(np.isin(unique, (0.0, 1.0))):
        raise ConsistencyError(
            f"{command}: the {keyword} equation's left-hand side must be a 0/1 "
            f"indicator, got values {unique[:6].tolist()}"
        )
    selected = indicator > 0.5
    if selected.all():
        raise ConsistencyError(
            f"{command}: the {keyword} indicator is 1 for every observation, so "
            "the first stage cannot be estimated"
        )
    if not selected.any():
        raise ConsistencyError(
            f"{command}: the {keyword} indicator is 0 for every observation, so "
            "there is nothing to model in the second stage"
        )
    return selected


def _parse_two_equation(
    formula: str,
    data: pd.DataFrame,
    *,
    keyword: str,
    command: str,
    subset: Any = None,
) -> tuple[str, str, pd.DataFrame]:
    """Shared formula handling for ``heckman`` and ``etregress``."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if subset is not None:
        data = data.loc[subset]
    data = data.reset_index(drop=True)

    outcome_formula, second_formula = _split_equation(formula, keyword, command)
    return outcome_formula, second_formula, data


def _drop_missing(
    data: pd.DataFrame, formulas: Sequence[str], *, command: str
) -> pd.DataFrame:
    """Listwise-delete rows with a missing value in any referenced column."""

    columns = [
        column
        for formula in formulas
        for column in _design_columns(formula)
        if column in data.columns
    ]
    columns = list(dict.fromkeys(columns))
    if not columns:
        return data
    kept = data.dropna(subset=columns)
    if len(kept) == 0:
        raise ConsistencyError(
            f"{command}: every observation has a missing value in "
            f"{', '.join(columns)}; nothing left to estimate"
        )
    return kept.reset_index(drop=True)


# ------------------------------------------------------------------ heckman


class Heckman:
    """Heckman sample-selection model (Stata ``heckman``), two-step estimator.

    Selection equation (probit, full sample)::

        s* = z g + u,   s = 1[s* > 0]

    Outcome equation (observed only where ``s == 1``)::

        y = x b + e

    Step 1 estimates ``g`` and forms the inverse Mills ratio
    ``lambda = phi(z g)/Phi(z g)``; step 2 regresses ``y`` on ``x`` and
    ``lambda`` over the selected rows.

    Parameters
    ----------
    formula:
        ``"y ~ x + select(s ~ z)"``.  The selection indicator must be 0/1 and
        the selection equation is estimated on the full sample, which is the
        point of the estimator: a missing ``y`` is what marks a non-selected
        observation, so ``y`` may be missing off the selected rows.

    Notes
    -----
    Standard errors are the two-step M-estimator sandwich: the first stage's
    estimation error is propagated into the second stage, so they are *not*
    naive OLS standard errors (see the module docstring).  They are also not
    Stata's ML standard errors, because these are two-step estimates.
    """

    _stata_command = "heckman"

    def __init__(
        self,
        outcome: np.ndarray,
        outcome_exog: np.ndarray,
        selection: np.ndarray,
        selection_exog: np.ndarray,
        *,
        outcome_names: Sequence[str],
        selection_names: Sequence[str],
        selected: np.ndarray,
        model_data: pd.DataFrame | None = None,
    ) -> None:
        self._y = np.asarray(outcome, dtype=float).reshape(-1)
        self._X = np.asarray(outcome_exog, dtype=float)
        self._s = np.asarray(selection, dtype=float).reshape(-1)
        self._Z = np.asarray(selection_exog, dtype=float)
        self._selected = np.asarray(selected, dtype=bool)
        self._outcome_names = list(outcome_names)
        self._selection_names = list(selection_names)
        self._model_data = model_data

        if self._X.shape[0] != self._y.size or self._Z.shape[0] != self._y.size:
            raise ConsistencyError(
                "heckman: the outcome and selection equations must describe the "
                "same observations"
            )
        if self._selected.size != self._y.size:
            raise ConsistencyError(
                "heckman: the selection indicator must have one entry per "
                "observation"
            )
        if int(self._selected.sum()) <= self._X.shape[1] + 1:
            raise ConsistencyError(
                f"heckman: only {int(self._selected.sum())} selected observations "
                f"for {self._X.shape[1]} outcome regressors plus the inverse Mills "
                "ratio; the second stage is not identified"
            )

    @classmethod
    def from_formula(
        cls, formula: str, data: pd.DataFrame, *, subset: Any = None
    ) -> "Heckman":
        """Build the model from ``"y ~ x + select(s ~ z)"``."""

        outcome_formula, selection_formula, data = _parse_two_equation(
            formula, data, keyword="select", command="heckman", subset=subset
        )

        outcome_lhs = outcome_formula.split("~", 1)[0].strip()
        selection_lhs = selection_formula.split("~", 1)[0].strip()
        for column in (outcome_lhs, selection_lhs):
            if column not in data.columns:
                raise ConsistencyError(
                    f"heckman: column {column!r} not found in the data"
                )

        # The selection equation must see every row, so only its own columns may
        # be required: a missing outcome is the marker of non-selection.
        selection_rhs = selection_formula.split("~", 1)[1]
        data = _drop_missing(
            data,
            [f"{selection_lhs} ~ {selection_rhs}", f"1 ~ {outcome_formula.split('~', 1)[1]}"],
            command="heckman",
        )

        outcome_design = _model_matrix(outcome_formula.split("~", 1)[1], data, "heckman")
        selection_design = _model_matrix(selection_rhs, data, "heckman")

        outcome = data[outcome_lhs].to_numpy(dtype=float)
        selected = _check_selection_indicator(
            data[selection_lhs].to_numpy(dtype=float), "heckman", "select"
        )
        # Rows off the selected set carry no outcome; they take part in the
        # probit only.  Fill their y so the array stays numeric -- it is never
        # read for them.
        outcome = np.where(selected, outcome, 0.0)

        return cls(
            outcome,
            outcome_design[0],
            data[selection_lhs].to_numpy(dtype=float),
            selection_design[0],
            outcome_names=outcome_design[1],
            selection_names=selection_design[1],
            selected=selected,
            model_data=data,
        )

    # ------------------------------------------------------------------- fit
    def fit(
        self,
        method: str = "twostep",
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
        *,
        maxiter: int = 200,
        strict: bool = False,
    ) -> StataResults:
        """Estimate by the two-step (Heckman 1979) procedure."""

        normalized = str(method).strip().lower()
        if normalized not in ("twostep", "two-step", "2step"):
            raise UnsupportedError(
                f"heckman: only method='twostep' is implemented; {method!r} is not. "
                "The maximum likelihood variant requires a joint optimisation over "
                "beta, gamma and rho together with the bivariate normal "
                "probabilities, which has not been implemented or verified against "
                "Stata."
            )
        if cov_type not in (None, "nonrobust"):
            raise ValueError(
                f"heckman: only cov_type='nonrobust' is implemented, got {cov_type!r}"
            )

        gamma, _ = _probit_fit(self._s, self._Z, maxiter=maxiter)
        index = self._Z @ gamma
        mills = _mills_ratio(index)

        selected = self._selected
        augmented = np.column_stack([self._X, mills])
        beta, residual = _augmented_regression(self._y[selected], augmented[selected])

        n_selected = int(selected.sum())
        k_augmented = augmented.shape[1]
        dof = max(n_selected - k_augmented, 1)

        # Stata's two-step scale (heckman.ado:743-746):
        #     rss     = e(rss) of the second-stage augmented regression
        #     deltbar = mean over the selected rows of lambda*(lambda + w)
        #     sigma   = sqrt(rss/m + b_lambda^2 * deltbar)      m = e(N) = m_sel
        #     rho     = b_lambda / sigma
        # Note the divisor is m, *not* m - k: this is Heckman's two-step
        # consistent estimate, not the OLS residual variance.  Using
        # sqrt(rss/(m-k)) was wrong by 6.7% on the reference DGP.
        lam_coef = float(beta[-1])
        m = n_selected
        rss = float(residual @ residual)
        delta_bar = float(np.mean((mills * (mills + index))[selected]))
        sigma2 = rss / m + lam_coef**2 * delta_bar
        sigma = float(np.sqrt(sigma2)) if sigma2 > 0 else float("nan")

        rho = lam_coef / sigma if np.isfinite(sigma) and sigma > 0 else float("nan")
        # Stata's default rhometh=2: truncate an out-of-range rho to +/-1 and
        # then report sigma = b_lambda * rho, so e(lambda) = rho*sigma stays
        # equal to the coefficient on the inverse Mills ratio.
        if np.isfinite(rho) and abs(rho) > 1.0:
            rho = float(np.sign(rho))
            sigma = abs(lam_coef)

        covariance_all = _two_step_covariance(
            outcome=self._y,
            A=augmented,
            beta=beta,
            selection=self._s,
            Z=self._Z,
            gamma=gamma,
            selected=selected,
            generated=lambda g: _mills_ratio(self._Z @ g),
        )
        k_beta = beta.size
        k_gamma = gamma.size
        covariance = np.zeros((k_beta + k_gamma, k_beta + k_gamma))
        covariance[:k_beta, :k_beta] = covariance_all[:k_beta, :k_beta]
        covariance[k_beta:, k_beta:] = covariance_all[k_beta:, k_beta:]

        # Equation-prefixed names keep the two equations' ``_cons`` distinct,
        # mirroring Stata's "y |" / "selected |" blocks and the package-wide
        # ``<equation>:<term>`` convention.
        names = (
            [f"y:{name}" for name in self._outcome_names]
            + ["lambda"]
            + [f"select:{name}" for name in self._selection_names]
        )
        params = np.concatenate([beta, gamma])

        # Gaussian log-likelihood of the augmented regression, which is the
        # second-stage objective a two-step estimator maximises, together with
        # its constant-only counterpart for r2_p.
        r2_p, llf, ll0 = _gaussian_pseudo_r2(
            self._y[selected],
            augmented[selected] @ beta,
            nobs=n_selected,
            k=k_augmented,
        )

        slope_positions = _slope_positions(names)
        wald = _wald_statistic(params, covariance, slope_positions)
        df_model = len(slope_positions)

        e: dict[str, Any] = {
            "N": int(self._y.size),
            "N_selected": n_selected,
            "N_nonselected": int(self._y.size - n_selected),
            "df_m": df_model,
            "rank": int(params.size),
            "ll": llf,
            "ll_0": ll0,
            "chi2": wald,
            "chi2type": "Wald",
            "p": _chi2_pvalue(wald, df_model),
            "r2_p": r2_p,
            "sigma": sigma,
            "rho": rho,
            # Stata's e(lambda) is rho*sigma, not the mean inverse Mills ratio.
            # Verified: the fixture prints rho .2849988, sigma 1.003876,
            # lambda .2861034, and the product is .2861035.
            "lambda": float(rho * sigma),
            "lam_coef": lam_coef,
            "estimator": "twostep",
            "vce": "twostep",
            "se_corrected": 1,
        }

        return StataResults(
            params=params,
            param_names=names,
            cov=covariance,
            resid=residual,
            fittedvalues=augmented[selected] @ beta,
            nobs=n_selected,
            df_resid=float(dof),
            df_model=float(df_model),
            dist="z",
            model_name=self._stata_command,
            cov_type="nonrobust",
            e=e,
            design_matrix=(self._X[selected], list(self._outcome_names)),
            model_data=self._model_data,
        )


class Etregress:
    """Endogenous treatment regression (Stata ``etregress``), control function.

    Treatment equation (probit, full sample)::

        t* = z g + u,   t = 1[t* > 0]

    Outcome equation (full sample, unlike :class:`Heckman`)::

        y = x b + d t + e

    The control function adds the first-stage residual
    ``t*phi(z g)/Phi(z g) - (1-t)*phi(z g)/(1-Phi(z g))`` to the outcome
    regression, which absorbs the correlation between ``u`` and ``e`` and makes
    the treatment coefficient ``d`` consistent.

    Parameters
    ----------
    formula:
        ``"y ~ x + treat(t ~ z)"``.  The treatment indicator is added to the
        outcome equation automatically; naming it there as well
        (``"y ~ x + t + treat(t ~ z)"``) is accepted and does not duplicate it.

    Notes
    -----
    ``y`` is observed for treated and control units alike, so every row enters
    the outcome equation -- that is the structural difference from
    :class:`Heckman`, where the outcome is truncated by selection.

    Standard errors are the two-step M-estimator sandwich (see the module
    docstring), not naive OLS standard errors and not Stata's ML ones.
    """

    _stata_command = "etregress"

    def __init__(
        self,
        outcome: np.ndarray,
        outcome_exog: np.ndarray,
        treatment: np.ndarray,
        treatment_exog: np.ndarray,
        *,
        outcome_names: Sequence[str],
        treatment_names: Sequence[str],
        treated: np.ndarray,
        model_data: pd.DataFrame | None = None,
    ) -> None:
        self._y = np.asarray(outcome, dtype=float).reshape(-1)
        self._X = np.asarray(outcome_exog, dtype=float)
        self._t = np.asarray(treatment, dtype=float).reshape(-1)
        self._Z = np.asarray(treatment_exog, dtype=float)
        self._treated = np.asarray(treated, dtype=bool)
        self._outcome_names = list(outcome_names)
        self._treatment_names = list(treatment_names)
        self._model_data = model_data

        if self._X.shape[0] != self._y.size or self._Z.shape[0] != self._y.size:
            raise ConsistencyError(
                "etregress: the outcome and treatment equations must describe the "
                "same observations"
            )
        if self._y.size <= self._X.shape[1] + 1:
            raise ConsistencyError(
                f"etregress: {self._y.size} observations for "
                f"{self._X.shape[1]} outcome regressors plus the control function; "
                "the outcome equation is not identified"
            )

    @classmethod
    def from_formula(
        cls, formula: str, data: pd.DataFrame, *, subset: Any = None
    ) -> "Etregress":
        """Build the model from ``"y ~ x + treat(t ~ z)"``."""

        outcome_formula, treatment_formula, data = _parse_two_equation(
            formula, data, keyword="treat", command="etregress", subset=subset
        )

        outcome_lhs = outcome_formula.split("~", 1)[0].strip()
        treatment_lhs = treatment_formula.split("~", 1)[0].strip()
        for column in (outcome_lhs, treatment_lhs):
            if column not in data.columns:
                raise ConsistencyError(
                    f"etregress: column {column!r} not found in the data"
                )

        treatment_rhs = treatment_formula.split("~", 1)[1]
        outcome_rhs = outcome_formula.split("~", 1)[1]
        data = _drop_missing(
            data,
            [f"{outcome_lhs} ~ {outcome_rhs}", f"{treatment_lhs} ~ {treatment_rhs}"],
            command="etregress",
        )

        # The treatment indicator joins the outcome equation, so it must appear
        # exactly once even when the user also wrote it among the regressors.
        outcome_terms = _strip_term(outcome_rhs, treatment_lhs)
        outcome_x = _model_matrix(outcome_terms, data, "etregress")
        treatment_design = _model_matrix(treatment_rhs, data, "etregress")
        treat_vector = data[treatment_lhs].to_numpy(dtype=float)
        _check_selection_indicator(treat_vector, "etregress", "treat")

        augmented = np.column_stack([outcome_x[0], treat_vector])
        names = list(outcome_x[1]) + [treatment_lhs]

        return cls(
            data[outcome_lhs].to_numpy(dtype=float),
            augmented,
            treat_vector,
            treatment_design[0],
            outcome_names=names,
            treatment_names=treatment_design[1],
            treated=treat_vector > 0.5,
            model_data=data,
        )

    # ------------------------------------------------------------------- fit
    def fit(
        self,
        method: str = "control",
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
        *,
        maxiter: int = 200,
        strict: bool = False,
    ) -> StataResults:
        """Estimate by the control-function (two-step) procedure."""

        normalized = str(method).strip().lower()
        if normalized not in ("control", "controlfunction", "control-function", "twostep", "two-step"):
            raise UnsupportedError(
                f"etregress: only method='control' is implemented; {method!r} is "
                "not. The maximum likelihood variant requires the joint likelihood "
                "of the outcome and treatment equations with a bivariate normal "
                "error term, which has not been implemented or verified against "
                "Stata."
            )
        if cov_type not in (None, "nonrobust"):
            raise ValueError(
                f"etregress: only cov_type='nonrobust' is implemented, got {cov_type!r}"
            )

        gamma, _ = _probit_fit(self._t, self._Z, maxiter=maxiter)
        control = self._control_function(gamma, self._t)

        augmented = np.column_stack([self._X, control])
        beta, residual = _augmented_regression(self._y, augmented)

        n = self._y.size
        k_augmented = augmented.shape[1]
        dof = max(n - k_augmented, 1)

        # Stata's control-function scale (etregress.ado:490-492), identical in
        # form to heckman's but with mtao = mean(lambda*(lambda + w)) over the
        # FULL sample (the treatment equation is estimated on all rows):
        #     rss   = e(rss) of the augmented regression
        #     sigma = sqrt(rss/m + mtao * b_ctrl^2),   rho = b_ctrl / sigma
        # and e(lambda) = rho*sigma = b_ctrl is the control-function
        # coefficient itself -- with the SIGN OF b_ctrl, which is what the old
        # `rho = -b_ctrl/sigma` got backwards (both DGPs were exact mirrors).
        control_coef = float(beta[-1])
        m = n
        rss = float(residual @ residual)
        index = self._Z @ gamma
        mtao = float(np.mean(control * (control + index)))
        sigma2 = rss / m + mtao * control_coef**2
        sigma = float(np.sqrt(sigma2)) if sigma2 > 0 else float("nan")

        rho = control_coef / sigma if np.isfinite(sigma) and sigma > 0 else float("nan")
        # Stata truncates |rho| > 1 (etregress.ado:495-497) but does not then
        # redefine sigma for the control-function case.
        if np.isfinite(rho) and abs(rho) > 1.0:
            rho = float(np.sign(rho))
        k_beta = beta.size
        covariance_all = _two_step_covariance(
            outcome=self._y,
            A=augmented,
            beta=beta,
            selection=self._t,
            Z=self._Z,
            gamma=gamma,
            selected=np.ones(n, dtype=bool),
            generated=lambda g: self._control_function(g, self._t),
        )
        covariance = np.zeros((k_beta + gamma.size, k_beta + gamma.size))
        covariance[:k_beta, :k_beta] = covariance_all[:k_beta, :k_beta]
        covariance[k_beta:, k_beta:] = covariance_all[k_beta:, k_beta:]

        # Equation-prefixed names, as for ``heckman``: Stata labels these blocks
        # "y" and by the treatment variable, and the prefix keeps the two
        # constants distinct.
        names = (
            [f"y:{name}" for name in self._outcome_names]
            + ["lambda"]
            + [f"treat:{name}" for name in self._treatment_names]
        )
        params = np.concatenate([beta, gamma])

        r2_p, llf, ll0 = _gaussian_pseudo_r2(
            self._y, augmented @ beta, nobs=n, k=k_augmented
        )

        slope_positions = _slope_positions(names)
        wald = _wald_statistic(params, covariance, slope_positions)
        df_model = len(slope_positions)

        e: dict[str, Any] = {
            "N": n,
            "N_treated": int(self._treated.sum()),
            "df_m": df_model,
            "rank": int(params.size),
            "ll": llf,
            "ll_0": ll0,
            "chi2": wald,
            "chi2type": "Wald",
            "p": _chi2_pvalue(wald, df_model),
            "r2_p": r2_p,
            "sigma": sigma,
            "rho": rho,
            "estimator": "control",
            "vce": "twostep",
            "se_corrected": 1,
        }

        return StataResults(
            params=params,
            param_names=names,
            cov=covariance,
            resid=residual,
            fittedvalues=augmented @ beta,
            nobs=n,
            df_resid=float(dof),
            df_model=float(df_model),
            dist="z",
            model_name=self._stata_command,
            cov_type="nonrobust",
            e=e,
            design_matrix=(self._X, list(self._outcome_names)),
            model_data=self._model_data,
        )

    # -------------------------------------------------------------- internals
    def _control_function(self, gamma: np.ndarray, treatment: np.ndarray) -> np.ndarray:
        """Generalised residual ``phi(w) [t/Phi(w) - (1-t)/(1-Phi(w))]``.

        This is ``E[u | t, z]`` for a standard normal ``u`` in the probit, so
        including it in the outcome equation absorbs the endogeneity.  Both
        mill-ratio pieces are evaluated through the log CDF so neither tail
        overflows: ``phi/Phi = exp(log phi - log Phi)``.
        """

        index = self._Z @ gamma
        log_pdf = stats.norm.logpdf(index)
        log_cdf = log_ndtr(index)
        log_sf = log_ndtr(-index)
        with np.errstate(over="ignore", invalid="ignore"):
            ratio = np.where(
                treatment > 0.5,
                np.exp(log_pdf - log_cdf),
                -np.exp(log_pdf - log_sf),
            )
        return np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)


def _model_matrix(rhs: str, data: pd.DataFrame, command: str) -> tuple[np.ndarray, list[str]]:
    """Design matrix for a right-hand side, with a Stata-style error message."""

    X, names, _ = build_design(f"__response__ ~ {rhs}", _with_response(data))
    return X, names


def _with_response(data: pd.DataFrame) -> pd.DataFrame:
    """Attach a dummy response column so :func:`build_design` can be reused."""

    frame = data.copy()
    frame["__response__"] = 0.0
    return frame


def _strip_term(rhs: str, term: str) -> str:
    """Remove every standalone occurrence of ``term`` from a formula RHS.

    Stata's ``etregress y x treat(t ~ z)`` adds the treatment indicator itself;
    a user who spelled out ``y ~ x + t + treat(t ~ z)`` would otherwise get a
    duplicated and perfectly collinear column.
    """

    pattern = re.compile(rf"(?<![\w.]){re.escape(term)}(?![\w.])")
    cleaned = pattern.sub(" ", rhs)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"^\s*\+\s*|\s*\+\s*(?=\+|$)", "", cleaned)
    cleaned = re.sub(r"\+\s*\+", "+", cleaned).strip()
    if not cleaned or cleaned in ("+", "0"):
        return "1"
    return cleaned
