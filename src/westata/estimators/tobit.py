"""Tobit censored regression, targeting Stata's ``tobit``.

Reference: ``[R] tobit`` (description of the model on p. 4, likelihood in
"Methods and formulas").

The latent model is ``y* = x b + e``, ``e ~ N(0, sigma^2)``, with the observed
outcome

    y_i = a            if y*_i <= a          (left-censored)
          y*_i         if a < y*_i < b       (uncensored)
          b            if y*_i >= b          (right-censored)

The log-likelihood Stata maximises is

    lnL = sum_{uncensored} [ -ln(sigma) - 0.5*ln(2*pi) - 0.5*((y-xb)/sigma)^2 ]
        + sum_{left}       ln( Phi((a - xb)/sigma) )
        + sum_{right}      ln( 1 - Phi((b - xb)/sigma) )

so ``sigma`` is estimated, not profiled away.  westata reports ``/sigma`` the
way Stata reports ``var(e.y)`` = ``sigma^2`` in the coefficient table; here the
``/sigma`` parameter appears as ``/sigma`` and ``e(sigma)`` holds its estimate.

Standard errors use the observed information (the Hessian of the log-
likelihood), which is Stata's default.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy import optimize, stats

from westata.errors import ConsistencyError, ConvergenceError
from westata.formula.factor import build_design
from westata.results.base import StataResults

__all__ = ["Tobit"]


class Tobit:
    """Tobit regression with lower and/or upper censoring (Stata ``tobit``).

    Parameters
    ----------
    endog:
        Observed outcome, length ``n``.
    exog:
        Design matrix ``n x k`` including a constant.
    lower:
        Lower censoring limit ``a``, or ``None`` for no left censoring.
    upper:
        Upper censoring limit ``b``, or ``None`` for no right censoring.
    """

    _stata_command = "tobit"

    def __init__(
        self,
        endog: Any,
        exog: Any,
        *,
        lower: float | None = 0.0,
        upper: float | None = None,
        param_names: list[str] | None = None,
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
        if lower is None and upper is None:
            raise ConsistencyError(
                "tobit: at least one censoring limit is required; with no "
                "censoring use OLS (Stata: regress)"
            )
        if lower is not None and upper is not None and not lower < upper:
            raise ConsistencyError(
                f"tobit: lower limit ({lower}) must be below upper limit ({upper})"
            )
        self._lower = lower
        self._upper = upper
        #: Estimation-sample covariates, retained for margins(over=...).
        self._model_data: pd.DataFrame | None = None

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        lower: float | None = 0.0,
        upper: float | None = None,
        subset: Any = None,
    ) -> "Tobit":
        """Build the model from a formula (Stata ``tobit y x, ll(#) ul(#)``)."""

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if subset is not None:
            data = data.loc[subset]
        data = data.reset_index(drop=True)

        X, names, kept_index = build_design(formula, data)
        lhs = formula.split("~", 1)[0].strip()
        if lhs not in data.columns:
            raise ConsistencyError(
                f"tobit: left-hand side must be a plain column name, got {lhs!r}"
            )
        y = data.loc[kept_index, lhs].to_numpy(dtype=float)
        model = cls(
            y, X, lower=lower, upper=upper, param_names=names
        )
        model._model_data = data.loc[kept_index].reset_index(drop=True)
        return model

    # ------------------------------------------------------------ likelihood
    def _censoring(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y = self._y
        left = np.zeros(y.size, dtype=bool)
        right = np.zeros(y.size, dtype=bool)
        if self._lower is not None:
            left = y <= self._lower
        if self._upper is not None:
            right = y >= self._upper
        uncensored = ~(left | right)
        return left, right, uncensored

    def _loglike(self, theta: np.ndarray) -> float:
        k = self._X.shape[1]
        beta = theta[:k]
        # Parameterise sigma on the log scale so it stays positive.
        sigma = float(np.exp(theta[k]))
        xb = self._X @ beta
        left, right, uncensored = self._censoring()

        log_likelihood = 0.0
        if uncensored.any():
            resid = (self._y[uncensored] - xb[uncensored]) / sigma
            log_likelihood -= float(
                np.sum(0.5 * resid**2 + np.log(sigma) + 0.5 * np.log(2 * np.pi))
            )
        if left.any():
            z = (self._lower - xb[left]) / sigma
            log_likelihood += float(np.sum(stats.norm.logcdf(z)))
        if right.any():
            z = (self._upper - xb[right]) / sigma
            log_likelihood += float(np.sum(stats.norm.logsf(z)))
        return log_likelihood

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
        *,
        maxiter: int = 1000,
        tol: float = 1e-10,
        strict: bool = False,
        start_params: np.ndarray | None = None,
    ) -> StataResults:
        """Maximise the censored likelihood and return a Stata-style result."""

        requested = "nonrobust" if cov_type is None else str(cov_type).strip()
        if requested.lower() != "nonrobust":
            raise ValueError(
                f"tobit: only cov_type='nonrobust' is implemented; got "
                f"{cov_type!r}"
            )

        n, k = self._X.shape
        if n <= k:
            raise ConsistencyError(
                f"tobit: {n} observations but {k} regressors; not identified"
            )

        if start_params is None:
            # Stata starts with an OLS fit on the uncensored/latent scale; for
            # the right-censored case it uses the observed values, which is
            # what ``lstsq`` on all observations approximates.
            beta0, *_ = np.linalg.lstsq(self._X, self._y, rcond=None)
            resid0 = self._y - self._X @ beta0
            sigma0 = max(float(np.std(resid0)), 1e-3)
            theta0 = np.concatenate([beta0, [np.log(sigma0)]])
        else:
            theta0 = np.asarray(start_params, dtype=float).reshape(-1)
            if theta0.size != k + 1:
                raise ValueError(
                    f"start_params must have {k + 1} entries (k slopes + "
                    f"log sigma), got {theta0.size}"
                )

        def objective(theta: np.ndarray) -> float:
            value = self._loglike(theta)
            return -value if np.isfinite(value) else 1e12

        result = optimize.minimize(
            objective,
            theta0,
            method="BFGS",
            options={"maxiter": maxiter, "gtol": tol},
        )
        # BFGS frequently stops on "precision loss" even at the optimum. Polish
        # with Nelder-Mead, which is derivative-free and does not need a
        # well-scaled gradient; Stata likewise refines its starting values.
        if not result.success:
            refined = optimize.minimize(
                objective,
                np.asarray(result.x, dtype=float),
                method="Nelder-Mead",
                options={
                    "maxiter": maxiter,
                    "xatol": 1e-10,
                    "fatol": 1e-10,
                },
            )
            if refined.fun <= result.fun:
                result = refined

        converged = bool(result.success)
        if not converged:
            message = (
                f"tobit: optimiser did not converge ({result.message}); "
                "estimates may be unreliable"
            )
            if strict:
                raise ConvergenceError(message)
            import warnings

            warnings.warn(message, RuntimeWarning, stacklevel=2)

        theta = np.asarray(result.x, dtype=float)
        beta = theta[:k]
        sigma = float(np.exp(theta[k]))

        cov = self._observed_information(theta, k)

        llf = float(self._loglike(theta))
        ll0 = self._null_loglike(k)
        chi2 = 2.0 * (llf - ll0) if np.isfinite(ll0) else float("nan")
        pseudo_r2 = 1.0 - llf / ll0 if np.isfinite(ll0) and ll0 != 0 else float("nan")

        left, right, uncensored = self._censoring()
        names = list(self._param_names) + ["/sigma"]
        params = np.concatenate([beta, [sigma]])
        if cov.shape != (k + 1, k + 1):
            cov = np.full((k + 1, k + 1), np.nan)

        e: dict[str, Any] = {
            "N": n,
            "n_uncensored": int(uncensored.sum()),
            "n_left": int(left.sum()),
            "n_right": int(right.sum()),
            "df_m": float(k - 1),
            "rank": k + 1,
            "ll": llf,
            "ll_0": ll0,
            "chi2": chi2,
            "chi2type": "LR",
            "p": float(stats.chi2.sf(chi2, k - 1)) if k > 1 else float("nan"),
            "r2_p": pseudo_r2,
            "sigma": sigma,
            "var_e": sigma**2,
            "converged": int(converged),
            "lower": self._lower,
            "upper": self._upper,
        }

        return StataResults(
            params=params,
            param_names=names,
            cov=cov,
            resid=self._y - self._X @ beta,
            fittedvalues=self._X @ beta,
            nobs=n,
            df_resid=float(n - k),
            df_model=float(k - 1),
            dist="t",
            model_name=self._stata_command,
            cov_type=requested,
            e=e,
            design_matrix=(self._X, list(self._param_names)),
            model_data=self._model_data,
        )

    # -------------------------------------------------------------- internals
    def _observed_information(self, theta: np.ndarray, k: int) -> np.ndarray:
        """Numerical Hessian of the *negative* log-likelihood, inverted."""

        def negative_loglike(params: np.ndarray) -> float:
            return -self._loglike(params)

        hessian = _numerical_hessian(negative_loglike, theta)
        try:
            cov = np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(hessian)
        if not np.all(np.isfinite(cov)):
            raise ConvergenceError(
                "tobit: the observed information matrix is not invertible; the "
                "model is likely not identified (check censoring limits and "
                "collinearity)"
            )
        return cov

    def _null_loglike(self, k: int) -> float:
        """Log-likelihood of the intercept-only model (for LR chi2 / pseudo R2)."""

        ones = np.ones((self._y.size, 1))
        original_X, original_names = self._X, self._param_names
        try:
            self._X = ones
            self._param_names = ["_cons"]
            beta0, *_ = np.linalg.lstsq(ones, self._y, rcond=None)
            sigma0 = max(float(np.std(self._y - ones @ beta0)), 1e-3)
            theta0 = np.array([float(beta0[0]), np.log(sigma0)])

            def objective(theta: np.ndarray) -> float:
                value = self._loglike(theta)
                return -value if np.isfinite(value) else 1e12

            result = optimize.minimize(
                objective, theta0, method="BFGS", options={"maxiter": 1000}
            )
            return float(self._loglike(np.asarray(result.x, dtype=float)))
        finally:
            self._X, self._param_names = original_X, original_names


def _numerical_hessian(func: Any, x: np.ndarray, step: float = 1e-5) -> np.ndarray:
    """Central-difference Hessian with a scaled step."""

    x = np.asarray(x, dtype=float)
    n = x.size
    hessian = np.zeros((n, n), dtype=float)
    h = step * np.maximum(np.abs(x), 1.0)

    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n)
            ej = np.zeros(n)
            ei[i] = h[i]
            ej[j] = h[j]
            if i == j:
                value = (
                    func(x + ei) - 2.0 * func(x) + func(x - ei)
                ) / (h[i] ** 2)
            else:
                value = (
                    func(x + ei + ej)
                    - func(x + ei - ej)
                    - func(x - ei + ej)
                    + func(x - ei - ej)
                ) / (4.0 * h[i] * h[j])
            hessian[i, j] = value
            hessian[j, i] = value
    return hessian
