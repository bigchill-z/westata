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
likelihood), which is Stata's default.  The Hessian is **analytic** (see
:meth:`Tobit._observed_information`): the cross-derivatives of the censored
terms mix ``beta`` and the scale, and a finite-difference Hessian at
``step=1e-5`` left the coefficient standard errors 8.9e-5 away from Stata's.
Because the optimiser works with ``ln sigma`` but the reported parameter is
``sigma``, the covariance needs the Jacobian ``d sigma / d ln sigma = sigma``
on the scale row and column; without it the reported ``se(/sigma)`` is
``se(ln sigma)``, which is ``1/sigma`` times too small (Stata's ``se(var(e.y))``
mapped to the sigma scale is the check).

``e(df_m)``/``e(df_r)`` follow ``tobit.ado`` (StataNow 19.5)::

    df_m = e(rank) - 1 - hascons
    df_r = e(N) - e(df_m)

so the estimated scale consumes no degree of freedom and, with a constant,
neither does the intercept.  ``e(df_r)`` is therefore ``N - (slopes)``, not
``N - (slopes + intercept)``; the ratio is Stata's ``t`` denominator.

Accuracy of the reported standard errors
----------------------------------------
Against StataNow 19.5, on three censoring variants (``ll``, ``ul``, ``ll+ul``):
parameters and standard errors agree to ~3e-8, ``se(/sigma)`` to ~5e-8, and
``predict, stdp`` to ~3e-8.  **That residual is Stata's stopping tolerance, not
ours**, and it will not go away::

    |score(theta_hat_stata)|_inf  ~ 1e-6 .. 1.5e-2   (analytic score, evaluated
                                                      at Stata's OWN estimates)
    |score(theta_hat_ours)|_inf   ~ 1e-13

One Newton step from Stata's estimates lands back on ours to 1e-9, and the two
log-likelihoods agree to 1e-13.  Stata's ``technique(nr)`` reaches that same
point regardless of ``tolerance(1e-12)`` or ``tolerance(1e-14)``, while
``technique(dfp)``/``(bhhh)`` move it by 5e-5 / 7e-5 -- so it is a fixed point of
its own iteration, just not the exact maximiser.

**Do not "fix" this by loosening or tightening anything.**  The scoring rule
above is the whole diagnosis: if a future change makes the gap smaller than
~1e-8 it has started chasing Stata's iteration noise; if it grows past 1e-6,
the analytic Hessian or the score has been broken (both are checked against
central differences in ``tests/test_tobit.py``).
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

    def _score(self, theta: np.ndarray) -> np.ndarray:
        """Analytic gradient of the log-likelihood at ``theta = [beta, ln sigma]``.

        Stata maximises this likelihood with analytic derivatives; BFGS on a
        numerical gradient stops a few ulps short of the exact maximiser, which
        is enough to leave ``stdp`` 4e-8 away from Stata's.  Supplying the score
        and the analytic Hessian lets the final step be a Newton step that lands
        on the same point the closed-form likelihood defines.
        """

        k = self._X.shape[1]
        beta = np.asarray(theta[:k], dtype=float)
        sigma = float(np.exp(theta[k]))
        xb = self._X @ beta
        left, right, uncensored = self._censoring()

        score = np.zeros(k + 1, dtype=float)

        if uncensored.any():
            r = (self._y[uncensored] - xb[uncensored]) / sigma
            Xu = self._X[uncensored]
            score[:k] += Xu.T @ r / sigma
            score[k] += float(r @ r) - float(uncensored.sum())

        for mask, sign, limit in (
            (left, 1.0, self._lower),
            (right, -1.0, self._upper),
        ):
            if limit is None or not mask.any():
                continue
            z = sign * (float(limit) - xb[mask]) / sigma
            lam = np.exp(stats.norm.logpdf(z) - stats.norm.logcdf(z))
            Xc = self._X[mask]
            # d ln Phi(z)/d b = -sign lam x / sigma ; d/d ln sigma = -lam z
            score[:k] += -(sign / sigma) * (Xc.T @ lam)
            score[k] += float(np.sum(-lam * z))
        return score

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

        def gradient(theta: np.ndarray) -> np.ndarray:
            return -self._score(theta)

        def hessian(theta: np.ndarray) -> np.ndarray:
            return -self._information(theta, k)

        result = optimize.minimize(
            objective,
            theta0,
            jac=gradient,
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

        # Newton polish on the analytic score/Hessian.  Stata estimates by
        # Newton-Raphson with analytic derivatives; a quasi-Newton stop leaves
        # the parameters a few 1e-11 away, which is visible in ``stdp``.
        polished = self._newton_polish(
            np.asarray(result.x, dtype=float), objective, k, maxiter=maxiter
        )
        if polished is not None and polished[1] >= -float(result.fun):
            theta_hat, value = polished
            result = optimize.OptimizeResult(
                x=theta_hat,
                fun=-value,
                success=True,
                message="converged (analytic Newton polish)",
                nit=0,
            )

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

        # ``_observed_information`` returns the covariance of ``[beta, ln sigma]``
        # because that is the parameterisation the likelihood is written in.
        cov = self._observed_information(theta, k)
        # Reported scale is ``sigma``; ``d sigma / d ln sigma = sigma``, so the
        # last row and column carry that Jacobian.  Without it the printed
        # ``se(/sigma)`` is the standard error of ``ln sigma`` -- smaller by
        # exactly ``1/sigma`` and enough to inflate the reported z.
        cov = cov.copy()
        cov[k, :] *= sigma
        cov[:, k] *= sigma

        llf = float(self._loglike(theta))
        ll0 = self._null_loglike(k)
        chi2 = 2.0 * (llf - ll0) if np.isfinite(ll0) else float("nan")
        pseudo_r2 = 1.0 - llf / ll0 if np.isfinite(ll0) and ll0 != 0 else float("nan")

        left, right, uncensored = self._censoring()
        names = list(self._param_names) + ["/sigma"]
        params = np.concatenate([beta, [sigma]])
        if cov.shape != (k + 1, k + 1):
            cov = np.full((k + 1, k + 1), np.nan)

        # Stata (tobit.ado, StataNow 19.5):
        #     df_m = e(rank) - 1 - hascons        rank = k + 1
        #     df_r = e(N) - e(df_m)
        # so the estimated scale consumes no degree of freedom and, when there is
        # a constant, neither does the intercept.  Verified on five datasets
        # spanning N/k = 74/3, 200/2, 350/4, 500/6, 120/3 and on `noconstant`.
        hascons = 1 if any(
            name in ("Intercept", "_cons", "const") for name in self._param_names
        ) else 0
        df_m = float(k - hascons)
        df_r = float(n - df_m)

        e: dict[str, Any] = {
            "N": n,
            "n_uncensored": int(uncensored.sum()),
            "n_left": int(left.sum()),
            "n_right": int(right.sum()),
            "df_m": df_m,
            "df_r": df_r,
            "rank": k + 1,
            "ll": llf,
            "ll_0": ll0,
            "chi2": chi2,
            "chi2type": "LR",
            "p": float(stats.chi2.sf(chi2, df_m)) if df_m > 0 else float("nan"),
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
            df_resid=df_r,
            df_model=df_m,
            dist="t",
            model_name=self._stata_command,
            cov_type=requested,
            e=e,
            design_matrix=(self._X, list(self._param_names)),
            model_data=self._model_data,
        )

    # -------------------------------------------------------------- internals
    def _observed_information(self, theta: np.ndarray, k: int) -> np.ndarray:
        """Covariance of ``[beta, ln sigma]``: inverse of ``-H`` at ``theta``."""

        hessian = self._information(theta, k)
        hessian = (hessian + hessian.T) / 2.0
        try:
            cov = np.linalg.inv(-hessian)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(-hessian)
        if not np.all(np.isfinite(cov)):
            raise ConvergenceError(
                "tobit: the observed information matrix is not invertible; the "
                "model is likely not identified (check censoring limits and "
                "collinearity)"
            )
        return cov

    def _information(self, theta: np.ndarray, k: int) -> np.ndarray:
        """Analytic Hessian of the log-likelihood at ``[beta, ln sigma]``.

        The likelihood mixes three regimes (uncensored density, left- and
        right-censored normal probabilities), and their cross-derivatives with
        respect to ``beta`` and the scale do not vanish.  A central-difference
        Hessian at the old ``step=1e-5`` lost enough digits on the mixed block
        to leave every coefficient standard error 8.9e-5 away from Stata; the
        analytic form below is exact up to the floating-point sum.

        Write ``ell = ln sigma``, ``z_c`` for the censored index of an
        observation and ``lam = phi(z)/Phi(z)``.  With ``s = +1`` for a left
        censoring limit and ``s = -1`` for a right one, so that the contribution
        is always ``ln Phi(z_c)`` with ``z_c = s (limit - x'b)/sigma``::

            d ln Phi(z_c)/d b   = -s lam x / sigma
            d2/db db'           = -lam (z_c + lam) x x' / sigma^2
            d/d ell             = -lam z_c
            d2/d ell2           = -lam (z_c + lam) z_c^2 + lam z_c
            d2/d b d ell        =  s lam x (1 - z_c (z_c + lam)) / sigma

        and, for an uncensored observation with ``r = (y - x'b)/sigma``::

            d2/db db'  = -x x'/sigma^2,   d2/db d ell = -2 r x/sigma,
            d2/d ell2  = -2 r^2
        """

        beta = np.asarray(theta[:k], dtype=float)
        sigma = float(np.exp(theta[k]))
        xb = self._X @ beta
        left, right, uncensored = self._censoring()

        hessian = np.zeros((k + 1, k + 1), dtype=float)

        # -- uncensored observations: Gaussian density ----------------------
        if uncensored.any():
            r = (self._y[uncensored] - xb[uncensored]) / sigma
            Xu = self._X[uncensored]
            hessian[:k, :k] -= (Xu.T @ Xu) / sigma**2
            hessian[:k, k] -= 2.0 * (Xu.T @ r) / sigma
            hessian[k, :k] = hessian[:k, k]
            hessian[k, k] -= 2.0 * float(r @ r)

        # -- censored observations: ln Phi(+-z) -----------------------------
        for mask, sign, limit in (
            (left, 1.0, self._lower),
            (right, -1.0, self._upper),
        ):
            if limit is None or not mask.any():
                continue
            z = sign * (float(limit) - xb[mask]) / sigma
            lam = np.exp(stats.norm.logpdf(z) - stats.norm.logcdf(z))
            Xc = self._X[mask]
            # -lam (z + lam) multiplies x x' / sigma^2; the sign is folded in.
            weight = -lam * (z + lam)
            hessian[:k, :k] += (Xc.T * weight) @ Xc / sigma**2
            # d2 / d b d ell
            cross = sign * lam * (1.0 - z * (z + lam)) / sigma
            hessian[:k, k] += Xc.T @ cross
            hessian[k, :k] = hessian[:k, k]
            hessian[k, k] += float(
                np.sum(-lam * (z + lam) * z**2 + lam * z)
            )

        return hessian

    def _newton_polish(
        self,
        theta: np.ndarray,
        objective: Any,
        k: int,
        *,
        maxiter: int = 100,
        max_steps: int = 50,
    ) -> tuple[np.ndarray, float] | None:
        """Newton-Raphson on the analytic score, returning ``(theta, loglike)``.

        Returns ``None`` when the iteration cannot be taken (singular
        information, a non-finite value, or no improvement after backtracking);
        the caller then keeps the optimiser's answer, so this can only improve
        the estimate, never break a case that used to fit.
        """

        x = np.asarray(theta, dtype=float).copy()
        value = -objective(x)
        if not np.isfinite(value):
            return None
        for _ in range(min(max_steps, maxiter)):
            score = self._score(x)
            if not np.all(np.isfinite(score)):
                return None
            if float(np.max(np.abs(score))) <= 1e-12:
                break
            hessian = self._information(x, k)
            try:
                # Newton step for a maximum: x+ = x - H^-1 s (H of the
                # log-likelihood is negative definite at the optimum).
                step = -np.linalg.solve(hessian, score)
            except np.linalg.LinAlgError:
                return None
            if not np.all(np.isfinite(step)):
                return None
            # Backtracking: a Newton step on a non-concave region can overshoot.
            scale = 1.0
            improved = False
            for _ in range(40):
                candidate = x + scale * step
                value_candidate = -objective(candidate)
                if np.isfinite(value_candidate) and value_candidate >= value:
                    x, value = candidate, value_candidate
                    improved = True
                    break
                scale *= 0.5
            if not improved:
                break
            if scale * float(np.max(np.abs(step))) <= 1e-14 * max(
                1.0, float(np.max(np.abs(x)))
            ):
                break
        return x, value

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
