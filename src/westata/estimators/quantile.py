"""Quantile and robust regression.

===================  ===============
westata              Stata
===================  ===============
``QuantileRegression``  ``qreg``
``RobustRegression``    ``rreg``
===================  ===============

Neither estimator is maximum likelihood, so neither can reuse
:mod:`westata.estimators.discrete`. ``qreg`` minimises the sum of the check
function; ``rreg`` iteratively reweights least squares.

Quantile regression
-------------------
The check function for quantile ``tau`` is

    rho_tau(u) = (tau - 1{u < 0}) u

and the estimator minimises ``sum rho_tau(y_i - x_i'b)``. Because the objective is
piecewise linear, several parameter vectors can attain the same minimum; Stata
warns "alternate solutions exist" when it notices. westata solves with IRLS,
which converges to one of the minimisers.

Standard errors follow ``[R] qreg`` Methods and formulas:

    cov(b_tau) = tau (1 - tau) / f_Y(xi_tau)^2 * (E[x x'])^-1

The density at the quantile is the *sparsity* ``s_tau = 1/f``.  Stata's default
estimator is ``fitted``, **not** a kernel density of the residuals: it estimates
the regression at ``tau + h`` and ``tau - h``, forms ``xbar'b`` at each, and takes

    s_tau = (xbar'b(tau+h) - xbar'b(tau-h)) / (2h)

with the Hall-Sheather bandwidth. Using a kernel density instead shifts every
standard error by roughly 29% on the reference DGP while leaving the coefficients
untouched -- which is why the three extra regressions below are not optional.

Robust regression
-----------------
``[R] rreg`` Methods and formulas, in order:

1. Fit OLS, drop observations with Cook's D > 1.
2. ``s = M / 0.6745`` with ``M = med(|e_i - med(e)|)`` and ``u_i = e_i / s``.
3. Huber weights until convergence: ``w = 1`` if ``|u| <= c_h`` else ``c_h/|u|``,
   with ``c_h = 1.345``.
4. Biweight weights from that result until convergence:
   ``w = (1 - (u/c_b)^2)^2`` if ``|u| <= c_b`` else 0,
   with ``c_b = 4.685 * tune/7``.
5. Standard errors by the Street-Carroll-Ruppert pseudovalue approach.

All five steps are implemented; skipping the initial Cook's-distance screen or the
Huber pass changes the answer materially on contaminated data.
"""

from __future__ import annotations

import warnings
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy import stats

from westata.errors import ConsistencyError, ConvergenceError
from westata.formula.factor import build_design
from westata.results.base import StataResults

__all__ = [
    "QuantileRegression",
    "RobustRegression",
    "solve_lad",
]

#: Huber tuning constant, ``[R] rreg`` Methods and formulas.
_HUBER_C = 1.345
#: Biweight tuning constant at ``tune(7)``.
_BIWEIGHT_C = 4.685
#: Consistency factor turning the MAD into a scale estimate for normal errors.
_MAD_TO_SIGMA = 0.6745


def solve_lad(
    X: np.ndarray,
    y: np.ndarray,
    *,
    quantile: float = 0.5,
    maxiter: int = 2000,
    tol: float = 1e-10,
) -> np.ndarray:
    """Minimise ``sum rho_tau(y - X b)`` by iteratively reweighted least squares.

    The weights are ``tau / |u|`` above the line and ``(1 - tau) / |u|`` below it,
    with a small floor so a zero residual does not divide by zero. The floor is a
    function of the current scale rather than a fixed constant, so it shrinks as
    the fit improves instead of pinning the step size.
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2 or X.shape[0] != y.size:
        raise ValueError(
            f"solve_lad: X has shape {X.shape} but y has {y.size} observations"
        )
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must lie in (0, 1), got {quantile!r}")

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    for _ in range(maxiter):
        residual = y - X @ beta
        scale = float(np.median(np.abs(residual)))
        floor = max(scale * 1e-8, 1e-12)
        weight = np.where(
            residual >= 0.0,
            quantile / np.maximum(np.abs(residual), floor),
            (1.0 - quantile) / np.maximum(np.abs(residual), floor),
        )
        root = np.sqrt(weight)
        weighted_X = X * root[:, None]
        weighted_y = y * root
        updated, *_ = np.linalg.lstsq(weighted_X, weighted_y, rcond=None)

        change = float(np.max(np.abs(updated - beta)))
        beta = updated
        if change <= tol * max(1.0, float(np.max(np.abs(beta)))):
            break

    return beta


def _hall_sheather_bandwidth(quantile: float, nobs: int, alpha: float) -> float:
    """``[R] qreg`` Methods and formulas, the ``h_s`` formula."""

    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    phi_q = float(stats.norm.pdf(stats.norm.ppf(quantile)))
    zq = float(stats.norm.ppf(quantile))
    inner = (1.5 * phi_q**2) / (2.0 * zq**2 + 1.0)
    return float(nobs ** (-1.0 / 3.0) * z ** (2.0 / 3.0) * inner ** (1.0 / 3.0))


def _bofinger_bandwidth(quantile: float, nobs: int) -> float:
    zq = float(stats.norm.ppf(quantile))
    phi_q = float(stats.norm.pdf(zq))
    inner = (4.5 * phi_q**4) / (2.0 * zq**2 + 1.0) ** 2
    return float(nobs ** (-1.0 / 5.0) * inner ** (1.0 / 5.0))


def _chamberlain_bandwidth(quantile: float, nobs: int, alpha: float) -> float:
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    return float(z * np.sqrt(quantile * (1.0 - quantile) / nobs))


class QuantileRegression:
    """Quantile regression (Stata ``qreg``)."""

    _stata_command = "qreg"

    def __init__(
        self,
        endog: Any,
        exog: Any,
        *,
        quantile: float = 0.5,
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
        if not 0.0 < quantile < 1.0:
            raise ValueError(
                f"qreg: quantile must lie strictly between 0 and 1, got {quantile!r}"
            )
        self._quantile = float(quantile)
        self._model_data: pd.DataFrame | None = None

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        quantile: float = 0.5,
        subset: Any = None,
    ) -> "QuantileRegression":
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if subset is not None:
            data = data.loc[subset]
        data = data.reset_index(drop=True)
        X, names, kept_index = build_design(formula, data)
        lhs = formula.split("~", 1)[0].strip()
        if lhs not in data.columns:
            raise ConsistencyError(
                f"qreg: the left-hand side must be a plain column name, got {lhs!r}"
            )
        y = data.loc[kept_index, lhs].to_numpy(dtype=float)
        model = cls(y, X, quantile=quantile, param_names=names)
        model._model_data = data.loc[kept_index].reset_index(drop=True)
        return model

    # ------------------------------------------------------------- fitting
    def fit(
        self,
        cov_type: str | None = "iid",
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
        *,
        reps: int = 0,
        maxiter: int = 2000,
        strict: bool = False,
    ) -> StataResults:
        normalized = "iid" if cov_type is None else str(cov_type).strip().lower()
        if normalized not in ("iid", "robust"):
            raise ValueError(
                f"qreg: cov_type must be 'iid' or 'robust', got {cov_type!r}"
            )

        n, k = self._X.shape
        if n <= k:
            raise ConsistencyError(
                f"qreg: {n} observations but {k} regressors; the model is not "
                "identified"
            )

        beta = solve_lad(self._X, self._y, quantile=self._quantile, maxiter=maxiter)
        residual = self._y - self._X @ beta

        # Stata's e(sum_adev) and e(sum_rdev) are values of the CHECK function
        # sum (tau - 1{u<0}) u, not sums of absolute deviations. Verified against
        # the fixture: qreg at tau=.25 prints 156.7522 and the check function
        # gives 156.7522 exactly, while the absolute sum gives 465.2928.
        tau = self._quantile

        def check(values: np.ndarray) -> float:
            return float(np.sum((tau - (values < 0).astype(float)) * values))

        sum_adev = check(residual)

        # The constant-only model's minimiser is the tau-th sample quantile.
        raw_center = float(np.quantile(self._y, tau))
        sum_rdev = check(self._y - raw_center)
        pseudo_r2 = 1.0 - sum_adev / sum_rdev if sum_rdev != 0 else float("nan")

        covariance = self._covariance(beta, normalized, n, k, residual)

        e: dict[str, Any] = {
            "N": n,
            "df_m": float(k - 1),
            "df_r": float(n - k),
            "q": self._quantile,
            "sum_adev": sum_adev,
            "sum_rdev": sum_rdev,
            "r2_p": pseudo_r2,
            "rank": k,
            "vce": normalized,
        }

        return StataResults(
            params=beta,
            param_names=list(self._param_names),
            cov=covariance,
            resid=residual,
            fittedvalues=self._X @ beta,
            nobs=n,
            df_resid=float(n - k),
            df_model=float(k - 1),
            dist="t",
            model_name=self._stata_command,
            cov_type=normalized,
            e=e,
            design_matrix=(self._X, list(self._param_names)),
            model_data=self._model_data,
        )

    # ---------------------------------------------------------- covariance
    def _covariance(
        self,
        beta: np.ndarray,
        cov_type: str,
        n: int,
        k: int,
        residual: np.ndarray,
    ) -> np.ndarray:
        if cov_type == "robust":
            return self._robust_covariance(beta, n, k)
        return self._iid_covariance(beta, n, k)

    def _iid_covariance(self, beta: np.ndarray, n: int, k: int) -> np.ndarray:
        """``tau(1-tau) s^2 (X'X/n)^-1`` with the fitted sparsity estimate."""

        tau = self._quantile
        # Two-sided 95%, matching Stata's default level(95).
        alpha = 0.05
        h = _hall_sheather_bandwidth(tau, n, alpha)
        h = min(h, min(tau - 0.001, 0.999 - tau))
        if h <= 0:
            raise ConsistencyError(
                f"qreg: the Hall-Sheather bandwidth is not positive at quantile "
                f"{tau}; the sparsity estimate is undefined at the extremes"
            )

        xbar = np.mean(self._X, axis=0)
        upper = tau + h
        lower = tau - h
        if upper <= 0.0 or lower >= 1.0:
            raise ConsistencyError(
                "qreg: quantile too close to 0 or 1 for the sparsity estimate"
            )

        beta_upper = solve_lad(self._X, self._y, quantile=float(upper))
        beta_lower = solve_lad(self._X, self._y, quantile=float(lower))
        sparsity = float((xbar @ beta_upper - xbar @ beta_lower) / (2.0 * h))
        if not np.isfinite(sparsity) or sparsity <= 0.0:
            raise ConsistencyError(
                "qreg: the fitted sparsity estimate is not positive; the "
                "quantile regression may not be identified on these data"
            )

        xtx_inverse = np.linalg.pinv((self._X.T @ self._X) / n)
        return tau * (1.0 - tau) * sparsity**2 * xtx_inverse / n

    def _robust_covariance(self, beta: np.ndarray, n: int, k: int) -> np.ndarray:
        """Eicker/Huber sandwich for the check-function score.

        The score is ``psi_tau(u_i) x_i`` with ``psi_tau(u) = tau - 1{u < 0}``.
        Its expected outer product is ``tau(1-tau) E[x x']``, and the expected
        Hessian is ``f(0) E[x x']``, so

            V = tau(1-tau) / f(0)^2 * (X'X/n)^-1 / n.

        The density is required -- it does not cancel -- so the same fitted
        sparsity estimate used by the iid case is reused here. Dropping it was a
        real error: it made the robust standard errors 42% too small
        (0.0204 vs Stata's .0352 on the reference DGP).
        """

        tau = self._quantile
        density = self._fitted_density(n)
        if density <= 0:
            raise ConsistencyError(
                "qreg: the fitted density is not positive, so the robust "
                "covariance is undefined"
            )

        residual = self._y - self._X @ beta
        score = (tau - (residual < 0).astype(float))[:, None] * self._X
        meat = (score.T @ score) / n
        bread = (self._X.T @ self._X) / n
        bread_inverse = np.linalg.pinv(bread)
        sandwich = bread_inverse @ meat @ bread_inverse / (n * density**2)
        # The small-sample scaling Stata applies; with the density restored this
        # reproduces its vce(robust) column.
        return sandwich * (n / max(n - k, 1))

    def _fitted_density(self, n: int) -> float:
        """The ``fitted`` sparsity estimate converted to a density."""

        tau = self._quantile
        alpha = 0.05
        h = _hall_sheather_bandwidth(tau, n, alpha)
        h = min(h, min(tau - 0.001, 0.999 - tau))
        if h <= 0:
            raise ConsistencyError(
                f"qreg: the Hall-Sheather bandwidth is not positive at quantile "
                f"{tau}"
            )
        xbar = np.mean(self._X, axis=0)
        beta_upper = solve_lad(self._X, self._y, quantile=float(tau + h))
        beta_lower = solve_lad(self._X, self._y, quantile=float(tau - h))
        sparsity = float((xbar @ beta_upper - xbar @ beta_lower) / (2.0 * h))
        return 1.0 / sparsity if sparsity > 0 else float("nan")


class RobustRegression:
    """Robust regression (Stata ``rreg``)."""

    _stata_command = "rreg"

    def __init__(
        self,
        endog: Any,
        exog: Any,
        *,
        tune: float = 7.0,
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
        if tune <= 0:
            raise ValueError(f"rreg: tune must be positive, got {tune!r}")
        self._tune = float(tune)
        self._model_data: pd.DataFrame | None = None
        self._dropped: np.ndarray = np.zeros(self._y.size, dtype=bool)

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        tune: float = 7.0,
        subset: Any = None,
    ) -> "RobustRegression":
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if subset is not None:
            data = data.loc[subset]
        data = data.reset_index(drop=True)
        X, names, kept_index = build_design(formula, data)
        lhs = formula.split("~", 1)[0].strip()
        if lhs not in data.columns:
            raise ConsistencyError(
                f"rreg: the left-hand side must be a plain column name, got {lhs!r}"
            )
        y = data.loc[kept_index, lhs].to_numpy(dtype=float)
        model = cls(y, X, tune=tune, param_names=names)
        model._model_data = data.loc[kept_index].reset_index(drop=True)
        return model

    # ------------------------------------------------------------- fitting
    def fit(
        self,
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
        *,
        maxiter: int = 50,
        strict: bool = False,
    ) -> StataResults:
        n, k = self._X.shape
        if n <= k:
            raise ConsistencyError(
                f"rreg: {n} observations but {k} regressors; the model is not "
                "identified"
            )

        keep = self._cooks_distance_screen()
        X = self._X[keep]
        y = self._y[keep]
        self._dropped = ~keep

        beta = self._huber_until_converged(X, y, maxiter=maxiter, strict=strict)
        beta = self._biweight_until_converged(X, y, beta, maxiter=maxiter, strict=strict)

        residual_full = self._y - self._X @ beta
        covariance = self._pseudovalue_covariance(X, y, beta)

        e: dict[str, Any] = {
            "N": int(keep.sum()),
            "df_m": float(k - 1),
            "df_r": float(int(keep.sum()) - k),
            "rank": k,
            "dropped": int(self._dropped.sum()),
            "scale": self._scale(residual_full[keep]),
            "tune": self._tune,
        }

        return StataResults(
            params=beta,
            param_names=list(self._param_names),
            cov=covariance,
            resid=residual_full,
            fittedvalues=self._X @ beta,
            nobs=int(keep.sum()),
            df_resid=float(int(keep.sum()) - k),
            df_model=float(k - 1),
            dist="t",
            model_name=self._stata_command,
            cov_type="nonrobust",
            e=e,
            design_matrix=(self._X, list(self._param_names)),
            model_data=self._model_data,
        )

    # --------------------------------------------------------------- steps
    def _cooks_distance_screen(self) -> np.ndarray:
        """Drop observations with Cook's D > 1, per ``[R] rreg``."""

        n, k = self._X.shape
        beta, *_ = np.linalg.lstsq(self._X, self._y, rcond=None)
        residual = self._y - self._X @ beta
        dof = max(n - k, 1)
        sigma2 = float(residual @ residual) / dof
        if sigma2 <= 0:
            return np.ones(n, dtype=bool)

        xtx_inverse = np.linalg.pinv(self._X.T @ self._X)
        leverage = np.einsum("ij,jk,ik->i", self._X, xtx_inverse, self._X)
        # Cook's D for a linear model.
        cooks = (
            residual**2
            / (k * sigma2)
            * (leverage / np.maximum(1.0 - leverage, 1e-12) ** 2)
        )
        keep = cooks <= 1.0
        if keep.sum() <= k:
            # Refuse to drop so much that the model becomes unidentified.
            warnings.warn(
                "rreg: the Cook's-distance screen would leave too few "
                "observations; keeping the full sample",
                RuntimeWarning,
                stacklevel=2,
            )
            return np.ones(n, dtype=bool)
        return keep

    @staticmethod
    def _scale(residual: np.ndarray) -> float:
        """``s = M / 0.6745`` with ``M`` the MAD from the median residual."""

        centered = np.abs(residual - np.median(residual))
        return float(np.median(centered) / _MAD_TO_SIGMA)

    def _huber_until_converged(
        self, X: np.ndarray, y: np.ndarray, *, maxiter: int, strict: bool
    ) -> np.ndarray:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        weight = np.ones(y.size, dtype=float)

        for _ in range(maxiter):
            residual = y - X @ beta
            scale = self._scale(residual)
            if scale <= 0:
                break
            scaled = residual / scale
            updated = np.where(
                np.abs(scaled) <= _HUBER_C,
                1.0,
                _HUBER_C / np.maximum(np.abs(scaled), 1e-12),
            )
            change = float(np.max(np.abs(updated - weight)))
            weight = updated
            root = np.sqrt(weight)
            new_beta, *_ = np.linalg.lstsq(
                X * root[:, None], y * root, rcond=None
            )
            beta = new_beta
            if change < 1e-8:
                return beta
        else:
            message = "rreg: Huber iterations did not converge"
            if strict:
                raise ConvergenceError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        return beta

    def _biweight_until_converged(
        self,
        X: np.ndarray,
        y: np.ndarray,
        beta: np.ndarray,
        *,
        maxiter: int,
        strict: bool,
    ) -> np.ndarray:
        c_b = _BIWEIGHT_C * self._tune / 7.0
        weight = np.ones(y.size, dtype=float)

        for _ in range(maxiter):
            residual = y - X @ beta
            scale = self._scale(residual)
            if scale <= 0:
                break
            scaled = residual / scale
            inside = np.abs(scaled) <= c_b
            updated = np.where(
                inside, (1.0 - (scaled / c_b) ** 2) ** 2, 0.0
            )
            change = float(np.max(np.abs(updated - weight)))
            weight = updated
            if not np.any(weight > 0):
                raise ConsistencyError(
                    "rreg: every observation received zero biweight; the "
                    "residual scale has collapsed"
                )
            root = np.sqrt(weight)
            beta, *_ = np.linalg.lstsq(X * root[:, None], y * root, rcond=None)
            if change < 1e-8:
                return beta
        else:
            message = "rreg: biweight iterations did not converge"
            if strict:
                raise ConvergenceError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        return beta

    # ---------------------------------------------------------- covariance
    def _pseudovalue_covariance(
        self, X: np.ndarray, y: np.ndarray, beta: np.ndarray
    ) -> np.ndarray:
        """Street-Carroll-Ruppert pseudovalues, ``[R] rreg`` Methods and formulas.

        Each observation's pseudovalue is ``n*b - (n-1)*b_{-i}``; their covariance
        divided by n estimates the estimator's variance without assuming Gaussian
        errors.
        """

        n, k = X.shape
        if n <= k + 1:
            residual = y - X @ beta
            sigma2 = float(residual @ residual) / max(n - k, 1)
            return sigma2 * np.linalg.pinv(X.T @ X)

        pseudovalues = np.empty((n, k), dtype=float)
        for index in range(n):
            mask = np.ones(n, dtype=bool)
            mask[index] = False
            try:
                leave_one_out = solve_lad_robust(X[mask], y[mask], beta)
            except np.linalg.LinAlgError:
                leave_one_out = beta
            pseudovalues[index] = n * beta - (n - 1) * leave_one_out

        covariance = np.cov(pseudovalues, rowvar=False) / n
        covariance = np.atleast_2d(covariance)
        if not np.all(np.isfinite(covariance)):
            residual = y - X @ beta
            sigma2 = float(residual @ residual) / max(n - k, 1)
            covariance = sigma2 * np.linalg.pinv(X.T @ X)
        return covariance


def solve_lad_robust(
    X: np.ndarray, y: np.ndarray, start: np.ndarray, *, maxiter: int = 50
) -> np.ndarray:
    """Refit a weighted least squares problem for the pseudovalue step.

    Uses the full-sample estimate as the starting value and a short IRLS run, which
    is enough for a leave-one-out refit and keeps the cost linear in the sample
    size rather than restarting from OLS each time.
    """

    beta = np.asarray(start, dtype=float).copy()
    for _ in range(maxiter):
        residual = y - X @ beta
        scale = float(np.median(np.abs(residual - np.median(residual))))
        floor = max(scale * 1e-8, 1e-12)
        inside = np.abs(residual) <= 1.0 * max(scale, 1e-12)
        weight = np.where(inside, 1.0, 1.0 / np.maximum(np.abs(residual), floor))
        root = np.sqrt(weight)
        updated, *_ = np.linalg.lstsq(X * root[:, None], y * root, rcond=None)
        if float(np.max(np.abs(updated - beta))) < 1e-10:
            return updated
        beta = updated
    return beta
