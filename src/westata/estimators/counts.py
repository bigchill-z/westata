"""Zero-inflated and truncated count models.

==========================  ==================
westata                     Stata
==========================  ==================
``ZeroInflatedPoisson``     ``zip``
``ZeroInflatedNegativeBinomial``  ``zinb``
``TruncatedPoisson``        ``ztp`` / ``tpoisson``
``TruncatedNegativeBinomial``  ``ztnb`` / ``tnbreg``
==========================  ==================

All four maximise a self-written likelihood rather than wrapping statsmodels.
The reason is parameterisation: statsmodels' ``ZeroInflatedPoisson`` models the
zero-inflation probability through ``exog_infl`` with its own sign convention,
and it does not expose a truncated-count model at all. Sorting that out for every
family would leave four different conventions to keep straight; one explicit
likelihood per family is easier to verify against Stata and impossible to
"accidentally" mis-specify.

Likelihoods
-----------
Zero-inflated::

    P(y = 0) = pi + (1 - pi) f(0; mu)
    P(y = k) = (1 - pi) f(k; mu),  k > 0
    pi = logistic(w' gamma)

with ``f`` the Poisson or negative-binomial mass.  When ``inflate is None`` the
model collapses to the plain Poisson/NB because ``pi`` is then fixed at 0.

Truncated (lower limit ``a``, optional upper ``b``)::

    P(y = k | a <= k <= b) = f(k; mu) / (F(b; mu) - F(a - 1; mu))

Zero truncation is the special case ``a = 1``.
"""

from __future__ import annotations

import warnings
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import gammaln, logsumexp

from westata.errors import ConsistencyError, ConvergenceError, UnsupportedError
from westata.formula.factor import build_design
from westata.results.base import StataResults

__all__ = [
    "TruncatedNegativeBinomial",
    "TruncatedPoisson",
    "ZeroInflatedNegativeBinomial",
    "ZeroInflatedPoisson",
]


# ---------------------------------------------------------------- mass functions


def _poisson_logpmf(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return y * np.log(np.maximum(mu, 1e-300)) - mu - gammaln(y + 1.0)


def _nb_logpmf(y: np.ndarray, mu: np.ndarray, alpha: float) -> np.ndarray:
    """Negative-binomial mass with variance ``mu + alpha * mu^2`` (Stata's)."""

    size = 1.0 / max(alpha, 1e-12)
    return (
        gammaln(y + size)
        - gammaln(size)
        - gammaln(y + 1.0)
        + size * np.log(size / (size + mu))
        + y * np.log(np.maximum(mu / (size + mu), 1e-300))
    )


def _poisson_cdf_counts(mu: np.ndarray, k: np.ndarray) -> np.ndarray:
    """``P(Y <= k)`` for each observation, vectorised over observations."""

    return stats.poisson.cdf(k, mu)


class _CountBase:
    """Shared plumbing for the self-implemented count likelihoods."""

    _stata_command = ""
    _optimizer = "bfgs"

    def __init__(self, endog: Any, exog: Any, *, param_names: list[str] | None = None) -> None:
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
        self._model_data: pd.DataFrame | None = None

    @staticmethod
    def _formula_parts(formula: str) -> tuple[str, str]:
        lhs, rhs = formula.split("~", 1)
        return lhs.strip(), rhs.strip()

    def _optimise(
        self, objective: Any, start: np.ndarray, *, maxiter: int, strict: bool, label: str
    ) -> np.ndarray:
        result = optimize.minimize(
            objective, start, method=self._optimizer.upper(),
            options={"maxiter": maxiter},
        )
        if not result.success:
            refined = optimize.minimize(
                objective,
                np.asarray(result.x, dtype=float),
                method="Nelder-Mead",
                options={"maxiter": maxiter, "xatol": 1e-9, "fatol": 1e-9},
            )
            if refined.fun <= result.fun:
                result = refined
        if not result.success:
            message = f"{label}: optimiser did not converge ({result.message})"
            if strict:
                raise ConvergenceError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        return np.asarray(result.x, dtype=float)

    def _covariance(self, objective: Any, point: np.ndarray) -> np.ndarray:
        size = point.size
        step = 1e-5
        hessian = np.zeros((size, size), dtype=float)
        increments = step * np.maximum(np.abs(point), 1.0)
        for i in range(size):
            for j in range(i, size):
                ei = np.zeros(size)
                ej = np.zeros(size)
                ei[i] = increments[i]
                ej[j] = increments[j]
                if i == j:
                    value = (
                        objective(point + ei) - 2 * objective(point) + objective(point - ei)
                    ) / increments[i] ** 2
                else:
                    value = (
                        objective(point + ei + ej)
                        - objective(point + ei - ej)
                        - objective(point - ei + ej)
                        + objective(point - ei - ej)
                    ) / (4 * increments[i] * increments[j])
                hessian[i, j] = value
                hessian[j, i] = value
        try:
            covariance = np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            covariance = np.linalg.pinv(hessian)
        if not np.all(np.isfinite(covariance)):
            raise ConvergenceError(
                f"{self._stata_command}: the observed information is not "
                "invertible; the model is likely not identified"
            )
        return covariance

    def _assemble(
        self,
        params: np.ndarray,
        names: list[str],
        covariance: np.ndarray,
        *,
        llf: float,
        ll0: float,
        nobs: int,
        n_parameters: int,
        extra: Mapping[str, Any] | None = None,
    ) -> StataResults:
        chi2 = 2.0 * (llf - ll0) if np.isfinite(ll0) else float("nan")
        pseudo = 1.0 - llf / ll0 if np.isfinite(ll0) and ll0 != 0 else float("nan")
        e: dict[str, Any] = {
            "N": nobs,
            "df_m": float(n_parameters - 1),
            "rank": int(params.size),
            "ll": float(llf),
            "ll_0": float(ll0),
            "chi2": chi2,
            "chi2type": "LR",
            "p": float(stats.chi2.sf(chi2, max(n_parameters - 1, 1))),
            "r2_p": pseudo,
        }
        if extra:
            e.update(dict(extra))

        mu = np.exp(self._X @ params[: self._X.shape[1]])
        return StataResults(
            params=params,
            param_names=names,
            cov=covariance,
            resid=self._y - mu,
            fittedvalues=mu,
            nobs=nobs,
            df_resid=float(nobs),
            df_model=float(n_parameters - 1),
            dist="z",
            model_name=self._stata_command,
            cov_type="nonrobust",
            e=e,
            design_matrix=(self._X, list(self._param_names)),
            model_data=self._model_data,
        )


# ------------------------------------------------------------- zero-inflated


class ZeroInflatedPoisson(_CountBase):
    """Zero-inflated Poisson (Stata ``zip``)."""

    _stata_command = "zip"

    def __init__(
        self,
        endog: Any,
        exog: Any,
        *,
        inflate_exog: Any = None,
        param_names: list[str] | None = None,
        inflate_names: list[str] | None = None,
    ) -> None:
        super().__init__(endog, exog, param_names=param_names)
        self._inflate_exog = (
            None
            if inflate_exog is None
            else np.asarray(inflate_exog, dtype=float)
        )
        self._inflate_names = (
            list(inflate_names) if inflate_names is not None else ["_cons"]
        )

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        inflate: str | None = None,
        subset: Any = None,
    ) -> "ZeroInflatedPoisson":
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if subset is not None:
            data = data.loc[subset]
        data = data.reset_index(drop=True)

        X, names, kept_index = build_design(formula, data)
        lhs, _ = cls._formula_parts(formula)
        y = data.loc[kept_index, lhs].to_numpy(dtype=float)

        inflate_exog = None
        inflate_names = None
        if inflate is not None:
            # `_cons` is Stata's spelling for a constant-only equation; patsy
            # would try to look it up as a data column.
            inflate_formula = (
                f"{lhs} ~ 1" if inflate.strip() == "_cons" else f"{lhs} ~ {inflate}"
            )
            inflate_exog, inflate_names, inflate_index = build_design(
                inflate_formula, data
            )
            if len(inflate_index) != len(kept_index):
                raise ConsistencyError(
                    "zip: the inflation equation and the count equation must "
                    "describe the same observations"
                )

        model = cls(
            y,
            X,
            inflate_exog=inflate_exog,
            param_names=names,
            inflate_names=inflate_names,
        )
        model._model_data = data.loc[kept_index].reset_index(drop=True)
        return model

    # -- likelihood -------------------------------------------------------
    def _loglike(self, theta: np.ndarray) -> float:
        k = self._X.shape[1]
        beta = theta[:k]
        mu = np.exp(self._X @ beta)

        if self._inflate_exog is None:
            return float(np.sum(_poisson_logpmf(self._y, mu)))

        gamma = theta[k:]
        pi = 1.0 / (1.0 + np.exp(-(self._inflate_exog @ gamma)))
        log_f = _poisson_logpmf(self._y, mu)
        zero = self._y == 0

        # log of the mixture probability, computed in log space so a
        # near-certain structural zero does not underflow.
        log_zero = np.logaddexp(
            np.log(np.maximum(pi, 1e-300)),
            np.log(np.maximum(1.0 - pi, 1e-300)) + log_f,
        )
        log_positive = np.log(np.maximum(1.0 - pi, 1e-300)) + log_f
        return float(np.sum(np.where(zero, log_zero, log_positive)))

    def _null_loglike(self) -> float:
        ybar = float(np.mean(self._y))
        return float(np.sum(_poisson_logpmf(self._y, np.full(self._y.size, ybar))))

    def fit(
        self,
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
        *,
        maxiter: int = 500,
        strict: bool = False,
    ) -> StataResults:
        if cov_type not in (None, "nonrobust"):
            raise ValueError(f"zip: only cov_type='nonrobust' is supported, got {cov_type!r}")

        k = self._X.shape[1]
        beta0, *_ = np.linalg.lstsq(self._X, np.log(np.maximum(self._y, 0.5)), rcond=None)
        start = beta0.copy()
        if self._inflate_exog is not None:
            start = np.concatenate([start, np.zeros(self._inflate_exog.shape[1])])

        def objective(theta: np.ndarray) -> float:
            value = self._loglike(theta)
            return -value if np.isfinite(value) else 1e12

        theta = self._optimise(objective, start, maxiter=maxiter, strict=strict, label="zip")
        covariance = self._covariance(objective, theta)
        names = list(self._param_names)
        if self._inflate_exog is not None:
            names = names + [f"inflate:{name}" for name in self._inflate_names]
        return self._assemble(
            theta,
            names,
            covariance,
            llf=self._loglike(theta),
            ll0=self._null_loglike(),
            nobs=self._y.size,
            n_parameters=theta.size,
        )


class ZeroInflatedNegativeBinomial(ZeroInflatedPoisson):
    """Zero-inflated negative binomial (Stata ``zinb``)."""

    _stata_command = "zinb"

    def _loglike(self, theta: np.ndarray) -> float:
        k = self._X.shape[1]
        beta = theta[:k]
        mu = np.exp(self._X @ beta)
        # The dispersion is the last parameter, on a log scale for positivity.
        alpha = float(np.exp(theta[-1]))
        log_f = _nb_logpmf(self._y, mu, alpha)

        if self._inflate_exog is None:
            return float(np.sum(log_f))

        gamma = theta[k:-1]
        pi = 1.0 / (1.0 + np.exp(-(self._inflate_exog @ gamma)))
        zero = self._y == 0
        log_zero = np.logaddexp(
            np.log(np.maximum(pi, 1e-300)),
            np.log(np.maximum(1 - pi, 1e-300)) + log_f,
        )
        positive_part = np.where(zero, 0.0, np.log(np.maximum(1 - pi, 1e-300)) + log_f)
        return float(np.sum(np.where(zero, log_zero, positive_part)))

    def _null_loglike(self) -> float:
        ybar = float(np.mean(self._y))
        variance = float(np.var(self._y))
        alpha = max((variance - ybar) / max(ybar**2, 1e-12), 1e-6)
        return float(np.sum(_nb_logpmf(self._y, np.full(self._y.size, ybar), alpha)))

    def fit(
        self,
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
        *,
        maxiter: int = 800,
        strict: bool = False,
    ) -> StataResults:
        if cov_type not in (None, "nonrobust"):
            raise ValueError(f"zinb: only cov_type='nonrobust' is supported, got {cov_type!r}")

        k = self._X.shape[1]
        beta0, *_ = np.linalg.lstsq(self._X, np.log(np.maximum(self._y, 0.5)), rcond=None)
        alpha0 = max((float(np.var(self._y)) - float(np.mean(self._y))) / max(float(np.mean(self._y)) ** 2, 1e-12), 1e-3)
        start = np.concatenate([beta0, [np.log(alpha0)]])
        if self._inflate_exog is not None:
            start = np.concatenate([start[:k], np.zeros(self._inflate_exog.shape[1]), start[k:]])

        def objective(theta: np.ndarray) -> float:
            value = self._loglike(theta)
            return -value if np.isfinite(value) else 1e12

        theta = self._optimise(objective, start, maxiter=maxiter, strict=strict, label="zinb")
        covariance = self._covariance(objective, theta)

        names = list(self._param_names)
        if self._inflate_exog is not None:
            names = names + [f"inflate:{name}" for name in self._inflate_names]
        names = names + ["/lnalpha"]
        return self._assemble(
            theta,
            names,
            covariance,
            llf=self._loglike(theta),
            ll0=self._null_loglike(),
            nobs=self._y.size,
            n_parameters=theta.size - 1,
            extra={"alpha": float(np.exp(theta[-1]))},
        )


# ---------------------------------------------------------------- truncated


class _TruncatedBase(_CountBase):
    """Shared normalisation for count data observed inside ``[lower, upper]``."""

    _negative_binomial = False

    def __init__(
        self,
        endog: Any,
        exog: Any,
        *,
        lower: float = 1,
        upper: float | None = None,
        param_names: list[str] | None = None,
    ) -> None:
        super().__init__(endog, exog, param_names=param_names)
        if lower is None:
            lower = 1
        if upper is not None and not lower <= upper:
            raise ConsistencyError(
                f"{self._stata_command}: lower truncation point ({lower}) must not "
                f"exceed the upper one ({upper})"
            )
        if np.any(self._y < lower):
            raise ConsistencyError(
                f"{self._stata_command}: the data contain observations below the "
                f"truncation point ({lower}); truncation requires every y >= {lower}"
            )
        if upper is not None and np.any(self._y > upper):
            raise ConsistencyError(
                f"{self._stata_command}: the data contain observations above the "
                f"truncation point ({upper})"
            )
        self._lower = float(lower)
        self._upper = None if upper is None else float(upper)

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        lower: float = 1,
        upper: float | None = None,
        subset: Any = None,
    ) -> "_TruncatedBase":
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if subset is not None:
            data = data.loc[subset]
        data = data.reset_index(drop=True)
        X, names, kept_index = build_design(formula, data)
        lhs, _ = cls._formula_parts(formula)
        y = data.loc[kept_index, lhs].to_numpy(dtype=float)
        model = cls(y, X, lower=lower, upper=upper, param_names=names)
        model._model_data = data.loc[kept_index].reset_index(drop=True)
        return model

    def _log_mass_denominator(self, mu: np.ndarray, alpha: float | None) -> np.ndarray:
        """``log(F(upper) - F(lower - 1))``, the truncation normaliser."""

        lower_cdf = self._cdf(self._lower - 1.0, mu, alpha)
        if self._upper is None:
            return np.log(np.maximum(1.0 - lower_cdf, 1e-300))
        upper_cdf = self._cdf(self._upper, mu, alpha)
        return np.log(np.maximum(upper_cdf - lower_cdf, 1e-300))

    def _cdf(self, k: float, mu: np.ndarray, alpha: float | None) -> np.ndarray:
        if alpha is None:
            return stats.poisson.cdf(k, mu)
        size = 1.0 / max(alpha, 1e-12)
        return stats.nbinom.cdf(k, size, size / (size + mu))

    def _logpmf(self, mu: np.ndarray, alpha: float | None) -> np.ndarray:
        if alpha is None:
            return _poisson_logpmf(self._y, mu)
        return _nb_logpmf(self._y, mu, alpha)


class TruncatedPoisson(_TruncatedBase):
    """Poisson truncated to ``[lower, upper]`` (Stata ``ztp``/``tpoisson``)."""

    _stata_command = "ztp"

    def _loglike(self, theta: np.ndarray) -> float:
        k = self._X.shape[1]
        mu = np.exp(self._X @ theta[:k])
        log_f = _poisson_logpmf(self._y, mu)
        return float(np.sum(log_f - self._log_mass_denominator(mu, None)))

    def _null_loglike(self) -> float:
        ybar = max(float(np.mean(self._y)), 1e-6)
        mu = np.full(self._y.size, ybar)
        return float(
            np.sum(_poisson_logpmf(self._y, mu) - self._log_mass_denominator(mu, None))
        )

    def fit(
        self,
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
        *,
        maxiter: int = 500,
        strict: bool = False,
    ) -> StataResults:
        if cov_type not in (None, "nonrobust"):
            raise ValueError(f"ztp: only cov_type='nonrobust' is supported, got {cov_type!r}")
        beta0, *_ = np.linalg.lstsq(self._X, np.log(np.maximum(self._y, 0.5)), rcond=None)

        def objective(theta: np.ndarray) -> float:
            value = self._loglike(theta)
            return -value if np.isfinite(value) else 1e12

        theta = self._optimise(objective, beta0, maxiter=maxiter, strict=strict, label="ztp")
        covariance = self._covariance(objective, theta)
        return self._assemble(
            theta,
            list(self._param_names),
            covariance,
            llf=self._loglike(theta),
            ll0=self._null_loglike(),
            nobs=self._y.size,
            n_parameters=theta.size,
            extra={"lower": self._lower, "upper": self._upper},
        )


class TruncatedNegativeBinomial(_TruncatedBase):
    """Negative binomial truncated to ``[lower, upper]`` (Stata ``tnbreg``/``ztnb``)."""

    _stata_command = "tnbreg"
    _negative_binomial = True

    def _split(self, theta: np.ndarray) -> tuple[np.ndarray, float]:
        k = self._X.shape[1]
        return theta[:k], float(np.exp(theta[k]))

    def _loglike(self, theta: np.ndarray) -> float:
        beta, alpha = self._split(theta)
        mu = np.exp(self._X @ beta)
        log_f = _nb_logpmf(self._y, mu, alpha)
        return float(np.sum(log_f - self._log_mass_denominator(mu, alpha)))

    def _null_loglike(self) -> float:
        ybar = max(float(np.mean(self._y)), 1e-6)
        variance = float(np.var(self._y))
        alpha = max((variance - ybar) / max(ybar**2, 1e-12), 1e-4)
        mu = np.full(self._y.size, ybar)
        return float(np.sum(_nb_logpmf(self._y, mu, alpha) - self._log_mass_denominator(mu, alpha)))

    def fit(
        self,
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
        *,
        maxiter: int = 800,
        strict: bool = False,
    ) -> StataResults:
        if cov_type not in (None, "nonrobust"):
            raise ValueError(f"tnbreg: only cov_type='nonrobust' is supported, got {cov_type!r}")
        k = self._X.shape[1]
        beta0, *_ = np.linalg.lstsq(self._X, np.log(np.maximum(self._y, 0.5)), rcond=None)
        ybar = max(float(np.mean(self._y)), 1e-6)
        alpha0 = max((float(np.var(self._y)) - ybar) / max(ybar**2, 1e-12), 1e-3)
        start = np.concatenate([beta0, [np.log(alpha0)]])

        def objective(theta: np.ndarray) -> float:
            value = self._loglike(theta)
            return -value if np.isfinite(value) else 1e12

        theta = self._optimise(objective, start, maxiter=maxiter, strict=strict, label="tnbreg")
        covariance = self._covariance(objective, theta)
        return self._assemble(
            theta,
            list(self._param_names) + ["/lnalpha"],
            covariance,
            llf=self._loglike(theta),
            ll0=self._null_loglike(),
            nobs=self._y.size,
            n_parameters=k,
            extra={
                "alpha": float(np.exp(theta[k])),
                "lower": self._lower,
                "upper": self._upper,
            },
        )
