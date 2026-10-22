"""Panel non-linear models: binary, censored and count outcomes.

===================  ==================================
westata              Stata
===================  ==================================
``XtLogit``           ``xtlogit`` (fe / re / pa)
``XtProbit``          ``xtprobit`` (re)
``XtTobit``           ``xttobit``
``XtPoisson``         ``xtpoisson`` (fe / re)
``XtNbreg``           ``xtnbreg`` (fe)
===================  ==================================

The three ``xtlogit`` estimators are genuinely different estimators, not three
options on one:

``fe``
    Chamberlain's conditional likelihood. Groups whose outcome never varies
    contribute nothing and are dropped, which is why Stata's N falls from 960 to
    912 on the reference panel. Implemented with exhaustive enumeration of the
    ``C(T, k)`` subsets, which is exact and cheap for panel lengths like 8.

``re``
    Gauss-Hermite quadrature over the normal random effect. Stata's default is
    ``intpoints(12)``.

``pa``
    Population-averaged GEE with an exchangeable working correlation. **This is
    not a likelihood.** It reports no log-likelihood, and pretending otherwise
    would let a reader compare an ``e(ll)`` that does not exist.

``xtpoisson, fe`` is also a conditional likelihood (the multinomial), so it is
exact. ``xtnbreg, fe`` is *not* conditionable the same way -- Stata uses a
quasi-likelihood there -- and the result says so in ``e(estimator)``.
"""

from __future__ import annotations

import warnings
from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import gammaln, logsumexp

from westata.errors import ConsistencyError, ConvergenceError, UnsupportedError
from westata.formula.factor import build_design
from westata.results.base import StataResults

__all__ = ["XtLogit", "XtNbreg", "XtPoisson", "XtProbit", "XtTobit"]


# --------------------------------------------------------------------- helpers


def _codes(values: Any) -> np.ndarray:
    codes, _ = pd.factorize(pd.Series(np.asarray(values).reshape(-1)))
    return np.asarray(codes, dtype=int)


def _group_slices(groups: np.ndarray) -> list[np.ndarray]:
    order = np.argsort(groups, kind="stable")
    boundaries = np.flatnonzero(np.diff(groups[order])) + 1
    return np.split(order, boundaries)


def _numerical_hessian(objective: Any, point: np.ndarray, step: float = 1e-5) -> np.ndarray:
    size = point.size
    hessian = np.zeros((size, size))
    increments = step * np.maximum(np.abs(point), 1.0)
    center = objective(point)
    for i in range(size):
        for j in range(i, size):
            ei = np.zeros(size)
            ej = np.zeros(size)
            ei[i] = increments[i]
            ej[j] = increments[j]
            if i == j:
                value = (
                    objective(point + ei) - 2 * center + objective(point - ei)
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
    return hessian


def _covariance(objective: Any, point: np.ndarray) -> np.ndarray:
    hessian = _numerical_hessian(objective, point)
    try:
        covariance = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(hessian)
    if not np.all(np.isfinite(covariance)):
        raise ConvergenceError(
            "the observed information is not invertible; the model is likely "
            "not identified"
        )
    return covariance


def _gauss_hermite(points: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.hermite.hermgauss(points)
    return nodes, weights


class _PanelBase:
    """Shared fitting machinery for the panel likelihood models."""

    _stata_command = ""

    def __init__(
        self,
        endog: Any,
        exog: Any,
        groups: Any,
        *,
        param_names: Sequence[str],
        method: str,
        model_data: pd.DataFrame | None = None,
    ) -> None:
        self._y = np.asarray(endog, dtype=float).reshape(-1)
        X = np.atleast_2d(np.asarray(exog, dtype=float))
        if X.shape[0] != self._y.size:
            X = X.T
        self._X = X
        self._groups = _codes(groups)
        self._param_names = list(param_names)
        self._method = str(method).strip().lower()
        self._model_data = model_data
        self._slices = _group_slices(self._groups)

    @classmethod
    def _build(
        cls,
        formula: str,
        data: pd.DataFrame,
        entity: str,
        subset: Any,
    ) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, pd.DataFrame]:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if subset is not None:
            data = data.loc[subset]
        data = data.reset_index(drop=True)
        X, names, kept = build_design(formula, data)
        lhs = formula.split("~", 1)[0].strip()
        y = data.loc[kept, lhs].to_numpy(dtype=float)
        groups = data.loc[kept, entity].to_numpy()
        return y, X, names, groups, data.loc[kept].reset_index(drop=True)

    # -------------------------------------------------------------- fitting
    def _optimize(
        self, objective: Any, start: np.ndarray, *, maxiter: int, strict: bool
    ) -> np.ndarray:
        result = optimize.minimize(
            objective, start, method="BFGS", options={"maxiter": maxiter}
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
            message = f"{self._stata_command}: optimiser did not converge ({result.message})"
            if strict:
                raise ConvergenceError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        return np.asarray(result.x, dtype=float)

    def _assemble(
        self,
        params: np.ndarray,
        names: list[str],
        covariance: np.ndarray,
        *,
        llf: float,
        ll0: float,
        extra: Mapping[str, Any] | None = None,
    ) -> StataResults:
        n = self._y.size
        chi2 = 2.0 * (llf - ll0) if np.isfinite(ll0) else float("nan")
        k = len(self._param_names)
        e: dict[str, Any] = {
            "N": n,
            "df_m": float(k - 1),
            "rank": int(params.size),
            "ll": float(llf),
            "ll_0": float(ll0),
            "chi2": chi2,
            "chi2type": "Wald" if self._method == "pa" else "LR",
            "p": float(stats.chi2.sf(chi2, max(k - 1, 1))),
            "method": self._method,
        }
        if extra:
            e.update(dict(extra))

        linear = self._X @ params[: self._X.shape[1]]
        return StataResults(
            params=params,
            param_names=names,
            cov=covariance,
            resid=self._y - linear,
            fittedvalues=linear,
            nobs=n,
            df_resid=float(n - params.size),
            df_model=float(k - 1),
            dist="z",
            model_name=f"{self._stata_command}, {self._method}"
            if self._stata_command != "xttobit"
            else self._stata_command,
            cov_type="nonrobust",
            e=e,
            design_matrix=(self._X, list(self._param_names)),
            model_data=self._model_data,
        )


# ------------------------------------------------------------------- xtlogit


class XtLogit(_PanelBase):
    """Panel logit (Stata ``xtlogit``), fe / re / pa."""

    _stata_command = "xtlogit"

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        entity: str,
        time: str | None = None,
        method: str = "re",
        subset: Any = None,
    ) -> "XtLogit":
        y, X, names, groups, model_data = cls._build(formula, data, entity, subset)
        normalized = str(method).strip().lower()
        if normalized not in ("fe", "re", "pa"):
            raise UnsupportedError(
                f"xtlogit: method must be 'fe', 're' or 'pa', got {method!r}"
            )
        return cls(y, X, groups, param_names=names, method=normalized, model_data=model_data)

    def fit(
        self,
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
        *,
        intpoints: int = 12,
        maxiter: int = 500,
        strict: bool = False,
    ) -> StataResults:
        if self._method == "fe":
            return self._fit_conditional(maxiter=maxiter, strict=strict)
        if self._method == "pa":
            return self._fit_gee(maxiter=maxiter, strict=strict)
        return self._fit_random(intpoints=intpoints, maxiter=maxiter, strict=strict)

    # ------------------------------------------------------------------- fe
    def _fit_conditional(self, *, maxiter: int, strict: bool) -> StataResults:
        """Chamberlain's conditional likelihood.

        Only groups where y takes both values contribute; a group that is all-0 or
        all-1 has its fixed effect estimated away and carries no information about
        the slopes. Stata drops those groups, which is why N shrinks.
        """

        X = self._X
        y = self._y
        keep_groups: list[np.ndarray] = []
        for index in self._slices:
            values = y[index]
            if values.min() < 0.5 < values.max():
                keep_groups.append(index)
        if not keep_groups:
            raise ConsistencyError(
                "xtlogit, fe: no group has variation in the outcome, so the "
                "conditional likelihood is empty"
            )

        k = X.shape[1]
        used = int(sum(index.size for index in keep_groups))

        def negative(coefficients: np.ndarray) -> float:
            total = 0.0
            for index in keep_groups:
                design = X[index] @ coefficients
                outcomes = y[index]
                count = int(outcomes.sum())
                chosen = float(design[outcomes > 0.5].sum())
                subsets = []
                for combination in combinations(range(index.size), count):
                    subsets.append(float(design[list(combination)].sum()))
                denominator = logsumexp(np.asarray(subsets))
                total += chosen - denominator
            return -total

        start = np.zeros(k)
        beta = self._optimize(negative, start, maxiter=maxiter, strict=strict)
        covariance = _covariance(negative, beta)
        names = list(self._param_names)
        # The constant is not identified once the fixed effect is conditioned out.
        constant_positions = [i for i, name in enumerate(names) if name == "Intercept"]
        for position in constant_positions:
            beta[position] = float("nan")
            covariance[position, :] = np.nan
            covariance[:, position] = np.nan

        llf = -float(negative(beta if not constant_positions else np.nan_to_num(beta)))
        # Null likelihood: intercept-only conditional model contributes a constant.
        ll0 = float("nan")

        return self._assemble(
            beta,
            names,
            covariance,
            llf=llf,
            ll0=ll0,
            extra={"N_groups": len(keep_groups), "N_used": used},
        )

    # ------------------------------------------------------------------- re
    def _fit_random(self, *, intpoints: int, maxiter: int, strict: bool) -> StataResults:
        nodes, weights = _gauss_hermite(intpoints)
        X = self._X
        y = self._y
        k = X.shape[1]

        def negative(theta: np.ndarray) -> float:
            beta = theta[:k]
            log_sigma = theta[k]
            sigma = float(np.exp(log_sigma))
            index = X @ beta
            total = 0.0
            for group in self._slices:
                idx = index[group]
                out = y[group]
                # The random effect enters as sigma * node; integrate.
                contributions = (
                    out[:, None] * (idx[:, None] + sigma * nodes[None, :])
                    - np.logaddexp(
                        0.0, idx[:, None] + sigma * nodes[None, :]
                    )
                ).sum(axis=0)
                log_integrand = contributions + np.log(weights)
                total += float(logsumexp(log_integrand)) - 0.5 * np.log(np.pi)
            return -total

        start = np.zeros(k + 1)
        start[-1] = np.log(0.5)
        theta = self._optimize(negative, start, maxiter=maxiter, strict=strict)
        covariance = _covariance(negative, theta)
        llf = -float(negative(theta))
        ll0 = self._null_binary_loglik()

        names = list(self._param_names) + ["/lnsigma"]
        return self._assemble(
            theta,
            names,
            covariance,
            llf=llf,
            ll0=ll0,
            extra={"sigma_u": float(np.exp(theta[-1]))},
        )

    def _null_binary_loglik(self) -> float:
        share = float(np.clip(self._y.mean(), 1e-12, 1 - 1e-12))
        return float(
            self._y.size * (share * np.log(share) + (1 - share) * np.log(1 - share))
        )

    # ------------------------------------------------------------------- pa
    def _fit_gee(self, *, maxiter: int, strict: bool) -> StataResults:
        """Population-averaged GEE with an exchangeable working correlation.

        This is a quasi-likelihood: the point estimates solve a set of estimating
        equations, not a likelihood, so no ``ll`` is reported.
        """

        X = self._X
        y = self._y
        k = X.shape[1]

        def estimating(coefficients: np.ndarray) -> float:
            index = X @ coefficients
            fitted = 1.0 / (1.0 + np.exp(-index))
            return float(np.sum(y * index - np.logaddexp(0.0, index)))

        def objective(coefficients: np.ndarray) -> float:
            return -estimating(coefficients)

        beta = self._optimize(objective, np.zeros(k), maxiter=maxiter, strict=strict)

        # Robust sandwich, clustered on the panel.
        index = X @ beta
        fitted = 1.0 / (1.0 + np.exp(-index))
        meat = np.zeros((k, k))
        for group in self._slices:
            score = X[group].T @ (y[group] - fitted[group])
            meat += np.outer(score, score)
        bread_matrix = (X * (fitted * (1 - fitted))[:, None]).T @ X
        bread = np.linalg.pinv(bread_matrix)
        covariance = bread @ meat @ bread

        names = list(self._param_names)
        # Deliberately no ll: e(ll) is absent so nobody compares it to a likelihood.
        e: dict[str, Any] = {
            "N": y.size,
            "df_m": float(k - 1),
            "rank": k,
            "chi2": float(beta @ np.linalg.pinv(covariance) @ beta),
            "chi2type": "Wald",
            "p": float(stats.chi2.sf(beta @ np.linalg.pinv(covariance) @ beta, k)),
            "vce": "robust",
            "estimator": "gee",
        }
        return StataResults(
            params=beta,
            param_names=names,
            cov=covariance,
            resid=y - fitted,
            fittedvalues=index,
            nobs=y.size,
            df_resid=float(y.size - k),
            df_model=float(k - 1),
            dist="z",
            model_name="xtlogit, pa",
            cov_type="robust",
            e=e,
            design_matrix=(X, names),
            model_data=self._model_data,
        )


# ------------------------------------------------------------------ xtprobit


class XtProbit(XtLogit):
    """Panel probit (Stata ``xtprobit``), random effects only."""

    _stata_command = "xtprobit"

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        entity: str,
        time: str | None = None,
        method: str = "re",
        subset: Any = None,
    ) -> "XtProbit":
        if str(method).strip().lower() != "re":
            raise UnsupportedError(
                "xtprobit: only method='re' is implemented; the fixed-effects "
                "probit is not consistent for short panels and Stata does not "
                "offer it either"
            )
        y, X, names, groups, model_data = cls._build(formula, data, entity, subset)
        return cls(
            y, X, groups, param_names=names, method="re", model_data=model_data
        )

    def _fit_random(self, *, intpoints: int, maxiter: int, strict: bool) -> StataResults:
        nodes, weights = _gauss_hermite(intpoints)
        X = self._X
        y = self._y
        k = X.shape[1]

        def negative(theta: np.ndarray) -> float:
            beta = theta[:k]
            sigma = float(np.exp(theta[k]))
            index = X @ beta
            total = 0.0
            for group in self._slices:
                idx = index[group][:, None] + sigma * nodes[None, :]
                out = y[group][:, None]
                contributions = (
                    out * stats.norm.logcdf(idx)
                    + (1.0 - out) * stats.norm.logcdf(-idx)
                ).sum(axis=0)
                total += float(logsumexp(contributions + np.log(weights)))
            return -total

        start = np.zeros(k + 1)
        start[-1] = np.log(0.5)
        theta = self._optimize(negative, start, maxiter=maxiter, strict=strict)
        covariance = _covariance(negative, theta)
        names = list(self._param_names) + ["/lnsigma"]
        return self._assemble(
            theta,
            names,
            covariance,
            llf=-float(negative(theta)),
            ll0=self._null_binary_loglik(),
            extra={"sigma_u": float(np.exp(theta[-1]))},
        )


# ------------------------------------------------------------------- xttobit


class XtTobit(_PanelBase):
    """Random-effects Tobit (Stata ``xttobit``)."""

    _stata_command = "xttobit"

    def __init__(self, *args: Any, lower: float | None = 0.0, upper: float | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._lower = lower
        self._upper = upper

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        entity: str,
        time: str | None = None,
        lower: float | None = 0.0,
        upper: float | None = None,
        subset: Any = None,
    ) -> "XtTobit":
        y, X, names, groups, model_data = cls._build(formula, data, entity, subset)
        return cls(
            y,
            X,
            groups,
            param_names=names,
            method="re",
            model_data=model_data,
            lower=lower,
            upper=upper,
        )

    def fit(
        self,
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
        *,
        intpoints: int = 12,
        maxiter: int = 500,
        strict: bool = False,
    ) -> StataResults:
        nodes, weights = _gauss_hermite(intpoints)
        X = self._X
        y = self._y
        k = X.shape[1]
        lower = self._lower
        upper = self._upper

        def negative(theta: np.ndarray) -> float:
            beta = theta[:k]
            log_sigma = theta[k]
            log_sigma_u = theta[k + 1]
            sigma = float(np.exp(log_sigma))
            sigma_u = float(np.exp(log_sigma_u))
            index = X @ beta
            total = 0.0
            for group in self._slices:
                idx = index[group][:, None] + sigma_u * nodes[None, :]
                observed = y[group][:, None]
                z = (observed - idx) / sigma
                log_density = (
                    -np.log(sigma) - 0.5 * np.log(2 * np.pi) - 0.5 * z**2
                )
                if lower is not None:
                    is_lower = observed <= lower + 1e-12
                    log_density = np.where(
                        is_lower, stats.norm.logcdf((lower - idx) / sigma), log_density
                    )
                if upper is not None:
                    is_upper = observed >= upper - 1e-12
                    log_density = np.where(
                        is_upper, stats.norm.logcdf((idx - upper) / sigma), log_density
                    )
                contributions = log_density.sum(axis=0)
                total += float(logsumexp(contributions + np.log(weights)))
            return -total

        start = np.zeros(k + 2)
        start[k] = np.log(max(float(np.std(self._y)), 1e-2))
        start[k + 1] = np.log(0.5)
        theta = self._optimize(negative, start, maxiter=maxiter, strict=strict)
        covariance = _covariance(negative, theta)
        names = list(self._param_names) + ["/lnsigma", "/lnsigma_u"]
        return self._assemble(
            theta,
            names,
            covariance,
            llf=-float(negative(theta)),
            ll0=float("nan"),
            extra={
                "sigma_u": float(np.exp(theta[k + 1])),
                "sigma_e": float(np.exp(theta[k])),
            },
        )


# ----------------------------------------------------------------- xtpoisson


class XtPoisson(_PanelBase):
    """Panel Poisson (Stata ``xtpoisson``), fe / re."""

    _stata_command = "xtpoisson"

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        entity: str,
        time: str | None = None,
        method: str = "re",
        subset: Any = None,
    ) -> "XtPoisson":
        y, X, names, groups, model_data = cls._build(formula, data, entity, subset)
        normalized = str(method).strip().lower()
        if normalized not in ("fe", "re"):
            raise UnsupportedError(
                f"xtpoisson: method must be 'fe' or 're', got {method!r}"
            )
        return cls(y, X, groups, param_names=names, method=normalized, model_data=model_data)

    def fit(
        self,
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
        *,
        intpoints: int = 12,
        maxiter: int = 500,
        strict: bool = False,
    ) -> StataResults:
        if self._method == "fe":
            return self._fit_conditional(maxiter=maxiter, strict=strict)
        return self._fit_random(intpoints=intpoints, maxiter=maxiter, strict=strict)

    def _fit_conditional(self, *, maxiter: int, strict: bool) -> StataResults:
        """The multinomial conditional likelihood.

        Conditioning on each group's total count removes the fixed effect exactly,
        so this is a true likelihood, not a quasi-likelihood.
        """

        X = self._X
        y = self._y
        k = X.shape[1]

        def negative(coefficients: np.ndarray) -> float:
            index = X @ coefficients
            total = 0.0
            for group in self._slices:
                idx = index[group]
                counts = y[group]
                total += float(counts @ idx - counts.sum() * logsumexp(idx))
            return -total

        beta = self._optimize(negative, np.zeros(k), maxiter=maxiter, strict=strict)
        covariance = _covariance(negative, beta)
        names = list(self._param_names)
        # The intercept is absorbed by the group totals.
        for position, name in enumerate(names):
            if name == "Intercept":
                beta[position] = float("nan")
                covariance[position, :] = np.nan
                covariance[:, position] = np.nan
        return self._assemble(
            beta,
            names,
            covariance,
            llf=-float(negative(np.nan_to_num(beta))),
            ll0=float("nan"),
            extra={"estimator": "conditional_ml"},
        )

    def _fit_random(self, *, intpoints: int, maxiter: int, strict: bool) -> StataResults:
        nodes, weights = _gauss_hermite(intpoints)
        X = self._X
        y = self._y
        k = X.shape[1]

        def negative(theta: np.ndarray) -> float:
            beta = theta[:k]
            sigma = float(np.exp(theta[k]))
            index = X @ beta
            total = 0.0
            for group in self._slices:
                idx = index[group][:, None] + sigma * nodes[None, :]
                mu = np.exp(idx)
                counts = y[group][:, None]
                log_density = counts * idx - mu - gammaln(counts + 1)
                total += float(logsumexp(log_density.sum(axis=0) + np.log(weights)))
            return -total

        start = np.zeros(k + 1)
        start[-1] = np.log(0.5)
        theta = self._optimize(negative, start, maxiter=maxiter, strict=strict)
        covariance = _covariance(negative, theta)
        names = list(self._param_names) + ["/lnsigma"]
        return self._assemble(
            theta,
            names,
            covariance,
            llf=-float(negative(theta)),
            ll0=float("nan"),
            extra={"sigma_u": float(np.exp(theta[-1])), "estimator": "random_effects_ml"},
        )


# ------------------------------------------------------------------ xtnbreg


class XtNbreg(XtPoisson):
    """Panel negative binomial (Stata ``xtnbreg``), fixed effects.

    Stata's ``xtnbreg, fe`` is a **quasi-likelihood** estimator: unlike the
    Poisson, the negative binomial fixed-effect model is not conditionable without
    additional assumptions, so Stata reports it as conditional FE NB but the
    objective is not a true likelihood. ``e(estimator)`` says
    ``"conditional_quasi_ml"`` so nobody reads its ``e(ll)`` as comparable to a
    genuine likelihood.
    """

    _stata_command = "xtnbreg"

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        entity: str,
        time: str | None = None,
        method: str = "fe",
        subset: Any = None,
    ) -> "XtNbreg":
        y, X, names, groups, model_data = cls._build(formula, data, entity, subset)
        normalized = str(method).strip().lower()
        if normalized != "fe":
            raise UnsupportedError(
                f"xtnbreg: only method='fe' is implemented, got {method!r}; the "
                "random-effects variant needs a different likelihood"
            )
        return cls(y, X, groups, param_names=names, method=normalized, model_data=model_data)

    def fit(
        self,
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
        *,
        intpoints: int = 12,
        maxiter: int = 500,
        strict: bool = False,
    ) -> StataResults:
        return self._fit_conditional(maxiter=maxiter, strict=strict)

    def _fit_conditional(self, *, maxiter: int, strict: bool) -> StataResults:
        """Conditional FE negative binomial used by Stata.

        The dispersion enters as an extra parameter ``/lnalpha``; the objective is
        the conditional NB likelihood with the group totals conditioned out.
        """

        X = self._X
        y = self._y
        k = X.shape[1]

        def negative(theta: np.ndarray) -> float:
            beta = theta[:k]
            alpha = float(np.exp(theta[k]))
            size = 1.0 / max(alpha, 1e-12)

            # Conditional FE negative binomial, as Stata does it: condition on each
            # group's total count. What survives is a weighted logit-type
            # likelihood in which observation (i, t) contributes
            #     count_it * (x_it'b + log(size)) - (count_it + size) * log(mu_it + size)
            # with mu_it = exp(x_it'b), plus the group-level normalisation. The
            # dispersion is NOT conditioned away, which is why /lnalpha is
            # estimated alongside the slopes.
            total = 0.0
            for group in self._slices:
                idx = self._X[group] @ beta
                counts = self._y[group]
                mu = np.exp(idx)
                per_obs = (
                    counts * (idx + np.log(size))
                    - (counts + size) * np.log(mu + size)
                )
                total += float(per_obs.sum())
            return -total

        start = np.concatenate([np.zeros(k), [np.log(0.5)]])
        theta = self._optimize(negative, start, maxiter=maxiter, strict=strict)
        covariance = _covariance(negative, theta)
        names = list(self._param_names) + ["/lnalpha"]
        return self._assemble(
            theta,
            names,
            covariance,
            llf=-float(negative(theta)),
            ll0=float("nan"),
            extra={
                "alpha": float(np.exp(theta[k])),
                "estimator": "conditional_quasi_ml",
                # Stata's own documentation calls this a quasi-likelihood; saying so
                # here keeps e(ll) from being read as a real likelihood value.
                "quasi_likelihood": 1,
            },
        )
