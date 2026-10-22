"""Limited dependent-variable models.

======================  ==================
westata                 Stata
======================  ==================
``Truncreg``            ``truncreg``
``Intreg``              ``intreg``
``Heckman``             ``heckman``
``Etregress``           ``etregress``
======================  ==================

All four are fitted by maximum likelihood except ``Heckman``/``Etregress``,
which implement the two-step / control-function estimators and refuse the MLE
variant explicitly: a two-step Heckman whose standard errors were naive would be
indistinguishable from a correct one to a reader, so the choice is stated in
``e(estimator)`` and the docstring rather than hidden.

Likelihoods
-----------
Truncated regression (``[R] truncreg``), observations only seen inside
``(a, b)``::

    lnL = sum[ -ln(sigma) - 0.5 ln(2 pi) - 0.5 ((y - xb)/sigma)^2
               - ln( Phi((b - xb)/sigma) - Phi((a - xb)/sigma) ) ]

Interval regression (``[R] intreg``), only the interval ``[a, b]`` is known::

    lnL = sum[ ln( Phi((b - xb)/sigma) - Phi((a - xb)/sigma) ) ]

Both are maximised with sigma on a log scale so it stays positive, following
:mod:`westata.estimators.tobit`.
"""

from __future__ import annotations

import warnings
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, stats

from westata.errors import ConsistencyError, ConvergenceError, UnsupportedError
from westata.formula.factor import build_design
from westata.results.base import StataResults

__all__ = ["Etregress", "Heckman", "Intreg", "Truncreg"]


def _central_difference_hessian(
    function: Any, point: np.ndarray, step: float = 1e-5
) -> np.ndarray:
    """Numerical Hessian, used for the observed information matrix."""

    point = np.asarray(point, dtype=float)
    size = point.size
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
                    function(point + ei) - 2.0 * function(point) + function(point - ei)
                ) / (increments[i] ** 2)
            else:
                value = (
                    function(point + ei + ej)
                    - function(point + ei - ej)
                    - function(point - ei + ej)
                    + function(point - ei - ej)
                ) / (4.0 * increments[i] * increments[j])
            hessian[i, j] = value
            hessian[j, i] = value
    return hessian


def _covariance_from_hessian(point: np.ndarray, function: Any) -> np.ndarray:
    hessian = _central_difference_hessian(function, point)
    try:
        covariance = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(hessian)
    if not np.all(np.isfinite(covariance)):
        raise ConvergenceError(
            "the observed information matrix is not invertible; the model is "
            "likely not identified"
        )
    return covariance


def _pseudo_r2(llf: float, ll0: float) -> float:
    if not np.isfinite(ll0) or ll0 == 0:
        return float("nan")
    return 1.0 - llf / ll0


class _LimitedBase:
    """Shared plumbing for the self-implemented censored/truncated likelihoods."""

    _stata_command = ""

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
        #: Estimation-sample covariates, retained for margins(over=...).
        self._model_data: pd.DataFrame | None = None

    # -------------------------------------------------------------- fitting
    def _fit_likelihood(
        self,
        *,
        maxiter: int,
        strict: bool,
        extra_names: Sequence[str] = ("/sigma",),
        scalars: Mapping[str, Any] | None = None,
    ) -> StataResults:
        """Maximise ``-self._loglike`` and assemble a Stata-style result."""

        n, k = self._X.shape
        if n <= k:
            raise ConsistencyError(
                f"{self._stata_command}: {n} observations but {k} regressors; the "
                "model is not identified"
            )

        beta0, *_ = np.linalg.lstsq(self._X, self._y, rcond=None)
        resid0 = self._y - self._X @ beta0
        scale0 = max(float(np.std(resid0)), 1e-3)
        start = np.concatenate([beta0, [np.log(scale0)]])
        if len(extra_names) != 1:
            raise UnsupportedError(
                f"{self._stata_command}: only a single scale parameter is "
                f"supported, got {extra_names}"
            )

        def objective(theta: np.ndarray) -> float:
            value = self._loglike(theta)
            return -value if np.isfinite(value) else 1e12

        result = optimize.minimize(
            objective, start, method="BFGS", options={"maxiter": maxiter}
        )
        if not result.success:
            # BFGS routinely stops on "precision loss" at the optimum; refine with
            # a derivative-free method as tobit does.
            refined = optimize.minimize(
                objective,
                np.asarray(result.x, dtype=float),
                method="Nelder-Mead",
                options={"maxiter": maxiter, "xatol": 1e-9, "fatol": 1e-9},
            )
            if refined.fun <= result.fun:
                result = refined

        converged = bool(result.success) or bool(np.isfinite(result.fun))
        if not result.success:
            message = (
                f"{self._stata_command}: optimiser did not converge ({result.message})"
            )
            if strict:
                raise ConvergenceError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)

        theta = np.asarray(result.x, dtype=float)
        beta = theta[:k]
        scale = float(np.exp(theta[k]))
        covariance = _covariance_from_hessian(theta, objective)

        llf = float(self._loglike(theta))
        ll0 = self._null_loglike(k)
        chi2 = 2.0 * (llf - ll0) if np.isfinite(ll0) else float("nan")
        pseudo = _pseudo_r2(llf, ll0)

        names = list(self._param_names) + list(extra_names)
        params = np.concatenate([beta, [scale]])

        e: dict[str, Any] = {
            "N": n,
            "df_m": float(k - 1),
            "rank": k + 1,
            "ll": llf,
            "ll_0": ll0,
            "chi2": chi2,
            "chi2type": "LR",
            "p": float(stats.chi2.sf(chi2, k - 1)) if k > 1 else float("nan"),
            "r2_p": pseudo,
            "sigma": scale,
            "converged": int(converged),
        }
        if scalars:
            e.update(dict(scalars))

        return StataResults(
            params=params,
            param_names=names,
            cov=covariance,
            resid=self._y - self._X @ beta,
            fittedvalues=self._X @ beta,
            nobs=n,
            df_resid=float(n - k),
            df_model=float(k - 1),
            dist="z",
            model_name=self._stata_command,
            cov_type="nonrobust",
            e=e,
            design_matrix=(self._X, list(self._param_names)),
            model_data=self._model_data,
        )

    def _loglike(self, theta: np.ndarray) -> float:
        raise NotImplementedError

    def _null_loglike(self, k: int) -> float:
        """Intercept-only likelihood, for the LR statistic and pseudo R-squared."""

        ones = np.ones((self._y.size, 1))
        original_X, original_names = self._X, self._param_names
        try:
            self._X = ones
            self._param_names = ["_cons"]
            beta0, *_ = np.linalg.lstsq(ones, self._y, rcond=None)
            scale0 = max(float(np.std(self._y - ones @ beta0)), 1e-3)
            start = np.array([float(beta0[0]), np.log(scale0)])

            def objective(theta: np.ndarray) -> float:
                value = self._loglike(theta)
                return -value if np.isfinite(value) else 1e12

            result = optimize.minimize(
                objective, start, method="BFGS", options={"maxiter": 1000}
            )
            return float(self._loglike(np.asarray(result.x, dtype=float)))
        finally:
            self._X, self._param_names = original_X, original_names


class Truncreg(_LimitedBase):
    """Truncated regression (Stata ``truncreg``).

    Observations outside ``(lower, upper)`` are absent from the data, so the
    likelihood is divided by the probability of falling inside the interval.
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
        super().__init__(endog, exog, param_names=param_names)
        if lower is None and upper is None:
            raise ConsistencyError(
                "truncreg: at least one truncation limit is required; with no "
                "truncation use OLS (Stata: regress)"
            )
        if lower is not None and upper is not None and not lower < upper:
            raise ConsistencyError(
                f"truncreg: lower limit ({lower}) must be below upper limit ({upper})"
            )
        self._lower = lower
        self._upper = upper

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
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if subset is not None:
            data = data.loc[subset]
        data = data.reset_index(drop=True)
        X, names, kept_index = build_design(formula, data)
        lhs = formula.split("~", 1)[0].strip()
        if lhs not in data.columns:
            raise ConsistencyError(
                f"truncreg: the left-hand side must be a plain column name, got {lhs!r}"
            )
        y = data.loc[kept_index, lhs].to_numpy(dtype=float)
        model = cls(y, X, lower=lower, upper=upper, param_names=names)
        model._model_data = data.loc[kept_index].reset_index(drop=True)
        return model

    def _loglike(self, theta: np.ndarray) -> float:
        k = self._X.shape[1]
        beta = theta[:k]
        sigma = float(np.exp(theta[k]))
        xb = self._X @ beta
        z_lower = (self._lower - xb) / sigma if self._lower is not None else None
        z_upper = (self._upper - xb) / sigma if self._upper is not None else None

        # Probability of the observed interval, computed with sf() in the upper
        # tail so it does not cancel to zero.
        if z_lower is None and z_upper is None:
            mass = 1.0
        elif z_lower is None:
            mass = stats.norm.cdf(z_upper)
        elif z_upper is None:
            mass = stats.norm.sf(z_lower)
        else:
            mass = stats.norm.cdf(z_upper) - stats.norm.cdf(z_lower)
        if np.any(mass <= 0):
            return float("-inf")

        residual = (self._y - xb) / sigma
        return float(
            np.sum(
                -np.log(sigma)
                - 0.5 * np.log(2 * np.pi)
                - 0.5 * residual**2
                - np.log(mass)
            )
        )

    def fit(
        self,
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
        *,
        maxiter: int = 1000,
        strict: bool = False,
    ) -> StataResults:
        if cov_type not in (None, "nonrobust"):
            raise ValueError(
                f"truncreg: only cov_type='nonrobust' is implemented, got {cov_type!r}"
            )
        return self._fit_likelihood(
            maxiter=maxiter,
            strict=strict,
            scalars={
                "lower": self._lower,
                "upper": self._upper,
                "n_truncated": 0,
            },
        )


class Intreg(_LimitedBase):
    """Interval regression (Stata ``intreg``).

    ``y`` is known only to lie in ``[lower, upper]``; ``lower == upper`` marks an
    exactly observed value, ``-inf``/``+inf`` a censored one.
    """

    _stata_command = "intreg"

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
                f"intreg: lower and upper must have the same length ({lo.size} vs {hi.size})"
            )
        if np.any(lo > hi):
            raise ConsistencyError(
                "intreg: every lower bound must be <= its upper bound"
            )
        self._lower = lo
        self._upper = hi
        # The base class wants a response; use the midpoint only as an optimiser
        # starting value, never as data.
        midpoint = np.where(
            np.isfinite(lo) & np.isfinite(hi),
            (lo + hi) / 2.0,
            np.where(np.isfinite(lo), lo + 1.0, hi - 1.0),
        )
        super().__init__(midpoint, exog, param_names=param_names)

    @classmethod
    def from_formula(
        cls, formula: str, data: pd.DataFrame, *, subset: Any = None
    ) -> "Intreg":
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if subset is not None:
            data = data.loc[subset]
        data = data.reset_index(drop=True)

        lhs = formula.split("~", 1)[0].strip()
        pieces = lhs.split()
        if len(pieces) != 2:
            raise ConsistencyError(
                "intreg: the left-hand side must name two columns, the lower and "
                f"upper bounds (e.g. 'lo hi ~ x'), got {lhs!r}"
            )
        lower_column, upper_column = pieces
        for column in (lower_column, upper_column):
            if column not in data.columns:
                raise ConsistencyError(
                    f"intreg: column {column!r} not found in the data"
                )

        # Build the design matrix from the right-hand side only.
        rhs_formula = f"{upper_column} ~ " + formula.split("~", 1)[1].strip()
        X, names, kept_index = build_design(rhs_formula, data)
        lower = data.loc[kept_index, lower_column].to_numpy(dtype=float)
        upper = data.loc[kept_index, upper_column].to_numpy(dtype=float)

        model = cls(lower, upper, X, param_names=names)
        model._model_data = data.loc[kept_index].reset_index(drop=True)
        return model

    def _loglike(self, theta: np.ndarray) -> float:
        k = self._X.shape[1]
        beta = theta[:k]
        sigma = float(np.exp(theta[k]))
        xb = self._X @ beta
        z_lower = (self._lower - xb) / sigma
        z_upper = (self._upper - xb) / sigma

        upper_cdf = np.where(np.isfinite(z_upper), stats.norm.cdf(z_upper), 1.0)
        lower_cdf = np.where(np.isfinite(z_lower), stats.norm.cdf(z_lower), 0.0)
        mass = upper_cdf - lower_cdf
        if np.any(mass <= 0):
            return float("-inf")
        return float(np.sum(np.log(mass)))

    def fit(
        self,
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
        *,
        maxiter: int = 1000,
        strict: bool = False,
    ) -> StataResults:
        if cov_type not in (None, "nonrobust"):
            raise ValueError(
                f"intreg: only cov_type='nonrobust' is implemented, got {cov_type!r}"
            )
        result = self._fit_likelihood(maxiter=maxiter, strict=strict)
        # Stata reports the log of the scale, not sigma itself.
        names = list(result.param_names)
        position = names.index("/sigma")
        params = result.params.to_numpy().copy()
        covariance = np.asarray(result.cov_params(), dtype=float).copy()
        sigma = float(params[position])
        scale = 1.0 / sigma if sigma != 0 else float("nan")
        params[position] = float(np.log(sigma)) if sigma > 0 else float("nan")
        covariance[position, :] *= scale
        covariance[:, position] *= scale
        names[position] = "/lnsigma"

        e = dict(result.e)
        e["sigma"] = sigma
        e["lnsigma"] = params[position]

        return StataResults(
            params=params,
            param_names=names,
            cov=covariance,
            resid=result.resid,
            fittedvalues=result.fittedvalues,
            nobs=result.nobs,
            df_resid=result.df_resid,
            df_model=result.df_model,
            dist="z",
            model_name=self._stata_command,
            cov_type="nonrobust",
            e=e,
            design_matrix=result.design_matrix,
            model_data=result.model_data,
        )


class Heckman:
    """Heckman sample-selection model (Stata ``heckman``), two-step estimator.

    Step 1 fits a probit for participation. Step 2 regresses the outcome on the
    covariates plus the inverse Mills ratio ``lambda = phi(z g) / Phi(z g)`` on the
    selected observations.

    **Standard errors**: the second-stage covariance reported here is the OLS
    covariance of the augmented regression, i.e. it does *not* include the
    Heckman correction for the estimated ``lambda``. Stata's two-step standard
    errors do. This is stated in ``e(se_corrected) = 0`` so a reader cannot
    mistake one for the other; `method="ml"` is not implemented.
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

    @classmethod
    def from_formula(
        cls, formula: str, data: pd.DataFrame, *, subset: Any = None
    ) -> "Heckman":
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if subset is not None:
            data = data.loc[subset]
        data = data.reset_index(drop=True)

        if "select(" not in formula:
            raise ConsistencyError(
                "heckman: the formula must contain a selection equation, as in "
                "'y ~ x + select(selected ~ z)'"
            )
        main, selection_part = formula.split("select(", 1)
        main = main.rstrip()
        if not main.endswith("+"):
            raise ConsistencyError(
                "heckman: expected '+' before select(...), as in 'y ~ x + select(s ~ z)'"
            )
        main = main[:-1].strip()
        selection_part = selection_part.rstrip()
        if not selection_part.endswith(")"):
            raise ConsistencyError("heckman: select(...) is missing its closing ')'")
        selection_formula = selection_part[:-1].strip()

        import patsy

        lhs_main = main.split("~", 1)[0].strip()
        rhs_main = main.split("~", 1)[1].strip()
        lhs_selection = selection_formula.split("~", 1)[0].strip()
        rhs_selection = selection_formula.split("~", 1)[1].strip()

        for column in (lhs_main, lhs_selection):
            if column not in data.columns:
                raise ConsistencyError(
                    f"heckman: column {column!r} not found in the data"
                )

        # The selection equation uses the FULL sample -- that is the whole point
        # of the estimator: the probit must see the non-selected rows too. Only
        # the covariates and the selector may not be missing; a missing outcome
        # is exactly what marks a non-selected observation.
        covariate_columns: list[str] = []
        for term in (rhs_main, rhs_selection):
            for token in term.replace("+", " ").split():
                if token in data.columns:
                    covariate_columns.append(token)
        required = list(dict.fromkeys([lhs_selection, *covariate_columns]))
        aligned = data.dropna(subset=required).reset_index(drop=True)

        x_frame = patsy.dmatrix(rhs_main, aligned, return_type="dataframe")
        z_frame = patsy.dmatrix(rhs_selection, aligned, return_type="dataframe")

        outcome = aligned[lhs_main].to_numpy(dtype=float)
        selection = aligned[lhs_selection].to_numpy(dtype=float)
        selected = selection > 0.5

        if selected.all():
            raise ConsistencyError(
                "heckman: the selection indicator is 1 for every observation, so "
                "the selection equation cannot be estimated"
            )
        if not selected.any():
            raise ConsistencyError(
                "heckman: the selection indicator is 0 for every observation, so "
                "there is no outcome to model"
            )

        return cls(
            outcome,
            np.asarray(x_frame, dtype=float),
            selection,
            np.asarray(z_frame, dtype=float),
            outcome_names=list(x_frame.columns),
            selection_names=list(z_frame.columns),
            selected=selected,
            model_data=aligned,
        )

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
        normalized = str(method).strip().lower()
        if normalized not in ("twostep", "two-step", "2sls"):
            raise UnsupportedError(
                f"heckman: only method='twostep' is implemented; {method!r} is not. "
                "The maximum-likelihood variant needs a joint optimisation over "
                "beta, gamma and rho that has not been implemented or verified "
                "against Stata."
            )
        if cov_type not in (None, "nonrobust"):
            raise ValueError(
                f"heckman: only cov_type='nonrobust' is implemented, got {cov_type!r}"
            )

        # Step 1: probit for participation, via statsmodels.
        import statsmodels.api as sm

        probit = sm.Probit(self._s, self._Z).fit(
            method="newton", disp=0, maxiter=maxiter
        )
        gamma = np.asarray(probit.params, dtype=float)
        z_index = self._Z @ gamma
        # Inverse Mills ratio for the selected observations.
        mills = stats.norm.pdf(z_index) / np.maximum(stats.norm.cdf(z_index), 1e-300)

        # Step 2: OLS of the outcome on X and the IMR, selected rows only.
        mask = self._selected
        y_sel = self._y[mask]
        X_sel = self._X[mask]
        augmented = np.column_stack([X_sel, mills[mask]])
        beta, *_ = np.linalg.lstsq(augmented, y_sel, rcond=None)
        resid = y_sel - augmented @ beta

        dof = max(len(y_sel) - augmented.shape[1], 1)
        sigma2 = float(resid @ resid) / dof
        bread = np.linalg.pinv(augmented.T @ augmented)
        covariance = sigma2 * bread * (dof / max(dof - 1, 1))

        # `beta` already contains the coefficient on the inverse Mills ratio
        # appended as an extra column, so the names must include it once here and
        # must not repeat it at the end.
        # Both equations carry an intercept, so the names are prefixed the way
        # Stata labels its two blocks (y: and selected:) to stay unique.
        names = (
            [f"y:{name}" for name in self._outcome_names]
            + ["lambda"]
            + [f"select:{name}" for name in self._selection_names]
        )
        params = np.concatenate([beta, gamma])
        augmented_covariance = np.zeros(
            (params.size, params.size), dtype=float
        )
        # Only the second-stage block is estimated here; the selection block stays
        # zero, which is why e(se_corrected) is 0.
        augmented_covariance[: beta.size, : beta.size] = covariance

        llf = float(
            -0.5 * len(y_sel) * (np.log(2 * np.pi) + np.log(sigma2) + 1.0)
        ) if sigma2 > 0 else float("nan")
        # rho is the correlation between the two equations' errors. The
        # second-stage residual and the probit index are the closest available
        # proxies; note this is a *plug-in* estimate, not the MLE of rho, which is
        # why the two-step estimator's rho can differ from Stata's ML rho.
        rho = (
            float(np.corrcoef(resid, -z_index[mask])[0, 1])
            if resid.size > 2
            else float("nan")
        )

        e: dict[str, Any] = {
            "N": int(self._y.size),
            "N_selected": int(mask.sum()),
            "df_m": float(self._X.shape[1] - 1),
            "rank": int(params.size),
            "ll": llf,
            "chi2": float("nan"),
            "chi2type": "Wald",
            "p": float("nan"),
            "r2_p": float("nan"),
            "sigma": float(np.sqrt(sigma2)),
            "rho": rho,
            "lambda": float(mills[mask].mean()),
            "estimator": "twostep",
            "se_corrected": 0,
        }

        return StataResults(
            params=params,
            param_names=names,
            cov=augmented_covariance,
            resid=resid,
            fittedvalues=augmented @ beta,
            nobs=int(y_sel.size),
            df_resid=float(dof),
            df_model=float(self._X.shape[1] - 1),
            dist="z",
            model_name=self._stata_command,
            cov_type="nonrobust",
            e=e,
            design_matrix=(X_sel, list(self._outcome_names)),
            model_data=self._model_data,
        )


class Etregress:
    """Endogenous treatment regression (Stata ``etregress``), control function.

    Like :class:`Heckman`, only the two-step control-function estimator is
    implemented; the MLE variant raises :class:`UnsupportedError`.
    """

    _stata_command = "etregress"

    def __init__(self, model: Heckman) -> None:
        self._model = model

    @classmethod
    def from_formula(
        cls, formula: str, data: pd.DataFrame, *, subset: Any = None
    ) -> "Etregress":
        if "treat(" not in formula:
            raise ConsistencyError(
                "etregress: the formula must contain a treatment equation, as in "
                "'y ~ x + treat(t ~ z)'"
            )
        transformed = formula.replace("treat(", "select(")
        return cls(Heckman.from_formula(transformed, data, subset=subset))

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
        normalized = str(method).strip().lower()
        if normalized not in ("control", "twostep", "two-step"):
            raise UnsupportedError(
                f"etregress: only method='control' is implemented; {method!r} is "
                "not. The ML variant needs a joint likelihood that has not been "
                "implemented or verified."
            )
        result = self._model.fit(
            method="twostep",
            cov_type=cov_type,
            cov_kwds=cov_kwds,
            ssc=ssc,
            maxiter=maxiter,
            strict=strict,
        )
        e = dict(result.e)
        e["estimator"] = "control"
        return StataResults(
            params=result.params,
            param_names=result.param_names,
            cov=result.cov_params(),
            resid=result.resid,
            fittedvalues=result.fittedvalues,
            nobs=result.nobs,
            df_resid=result.df_resid,
            df_model=result.df_model,
            dist="z",
            model_name=self._stata_command,
            cov_type="nonrobust",
            e=e,
            design_matrix=result.design_matrix,
            model_data=result.model_data,
        )
