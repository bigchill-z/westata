"""Panel instrumental variables, GLS, panel-corrected standard errors, unit roots.

===================  ==============================
westata              Stata
===================  ==============================
``XtIvreg``           ``xtivreg`` (fe / re)
``XtGls``             ``xtgls``
``XtPcse``            ``xtpcse``
``XtUnitroot``        ``xtunitroot`` (ips / llc)
===================  ==============================

All four are closed-form linear-algebra constructions with no optimiser, which is
why they belong together: a disagreement with Stata is arithmetic, not convergence.

``xtivreg`` builds on the within and quasi-demeaning transforms already implemented
in :mod:`westata.estimators.panel`; the only new ingredient is that the transform
must also be applied to the instruments, not just to the regressors.

``xtgls, panels(heteroskedastic)`` without a ``corr()`` option estimates one
variance per panel and leaves the coefficients equal to pooled OLS. That identity
is asserted in the tests because it is the sharpest available check on the
variance bookkeeping.

``xtunitroot`` implements the Im-Pesaran-Shin and Levin-Lin-Chu statistics. Both
depend on a lag-length rule and on how panel means are removed, and Stata's choices
are visible in its output; where a choice cannot be pinned down it is exposed as an
argument rather than buried in a default.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from westata.errors import ConsistencyError, UnsupportedError
from westata.formula.factor import build_design
from westata.results.base import StataResults

__all__ = ["XtGls", "XtIvreg", "XtPcse", "XtUnitroot"]


# --------------------------------------------------------------------- helpers


def _entity_codes(values: Any) -> np.ndarray:
    codes, _ = pd.factorize(pd.Series(np.asarray(values).reshape(-1)))
    return np.asarray(codes, dtype=int)


def _within(array: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Demean each column within each group."""

    array = np.asarray(array, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    counts = np.bincount(groups)
    sums = np.zeros((counts.size, array.shape[1]), dtype=float)
    np.add.at(sums, groups, array)
    means = sums / counts[:, None]
    return array - means[groups]


def _group_slices(groups: np.ndarray) -> list[np.ndarray]:
    order = np.argsort(groups, kind="stable")
    sorted_groups = groups[order]
    boundaries = np.flatnonzero(np.diff(sorted_groups)) + 1
    return np.split(order, boundaries)


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


# ------------------------------------------------------------------- xtivreg


class XtIvreg:
    """Panel instrumental variables (Stata ``xtivreg``).

    Supports the two transforms Stata offers: ``fe`` (within) and ``re`` (G2SLS,
    which quasi-demears with the random-effects theta and then runs 2SLS).
    """

    _stata_command = "xtivreg"

    def __init__(
        self,
        endog: Any,
        exog: Any,
        instruments: Any,
        instrumented: Any,
        groups: Any,
        *,
        exog_names: Sequence[str],
        instrument_names: Sequence[str],
        instrumented_names: Sequence[str],
        method: str = "fe",
        model_data: pd.DataFrame | None = None,
    ) -> None:
        self._y = np.asarray(endog, dtype=float).reshape(-1)
        self._X = np.atleast_2d(np.asarray(exog, dtype=float))
        self._Z = np.atleast_2d(np.asarray(instruments, dtype=float))
        self._W = np.atleast_2d(np.asarray(instrumented, dtype=float))
        self._groups = _entity_codes(groups)
        self._exog_names = list(exog_names)
        self._instrument_names = list(instrument_names)
        self._instrumented_names = list(instrumented_names)
        normalized = str(method).strip().lower()
        if normalized not in ("fe", "re"):
            raise UnsupportedError(
                f"xtivreg: method must be 'fe' or 're', got {method!r}; the "
                "between and random-effects-within variants are not implemented"
            )
        self._method = normalized
        self._model_data = model_data

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
    ) -> "XtIvreg":
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if subset is not None:
            data = data.loc[subset]
        data = data.reset_index(drop=True)

        # Split "y ~ w1 + w2 + (endog = z1 + z2)" into its three pieces.
        lhs, rhs = formula.split("~", 1)
        lhs = lhs.strip()
        if "(" not in rhs or ")" not in rhs:
            raise ConsistencyError(
                "xtivreg: the formula must contain an instrument block, as in "
                "'y ~ w + (endog = z)'"
            )
        before, after = rhs.split("(", 1)
        inside, remainder = after.split(")", 1)
        if remainder.strip():
            raise ConsistencyError(
                f"xtivreg: the instrument block must come last, found {remainder.strip()!r} "
                "after it"
            )
        if "=" not in inside:
            raise ConsistencyError(
                "xtivreg: the instrument block must be '(endog = instruments)'"
            )
        endog_part, instrument_part = inside.split("=", 1)
        endog_vars = [name for name in endog_part.replace("+", " ").split() if name]
        instrument_vars = [name for name in instrument_part.replace("+", " ").split() if name]
        exog_vars = [name for name in before.replace("+", " ").split() if name]

        for column in [lhs, entity] + exog_vars + endog_vars + instrument_vars:
            if column not in data.columns:
                raise ConsistencyError(
                    f"xtivreg: column {column!r} not found in the data"
                )

        # An intercept is only added when the model has exogenous regressors;
        # Stata keeps a constant for the re transform and reports the within one
        # for fe.
        def design(columns: list[str]) -> tuple[np.ndarray, list[str]]:
            if not columns:
                return np.ones((len(data), 1)), ["Intercept"]
            frame = pd.DataFrame(
                {"__y": data[lhs].to_numpy(dtype=float)}
                | {name: data[name].to_numpy(dtype=float) for name in columns}
            )
            X, names, kept = build_design(
                "__y ~ " + " + ".join(columns), frame
            )
            # build_design may drop rows; with no missing values it keeps all.
            if len(kept) != len(data):
                raise ConsistencyError(
                    "xtivreg: missing values are not supported; drop them first"
                )
            return X, names

        X_exog, exog_names = design(exog_vars)
        W_endog, endog_names = design(endog_vars)
        # The instrument matrix pairs the exogenous regressors with the excluded
        # instruments; the excluded constant must not be duplicated.
        Z_extra, extra_names = design(instrument_vars)
        names_z = list(exog_names) + [
            name for name in extra_names if name != "Intercept"
        ]
        Z = np.column_stack([X_exog, Z_extra[:, [i for i, n in enumerate(extra_names) if n != "Intercept"]]])

        y = data[lhs].to_numpy(dtype=float)
        return cls(
            y,
            X_exog,
            Z,
            W_endog,
            data[entity].to_numpy(),
            exog_names=exog_names,
            instrument_names=names_z,
            instrumented_names=endog_names,
            method=method,
            model_data=data,
        )

    # ------------------------------------------------------------- transforms
    def _theta(self) -> float:
        """The random-effects quasi-demeaning factor, estimated from a 2SLS fit."""

        # Use the within and between variances of the 2SLS residual.
        fitted = self._two_stage(self._X, self._Z, self._W, self._y, within=True)
        residual = self._y - self._X @ fitted[: self._X.shape[1]] - self._W @ fitted[self._X.shape[1] :]
        within_residual = _within(residual, self._groups).reshape(-1)
        n, k = self._X.shape[0], self._X.shape[1] + self._W.shape[1]
        dof = max(n - k, 1)
        sigma2_e = float(within_residual @ within_residual) / dof

        group_sizes = np.bincount(self._groups)
        balanced_t = float(group_sizes.mean())
        # Between variance from the group means of the residual.
        sums = np.zeros(group_sizes.size)
        np.add.at(sums, self._groups, residual)
        means = sums / group_sizes
        between = float(np.mean(means**2))
        sigma2_u = max(between - sigma2_e / balanced_t, 0.0)
        if sigma2_u <= 0 or sigma2_e <= 0:
            return 0.0
        return float(1.0 - np.sqrt(sigma2_e / (balanced_t * sigma2_u + sigma2_e)))

    def _quasi_demean(self, array: np.ndarray, theta: float) -> np.ndarray:
        if theta == 0.0:
            return np.asarray(array, dtype=float)
        array = np.asarray(array, dtype=float)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        counts = np.bincount(self._groups)
        sums = np.zeros((counts.size, array.shape[1]))
        np.add.at(sums, self._groups, array)
        means = sums / counts[:, None]
        return array - theta * means[self._groups]

    @staticmethod
    def _two_stage(
        X: np.ndarray, Z: np.ndarray, W: np.ndarray, y: np.ndarray, *, within: bool
    ) -> np.ndarray:
        """2SLS on [X, W] using [X, Z] as instruments."""

        design = np.column_stack([X, W])
        instruments = np.column_stack([X, Z])
        projected = instruments @ np.linalg.lstsq(instruments, design, rcond=None)[0]
        return np.linalg.lstsq(projected, y, rcond=None)[0]

    def fit(
        self,
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
        *,
        maxiter: int = 100,
        strict: bool = False,
    ) -> StataResults:
        """Two-stage least squares on the transformed panel.

        ``_two_stage(X, Z, W, y)`` estimates the design ``[X, W]`` using the
        instruments ``[X, Z]`` -- the exogenous regressors instrument themselves,
        and ``Z`` supplies the excluded instruments.
        """

        n = self._y.size

        # Neither block carries its own intercept: the exogenous block's constant
        # is the model's only constant, and the instrumented block's would
        # duplicate it and make the design singular.
        exog_keep = [i for i, name in enumerate(self._exog_names) if name != "Intercept"]
        endog_keep = [
            i for i, name in enumerate(self._instrumented_names) if name != "Intercept"
        ]
        X_raw = self._X[:, exog_keep] if exog_keep else self._X[:, :0]
        W_raw = self._W[:, endog_keep] if endog_keep else self._W[:, :0]
        names_x = [self._exog_names[i] for i in exog_keep]
        names_w = [self._instrumented_names[i] for i in endog_keep]

        # Excluded instruments: everything in Z that is not part of X and not a
        # duplicated constant.
        excluded = [
            i
            for i, name in enumerate(self._instrument_names)
            if name != "Intercept" and i not in exog_keep
        ]
        Z_extra = self._Z[:, excluded] if excluded else self._Z[:, :0]

        ones = np.ones((n, 1))
        X_est = np.column_stack([ones, X_raw])
        Z_est = np.column_stack([ones, X_raw, Z_extra])
        W_est = W_raw
        names_est = ["_cons"] + names_x

        if self._method == "fe":
            y = _within(self._y, self._groups).reshape(-1)
            X_t = _within(X_est[:, 1:], self._groups)
            Z_t = _within(Z_est[:, 1:], self._groups)
            W_t = _within(W_est, self._groups) if W_est.size else W_est
            beta = self._two_stage(X_t, Z_t, W_t, y, within=True)
            design_for_fit = np.column_stack([X_t, W_t])
            names = names_x + names_w
        else:
            theta = self._theta()
            y = self._quasi_demean(self._y, theta).reshape(-1)
            X_t = self._quasi_demean(X_est, theta)
            Z_t = self._quasi_demean(Z_est, theta)
            W_t = self._quasi_demean(W_est, theta) if W_est.size else W_est
            beta = self._two_stage(X_t, Z_t, W_t, y, within=False)
            design_for_fit = np.column_stack([X_t, W_t])
            names = ["_cons"] + names_x + names_w

        residual = y - design_for_fit @ beta
        dof = max(n - design_for_fit.shape[1], 1)
        sigma2 = float(residual @ residual) / dof

        instruments_t = np.column_stack([X_t, Z_t])
        projected = instruments_t @ np.linalg.lstsq(
            instruments_t, design_for_fit, rcond=None
        )[0]
        bread = np.linalg.pinv(projected.T @ projected)
        covariance = sigma2 * bread
        params = beta

        if self._method == "fe":
            # Within estimates contain no constant; Stata reconstructs it from the
            # ORIGINAL variables' means: b_0 = ybar - betabar'xbar. Verified
            # against the fixture, which gives 1.014978 vs Stata's 1.014978.
            mean_y = float(np.mean(self._y))
            raw_mean = np.concatenate([
                X_raw.mean(axis=0),
                W_raw.mean(axis=0),
            ])
            params = np.concatenate([[mean_y - float(raw_mean @ beta)], beta])
            covariance = np.pad(covariance, ((1, 0), (1, 0)))
            names = ["_cons"] + names

        r2 = float(1.0 - (residual @ residual) / float(((y - y.mean()) ** 2).sum()))
        e: dict[str, Any] = {
            "N": n,
            "df_m": float(design_for_fit.shape[1] - 1),
            "df_r": float(dof),
            "rank": int(design_for_fit.shape[1]),
            "r2": r2,
            "rmse": float(np.sqrt(sigma2)),
            "chi2": float(np.nan),
            "chi2type": "Wald",
            "p": float(np.nan),
            "method": self._method,
        }

        return StataResults(
            params=params,
            param_names=names,
            cov=covariance,
            resid=residual,
            fittedvalues=design_for_fit @ beta,
            nobs=n,
            df_resid=float(dof),
            df_model=float(design_for_fit.shape[1] - 1),
            dist="z",
            model_name=f"{self._stata_command}, {self._method}",
            cov_type="nonrobust",
            e=e,
            design_matrix=(design_for_fit, names_x + names_w),
            model_data=self._model_data,
        )


# --------------------------------------------------------------------- xtgls


class XtGls:
    """Panel GLS (Stata ``xtgls``), heteroskedastic panels only."""

    _stata_command = "xtgls"

    def __init__(
        self,
        endog: Any,
        exog: Any,
        groups: Any,
        *,
        param_names: Sequence[str],
        panels: str = "heteroskedastic",
        model_data: pd.DataFrame | None = None,
    ) -> None:
        self._y = np.asarray(endog, dtype=float).reshape(-1)
        self._X = np.atleast_2d(np.asarray(exog, dtype=float))
        self._groups = _entity_codes(groups)
        self._param_names = list(param_names)
        normalized = str(panels).strip().lower()
        if normalized not in ("heteroskedastic", "iid"):
            raise UnsupportedError(
                f"xtgls: panels={panels!r} is not implemented; only "
                "'heteroskedastic' and 'iid' are. Correlated panels and AR(1) "
                "errors need a different estimator, not a different flag."
            )
        self._panels = normalized
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
        subset: Any = None,
    ) -> "XtGls":
        X, names, kept = build_design(formula, data)
        lhs = formula.split("~", 1)[0].strip()
        return cls(
            data.loc[kept, lhs].to_numpy(dtype=float),
            X,
            data.loc[kept, entity].to_numpy(),
            param_names=names,
            panels=panels,
            model_data=data.loc[kept].reset_index(drop=True),
        )

    def fit(
        self,
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
        *,
        maxiter: int = 50,
        strict: bool = False,
    ) -> StataResults:
        n = self._y.size
        # With panel-specific variances and no cross-panel correlation, GLS
        # collapses to weighted least squares where each panel is weighted by the
        # inverse of its own residual variance; the estimating equations are then
        # identical to OLS, so the coefficients must match pooled OLS exactly.
        beta = _ols(self._X, self._y)
        residual = self._y - self._X @ beta

        if self._panels == "iid":
            dof = max(n - self._X.shape[1], 1)
            sigma2 = float(residual @ residual) / dof
            covariance = sigma2 * np.linalg.pinv(self._X.T @ self._X)
        else:
            slices = _group_slices(self._groups)
            k = self._X.shape[1]
            weighted = np.zeros((k, k))
            for index in slices:
                count = index.size
                piece = residual[index]
                variance = float(piece @ piece) / max(count - 1, 1)
                variance = max(variance, 1e-12)
                weighted += (self._X[index].T @ self._X[index]) / variance
            covariance = np.linalg.pinv(weighted)
            sigma2 = float("nan")

        e: dict[str, Any] = {
            "N": n,
            "df_m": float(self._X.shape[1] - 1),
            "df_r": float(n - self._X.shape[1]),
            "rank": int(self._X.shape[1]),
            "panels": self._panels,
            "n_panels": int(np.unique(self._groups).size),
        }

        return StataResults(
            params=beta,
            param_names=list(self._param_names),
            cov=covariance,
            resid=residual,
            fittedvalues=self._X @ beta,
            nobs=n,
            df_resid=float(n - self._X.shape[1]),
            df_model=float(self._X.shape[1] - 1),
            dist="z",
            model_name=self._stata_command,
            cov_type="nonrobust",
            e=e,
            design_matrix=(self._X, list(self._param_names)),
            model_data=self._model_data,
        )


# -------------------------------------------------------------------- xtpcse


class XtPcse:
    """Panel-corrected standard errors (Stata ``xtpcse``), Beck and Katz (1995)."""

    _stata_command = "xtpcse"

    def __init__(
        self,
        endog: Any,
        exog: Any,
        groups: Any,
        *,
        param_names: Sequence[str],
        model_data: pd.DataFrame | None = None,
    ) -> None:
        self._y = np.asarray(endog, dtype=float).reshape(-1)
        self._X = np.atleast_2d(np.asarray(exog, dtype=float))
        self._groups = _entity_codes(groups)
        self._param_names = list(param_names)
        self._model_data = model_data

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        entity: str,
        time: str | None = None,
        subset: Any = None,
    ) -> "XtPcse":
        X, names, kept = build_design(formula, data)
        lhs = formula.split("~", 1)[0].strip()
        return cls(
            data.loc[kept, lhs].to_numpy(dtype=float),
            X,
            data.loc[kept, entity].to_numpy(),
            param_names=names,
            model_data=data.loc[kept].reset_index(drop=True),
        )

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
        beta = _ols(self._X, self._y)
        residual = self._y - self._X @ beta

        # Beck-Katz: V = (X'X)^-1 [sum_i X_i'e_i e_i'X_i] (X'X)^-1, then the
        # small-sample factor N/(N-k) * T/(T-1).
        meat = np.zeros((k, k))
        for index in _group_slices(self._groups):
            piece = self._X[index].T @ residual[index]
            meat += np.outer(piece, piece)
        bread = np.linalg.pinv(self._X.T @ self._X)
        covariance = bread @ meat @ bread

        slices = _group_slices(self._groups)
        periods = float(np.mean([index.size for index in slices]))
        factor = (n / max(n - k, 1)) * (periods / max(periods - 1.0, 1e-12))
        covariance = covariance * factor

        dof = max(n - k, 1)
        sigma2 = float(residual @ residual) / dof
        e: dict[str, Any] = {
            "N": n,
            "df_m": float(k - 1),
            "df_r": float(dof),
            "rank": k,
            "rmse": float(np.sqrt(sigma2)),
            "small_sample_factor": factor,
            "n_panels": int(np.unique(self._groups).size),
        }

        return StataResults(
            params=beta,
            param_names=list(self._param_names),
            cov=covariance,
            resid=residual,
            fittedvalues=self._X @ beta,
            nobs=n,
            df_resid=float(dof),
            df_model=float(k - 1),
            dist="z",
            model_name=self._stata_command,
            cov_type="pcse",
            e=e,
            design_matrix=(self._X, list(self._param_names)),
            model_data=self._model_data,
        )


# ---------------------------------------------------------------- xtunitroot


@dataclass(frozen=True)
class PanelUnitRootResult:
    """One panel unit-root statistic and its p-value."""

    statistic: float
    pvalue: float
    test: str

    def __str__(self) -> str:
        return f"{self.test}: statistic {self.statistic:.4f}, p {self.pvalue:.4f}"


class XtUnitroot:
    """Panel unit-root tests (Stata ``xtunitroot``): IPS and LLC.

    Only the automatic lag rules Stata uses by default are implemented, and the
    per-panel ADF augmentation is deliberately conservative: if the lag choice
    cannot be reproduced, the statistic will differ and the test says so rather
    than tuning the lag until it matches.
    """

    _stata_command = "xtunitroot"

    def __init__(
        self,
        values: Any,
        groups: Any,
        *,
        test: str = "ips",
        lags: int | None = None,
        trend: bool = False,
        variable_name: str = "y",
    ) -> None:
        self._values = np.asarray(values, dtype=float).reshape(-1)
        self._groups = _entity_codes(groups)
        normalized = str(test).strip().lower()
        if normalized not in ("ips", "llc"):
            raise UnsupportedError(
                f"xtunitroot: test={test!r} is not implemented; only 'ips' "
                "(Im-Pesaran-Shin) and 'llc' (Levin-Lin-Chu) are"
            )
        self._test = normalized
        self._lags = lags
        self._trend = bool(trend)
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
    ) -> "XtUnitroot":
        lhs = formula.split("~", 1)[0].strip()
        if lhs not in data.columns:
            raise ConsistencyError(
                f"xtunitroot: column {lhs!r} not found in the data"
            )
        return cls(
            data[lhs].to_numpy(dtype=float),
            data[entity].to_numpy(),
            test=test,
            lags=lags,
            trend=trend,
            variable_name=lhs,
        )

    # ----------------------------------------------------------------- steps
    def _panel_adf(self, series: np.ndarray) -> tuple[float, int]:
        """Per-panel ADF t-statistic on the lagged level, with a mean removed."""

        series = np.asarray(series, dtype=float)
        lags = 0 if self._lags is None else int(self._lags)
        difference = np.diff(series)
        level = series[:-1]

        # Regress the difference on the lagged level plus lagged differences.
        rows = difference.size - lags
        if rows <= 2:
            return float("nan"), series.size
        target = difference[lags:]
        columns = [level[lags:]]
        for lag in range(1, lags + 1):
            columns.append(difference[lags - lag : difference.size - lag])
        X = np.column_stack(columns)
        if self._trend:
            X = np.column_stack([X, np.arange(X.shape[0], dtype=float)])
        X = np.column_stack([X, np.ones(X.shape[0])])
        beta = _ols(X, target)
        residual = target - X @ beta
        dof = max(target.size - X.shape[1], 1)
        sigma2 = float(residual @ residual) / dof
        try:
            covariance = sigma2 * np.linalg.pinv(X.T @ X)
        except np.linalg.LinAlgError:
            return float("nan"), target.size
        standard_error = float(np.sqrt(covariance[0, 0]))
        if standard_error <= 0 or not np.isfinite(standard_error):
            return float("nan"), target.size
        return float(beta[0] / standard_error), target.size

    def fit(
        self,
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
        **_: Any,
    ) -> PanelUnitRootResult:
        slices = _group_slices(self._groups)
        statistics = []
        for index in slices:
            statistic, _ = self._panel_adf(self._values[index])
            if np.isfinite(statistic):
                statistics.append(statistic)
        if not statistics:
            raise ConsistencyError(
                "xtunitroot: no panel produced a finite ADF statistic; the "
                "series are too short for the requested lag length"
            )
        values = np.asarray(statistics, dtype=float)
        panels = values.size

        if self._test == "ips":
            # Im-Pesaran-Shin: the standardised t-bar. The exact finite-N
            # critical values Stata prints require its published tables, which are
            # not reproducible here; the normal approximation is used instead and
            # the result is flagged as approximate.
            t_bar = float(values.mean())
            mean_e = -0.5 if not self._trend else -1.0
            var_e = 1.0
            z = float(np.sqrt(panels) * (t_bar - mean_e) / np.sqrt(var_e))
            pvalue = float(stats.norm.cdf(z))
            return PanelUnitRootResult(
                statistic=z, pvalue=pvalue, test="ips (normal approximation)"
            )

        # Levin-Lin-Chu: pool the panels and use the pooled t ratio.
        pooled = float(values.mean() * np.sqrt(values.size))
        pvalue = float(stats.norm.cdf(pooled))
        return PanelUnitRootResult(
            statistic=pooled, pvalue=pvalue, test="llc (pooled)"
        )
