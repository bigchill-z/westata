"""Difference-in-differences (Stata ``didregress`` / ``xtdidregress``).

=================  ====================
westata             Stata
=================  ====================
``Didregress``       ``didregress``
``Xtdidregress``     ``xtdidregress``
=================  ====================

The model is the two-way fixed-effects specification

    y_it = alpha_i + lambda_t + beta * D_it + x_it' gamma + eps_it

and ``beta`` is the average treatment effect on the treated. Both Stata spellings
are the same estimator on a balanced panel, so ``Xtdidregress`` is a subclass
rather than a second implementation -- they cannot drift apart that way.

Absorbing versus dummying
-------------------------
Unit and time effects are absorbed rather than expanded into indicator columns.
Dummying 200 units would add 199 regressors and make ``X'X`` singular the moment
the treatment is collinear with them, which is exactly the failure mode the
treatment variable must avoid:

**the treatment must vary within a unit.** A time-invariant group indicator IS the
fixed effect. Stata rejects that with ``r(198) treatment variable omitted because
of collinearity``, and so does this module, with a message that says why.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy import stats

from westata.errors import ConsistencyError
from westata.formula.factor import build_design
from westata.results.base import StataResults

__all__ = ["Didregress", "Xtdidregress"]


def _codes(values: Any) -> np.ndarray:
    codes, _ = pd.factorize(pd.Series(np.asarray(values).reshape(-1)))
    return np.asarray(codes, dtype=int)


def _within(matrix: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Demean every column within each group."""

    matrix = np.atleast_2d(np.asarray(matrix, dtype=float))
    if matrix.shape[0] != groups.size:
        matrix = matrix.T
    counts = np.bincount(groups)
    sums = np.zeros((counts.size, matrix.shape[1]), dtype=float)
    np.add.at(sums, groups, matrix)
    means = sums / counts[:, None]
    return matrix - means[groups]


def _absorb(
    matrix: np.ndarray, unit: np.ndarray, time: np.ndarray, *, tolerance: float = 1e-10,
    maxiter: int = 200,
) -> np.ndarray:
    """Alternating projections: unit effects, then time effects, until stable.

    This is the same two-way within transform the HDFE estimators use. It avoids
    materialising one indicator per unit and per period.
    """

    residual = np.atleast_2d(np.asarray(matrix, dtype=float))
    if residual.shape[0] != unit.size:
        residual = residual.T
    for _ in range(maxiter):
        previous = residual.copy()
        residual = _within(residual, unit)
        residual = _within(residual, time)
        change = float(np.max(np.abs(residual - previous)))
        if change < tolerance:
            break
    return residual


class Didregress:
    """Two-way fixed-effects difference-in-differences (Stata ``didregress``)."""

    _stata_command = "didregress"

    def __init__(
        self,
        endog: Any,
        treatment: Any,
        covariates: np.ndarray,
        unit: np.ndarray,
        time: np.ndarray,
        *,
        treatment_name: str,
        covariate_names: list[str],
        model_data: pd.DataFrame | None = None,
    ) -> None:
        self._y = np.asarray(endog, dtype=float).reshape(-1)
        self._treatment = np.asarray(treatment, dtype=float).reshape(-1)
        self._X = np.atleast_2d(np.asarray(covariates, dtype=float))
        if self._X.shape[0] != self._y.size:
            self._X = self._X.T
        self._unit = _codes(unit)
        self._time = _codes(time)
        self._treatment_name = treatment_name
        self._covariate_names = list(covariate_names)
        self._model_data = model_data

    @classmethod
    def from_formula(
        cls,
        outcome: str,
        treatment: str,
        data: pd.DataFrame,
        *,
        group: str,
        time: str,
        covariates: list[str] | None = None,
        subset: Any = None,
    ) -> "Didregress":
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if subset is not None:
            data = data.loc[subset]
        data = data.reset_index(drop=True)

        for column in [outcome, treatment, group, time]:
            if column not in data.columns:
                raise ConsistencyError(
                    f"didregress: column {column!r} not found in the data"
                )
        if covariates is None:
            covariates = [
                column
                for column in data.columns
                if column not in (outcome, treatment, group, time)
            ]

        y = data[outcome].to_numpy(dtype=float)
        treatment_values = data[treatment].to_numpy(dtype=float)
        unit = data[group].to_numpy()
        period = data[time].to_numpy()

        # A treatment that never varies within a unit is the fixed effect itself,
        # so beta is not identified. Stata says "omitted because of collinearity";
        # saying so up front is far better than returning a coefficient of zero.
        unit_codes = _codes(unit)
        frame = pd.DataFrame({"treatment": treatment_values, "unit": unit_codes})
        variation = frame.groupby("unit")["treatment"].nunique()
        if int(variation.max()) < 2:
            raise ConsistencyError(
                "didregress: the treatment variable never changes within a unit. "
                "A time-invariant treatment is collinear with the unit fixed "
                "effects (Stata: 'treatment variable omitted because of "
                "collinearity', r(198)); the effect is not identified. Pass the "
                "group-by-period interaction instead of the group indicator."
            )

        if covariates:
            frame_design = pd.DataFrame(
                {"__y": y}
                | {name: data[name].to_numpy(dtype=float) for name in covariates}
            )
            X, names, kept = build_design(
                "__y ~ " + " + ".join(covariates), frame_design
            )
            if len(kept) != len(data):
                raise ConsistencyError(
                    "didregress: missing values are not supported; drop them first"
                )
            names = [name for name in names if name != "Intercept"]
            keep_columns = [
                position for position, name in enumerate(names)
                if name != "Intercept"
            ]
            X = np.asarray(X, dtype=float)[:, keep_columns]
        else:
            X = np.empty((y.size, 0), dtype=float)
            names = []

        return cls(
            y,
            treatment_values,
            X,
            unit,
            period,
            treatment_name=treatment,
            covariate_names=names,
            model_data=data,
        )

    def fit(
        self,
        cov_type: str | None = "cluster",
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
    ) -> StataResults:
        y = self._y
        n = y.size
        design = np.column_stack([self._treatment, self._X])

        # Absorb the two-way fixed effects from both the outcome and the design.
        y_within = _absorb(y.reshape(-1, 1), self._unit, self._time).reshape(-1)
        design_within = _absorb(design, self._unit, self._time)

        if np.max(np.abs(design_within[:, 0])) < 1e-12:
            raise ConsistencyError(
                "didregress: the treatment is fully absorbed by the fixed effects, "
                "so its coefficient is not identified"
            )

        beta = np.linalg.lstsq(design_within, y_within, rcond=None)[0]
        residual = y_within - design_within @ beta

        # Cluster on the unit, which is what group(unit) asks for.
        k = design_within.shape[1]
        clusters = np.unique(self._unit)
        meat = np.zeros((k, k), dtype=float)
        for cluster in clusters:
            mask = self._unit == cluster
            score = design_within[mask].T @ residual[mask]
            meat += np.outer(score, score)
        bread = np.linalg.pinv(design_within.T @ design_within)
        covariance = bread @ meat @ bread

        # Cluster-robust small-sample correction, matching Stata's default.
        g = clusters.size
        if g > 1:
            covariance *= (g / max(g - 1, 1)) * ((n - 1) / max(n - k, 1))

        names = [self._treatment_name] + list(self._covariate_names)
        dof = max(n - k, 1)
        r2 = float(
            1.0
            - (residual @ residual)
            / float(((y_within - y_within.mean()) ** 2).sum())
        )
        e: dict[str, Any] = {
            "N": n,
            "df_m": float(k - 1),
            "df_r": float(dof),
            "rank": k,
            "n_groups": int(g),
            "r2": r2,
            "estimand": "ATET",
            "absorbed": ["unit", "time"],
            "vce": "cluster",
        }

        return StataResults(
            params=beta,
            param_names=names,
            cov=covariance,
            resid=residual,
            fittedvalues=design_within @ beta,
            nobs=n,
            df_resid=float(dof),
            df_model=float(k - 1),
            dist="t",
            model_name=self._stata_command,
            cov_type="cluster",
            e=e,
            design_matrix=(design_within, names),
            model_data=self._model_data,
        )


class Xtdidregress(Didregress):
    """Panel spelling of the same estimator (Stata ``xtdidregress``).

    ``[R] didregress``: the two commands are the same estimator; only the data
    declaration differs. Subclassing keeps that equivalence structural.
    """

    _stata_command = "xtdidregress"
