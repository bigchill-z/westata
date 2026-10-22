"""Panel estimators targeting Stata's ``xtreg`` family.

=========================  ==================
westata                    Stata
=========================  ==================
``PanelOLS``               ``xtreg, fe``
``RandomEffects``          ``xtreg, re``
``BetweenOLS``             ``xtreg, be``
``FirstDifferenceOLS``     ``xtreg, fd``
=========================  ==================

Implementation notes
--------------------
``PanelOLS.from_formula`` does **not** accept ``entity_effects`` or
``time_effects`` in linearmodels 7.1 -- those are ``__init__``-only arguments.
Passing them through ``fit()`` (as an earlier draft did) silently produces
pooled OLS.  This module therefore builds the model with
``PanelOLS(dependent, exog, entity_effects=..., time_effects=...)`` explicitly
after constructing the design matrices with ``patsy``/formulaic-free parsing.

Degrees of freedom
------------------
Stata's ``xtreg, fe`` reports coefficient t tests with

    df_r = N - K - n_entities + 1

where ``N`` is the number of observations, ``K`` the number of reported
regressors *including* the constant, and ``n_entities`` the number of groups.
The ``+1`` is the intercept that within-transformation absorbs.

Under clustered VCE the reference distribution uses ``G - 1`` while
``e(df_r)`` keeps ``N - K - n_entities + 1``, mirroring
:mod:`westata.estimators.linear`.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
import patsy

from westata.errors import ConsistencyError
from westata.results.base import StataResults

__all__ = [
    "PanelOLS",
    "RandomEffects",
    "BetweenOLS",
    "FirstDifferenceOLS",
]

_COV_TYPE_MAP: dict[str, str] = {
    "nonrobust": "unadjusted",
    "unadjusted": "unadjusted",
    "hc0": "robust",
    "hc1": "robust",
    "robust": "robust",
    "cluster": "clustered",
    "clustered": "clustered",
}


def _normalize_cov_type(cov_type: str | None) -> str:
    key = "nonrobust" if cov_type is None else str(cov_type).strip().lower()
    try:
        return _COV_TYPE_MAP[key]
    except KeyError:
        raise ValueError(
            f"Unknown cov_type {cov_type!r}; supported cov_type values are: "
            "nonrobust, HC0, HC1, cluster"
        ) from None


class PanelOLS:
    """Fixed-effects panel regression (Stata ``xtreg, fe``).

    Parameters
    ----------
    formula:
        ``"y ~ x1 + x2"``; the constant is implied (Stata's default).
    data:
        Long-format frame containing ``entity`` and ``time`` columns.
    entity, time:
        Column names identifying the panel dimensions.
    """

    _stata_command = "xtreg, fe"

    def __init__(
        self,
        formula: str,
        data: pd.DataFrame,
        *,
        entity: str,
        time: str,
    ) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        for column in (entity, time):
            if column not in data.columns:
                raise ConsistencyError(
                    f"{self._stata_command}: column {column!r} not found in data; "
                    f"available columns: {list(data.columns)}"
                )
        if "~" not in formula:
            raise ConsistencyError(
                f"{self._stata_command}: formula must contain '~', got {formula!r}"
            )

        self._formula = formula
        self._entity = entity
        self._time = time
        self._data = data.reset_index(drop=True)
        self._param_names: list[str] = []

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        entity: str,
        time: str,
    ) -> "PanelOLS":
        """Build the model from a formula and a long-format frame."""

        return cls(formula, data, entity=entity, time=time)

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        entity_effects: bool = True,
        time_effects: bool = False,
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
    ) -> StataResults:
        """Estimate and return a Stata-conformant result object."""

        backend_cov = _normalize_cov_type(cov_type)
        clusters = self._cluster_series(backend_cov, cov_kwds)

        y, X, names = self._design()
        nobs, k = X.shape
        panel_index = pd.MultiIndex.from_frame(
            self._data[[self._entity, self._time]], names=[self._entity, self._time]
        )

        model = self._build_model(
            y, X, panel_index, entity_effects, time_effects
        )
        ref = self._fit_backend(model, backend_cov, clusters)

        params = np.asarray(ref.params, dtype=float).reshape(-1)
        cov = np.asarray(ref.cov, dtype=float)
        resid = np.asarray(ref.resids, dtype=float).reshape(-1)
        n_entities = int(self._data[self._entity].nunique())
        # FD drops the constant, so the backend reports fewer parameters than
        # the patsy design matrix had columns.
        design_names = list(names)
        names = [str(name) for name in ref.params.index]

        e = self._stata_scalars(ref, params, resid, n_entities, backend_cov)
        df_resid = (
            float(clusters.nunique() - 1) if clusters is not None else float(e["df_r"])
        )

        # ``StataResults`` requires ``resid``/``fittedvalues`` to have one entry
        # per *regression* observation.  BetweenOLS regresses on entity means
        # (n_entities rows) while Stata still reports e(N) as the number of raw
        # observations, so the two counts are kept separate.
        nobs_regression = int(resid.size)
        if nobs_regression == int(np.asarray(y).size):
            full_X = np.asarray(X, dtype=float)
            if full_X.shape[1] != params.size:
                keep = [i for i, name in enumerate(design_names) if name in set(names)]
                full_X = full_X[:, keep]
            fitted = full_X @ params
            resid = np.asarray(y, dtype=float).reshape(-1) - fitted
        else:
            fitted = np.asarray(ref.fitted_values, dtype=float).reshape(-1)
            if fitted.size != nobs_regression:
                fitted = np.full(nobs_regression, np.nan)

        return StataResults(
            params=params,
            param_names=names,
            cov=cov,
            resid=resid,
            fittedvalues=fitted,
            nobs=nobs_regression,
            df_resid=df_resid,
            df_model=e["df_m"],
            dist="t",
            model_name=self._stata_command,
            cov_type=str(cov_type or "nonrobust"),
            e=e,
        )

    # -------------------------------------------------------------- internals
    def _design(self) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        """Build ``y`` and ``X`` with patsy, keeping the row order."""

        y_frame, x_frame = patsy.dmatrices(
            self._formula, self._data, return_type="dataframe"
        )
        names = [str(name) for name in x_frame.columns]
        return (
            pd.DataFrame(y_frame.iloc[:, 0]),
            pd.DataFrame(x_frame, columns=names),
            names,
        )

    def _build_model(
        self,
        y: pd.DataFrame,
        X: pd.DataFrame,
        index: pd.MultiIndex,
        entity_effects: bool,
        time_effects: bool,
    ) -> Any:
        from linearmodels.panel import PanelOLS as _PanelOLS

        if not entity_effects and not time_effects:
            raise ConsistencyError(
                f"{self._stata_command}: at least one effect is required; "
                "use westata.OLS for pooled OLS (Stata: regress)"
            )
        return _PanelOLS(
            y.set_axis(index),
            X.set_axis(index),
            entity_effects=entity_effects,
            time_effects=time_effects,
            drop_absorbed=False,
        )

    def _fit_backend(
        self, model: Any, backend_cov: str, clusters: pd.Series | None
    ) -> Any:
        kwargs: dict[str, Any] = {"cov_type": backend_cov}
        if backend_cov == "clustered":
            kwargs["clusters"] = clusters
        return model.fit(**kwargs)

    def _cluster_series(
        self, backend_cov: str, cov_kwds: Mapping[str, Any] | None
    ) -> pd.Series | None:
        if backend_cov != "clustered":
            return None
        kwds = {} if cov_kwds is None else dict(cov_kwds)
        clusters = kwds.get("clusters", kwds.get("groups"))
        if clusters is None:
            raise ValueError(
                "cov_type='cluster' requires cov_kwds['clusters']; pass the "
                "cluster labels as cov_kwds={'clusters': ...}"
            )
        index = pd.MultiIndex.from_frame(
            self._data[[self._entity, self._time]], names=[self._entity, self._time]
        )
        # linearmodels requires a 2-level MultiIndex on the cluster Series.
        return pd.Series(np.asarray(clusters).reshape(-1), index=index)

    def _stata_scalars(
        self,
        ref: Any,
        params: np.ndarray,
        resid: np.ndarray,
        n_entities: int,
        backend_cov: str,
    ) -> dict[str, Any]:
        nobs = int(ref.nobs)
        rank = int(len(params))
        # Stata xtreg, fe: df_r = N - K - n_entities + 1.
        df_r = nobs - rank - n_entities + 1
        rss = float(resid @ resid)
        f_stat = getattr(ref, "f_statistic", None)
        return {
            "N": nobs,
            "n_entities": n_entities,
            "rank": rank,
            "df_m": rank - 1,
            "df_r": df_r,
            "r2": float(getattr(ref, "rsquared", float("nan"))),
            "r2_a": float(getattr(ref, "rsquared_adj", float("nan"))),
            "rmse": float(np.sqrt(rss / df_r)) if df_r > 0 else float("nan"),
            "F": float(getattr(f_stat, "stat", float("nan"))),
            "ll": float("nan"),
            "rss": rss,
        }


class RandomEffects(PanelOLS):
    """Random-effects GLS (Stata ``xtreg, re``), Swamy-Arora by default."""

    _stata_command = "xtreg, re"

    def _build_model(
        self,
        y: pd.DataFrame,
        X: pd.DataFrame,
        index: pd.MultiIndex,
        entity_effects: bool,
        time_effects: bool,
    ) -> Any:
        from linearmodels.panel import RandomEffects as _RandomEffects

        return _RandomEffects(y.set_axis(index), X.set_axis(index))

    def _fit_backend(
        self, model: Any, backend_cov: str, clusters: pd.Series | None
    ) -> Any:
        kwargs: dict[str, Any] = {"cov_type": backend_cov}
        if backend_cov == "clustered":
            kwargs["clusters"] = clusters
        return model.fit(**kwargs)

    def _stata_scalars(
        self,
        ref: Any,
        params: np.ndarray,
        resid: np.ndarray,
        n_entities: int,
        backend_cov: str,
    ) -> dict[str, Any]:
        nobs = int(ref.nobs)
        rank = int(len(params))
        df_r = nobs - rank
        rss = float(resid @ resid)
        f_stat = getattr(ref, "f_statistic", None)
        return {
            "N": nobs,
            "n_entities": n_entities,
            "rank": rank,
            "df_m": rank - 1,
            "df_r": df_r,
            "r2": float(getattr(ref, "rsquared", float("nan"))),
            "r2_a": float(getattr(ref, "rsquared_adj", float("nan"))),
            "rmse": float(np.sqrt(rss / df_r)) if df_r > 0 else float("nan"),
            "F": float(getattr(f_stat, "stat", float("nan"))),
            "ll": float("nan"),
            "rss": rss,
        }


class BetweenOLS(RandomEffects):
    """Between estimator on group means (Stata ``xtreg, be``).

    The backend reports one residual per *entity*, not per observation, so the
    degrees of freedom and ``N`` are adjusted accordingly.
    """

    _stata_command = "xtreg, be"

    def _build_model(
        self,
        y: pd.DataFrame,
        X: pd.DataFrame,
        index: pd.MultiIndex,
        entity_effects: bool,
        time_effects: bool,
    ) -> Any:
        from linearmodels.panel import BetweenOLS as _BetweenOLS

        return _BetweenOLS(y.set_axis(index), X.set_axis(index))

    def _scalar_nobs(self, ref: Any) -> int:
        return int(ref.nobs)

    def _stata_scalars(
        self,
        ref: Any,
        params: np.ndarray,
        resid: np.ndarray,
        n_entities: int,
        backend_cov: str,
    ) -> dict[str, Any]:
        rank = int(len(params))
        df_r = int(ref.nobs) - rank
        rss = float(resid @ resid)
        f_stat = getattr(ref, "f_statistic", None)
        return {
            "N": int(self._data.shape[0]),
            "n_entities": n_entities,
            "rank": rank,
            "df_m": rank - 1,
            "df_r": df_r,
            "r2": float(getattr(ref, "rsquared", float("nan"))),
            "r2_a": float(getattr(ref, "rsquared_adj", float("nan"))),
            "rmse": float(np.sqrt(rss / df_r)) if df_r > 0 else float("nan"),
            "F": float(getattr(f_stat, "stat", float("nan"))),
            "ll": float("nan"),
            "rss": rss,
        }


class FirstDifferenceOLS(RandomEffects):
    """First-difference estimator (Stata ``xtreg, fd``).

    Stata's ``xtreg, fd`` differences consecutive periods and then fits without
    a constant, so the intercept column is dropped before handing the design
    matrix to linearmodels (which rejects constants outright).
    """

    _stata_command = "xtreg, fd"

    def _build_model(
        self,
        y: pd.DataFrame,
        X: pd.DataFrame,
        index: pd.MultiIndex,
        entity_effects: bool,
        time_effects: bool,
    ) -> Any:
        from linearmodels.panel import FirstDifferenceOLS as _FDOLS

        if "Intercept" in X.columns:
            X = X.drop(columns=["Intercept"])
        return _FDOLS(y.set_axis(index), X.set_axis(index))

    def _stata_scalars(
        self,
        ref: Any,
        params: np.ndarray,
        resid: np.ndarray,
        n_entities: int,
        backend_cov: str,
    ) -> dict[str, Any]:
        rank = int(len(params))
        df_r = int(ref.nobs) - rank
        rss = float(resid @ resid)
        f_stat = getattr(ref, "f_statistic", None)
        return {
            "N": int(ref.nobs),
            "n_entities": n_entities,
            "rank": rank,
            "df_m": rank,
            "df_r": df_r,
            "r2": float(getattr(ref, "rsquared", float("nan"))),
            "r2_a": float(getattr(ref, "rsquared_adj", float("nan"))),
            "rmse": float(np.sqrt(rss / df_r)) if df_r > 0 else float("nan"),
            "F": float(getattr(f_stat, "stat", float("nan"))),
            "ll": float("nan"),
            "rss": rss,
        }
