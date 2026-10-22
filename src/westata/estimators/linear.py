"""Ordinary least squares, targeting Stata's ``regress``.

The estimation code only computes the point estimates and passes the residual
vector to :mod:`westata.vce`, which owns every small-sample correction.  This
keeps Stata conformance in one place instead of duplicating it per model.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from westata.data.weights import normalize_weights
from westata.errors import ConsistencyError, UnsupportedError, WeightError
from westata.formula.factor import build_design
from westata.protocols import SSC
from westata.results.base import StataResults
from westata.vce.dispatch import compute_cov


class OLS:
    """Ordinary least squares (Stata ``regress``).

    Parameters
    ----------
    endog:
        Dependent variable, length ``n``.
    exog:
        Design matrix, ``n x k``.  An intercept must be included explicitly,
        matching statsmodels semantics.
    weights:
        Not implemented in P0; passing it raises :class:`UnsupportedError`.
    weight_type:
        One of ``"aweight"``, ``"pweight"``, ``"fweight"``, ``"iweight"``.
        Required in P1 once weights are implemented.
    """

    def __init__(
        self,
        endog: Any,
        exog: Any,
        *,
        weights: Any = None,
        weight_type: str | None = None,
        param_names: Sequence[str] | None = None,
    ) -> None:
        self._y = np.asarray(endog, dtype=float).ravel()
        X = np.asarray(exog, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self._X = X
        self._param_names = (
            list(param_names) if param_names is not None else [f"x{i}" for i in range(X.shape[1])]
        )
        if len(self._param_names) != X.shape[1]:
            raise ValueError(
                f"param_names has {len(self._param_names)} entries but exog has "
                f"{X.shape[1]} columns"
            )
        if self._X.shape[0] != self._y.shape[0]:
            raise ValueError(
                f"endog has {self._y.shape[0]} rows but exog has {self._X.shape[0]} rows"
            )
        self._weight_spec = None
        if weights is not None:
            if weight_type is None:
                raise WeightError(
                    "weights were supplied without weight_type; specify one of "
                    "'fweight', 'aweight', 'pweight', 'iweight' (Stata never "
                    "guesses the weight type)"
                )
            spec = normalize_weights(weights, weight_type)
            if spec.weights.shape[0] != self._X.shape[0]:
                raise WeightError(
                    f"weights have {spec.weights.shape[0]} entries but the data "
                    f"have {self._X.shape[0]} rows"
                )
            self._weight_spec = spec
        elif weight_type is not None:
            raise WeightError("weight_type was supplied without weights")

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        weights: Any = None,
        weight_type: str | None = None,
        subset: Any = None,
    ) -> "OLS":
        """Build a model from a patsy-style formula, e.g. ``"y ~ x + C(g)"``."""

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if subset is not None:
            data = data.loc[subset]
        # Positional alignment: patsy's kept_index is only safe to use with
        # .loc when the index is unique, so normalise it to a RangeIndex.
        data = data.reset_index(drop=True)

        X, names, kept_index = build_design(formula, data)

        lhs = formula.split("~", 1)[0].strip()
        if lhs not in data.columns:
            raise UnsupportedError(
                "P0 only supports a plain column name on the left-hand side; "
                f"got {lhs!r}"
            )
        y = data.loc[kept_index, lhs].to_numpy(dtype=float)

        return cls(y, X, weights=weights, weight_type=weight_type, param_names=names)

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: SSC | None = None,
    ) -> StataResults:
        """Estimate the model and return a Stata-conformant result object.

        Parameters
        ----------
        cov_type:
            ``None`` selects the Stata default for this model: ``"nonrobust"``
            for unweighted/aweighted/fweighted/iweighted OLS, ``"HC1"`` when the
            model carries ``pweight`` (Stata: ``[pw=w]`` implies ``vce(robust)``).
            Passing ``"nonrobust"`` explicitly for a pweighted model raises
            :class:`ConsistencyError` because Stata cannot express that
            combination either.
        """

        X, y = self._X, self._y
        n, k = X.shape
        if n == 0:
            raise ConsistencyError(
                "regress: no observations remain after listwise deletion"
            )
        if n <= k:
            raise ConsistencyError(
                f"regress: {n} observations but {k} parameters; "
                "the model is not identified"
            )

        rank = int(np.linalg.matrix_rank(X))
        if rank < k:
            raise ConsistencyError(
                "regress: design matrix is rank deficient "
                f"(rank {rank} < {k} columns); drop collinear regressors"
            )

        spec = self._weight_spec
        if spec is None:
            w = np.ones(n, dtype=float)
            effective_n = float(n)
        else:
            w = spec.weights
            effective_n = float(spec.effective_n)

        if cov_type is None:
            effective_cov_type = "HC1" if (spec is not None and spec.implies_robust) else "nonrobust"
        else:
            effective_cov_type = cov_type
            if (
                spec is not None
                and spec.implies_robust
                and str(cov_type).strip().lower() == "nonrobust"
            ):
                raise ConsistencyError(
                    "pweight implies vce(robust): Stata's 'regress y x [pw=w]' is "
                    "identical to adding vce(robust), so a nonrobust covariance "
                    "cannot be requested for a pweighted model"
                )

        # Weighted least squares via the square-root transform keeps a single
        # well-conditioned least-squares solve for both weighted and unweighted
        # models and matches Stata's (X'WX)^-1 X'Wy exactly.
        sw = np.sqrt(w)
        Xw = X * sw[:, None]
        yw = y * sw
        beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        resid = y - X @ beta

        df_resid = effective_n - k
        if df_resid <= 0:
            raise ConsistencyError(
                f"regress: {effective_n:g} effective observations but {k} "
                "parameters; the model is not identified"
            )

        cov, df_t = compute_cov(
            Xw,
            resid * sw,
            cov_type=effective_cov_type,
            cov_kwds=cov_kwds,
            ssc=ssc,
            nobs=n,
            df_resid=df_resid,
        )

        rss = float((w * resid**2).sum())
        ybar = float((w * y).sum() / w.sum())
        tss = float((w * (y - ybar) ** 2).sum())
        r2 = 1.0 - rss / tss if tss > 0 else float("nan")
        r2_adj = (
            1.0 - (1.0 - r2) * (effective_n - 1) / df_resid
            if np.isfinite(r2)
            else float("nan")
        )
        sigma2 = rss / df_resid
        rmse = float(np.sqrt(sigma2))
        llf = float(
            -0.5 * effective_n * (np.log(2 * np.pi) + np.log(rss / effective_n) + 1.0)
        )
        has_intercept = any(name == "Intercept" for name in self._param_names)
        df_model = k - 1 if has_intercept else k
        fvalue = (
            (r2 / df_model) / ((1.0 - r2) / df_resid)
            if has_intercept and df_model > 0 and r2 < 1.0
            else float("nan")
        )
        # Stata e(aic)/e(bic) for regress use the Gaussian likelihood with the
        # MLE of sigma; statsmodels' OLS estimates sigma by sqrt(RSS/df_resid),
        # so report the Stata-consistent values here.
        aic = float(effective_n * np.log(rss / effective_n) + 2 * k)
        bic = float(effective_n * np.log(rss / effective_n) + k * np.log(effective_n))

        e: dict[str, Any] = {
            "N": effective_n,
            "df_m": df_model,
            "df_r": df_resid,
            "r2": r2,
            "r2_a": r2_adj,
            "rmse": rmse,
            "sigma": rmse,
            "F": fvalue,
            "ll": llf,
            "rank": k,
            "aic": aic,
            "bic": bic,
        }

        return StataResults(
            params=beta,
            param_names=self._param_names,
            cov=cov,
            resid=resid,
            fittedvalues=X @ beta,
            nobs=int(n),
            df_resid=df_t,
            df_model=df_model,
            dist="t",
            model_name="regress",
            cov_type=effective_cov_type,
            e=e,
        )
