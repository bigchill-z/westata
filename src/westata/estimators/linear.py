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
from westata.vce.cluster import encode_groups
from westata.vce.dispatch import compute_cov


def _wald_f(params: np.ndarray, cov: np.ndarray, tested: Sequence[int]) -> float:
    """Return ``b' V^-1 b / q`` over the ``tested`` coefficients.

    This is Stata's ``e(F)`` for ``regress``: the Wald statistic built from the
    covariance matrix that the model actually reports, divided by the number of
    tested coefficients (``[R] regress``, "Methods and formulas").  It reduces
    to the textbook ``(R2/q)/((1-R2)/(N-K))`` only when the covariance is the
    nonrobust one; under ``vce(robust)``/``vce(cluster)`` Stata reports the
    robust Wald version instead, so the classic formula would be the wrong
    number there.
    """

    idx = np.asarray(list(tested), dtype=int)
    b = np.asarray(params, dtype=float)[idx]
    V = np.asarray(cov, dtype=float)[np.ix_(idx, idx)]
    df_m = int(idx.size)
    if df_m == 0 or b.size == 0:
        return float("nan")
    try:
        return float(b @ np.linalg.solve(V, b) / df_m)
    except np.linalg.LinAlgError:
        return float("nan")


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
        Optional Stata-weighted observation weights.
    weight_type:
        One of ``"fweight"``, ``"aweight"``, ``"pweight"``, ``"iweight"``.
        Required whenever ``weights`` is given; Stata never guesses the type.
        See :mod:`westata.data.weights` and Stata ``[U] 20.24``.
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
        #: Raw (un-normalised) ``sum(w)``: Stata's ``e(sum_w)``.  ``aweight`` and
        #: ``pweight`` are normalised to sum to ``N`` for estimation, so the
        #: normalized vector cannot be used to recover this scalar.
        self._raw_weight_sum = float(self._X.shape[0])
        #: Estimation-sample covariates, retained for margins(over=...).
        self._model_data: pd.DataFrame | None = None
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
            self._raw_weight_sum = float(np.asarray(weights, dtype=float).sum())
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
        model = cls(y, X, weights=weights, weight_type=weight_type, param_names=names)
        model._model_data = data.loc[kept_index].reset_index(drop=True)
        return model

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
            X,
            resid,
            cov_type=effective_cov_type,
            cov_kwds=cov_kwds,
            ssc=ssc,
            nobs=n,
            df_resid=df_resid,
            nobs_adj=effective_n,
            weights=None if spec is None else w,
            weight_type=None if spec is None else spec.weight_type,
        )

        rss = float((w * resid**2).sum())
        ybar = float((w * y).sum() / w.sum())
        has_intercept = any(name == "Intercept" for name in self._param_names)
        if has_intercept:
            # Centred TSS: Stata's e(mss) is the explained sum of squares about
            # the mean, so e(rss) + e(mss) = e(tss) with tss = sum w (y - ybar)^2.
            tss = float((w * (y - ybar) ** 2).sum())
        else:
            # With noconstant Stata does not centre: r2 = 1 - rss/sum(w y^2) and
            # mss = sum(w y^2) - rss.  Verified on auto.dta,
            # ``regress mpg weight foreign, noconstant``: r2 = 0.8576921885397891,
            # mss = 30883.780324940726 (= 36008 - 5124.219675059).
            tss = float((w * y**2).sum())
        mss = tss - rss
        r2 = 1.0 - rss / tss if tss > 0 else float("nan")
        # Stata's e(r2_a) = 1 - (1 - r2) * (N - c) / (N - K), where c = 1 when the
        # model has a constant and c = 0 with ``noconstant`` (verified on
        # auto.dta: 0.6532015851691599 and, for ``regress mpg weight foreign,
        # noconstant``, 0.8537391937770055 = 1 - (1-0.8576921885397891)*74/72).
        r2_adj = (
            1.0 - (1.0 - r2) * (effective_n - (1.0 if has_intercept else 0.0)) / df_resid
            if np.isfinite(r2)
            else float("nan")
        )
        sigma2 = rss / df_resid
        rmse = float(np.sqrt(sigma2))
        llf = float(
            -0.5 * effective_n * (np.log(2 * np.pi) + np.log(rss / effective_n) + 1.0)
        )
        # Stata's e(ll_0) is the same Gaussian likelihood for the intercept-only
        # model, i.e. with sigma^2_0 = TSS/N.  Verified on auto.dta for
        # nonrobust, vce(robust), vce(cluster), pweight and fweight (all 0.0
        # relative).  ``noconstant`` models have no e(ll_0) in Stata.
        # NOTE: under aweight Stata's e(ll)/e(ll_0) carry a weight-convention
        # offset that westata does not reproduce (both are ~2.2e-2 off, by the
        # same absolute 4.4044), so ``e(ll) - e(ll_0)`` -- the LR statistic -- is
        # still exact.  See ``retest/fix/linear-iv/FIXES.md``.
        ll_0 = (
            float(-0.5 * effective_n * (np.log(2 * np.pi) + np.log(tss / effective_n) + 1.0))
            if has_intercept and tss > 0
            else None
        )
        df_model = k - 1 if has_intercept else k
        # Stata's e(F) is a Wald test of ALL non-constant coefficients using the
        # covariance matrix actually reported, divided by its degrees of freedom
        # ([R] regress, "Methods and formulas"; verified against StataNow 19.5
        # for nonrobust, vce(robust), vce(cluster) and pweight at <= 1e-15).
        # The classic R^2-form F is what this reduces to for the nonrobust VCE,
        # so one definition covers every cov_type instead of only the default.
        tested = [i for i, name in enumerate(self._param_names) if name != "Intercept"]
        if df_model == 0:
            # Stata reports e(F) = 0 for the intercept-only model (``regress y``).
            fvalue = 0.0
        elif tested:
            fvalue = _wald_f(np.asarray(beta, dtype=float), cov, tested)
        else:
            fvalue = float("nan")
        # Stata e(aic)/e(bic) for regress use the Gaussian likelihood with the
        # MLE of sigma; statsmodels' OLS estimates sigma by sqrt(RSS/df_resid),
        # so report the Stata-consistent values here.
        aic = float(effective_n * np.log(rss / effective_n) + 2 * k)
        bic = float(effective_n * np.log(rss / effective_n) + k * np.log(effective_n))

        # Stata stores the *reference-distribution* degrees of freedom in
        # e(df_r): N - K normally, G - 1 under vce(cluster id) ([R] regress,
        # "Degrees of freedom"; verified on auto.dta: 12 clusters -> e(df_r) =
        # 11 while e(rmse) keeps sqrt(rss/(N-K))).  ``results.df_resid`` already
        # carries that same value, so the two stay in step.
        e: dict[str, Any] = {
            "N": effective_n,
            "sum_w": self._raw_weight_sum,
            "df_m": df_model,
            "df_r": df_t,
            "r2": r2,
            "r2_a": r2_adj,
            "rmse": rmse,
            "sigma": rmse,
            "F": fvalue,
            "ll": llf,
            "rss": rss,
            "mss": mss,
            "rank": k,
            "aic": aic,
            "bic": bic,
        }
        if ll_0 is not None:
            e["ll_0"] = ll_0
        if str(effective_cov_type).strip().lower() == "cluster":
            groups = (cov_kwds or {}).get("groups") if cov_kwds else None
            e["N_clust"] = float(np.unique(encode_groups(groups)).size)

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
            design_matrix=(X, self._param_names),
            model_data=self._model_data,
        )
