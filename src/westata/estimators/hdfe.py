"""High-dimensional fixed effects, targeting Stata ``reghdfe`` / ``areg``.

The estimator is a thin, Stata-conformance layer over :func:`pyfixest.feols`:
PyFixest owns the alternating-projection demeaning and the ``(N-1)/(N-K)`` and
``G/(G-1)`` small-sample factors (which is why its covariance matrix is reused
rather than recomputed), while this module owns the Stata mapping:

* ``cov_type`` -> PyFixest ``vcov`` (``"nonrobust"`` -> ``None``, ``"HC1"`` ->
  ``"HC1"``, ``"cluster"`` -> ``{"CRV1": <var>}``),
* Westata's :class:`~westata.protocols.SSC` -> :func:`pyfixest.ssc`
  (``adj`` -> ``k_adj``, ``cluster_adj`` -> ``G_adj``, ``G_df="min"``),
* the ``StataResults`` payload, including Stata's residual ``e(df_r)`` (which
  counts every absorbed fixed-effect parameter) and ``e(N_clust)``.

Formula syntax is PyFixest's native one::

    "y ~ x1 + x2 | fe1 + fe2"      # two-way fixed effects
    "y ~ x1"                       # degrades to plain OLS (with intercept)

Absorbed (collinear) regressors are never silently dropped: their names are
warned about as ``RuntimeWarning`` and stored on the result object as
``results.absorbed``.

Conformance notes
-----------------
``e(rank)``
    The number of estimated parameters *counted by the degrees-of-freedom
    correction*: reported regressors plus the absorbed fixed-effect parameters
    (PyFixest's ``_df_k``).  Consequently ``e(df_r) == e(N) - e(rank)``, which
    is the relation ``reghdfe``'s ``e(df_r)`` satisfies.  It is deliberately
    larger than ``len(results.params)``.
``e(df_m)``
    Reported slope parameters only (absorbed fixed effects excluded), matching
    the number of coefficients shown in the Stata output.
``e(rmse)``
    ``sqrt(rss / e(df_r))`` -- a residual standard deviation, not PyFixest's
    ``sqrt(rss / N)``.
``results.df_resid`` vs ``e(df_r)``
    The two differ under clustering, exactly as for westata's ``OLS``:
    inference uses ``G - 1`` (``results.df_resid``) while the residual df used
    by ``rmse`` and the adjusted R-squared stays ``N - K`` (``e(df_r)``).

Nested fixed effects
--------------------
Following ``reghdfe``'s default ``dofadjustments(nested)`` (and fixest's
``k_fixef="nonnested"``), a fixed effect that is *fully nested* in the cluster
variable contributes no parameters to ``K``: the cluster absorbs it.  So
``e(rank)`` and therefore ``e(df_r)``/``e(rmse)`` legitimately shrink when the
cluster variable nests a fixed effect, while the slopes are unchanged.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import pyfixest

from westata.data.weights import WeightSpec, normalize_weights
from westata.errors import ConsistencyError, WeightError
from westata.protocols import SSC
from westata.results.base import StataResults

__all__ = ["FEOLS"]

SUPPORTED_COV_TYPES = ("nonrobust", "HC0", "HC1", "cluster")
"""Canonical ``cov_type`` values supported by :class:`FEOLS`.

``HC2``/``HC3`` are deliberately absent: they need the leverage of the *full*
design (including every absorbed dummy), PyFixest refuses them for
fixed-effect models, and ``reghdfe`` itself only offers ``unadjusted``,
``robust`` (HC1) and ``cluster``.  Guessing an approximation here would produce
standard errors westata cannot stand behind, so the request is rejected.
"""

_COV_ALIASES = {
    "NONROBUST": "nonrobust",
    "IID": "nonrobust",
    "UNROBUST": "nonrobust",
    "HC0": "HC0",
    "HC1": "HC1",
    "CLUSTER": "cluster",
    "CLUSTERED": "cluster",
    "CRV1": "cluster",
}

_UNSUPPORTED_COV_TYPES = {
    "HC2": "HC2 needs the leverage of the full dummy design, which PyFixest "
    "does not expose for fixed-effect models",
    "HC3": "HC3 needs the leverage of the full dummy design, which PyFixest "
    "does not expose for fixed-effect models",
    "HAC": "reghdfe supports HAC/Driscoll-Kraay inference, but westata P1 does "
    "not implement it for FEOLS yet",
    "TWOWAY": "westata uses cov_type='cluster' with two cluster variables for "
    "multi-way clustering",
    "ROBUST": "use cov_type='HC1' (Stata's vce(robust) default)",
}

# PyFixest accepts weights as a *column name*, so westata materialises them as
# a scratch column.  The prefix keeps collisions with user columns unlikely.
_WEIGHT_COLUMN = "__westata_weight__"
_CLUSTER_COLUMN_PREFIX = "__westata_cluster_"

# PyFixest's `ssc()` values are `k_fixef="nonnested"` and `G_df="min"` by
# default, i.e. exactly the fixest/reghdfe treatment of nested fixed effects and
# multi-way clusters.  westata keeps those and only maps the two switches.
_PYFIXEST_SSC_DEFAULTS = {"k_fixef": "nonnested", "G_df": "min"}


def _normalize_cov_type(cov_type: str) -> str:
    """Return the canonical ``cov_type`` or raise ``ValueError``."""

    if not isinstance(cov_type, str):
        raise TypeError(
            "cov_type must be a string such as 'nonrobust', 'HC1' or 'cluster'; "
            f"got {type(cov_type).__name__}"
        )
    key = cov_type.strip().upper()
    if key in _UNSUPPORTED_COV_TYPES:
        raise ValueError(
            f"cov_type {cov_type!r} is not supported by FEOLS: "
            + _UNSUPPORTED_COV_TYPES[key]
            + ". Supported cov_type values are: "
            + ", ".join(SUPPORTED_COV_TYPES)
        )
    try:
        return _COV_ALIASES[key]
    except KeyError:
        raise ValueError(
            f"Unknown cov_type {cov_type!r} for FEOLS; supported cov_type values "
            "are: " + ", ".join(SUPPORTED_COV_TYPES)
        ) from None


class FEOLS:
    """Fixed-effects OLS with high-dimensional fixed effects (Stata ``reghdfe``).

    Parameters
    ----------
    formula:
        PyFixest formula with optional fixed-effect block after ``|``, e.g.
        ``"y ~ x1 + x2 | firm + year"``.
    data:
        Estimation sample; must contain every variable named in ``formula``
        (and every ``weights`` / cluster variable).
    weights:
        Optional Stata-weighted observation weights, aligned positionally with
        ``data``.
    weight_type:
        One of ``"fweight"``, ``"aweight"``, ``"pweight"``, ``"iweight"``.
        Required whenever ``weights`` is given; Stata never guesses the type
        (``[U] 20.24``).

    Notes
    -----
    ``pweight`` implies ``vce(robust)``: :meth:`fit` therefore selects ``HC1``
    when ``cov_type`` is left at its default.
    """

    def __init__(
        self,
        formula: str,
        data: pd.DataFrame,
        *,
        weights: Any = None,
        weight_type: str | None = None,
    ) -> None:
        if not isinstance(formula, str):
            raise TypeError(
                f"formula must be a string; got {type(formula).__name__}"
            )
        if "~" not in formula:
            raise ValueError(
                "formula must be of the form 'y ~ x1 + x2 | fe1 + fe2'; got "
                f"{formula!r}"
            )
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        self._formula = formula.strip()
        # Positional alignment: PyFixest performs listwise deletion and keeps the
        # surviving labels, so a RangeIndex makes row-alignment unambiguous.
        self._data = data.copy().reset_index(drop=True)
        self._weight_spec: WeightSpec | None = None

        if weights is not None:
            if weight_type is None:
                raise WeightError(
                    "weights were supplied without weight_type; specify one of "
                    "'fweight', 'aweight', 'pweight', 'iweight' (Stata never "
                    "guesses the weight type)"
                )
            w = np.asarray(weights, dtype=float).reshape(-1)
            if w.shape[0] != self._data.shape[0]:
                raise WeightError(
                    f"weights have {w.shape[0]} entries but the data have "
                    f"{self._data.shape[0]} rows"
                )
            self._weight_spec = normalize_weights(w, weight_type)
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
    ) -> "FEOLS":
        """Build a model from a PyFixest formula (see :class:`FEOLS`)."""

        return cls(formula, data, weights=weights, weight_type=weight_type)

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
            ``None`` selects the Stata default (``"nonrobust"``, or ``"HC1"``
            for a ``pweight`` model), ``"nonrobust"``, ``"HC0"``, ``"HC1"``,
            ``"HC2"``, ``"HC3"`` or ``"cluster"``.
        cov_kwds:
            ``{"cluster": <name | list of names | array>}`` for
            ``cov_type="cluster"``; ``{"groups": ...}`` is accepted as an alias.
        ssc:
            Stata small-sample settings; defaults to
            ``SSC(adj=True, cluster_adj=True, df="stata")``.
        """

        spec = self._weight_spec
        effective_cov_type = self._resolve_cov_type(cov_type, spec)
        data = self._data.copy()
        weight_column = self._attach_weights(data, spec)
        cluster_columns = self._resolve_cluster_columns(cov_kwds, effective_cov_type, data)

        vcov = self._pyfixest_vcov(effective_cov_type, cluster_columns)
        ssc_dict = self._pyfixest_ssc(ssc, effective_cov_type)

        try:
            model = pyfixest.feols(
                self._formula,
                data,
                vcov=vcov,
                weights=weight_column,
                ssc=ssc_dict,
                weights_type=self._pyfixest_weights_type(spec),
            )
        except ValueError as exc:  # pragma: no cover - defensive
            message = str(exc)
            if "All variables are collinear" in message:
                raise ConsistencyError(
                    "reghdfe: every regressor is collinear with the absorbed "
                    "fixed effects; the model is not identified"
                ) from exc
            raise

        return self._build_results(model, data, effective_cov_type, spec)

    # ------------------------------------------------------------- mapping
    def _resolve_cov_type(self, cov_type: str | None, spec: WeightSpec | None) -> str:
        """Return the canonical covariance type, applying the Stata defaults."""

        if cov_type is None:
            # Stata's `[pw=w]` is identical to adding vce(robust).
            return "HC1" if (spec is not None and spec.implies_robust) else "nonrobust"

        effective = _normalize_cov_type(cov_type)
        if spec is not None and spec.implies_robust and effective == "nonrobust":
            raise ConsistencyError(
                "pweight implies vce(robust): Stata's 'reghdfe y x [pw=w]' is "
                "identical to adding vce(robust), so a nonrobust covariance "
                "cannot be requested for a pweighted model"
            )
        return effective

    def _attach_weights(self, data: pd.DataFrame, spec: WeightSpec | None) -> str | None:
        """Materialise the (normalized) weights as a column PyFixest can read."""

        if spec is None:
            return None
        if _WEIGHT_COLUMN in data.columns:
            raise ValueError(
                f"data already contains a column named {_WEIGHT_COLUMN!r}; "
                "rename it before passing weights to FEOLS"
            )
        data[_WEIGHT_COLUMN] = spec.weights
        return _WEIGHT_COLUMN

    def _resolve_cluster_columns(
        self,
        cov_kwds: Mapping[str, Any] | None,
        cov_type: str,
        data: pd.DataFrame,
    ) -> list[str] | None:
        """Normalise ``cov_kwds`` into a list of PyFixest cluster column names."""

        if cov_type != "cluster":
            if cov_kwds:
                raise ValueError(
                    f"cov_kwds was supplied but cov_type={cov_type!r} does not "
                    "use cluster labels; pass cov_type='cluster'"
                )
            return None

        if cov_kwds is None:
            raise ValueError(
                "cov_type='cluster' requires cov_kwds['cluster'] (or "
                "cov_kwds['groups']) naming the cluster variable"
            )
        if not isinstance(cov_kwds, Mapping):
            raise TypeError(
                "cov_kwds must be a mapping (dict-like); got "
                f"{type(cov_kwds).__name__}"
            )

        labels = cov_kwds.get("cluster", cov_kwds.get("groups"))
        if labels is None:
            raise ValueError(
                "cov_type='cluster' requires cov_kwds['cluster'] (or "
                "cov_kwds['groups']); got keys " + ", ".join(map(str, cov_kwds))
            )

        if isinstance(labels, str):
            if labels not in data.columns:
                raise ValueError(
                    f"cov_kwds['cluster']={labels!r} is not a column of the data"
                )
            return [labels]

        if isinstance(labels, Sequence) and not isinstance(labels, (np.ndarray, pd.Series)):
            names = []
            for item in labels:
                if not isinstance(item, str) or item not in data.columns:
                    raise ValueError(
                        "cov_kwds['cluster'] must be a column name, a list of "
                        f"column names or an array of labels; got {item!r}"
                    )
                names.append(item)
            if not names:
                raise ValueError("cov_kwds['cluster'] must not be empty")
            return names

        # Array/Series of labels: add scratch columns so PyFixest can cluster.
        if isinstance(labels, pd.Series):
            values = labels.to_numpy()
        else:
            values = np.asarray(labels)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.shape[0] != data.shape[0]:
            raise ValueError(
                "cov_kwds['cluster'] has "
                f"{values.shape[0]} labels but the data have {data.shape[0]} rows"
            )
        names = []
        for position in range(values.shape[1]):
            name = f"{_CLUSTER_COLUMN_PREFIX}{position}__"
            data[name] = values[:, position]
            names.append(name)
        return names

    @staticmethod
    def _pyfixest_vcov(cov_type: str, cluster_columns: list[str] | None) -> Any:
        """Map the canonical ``cov_type`` onto a PyFixest ``vcov`` argument."""

        if cov_type == "nonrobust":
            return None
        if cov_type == "cluster":
            if not cluster_columns:
                raise ConsistencyError(
                    "reghdfe: cov_type='cluster' without any cluster variable; "
                    "this is a bug in westata, please report it"
                )
            return {"CRV1": "+".join(cluster_columns)}
        if cov_type == "HC0":
            # No small-sample factor by definition; the HC0 variance is the raw
            # sandwich, so the k_adj switch is forced off for this cov_type.
            return "hetero"
        # HC1/HC2/HC3 map onto PyFixest's heteroskedastic variants.
        return cov_type

    @staticmethod
    def _pyfixest_ssc(ssc: SSC | None, cov_type: str) -> dict[str, str | bool]:
        """Map a westata :class:`SSC` onto :func:`pyfixest.ssc`'s dict."""

        if ssc is None:
            settings = SSC()
        elif isinstance(ssc, SSC):
            settings = ssc
        else:
            raise TypeError(
                "ssc must be a westata SSC instance (or None); got "
                f"{type(ssc).__name__}"
            )

        if not isinstance(settings.df, str):
            raise TypeError(
                f"ssc.df must be a string; got {type(settings.df).__name__}"
            )

        k_adj = bool(settings.adj)
        if cov_type == "HC0":
            # HC0 is *defined* by the absence of the N/(N-K) factor; respecting
            # ssc.adj here would silently turn HC0 into HC1.
            k_adj = False

        return dict(
            k_adj=k_adj,
            G_adj=bool(settings.cluster_adj),
            **_PYFIXEST_SSC_DEFAULTS,
        )

    @staticmethod
    def _pyfixest_weights_type(spec: WeightSpec | None) -> str:
        """Map a Stata weight type onto PyFixest's ``weights_type`` argument."""

        if spec is None:
            return "aweights"
        # PyFixest implements analytic (aweights) and frequency (fweights)
        # weights only.  pweights share the aweight point estimate and differ
        # solely in the variance, which westata already forces to HC1; iweights
        # are used unnormalised like fweights (``[U] 20.24``).
        if spec.weight_type in ("fweight", "iweight"):
            return "fweights"
        return "aweights"

    # ------------------------------------------------------------ results
    def _build_results(
        self,
        model: Any,
        data: pd.DataFrame,
        cov_type: str,
        spec: WeightSpec | None,
    ) -> StataResults:
        """Translate a fitted ``pyfixest.Feols`` into :class:`StataResults`."""

        names = [str(name) for name in model.coef().index]
        params = np.asarray(model.coef().to_numpy(), dtype=float)
        cov = np.asarray(model._vcov, dtype=float)
        if cov.shape != (len(names), len(names)):
            raise ConsistencyError(
                "reghdfe: PyFixest returned a covariance matrix of shape "
                f"{cov.shape} for {len(names)} coefficients"
            )

        absorbed = [str(name) for name in (model._collin_vars or [])]
        if absorbed:
            warnings.warn(
                "reghdfe: "
                + ", ".join(absorbed)
                + f" dropped from the estimation sample because "
                + ("it is" if len(absorbed) == 1 else "they are")
                + " collinear with the absorbed fixed effects or with another "
                "regressor; the estimate is still reported but has no "
                "coefficient",
                RuntimeWarning,
                stacklevel=3,
            )

        resid = np.asarray(model.resid(), dtype=float).reshape(-1)
        fitted = np.asarray(model.predict(), dtype=float).reshape(-1)
        nobs = int(resid.size)
        if fitted.size != nobs:
            raise ConsistencyError(
                "reghdfe: PyFixest returned "
                f"{fitted.size} fitted values for {nobs} residuals"
            )

        y = fitted + resid
        kept_weights = self._kept_weights(model, spec, nobs)
        n_effective = self._effective_n(spec, kept_weights, nobs)

        rank = int(round(float(getattr(model, "_df_k", nobs - float(model._df_t)))))
        # `results.df_resid` follows the reference distribution (G - 1 under
        # clustering, matching westata's OLS); the *residual* df used by rmse and
        # the adjusted R^2 stays N - K, as in Stata's e(df_r).
        df_resid = float(model._df_t)
        residual_df = float(n_effective - rank)
        if residual_df <= 0:
            raise ConsistencyError(
                "reghdfe: "
                f"{n_effective:g} effective observations but {rank} parameters "
                "(including absorbed fixed effects); the model is not identified"
            )
        has_intercept = any(name == "Intercept" for name in names)
        tested = [index for index, name in enumerate(names) if name != "Intercept"]
        df_model = float(len(tested)) if has_intercept else float(len(names))

        rss = float(np.sum(kept_weights * resid**2))
        mean_y = float(np.sum(kept_weights * y) / np.sum(kept_weights))
        tss = float(np.sum(kept_weights * (y - mean_y) ** 2))
        r2 = 1.0 - rss / tss if tss > 0 else float("nan")
        r2_adj = (
            1.0 - (1.0 - r2) * (n_effective - 1.0) / residual_df
            if np.isfinite(r2)
            else float("nan")
        )
        rmse = float(np.sqrt(rss / residual_df))
        llf = float(
            -0.5 * n_effective * (np.log(2 * np.pi) + np.log(rss / n_effective) + 1.0)
        ) if rss > 0 else float("nan")

        fvalue = self._wald_f(params, cov, tested)

        e: dict[str, Any] = {
            "N": n_effective,
            "df_m": df_model,
            "df_r": residual_df,
            "r2": r2,
            "r2_a": r2_adj,
            "rmse": rmse,
            "F": fvalue,
            "ll": llf,
            "rank": rank,
        }
        if cov_type == "cluster":
            e["N_clust"] = float(self._n_clusters(model))

        results = StataResults(
            params=params,
            param_names=names,
            cov=cov,
            resid=resid,
            fittedvalues=fitted,
            nobs=nobs,
            df_resid=df_resid,
            df_model=df_model,
            dist="t",
            model_name="reghdfe",
            cov_type=cov_type,
            e=e,
        )
        # Reported, not hidden: absorbed regressors are part of the result.
        results.absorbed = absorbed
        return results

    @staticmethod
    def _wald_f(params: np.ndarray, cov: np.ndarray, tested: list[int]) -> float:
        """Wald F for ``H0: beta_tested = 0`` using the reported covariance."""

        if not tested:
            return float("nan")
        beta = params[tested]
        sub = cov[np.ix_(tested, tested)]
        try:
            quadratic = float(beta @ np.linalg.solve(sub, beta))
        except np.linalg.LinAlgError:
            return float("nan")
        return quadratic / len(tested)

    @staticmethod
    def _kept_weights(
        model: Any, spec: WeightSpec | None, nobs: int
    ) -> np.ndarray:
        """Weights of the observations PyFixest actually used."""

        if spec is None or _WEIGHT_COLUMN not in getattr(model, "_data", pd.DataFrame()):
            return np.ones(nobs, dtype=float)
        kept = np.asarray(model._data[_WEIGHT_COLUMN], dtype=float).reshape(-1)
        if kept.size != nobs:
            raise ConsistencyError(
                "reghdfe: could not align the weights with the estimation "
                f"sample ({kept.size} weights for {nobs} observations)"
            )
        return kept

    @staticmethod
    def _effective_n(spec: WeightSpec | None, weights: np.ndarray, nobs: int) -> float:
        """Sample size for the degrees of freedom (Stata ``[U] 20.24``)."""

        if spec is None:
            return float(nobs)
        if spec.weight_type in ("fweight", "iweight"):
            return float(np.sum(weights))
        return float(nobs)

    @staticmethod
    def _n_clusters(model: Any) -> int:
        """Number of clusters used for the clustered VCE."""

        groups = getattr(model, "_G", None)
        if groups is None:
            raise ConsistencyError(
                "reghdfe: PyFixest did not record the number of clusters"
            )
        return int(min(groups))
