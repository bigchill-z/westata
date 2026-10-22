"""Panel estimators targeting Stata's ``xtreg`` family (P1 Task 5).

=====================  =============  ================================
westata                Stata          backend
=====================  =============  ================================
``PanelOLS``           ``xtreg, fe``  ``linearmodels.PanelOLS``
``RandomEffects``      ``xtreg, re``  ``linearmodels.RandomEffects``
``BetweenOLS``         ``xtreg, be``  ``linearmodels.BetweenOLS``
``FirstDifferenceOLS`` ``xtreg, fd``  ``linearmodels.FirstDifferenceOLS``
=====================  =============  ================================

These are Stata-semantics adapters over ``linearmodels.panel``: point estimates
and covariances come from the backend, while the degrees of freedom and the
``e()`` scalars are recomputed here from Stata's formulas (``[R] xtreg``) so the
two cannot silently disagree.

Degrees of freedom: what linearmodels returns vs what Stata reports
------------------------------------------------------------------
Measured on linearmodels 7.1 with the balanced 5-entity x 4-period panel that
``tests/test_panel.py`` uses (``y ~ 1 + x1 + x2``; ``N = 20``; ``K = 3`` with
the constant; ``n_entities = 5``):

==============================  ==============  =======
quantity                        source          value
==============================  ==============  =======
``PanelOLS(...).fit(auto_df=True).df_resid``   attribute      13
``PanelOLS(...).fit(auto_df=False).df_resid``  attribute      13
``PanelOLS(...).fit().df_model``               attribute       7
Stata ``xtreg, fe`` ``df_r``                   ``N-K-n+1``    13
==============================  ==============  =======

For a one-way model with a constant the two conventions coincide, because
linearmodels counts the absorbed effect columns in its model rank::

    linearmodels: df_resid = N - df_model = N - K - (n_entities - 1) = 13
    Stata:        df_r     = N - K - n_entities + 1                  = 13

so ``N - K - (n_entities - 1)`` and ``N - K - n_entities + 1`` are the same
number once the intercept is present.  Two consequences matter:

* ``auto_df`` does **not** change ``df_resid`` here - both settings report 13.
  What ``auto_df`` changes is the *covariance*: with ``cov_type="clustered"`` and
  a single effect nested in the cluster variable, ``auto_df=True`` (the backend
  default) stops counting the effects, so the standard errors differ from
  Stata's ``vce(cluster id)``.  westata always passes ``auto_df=False`` plus
  ``group_debias=True``, which reproduces Stata's
  ``G/(G-1) * (N-1)/(N-K)`` CR1 exactly (see
  ``test_auto_df_off_reproduces_stata_clustered_standard_errors``).
* Without a constant (``"y ~ x1 + x2 - 1"``) the ``+1`` disappears from both
  spellings and both report 13 - i.e. ``N - K - n_entities``.  westata derives
  the effect count from whether the fitted model actually carries an intercept,
  so this case stays conformant instead of assuming a constant exists.

westata never forwards the backend's ``df_resid``; :meth:`PanelOLS._stata_df_resid`
recomputes ``N - K - sum(levels - 1)`` (a constant absorbs one base level per
effect, which is where Stata's ``+1`` comes from) and that number is what
reaches :class:`westata.results.StataResults`.

The other family members use the plain Stata rules: ``xtreg, re`` and
``xtreg, fd`` report ``N - K``; ``xtreg, be`` reports ``n_entities - K`` because
the between regression actually runs on the group means.

Covariance mapping
------------------
===========================  ===================================================
westata ``cov_type``         linearmodels
===========================  ===================================================
``None`` / ``"nonrobust"``   ``cov_type="unadjusted"``
``"HC1"``                    ``cov_type="robust", debiased=True`` (Stata's
                             ``vce(robust)`` is HC1)
``"cluster"``                ``cov_type="clustered"``, ``cov_kwds["clusters"]``,
                             ``group_debias=True``
===========================  ===================================================

``HC0``/``HC2``/``HC3`` are rejected with :class:`UnsupportedError` instead of
being folded into Stata's ``vce(robust)``: returning an HC2 number under the
label "robust" would be a quiet numerical lie.

Known gaps (documented, not silently approximated)
--------------------------------------------------
* ``xtreg, fe``'s ``e(r2)`` is the *within* R-squared (as Stata reports it);
  ``e(r2_a)`` follows ``1 - (1 - r2)(N - 1)/df_r``, which is Stata's
  ``regress``/``xtreg`` adjustment applied to the reported R-squared.
* ``xtreg, be`` reports ``e(N)`` as the number of group means the regression
  consumes; Stata's header may show the underlying observation count instead.
* ``e(ll)`` is the Gaussian log-likelihood evaluated at the ML variance,
  ``-N/2 * (ln 2pi + ln(rss/N) + 1)``, matching ``regress``'s definition; for
  ``xtreg, re`` Stata reports the GLS log-likelihood, which is not reproduced.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from westata.errors import ConsistencyError, UnsupportedError
from westata.protocols import SSC
from westata.results.base import StataResults
from westata.vce.ssc import resolve_ssc

__all__ = [
    "PanelOLS",
    "RandomEffects",
    "BetweenOLS",
    "FirstDifferenceOLS",
]

#: ``cov_type`` (lower-case) -> linearmodels covariance name.
_BACKEND_COV_TYPES: dict[str, str] = {
    "nonrobust": "unadjusted",
    "unadjusted": "unadjusted",
    "hc1": "robust",
    "robust": "robust",
    "cluster": "clustered",
    "clustered": "clustered",
}

#: ``cov_type`` values Stata can express for ``xtreg`` but this adapter cannot.
_UNSUPPORTED_COV_TYPES: dict[str, str] = {
    "hc0": "HC0 is not Stata's vce(robust) (Stata applies the HC1 factor N/(N-K))",
    "hc2": "panel HC2 has no Stata xtreg counterpart",
    "hc3": "panel HC3 has no Stata xtreg counterpart",
    "hac": "vce(HAC) needs newey/xtscc, not xtreg",
    "twoway": "two-way clustering needs a pairs argument this adapter does not expose",
}

_CONSTANT_NAMES = frozenset({"intercept", "const", "_cons"})


def _normalize_cov_type(cov_type: str | None) -> str:
    """Map a westata ``cov_type`` onto a linearmodels covariance name."""

    if cov_type is None:
        return "unadjusted"
    key = str(cov_type).strip().lower()
    if key in _UNSUPPORTED_COV_TYPES:
        raise UnsupportedError(
            f"cov_type={cov_type!r} is not available for the xtreg family: "
            f"{_UNSUPPORTED_COV_TYPES[key]}"
        )
    try:
        return _BACKEND_COV_TYPES[key]
    except KeyError:
        raise ValueError(
            f"Unknown cov_type {cov_type!r}; supported cov_type values for the "
            "xtreg family are: nonrobust, HC1, cluster"
        ) from None


def _cov_label(cov_type: str | None) -> str:
    """Label recorded on the result, in westata's own vocabulary."""

    if cov_type is None:
        return "nonrobust"
    key = str(cov_type).strip().lower()
    if key in ("robust",):
        return "HC1"
    if key in ("clustered",):
        return "cluster"
    return key.upper() if key.startswith("hc") else key


class PanelOLS:
    """Fixed-effects panel regression (Stata ``xtreg, fe``).

    Parameters
    ----------
    formula:
        Two-sided formula in statsmodels/patsy syntax, for example
        ``"y ~ x1 + x2"``.  The constant is implied by patsy exactly as in
        :class:`westata.estimators.linear.OLS`, so ``e(b)`` includes
        ``Intercept`` even though the within transformation absorbs it.
        Drop it with ``"y ~ x1 + x2 - 1"`` to follow Stata's ``noconstant``.
    data:
        Panel frame.  Either a long frame with the ``entity``/``time`` columns,
        or a frame already indexed by a two-level ``(entity, time)``
        ``MultiIndex`` (``entity``/``time`` must then name two of its levels).
    entity, time:
        Names of the panel dimensions.

    Notes
    -----
    Rows with missing values are dropped listwise *before* estimation, and the
    retained sample is what ``e(N)`` counts, so ``e(N)`` can be smaller than
    ``len(data)`` - the same behaviour as Stata's ``regress``/``xtreg``.
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
        if not isinstance(formula, str) or "~" not in formula:
            raise ConsistencyError(
                f"{self._stata_command}: formula must be two-sided (contain "
                f"'~'), got {formula!r}"
            )

        self._formula = formula
        self._entity = entity
        self._time = time
        self._data = _panel_frame(data, entity=entity, time=time, command=self._stata_command)
        # Remember the full sample so that user-supplied, observation-aligned
        # arguments (cluster labels) can be mapped onto the estimation sample
        # *after* listwise deletion.
        self._full_index = self._data.index
        self._kept_index = self._data.index

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        entity: str,
        time: str,
    ) -> "PanelOLS":
        """Build a model from a formula and a panel frame.

        A long frame carrying the ``entity``/``time`` columns is converted with
        ``set_index([entity, time])``; a frame that already carries the
        two-level ``MultiIndex`` is used as is.
        """

        return cls(formula, data, entity=entity, time=time)

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        entity_effects: bool = True,
        time_effects: bool = False,
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: SSC | None = None,
    ) -> StataResults:
        """Estimate the model and return a Stata-conformant result object.

        Parameters
        ----------
        entity_effects, time_effects:
            Which fixed effects to absorb.  ``False``/``False`` estimates the
            pooled model (Stata's ``xtreg`` baseline, i.e. ``regress``).
        cov_type:
            ``None``/``"nonrobust"``, ``"HC1"`` or ``"cluster"``.
        cov_kwds:
            ``{"clusters": ...}`` (or ``{"groups": ...}``) is required for
            ``cov_type="cluster"``.  Labels may be any 1-D array-like with one
            entry per observation in ``data`` row order, or a ``Series``.
        ssc:
            Stata small-sample settings (``SSC(adj=True, cluster_adj=True)``
            defaults).  ``adj=False`` turns off the observation-based
            correction (``debiased=False``); ``cluster_adj=False`` drops the
            ``G/(G-1)`` factor.  Both are the switches ``[R] regress`` exposes.

        Raises
        ------
        ValueError
            Unknown ``cov_type``, or cluster labels not supplied.
        UnsupportedError
            A ``cov_type`` Stata expresses for ``xtreg`` that this adapter
            cannot reproduce (HC0/HC2/HC3/HAC/two-way).
        ConsistencyError
            The panel is unusable as given (duplicate cells, missing columns,
            too few clusters, no residual degrees of freedom).
        """

        y, X = self._design()
        self._kept_index = self._data.index
        backend_cov, clusters, n_clust = self._resolve_cov(cov_type, cov_kwds)
        settings = resolve_ssc(ssc)

        model = self._build_model(y, X, entity_effects, time_effects)
        ref = self._fit_backend(model, backend_cov, clusters, settings)

        params = np.asarray(ref.params, dtype=float).reshape(-1)
        param_names = [str(name) for name in ref.params.index]
        cov = np.asarray(ref.cov, dtype=float)

        resid = _aligned_resid(ref, command=self._stata_command)
        nobs = int(resid.size)
        response = self._residual_response(y).reindex(_resid_index(ref, self._data.index))
        fitted = response.to_numpy(dtype=float) - resid

        has_constant = any(str(name).lower() in _CONSTANT_NAMES for name in param_names)
        rank = len(param_names)
        effect_df = self._effect_df(model, has_constant)
        df_model = self._stata_df_model(rank, has_constant)
        df_resid = self._stata_df_resid(nobs, rank, effect_df)

        e = self._stata_scalars(
            ref,
            nobs=nobs,
            resid=resid,
            df_model=df_model,
            df_resid=df_resid,
            rank=rank,
            n_clust=n_clust,
        )

        return StataResults(
            params=params,
            param_names=param_names,
            cov=cov,
            resid=resid,
            fittedvalues=fitted,
            nobs=nobs,
            df_resid=df_resid,
            df_model=df_model,
            dist="t",
            model_name=self._stata_command,
            cov_type=_cov_label(cov_type),
            e=e,
        )

    # -------------------------------------------------------------- internals
    def _design(self) -> tuple[pd.Series, pd.DataFrame]:
        """Build ``y``/``X`` with patsy and apply listwise deletion."""

        import patsy

        try:
            y_frame, x_frame = patsy.dmatrices(
                self._formula, self._data, return_type="dataframe"
            )
        except patsy.PatsyError as exc:
            raise ConsistencyError(
                f"{self._stata_command}: cannot build the design matrix for "
                f"{self._formula!r}: {exc}"
            ) from exc

        kept = x_frame.index
        if len(kept) == 0:
            raise ConsistencyError(
                f"{self._stata_command}: no observations remain after listwise "
                "deletion of missing values"
            )
        # Restrict the panel to the estimation sample so that cluster labels,
        # effect counts and e(N) all describe the same rows.
        self._data = self._data.loc[kept]

        y = pd.Series(
            np.asarray(y_frame.iloc[:, 0], dtype=float),
            index=kept,
            name=str(y_frame.columns[0]),
        )
        X = pd.DataFrame(
            np.asarray(x_frame, dtype=float), index=kept, columns=list(x_frame.columns)
        )
        return y, X

    def _build_model(
        self,
        y: pd.Series,
        X: pd.DataFrame,
        entity_effects: bool,
        time_effects: bool,
    ) -> Any:
        from linearmodels.panel import PanelOLS as _BackendPanelOLS

        return _BackendPanelOLS(
            y.to_frame(),
            X,
            entity_effects=bool(entity_effects),
            time_effects=bool(time_effects),
        )

    def _fit_backend(
        self,
        model: Any,
        backend_cov: str,
        clusters: np.ndarray | None,
        settings: SSC,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "cov_type": backend_cov,
            "debiased": bool(settings.adj),
        }
        extra = self._extra_fit_kwargs(model)
        kwargs.update(extra)
        if backend_cov == "clustered":
            kwargs["clusters"] = clusters
            # ``group_debias`` adds Stata's G/(G-1) factor on top of the
            # observation-based correction -> exactly CR1.
            kwargs["group_debias"] = bool(settings.cluster_adj)
        return model.fit(**kwargs)

    def _extra_fit_kwargs(self, model: Any) -> dict[str, Any]:
        """Backend-specific ``fit`` keywords (``PanelOLS`` has ``auto_df``)."""

        from linearmodels.panel import PanelOLS as _BackendPanelOLS

        if not isinstance(model, _BackendPanelOLS):
            return {}
        return {
            # Stata always counts the absorbed effects in the covariance
            # correction; ``auto_df=True`` (the backend default) does not when
            # an effect is nested in the clusters, so pin it off.
            "auto_df": False,
        }

    def _resolve_cov(
        self, cov_type: str | None, cov_kwds: Mapping[str, Any] | None
    ) -> tuple[str, np.ndarray | None, int | None]:
        backend_cov = _normalize_cov_type(cov_type)
        kwds = {} if cov_kwds is None else dict(cov_kwds)

        if backend_cov != "clustered":
            unused = set(kwds) - {"clusters", "groups"}
            if unused:
                raise ValueError(
                    "cov_kwds is only consumed for cov_type='cluster'; got "
                    f"unexpected key(s) {sorted(unused)}"
                )
            return backend_cov, None, None

        clusters = kwds.get("clusters", kwds.get("groups"))
        if clusters is None:
            raise ValueError(
                "cov_type='cluster' requires cov_kwds['clusters']; pass the "
                "cluster labels as cov_kwds={'clusters': ...}"
            )
        labels = _cluster_labels(clusters, self._full_index, self._kept_index)
        n_clust = int(pd.unique(labels).size)
        if n_clust < 2:
            raise ConsistencyError(
                f"{self._stata_command}: cluster-robust covariance requires at "
                f"least 2 clusters; got {n_clust}"
            )
        return backend_cov, labels, n_clust

    # ---------------------------------------------------- Stata d.o.f. rules
    def _effect_df(self, model: Any, has_constant: bool) -> int:
        """Effect columns linearmodels adds to the model rank.

        With a constant, each set of effects contributes ``levels - 1`` (one
        level is the base); that subtracted ``1`` per effect is exactly the
        ``+1`` in Stata's ``N - K - n_entities + 1``.
        """

        absorbed = 0
        for name, requested in self._effect_requests(model):
            if not requested:
                continue
            levels = int(self._data.index.get_level_values(name).nunique())
            absorbed += levels - 1 if has_constant else levels
        return absorbed

    def _effect_requests(self, model: Any) -> list[tuple[str, bool]]:
        """``[(dimension, requested)]`` for the effects this model absorbs."""

        return [
            (self._entity, bool(getattr(model, "entity_effects", False))),
            (self._time, bool(getattr(model, "time_effects", False))),
        ]

    def _stata_df_model(self, rank: int, has_constant: bool) -> float:
        """``e(df_m)``: the number of reported parameters excluding the constant."""

        return float(rank - 1 if has_constant else rank)

    def _stata_df_resid(self, nobs: int, rank: int, effect_df: int) -> float:
        """``xtreg, fe``: ``df_r = N - K - n_entities + 1`` (``[R] xtreg``)."""

        df_resid = float(nobs - rank - effect_df)
        if df_resid <= 0:
            raise ConsistencyError(
                f"{self._stata_command}: no residual degrees of freedom "
                f"(N={nobs}, K={rank}, absorbed effects={effect_df})"
            )
        return df_resid

    def _residual_response(self, y: pd.Series) -> pd.Series:
        """Response aligned with the residuals (levels for FE/RE/pooled)."""

        return y

    # ------------------------------------------------------------- e() scalars
    def _stata_scalars(
        self,
        ref: Any,
        *,
        nobs: int,
        resid: np.ndarray,
        df_model: float,
        df_resid: float,
        rank: int,
        n_clust: int | None,
    ) -> dict[str, Any]:
        rss = float(resid @ resid)
        r2 = float(getattr(ref, "rsquared", float("nan")))
        r2_a = self._stata_r2_adjusted(r2, nobs, df_resid)
        e: dict[str, Any] = {
            "N": nobs,
            "df_m": df_model,
            "df_r": df_resid,
            "r2": r2,
            "r2_a": r2_a,
            "rmse": float(np.sqrt(rss / df_resid)) if df_resid > 0 else float("nan"),
            "F": self._stata_f(ref, df_model),
            "ll": _gaussian_loglik(rss, nobs),
            "rank": rank,
            "n_entities": int(self._data.index.get_level_values(self._entity).nunique()),
            "rss": rss,
        }
        if n_clust is not None:
            e["N_clust"] = n_clust
        return e

    def _stata_r2_adjusted(self, r2: float, nobs: int, df_resid: float) -> float:
        if not np.isfinite(r2) or df_resid <= 0 or nobs <= 1:
            return float("nan")
        return float(1.0 - (1.0 - r2) * (nobs - 1.0) / df_resid)

    def _stata_f(self, ref: Any, df_model: float) -> float:
        """Joint F over the reported slopes, using the fitted covariance.

        Stata's ``xtreg`` F is the Wald statistic on the reported non-constant
        coefficients divided by ``df_m``; ``linearmodels`` exposes exactly that
        as ``f_statistic_robust`` (its ``f_statistic`` is always homoskedastic,
        which is why a robust/clustered VCE would leave it unchanged).
        """

        if df_model <= 0:
            return float("nan")
        for attribute in ("f_statistic_robust", "f_statistic"):
            candidate = getattr(ref, attribute, None)
            if candidate is None:
                continue
            value = float(getattr(candidate, "stat", float("nan")))
            if np.isfinite(value):
                return value
        return float("nan")


class RandomEffects(PanelOLS):
    """Random-effects GLS (Stata ``xtreg, re``).

    Uses linearmodels' default Swamy-Arora variance-component estimator; the
    optional ``small_sample`` refinement is deliberately left off so that the
    estimator is exactly the documented default.
    """

    _stata_command = "xtreg, re"

    def fit(  # type: ignore[override]
        self,
        entity_effects: bool = False,
        time_effects: bool = False,
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: SSC | None = None,
    ) -> StataResults:
        return super().fit(
            entity_effects=entity_effects,
            time_effects=time_effects,
            cov_type=cov_type,
            cov_kwds=cov_kwds,
            ssc=ssc,
        )

    def _build_model(
        self,
        y: pd.Series,
        X: pd.DataFrame,
        entity_effects: bool,
        time_effects: bool,
    ) -> Any:
        from linearmodels.panel import RandomEffects as _BackendRandomEffects

        self._reject_effects(entity_effects, time_effects)
        return _BackendRandomEffects(y.to_frame(), X)

    def _effect_requests(self, model: Any) -> list[tuple[str, bool]]:
        return []

    def _stata_df_resid(self, nobs: int, rank: int, effect_df: int) -> float:
        """``xtreg, re``: ``df_r = N - K``."""

        df_resid = float(nobs - rank)
        if df_resid <= 0:
            raise ConsistencyError(
                f"{self._stata_command}: no residual degrees of freedom "
                f"(N={nobs}, K={rank})"
            )
        return df_resid

    def _reject_effects(self, entity_effects: bool, time_effects: bool) -> None:
        requested = [
            name
            for name, flag in (
                ("entity_effects", entity_effects),
                ("time_effects", time_effects),
            )
            if flag
        ]
        if requested:
            raise UnsupportedError(
                f"{self._stata_command}: {' and '.join(requested)} is not a "
                "Stata option for this estimator; xtreg's re/be/fd estimators do "
                "not absorb fixed effects (use PanelOLS, i.e. xtreg, fe)"
            )


class BetweenOLS(RandomEffects):
    """Between estimator on entity means (Stata ``xtreg, be``).

    The regression runs on one row per entity, so ``e(N)`` is the number of
    group means actually consumed and ``df_r = n_entities - K``.
    """

    _stata_command = "xtreg, be"

    def _build_model(
        self,
        y: pd.Series,
        X: pd.DataFrame,
        entity_effects: bool,
        time_effects: bool,
    ) -> Any:
        from linearmodels.panel import BetweenOLS as _BackendBetweenOLS

        self._reject_effects(entity_effects, time_effects)
        return _BackendBetweenOLS(y.to_frame(), X)

    def _residual_response(self, y: pd.Series) -> pd.Series:
        """The between regression's response is the entity mean."""

        return y.groupby(level=self._entity).mean()

    def _stata_df_resid(self, nobs: int, rank: int, effect_df: int) -> float:
        """``xtreg, be``: ``df_r = n_entities - K`` (group means are the sample)."""

        df_resid = float(nobs - rank)
        if df_resid <= 0:
            raise ConsistencyError(
                f"{self._stata_command}: no residual degrees of freedom "
                f"(n_entities={nobs}, K={rank}); the between regression needs "
                "more entities than regressors"
            )
        return df_resid


class FirstDifferenceOLS(RandomEffects):
    """First-difference estimator (Stata ``xtreg, fd``).

    Periods are differenced (``t`` against ``t - 1``, so a missing period is not
    bridged) and the model is fitted without a constant; the patsy intercept is
    therefore removed before the design matrix reaches linearmodels, which
    rejects constants in first-difference regressions.
    """

    _stata_command = "xtreg, fd"

    def _build_model(
        self,
        y: pd.Series,
        X: pd.DataFrame,
        entity_effects: bool,
        time_effects: bool,
    ) -> Any:
        from linearmodels.panel import FirstDifferenceOLS as _BackendFirstDifferenceOLS

        self._reject_effects(entity_effects, time_effects)
        exog = X.drop(
            columns=[column for column in X.columns if str(column).lower() in _CONSTANT_NAMES]
        )
        return _BackendFirstDifferenceOLS(y.to_frame(), exog)

    def _residual_response(self, y: pd.Series) -> pd.Series:
        """``y_t - y_{t-1}`` per entity.

        Stata's ``xtreg, fd`` differences *adjacent periods*, and a missing
        period breaks the pair (the ``L.`` operator does not skip over a hole).
        Building the wide matrix on the complete period grid is what makes that
        true: without it, pandas' positional ``diff`` would silently bridge the
        hole and difference ``t`` against ``t - 2``.
        """

        wide = y.unstack(level=self._time).sort_index(axis=1)
        grid = _period_grid(wide.columns)
        wide = wide.reindex(columns=grid)
        differenced = wide.diff(axis=1).stack(future_stack=True)
        differenced.index = differenced.index.set_names([self._entity, self._time])
        return differenced.dropna()

    def _stata_df_model(self, rank: int, has_constant: bool) -> float:
        # No constant in a first-difference regression: every parameter is a slope.
        return float(rank)

    def _stata_df_resid(self, nobs: int, rank: int, effect_df: int) -> float:
        """``xtreg, fd``: ``df_r = N_differences - K``."""

        df_resid = float(nobs - rank)
        if df_resid <= 0:
            raise ConsistencyError(
                f"{self._stata_command}: no residual degrees of freedom "
                f"(N_differences={nobs}, K={rank})"
            )
        return df_resid


# --------------------------------------------------------------------- helpers
def _period_grid(observed: pd.Index) -> pd.Index:
    """Complete period axis spanning the observed periods.

    Inserting the unobserved periods makes ``DataFrame.diff`` treat a missing
    period as a hole (and produce ``NaN`` for the following pair) instead of
    silently differencing across it.  Integer periods - the Stata default for
    ``xtset`` panels - are filled with ``range``; anything else keeps the
    observed periods, where a hole cannot be told apart from the intended grid
    without a declared frequency.
    """

    if observed.empty or not pd.api.types.is_integer_dtype(observed.dtype):
        return observed
    return pd.Index(
        np.arange(int(observed.min()), int(observed.max()) + 1), name=observed.name
    )


def _panel_frame(data: pd.DataFrame, *, entity: str, time: str, command: str) -> pd.DataFrame:
    """Return ``data`` indexed by a sorted ``(entity, time)`` MultiIndex."""

    frame = data
    if isinstance(frame.index, pd.MultiIndex):
        levels = list(frame.index.names)
        if entity not in levels or time not in levels:
            raise ConsistencyError(
                f"{command}: the data already carry a MultiIndex {levels}, so "
                f"entity={entity!r} and time={time!r} must name two of its "
                "levels (or pass a long frame carrying those columns)"
            )
        extras = [level for level in levels if level not in (entity, time)]
        frame = frame.reorder_levels([entity, time] + extras)
    else:
        missing = [name for name in (entity, time) if name not in frame.columns]
        if missing:
            raise ConsistencyError(
                f"{command}: column(s) {missing} not found in data; available "
                f"columns: {list(frame.columns)}"
            )
        frame = frame.set_index([entity, time])

    if len(frame) == 0:
        raise ConsistencyError(f"{command}: the panel is empty")

    frame = frame.sort_index()
    duplicated = frame.index.duplicated()
    if bool(np.any(duplicated)):
        raise ConsistencyError(
            f"{command}: the panel has repeated (entity, time) cells, e.g. "
            f"{frame.index[duplicated][0]}; every entity-time pair must be unique"
        )
    if frame.index.nlevels < 3:
        return frame
    return frame.droplevel(list(range(2, frame.index.nlevels)))


def _cluster_labels(
    clusters: Any, full_index: pd.MultiIndex, kept_index: pd.MultiIndex
) -> np.ndarray:
    """Normalize cluster labels to one value per *estimation* observation.

    Labels may arrive aligned with the full input frame (the usual case: the
    caller passes ``df["industry"]``) or already restricted to the estimation
    sample, and either as a ``Series``/``Index`` or a plain array.  The three
    cases are told apart by length, and the labels are then mapped onto the
    kept rows so listwise deletion cannot misalign them.
    """

    if isinstance(clusters, (pd.Series, pd.Index)) and isinstance(
        getattr(clusters, "index", None), pd.MultiIndex
    ):
        # A real panel-indexed Series: align on the panel index, not by position.
        aligned = pd.Series(np.asarray(clusters), index=clusters.index).reindex(kept_index)
        if aligned.isna().any():
            raise ConsistencyError(
                "cluster labels do not cover the estimation sample; the panel "
                "index of the labels must contain every retained observation"
            )
        return aligned.to_numpy()

    values = np.asarray(clusters)
    if values.ndim != 1:
        raise ConsistencyError(
            f"cluster labels must be one-dimensional; got {values.ndim} dimensions"
        )

    if values.size == len(kept_index):
        labels = values
    elif values.size == len(full_index):
        labels = _restrict(values, full_index, kept_index)
    else:
        raise ConsistencyError(
            f"cluster labels have {values.size} entries but the panel has "
            f"{len(full_index)} observations ({len(kept_index)} in the "
            "estimation sample)"
        )

    if pd.Series(labels).isna().any():
        raise ConsistencyError(
            "cluster labels contain missing values; every observation must "
            "belong to a cluster"
        )
    return labels


def _restrict(values: np.ndarray, full_index: pd.MultiIndex, kept_index: pd.MultiIndex) -> np.ndarray:
    """Take the entries of ``values`` that belong to ``kept_index``."""

    if full_index.equals(kept_index):
        return values
    position = pd.Series(np.arange(len(full_index)), index=full_index)
    rows = position.reindex(kept_index)
    if rows.isna().any():  # pragma: no cover - kept rows come from full_index
        raise ConsistencyError(
            "the estimation sample is not a subset of the supplied panel"
        )
    return values[rows.to_numpy(dtype=int)]


def _resid_index(ref: Any, fallback: pd.Index) -> pd.Index:
    """Index carried by the backend residuals, or ``fallback`` when absent."""

    resid = ref.resids
    if isinstance(resid, pd.Series) and resid.index is not None:
        return resid.index
    return fallback


def _aligned_resid(ref: Any, *, command: str) -> np.ndarray:
    """Backend residuals, in the backend's own (panel-index) order."""

    values = np.asarray(ref.resids, dtype=float).reshape(-1)
    if values.size != int(ref.nobs):
        raise ConsistencyError(
            f"{command}: the backend returned {values.size} residuals for "
            f"N={int(ref.nobs)} observations"
        )
    return values


def _gaussian_loglik(rss: float, nobs: int) -> float:
    """Gaussian log-likelihood at the ML variance, as Stata's ``e(ll)``."""

    if nobs <= 0 or not np.isfinite(rss) or rss <= 0:
        return float("nan")
    return float(-0.5 * nobs * (np.log(2.0 * np.pi) + np.log(rss / nobs) + 1.0))
