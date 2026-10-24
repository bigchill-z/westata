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
number once the intercept is present.

``auto_df`` does **not** change ``df_resid`` (both settings report 13); what it
changes is the *small-sample factor inside the clustered covariance*.  Its
correct setting was measured, not assumed, because an earlier version of this
module had it backwards:

* ``auto_df=True`` counts only the *reported* coefficients in that factor, i.e.
  ``K = 3``.  That is Stata's convention: on a 30-entity x 6-period panel,
  ``xtreg y x1 x2, fe vce(cluster id)`` reproduces linearmodels' clustered
  standard errors **to 9.4e-15** with ``auto_df=True``.
* ``auto_df=False`` also counts the 29 absorbed entity effects (``K = 32``) and
  therefore reproduces Stata's ``vce(cluster id) dfadj`` - an *undocumented*
  Stata option - with the same 2.1e-15 agreement.  It is 9.36% too large
  (``sqrt(177/148)``) against Stata's default.

westata therefore passes ``auto_df=True`` plus ``group_debias=True``, which is
Stata's ``G/(G-1) * (N-1)/(N-K)`` CR1 with Stata's ``K`` (see
``test_auto_df_on_reproduces_stata_clustered_standard_errors``).

Without a constant (``"y ~ x1 + x2 - 1"``) the ``+1`` disappears from both
spellings and both report 13 - i.e. ``N - K - n_entities``.  westata derives
the effect count from whether the fitted model actually carries an intercept,
so this case stays conformant instead of assuming a constant exists.

Under a clustered VCE Stata also reports ``e(df_r) = G - 1`` (29, not 148), and
that number - not ``N - K`` - is the reference distribution of the reported
t statistics.  westata recomputes it here for the same reason it recomputes the
rest of the ``e()`` block: so the p-values and confidence intervals cannot
silently disagree with Stata.

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
westata ``cov_type``         Stata
===========================  ===================================================
``None`` / ``"nonrobust"``   (no ``vce()``) -> ``cov_type="unadjusted"``
``"robust"`` / ``"HC1"``     ``vce(robust)`` -> **``vce(cluster panelvar)``**
``"cluster"``                ``vce(cluster clustvar)``
===========================  ===================================================

``[XT] xtreg`` states it outright: "Specifying ``vce(robust)`` is equivalent to
specifying ``vce(cluster panelvar)``".  Measured on a 30-entity x 6-period
panel, ``max|V_robust - V_cluster(id)| = 0.0`` for ``xtreg, fe`` (and 6.9e-18
for ``xtreg, re``), while ``areg y x1 x2, absorb(id) vce(robust)`` gives a
*different* matrix (constants' SE ``.0372797`` vs ``.0020300``, i.e. 18.4x).
So the HC1 mapping belongs to ``areg``/``FEOLS`` and **not** to ``xtreg``:
``cov_type="robust"`` here is Stata's clustered ``vce(robust)``, never a
heteroskedasticity-only sandwich.  ``HC1`` is kept as an accepted alias
because that is the spelling older westata code used, but it now names the
same covariance Stata's ``vce(robust)`` returns for this estimator.

Because the two are the same thing in Stata, ``xtreg`` has no way to ask for a
non-clustered heteroskedasticity-robust covariance at all; ``HC0``/``HC2``/
``HC3`` are rejected with :class:`UnsupportedError` rather than being folded
into a number Stata cannot produce.

Known deviations: ``e(df_m)`` and ``e(rank)`` under a clustered VCE
--------------------------------------------------------------------
Deliberate, and approved by the Lead: **westata does not reproduce Stata's
``e(df_m)``/``e(rank)`` when the covariance is clustered**, because Stata's own
``e()`` block is self-contradictory there.

Measured with StataNow 19.5 on 2026-10-24, 30 entities x 6 periods,
``xtset id t``::

    command                                  e(df_m) e(df_b) e(rank)  prints
    xtreg y x1,     fe vce(cluster id)          0      1       1      F(1, 29) = 211.29
    xtreg y x1 x2,  fe vce(cluster id)          1      2       2      F(2, 29) = 303.80
    xtreg y x1 x2 z, fe vce(cluster id)         2      3       3      F(3, 29) = 254.47

So Stata stores ``e(df_m) = slopes - 1`` and ``e(rank) = slopes`` - both defined
against the *free parameters of the clustered problem*, not against the model's
columns - while the F it prints and the ``e(p)`` it stores use
``F(e(df_b), e(df_r))``.  They disagree on every clustered ``xtreg, fe``: with
two slopes, ``e(df_m)`` says 1, ``e(rank)`` says 2, and the printed test is
``F(2, 29)``.  Stata keeps the unclustered values in ``e(df_r_mb) = 148`` and
``e(rank_mb) = 3``, which shows the shift is a convention and not an accident.

westata stores the parameter vector's own quantities: ``e(df_m)`` = the number
of slopes (2), ``e(rank)`` = ``len(e(b))`` = slopes + 1 (3, constant included),
plus ``e(df_b)`` = 2.  So ``e(df_m)`` is +1 and ``e(rank)`` is +1 against
Stata's clustered values, everything else matches:

=========================  =========  =========  =======
quantity                   Stata      westata    delta
=========================  =========  =========  =======
``e(df_m)`` (2 slopes)           1          2      +1
``e(rank)``  (2 slopes)          2          3      +1
``e(df_b)``                      2          2       0
``e(df_r)``                     29         29       0
``e(F)``                    303.80     303.80       0  (6.5e-16 relative)
=========================  =========  =========  =======

Copying Stata's ``e(df_m)`` would make
:mod:`westata.results.summary` print ``F(1, 29) = 303.80`` - a coefficient block
whose degrees of freedom contradict its own p-value.  A wrong *label* baked into
a downstream CI is the lesser evil only because it is visible; a wrong *number*
is not.  ``e(rank)`` stays at the honest rank of the fitted model for the same
class of reason: it is the size of ``e(b)``/``e(V)`` that
:class:`westata.results.StataResults` and the postestimation code rely on, and
shrinking it to Stata's clustered convention would quietly change unrelated
scalars that read it.  The real fix, if bit-for-bit ``e()`` equality is ever
wanted, is to store Stata's values and teach ``results/summary.py`` to label F
tests from ``e(df_b)`` - that needs a file outside this module's ownership.

Everything else about a clustered ``xtreg, fe`` *is* reproduced, including
``e(df_r) = G - 1``:
``xtreg y x1 x2, fe vce(cluster id)`` on that panel gives ``e(df_r) = 29`` with
``e(df_r_mb) = 148`` for the model-based value.

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
from scipy import stats

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

#: ``cov_type`` (lower-case) -> the covariance Stata actually produces here.
#:
#: ``xtreg``'s ``vce(robust)`` *is* ``vce(cluster panelvar)`` (``[XT] xtreg``,
#: verified: ``max|V_robust - V_cluster| = 0.0``), so both spellings map onto the
#: clustered backend estimator.  The heteroskedasticity-only HC1 sandwich that
#: ``areg``/``FEOLS`` use is a *different* matrix and must not be substituted
#: here: on a 30x6 panel its constant SE is 18.4x Stata's.
_BACKEND_COV_TYPES: dict[str, str] = {
    "nonrobust": "unadjusted",
    "unadjusted": "unadjusted",
    "hc1": "clustered",
    "robust": "clustered",
    "cluster": "clustered",
    "clustered": "clustered",
}

#: ``cov_type`` values Stata can express for ``xtreg`` but this adapter cannot.
_UNSUPPORTED_COV_TYPES: dict[str, str] = {
    "hc0": "HC0 has no xtreg counterpart (Stata's vce(robust) is a cluster VCE)",
    "hc2": "panel HC2 has no Stata xtreg counterpart",
    "hc3": "panel HC3 has no Stata xtreg counterpart",
    "hac": "vce(HAC) needs newey/xtscc, not xtreg",
    "twoway": "two-way clustering needs a pairs argument this adapter does not expose",
}

_CONSTANT_NAMES = frozenset({"intercept", "const", "_cons"})

#: ``cov_type`` values that request Stata's clustered ``vce(robust)``.
_ROBUST_COV_TYPES = frozenset({"robust", "hc1"})


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
            "xtreg family are: nonrobust, robust (Stata's vce(robust), i.e. "
            "vce(cluster panelvar)), cluster"
        ) from None


def _cov_label(cov_type: str | None) -> str:
    """Label recorded on the result, in westata's own vocabulary."""

    if cov_type is None:
        return "nonrobust"
    key = str(cov_type).strip().lower()
    if key in _ROBUST_COV_TYPES:
        return "robust"
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

    Known deviations: ``e(df_m)``, ``e(rank)`` and ``e(df_b)`` under clustering
    ---------------------------------------------------------------------------
    Deliberate, and approved by the Lead.  Stata's own ``e()`` block disagrees
    with itself here, so westata does not copy it.

    Measured with StataNow 19.5 on 2026-10-24, 30 entities x 6 periods,
    ``xtset id t``::

        command                                  e(df_m) e(df_b) e(rank)  prints
        xtreg y x1,     fe vce(cluster id)          0      1       1      F(1, 29) = 211.29
        xtreg y x1 x2,  fe vce(cluster id)          1      2       2      F(2, 29) = 303.80
        xtreg y x1 x2 z, fe vce(cluster id)         2      3       3      F(3, 29) = 254.47

    Stata's clustered ``xtreg, fe`` redefines both scalars relative to the
    *reported* model, not to the parameter vector:

    * ``e(df_m) = slopes - 1`` (the reported rank minus one),
    * ``e(rank)  = slopes``,

    while the F it prints and the ``e(p)`` it stores are
    ``F(e(df_b), e(df_r))``.  With two slopes, ``e(df_m)`` says 1, ``e(rank)``
    says 2, and the printed test is ``F(2, 29)``.  Stata also keeps the
    unclustered values in ``e(df_r_mb) = 148`` and ``e(rank_mb) = 3``, which is
    what shows the redefinition is deliberate: under clustering both scalars
    shift down by one because Stata counts the *free* parameters of the
    clustered problem (the slopes alone) rather than the model's columns.

    westata reports the parameter vector's own quantities - ``e(df_m)`` = the
    number of slopes (2), ``e(rank)`` = ``len(e(b))`` = slopes + 1 (3, the
    constant included) - plus ``e(df_b)`` = 2.  So against Stata's clustered
    values, ``e(df_m)`` is +1 and ``e(rank)`` is +1:

    =====================  =========  =========  =======
    quantity               Stata      westata    delta
    =====================  =========  =========  =======
    ``e(df_m)`` (2 slopes)       1          2      +1
    ``e(rank)``  (2 slopes)      2          3      +1
    ``e(df_b)``                  2          2       0
    ``e(df_r)``                 29         29       0
    ``e(F)``                 303.80     303.80       0  (6.5e-16 relative)
    =====================  =========  =========  =======

    Copying Stata's ``e(df_m)`` would make the summary read ``F(1, 29) = 303.80``
    - a degrees-of-freedom label contradicting its own p-value - so a visibly
    wrong *label* is the lesser evil than a wrong *number*.  ``e(rank)`` is kept
    as the honest rank of the fitted model for the same class of reason: it is
    what :class:`westata.results.StataResults` and the postestimation machinery
    use for the size of ``e(b)``/``e(V)``, and shrinking it to Stata's
    clustered-scalar convention would silently change unrelated scalars
    (``aic``/``bic``-style counts, ``/sigma`` handling) that read it.  Changing
    either number is therefore out of scope for a documentation fix; only the
    F statistic has to match, and it does.

    The real fix, if the Lead ever wants bit-for-bit ``e()`` equality, is to
    store Stata's values and teach :mod:`westata.results.summary` to label F
    tests from ``e(df_b)`` - that needs a file outside this module's ownership.

    Everything else about a clustered ``xtreg, fe`` is reproduced, including
    ``e(df_r) = G - 1`` (``29`` on that panel, with the model-based ``148`` in
    ``e(df_r_mb)``) and the ``N - rank`` denominator Stata uses for
    ``e(rmse)``/``e(r2_a)`` under clustering.
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
        model_df = self._stata_df_resid(nobs, rank, effect_df)
        df_b = rank - (1 if has_constant else 0)
        df_resid = model_df
        scalar_df = model_df
        if n_clust is not None:
            df_resid = self._stata_clustered_df_resid(n_clust, nobs, rank, model_df)
            scalar_df = self._stata_clustered_scalar_df(nobs, rank, model_df)

        e = self._stata_scalars(
            ref,
            nobs=nobs,
            resid=resid,
            df_model=df_model,
            df_resid=df_resid,
            scalar_df=scalar_df,
            df_b=df_b,
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
            # ``RandomEffects``/``BetweenOLS``/``FirstDifferenceOLS`` reject
            # ``auto_df`` outright, and their clustered covariance already
            # matches Stata without it.
            return {}
        return {
            # ``auto_df=True`` (the backend default) counts only the *reported*
            # coefficients in the clustered small-sample factor, i.e. Stata's
            # K.  Measured on 30 entities x 6 periods: auto_df=True reproduces
            # ``xtreg, fe vce(cluster id)`` to 9.4e-15, auto_df=False counts the
            # 29 absorbed effects too (K=32) and reproduces the undocumented
            # ``dfadj`` variant, i.e. 9.36% = sqrt(177/148) too large against
            # Stata's default.  Pin it explicitly rather than relying on the
            # backend's default, which has changed between releases.
            "auto_df": True,
        }

    def _resolve_cov(
        self, cov_type: str | None, cov_kwds: Mapping[str, Any] | None
    ) -> tuple[str, np.ndarray | None, int | None]:
        backend_cov = _normalize_cov_type(cov_type)
        kwds = {} if cov_kwds is None else dict(cov_kwds)
        wants_cluster = backend_cov == "clustered"
        key = "" if cov_type is None else str(cov_type).strip().lower()

        clusters = kwds.pop("clusters", kwds.pop("groups", None))
        if key in _ROBUST_COV_TYPES:
            # ``vce(robust)`` == ``vce(cluster panelvar)``: the panel variable is
            # the cluster variable, so callers must not pass their own labels --
            # doing so would silently give a different estimator than Stata's.
            if clusters is not None:
                raise ValueError(
                    "cov_type='robust' is Stata's vce(robust), which for xtreg "
                    "clusters on the panel variable; do not pass cov_kwds -- use "
                    "cov_type='cluster' with cov_kwds={'clusters': ...} to "
                    "cluster on a different variable"
                )
            if not wants_cluster:  # pragma: no cover - guarded by the mapping
                return backend_cov, None, None
            labels = self._data.index.get_level_values(self._entity).to_numpy()
            n_clust = int(pd.unique(labels).size)
            if n_clust < 2:
                raise ConsistencyError(
                    f"{self._stata_command}: Stata's vce(robust) clusters on the "
                    f"panel variable, so at least 2 panels are required; got "
                    f"{n_clust}"
                )
            return backend_cov, labels, n_clust

        if not wants_cluster:
            if kwds:
                raise ValueError(
                    "cov_kwds is only consumed for cov_type='cluster'; got "
                    f"unexpected key(s) {sorted(kwds)}"
                )
            return backend_cov, None, None

        if clusters is None:
            raise ValueError(
                "cov_type='cluster' requires cov_kwds['clusters']; pass the "
                "cluster labels as cov_kwds={'clusters': ...}"
            )
        if kwds:
            raise ValueError(
                "cov_kwds is only consumed for cov_type='cluster'; got "
                f"unexpected key(s) {sorted(kwds)}"
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

    def _stata_clustered_df_resid(
        self, n_clust: int, nobs: int, rank: int, df_resid: float
    ) -> float:
        """``e(df_r)`` under a clustered VCE: Stata reports ``G - 1``.

        Measured on 30 entities x 6 periods, ``xtreg y x1 x2, fe vce(cluster id)``
        reports ``e(df_r) = 29`` while the model-based value stays in
        ``e(df_r_mb) = 148``, and the reported t statistics use the clustered
        value.  Keeping ``N - K`` here would give Stata's standard errors with
        the wrong p-values and confidence intervals.
        """

        if n_clust < 2:  # pragma: no cover - guarded by _resolve_cov
            return df_resid
        return float(n_clust - 1)

    def _stata_clustered_scalar_df(self, nobs: int, rank: int, model_df: float) -> float:
        """Denominator Stata uses for ``e(rmse)``/``e(r2_a)`` under clustering.

        Measured: Stata divides by the *unadjusted* ``N - rank`` there
        (``rss/rmse^2 = 177.0000`` for ``xtreg y x1 x2, fe vce(robust)`` on
        30 x 6), not by the effects-adjusted ``N - K - n + 1 = 148`` it uses
        when the VCE is conventional and not by the clustered ``G - 1 = 29``.
        """

        return float(nobs - rank)

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
        df_b: float,
        scalar_df: float | None = None,
        rank: int,
        n_clust: int | None,
    ) -> dict[str, Any]:
        """``e()`` scalars, including Stata's clustered-VCE convention.

        Measured on 30 entities x 6 periods, ``xtreg y x1 x2, fe vce(robust)``
        reports ``e(df_r) = G - 1 = 29`` for the reference distribution but
        divides ``e(rmse)``/``e(r2_a)`` by a *model-based* ``N - rank = 177``
        (``rss/rmse^2 = 177.0000``; plain ``xtreg, fe`` uses the effects-adjusted
        ``df_r = 148``).  ``e(df_b)`` is the number of reported slopes, which
        ``xtreg, fe`` keeps separate from ``e(df_m)`` because the latter also
        counts the absorbed effects when the VCE is conventional.
        """

        scalar_df = (df_resid if scalar_df is None else scalar_df)
        rss = float(resid @ resid)
        r2 = float(getattr(ref, "rsquared", float("nan")))
        f_stat = self._stata_f(ref, df_model)
        r2_a = self._stata_r2_adjusted(r2, nobs, scalar_df)
        e: dict[str, Any] = {
            "N": nobs,
            "df_m": df_model,
            "df_r": df_resid,
            "df_b": float(df_b),
            "r2": r2,
            "r2_a": r2_a,
            "rmse": float(np.sqrt(rss / scalar_df)) if scalar_df > 0 else float("nan"),
            "F": f_stat,
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

    def _stata_clustered_df_resid(
        self, n_clust: int, nobs: int, rank: int, df_resid: float
    ) -> float:
        """``xtreg, re`` keeps ``N - K``: Stata reports no ``e(df_r)`` at all.

        ``xtreg, re`` is a GLS estimator whose inference is normal-based, so
        unlike ``xtreg, fe`` it has no ``e(df_r)`` to replace with ``G - 1``
        (measured: the key is absent from ``e()`` under every VCE).  Only the
        covariance changes when the VCE is clustered, which this class inherits.
        """

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
