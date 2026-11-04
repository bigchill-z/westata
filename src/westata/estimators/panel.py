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
from westata.estimators.panel_iv import _corr, _corr2 as _squared_correlation
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


#: ``e(title)`` per ``xtreg`` variant -- the banner each one prints. Measured from
#: StataNow 19.5 by running each estimator and reading the first line of its output;
#: ``e(title)`` is not retrievable through the macro interface for ``fe``/``re``, so the
#: printed banner is the authority.
#:
#: ``xtreg, fd`` is deliberately ABSENT: it could not be run on this installation (the
#: command errors out before producing a table), so its banner is unknown. An unknown
#: variant falls back to the generic header rather than to a guessed string.
def _obs_per_group(data: Any, entity: str) -> dict[str, float]:
    """``min``/``avg``/``max`` observations per entity, for the ``xtreg`` header.

    Returns an empty dict when the entity column is unknown, so the header omits the
    column instead of printing made-up sizes.
    """

    if not entity:
        return {}
    # The entity lives in the INDEX after ``_panel_frame`` converts the frame to a
    # MultiIndex, so grouping on a COLUMN finds nothing and silently returns no counts.
    try:
        if entity in getattr(data.index, "names", []):
            sizes = data.groupby(level=entity, observed=True).size()
        elif entity in getattr(data, "columns", []):
            sizes = data.groupby(entity, observed=True).size()
        else:
            return {}
    except Exception:  # pragma: no cover - defensive
        return {}
    if sizes.empty:
        return {}
    return {
        "min": float(sizes.min()),
        "avg": float(sizes.mean()),
        "max": float(sizes.max()),
    }


_XTREG_TITLES = {
    "xtreg, fe": "Fixed-effects (within) regression",
    "xtreg, re": "Random-effects GLS regression",
    "xtreg, be": "Between regression (regression on group means)",
}


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

    #: Whether the backend's ``rsquared`` is the *within* R-squared, so that it
    #: can also be stored under Stata's ``e(r2_w)``.  True only for the within
    #: estimator; see :meth:`_stata_scalars` for why this is not generalised.
    _within_r2_is_backend_r2 = True

    #: ``xtreg, fe`` is the only variant whose Stata fixture stores ``e(tss)``.
    _stores_totals = True

    # ``xtreg_fe.json`` has ll_0 and the absorbed-effects block.
    _stores_ll_0 = True
    _stores_absorbed = True

    #: Whether the estimator regresses on one row per entity (``xtreg, be``),
    #: so that ``e(N)`` counts entities and the response is already a group mean.
    _estimates_on_entity_means = False

    #: Whether Stata stores ``e(mss)`` WITHOUT ``e(tss)`` -- true only for
    #: ``xtreg, be``, whose ``xtreg.sthlp:1014-1067`` block has ``mss``/``rss`` and no
    #: ``tss``.  This is NOT "does the estimator store mss": ``xtreg, fe`` stores
    #: ``mss`` too (``:1068-1144``), but together with ``tss``, so it is covered by
    #: ``_stores_totals`` instead.  Line 1089 reads ``if self._stores_totals or
    #: self._stores_mss``, which is why both routes exist.
    _stores_mss = False

    #: Whether the estimator quasi-demeans, i.e. whether ``e(theta)`` applies.
    #: True only for ``xtreg, re`` (``xtreg_re.ado:265-266``).
    _stores_theta = False

    #: Whether Stata stores the ``sigma_u``/``sigma_e``/``sigma``/``rho``/``corr``
    #: family.  True for ``xtreg, fe``/``re``; FALSE for ``xtreg, be`` (no
    #: decomposition -- ``xtreg_be.ado`` only uses the ``corr`` COMMAND) and for
    #: ``xtreg, fd`` (differencing removes the entity effect).
    _stores_sigma = True

    #: Whether Stata reports a Wald ``e(chi2)`` instead of ``e(F)`` -- true for
    #: ``xtreg, re``, whose fixture has ``chi2``/``p`` and no ``F``.
    _stores_wald_chi2 = False

    def __init__(
        self,
        formula: str,
        data: pd.DataFrame,
        *,
        entity: str,
        time: str,
        weights: Any = None,
        weight_type: str | None = None,
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
        # ``xtreg, fe [aweight=...]`` is the combination this paper uses. Stored on the
        # instance and applied at estimation time, matching ``FEOLS``/``Areg``; the
        # weights are aligned to the estimation sample AFTER listwise deletion, which is
        # why the raw argument is kept rather than an already-filtered array.
        self._weights_argument = weights
        self._weight_type = weight_type
        if weights is not None and weight_type is None:
            raise ValueError(
                "weights were supplied without weight_type; Stata never guesses the "
                "weight type ([U] 20.24). Pass weight_type='aweight' (or 'fweight', "
                "'pweight', 'iweight')."
            )
        self._data = _panel_frame(data, entity=entity, time=time, command=self._stata_command)
        # Remember the full sample so that user-supplied, observation-aligned
        # arguments (cluster labels) can be mapped onto the estimation sample
        # *after* listwise deletion.
        self._full_index = self._data.index
        #: The index BEFORE _design drops rows. _weights_argument describes this
        #: one, and _full_index is overwritten as the sample shrinks, so the original
        #: has to be kept or positional re-labelling breaks on the second pass.
        self._weight_source_index = self._data.index
        self._kept_index = self._data.index

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        entity: str,
        time: str,
        subset: Any = None,
        weights: Any = None,
        weight_type: str | None = None,
    ) -> "PanelOLS":
        """Build a model from a formula and a panel frame.

        A long frame carrying the ``entity``/``time`` columns is converted with
        ``set_index([entity, time])``; a frame that already carries the
        two-level ``MultiIndex`` is used as is.

        ``subset=`` restricts the estimation sample, matching ``OLS``/``Logit``/
        ``Poisson`` and Stata's ``if``/``in``. The previous workaround -- pre-filtering the
        frame and calling ``reset_index`` -- gives identical numbers (verified), so this is
        a consistency fix rather than a correctness one: one spelling for the same
        operation across estimators.

        ``weights=``/``weight_type=`` reproduce ``xtreg, fe [aweight=...]``. They are
        accepted here because ``xtreg`` is the command ``PanelOLS`` documents itself as
        replacing: leaving them out forced callers onto ``FEOLS`` to get the same
        estimator.
        """

        if subset is not None:
            data = data.loc[subset]
        return cls(
            formula,
            data,
            entity=entity,
            time=time,
            weights=weights,
            weight_type=weight_type,
        )

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
        # The response on the estimator's own scale, for e(ll_0).
        self._estimation_response = np.asarray(
            response.to_numpy(dtype=float), dtype=float
        )

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
            response=response,
            fitted=fitted,
            design=(X, param_names),
            params=params,
        )

        results = StataResults(
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
        #: The estimation design, retained so ``predict`` can run.
        #: ``xtreg_postestimation.sthlp:128`` defines ``xb`` as ``a + xb`` -- the
        #: linear prediction WITHOUT the panel effect ``u_i`` -- and that is the
        #: default; ``xbu`` is the variant that adds ``u_i``.  The default is
        #: therefore not ambiguous, so predict should offer it rather than refuse.
        #:
        #: The column count must match ``param_names``: ``xtreg, fd`` drops the
        #: intercept when differencing, so it has one coefficient but the built
        #: ``X`` still carries two columns.  Retaining that mismatch made every
        #: predict type fail with a matmul dimension error, which the registry
        #: audit surfaced as a "registered but broken" entry.
        design = np.asarray(X, dtype=float)
        if design.ndim == 2 and design.shape[1] != len(param_names):
            # Column-aligned truncation: the dropped column is the constant, which
            # sits last in this package's designs (Stata's e(b) order).
            if design.shape[1] > len(param_names):
                design = design[:, : len(param_names)]
        results.design_matrix = (design, list(param_names))
        #: Entity labels aligned to the estimation sample, so ``predict`` can
        #: separate ``u_i`` from ``e_it`` (``xtreg_postestimation.sthlp:127-134``).
        try:
            results.panel_entity = (
                self._data.index.get_level_values(self._entity).to_numpy()
            )
        except (KeyError, ValueError):  # pragma: no cover - unnamed index
            results.panel_entity = None
        #: The estimation-sample frame, retained so ``estat summarize`` can run.
        #: ``xtreg_estat.ado:18,36,48`` forwards unrecognised subcommands to
        #: ``estat_default``, whose ``summarize`` is therefore available after
        #: ``xtreg``; refusing it here was a gap, not a limitation.
        try:
            flat = self._data.reset_index()
            results.model_data = flat
        except (AttributeError, ValueError):  # pragma: no cover
            flat = None
        #: ``e(depvar)``: the response name, which ``estat summarize`` lists first.
        try:
            lhs = str(getattr(self, "_formula", "")).split("~", 1)[0].strip()
            response = lhs.split()[0] if lhs else None
            if response and flat is not None and response in flat.columns:
                results.response_name = response
                results.e["depvar"] = response
        except (AttributeError, IndexError):  # pragma: no cover
            pass
        return results

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
        # A MISSING WEIGHT drops the row, exactly like a missing regressor: Stata's
        # ``[aweight=]`` marks an observation "not to be used" when the weight is missing,
        # so the estimation sample is the intersection. The AJR data has 95 missing weights
        # out of 209, so ignoring this both shrank N wrongly and then tripped the
        # "weights contain missing values" check.
        if self._weights_argument is not None:
            aligned = self._raw_weight_values().reindex(kept)
            mask = aligned.notna().to_numpy()
            kept = kept[mask]
            # ``y_frame``/``x_frame`` were built on the pre-weight sample, so they have to
            # be cut by the SAME mask -- shrinking only ``kept`` left a 169-row response
            # against a 109-row index.
            y_frame = y_frame.loc[kept]
            x_frame = x_frame.loc[kept]
            if len(kept) == 0:
                raise ConsistencyError(
                    f"{self._stata_command}: no observations remain after dropping "
                    "missing weights"
                )
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
        #: The untransformed response over the estimation sample.  Retained so
        #: that ``xtreg, be`` can report ``e(r2_w)``, which Stata computes on the
        #: long panel even though the between regression runs on group means.
        self._long_response: np.ndarray | None = np.asarray(
            y.to_numpy(dtype=float), dtype=float
        )
        #: The response the estimator actually regresses on -- entity means for
        #: ``be``, differences for ``fd``, levels otherwise.  ``e(ll_0)`` is the
        #: intercept-only likelihood on this scale, so it must match ``e(N)``.
        self._estimation_response: np.ndarray | None = None
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

        weights = self._aligned_weights(y.index)
        return _BackendPanelOLS(
            y.to_frame(),
            X,
            weights=None if weights is None else pd.Series(weights, index=y.index),
            entity_effects=bool(entity_effects),
            time_effects=bool(time_effects),
        )

    def _raw_weight_values(self) -> pd.Series:
        """The weights as a Series indexed like ``self._full_index``, NaNs included.

        The caller hands over a Series carrying the LONG frame's index; the panel index is
        a ``(entity, time)`` MultiIndex, so the two cannot be joined by label. Row order is
        the contract -- "one weight per row of data" -- so the values are re-labelled
        positionally.
        """

        raw = self._weights_argument
        source = self._weight_source_index
        if isinstance(raw, str):
            raise ConsistencyError(
                f"{self._stata_command}: pass the weight VALUES, not the column name "
                f"{raw!r}; column names cannot be aligned to the original rows once the "
                "panel index has been built"
            )
        if isinstance(raw, pd.Series):
            if raw.index.equals(source):
                return raw.astype(float)
            return pd.Series(np.asarray(raw.to_numpy(), dtype=float), index=source)
        array = np.asarray(raw, dtype=float).reshape(-1)
        if array.size != len(source):
            raise ConsistencyError(
                f"{self._stata_command}: weights have {array.size} entries but the data "
                f"have {len(source)} rows"
            )
        return pd.Series(array, index=source)

    def _aligned_weights(self, index: pd.Index) -> np.ndarray | None:
        """Weights for the ESTIMATION sample, or ``None`` when unweighted.

        The argument the caller passed describes ``self._full_index``; ``_design`` has
        already dropped rows with missing values, so it is reindexed onto the surviving
        rows here. Skipping this step would either misalign silently (a longer array would
        raise, but a same-length one from a different order would not) or, worse, hand the
        backend weights in frame order while it expects sample order.
        """

        if self._weights_argument is None:
            return None

        values = np.asarray(self._raw_weight_values().reindex(index), dtype=float)
        if np.any(np.isnan(values)):
            raise ConsistencyError(
                f"{self._stata_command}: weights are missing on rows of the estimation "
                "sample; a missing weight should have dropped the row"
            )

        if not np.all(np.isfinite(values)):
            raise ConsistencyError(
                f"{self._stata_command}: weights contain missing or infinite values"
            )
        if np.any(values < 0):
            raise ConsistencyError(f"{self._stata_command}: weights must be non-negative")
        return values

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

        # All three spellings, each POPPED so the "unexpected key(s)" check below still
        # catches a real typo. ``groups`` is the OLS/ML documentation, ``clusters`` reads
        # naturally from Stata's ``vce(cluster clustvar)``, ``cluster`` is what ``hdfe``
        # used -- and every other estimator accepts all three now.
        #
        # The loop variable must NOT be called ``key``: the enclosing method already uses
        # that name for the normalised cov_type, and reusing it silently replaced
        # ``"robust"``/``"cluster"`` with ``"cluster"`` so every robust fit fell through to
        # the wrong branch. Caught by test_robust_is_statas_clustered_vce_not_hc1.
        clusters = None
        for _label_key in ("clusters", "groups", "cluster"):
            candidate = kwds.pop(_label_key, None)
            if candidate is not None and clusters is None:
                clusters = candidate
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
        response: pd.Series | None = None,
        fitted: np.ndarray | None = None,
        design: tuple[pd.DataFrame, list[str]] | None = None,
        params: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """``e()`` scalars, including Stata's clustered-VCE convention.

        Measured on 30 entities x 6 periods, ``xtreg y x1 x2, fe vce(robust)``
        reports ``e(df_r) = G - 1 = 29`` for the reference distribution but
        divides ``e(rmse)``/``e(r2_a)`` by a *model-based* ``N - rank = 177``
        (``rss/rmse^2 = 177.0000``; plain ``xtreg, fe`` uses the effects-adjusted
        ``df_r = 148``).  ``e(df_b)`` is the number of reported slopes, which
        ``xtreg, fe`` keeps separate from ``e(df_m)`` because the latter also
        counts the absorbed effects when the VCE is conventional.

        ``response``/``fitted`` are the estimation-sample response and fitted
        values; when supplied they enable the R-squared decomposition
        (``r2_w``/``r2_b``/``r2_o``), the sums of squares and the variance
        components that Stata's ``xtreg`` stores.
        """

        scalar_df = (df_resid if scalar_df is None else scalar_df)
        rss = float(resid @ resid)
        r2 = float(getattr(ref, "rsquared", float("nan")))
        f_stat = self._stata_f(ref, df_model)
        r2_a = self._stata_r2_adjusted(r2, nobs, scalar_df)
        # Stata's spellings for quantities westata already computed:
        #   e(N_g)     == the entity count (was ``n_entities``)
        # A Stata-porting script reading ``e(N_g)`` previously got ``None``
        # with no error, which is the failure mode this removes.
        #
        # ``e(r2_w)`` is stored ONLY for the within estimator, and only because
        # the backend's ``rsquared`` there *is* the within R-squared -- verified
        # independently on the balanced test panel (0.838219091483202 against a
        # hand-computed within R2 of 0.838219091483203) and against
        # ``xtreg_fe.json`` where Stata's ``e(r2) == e(r2_w) ==
        # 0.9299367179801631``.  It is deliberately NOT aliased for ``re``/``be``:
        # ``xtreg_re.json`` stores no ``e(r2)`` at all and ``xtreg_be.json``
        # stores ``r2 == r2_b = 0.28158005317996504`` while its ``r2_w`` is
        # 0.7540434430300107.  A blanket alias would report the wrong number
        # under Stata's name, which is worse than leaving the key absent.
        # See retest/gaps/linear_iv_ts_gaps.md §1.6.
        n_entities = int(
            self._data.index.get_level_values(self._entity).nunique()
        )
        rmse = float(np.sqrt(rss / scalar_df)) if scalar_df > 0 else float("nan")
        e: dict[str, Any] = {
            "N": nobs,
            "df_m": df_model,
            "df_r": df_resid,
            "df_b": float(df_b),
            "r2": r2,
            "r2_a": r2_a,
            "rmse": rmse,
            "F": f_stat,
            "ll": _gaussian_loglik(rss, nobs),
            "rank": rank,
            "n_entities": n_entities,
            "N_g": n_entities,
            # ``e(title)`` and ``e(group)``: what Stata prints as the banner and as the
            # variable it grouped on. Measured from ``xtreg y x1, fe``:
            #
            #   Fixed-effects (within) regression      Number of obs     =        180
            #   Group variable: id                     Number of groups  =         30
            #
            # Neither was stored, so the header fell back to a generic string and the
            # group column showed "." instead of the variable the caller passed.
            "title": _XTREG_TITLES.get(self._stata_command, ""),
            "group": str(getattr(self, "_entity", "") or ""),
            # ``Obs per group``: Stata prints min / avg / max. Computed from the estimation
            # frame because they describe the SAMPLE, not the estimator; ``avg`` keeps one
            # decimal even when it is whole ("6.0" next to "min = 6").
            "_obs_per_group": _obs_per_group(
                self._data, str(getattr(self, "_entity", "") or "")
            ),
            "rss": rss,
        }
        if self._within_r2_is_backend_r2:
            e["r2_w"] = r2
        if self._stores_wald_chi2:
            # ``xtreg, re`` reports a Wald chi-square on the reported slopes
            # rather than an F: ``xtreg_re.json`` has e(chi2) = 1965.988... with
            # e(df_m) = 2 and no e(F).  ``_stata_f`` already computes the same
            # Wald statistic divided by df_m, so chi2 is that times df_m.
            if np.isfinite(f_stat) and df_model > 0:
                chi2 = float(f_stat * df_model)
                e["chi2"] = chi2
                e["chi2type"] = "Wald"
                e["p"] = float(stats.chi2.sf(chi2, df_model))
        if n_clust is not None:
            e["N_clust"] = n_clust

        # e(ll_0): the Gaussian likelihood of the intercept-only model, using
        # the centred total.  Same construction as westata.OLS, which was
        # verified against StataNow 19.5 for all four weight types.  Stata's
        # xtreg stores it for both fe and be (xtreg_fe.json: -313.83365173076487,
        # xtreg_be.json has one too); ``re`` does not.
        if self._stores_ll_0 and nobs > 1:
            # ``e(ll_0)`` is the Gaussian likelihood of the intercept-only model,
            # using the centred total of the ESTIMATION sample.  Same
            # construction as westata.OLS (verified there against StataNow 19.5
            # for every weight type).  Stata stores it for ``fe`` and ``be``
            # but not ``re``.
            #
            # The estimation sample differs by estimator: ``be`` fits on entity
            # means, so its response is shorter than the long panel.  Rather
            # than guess, take whichever response matches ``nobs``.
            candidates = [
                getattr(self, "_estimation_response", None),
                self._long_response,
            ]
            centred_total = float("nan")
            for candidate in candidates:
                if candidate is None:
                    continue
                values = np.asarray(candidate, dtype=float).reshape(-1)
                if values.size != nobs:
                    continue
                total = float(np.sum((values - values.mean()) ** 2))
                if total > 0:
                    centred_total = total
                    break
            if np.isfinite(centred_total):
                e["ll_0"] = float(
                    -0.5
                    * nobs
                    * (
                        np.log(2 * np.pi)
                        + np.log(centred_total / nobs)
                        + 1.0
                    )
                )
        # e(sum_w): the raw weight sum.  With no weights it equals N.
        e["sum_w"] = float(nobs)
        # e(p): the model test's p-value.  For the F-based estimators it is the
        # F upper tail; ``re`` already set it from its Wald chi-square above.
        if "p" not in e and np.isfinite(f_stat) and df_model > 0:
            e["p"] = float(stats.f.sf(f_stat, df_model, df_resid))
        # e(rc): Stata's return code; 0 means the fit succeeded.
        e["rc"] = 0.0

        # --- R-squared decomposition, sums of squares, variance components ---
        # Stata's xtreg stores all of these (xtreg_fe.json has r2_w/r2_b/r2_o/
        # r2_c/sigma_u/sigma_e/rho/corr/Tbar/g_*/k_absorb/df_a/F_f; xtreg_re.json
        # has theta as well).  They are derivable from the estimation sample, so
        # nothing new is estimated -- only the decomposition is written out.
        if response is not None and fitted is not None:
            e.update(
                self._panel_decomposition(
                    response=response,
                    fitted=np.asarray(fitted, dtype=float),
                    resid=np.asarray(resid, dtype=float),
                    nobs=nobs,
                    n_entities=n_entities,
                    scalar_df=scalar_df,
                    design=design,
                    params=params,
                )
            )
        return e

    def _panel_decomposition(
        self,
        *,
        response: pd.Series,
        fitted: np.ndarray,
        resid: np.ndarray,
        nobs: int,
        n_entities: int,
        scalar_df: float,
        design: tuple[pd.DataFrame, list[str]] | None = None,
        params: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Stata's ``xtreg`` R-squared decomposition and variance components.

        Read from ``xtreg_fe.ado`` (StataNow 19.5, version 2.0.2, lines
        358-430), not inferred from output:

        .. code-block:: stata

            quietly _predict double `XB' if `touse', xb
            quietly gen double `U' = `y' - `XB'        // LEVEL residual
            `BY' replace `U' = cond(_n==_N,sum(`U')/`Ti',.)   // panel mean
            quietly summ `U'
            ereturn scalar sigma_u = r(sd)             // sd of the panel means
            ereturn scalar ui      = r(sd)             // same value
            `BY' replace `U' = `U'[_N]                 // expand back
            quietly corr `XB' `U' if `touse'
            ereturn scalar corr = r(rho)               // corr(Xb, u_i)
            quietly corr `y' `XB' if `touse'
            ereturn scalar r2_o = r(rho)^2
            `BY' replace `XB' = cond(_n==_N,sum(`XB')/`Ti',.)  // panel mean
            ... corr(y_mean, XB_mean)
            ereturn scalar r2_b = r(rho)^2
            ereturn scalar sigma_e = sqrt(e(rss)/e(df_r))
            ereturn scalar sigma   = sqrt(e(sigma_u)^2 + e(sigma_e)^2)

        Two consequences worth stating, because both are easy to get wrong:

        * ``sigma_u`` is the **standard deviation of the panel means of the
          level residual** ``y - Xb``, not a Swamy-Arora expression in the
          within residual.  The within residual has entity means of exactly
          zero by construction, so reusing it makes ``sigma_u`` collapse to 0
          and ``rho`` to 0 -- a silent, plausible-looking wrong answer.
        * ``r2_b`` uses the entity means of ``y`` and ``Xb``, while ``r2_o``
          uses the raw series; ``r2_w`` uses the demeaned series.

        Every key is omitted when its inputs are degenerate rather than filled
        with a placeholder.
        """

        y = np.asarray(response.to_numpy(dtype=float), dtype=float).reshape(-1)
        if y.size != nobs:
            # The response could not be aligned with the residuals (e.g. a
            # transformed estimator); leave the block out rather than guess.
            return {}
        fitted = np.asarray(fitted, dtype=float).reshape(-1)
        if fitted.size != nobs:
            return {}

        entity_labels = self._data.index.get_level_values(self._entity)
        if entity_labels.size == nobs:
            codes = pd.factorize(entity_labels.to_numpy())[0]
        elif self._estimates_on_entity_means:
            # ``xtreg, be`` regresses on one row per entity, so the estimation
            # sample is the entity index rather than the full long panel.
            unique_labels = pd.unique(entity_labels.to_numpy())
            if unique_labels.size != nobs:
                return self._panel_structure_only(n_entities)
            codes = pd.factorize(unique_labels)[0]
        else:
            # ``xtreg, fd`` differences within entity, so the estimation sample
            # is shorter than the panel.  The R-squared decomposition is not
            # defined on the differenced response, but the panel-structure
            # block (Tbar / g_* / Tcon) still describes the data honestly.
            structure = self._panel_structure_only(n_entities)
            return structure

        sizes = np.bincount(codes, minlength=n_entities).astype(float)
        # ``e(Tbar)``/``e(g_*)`` describe the LONG panel's group sizes, not the
        # entity-level estimation sample: Stata's xtreg_be.json reports N=180
        # observations, N_g=30 entities, Tbar=6.  When the estimator regresses
        # on entity means the ``codes`` above are one-per-entity, so the panel
        # sizes must come from the full long data instead.
        if self._estimates_on_entity_means:
            long_labels = self._data.index.get_level_values(self._entity).to_numpy()
            long_codes = pd.factorize(long_labels)[0]
            long_sizes = np.bincount(long_codes).astype(float)
            panel_sizes = long_sizes
            long_nobs = float(long_labels.size)
        else:
            panel_sizes = sizes
            long_nobs = float(nobs)
        n_panels = int(panel_sizes.size)
        t_bar = long_nobs / float(n_panels) if n_panels else float("nan")

        def _entity_means(values: np.ndarray) -> np.ndarray:
            totals = np.bincount(codes, weights=values, minlength=n_entities)
            with np.errstate(divide="ignore", invalid="ignore"):
                return totals / sizes

        y_bar = _entity_means(y)
        f_bar = _entity_means(fitted)

        out: dict[str, Any] = {
            "Tbar": t_bar,
            "g_min": int(panel_sizes.min()),
            "g_avg": float(panel_sizes.mean()),
            "g_max": int(panel_sizes.max()),
            "Tcon": int(np.all(panel_sizes == panel_sizes[0])),
        }
        #: Whether every entity contributes the same number of periods.  Stata's
        #: ``e(Tcon)``; it decides whether ``theta`` is a single number or a
        #: distribution (``xtreg_re.ado:368-379``).
        t_con = bool(out["Tcon"])

        # Sums of squares, so that rss + mss = tss and the within/between
        # totals are visible.  Stata records these per estimator, and the
        # fixtures disagree: xtreg_fe.json has rss/mss/tss and
        # tss_within-family keys, xtreg_be.json has rss/mss but NO tss, and
        # xtreg_re.json has none of them.  So the totals are stored only where
        # Stata stores them, rather than uniformly.
        rss = float(resid @ resid)
        if self._stores_totals or self._stores_mss:
            tss = float(np.sum((y - y.mean()) ** 2))
            tss_within = float(np.sum((y - y_bar[codes]) ** 2))
            # Stata's panel fits mix TWO scales, and it is not an accident:
            #   e(tss) is the raw total sum of squares,
            #   e(rss) and e(mss) are on the WITHIN scale.
            # Measured on a 60x5 frame: tss 604.64155, rss 216.16809, mss 78.087606 -- and
            # mss + rss = 294.2557 = the within tss, NOT e(tss). Computing mss as
            # tss - rss instead gave 388.47345, which no Stata run produces.
            if tss_within > 0 and abs(1.0 - rss / tss_within) < 1.0:
                out["mss"] = tss_within - rss
            else:
                out["mss"] = tss - rss
            if self._stores_totals:
                tss_between = float(np.sum((y_bar - y.mean()) ** 2))
                out["tss"] = tss
                out["tss_within"] = tss_within
                out["tss_between"] = tss_between
                if tss_within > 0:
                    out["r2_within"] = 1.0 - rss / tss_within

        r2_w = _squared_correlation(fitted - f_bar[codes], y - y_bar[codes])
        # Stata's r2_b and r2_o are squared correlations involving the linear predictor on
        # the ORIGINAL scale (its entity means, and y itself). ``fitted`` reaches here as
        # ``response - resid``, which is NOT that predictor for the WITHIN estimator: its
        # backend residuals are already group-demeaned, so the subtraction mixes scales and
        # reports r2_b = 1.0000 / r2_o = .9638 where Stata reports .6650 / .7135 (measured
        # on lin_1992.dta; C12-01 of the 0.0.6 gap report).
        #
        # Recomputing the level predictor from the design leaves r2_w untouched -- it is
        # defined on deviations and is correct either way -- and leaves other estimators
        # alone, because there ``X @ params`` equals ``fitted``.
        _level_fitted = fitted
        if design is not None and params is not None:
            _matrix = np.asarray(design[0], dtype=float)
            _params = np.asarray(params, dtype=float).reshape(-1)
            if _matrix.ndim == 2 and _matrix.shape[0] == fitted.size:
                if _matrix.shape[1] == _params.size:
                    _level_fitted = _matrix @ _params
                elif _matrix.shape[1] - 1 == _params.size:
                    # The design carries an intercept column the estimator absorbed; the
                    # constant does not shift a correlation, so dropping it is harmless.
                    _level_fitted = _matrix[:, 1:] @ _params
        if _level_fitted.size != fitted.size or not np.all(np.isfinite(_level_fitted)):
            _level_fitted = fitted

        r2_b = _squared_correlation(_entity_means(_level_fitted), y_bar)
        r2_o = _squared_correlation(_level_fitted, y)
        for key, value in (("r2_w", r2_w), ("r2_b", r2_b), ("r2_o", r2_o)):
            if np.isfinite(value):
                out[key] = float(value)

        # ``xtreg, be`` estimates on entity means, so the r2_w above would be
        # degenerate (no within variation).  Stata still reports e(r2_w) for
        # ``be`` -- xtreg_be.json has 0.7540434430300107 alongside
        # e(r2) == e(r2_b) == 0.28158005317996504 -- describing how the *level*
        # fit explains the within variation.  That needs the long panel.
        if (
            self._estimates_on_entity_means
            and design is not None
            and params is not None
            and self._long_response is not None
        ):
            design_frame, _ = design
            matrix = np.asarray(design_frame.to_numpy(dtype=float), dtype=float)
            y_long = self._long_response
            if matrix.shape[0] == y_long.size:
                level_pred = matrix @ np.asarray(params, dtype=float)
                y_long_bar = _panel_means(y_long, long_codes, n_panels)
                pred_bar = _panel_means(level_pred, long_codes, n_panels)
                within_r2 = _squared_correlation(
                    level_pred - pred_bar[long_codes],
                    y_long - y_long_bar[long_codes],
                )
                if np.isfinite(within_r2):
                    out["r2_w"] = float(within_r2)

        # Variance components, following xtreg_fe.ado:358-430 exactly.
        #
        # Stata's ``XB`` is ``predict, xb`` -- the LEVEL linear predictor
        # ``b0 + b1*x1 + ...``, which does NOT include the entity effect.  The
        # ``fittedvalues`` westata carries for the within estimator is instead
        # ``y - within_resid``, i.e. the *demeaned* fit, whose entity means are
        # zero by construction; using it would make sigma_u and rho collapse to
        # 0 (a silent, plausible-looking wrong answer).  So the level predictor
        # is rebuilt from the design matrix and the reported coefficients.
        level_fitted = fitted
        if design is not None and params is not None:
            design_frame, design_names = design
            matrix = np.asarray(design_frame.to_numpy(dtype=float), dtype=float)
            if matrix.shape[0] == nobs and matrix.shape[1] == len(params):
                level_fitted = matrix @ np.asarray(params, dtype=float)
            elif design_names != []:  # pragma: no cover - defensive
                level_fitted = fitted

        level_residual = y - level_fitted
        u_panel = _entity_means(level_residual)

        # e(sigma_u) = r(sd) of the panel means.  ``summ`` reports the
        # df-adjusted standard deviation, so ddof=1 is deliberate.
        #
        # The whole sigma/rho/corr family is NOT universal.  ``xtreg_be.ado``
        # never sets them: its only uses of ``corr``/``r(rho)`` are the CORR
        # COMMAND, used to build ``r2_o`` and ``r2_w`` (``xtreg_be.ado:148-166``),
        # not e() keys.  The between estimator runs OLS on the entity means, so
        # there is no decomposed error structure to report.
        finite = np.isfinite(u_panel)
        if self._stores_sigma and finite.sum() > 1:
            deviations = u_panel[finite] - u_panel[finite].mean()
            sigma_u = float(np.sqrt(np.sum(deviations**2) / (finite.sum() - 1)))
        else:
            sigma_u = float("nan")

        # e(corr) = corr(Xb, u_i), the correlation Stata prints beside rho.
        if self._stores_sigma:
            correlation = _corr(level_fitted, u_panel[codes])
            if np.isfinite(correlation):
                out["corr"] = float(correlation)

        # e(sigma_e) = sqrt(e(rss)/e(df_r)) where e(df_r) is the reference df.
        sigma_e = (
            float(np.sqrt(rss / scalar_df))
            if self._stores_sigma and scalar_df and scalar_df > 0
            else float("nan")
        )

        if np.isfinite(sigma_u):
            out["sigma_u"] = sigma_u
            out["ui"] = sigma_u  # Stata stores both names with the same value
        if np.isfinite(sigma_e):
            out["sigma_e"] = sigma_e
        if np.isfinite(sigma_u) and np.isfinite(sigma_e):
            total = sigma_u**2 + sigma_e**2
            out["sigma"] = float(np.sqrt(total))
            if total > 0:
                out["rho"] = float(sigma_u**2 / total)

            # e(theta): the random-effects quasi-demeaning factor, from
            # ``xtreg_re.ado:265-266``:
            #     theta = 1 - s_e / sqrt(T * s_u^2 + s_e^2)
            # Verified digit for digit against ``xtreg_re.json``:
            # 1 - 0.4038449889037476/sqrt(6*0.8551376831355879^2
            #     + 0.4038449889037476^2) = 0.8106881439929379, which is the
            # fixture's e(theta).
            #
            # Only the random-effects estimator quasi-demeans, so the key is
            # stored only there (`xtreg_fe.json` has no e(theta)).
            if self._stores_theta and t_bar > 0:
                # ``xtreg_re.ado:368-379`` branches on ``e(Tcon)``:
                #
                #   if e(Tcon) { est scalar theta = <thetav> }
                #   else {
                #       by ivar: replace thetav = . if _n != _N
                #       summ thetav, d
                #       est scalar thta_min/p5/p50/p95/max
                #   }
                #
                # So the two forms are MUTUALLY EXCLUSIVE.  A balanced panel gets
                # the single ``theta``; an UNBALANCED one gets the five
                # percentiles of the per-entity theta -- because the group sizes
                # differ, there is no single theta to report.
                if t_con:
                    theta = 1.0 - sigma_e / float(
                        np.sqrt(t_bar * sigma_u**2 + sigma_e**2)
                    )
                    out["theta"] = theta
                else:
                    per_entity_theta = 1.0 - sigma_e / np.sqrt(
                        panel_sizes * sigma_u**2 + sigma_e**2
                    )
                    out["thta_min"] = float(np.min(per_entity_theta))
                    out["thta_5"] = float(
                        np.percentile(per_entity_theta, 5)
                    )
                    out["thta_50"] = float(
                        np.percentile(per_entity_theta, 50)
                    )
                    out["thta_95"] = float(
                        np.percentile(per_entity_theta, 95)
                    )
                    out["thta_max"] = float(np.max(per_entity_theta))

        # --- the absorbed-effects block (``xtreg, fe`` only) ----------------
        if self._stores_absorbed:
            # ``xtreg_fe.ado:505-524`` computes all three from ONE auxiliary
            # regression, not from the entity effects:
            #
            #     scalar `r2'  = 1 - e(rss)/e(tss)          // the within R2
            #     quietly _regress `y' `xvars' if `touse'   // plain OLS
            #     local r2c = e(r2)
            #     ereturn scalar r2_c = `r2c'
            #     ereturn scalar F_f = ((`r2'-`r2c')/e(df_a))/((1-`r2')/e(df_r))
            #     ereturn scalar p_f = fprob(e(df_a),e(df_r),e(F_f))
            #
            # so F_f is a comparison of the within R2 against the pooled OLS
            # R2, scaled by the absorbed degrees of freedom -- no entity effects
            # needed.  (An earlier version of this comment claimed the opposite,
            # which would have left F_f permanently unimplemented.)
            # ``xtreg_fe.ado:505-506`` is explicit that the r2 in this formula is
            # NOT ``e(r2)``:
            #
            #     /* not the same as e(r2) returned from _regress */
            #     scalar `r2' = 1 - e(rss)/e(tss)
            #     ereturn scalar F_f = ((`r2'-`r2c')/e(df_a))/((1-`r2')/e(df_r))
            #
            # Measured on ``xtreg_fe.json``: ``1 - rss/tss`` is 0.9531797 while
            # ``e(r2)`` is 0.9299367.  Substituting ``e(r2)`` gives 17.075 where the
            # fixture records 28.085 -- the fixture's ``p_f`` then confirms it, since
            # ``f.sf(28.085, 29, 148)`` reproduces the stored p-value exactly.
            out["k_absorb"] = float(len(u_panel))
            out["df_a"] = float(len(u_panel) - 1)
            if design is not None and params is not None:
                design_frame, _ = design
                matrix = np.asarray(design_frame.to_numpy(dtype=float), dtype=float)
                if matrix.shape[0] == nobs:
                    pooled = _pooled_r2(matrix, y)
                    if np.isfinite(pooled):
                        out["r2_c"] = float(pooled)
                        df_a = float(len(u_panel) - 1)
                        # The formula's r2 is ``1 - e(rss)/e(tss)`` as Stata computes
                        # it there -- both keys are already in ``out``.
                        tss_within = float(out.get("tss", float("nan")))
                        if np.isfinite(rss) and np.isfinite(tss_within) and tss_within > 0:
                            within_r2 = 1.0 - float(rss) / tss_within
                        else:  # pragma: no cover - defensive
                            within_r2 = float("nan")
                        if (
                            df_a > 0
                            and scalar_df > 0
                            and np.isfinite(within_r2)
                            and within_r2 < 1.0
                        ):
                            f_f = ((within_r2 - pooled) / df_a) / (
                                (1.0 - within_r2) / scalar_df
                            )
                            out["F_f"] = float(f_f)
                            out["p_f"] = float(
                                stats.f.sf(f_f, df_a, scalar_df)
                            )
        return out


    def _panel_structure_only(self, n_entities: int) -> dict[str, Any]:
        """The panel-shape block, for estimators whose sample is transformed.

        ``xtreg, fd`` estimates on first differences, so the response has no
        within-entity variation left to decompose -- but ``e(Tbar)``,
        ``e(g_min)``, ``e(g_avg)``, ``e(g_max)`` and ``e(Tcon)`` describe the
        underlying panel and are still meaningful (and still stored by Stata).
        """

        labels = self._data.index.get_level_values(self._entity).to_numpy()
        if labels.size == 0:
            return {}
        codes = pd.factorize(labels)[0]
        sizes = np.bincount(codes).astype(float)
        return {
            "Tbar": float(labels.size) / float(sizes.size),
            "g_min": int(sizes.min()),
            "g_avg": float(sizes.mean()),
            "g_max": int(sizes.max()),
            "Tcon": int(np.all(sizes == sizes[0])),
        }

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

    # Stata stores no ``e(r2)`` for ``xtreg, re`` and its ``e(r2_w)`` is not the
    # GLS ``rsquared`` the backend reports, so the alias is disabled.
    _within_r2_is_backend_r2 = False

    # The random-effects estimator quasi-demeans, so e(theta) is meaningful,
    # and it reports a Wald chi-square rather than an F.
    _stores_theta = True
    _stores_wald_chi2 = True

    # ``xtreg_re.json`` has no rss/mss/tss, no ll_0 and no absorbed block.
    _stores_totals = False
    _stores_mss = False
    _stores_ll_0 = False
    _stores_absorbed = False

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

    #: ``xtreg.sthlp:1014-1067`` (the be block) lists ``e(F)`` and NOT ``e(chi2)``.
    #: ``BetweenOLS`` inherits from ``RandomEffects``, which sets this flag True, so it
    #: has to be turned back off explicitly -- inheritance made the between estimator
    #: report a Wald chi-square that Stata does not store, and the value was just
    #: ``F * df_m`` under a different name.
    _stores_wald_chi2 = False

    _estimates_on_entity_means = True
    _stores_mss = True

    #: ``xtreg_be.ado`` stores NO sigma/rho/corr: the between estimator runs OLS
    #: on the entity means, so there is no panel/epsilon decomposition.  The
    #: ``corr`` occurrences in that ado are the CORR COMMAND feeding ``r2_o`` and
    #: ``r2_w``, not e() keys.
    _stores_sigma = False

    #: ``xtreg_be.ado`` contains no ``theta`` (or ``thta``) at all -- the between
    #: estimator does not quasi-demean, it runs OLS on the entity means.  This
    #: class inherits from ``RandomEffects``, which DOES store theta, so the flag
    #: has to be switched off explicitly; leaving it inherited put a theta on a
    #: be result where Stata stores none.
    _stores_theta = False

    # ``xtreg_be.json`` has ll_0 but not the fe-specific absorbed block.
    _stores_ll_0 = True

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

    #: ``xtreg.sthlp`` gives no chi-square for the first-difference estimator either
    #: (the ``be``/``fe`` blocks list only ``e(F)``).  Inherited from
    #: ``RandomEffects``, so it must be switched off here for the same reason as
    #: ``BetweenOLS``: otherwise ``fd`` reports ``F * df_m`` under a ``chi2`` name.
    _stores_wald_chi2 = False

    #: Differencing removes the entity effect entirely, so there is no
    #: ``sigma_u``/``rho``/``theta`` to report.  This class inherits from
    #: ``RandomEffects``, which stores all of them, so each flag is switched off
    #: explicitly.  Leaving them inherited "worked" only by accident: the theta
    #: block is guarded by ``isfinite(sigma_u)``, and ``fd`` happens to have no
    #: sigma_u -- so a future change that started computing one would silently
    #: start reporting a theta for a model that has no quasi-demeaning.
    _stores_theta = False
    _stores_absorbed = False
    #: No decomposed error structure: differencing removes the entity effect.
    _stores_sigma = False

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


def _pooled_r2(design: np.ndarray, y: np.ndarray) -> float:
    """R-squared of a plain OLS fit of ``y`` on ``design`` (Stata's ``r2_c``).

    ``xtreg_fe.ado:513-520`` obtains this by running ``_regress y xvars`` after
    the within fit; it is the pooled (between-and-within) explanatory power that
    ``e(F_f)`` compares the within R-squared against.
    """

    matrix = np.asarray(design, dtype=float)
    values = np.asarray(y, dtype=float).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != values.size:
        return float("nan")
    coefficients, *_ = np.linalg.lstsq(matrix, values, rcond=None)
    resid = values - matrix @ coefficients
    centred = values - values.mean()
    tss = float(centred @ centred)
    if tss <= 0.0:
        return float("nan")
    return 1.0 - float(resid @ resid) / tss


def _panel_means(values: np.ndarray, codes: np.ndarray, n_panels: int) -> np.ndarray:
    """Entity means of ``values`` in the order of ``codes``."""

    values = np.asarray(values, dtype=float).reshape(-1)
    sizes = np.bincount(codes, minlength=n_panels).astype(float)
    totals = np.bincount(codes, weights=values, minlength=n_panels)
    with np.errstate(divide="ignore", invalid="ignore"):
        means = totals / sizes
    return means


def _gaussian_loglik(rss: float, nobs: int) -> float:
    """Gaussian log-likelihood at the ML variance, as Stata's ``e(ll)``."""

    if nobs <= 0 or not np.isfinite(rss) or rss <= 0:
        return float("nan")
    return float(-0.5 * nobs * (np.log(2.0 * np.pi) + np.log(rss / nobs) + 1.0))
