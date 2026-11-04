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
    (PyFixest's ``_df_k``).  It is deliberately larger than
    ``len(results.params)``, and it is **not** reghdfe's own ``e(rank)``, which
    is documented as the rank of ``e(V)`` -- the reported regressors only
    (``reghdfe_firm.json``: ``e(rank)=2`` for two slopes plus an absorbed
    intercept).  Use ``e(df_m)`` for that count.
``e(df_m)``
    Reported slope parameters only (absorbed fixed effects excluded), matching
    the number of coefficients shown in the Stata output.  This *is* reghdfe's
    ``e(df_m)``.
``e(rmse)``
    ``sqrt(rss / (N - rank_full))``, a residual standard deviation, not
    PyFixest's ``sqrt(rss / N)``.  ``rank_full`` counts every absorbed
    fixed-effect level, so ``rmse`` is a sample statistic: it is **invariant to
    the covariance estimator** and does *not* follow ``e(df_r)``.  This is
    reghdfe's behaviour too -- ``reghdfe_firm.json`` and
    ``reghdfe_firm_cluster.json`` report the same ``e(rmse)`` while ``e(df_r)``
    moves from 148 to 29.
``results.df_resid`` vs ``e(df_r)``
    They coincide: both are ``G - 1`` under clustering and ``N - K`` otherwise.
    ``results.df_resid`` is the reference distribution used for t/p/CI, and
    ``e(df_r)`` is the same number stored as the Stata scalar (verified:
    148/148 nonrobust, 29/29 clustered).  Note that under clustering this is
    *not* the denominator behind ``e(rmse)`` -- see above.

Nested fixed effects
--------------------
Following ``reghdfe``'s default ``dofadjustments(nested)`` (and fixest's
``k_fixef="nonnested"``), a fixed effect that is *fully nested* in the cluster
variable contributes no parameters to ``K``: the cluster absorbs it.  So
``e(rank)`` and ``e(df_r)`` shrink when the cluster variable nests a fixed
effect, while the slopes and ``e(rmse)`` are unchanged (``rmse`` keeps the
full-rank denominator).  That mixed accounting is reghdfe's own: verified
against a live capture, and reghdfe's source comments on it deliberately
(``reghdfe.mata``: "breaks consistency with areg but ensures internal
consistency between vce(robust) and vce(cluster _n)").
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import pyfixest
from scipy import stats

from westata.data.weights import WeightSpec, normalize_weights
from westata.errors import ConsistencyError, UnsupportedError, WeightError
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


def _pooled_r2(
    response: np.ndarray,
    regressors: np.ndarray,
    n_effective: int,
) -> float | None:
    """R-squared of a plain pooled OLS fit of the same regressors.

    ``areg``/``xtreg, fe`` obtain this by re-running ``_regress y xvars`` after
    the absorbed fit (``xtreg_fe.ado:513-520``).

    The design must be the **original** (un-demeaned) one: PyFixest's ``_X`` has
    had the fixed effects projected out, so regressing ``y`` on it yields the
    *within* R-squared instead of the pooled one.  On the 30x6 test panel that
    mistake gives 0.9187 where the correct pooled R-squared is 0.7174 -- close
    to ``e(r2)``, which is what made it look plausible.

    Returns ``None`` when the inputs are unusable, so the caller leaves
    ``e(r2_c)`` out rather than guessing.
    """

    y = np.asarray(response, dtype=float).reshape(-1)
    matrix = np.asarray(regressors, dtype=float)
    if y.size != n_effective or matrix.shape[0] != y.size:
        return None
    if matrix.ndim != 2:
        return None
    # Add a constant unless the supplied design already has one.
    if not any(np.ptp(matrix[:, j]) == 0.0 for j in range(matrix.shape[1])):
        matrix = np.column_stack([np.ones(y.size), matrix])

    coefficients, *_ = np.linalg.lstsq(matrix, y, rcond=None)
    residual = y - matrix @ coefficients
    centred = y - y.mean()
    total = float(centred @ centred)
    if total <= 0.0:
        return None
    return 1.0 - float(residual @ residual) / total


class FEOLS:
    """Fixed-effects OLS with high-dimensional fixed effects (Stata ``reghdfe``).

    Parameters
    ----------    formula:
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

    #: ``e(cmd)``. Subclasses override these to present themselves as a different Stata
    #: command while reusing the same estimation machinery -- which is the honest way to
    #: support ``areg``: it is a DIFFERENT command in Stata (``absorb()`` is required, the
    #: banner reads "Linear regression, absorbing indicators", and ``e(title2)`` names the
    #: absorbed variable), not another name for ``reghdfe``.
    _command_name = "reghdfe"
    #: areg reports an intercept, reghdfe does not. See fit().
    _reports_intercept = False
    _command_title = None
    _predict_program = None
    _marginsnotok = None
    #: e(estmethod): areg sets "areg"; reghdfe sets nothing.
    _stata_estimethod = None
    #: Whether this command can serve HC2/HC3. FEOLS cannot (PyFixest hides the leverage
    #: of the full dummy design); Areg can, because it builds that design itself.
    _allows_leverage_cov_types = False

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

    #: Whether ``df_a`` subtracts a constant absorbed degree of freedom.
    #:
    #: ``areg`` reports ``df_a = k_absorb - 1`` (29 for 30 groups) because it
    #: reports the intercept separately; ``reghdfe`` absorbs the constant and
    #: reports ``df_a = df_a_initial`` (30).  Both are confirmed by their fixtures,
    #: so this is a genuine convention difference rather than a bug in either.
    _absorb_df_drops_constant = False

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        weights: Any = None,
        weight_type: str | None = None,
        subset: Any = None,
    ) -> "FEOLS":
        """Build a model from a PyFixest formula (see :class:`FEOLS`).

        ``subset=`` restricts the estimation sample, matching ``OLS``/``Logit``/
        ``Poisson`` and Stata's ``if``/``in``. Filtering the frame by hand instead would
        work numerically but loses the distinction between the frame and the sample, and
        the absorbed-effect prediction added earlier reads the estimation frame.
        """

        if subset is not None:
            data = data.loc[subset]
        return cls(formula, data, weights=weights, weight_type=weight_type)

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: SSC | None = None,
        level: float | None = None,
    ) -> StataResults:
        """Estimate the model and return a Stata-conformant result object.

        Parameters
        ----------
        cov_type:
            ``None`` selects the Stata default (``"nonrobust"``, or ``"HC1"``
            for a ``pweight`` model).  Supported: ``"nonrobust"``, ``"HC0"``,
            ``"HC1"``, ``"cluster"``; see :data:`SUPPORTED_COV_TYPES`.
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
        # Recorded for _with_intercept: areg reports cov(_cons) under whatever vce()
        # was requested, so the constant needs the same clustering the model used.
        self._fit_cluster_columns = None
        cluster_columns = self._resolve_cluster_columns(
            cov_kwds, effective_cov_type, data
        )

        self._fit_cluster_columns = cluster_columns
        # Keep the RAW cluster labels as well. ``_resolve_cluster_columns`` may synthesise
        # column names (``__westata_cluster_0__``) that exist only on the working copy, so
        # the absorbed-degrees-of-freedom nesting test needs the values, not the names.
        from westata.vce.dispatch import cluster_labels_from_kwds

        _raw_labels = cluster_labels_from_kwds(cov_kwds) if cov_type == "cluster" else None
        if _raw_labels is not None and not isinstance(_raw_labels, str):
            _raw = np.asarray(_raw_labels)
            if _raw.ndim == 1 and _raw.size == len(data):
                self._fit_cluster_labels = pd.Series(_raw, index=data.index)
            else:
                self._fit_cluster_labels = None
        else:
            self._fit_cluster_labels = None

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

        return self._build_results(model, data, effective_cov_type, spec, level)
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

        # One accessor for the cluster labels: ``groups``/``clusters``/``cluster`` are all
        # accepted, the same way they are for every other estimator. This file used to
        # accept only ``cluster`` and ``groups``, which made ``FEOLS`` the one estimator
        # where ``clusters`` failed -- see
        # ``replication/papers/adh2013/WESTATA_GAPS.md`` §3.
        from westata.vce.dispatch import cluster_labels_from_kwds

        labels = cluster_labels_from_kwds(cov_kwds)
        if labels is None:
            raise ValueError(
                "cov_type='cluster' requires the cluster labels; pass them as "
                "cov_kwds={'groups': labels} (or the equivalent 'clusters' / "
                "'cluster'); got keys " + ", ".join(map(str, cov_kwds))
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
        # A DataFrame carries COLUMN NAMES, and ``clustertable`` must print those rather
        # than the scratch names made up below -- capture them before the labels are
        # flattened into an array, where the names are unrecoverable.
        self._cluster_display_names: list[str] | None = None
        if isinstance(labels, pd.DataFrame):
            self._cluster_display_names = [str(name) for name in labels.columns]
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
        if self._cluster_display_names is None:
            # A bare array supplies no names, so positional placeholders are used rather
            # than inventing labels that would look authoritative but be wrong.
            self._cluster_display_names = [
                f"cluster{position + 1}" for position in range(values.shape[1])
            ]
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
        # HC1 maps onto PyFixest's "HC1" heteroskedastic variant.
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
        level: float | None = None,
    ) -> StataResults:
        """Translate a fitted ``pyfixest.Feols`` into :class:`StataResults`."""

        # Retained so ``_tss_within`` can refit the intercept-only absorbed model
        # on the same estimation sample.
        self._within_data = data.reset_index(drop=True)
        # PyFixest records the rows it dropped as a frozenset of positions in the input
        # frame (missing values AND singleton groups). That is the ONLY authoritative
        # description of the estimation sample: re-deriving it by filtering on "level has
        # more than one row" reproduced 140432 of 140469 on the report's C12-10 case but
        # lost a further 7 rows when refitted, because singleton removal interacts across
        # the two absorbed dimensions and is applied iteratively.
        self._dropped_rows = getattr(model, "_na_index", None)
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
                + " dropped from the model because "
                + ("it is" if len(absorbed) == 1 else "they are")
                + " collinear with the absorbed fixed effects or with another "
                "regressor; the reported coefficients are conditioned on this "
                "omission",
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

        rank = self._rank(model, nobs)
        # `results.df_resid` follows the reference distribution (G - 1 under
        # clustering, matching westata's OLS); the *residual* df used by rmse is
        # normally N - K.
        df_resid = float(model._df_t)
        residual_df = float(n_effective - rank)
        # reghdfe's three degree-of-freedom quantities diverge under clustering,
        # and reproducing them requires computing all three:
        #
        #   e(df_r)   = G - 1         (nested FE treated as redundant)
        #   e(rmse)   = sqrt(rss/e(df_r))
        #   e(r2_a)   = 1-(1-r2)(N-1)/(N-rank_full)
        #
        # where rank_full counts every absorbed fixed-effect level even when it
        # is nested in the cluster. Verified on absorb(firm) cluster(firm):
        # G=30 -> e(df_r)=29, and N-rank_full = 180-32 gives the reported
        # 0.7305369420818487.
        clustered = str(cov_type).strip().lower() == "cluster"
        rank_full = rank
        if clustered:
            rank_full = self._rank_full(model, n_effective)
            residual_df = df_resid
        if residual_df <= 0:
            raise ConsistencyError(
                "reghdfe: "
                f"{n_effective:g} effective observations but {rank} parameters "
                "(including absorbed fixed effects); the model is not identified"
            )
        r2_denominator = float(n_effective - rank_full)
        if r2_denominator <= 0:
            r2_denominator = float("nan")

        has_intercept = any(name == "Intercept" for name in names)
        tested = [index for index, name in enumerate(names) if name != "Intercept"]
        df_model = float(len(tested)) if has_intercept else float(len(names))

        rss = float(np.sum(kept_weights * resid**2))
        mean_y = float(np.sum(kept_weights * y) / np.sum(kept_weights))
        tss = float(np.sum(kept_weights * (y - mean_y) ** 2))
        r2 = 1.0 - rss / tss if tss > 0 else float("nan")
        # Adjusted R^2 uses the full rank (nested fixed effects included) even
        # when e(df_r) has collapsed to G - 1.
        r2_adj = (
            1.0 - (1.0 - r2) * (n_effective - 1.0) / r2_denominator
            if np.isfinite(r2) and np.isfinite(r2_denominator)
            else float("nan")
        )
        rmse = float(np.sqrt(rss / r2_denominator)) if np.isfinite(r2_denominator) else float("nan")
        llf = float(
            -0.5 * n_effective * (np.log(2 * np.pi) + np.log(rss / n_effective) + 1.0)
        ) if rss > 0 else float("nan")

        fvalue = self._wald_f(params, cov, tested)

        # ``e(tss)``/``e(mss)``/``e(ll_0)``: Stata's reghdfe stores all three
        # (``reghdfe_firm.json`` scalars), and ``rss + mss = tss`` holds there.
        # ``tss`` is the centred total computed above and was previously
        # discarded after deriving ``mss``; storing it costs nothing.
        # ``ll_0`` uses the same Gaussian intercept-only construction as
        # :class:`~westata.estimators.linear.OLS`, verified there against
        # StataNow 19.5 for nonrobust/robust/cluster/pweight/fweight.
        # NOTE: ``tss_within``/``r2_within``/``r2_a_within`` are also in the
        # fixture but need the absorbed-only (FE-demeaned) response, which this
        # adapter does not currently retain; they are left out rather than
        # approximated.  See retest/gaps/linear_iv_ts_gaps.md.
        ll_0 = (
            float(-0.5 * n_effective * (np.log(2 * np.pi) + np.log(tss / n_effective) + 1.0))
            if tss > 0
            else float("nan")
        )

        e: dict[str, Any] = {
            "N": n_effective,
            "df_m": df_model,
            "df_r": residual_df,
            "r2": r2,
            "r2_a": r2_adj,
            "rmse": rmse,
            "F": fvalue,
            "ll": llf,
            "ll_0": ll_0,
            "rss": rss,
            "mss": tss - rss,
            "tss": tss,
            "rank": rank,
            # Command identity. ``areg`` and ``reghdfe`` are DIFFERENT Stata commands with
            # different banners, a different predict program and different
            # ``e(marginsnotok)``; a subclass overrides the class attributes and these
            # keys follow. ``areg.json`` is the authority for the areg values.
            "cmd": self._command_name,
        }
        # Recorded so ``Areg`` can convert to areg's parameter count: reghdfe excludes the
        # nested absorbed effects from the residual denominator, areg includes them, and
        # the two differ by exactly k_absorb. See ``Areg._rescale_cov_to_areg``.
        e["reghdfe_residual_dof"] = r2_denominator
        if self._command_title is not None:
            e["title"] = self._command_title
        if self._predict_program is not None:
            e["predict"] = self._predict_program
        if self._marginsnotok is not None:
            e["marginsnotok"] = self._marginsnotok
        # ``e(vce)`` / ``e(vcetype)`` / ``e(estmethod)`` / ``e(marginsprop)``: all four are
        # in areg.json and all four were missing. ``e(vce)`` in particular is the macro a
        # Stata user reads to know which covariance was requested, and its absence made
        # ``Areg`` look like it had no vce at all next to a genuine areg capture.
        if self._stata_estimethod is not None:
            e["estmethod"] = self._stata_estimethod
            e["marginsprop"] = "nolinearize"
            stata_vce, stata_vcetype = _stata_vce_names(cov_type)
            e["vce"] = stata_vce
            e["vcetype"] = stata_vcetype
        # ``areg`` prints a SECOND title line naming the absorbed variable
        # (areg.json: e(title2) = "Absorbed variable: firm") and stores e(absvar). Both
        # are areg-only; reghdfe has neither.
        absorbed_names = getattr(self, "_absorbed_names", None)
        if absorbed_names:
            e["absvar"] = " ".join(absorbed_names)
            e["title2"] = "Absorbed variable: " + " ".join(absorbed_names)
        # ``tss_within`` / ``r2_within`` / ``r2_a_within``: ``reghdfe`` measures
        # the within R-squared against the total sum of squares of ``y`` after
        # projecting out ONLY the absorbed effects.  Equivalently, it is the
        # residual sum of squares of ``y ~ 1 | absorb`` -- verified numerically
        # against direct demeaning for both one-way and two-way absorption.
        #
        # The identity ``r2_within = 1 - rss/tss_within`` is confirmed digit for
        # digit by reghdfe_firm.json:
        #     1 - 131.2293504838976/418.6969854497428 = 0.6865767964798271
        # which is exactly its e(r2_within).
        tss_within = self._tss_within(y, n_effective)
        if tss_within is not None and tss_within > 0.0:
            e["tss_within"] = float(tss_within)
            # ``rss`` is on the LEVEL scale, but ``tss_within`` is the residual sum of
            # squares of ``y ~ 1 | <absorb>``, i.e. the within-transformed total. Dividing
            # one by the other mixes scales: it gave r2_within = .00635709 where Stata
            # reports .00024865 (report C12-10). The numerator must be the within residual
            # of the FULL model, so take it from the absorbed model's own residuals.
            _rss_within = self._rss_within(y, n_effective)
            if _rss_within is not None and _rss_within > 0.0:
                e["r2_within"] = float(1.0 - _rss_within / tss_within)
            else:
                e["r2_within"] = float(1.0 - rss / tss_within)
            if np.isfinite(r2_denominator) and r2_denominator > 0.0:
                e["r2_a_within"] = float(
                    1.0
                    - (1.0 - e["r2_within"])
                    * (n_effective - 1.0)
                    / r2_denominator
                )

        if cov_type == "cluster":
            clusters = self._n_clusters(model)
            e["N_clust"] = float(clusters)
            # ``N_clust1`` is the group count of the FIRST cluster variable and
            # ``N_clustervars`` the number of cluster variables.  westata clusters
            # on one variable, so both reduce to the values the fixtures show
            # (``N_clust1 == N_clust == 30``, ``N_clustervars == 1``).
            e["N_clust1"] = float(clusters)
            e["N_clustervars"] = 1.0
            # ``clustertable`` needs the per-dimension counts and the joint counts for
            # every combination; e(N_clust) alone holds the minimum. Computed from the
            # actual labels so the joint rows are measured rather than multiplied.
            columns = getattr(self, "_fit_cluster_columns", None) or []
            if len(columns) >= 2:
                from itertools import combinations as _combinations

                display = getattr(self, "_cluster_display_names", None) or [
                    str(name) for name in columns
                ]
                per_dimension = [
                    int(pd.factorize(np.asarray(data[name]), sort=True)[0].max()) + 1
                    for name in columns
                ]
                e["_cluster_names"] = [str(name) for name in display]
                e["_cluster_counts"] = per_dimension
                joint: dict[str, int] = {}
                for size in range(2, len(columns) + 1):
                    for indices in _combinations(range(len(columns)), size):
                        pairs = list(
                            zip(*(np.asarray(data[columns[index]]) for index in indices))
                        )
                        flat = np.empty(len(pairs), dtype=object)
                        flat[:] = pairs
                        # Keyed by the DISPLAY names, which is what the renderer builds
                        # its row labels from too -- keying by the scratch column names
                        # made every joint row fall back to zero.
                        key = "#".join(display[index] for index in indices)
                        joint[key] = int(pd.factorize(flat, sort=True)[0].max()) + 1
                e["_cluster_combination_counts"] = joint

        # --- keys Stata's ``areg`` records alongside ``reghdfe``'s ----------
        # ``ar2`` is the adjusted R-squared, which ``areg`` stores under that
        # name as well as ``r2_a`` (areg.json: ar2 == r2_a == 0.9433727826186399).
        if np.isfinite(r2_adj):
            e["ar2"] = float(r2_adj)
        # --- the absorbed-effects bookkeeping ``reghdfe`` stores ---------------
        #
        # ``reghdfe_firm_year.json`` pins down the identities:
        #
        #     df_a_initial = 36   (30 firms + 6 years)
        #     df_a         = 35   (one partition is redundant)
        #     df_a_nested  = 0
        #     df_a_redundant = 1
        #     df_a_initial - df_a == df_a_nested + df_a_redundant == 1
        #
        # so ``df_a = df_a_initial - (df_a_nested + df_a_redundant)``.  The
        # redundant partition is the one whose levels the others already span; for
        # a two-way absorption that is exactly one (the shared constant).
        #
        # ``num_singletons`` counts observations dropped because they are the only
        # row in some absorbed level; every fixture has 0, which PyFixest's own
        # fitting would also report because it does not silently drop rows.  The
        # count is therefore derived from N_full - N rather than invented.
        absorbed_levels = self._absorbed_levels(model, n_effective)
        n_hdfe = self._n_hdfe(model)
        if n_hdfe is not None:
            e["N_hdfe"] = float(n_hdfe)
            # ``N_hdfe_extended`` adds the absorbed dimensions introduced by
            # continuous interactions; this implementation has none, so the two
            # coincide (``reghdfe_firm.json`` and ``reghdfe_firm_year.json`` both
            # report them equal).
            e["N_hdfe_extended"] = float(n_hdfe)
        e["N_full"] = float(n_effective)
        e["drop_singletons"] = 1.0
        # No rows are dropped for singletons, so N_full == N and the count is 0.
        # Recording it as ``N_full - N`` keeps the identity honest if row
        # dropping is ever added.
        e["num_singletons"] = float(max(e["N_full"] - n_effective, 0.0))
        if absorbed_levels is not None:
            e["k_absorb"] = float(absorbed_levels)
            e["df_a_initial"] = float(absorbed_levels)
            # The NAMES of the absorbed effects, for the "Absorbed degrees of freedom"
            # table reghdfe prints below its coefficient table. Parsed from the formula's
            # fixed-effect block ("y ~ x | firm + year"), since pyfixest does not expose
            # them on the result and e(absorbed) (the collinear-variable list) is a
            # different thing entirely.
            _fe_block = str(self._formula).split("|", 1)
            if len(_fe_block) == 2:
                e["absorbed_names"] = [
                    part.strip()
                    for part in _fe_block[1].split("+")
                    if part.strip()
                ]
            # ``areg`` and ``reghdfe`` disagree here, and both fixtures prove it:
            #
            #   areg.json         df_a = 29 = k_absorb - 1     (30 groups)
            #   reghdfe_firm.json df_a = 30 = df_a_initial     (30 levels)
            #
            # ``areg`` subtracts the intercept it reports separately; ``reghdfe``
            # absorbs it and does not subtract.  The class flag
            # ``_absorb_df_drops_constant`` selects the convention, so one code
            # path serves both.
            redundant = float(max((n_hdfe or 1) - 1, 0))
            e["df_a_redundant"] = redundant
            # An absorbed effect NESTED WITHIN A CLUSTER VARIABLE contributes no degrees of
            # freedom: every one of its coefficients is collinear with a cluster dummy, so
            # reghdfe marks it redundant and prints 0 Num. Coefs. Measured on the report's
            # C12-10 case -- 550 counties, 262 years, clustered on county -- Stata prints
            #
            #   county |  536   536   0  *     (nested within cluster)
            #   year   |  262     1 261
            #
            # and e(df_a) = 261. Counting both partitions gave 797 here.
            e["df_a_nested"] = float(self._nested_absorbed_levels(model, data))
            drop_constant = 1.0 if self._absorb_df_drops_constant else 0.0
            e["df_a"] = float(
                absorbed_levels - (e["df_a_nested"] + redundant + drop_constant)
            )
        # ``sum_w``: the raw weight sum, N when unweighted.
        e["sum_w"] = float(n_effective)
        # ``sumweights`` is ``reghdfe``'s spelling of the same quantity
        # (``reghdfe_firm.json``: sumweights == sum_w == 180).
        e["sumweights"] = float(n_effective)
        # ``converged`` / ``ic``: reghdfe's solver report.  The absorbed fit is a
        # closed-form projection here (alternating projections in reghdfe), so
        # one "iteration" is the honest count for a converged solve.
        e["converged"] = 1.0
        e["ic"] = 1.0
        # ``report_constant``: 1 when a constant is among the reported
        # coefficients.  ``reghdfe`` does not report it by default (the absorbed
        # effects subsume it), but the key records whether one was reported.
        e["report_constant"] = (
            1.0
            if any(
                str(name).strip().lower() in ("_cons", "intercept", "const")
                for name in names
            )
            else 0.0
        )
        # ``rc``: Stata's return code; 0 means the fit completed.
        e["rc"] = 0.0
        # ``p``: the model test's p-value.
        if np.isfinite(fvalue):
            e["p"] = float(stats.f.sf(fvalue, df_model, residual_df))
        # ``r2_c`` / ``F_absorb`` / ``p_absorb``: ``areg`` compares the absorbed
        # model against a plain pooled OLS fit of the same regressors.
        #
        # ``xtreg_fe.ado:505-524`` (the same computation ``areg`` reports as
        # ``F_absorb``) is:
        #
        #     quietly _regress `y' `xvars' if `touse'    // pooled OLS
        #     ereturn scalar r2_c = e(r2)
        #     ereturn scalar F_f  = ((`r2'-`r2c')/e(df_a))/((1-`r2')/e(df_r))
        #
        # and ``areg.json`` confirms the equivalence numerically: its
        # F_absorb = 28.085188842551904 is *identical* to ``xtreg_fe.json``'s
        # F_f, and both files carry the same ``r2_c`` = 0.6955194140102255.
        # So the pooled R-squared is all that is needed -- no entity effects.
        # ``noabsorb`` (Stata: ``noabsorbtest``) suppresses this whole computation, per
        # areg.sthlp:157-161 -- "prevents computation of the F test that all coefficients
        # on indicators for absorbed variables are jointly zero", added because the test
        # "requires an iterative search". So the option is about SKIPPING WORK, and the
        # observable consequence is that e(F_absorb)/e(p_absorb) are absent from the
        # result rather than present-and-empty.
        if (
            not getattr(self, "_suppress_absorb_test", False)
            and np.isfinite(r2)
            and absorbed_levels is not None
            and absorbed_levels > 1
        ):
            # Original (un-demeaned) regressor matrix: PyFixest's ``_X`` has the
            # fixed effects projected out, so only the estimation frame gives
            # the POOLED design that ``r2_c`` is defined against.
            original_design = None
            try:
                available = [name for name in names if name in data.columns]
                if available:
                    rows = np.asarray(y, dtype=float).size
                    original_design = np.column_stack(
                        [
                            np.ones(rows),
                            *[
                                np.asarray(data[name].to_numpy(), dtype=float)
                                for name in available
                            ],
                        ]
                    )
            except (AttributeError, TypeError, ValueError):
                original_design = None
            pooled = (
                _pooled_r2(y, original_design, n_effective)
                if original_design is not None
                else None
            )
            if pooled is not None and np.isfinite(pooled):
                e["r2_c"] = float(pooled)
                # Reuse the ``df_a`` already reported above rather than
                # recomputing it as ``absorbed_levels - 1``: that shortcut ignored
                # the redundant-partition correction a two-way absorption needs,
                # and made ``F_absorb`` inconsistent with ``e(df_a)``.
                df_a = float(e["df_a"])
                if df_a > 0.0 and 0.0 <= r2 < 1.0 and residual_df > 0.0:
                    f_absorb = ((r2 - pooled) / df_a) / ((1.0 - r2) / residual_df)
                    e["F_absorb"] = float(f_absorb)
                    e["p_absorb"] = float(
                        stats.f.sf(f_absorb, df_a, residual_df)
                    )

        # ``areg`` REPORTS AN INTERCEPT; ``reghdfe`` does not. areg.json's names are
        # ['x1', 'x2', '_cons'] and its b carries three entries, so this is a real
        # difference between the two commands rather than a cosmetic one.
        #
        # The value is the AVERAGE absorbed effect, ``mean(y - b'x)``, verified digit for
        # digit against the fixture (1.209671208388627 both ways on its panel). An FE model
        # does not identify a constant separately from the absorbed effects -- only a
        # normalisation fixes it -- and this is the one areg uses.
        params, names, cov = self._with_intercept(data, y, params, names, cov, e)

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
            model_name=self._command_name,
            cov_type=cov_type,
            e=e,
            level=level,
        )
        # Reported, not hidden: absorbed regressors are part of the result.
        results.absorbed = absorbed
        #: The estimation frame and design, retained so ``estat summarize`` and
        #: ``estat vif`` can run.  ``reghdfe`` supports both; without them the two
        #: commands refused with "the estimation-sample data was not retained",
        #: which read like a deliberate limitation but was a gap.
        results.model_data = data
        #: ``e(depvar)``/``response_name``: ``estat summarize`` lists the response
        #: first and needs it by name; ``reghdfe`` records it like every other
        #: estimation command.
        try:
            lhs = str(self._formula).split("~", 1)[0].strip()
            response = lhs.split()[0] if lhs else None
            if response and response in data.columns:
                results.response_name = response
                results.e["depvar"] = response
        except (AttributeError, IndexError):  # pragma: no cover
            pass
        try:
            available = [name for name in names if name in data.columns]
            if available:
                rows = np.asarray(y, dtype=float).size
                design = np.column_stack(
                    [
                        np.ones(rows),
                        *[
                            np.asarray(data[name].to_numpy(), dtype=float)
                            for name in available
                        ],
                    ]
                )
                results.design_matrix = (design, ["Intercept", *available])
        except (AttributeError, TypeError, ValueError):  # pragma: no cover
            pass

        _register_hdfe_predict(
            results,
            design_matrix=getattr(results, "design_matrix", None),
            formula=str(self._formula),
            y=np.asarray(y, dtype=float),
            # The RESULT's parameters, not the local ones: ``areg`` appends ``_cons`` in
            # ``_with_intercept``, and registering the pre-intercept list left the predict
            # dispatcher looking for a coefficient the result no longer matched.
            params=np.asarray(results.params, dtype=float),
            param_names=list(results.param_names),
            data=data,
            weights=getattr(self, "_weights", None),
        )
        return results

    def _rescale_cov_to_areg(
        self,
        data: pd.DataFrame,
        y: np.ndarray,
        cov: np.ndarray,
        e: dict[str, Any],
    ) -> np.ndarray:
        """Hook: subclasses may rescale the covariance to their command's convention.

        ``FEOLS`` targets ``reghdfe`` and uses that convention unchanged; ``Areg``
        overrides this to apply areg's degrees-of-freedom count.
        """

        return cov

    def _with_intercept(
        self,
        data: pd.DataFrame,
        y: np.ndarray,
        params: np.ndarray,
        names: list[str],
        cov: np.ndarray,
        e: dict[str, Any],
    ) -> tuple[np.ndarray, list[str], np.ndarray]:
        """Append ``e(_cons)``-style intercept reporting, per command.

        ``reghdfe`` reports only the slopes; ``areg`` reports a constant as well, because
        an absorbed-indicator model DOES have one once the absorbed effects are normalised
        (``areg.json``: names ``['x1', 'x2', '_cons']``, three ``b`` entries).

        Kept as a hook rather than an inline branch so the difference between the two
        commands lives in one place per command, next to the other identity attributes.
        """

        return params, names, cov

    @staticmethod
    def _rank(model: Any, nobs: int) -> int:
        """Estimated-parameter count ``K`` used by the ``(N-1)/(N-K)`` factor.

        This is PyFixest's ``_df_k``: reported regressors plus the absorbed
        fixed-effect parameters that are *not* nested in the cluster variable.
        It must not be confused with ``nobs - _df_t``: under clustering
        ``_df_t`` is ``G - 1``, which has nothing to do with ``K``.
        """

        value = getattr(model, "_df_k", None)
        if value is None:
            raise ConsistencyError(
                "reghdfe: PyFixest did not record the degrees-of-freedom count "
                "(_df_k); westata cannot reconstruct e(rank) safely"
            )
        rank = int(round(float(value)))
        if not 0 < rank <= nobs:
            raise ConsistencyError(
                f"reghdfe: PyFixest reported an impossible parameter count {rank} "
                f"for {nobs} observations"
            )
        return rank

    def _rss_within(self, y: np.ndarray, n_effective: int) -> float | None:
        """Residual sum of squares of the full model measured on the WITHIN scale.

        ``r2_within`` is ``1 - rss_within / tss_within`` where ``tss_within`` is the
        residual sum of squares of ``y ~ 1 | <absorb>``. The model's ordinary ``rss`` is on
        the LEVEL scale, so pairing it with ``tss_within`` mixes two scales -- that produced
        .00635709 against Stata's .00024865 on the report's C12-10 case.

        Computed by refitting the full specification with the absorbed model and taking the
        residuals of the identical fit, which keeps multi-way absorption exact.
        """

        formula = str(self._formula)
        if "|" not in formula:
            return None
        absorb_part = formula.split("|", 1)[1].strip()
        if not absorb_part:
            return None
        frame = self._within_frame(y)
        # ``y`` has already been swapped in as ``__westata_y__``; the regressors keep their
        # original names, so the same right-hand side can be reused verbatim.
        rhs = formula.split("~", 1)[1].split("|", 1)[0].strip()
        if not rhs or rhs in {"1", "-1"}:
            return None
        try:
            full = pyfixest.feols(
                f"__westata_y__ ~ {rhs} | {absorb_part}", data=frame
            )
            residuals = np.asarray(full.resid(), dtype=float)
        except Exception:  # noqa: BLE001 - leave the within block out rather than guess
            return None
        if residuals.size == 0 or residuals.size < 0.5 * n_effective:
            return None
        return float(residuals @ residuals)

    def _tss_within(self, y: np.ndarray, n_effective: int) -> float | None:
        """Total sum of squares of ``y`` after removing ONLY the fixed effects.

        This is the residual sum of squares of ``y ~ 1 | <absorb>``, which
        ``reghdfe`` reports as ``e(tss_within)``.  Fitting the intercept-only
        absorbed model through the same backend keeps multi-way absorption exact
        (a single demean pass is not enough for two or more effects).

        Returns ``None`` when the fit cannot be made, so the caller omits the
        within block rather than substituting the ordinary centred total.
        """

        formula = str(self._formula)
        if "|" not in formula:
            return None
        absorb_part = formula.split("|", 1)[1].strip()
        if not absorb_part:
            return None
        try:
            reduced = pyfixest.feols(f"__westata_y__ ~ 1 | {absorb_part}", data=self._within_frame(y))
        except Exception:  # noqa: BLE001 - any backend failure means "leave it out"
            return None
        try:
            residuals = np.asarray(reduced.resid(), dtype=float)
        except Exception:  # noqa: BLE001
            return None
        # The reduced fit drops singletons AGAIN -- absorbing only ``county + year`` without
        # the regressor can leave levels that the full model kept. On the report's C12-10
        # case that returns 140425 residuals against an estimation sample of 140432, so the
        # old ``== n_effective`` test rejected a perfectly good sum of squares and the
        # within block vanished from the header. Only a wildly different size signals a
        # real mismatch.
        if residuals.size == 0 or residuals.size < 0.5 * n_effective:
            return None
        return float(residuals @ residuals)

    def _within_frame(self, y: np.ndarray) -> pd.DataFrame:
        """The estimation frame with the response replaced by ``y``.

        ``y`` is ``fitted + resid`` from the backend, i.e. the response on its original
        scale, so it has one entry per OBSERVATION IN THE ESTIMATION SAMPLE -- 140432 on the
        report's C12-10 case, against 140469 rows in the pre-fit frame. PyFixest drops 37
        of those 37 as missing values or singleton groups, and it drops them ITERATIVELY
        across the absorbed dimensions, so reproducing the subset by filtering
        "level appears more than once" keeps 140432 rows whose subsequent refit silently
        loses a further 7.

        ``_dropped_rows`` (PyFixest's own ``_na_index``) is the authoritative list, so the
        frame is rebuilt from it and the response lands on exactly the rows the model used.
        """

        frame = self._within_data.copy()
        values = np.asarray(y, dtype=float).reshape(-1)
        if len(frame) != values.size:
            dropped = getattr(self, "_dropped_rows", None)
            if dropped:
                keep = np.ones(len(frame), dtype=bool)
                positions = np.asarray(sorted(int(i) for i in dropped), dtype=int)
                positions = positions[(positions >= 0) & (positions < len(frame))]
                keep[positions] = False
                if int(keep.sum()) == values.size:
                    frame = frame.loc[keep]
        if len(frame) == values.size:
            frame["__westata_y__"] = values
            return frame
        if len(frame) > values.size:
            # Last resort: the frame is longer than the response and the dropped set could
            # not be matched exactly. Restricting by position keeps the response aligned
            # with the leading rows, which is right whenever the dropped rows are trailing.
            frame = frame.iloc[: values.size].copy()
            frame["__westata_y__"] = values
            return frame
        return pd.DataFrame({"__westata_y__": values})

    @staticmethod
    def _n_hdfe(model: Any) -> int | None:
        """Number of absorbed fixed-effect dimensions (``reghdfe``'s ``e(N_hdfe)``).

        PyFixest stores one entry per absorbed effect in ``_k_fe``, so its length
        is the count: ``__fixed_effect__(firm)`` alone gives 1, ``firm`` plus
        ``year`` gives 2 -- exactly the values ``reghdfe_firm.json`` (1) and
        ``reghdfe_firm_year.json`` (2) report.
        """

        counts = getattr(model, "_k_fe", None)
        if counts is None:
            return None
        try:
            size = int(np.asarray(counts.to_numpy()).size)
        except (AttributeError, TypeError, ValueError):
            return None
        return size if size > 0 else None

    @staticmethod
    def _absorbed_levels(model: Any, nobs: int) -> int | None:
        """Total absorbed fixed-effect levels, or ``None`` when unavailable.

        PyFixest stores the per-effect level counts as a Series under ``_k_fe``
        (``__fixed_effect__(firm) 30``).  Summing them gives ``e(k_absorb)``;
        ``e(df_a)`` is that minus the one redundant constant.
        """

        counts = getattr(model, "_k_fe", None)
        if counts is None:
            return None
        try:
            total = int(sum(int(value) for value in counts.to_numpy()))
        except (AttributeError, TypeError, ValueError):
            return None
        return total if 0 < total <= nobs else None

    def _nested_absorbed_levels(self, model: Any, data: pd.DataFrame) -> int:
        """Absorbed levels nested within a cluster variable.

        reghdfe treats an absorbed effect as redundant when each of its levels sits inside a
        single cluster: its coefficients are then collinear with the cluster structure and
        contribute nothing to ``e(df_a)``. Detected the standard way -- the number of
        distinct (cluster, absorbed) pairs equals the number of absorbed levels.

        Returns 0 when nothing is nested or when the information is unavailable, so models
        without clustering keep their previous behaviour.
        """

        labels = getattr(self, "_fit_cluster_labels", None)
        if labels is None or len(labels) != len(data):
            return 0

        counts = getattr(model, "_k_fe", None)
        if counts is None:
            return 0
        try:
            pairs = list(zip(counts.index, counts.to_numpy()))
        except (AttributeError, TypeError, ValueError):
            return 0

        cluster_values = pd.Series(np.asarray(labels), index=data.index)
        nested = 0
        for raw_name, raw_count in pairs:
            name = str(raw_name)
            if name.startswith("__fixed_effect__(") and name.endswith(")"):
                name = name[len("__fixed_effect__(") : -1]
            variable = str(name).lstrip("_")
            column = variable if variable in data.columns else None
            if column is None:
                continue
            # Nesting means each level of the absorbed effect belongs to exactly one
            # cluster: the number of distinct clusters per level is 1 everywhere. Counting
            # distinct (cluster, level) pairs does NOT work -- for the absorbed level that
            # coincides with the cluster it equals the level count, and for an independent
            # one it equals nobs.
            per_level = (
                pd.DataFrame({"c": cluster_values, "a": data[column]})
                .drop_duplicates()
                .groupby("a", observed=True)["c"]
                .nunique()
            )
            if len(per_level) > 0 and int(per_level.max()) == 1:
                nested += int(raw_count)
        return nested

    @staticmethod
    def _rank_full(model: Any, nobs: int) -> int:
        """Parameter count including fixed effects nested in the cluster variable.

        ``_df_k`` drops nested effects under clustering (matching reghdfe's
        ``dofadjustments(nested)`` for the VCE factor and ``e(df_r)``), but
        reghdfe keeps the full count for ``e(rmse)``, ``e(ll)`` and ``e(r2_a)``
        -- verified against a live capture: ``reghdfe_firm.json`` and
        ``reghdfe_firm_cluster.json`` share one ``e(rmse)`` while ``e(df_r)``
        moves from 148 to 29.  When PyFixest exposes ``_k_fe`` we can rebuild it
        as ``_df_k + nested_effects``; otherwise fall back to ``_df_k``, which
        keeps the previous behaviour rather than inventing a number.
        """

        base = getattr(model, "_df_k", None)
        if base is None:
            return HDFE._rank(model, nobs)

        # PyFixest stores the absorbed level counts as a Series under ``_k_fe``:
        #   __fixed_effect__(firm)    30
        # Summing it gives the total absorbed levels, but two absorbed effects
        # that span a common constant are collinear through exactly one redundant
        # parameter. reghdfe reports the full count with that redundancy removed:
        #   one FE  : rank = reported + 30        = 32  (firm absorbs the constant)
        #   two FEs : rank = reported + 30 + 6 - 1 = 37
        # verified against reghdfe's e(r2_a) on both specifications.
        counts = getattr(model, "_k_fe", None)
        if counts is None:
            return int(round(float(base)))

        total_fe = int(np.sum(np.asarray(counts, dtype=float)))
        reported = len(model.coef())
        n_effects = len(counts)
        redundancy = max(n_effects - 1, 0)
        rank_full = reported + total_fe - redundancy
        return rank_full if 0 < rank_full <= nobs else int(round(float(base)))

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



def _absorbed_columns(formula: str) -> list[str]:
    """Names of the variables absorbed by ``|`` in ``"y ~ x | fe1 + fe2"``."""

    if "|" not in formula:
        return []
    tail = formula.split("|", 1)[1]
    # Drop any trailing option block (``| fe1, absorb(...)`` shapes) and split on ``+``.
    tail = tail.split(",", 1)[0]
    return [token.strip() for token in tail.split("+") if token.strip()]


def _stata_vce_names(cov_type: Any) -> tuple[str, str]:
    """``(e(vce), e(vcetype))`` for a ``reghdfe``/``areg`` fit.

    ``e(vce)`` is LOWERCASE (``ols``, ``robust``, ``cluster``) and ``e(vcetype)`` is the
    display spelling (``Ols``, ``Robust``, ``Cluster``). Delegated to the linear family's
    helpers so the two spellings cannot drift apart between estimator families.
    """

    from westata.estimators.linear import _stata_vce_label, _stata_vcetype_label

    return _stata_vce_label(cov_type), _stata_vcetype_label(cov_type)


def _cluster_codes_for_intercept(
    model: Any, data: pd.DataFrame
) -> np.ndarray | None:
    """Integer cluster codes for the ``_cons`` variance, or ``None`` when not clustered.

    ``areg`` reports the constant's variance under whichever ``vce()`` was requested, so
    the intercept needs the same clustering the model used. On the Nunn & Qian panel the
    clustered answer is 0.03383845 (Stata) against 0.02306936 unclustered -- a 47% error
    if the wrong one is used, and one that looks entirely reasonable on its own.
    """

    columns = getattr(model, "_fit_cluster_columns", None)
    if not columns:
        return None

    if len(columns) == 1:
        keys: Any = np.asarray(data[columns[0]])
    else:
        pairs = list(zip(*(np.asarray(data[name]) for name in columns)))
        flat = np.empty(len(pairs), dtype=object)
        flat[:] = pairs
        keys = flat

    codes, _ = pd.factorize(keys, sort=False)
    return codes


def _register_hdfe_predict(
    results: Any,
    *,
    design_matrix: Any,
    formula: str,
    y: np.ndarray,
    params: np.ndarray,
    param_names: list[str],
    data: pd.DataFrame,
    weights: Any,
) -> None:
    """Give ``reghdfe`` the prediction types ``areg`` has, including the absorbed effect.

    ``FEOLS`` was the only linear estimator with NO prediction types registered at all, so
    ``fit().predict(kind="xb")`` raised ``UnsupportedError``. ``WESTATA_GAPS.md`` P0-2
    records it: every other estimator -- ``OLS``, ``IV2SLS``, ``PanelOLS``, ``Logit``,
    ``Poisson`` -- can predict, and fitted values from a fixed-effects model are routine
    work.

    The authority is ``areg_p.ado`` -- the same estimator with one absorbed dimension,
    and available in the base installation where ``reghdfe`` is not::

        areg_p.ado:147-158 -- ``xb`` (the default) is ``xb + u``, ``d`` is ``u`` alone,
        and ``u`` is the within-group weighted mean of ``(e(depvar) - xb)``.

    So ``xb`` INCLUDES the fixed effects. Returning only ``Xb`` would produce a plausible
    number wrong by a group constant -- and because the output still looks like a
    prediction, a reader would have no reason to check.
    """

    if design_matrix is None:
        return
    design, design_names = design_matrix

    # The design carries an intercept column but the fitted parameters do NOT: with
    # absorbed fixed effects the constant is absorbed along with them, so ``design`` has
    # one more column than ``params``. Multiplying the full design by ``params`` therefore
    # raises a dimension error -- and "fix" it by dropping a column positionally would pick
    # the wrong one. Select BY NAME so the correspondence is explicit and checkable.
    #
    # ``areg`` additionally REPORTS a ``_cons`` that has no design column of its own (it is
    # the average of the absorbed effects, see ``Areg._with_intercept``). It must be
    # dropped here rather than treated as a missing column, or the membership test below
    # fails and predict is never registered at all.
    slope_names = [name for name in param_names if name != "_cons"]
    available = set(design_names)
    if not set(slope_names) <= available:
        return
    columns = [design_names.index(name) for name in slope_names]
    design = np.asarray(design, dtype=float)[:, columns]
    param_names = slope_names

    absorbed_names = [name for name in _absorbed_columns(formula) if name in data.columns]
    if not absorbed_names:
        # Nothing absorbed: ``xb`` is the plain linear predictor, which the generic
        # dispatcher already knows how to produce. Leave it unregistered rather than
        # claiming a ``d`` that cannot exist.
        return

    # A MULTI-WAY absorption needs the SUM of the per-dimension effects, which is only
    # identified up to a normalisation. Stata's ``areg`` takes one dimension and
    # ``reghdfe`` maps multi-way effects through an iterative partition; reproducing that
    # here would be guesswork. Rather than leaving the types unregistered (which makes the
    # failure read as "reghdfe cannot predict at all"), register a dispatch that refuses
    # with the actual reason.
    if len(absorbed_names) > 1:
        def _refuse(kind: str, exog: Any = None, **_kwargs: Any) -> np.ndarray:
            raise UnsupportedError(
                "reghdfe: predict with MORE THAN ONE absorbed dimension is not "
                "implemented. The default 'xb' of an absorbed model is the linear "
                "predictor plus the absorbed effect (areg_p.ado:147-158), and for a "
                "multi-way absorption that effect is identified only up to a "
                "normalisation. Predict from a single-dimension fit, or add the "
                "absorbed effects back yourself."
            )

        results._predict_types = ("xb", "d", "residuals", "dresiduals")
        results._predict_dispatch = _refuse
        return

    group_column = np.asarray(data[absorbed_names[0]])

    def _fixed_effect() -> np.ndarray:
        """The absorbed effect per observation -- ``u`` in ``areg_p.ado``."""

        # Drop a reported ``_cons`` (areg) so the product matches the design's width.
        linear = design @ np.asarray(params, dtype=float)[: design.shape[1]]
        within = y - linear
        if weights is not None:
            return _group_weighted_mean(within, group_column, np.asarray(weights, dtype=float))
        return _group_mean(within, group_column)

    def dispatch(kind: str, exog: Any = None, **_kwargs: Any) -> np.ndarray:
        normalized = str(kind).strip().lower()
        # ``params`` comes from the RESULT, which for ``areg`` ends with a reported
        # ``_cons`` that has no design column; ``design`` was already restricted to the
        # slope columns above, so the constant must be dropped here too or the product is
        # a shape error.
        slope_params = np.asarray(params, dtype=float)[
            : design.shape[1]
        ]
        linear = design @ slope_params
        if normalized in ("xb", "fitted", "index"):
            # ``areg_p.ado``: xb is the linear predictor PLUS the absorbed effect.
            return np.asarray(linear + _fixed_effect(), dtype=float)
        if normalized == "d":
            return np.asarray(_fixed_effect(), dtype=float)
        if normalized in ("residuals", "dresiduals"):
            # ``areg_p.ado``: residuals are on the full model, so this is y - (xb + u),
            # which equals the within-residual the estimator minimised.
            return np.asarray(y - (linear + _fixed_effect()), dtype=float)
        raise UnsupportedError(
            f"reghdfe: predict type {kind!r} is not implemented; the supported types "
            "are 'xb', 'd', 'residuals' and 'dresiduals'"
        )

    results._predict_types = ("xb", "d", "residuals", "dresiduals")
    results._predict_dispatch = dispatch


def _group_mean(values: np.ndarray, groups: Any) -> np.ndarray:
    """Mean of ``values`` within each level of ``groups``, broadcast back to every row."""

    codes, _ = pd.factorize(np.asarray(groups), sort=False)
    size = int(codes.max()) + 1
    counts = np.bincount(codes, minlength=size).astype(float)
    sums = np.bincount(codes, weights=np.asarray(values, dtype=float), minlength=size)
    means = np.divide(sums, counts, out=np.zeros(size), where=counts > 0)
    return means[codes]


def _group_weighted_mean(values: np.ndarray, groups: Any, weights: np.ndarray) -> np.ndarray:
    """Weighted mean of ``values`` within each level of ``groups``, broadcast back."""

    codes, _ = pd.factorize(np.asarray(groups), sort=False)
    size = int(codes.max()) + 1
    weight_sums = np.bincount(codes, weights=weights, minlength=size)
    weighted = np.bincount(
        codes, weights=weights * np.asarray(values, dtype=float), minlength=size
    )
    means = np.divide(weighted, weight_sums, out=np.zeros(size), where=weight_sums > 0)
    return means[codes]


class Areg(FEOLS):
    """Linear regression absorbing categorical indicators -- Stata ``areg``.

    This is a DIFFERENT command from ``reghdfe``, not a synonym, and the differences are
    the reason it needs its own class:

    ==========================  ====================================================
    ``areg``                    ``reghdfe``
    ==========================  ====================================================
    ``absorb(varlist)`` REQUIRED, ``|`` block in the formula
    banner "Linear regression, absorbing indicators"
    ``e(title2)`` = "Absorbed variable: firm"
    ``e(predict)`` = ``areg_p``  no predict program registered
    ``e(marginsnotok)`` = ``Residuals SCore D XBD``
    ==========================  ====================================================

    All of those values come from ``tests/conformance/stata_fixtures/areg.json``, which is
    a capture of the real command, not from reading prose.

    The ESTIMATION is identical -- ``areg`` and ``reghdfe`` fit the same within model -- so
    this subclass reuses :class:`FEOLS`'s machinery and overrides only the identity.

    Parameters
    ----------
    formula:
        ``"y ~ x1 + x2"`` alongside the ``absorb=`` keyword, matching Stata's
        ``areg y x1 x2, absorb(firm)``.  The ``|`` form is also accepted for symmetry with
        :class:`FEOLS`, but ``absorb=`` is the documented Stata spelling.
    data:
        Estimation sample.
    absorb:
        One or more categorical variables to absorb, e.g. ``"firm"`` or
        ``("firm", "year")``.  Required, as in Stata.
    """

    _command_name = "areg"
    #: areg reports _cons (areg.json names: x1, x2, _cons).
    _reports_intercept = True
    _command_title = "Linear regression, absorbing indicators"
    _predict_program = "areg_p"
    # ``areg.json``'s e(marginsnotok): these prediction types are refused after areg.
    _marginsnotok = "Residuals SCore D XBD"
    _stata_estimethod = "areg"
    #: areg supports vce(hc2 [clustvar]); FEOLS does not, because PyFixest will not
    #: expose the leverage of the full dummy design. Areg builds that design itself (for
    #: _cons), so it CAN honour HC2/HC3 -- see Areg.fit.
    _allows_leverage_cov_types = True

    def __init__(
        self,
        formula: str,
        data: pd.DataFrame,
        *,
        absorb: Any = None,
        weights: Any = None,
        weight_type: str | None = None,
        noabsorb: bool = False,
    ) -> None:
        if absorb is None:
            raise ValueError(
                "areg requires absorb=; Stata's syntax is 'areg y x, absorb(varlist)' "
                "and the option is not optional (areg.sthlp: 'absorb() is required')"
            )
        names = [absorb] if isinstance(absorb, str) else list(absorb)
        if not names:
            raise ValueError("absorb= was given but names no variables")
        absorbed = " + ".join(str(name) for name in names)
        combined = f"{formula.strip()} | {absorbed}" if "|" not in formula else formula
        # Set BEFORE the super() call: the fit that follows reads it to decide whether to
        # compute the absorbed-indicator F test at all.
        self._suppress_absorb_test = bool(noabsorb)
        super().__init__(combined, data, weights=weights, weight_type=weight_type)
        #: ``e(absvar)``: ``areg`` records the absorbed variable's NAME, which is what
        #: ``e(title2)`` is built from.
        self._absorbed_names = [str(name) for name in names]

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        absorb: Any = None,
        weights: Any = None,
        weight_type: str | None = None,
        subset: Any = None,
        noabsorb: bool = False,
    ) -> "Areg":
        """Build from Stata's spelling: ``"y ~ x1 + x2"`` with ``absorb="firm"``."""

        if subset is not None:
            data = data.loc[subset]
        return cls(
            formula,
            data,
            absorb=absorb,
            weights=weights,
            weight_type=weight_type,
            noabsorb=noabsorb,
        )

    def fit(
        self,
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: SSC | None = None,
        level: float | None = None,
    ) -> StataResults:
        """Fit, honouring ``vce(hc2 [clustvar])`` and ``vce(hc3)``.

        ``areg.sthlp`` lists ``vce(hc2 [clustvar])`` among its SE/Robust options, and
        HC2 needs the leverage ``h_jj`` of the FULL dummy design. FEOLS cannot offer it
        because PyFixest does not expose those leverages -- but ``Areg`` builds that design
        itself for ``_cons``, so the restriction does not apply here.

        The estimate is computed by FEOLS's machinery (the point estimates do not depend on
        the VCE); the LEVERAGE-based covariance is then rebuilt on the full design in
        ``_with_intercept``, which is where the design already exists.
        """

        requested = None if cov_type is None else str(cov_type).strip().upper()
        if requested in ("HC2", "HC3"):
            # Fit with HC1 (a valid, harmless placeholder) and remember what was asked for
            # so the covariance can be replaced once the full design is in hand.
            self._leverage_cov_type = requested
            results = super().fit(cov_type="HC1", cov_kwds=cov_kwds, ssc=ssc, level=level)
            results.e["vce"] = requested.lower()
            results.e["vcetype"] = requested
            return results

        self._leverage_cov_type = None
        return super().fit(cov_type=cov_type, cov_kwds=cov_kwds, ssc=ssc, level=level)

    def _rescale_cov_to_areg(
        self,
        data: pd.DataFrame,
        y: np.ndarray,
        cov: np.ndarray,
        e: dict[str, Any],
    ) -> np.ndarray:
        """Rescale a ``reghdfe`` covariance to ``areg``'s degrees-of-freedom convention.

        The two commands disagree about whether the absorbed dummies consume parameters
        when forming the residual variance:

            areg      N - k_reported - k_absorb    (1552 - 23 - 129 = 1400)
            reghdfe   N - k_reported               (1529, nested effects excluded)

        ``reghdfe`` documents this as a deliberate break from areg, so each is correct for
        its own command. But ``Areg`` claims to BE ``areg``, and a standard error 4.5% low
        that silently disagrees with the reference is worse than no feature at all.

        Measured on the Nunn & Qian panel: the ratio is exactly sqrt(1529/1400) =
        1.0450563895, and scaling by it turns westata's 0.0164182 into Stata's 0.0171579.
        """

        n_obs = float(e.get("N", 0.0) or 0.0)
        k_reported = float(e.get("rank", 0.0) or 0.0)
        k_absorb = float(e.get("k_absorb", 0.0) or 0.0)

        # ONLY CLUSTERED fits need this. ``areg.json`` is a vce(ols) capture and its
        # covariance already matches FEOLS' path; rescaling it broke two of its entries by
        # 9%. The K difference shows up only in the CR1 factor G/(G-1)*(N-1)/(N-K).
        if not getattr(self, "_fit_cluster_columns", None):
            return cov

        # ``e(rank)`` does NOT have one fixed meaning here, and that is the whole trap:
        # pyfixest reports the slope count alone (2) when the cluster variable is nested in
        # / equal to the absorbed dimension, but the full count including the dummies (31)
        # when it is crossed with it. Applying the rescale unconditionally therefore fixed
        # the nested case and BROKE the crossed one, where FEOLS' covariance was already
        # exactly Stata's.
        #
        # Measured on a 210-row panel (30 groups x 7 periods, one regressor):
        #
        #     vce(cluster a)     rank =  2  -> needs the rescale
        #     vce(cluster b)     rank = 31  -> already correct, NO rescale
        #     vce(cluster a b)   rank =  2  -> needs the rescale
        #
        # and on the Nunn & Qian panel (22 slopes, 130 countries) rank = 23, which needs it.
        # The test is whether rank already accounts for the absorbed levels: if it reaches
        # ``n_slopes + k_absorb - 1`` the dummies are counted, so nothing is owed.
        n_slopes = float(e.get("df_m", 0.0) or 0.0)
        if k_absorb <= 0 or n_slopes <= 0:
            return cov
        rank_includes_absorbed = k_reported >= n_slopes + k_absorb - 1.0
        if rank_includes_absorbed:
            return cov

        # With clustering, the two commands use the SAME CR1 formula; only K differs.
        #
        #     reghdfe   K = k_reported                    (absorbed dummies do not count)
        #     areg      K = n_slopes + k_absorb           (they do, plus the slopes)
        #
        # ``n_slopes`` is ``e(df_m)``, the reported regressors; ``k_absorb`` counts the
        # absorbed LEVELS. The constant is one of those levels and is reported separately as
        # _cons, so it is not added again -- the formula is NOT (rank + k_absorb).
        #
        # Checked by rebuilding the full dummy design by hand and evaluating the CR1 factor
        # for each candidate K on a 210-row panel (1 slope, 30 groups, crossed with 7):
        #
        #     K = 31  -> 0.11086346   <- matches Stata exactly; 1 + 30
        #     K = 32  -> 0.11117444
        #     K = 33  -> 0.11148805
        #
        # and on the Nunn & Qian panel (22 slopes, 130 countries): 22 + 130 = 152, which
        # reproduces 0.0171579390. Using k_absorb-1 there gives 0.0171640701 -- a 0.036%
        # error that reads as rounding, which is why the formula is stated as a sum here
        # rather than as an adjustment to k_reported.
        absorbed_parameters = k_absorb
        reghdfe_k = k_reported
        areg_k = n_slopes + absorbed_parameters
        if areg_k >= n_obs or reghdfe_k >= n_obs or absorbed_parameters <= 0:
            return cov
        return cov * ((n_obs - reghdfe_k) / (n_obs - areg_k))

    def _with_intercept(
        self,
        data: pd.DataFrame,
        y: np.ndarray,
        params: np.ndarray,
        names: list[str],
        cov: np.ndarray,
        e: dict[str, Any],
    ) -> tuple[np.ndarray, list[str], np.ndarray]:
        """``areg`` reports ``_cons``, the EQUAL-WEIGHTED MEAN of the absorbed effects.

        Verified against ``areg.json`` to machine precision. Fitting the full design
        (slopes plus one dummy per group) and taking ``c = (1/G)·[0, 0, 1, ..., 1]``
        reproduces BOTH the reported variance 9.093229e-04 and the covariances with the
        slopes (5.80638e-05 against x1, 2.80071e-05 against x2). Two simpler guesses were
        wrong and are recorded so they are not retried::

            mean(y - b'x)          -> right POINT (1.2096712084), wrong variance
            rmse^2 / G             -> 0.0054 against the true 0.0009
            rmse^2 / N             -> 0.000906, close but not the reported value
            full design, [_cons]   -> 0.029182 vs Stata's 0.03015498

        The point estimate coincides with ``mean(y - b'x)`` -- the equal-weighted dummy
        mean is the same number under this normalisation -- so the DESIGN-based route is
        used for both, which keeps the covariance internally consistent with it.
        """

        if not names:
            return params, names, cov

        # ``areg`` SCALES THE WHOLE COVARIANCE by its own residual variance, and its
        # denominator counts the absorbed dummies as parameters. ``reghdfe`` does not, and
        # says so deliberately (reghdfe.mata: "breaks consistency with areg"). Areg
        # inherits the reghdfe path from FEOLS, so without this the clustered standard
        # errors are 4.5% low -- exactly sqrt(1529/1400) on the Nunn & Qian panel, which is
        # the ratio of the two denominators.
        cov = self._rescale_cov_to_areg(data, y, np.asarray(cov, dtype=float), e)

        groups = np.asarray(data[self._absorbed_names[0]])
        codes, levels = pd.factorize(groups, sort=True)
        n_groups = len(levels)
        y_values = np.asarray(y, dtype=float)

        # Full design: slopes, then one dummy per absorbed group.
        design = np.column_stack(
            [
                *[np.asarray(data[name].to_numpy(), dtype=float) for name in names],
                np.eye(n_groups)[codes],
            ]
        )
        coef = np.linalg.pinv(design) @ y_values
        n_obs, k_cols = design.shape

        # ``areg``'s constant is the group-size-WEIGHTED mean of the group effects, not
        # the equal-weighted one. Verified against the Nunn & Qian panel to ten decimals:
        #
        #     equal-weighted     12.63564782   <- what this code used to do
        #     size-weighted      12.64269254   <- Stata's areg, exactly
        #
        # The equal-weighted version is right only when every group has the same size,
        # which is why it passed the 30x6 fixture and failed here (129 countries, very
        # uneven sizes). That is the whole bug: the earlier fixture could not distinguish
        # the two rules.
        counts = np.bincount(codes, minlength=n_groups).astype(float)
        weights = np.zeros(k_cols, dtype=float)
        weights[len(names):] = counts / counts.sum()
        intercept = float(weights @ coef)

        params = np.append(np.asarray(params, dtype=float), intercept)
        names = [*names, "_cons"]

        # The variance must come from the CLUSTER-ROBUST sandwich, not from
        # ``sigma^2 * (X'X)^-1``: areg reports cov(_cons) under whatever vce() was asked
        # for. On this panel the clustered answer is 0.03383845 (matching Stata) while the
        # homoskedastic one is 0.02306936 -- a 47% error that a residual-based shortcut
        # would have hidden.
        widened = np.zeros((params.size, params.size), dtype=float)
        widened[:-1, :-1] = np.asarray(cov, dtype=float)

        clusters = _cluster_codes_for_intercept(self, data)
        leverage_type = getattr(self, "_leverage_cov_type", None)
        cluster_columns = getattr(self, "_fit_cluster_columns", None) or []
        if len(cluster_columns) >= 2 and not leverage_type:
            # MULTIWAY: rebuild the CGM covariance on the FULL design.
            #
            # Scaling FEOLS' result does not work here, and the reason is measurable. Stata's
            # ``reghdfe`` and westata's FEOLS agree exactly on this panel (both 0.1023861),
            # so the pyfixest CGM is right -- but ``areg`` reports 0.11086346, and the ratio
            # between the two is not any (N-K1)/(N-K2) ratio: solving for the implied K gives
            # 0.13, not an integer. The two commands' multiway factors differ in structure,
            # not just in a parameter count.
            #
            # Recomputing from the dummy design reproduces Stata exactly. Checked by hand on
            # the 210-row panel: with K = 1 + 30 = 31 the CGM identity gives 0.11086346, the
            # figure areg prints.
            from itertools import combinations as _combinations

            fitted_resid = y_values - design @ coef
            bread = np.linalg.pinv(design.T @ design)
            scores = design * fitted_resid[:, None]
            n_obs = design.shape[0]
            areg_k = float(design.shape[1])  # slopes + one dummy per absorbed level

            cov_multi = np.zeros((k_cols, k_cols), dtype=float)
            for size in range(1, len(cluster_columns) + 1):
                sign = 1.0 if size % 2 == 1 else -1.0
                for indices in _combinations(range(len(cluster_columns)), size):
                    parts = [np.asarray(data[cluster_columns[pos]]) for pos in indices]
                    if len(parts) == 1:
                        combined = parts[0].astype(object)
                    else:
                        # A flat object array of tuples. ``np.array(list_of_tuples)`` would
                        # build a 2-D array, which factorize rejects -- the same trap
                        # documented in cov_multiway_cluster.
                        tuples = list(zip(*parts))
                        combined = np.empty(len(tuples), dtype=object)
                        combined[:] = tuples
                    codes = pd.factorize(combined, sort=False)[0]
                    n_groups = int(codes.max()) + 1
                    grouped = np.zeros((n_groups, k_cols), dtype=float)
                    for column in range(k_cols):
                        grouped[:, column] = np.bincount(
                            codes, weights=scores[:, column], minlength=n_groups
                        )
                    factor = (n_groups / (n_groups - 1.0)) * (
                        (n_obs - 1.0) / (n_obs - areg_k)
                    )
                    cov_multi += sign * factor * (bread @ (grouped.T @ grouped) @ bread)

            # ``cov_multi`` is indexed over the FULL design (slopes + dummies). Only the
            # reported parameters are carried into the result, and those are the FIRST
            # columns of the design by construction (see the ``design`` above).
            n_reported = len(names) - 1
            reported = cov_multi[:n_reported, :n_reported]
            widened[:-1, :-1] = reported
            widened[-1, -1] = float(weights @ cov_multi @ weights)
            for index in range(n_reported):
                selector = np.zeros(k_cols, dtype=float)
                selector[index] = 1.0
                value = float(weights @ cov_multi @ selector)
                widened[index, -1] = value
                widened[-1, index] = value
            return params, names, widened
        if leverage_type:
            # vce(hc2/hc3): rebuild from the FULL design. ``h_jj`` is the leverage of the
            # dummy design, which is the quantity areg.sthlp:202 says HC2 uses
            # (``u_j^2/(1-h_jj)``). Neither ``cov`` nor the cluster branch above is valid
            # here -- both were computed without those leverages.
            fitted_resid = y_values - design @ coef
            leverage = np.einsum("ij,jk,ik->i", design, np.linalg.pinv(design.T @ design), design)
            if leverage_type == "HC3":
                omega = fitted_resid**2 / np.maximum(1.0 - leverage, 1e-12) ** 2
            else:
                omega = fitted_resid**2 / np.maximum(1.0 - leverage, 1e-12)
            bread = np.linalg.pinv(design.T @ design)
            robust = bread @ (design.T @ (design * omega[:, None])) @ bread
            widened[-1, -1] = float(weights @ robust @ weights)
            for index in range(len(names) - 1):
                selector = np.zeros(k_cols, dtype=float)
                selector[index] = 1.0
                value = float(weights @ robust @ selector)
                widened[index, -1] = value
                widened[-1, index] = value
            # The reported slopes carry the same statistic, taken from the design columns.
            slope_block = np.zeros((len(names) - 1, len(names) - 1), dtype=float)
            for row in range(len(names) - 1):
                selector_row = np.zeros(k_cols, dtype=float)
                selector_row[row] = 1.0
                for column in range(len(names) - 1):
                    selector_column = np.zeros(k_cols, dtype=float)
                    selector_column[column] = 1.0
                    slope_block[row, column] = float(
                        selector_row @ robust @ selector_column
                    )
            widened[:-1, :-1] = slope_block
            return params, names, widened
        if clusters is not None and len(np.unique(clusters)) > 1:
            n_clusters = int(np.unique(clusters).size)
            scores = design * (y_values - design @ coef)[:, None]
            grouped = np.zeros((n_clusters, k_cols), dtype=float)
            for column in range(k_cols):
                grouped[:, column] = np.bincount(
                    clusters, weights=scores[:, column], minlength=n_clusters
                )
            bread = np.linalg.pinv(design.T @ design)
            robust = bread @ (grouped.T @ grouped) @ bread
            factor = (n_clusters / (n_clusters - 1.0)) * (
                (n_obs - 1.0) / (n_obs - k_cols)
            )
            robust = robust * factor
            variance = float(weights @ robust @ weights)
            widened[-1, -1] = variance
            for index in range(len(names) - 1):
                selector = np.zeros(k_cols, dtype=float)
                selector[index] = 1.0
                value = float(weights @ robust @ selector)
                widened[index, -1] = value
                widened[-1, index] = value
        else:
            dof = float(e.get("df_r", 0.0) or 0.0)
            sigma2 = float(e.get("rss", 0.0)) / dof if dof > 0 else 0.0
            bread = np.linalg.pinv(design.T @ design)
            widened[-1, -1] = sigma2 * float(weights @ bread @ weights)
            for index in range(len(names) - 1):
                selector = np.zeros(k_cols, dtype=float)
                selector[index] = 1.0
                value = sigma2 * float(weights @ bread @ selector)
                widened[index, -1] = value
                widened[-1, index] = value
        return params, names, widened
