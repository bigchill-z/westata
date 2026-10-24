"""Two-way fixed-effects difference-in-differences (Stata ``didregress``).

``didregress`` -- and its panel spelling ``xtdidregress`` -- estimates the
average treatment effect on the treated from

.. math::

    y_{it} = \\alpha_i + \\lambda_t + \\beta D_{it} + x_{it}'\\gamma
             + \\varepsilon_{it}

where ``group()`` names the unit (:math:`\\alpha_i`), ``time()`` names the
period (:math:`\\lambda_t`) and :math:`D_{it}` is the treatment indicator.  The
reported ``ATET`` is :math:`\\hat\\beta`; the default covariance clusters on the
``group()`` variable, exactly like Stata.

The model is estimated by :class:`~westata.estimators.hdfe.FEOLS`, i.e. on the
within-transformed data with PyFixest owning the alternating projections, the
``(N-1)/(N-K)``/``G/(G-1)`` factors and the clustered sandwich.  This module
only owns the Stata mapping: the ``(y x1) (treated)`` argument block, the
collinearity policy, the degrees-of-freedom convention below and the ``e()``
payload.

Absorbed, not dummied out
-------------------------
No unit or period indicator ever enters the design: the estimator works on
``y - alpha_i - lambda_t``.  ``results.design_matrix`` exposes that transformed
design (one column per reported regressor), so the absorption is inspectable
rather than merely promised.  Because the transform is Frisch-Waugh-Lovell the
ATET equals the coefficient from the explicit dummy-variable regression, which
``tests/test_did.py`` verifies against ``statsmodels``.

Why the two spellings differ in their *standard errors*
-------------------------------------------------------
Both commands print the same ATET, but the P11 fixture records different
clustered standard errors -- ``.0590465`` for ``didregress`` and ``.055228`` for
``xtdidregress`` on identical data.  That is not sampling variation, it is
Stata's degrees-of-freedom convention:

* ``didregress`` treats the data as *repeated cross-sectional*, so the
  ``group()`` dummies are ordinary regressors and every absorbed fixed-effect
  parameter counts in ``(N-1)/(N-K)``: ``K = 209`` on the fixture
  (1 constant + 1 treatment + 1 covariate + 199 unit + 7 period).
* ``xtdidregress`` knows the unit effect is *nested in the cluster variable*
  (``group()`` is the panel id), so the unit effect adds nothing to ``K`` and
  only the period effects count: ``K = 10``.

The variance therefore differs by exactly
:math:`\\sqrt{(N-K_{rcs})/(N-K_{panel})} = \\sqrt{1391/1590} = .93530`, which
turns ``.0590465`` into ``.055228`` -- both reproduced to every digit Stata
displays.  This one degrees-of-freedom switch (PyFixest's
``ssc(k_fixef=...)``) is the *only* difference between the two classes, which
is why :class:`Xtdidregress` subclasses :class:`Didregress`: they cannot drift
apart.

Conformance (fixture ``stata_fixtures/p11_did.txt``, StataNow 19.5)
-------------------------------------------------------------------
200 units x 8 periods, half the units treated from period 4; 1600
observations, 200 clusters; DGP effect 1.2.

==================================  ==========  ==============  ==================
command                              ATET        Stata ``se``    westata ``se``
==================================  ==========  ==============  ==================
``didregress (y x1) (treated)``      1.18613     .0590465        .05904651458904
``xtdidregress (y x1) (treated)``    1.18613     .055228         .05522799411606
==================================  ==========  ==============  ==================

The collinearity trap
---------------------
A *time-invariant* treatment indicator is the unit fixed effect itself: after
absorption its column is identically zero, so nothing identifies :math:`\\beta`.
Stata refuses it with ``r(198) "treatment variable omitted because of
collinearity"`` and westata raises
:class:`~westata.errors.ConsistencyError` with the same diagnosis instead of
reporting whatever the solver leaves in the dropped column.  The check runs
before estimation (no within-group variation at all) and after it (a treatment
that survives the first check but is collinear with the period effects, e.g. a
pure ``post`` dummy that shifts every unit at once).

What is Stata output and what is a westata addition
---------------------------------------------------
Stata's ``didregress`` prints only the ATET row: no R-squared, F statistic or
root MSE block.  ``e(r2)``, ``e(r2_a)``, ``e(rmse)``, ``e(F)`` and ``e(ll)``
are therefore *westata additions* carried over from the underlying
fixed-effects regression -- meaningful, but not Stata-verified.  Likewise
``e(df_m)`` is westata's convention (the number of reported coefficients: the
ATET plus the controls) because Stata publishes no ``e(df_m)`` for this
command.  Everything the fixture does display -- ATET, its clustered standard
error, the t statistic, ``e(N)``, ``e(df_r)`` and ``e(N_clust)`` -- is
reproduced.

Repeated cross-sections and covariate inference
-----------------------------------------------
The fixture command is ``didregress (y x1) (treated)``, and Stata's first
parenthesised block is "dependent variable followed by covariates".  Pass that
block verbatim::

    Didregress.from_formula("y x1", "treated", data, group="unit", time="period")

**A bare ``"y"`` means no covariates**, exactly as it does in Stata: the
``(y)`` block names only the dependent variable.  This is the reversed default
of an earlier version, which *inferred* every other time-varying numeric column
as a covariate and merely warned about it -- so the ATET silently moved when an
unrelated column happened to sit in the DataFrame (measured on the retest's
two-period panel: 1.1927931 with the column inferred against Stata's
1.1925775 for the same ``(y)`` command).  Inference must not depend on which
unrelated columns the caller happens to be carrying.

The old behaviour is available explicitly, never by default::

    Didregress.from_formula("y", "treated", data, ..., covariates="all")

``covariates="all"`` selects every time-varying numeric column that is not the
outcome, the treatment, ``group()`` or ``time()``, warns, and exposes the list as
``model.inferred_covariates``.  A time-invariant column is never selected either
way: it is the unit fixed effect.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from westata.errors import ConsistencyError
from westata.estimators.hdfe import FEOLS
from westata.protocols import SSC
from westata.results.base import StataResults

__all__ = ["Didregress", "Xtdidregress"]

COV_TYPES = ("cluster", "HC1", "nonrobust")
"""Canonical ``cov_type`` values accepted by :meth:`Didregress.fit`.

``cluster`` is Stata's default (``vce(cluster group())``); ``HC1`` is Stata's
``vce(robust)`` and ``nonrobust`` is ``vce(unadjusted)``.  Only ``cluster`` is
covered by the P11 fixture; the other two reuse
:class:`~westata.estimators.hdfe.FEOLS`'s machinery and are offered for
completeness, not as verified conformance targets.
"""

_COV_ALIASES = {
    "CLUSTER": "cluster",
    "CLUSTERED": "cluster",
    "CR1": "cluster",
    "ROBUST": "HC1",
    "HC1": "HC1",
    "NONROBUST": "nonrobust",
    "UNROBUST": "nonrobust",
    "IID": "nonrobust",
    "UNADJUSTED": "nonrobust",
}

_REJECTED_COV_TYPES = {
    "HC0": "Stata's didregress offers vce(robust) (HC1), not HC0, and HC0 is "
    "defined by the absence of the (N-1)/(N-K) factor that the DID "
    "degrees-of-freedom convention applies; ask for cov_type='HC1'",
    "HC2": "HC2/HC3 need the leverage of the full dummy design, which is not "
    "available once the fixed effects are absorbed",
    "HC3": "HC2/HC3 need the leverage of the full dummy design, which is not "
    "available once the fixed effects are absorbed",
    "TWOWAY": "didregress clusters on group(); use cov_type='cluster'",
}


# --------------------------------------------------------------------------- utils
def _variable_names(value: Any, what: str) -> list[str]:
    """Split Stata's ``"(y x1)"``-style block into individual variable names.

    ``value`` may be a whitespace-separated string (Stata's own syntax for a
    parenthesised variable list) or a sequence of names whose entries may
    themselves hold several names.
    """

    if isinstance(value, str):
        tokens = [token for token in value.split() if token]
        if not tokens:
            raise ValueError(f"{what} must name at least one variable")
        return tokens

    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        names: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise TypeError(
                    f"{what} must be a variable name or a sequence of names; "
                    f"got an entry of type {type(item).__name__}"
                )
            names.extend(token for token in item.split() if token)
        if not names:
            raise ValueError(f"{what} must name at least one variable")
        return names

    raise TypeError(
        f"{what} must be a string such as 'y x1' or a sequence of variable "
        f"names; got {type(value).__name__}"
    )


def _column_name(value: Any, what: str) -> str:
    """Validate a single column name argument."""

    if not isinstance(value, str):
        raise TypeError(
            f"{what} must be a column name (a string); got {type(value).__name__}"
        )
    name = value.strip()
    if not name:
        raise ValueError(f"{what} must be a non-empty column name")
    return name


def _normalize_cov_type(cov_type: str) -> str:
    """Return the canonical ``cov_type`` or raise ``ValueError``."""

    if not isinstance(cov_type, str):
        raise TypeError(
            "cov_type must be a string such as 'cluster', 'HC1' or "
            f"'nonrobust'; got {type(cov_type).__name__}"
        )
    key = cov_type.strip().upper()
    if key in _REJECTED_COV_TYPES:
        raise ValueError(
            f"cov_type {cov_type!r} is not supported by the DID estimators: "
            + _REJECTED_COV_TYPES[key]
            + ". Supported cov_type values are: "
            + ", ".join(COV_TYPES)
        )
    try:
        return _COV_ALIASES[key]
    except KeyError:
        raise ValueError(
            f"Unknown cov_type {cov_type!r} for the DID estimators; supported "
            "cov_type values are: " + ", ".join(COV_TYPES)
        ) from None


def _collinearity_error(treatment: str, group: str, detail: str) -> ConsistencyError:
    """Build Stata's ``r(198)`` "omitted because of collinearity" error."""

    return ConsistencyError(
        f"treatment variable {treatment!r} was omitted because of collinearity. "
        + detail
        + ' Stata stops in exactly this situation with r(198) "treatment '
        'variable omitted because of collinearity", and westata refuses to '
        "report a coefficient that the fixed effects cannot identify. Pass the "
        "time-varying treatment indicator (the group x post interaction), not "
        f"the time-invariant group indicator; the latter is the group({group}) "
        "fixed effect itself."
    )


# ------------------------------------------------------------------- FE plumbing
class _DidFEOLS(FEOLS):
    """``FEOLS`` with a DID degrees-of-freedom convention and design capture.

    Everything numerical -- the within transform, the clustered sandwich, the
    ``e(rank)``/``e(df_r)`` reconstruction -- stays in
    :class:`~westata.estimators.hdfe.FEOLS`.  This subclass adds the single
    DID-specific decision (:attr:`K_FIXEF`, i.e. whether the absorbed
    fixed-effect parameters count in ``(N-1)/(N-K)``) plus the retained
    within-transformed design.
    """

    #: PyFixest ``ssc(k_fixef=...)``: ``"full"`` (repeated cross-section,
    #: Stata's ``didregress``) or ``"nonnested"`` (panel, ``xtdidregress``).
    K_FIXEF = "nonnested"

    #: The Stata command this subclass reproduces; written onto the result as
    #: ``results.model_name`` so summaries show ``didregress`` rather than the
    #: underlying ``reghdfe``.
    MODEL_NAME = "didregress"

    def __init__(self, formula: str, data: pd.DataFrame) -> None:
        super().__init__(formula, data)
        self.capture: dict[str, Any] = {}

    @classmethod
    def _pyfixest_ssc(  # type: ignore[override]
        cls, ssc: SSC | None, cov_type: str
    ) -> dict[str, str | bool]:
        """``FEOLS``'s SSC mapping with the DID ``k_fixef`` convention."""

        settings = FEOLS._pyfixest_ssc(ssc, cov_type)
        settings["k_fixef"] = cls.K_FIXEF
        return settings

    def _build_results(
        self,
        model: Any,
        data: pd.DataFrame,
        cov_type: str,
        spec: Any,
    ) -> StataResults:
        results = super()._build_results(model, data, cov_type, spec)

        names = [str(name) for name in model.coef().index]
        design = np.asarray(model._X, dtype=float)
        if design.ndim != 2 or design.shape[1] != len(names):
            raise ConsistencyError(
                "didregress: PyFixest returned a within-transformed design of "
                f"shape {design.shape} for {len(names)} coefficients; westata "
                "cannot prove that the fixed effects were absorbed"
            )
        nobs = int(results.nobs)
        # `model._G` is the per-dimension cluster count and is *empty* for the
        # non-clustered covariances, so it is checked for size, not just `None`.
        groups = getattr(model, "_G", None)
        n_clusters: int | None = None
        if groups is not None and np.size(groups):
            n_clusters = int(np.min(groups))
        self.capture = {
            "design": design,
            "design_names": names,
            # Under clustering `_df_k` reflects this subclass's `k_fixef`: 209
            # for the repeated cross-section, 10 for the panel.  Without
            # clustering the nested-effect rule does not apply and PyFixest
            # counts all 209 regardless -- correctly, because `N - K` must then
            # pay for every absorbed parameter.
            "rank": FEOLS._rank(model, nobs),
            "n_clusters": n_clusters,
        }
        # FEOLS labels every result "reghdfe", and StataResults exposes no
        # model-name setter (results/base.py is outside this module's write
        # scope), so the command label is applied here.
        results._model_name = self.MODEL_NAME
        return results


class _RepeatedCrossSectionFEOLS(_DidFEOLS):
    """Stata ``didregress``: unit dummies are ordinary regressors (``K`` full)."""

    K_FIXEF = "full"
    MODEL_NAME = "didregress"


class _PanelFEOLS(_DidFEOLS):
    """Stata ``xtdidregress``: the unit effect is nested in ``group()``."""

    K_FIXEF = "nonnested"
    MODEL_NAME = "xtdidregress"


# -------------------------------------------------------------------- estimator
class Didregress:
    """Two-way fixed-effects difference-in-differences (Stata ``didregress``).

    Parameters
    ----------
    outcome:
        Stata's first parenthesised block: the dependent variable, optionally
        followed by covariates -- ``"y"`` or ``"y x1 x2"``.  A sequence of
        names is accepted as well.
    treatment:
        Stata's second block: exactly one treatment indicator.  It must be 0/1
        and vary within units; see the module docstring for the collinearity
        trap.
    data:
        Estimation data in long (unit x period) form.
    group:
        Column naming the unit whose fixed effect is absorbed and on which the
        standard errors are clustered.
    time:
        Column naming the period whose fixed effect is absorbed.
    covariates:
        Optional explicit covariate names, added to the tokens of ``outcome``.
        ``None`` (the default) adds no covariate, matching Stata's ``(y)``
        block.  Pass ``covariates="all"`` to use every remaining time-varying
        numeric column (the previous, opt-in behaviour) -- that path warns and
        reports the list through :attr:`inferred_covariates`.

    Examples
    --------
    >>> result = Didregress.from_formula(              # doctest: +SKIP
    ...     "y x1", "treated", panel, group="unit", time="period"
    ... ).fit()
    >>> float(result.params["treated"])                # the ATET
    1.1861296209175862
    """

    #: Stata command reproduced by this class (also ``results.model_name``).
    model_name = "didregress"

    #: Stata's "Data type:" label.
    data_type = "Repeated cross-sectional"

    #: ``FEOLS`` subclass carrying the degrees-of-freedom convention.
    fe_class: type[_DidFEOLS] = _RepeatedCrossSectionFEOLS

    def __init__(
        self,
        outcome: Any,
        treatment: Any,
        data: pd.DataFrame,
        *,
        group: str,
        time: str,
        covariates: Any = None,
    ) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"data must be a pandas DataFrame; got {type(data).__name__}"
            )

        outcome_names = _variable_names(outcome, "outcome")
        treatment_names = _variable_names(treatment, "treatment")
        if len(treatment_names) != 1:
            raise ValueError(
                "the DID estimators take exactly one treatment variable; got "
                f"{len(treatment_names)} ({', '.join(treatment_names)})"
            )
        treatment_name = treatment_names[0]
        group_name = _column_name(group, "group")
        time_name = _column_name(time, "time")
        if group_name == time_name:
            raise ValueError(
                f"group and time must name different columns; both are {group_name!r}"
            )

        depvar = outcome_names[0]
        explicit = list(outcome_names[1:])
        infer_all = False
        if isinstance(covariates, str) and covariates.strip().lower() == "all":
            infer_all = True
        elif covariates is not None:
            explicit.extend(_variable_names(covariates, "covariates"))
        if treatment_name == depvar:
            raise ValueError(
                f"the dependent variable and the treatment variable are both {depvar!r}"
            )
        for role, name in (
            ("outcome", depvar),
            ("treatment", treatment_name),
            ("group", group_name),
            ("time", time_name),
        ):
            if name in explicit:
                raise ValueError(
                    f"{name!r} is named both as the {role} and as a covariate"
                )

        self._data = data.copy().reset_index(drop=True)
        self._depvar = depvar
        self._treatment = treatment_name
        self._group = group_name
        self._time = time_name

        required = [depvar, treatment_name, group_name, time_name, *explicit]
        missing = [name for name in required if name not in self._data]
        if missing:
            raise ValueError(
                "the data do not contain "
                + ", ".join(repr(name) for name in missing)
                + "; available columns are "
                + ", ".join(map(str, self._data.columns))
            )

        if explicit:
            self._covariates = self._dedupe(explicit)
            self._inferred_covariates: list[str] = []
        elif infer_all:
            self._covariates = self._infer_covariates()
            self._inferred_covariates = list(self._covariates)
            if self._covariates:
                warnings.warn(
                    "didregress: covariates='all' asked for every remaining "
                    "time-varying numeric column, so "
                    + ", ".join(self._covariates)
                    + " enter the regression. Stata's ("
                    + depvar
                    + ") block names no covariate and adjusts for none; pass "
                    + repr(f"{depvar} {' '.join(self._covariates)}")
                    + " if that is what you want, so the specification is explicit.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        else:
            # Stata's ``(y)`` block names no covariate, so none is added.  The old
            # behaviour inferred every time-varying column and only warned, which
            # made the ATET depend on unrelated columns in the frame.
            self._covariates = []
            self._inferred_covariates = []

        self._columns = self._dedupe(
            [depvar, *self._covariates, treatment_name, group_name, time_name]
        )
        self._check_numeric([depvar, *self._covariates, treatment_name])

        #: Observations dropped by listwise deletion in the last :meth:`fit`.
        self._n_dropped_missing = 0
        #: Rows in the estimation sample of the last :meth:`fit`.
        self._n_used = 0

    # ----------------------------------------------------------- construction
    @classmethod
    def from_formula(
        cls,
        outcome: Any,
        treatment: Any,
        data: pd.DataFrame,
        *,
        group: str,
        time: str,
        covariates: Any = None,
    ) -> "Didregress":
        """Build a model from Stata's ``didregress (y x1) (treated)`` spelling.

        Nothing is estimated until :meth:`fit` is called.
        """

        return cls(
            outcome, treatment, data, group=group, time=time, covariates=covariates
        )

    @staticmethod
    def _dedupe(names: Sequence[str]) -> list[str]:
        seen: list[str] = []
        for name in names:
            if name not in seen:
                seen.append(name)
        return seen

    def _infer_covariates(self) -> list[str]:
        """Time-varying numeric columns other than the model's own variables.

        A column that never varies within a group *is* the unit fixed effect, so
        it is skipped: it could not be estimated separately from :math:`\\alpha_i`
        and would only trigger a collinearity warning.
        """

        reserved = {self._depvar, self._treatment, self._group, self._time}
        inferred: list[str] = []
        for name in self._data.columns:
            # Only string column names are addressable by Stata-style variable
            # lists, so those are the only candidates.
            if not isinstance(name, str) or name in reserved:
                continue
            column = self._data[name]
            if not pd.api.types.is_numeric_dtype(column):
                continue
            spread = column.groupby(self._data[self._group]).nunique(dropna=True)
            if int(spread.max()) < 2:
                continue
            inferred.append(name)
        return inferred

    def _check_numeric(self, names: Sequence[str]) -> None:
        """The outcome, the covariates and the treatment must be numeric."""

        offenders = [
            name
            for name in names
            if not pd.api.types.is_numeric_dtype(self._data[name])
        ]
        if offenders:
            raise TypeError(
                "didregress: "
                + ", ".join(repr(name) for name in offenders)
                + " must be numeric; convert the variable first (Stata would "
                "also refuse a string dependent variable or a string treatment "
                "indicator)"
            )

    # --------------------------------------------------------------- metadata
    @property
    def depvar(self) -> str:
        """Name of the dependent variable."""

        return self._depvar

    @property
    def covariates(self) -> list[str]:
        """Names of the covariates (possibly empty)."""

        return list(self._covariates)

    @property
    def inferred_covariates(self) -> list[str]:
        """Covariates selected by ``covariates="all"`` (empty otherwise)."""

        return list(self._inferred_covariates)

    @property
    def treatment(self) -> str:
        """Name of the treatment indicator."""

        return self._treatment

    @property
    def group(self) -> str:
        """Name of the group variable (also the cluster variable)."""

        return self._group

    @property
    def time(self) -> str:
        """Name of the time variable."""

        return self._time

    @property
    def formula(self) -> str:
        """The PyFixest formula actually estimated (fixed effects absorbed)."""

        regressors = " + ".join([self._treatment, *self._covariates])
        return f"{self._depvar} ~ {regressors} | {self._group} + {self._time}"

    @property
    def n_dropped_missing(self) -> int:
        """Observations dropped for missing values in the last :meth:`fit`."""

        return self._n_dropped_missing

    @property
    def n_used(self) -> int:
        """Rows in the estimation sample of the last :meth:`fit`."""

        return self._n_used

    def __repr__(self) -> str:
        covariates = f" {' '.join(self._covariates)}" if self._covariates else ""
        return (
            f"<{type(self).__name__} {self.model_name}: "
            f"({self._depvar}{covariates}) ({self._treatment}) "
            f"| group({self._group}) time({self._time})>"
        )

    # ------------------------------------------------------------------- fit
    def fit(self, cov_type: str = "cluster") -> StataResults:
        """Estimate the model and return a Stata-conformant result object.

        Parameters
        ----------
        cov_type:
            ``"cluster"`` (the default: Stata's ``vce(cluster group())``),
            ``"HC1"`` (``vce(robust)``) or ``"nonrobust"``
            (``vce(unadjusted)``).

        Returns
        -------
        StataResults
            Coefficients -- the ATET first, then the covariates -- with
            ``e(N)``/``e(df_m)``/``e(df_r)``/``e(N_clust)``, the
            within-transformed ``design_matrix`` and the estimation-sample
            ``model_data``.

        Raises
        ------
        ConsistencyError
            If the treatment is collinear with the absorbed fixed effects
            (including the time-invariant-treatment trap), is not a 0/1
            indicator, or has only one arm present.
        """

        canonical = _normalize_cov_type(cov_type)
        sample = self._estimation_sample()
        self._require_identification(sample)

        model = self.fe_class(self.formula, sample)
        cov_kwds = {"cluster": self._group} if canonical == "cluster" else None

        # FEOLS reports absorbed regressors as a RuntimeWarning.  Capture them
        # so an absorbed *treatment* becomes the error Stata raises, while a
        # genuinely-collinear covariate keeps its warning.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = model.fit(cov_type=canonical, cov_kwds=cov_kwds)

        absorbed = [str(name) for name in (getattr(results, "absorbed", None) or [])]
        if self._treatment in absorbed:
            raise _collinearity_error(
                self._treatment,
                self._group,
                "The estimator dropped it from the within-transformed design "
                "because its column is a linear combination of the absorbed "
                "fixed effects.",
            )
        for record in caught:
            warnings.warn(str(record.message), record.category, stacklevel=2)

        self._finalize(results, model.capture, canonical, sample)
        return results

    # -------------------------------------------------------------- sampling
    def _estimation_sample(self) -> pd.DataFrame:
        """Return the complete-case sample Stata would estimate on."""

        used = self._data[self._columns]
        missing = used.isna().any(axis=1)
        self._n_dropped_missing = int(missing.sum())
        sample = (
            self._data.loc[~missing].reset_index(drop=True)
            if self._n_dropped_missing
            else self._data.copy()
        )
        if sample.empty:
            raise ConsistencyError(
                "didregress: no observations are left after dropping missing "
                "values in " + ", ".join(map(repr, self._columns))
            )
        self._n_used = int(sample.shape[0])
        self._validate_sample(sample)
        return sample

    def _validate_sample(self, sample: pd.DataFrame) -> None:
        """Check the sample shape both spellings require."""

        n_groups = int(sample[self._group].nunique())
        n_periods = int(sample[self._time].nunique())
        if n_groups < 2:
            raise ConsistencyError(
                f"didregress: group({self._group}) has {n_groups} distinct "
                "value(s); a difference-in-differences needs at least two "
                "groups (one treated, one control)"
            )
        if n_periods < 2:
            raise ConsistencyError(
                f"didregress: time({self._time}) has {n_periods} distinct "
                "value(s); a difference-in-differences needs at least two periods"
            )
        self._check_treatment_indicator(sample)

    def _check_treatment_indicator(self, sample: pd.DataFrame) -> None:
        """Stata's ``didregress`` requires a 0/1 treatment with both arms."""

        values = np.asarray(sample[self._treatment], dtype=float)
        distinct = np.unique(values)
        if distinct.size and not np.all(np.isin(distinct, (0.0, 1.0))):
            shown = ", ".join(f"{value:g}" for value in distinct[:6])
            if distinct.size > 6:
                shown += ", ..."
            raise ConsistencyError(
                f"didregress: the treatment variable {self._treatment!r} must "
                "be a 0/1 indicator (Stata's didregress estimates the ATET of a "
                f"binary treatment); it takes {distinct.size} distinct values: "
                f"{shown}"
            )
        if distinct.size < 2:
            arm = "treated" if distinct.size and distinct[0] == 1.0 else "control"
            raise ConsistencyError(
                f"didregress: every observation is a {arm} observation "
                f"({self._treatment!r} is constantly {distinct[0]:g}); a "
                "difference-in-differences needs both a treated and a control "
                "group"
            )

    def _require_identification(self, sample: pd.DataFrame) -> None:
        """Reject a treatment that the group fixed effect can absorb."""

        spread = sample.groupby(self._group, sort=False)[self._treatment].nunique()
        if bool((spread <= 1).all()):
            raise _collinearity_error(
                self._treatment,
                self._group,
                f"The treatment never changes within a {self._group!r}: each of "
                f"the {spread.size} groups takes a single value for it, so "
                "after the within transformation its column is exactly zero.",
            )
        self._check_panel_cells(sample)

    def _check_panel_cells(self, sample: pd.DataFrame) -> None:
        """Hook for the panel spelling; the pooled one accepts any layout."""

    # --------------------------------------------------------------- results
    def _finalize(
        self,
        results: StataResults,
        capture: dict[str, Any],
        cov_type: str,
        sample: pd.DataFrame,
    ) -> None:
        """Attach the DID ``e()`` conventions and the estimation provenance."""

        nobs = float(results.nobs)
        rank = float(capture["rank"])
        updates: dict[str, Any] = {
            "N": nobs,
            # Stata publishes no e(df_m) for didregress; westata reports the
            # number of coefficients shown in the table (ATET + controls), which
            # is exactly what the joint F statistic tests.
            "df_m": float(len(results.param_names)),
            "rank": rank,
            "estimand": "ATET",
            "data_type": self.data_type,
        }
        if cov_type == "cluster":
            # Verified against the fixture's 95% interval: .0590465 * t(199).
            # `rank` is the *nested* count here, so the two spellings differ.
            updates["df_r"] = float(results.df_resid)
            if capture["n_clusters"] is not None:
                updates["N_clust"] = float(capture["n_clusters"])
        else:
            # No clustering, so no nested-effect reduction: e(df_r) is N - K with
            # every absorbed fixed-effect parameter counted, which is what
            # `capture["rank"]` already holds for both spellings.
            updates["df_r"] = nobs - rank
        results.e.update(updates)

        # Provenance: the within-transformed design (no dummy columns) and the
        # rows actually used, so postestimation and audits need not rebuild the
        # absorption.
        results.design_matrix = (capture["design"], list(capture["design_names"]))
        results.model_data = sample.reset_index(drop=True)


class Xtdidregress(Didregress):
    """Panel difference-in-differences (Stata ``xtdidregress``).

    The estimator is the same two-way fixed-effects regression as
    :class:`Didregress`, and the ATET is identical on a balanced panel.  Only
    the degrees-of-freedom convention differs, and for the reason Stata
    documents: a panel unit effect is nested in the ``group()`` cluster
    variable, so it adds nothing to the ``(N-1)/(N-K)`` factor.  The clustered
    standard error is therefore the smaller one Stata prints -- ``.055228``
    against ``.0590465`` for ``didregress`` on the P11 fixture.

    Unlike ``didregress`` this spelling requires a genuine panel: each
    ``(group, time)`` cell may occur at most once, matching Stata's ``xtset``
    requirement ("repeated time values within panel").  Unbalanced panels are
    estimated (the within transform handles them); the exact agreement with
    :class:`Didregress` is guaranteed, and tested, for balanced panels.
    """

    model_name = "xtdidregress"
    data_type = "Longitudinal"
    fe_class: type[_DidFEOLS] = _PanelFEOLS

    def _check_panel_cells(self, sample: pd.DataFrame) -> None:
        repeated = sample.duplicated(subset=[self._group, self._time])
        if bool(repeated.any()):
            raise ConsistencyError(
                "xtdidregress: the panel is not identified by "
                f"({self._group}, {self._time}): {int(repeated.sum())} "
                "observation(s) repeat a (group, time) cell. Stata's "
                "xtdidregress requires xtset data without repeated time values "
                "within a panel; aggregate or deduplicate the data first, or "
                "use didregress for a genuinely repeated cross-section."
            )
