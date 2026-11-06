"""statsmodels-compatible results container with Stata ``e()`` scalars.

``StataResults`` is the single results object returned by every westata
estimator.  It implements the surface statsmodels downstream code expects
(``params``/``bse``/``tvalues``/``pvalues``/``conf_int``/``cov_params``/...) and
adds the Stata post-estimation scalars via :attr:`StataResults.e`.

The container is intentionally dumb: it renders what it is given.  Degrees of
freedom, which distribution to use for inference, and the ``e()`` scalars are
decided by the estimator (the Stata-conformance layer), not here.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy import stats

from westata.errors import ConsistencyError
from westata.results.stata_return import EReturn
from westata.results.summary import render_summary

__all__ = ["StataResults"]

_VALID_DISTRIBUTIONS = ("t", "z")


def _as_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


class StataResults:
    """Estimation results with a statsmodels-compatible API.

    Parameters
    ----------
    params:
        Point estimates, length ``k``.
    param_names:
        Variable names, length ``k``; used as the index of every coefficient
        Series.
    cov:
        Covariance matrix (``k x k``) already including all requested
        small-sample corrections.
    resid, fittedvalues:
        Observation-level residuals and fitted values, length ``nobs``.
    nobs, df_resid, df_model:
        Sample size and degrees of freedom.  ``df_resid`` is the reference
        distribution's degrees of freedom (for clustered VCE it is ``G - 1``,
        not necessarily ``nobs - k``).
    dist:
        ``"t"`` for coefficient t tests / t confidence intervals, ``"z"`` for
        normal tests (logit, probit, Poisson, ...).
    model_name:
        Label shown in summaries (``"OLS"``, ``"Logit"``, ...).
    cov_type:
        Covariance estimator label recorded for display and auditing.
    e:
        Optional Stata ``e()`` scalars.  They are merged over the computed
        defaults ``N``, ``df_m`` and ``df_r`` (an explicit ``e()`` value wins).
    """

    def __init__(
        self,
        *,
        params: np.ndarray,
        param_names: list[str],
        cov: np.ndarray,
        resid: np.ndarray,
        fittedvalues: np.ndarray,
        nobs: int,
        df_resid: float,
        df_model: float,
        dist: str = "t",
        model_name: str = "",
        cov_type: str = "nonrobust",
        e: Mapping[str, Any] | None = None,
        design_matrix: tuple[np.ndarray, list[str]] | None = None,
        model_data: pd.DataFrame | None = None,
        level: float | None = None,
    ) -> None:
        names = list(param_names)
        params_arr = np.asarray(params, dtype=float).reshape(-1)
        cov_arr = np.asarray(cov, dtype=float)
        resid_arr = np.asarray(resid, dtype=float).reshape(-1)
        fitted_arr = np.asarray(fittedvalues, dtype=float).reshape(-1)

        if params_arr.size != len(names):
            raise ValueError(
                f"params has {params_arr.size} entries but param_names has {len(names)}"
            )
        if len(set(names)) != len(names):
            raise ValueError("param_names must be unique")
        if cov_arr.shape != (len(names), len(names)):
            raise ValueError(
                f"cov must have shape ({len(names)}, {len(names)}), got {cov_arr.shape}"
            )

        nobs_int = int(nobs)
        if nobs_int != nobs or nobs_int < 1:
            raise ValueError(f"nobs must be a positive integer, got {nobs!r}")
        if resid_arr.size != nobs_int:
            raise ValueError(
                f"resid has {resid_arr.size} entries but nobs is {nobs_int}"
            )
        if fitted_arr.size != nobs_int:
            raise ValueError(
                f"fittedvalues has {fitted_arr.size} entries but nobs is {nobs_int}"
            )

        dist_value = str(dist).strip().lower()
        if dist_value not in _VALID_DISTRIBUTIONS:
            raise ValueError(
                f"dist must be one of {_VALID_DISTRIBUTIONS}, got {dist!r}"
            )

        df_resid_value = float(df_resid)
        df_model_value = float(df_model)

        if level is None:
            level_value = 0.95
        else:
            level_value = float(level)
            # Stata's level() takes a PERCENTAGE (level(90)); accept either spelling so a
            # caller cannot accidentally ask for level(0.95) and silently get a 0.95%
            # interval, whose critical value would be absurd and whose symptom (narrower
            # intervals than expected) is easy to miss.
            if 1.0 < level_value <= 100.0:
                level_value = level_value / 100.0
            if not 0.0 < level_value < 1.0:
                raise ValueError(
                    f"level must lie in (0, 1) as a fraction or (1, 100] as a "
                    f"percentage, got {level!r}"
                )

        self._param_names = names
        self._nobs = nobs_int
        self._df_resid = df_resid_value
        self._df_model = df_model_value
        self._dist = dist_value
        # Confidence level for the printed intervals, as a FRACTION (0.95 for Stata's
        # level(95)). Stata's ``level(#)`` takes a percentage; the models accept the
        # percentage and store the fraction here, so ``conf_int`` and the summary agree on
        # one representation.
        self._level = level_value
        self._model_name = str(model_name)
        self._cov_type = str(cov_type)
        self._cov = cov_arr
        self._resid = resid_arr
        self._fittedvalues = fitted_arr

        merged_e: dict[str, Any] = {
            "N": nobs_int,
            "df_m": df_model_value,
        }
        # ``e(df_r)`` is NOT universal.  ``logit``/``poisson``/``nbreg``/``ologit``
        # do not document it (only ``e(df_m)``), because an ML model's tests are
        # z-tests with no residual degrees of freedom; ``tobit`` DOES document it.
        # Injecting it unconditionally put a fabricated 200.0 in e() for a logit
        # fit, where Stata has no such key at all.
        # ``e(df_r)`` is NOT universal.  ``logit``/``poisson``/``nbreg``/``ologit``
        # do not document it (only ``e(df_m)``), because an ML model's tests are
        # z-tests with no residual degrees of freedom; ``tobit`` DOES document it,
        # and so does ``xtivreg`` -- even though both report z statistics.
        #
        # So the decision cannot be made from ``dist`` alone.  An estimator that
        # passes ``df_r`` in its own e() dict is stating that Stata stores it;
        # only the unconditional injection is suppressed for z-based models.
        explicit_df_r = e is not None and "df_r" in dict(e)
        if explicit_df_r or str(dist_value or "").strip().lower() != "z":
            merged_e["df_r"] = df_resid_value
        if e is not None:
            merged_e.update(dict(e))
        self._e = EReturn(merged_e)
        #: Retained estimation design (X, names) so postestimation can rebuild
        #: predictions, marginal effects and adjusted means without asking the
        #: user to re-supply the data.
        self.design_matrix = design_matrix
        #: Estimation-sample covariates, needed by ``margins(over=...)`` to
        #: resolve group labels given as column names.
        self.model_data = model_data

        index = pd.Index(names, dtype=object)
        self._params = pd.Series(params_arr, index=index, name="params")
        self._bse = pd.Series(np.sqrt(np.diag(cov_arr)), index=index, name="bse")
        with np.errstate(divide="ignore", invalid="ignore"):
            tvalues_arr = params_arr / self._bse.to_numpy()
        self._tvalues = pd.Series(tvalues_arr, index=index, name="tvalues")

        if dist_value == "t":
            pvalues_arr = 2.0 * stats.t.sf(np.abs(tvalues_arr), df_resid_value)
        else:
            pvalues_arr = 2.0 * stats.norm.sf(np.abs(tvalues_arr))
        self._pvalues = pd.Series(pvalues_arr, index=index, name="pvalues")

    # ------------------------------------------------------------------
    # Coefficient surface
    # ------------------------------------------------------------------
    @property
    def param_names(self) -> list[str]:
        """Variable names in coefficient order."""

        return list(self._param_names)

    # ------------------------------------------------------------------
    # Stata's coefficient ordering
    # ------------------------------------------------------------------
    #
    # ``regress mpg weight foreign`` reports ``e(b)`` in the order
    # ``weight foreign _cons`` -- the CONSTANT LAST.  patsy, which builds this
    # package's design matrices, puts the intercept FIRST and spells the factor
    # level ``foreign[T.Foreign]``.
    #
    # The numbers agree exactly once reordered: ``regress_auto.json``'s ``b``
    # matches to 1.6e-12 and its ``V`` to 1.6e-15.  But a caller who compares
    # ``result.params`` against Stata's ``e(b)`` directly sees a difference of
    # ~41.7 in the first element and concludes the fit is wrong.  These accessors
    # remove that trap.
    #
    # ``VARResults`` already had its own ``stata_params``/``stata_cov_params``
    # for the same reason (Stata orders a VAR equation by equation); this is the
    # equivalent for the single-equation results.

    _STATA_CONSTANT_NAMES = frozenset(
        {"intercept", "intercept_", "_cons", "const"}
    )

    def _stata_order(self) -> list[int]:
        """Indices that reorder ``param_names`` into Stata's ``e(b)`` order.

        Stata's layout, in order:

        1. the regressors, in the order the formula wrote them;
        2. ``_cons``, if the model has one;
        3. the auxiliary parameters -- ``cut1``.. (ordinal models) and
           ``lnalpha`` (negative binomial) -- LAST.

        Step 3 is confirmed by ``nbreg.json``, whose names are
        ``['x1', '_cons', 'lnalpha']``: the dispersion follows the constant, it
        does not sit among the regressors.  The ``ologit.json`` names
        ``['x1', 'x2', 'cut1', 'cut2']`` put the cuts after the slopes for the
        same reason.

        patsy alphabetises, so the written order is recovered from the
        estimator's ``_formula_rhs_order`` when available; without it the slope
        order is whatever patsy produced.
        """

        names = [str(name) for name in self._param_names]

        # MULTI-EQUATION models (``zip``, ``zinb``, ``heckman``, ``etregress``) store the
        # second equation's parameters prefixed. Stata prints one BLOCK per equation, each
        # ending with its own constant:
        #
        #     y            |          inflate      |
        #            x1    |                 x1    |
        #          _cons   |               _cons   |
        #
        # Sorting all constants to the very end put the main equation's _cons AFTER the
        # inflate block, which is not Stata's layout. Each equation is sorted on its own.
        #: Stata's equation names for multi-equation models. An interaction term also
        #: contains ``#`` (``weight#length``), so the marker alone cannot decide: only a
        #: KNOWN equation name counts, otherwise ``weight#length`` would be treated as an
        #: equation called "weight" and the constant would be reordered wrongly.
        _EQUATION_NAMES = frozenset({"inflate", "select", "tobit", "y1", "y2", "y3", "y4"})

        def _equation_of(name: str) -> str:
            for marker in ("#", ":"):
                if marker in name:
                    head = name.split(marker, 1)[0]
                    if head in _EQUATION_NAMES:
                        return head
            return ""

        equations = list(dict.fromkeys(_equation_of(name) for name in names))
        if len(equations) > 1:
            ordered_all: list[int] = []
            for equation in equations:
                members = [
                    index
                    for index, name in enumerate(names)
                    if _equation_of(name) == equation
                ]

                def _bare(index: int) -> str:
                    text = names[index]
                    for marker in ("#", ":"):
                        if marker in text:
                            text = text.split(marker, 1)[1]
                    return text.strip().lower()

                ordered_all.extend(
                    index
                    for index in members
                    if _bare(index) not in self._STATA_CONSTANT_NAMES
                )
                ordered_all.extend(
                    index
                    for index in members
                    if _bare(index) in self._STATA_CONSTANT_NAMES
                )
            return ordered_all

        constants = [
            index
            for index, name in enumerate(names)
            if name.strip().lower() in self._STATA_CONSTANT_NAMES
        ]
        # Auxiliary parameters: Stata stores them without the display slash, so
        # recognise the leading "/" of the internal spelling. ``var(e.<depvar>)`` is
        # tobit's scale and carries no slash at all. The variance/dispersion parameters of
        # the random-effects ML commands are stored as ``lnsig2u``/``lnalpha`` and are
        # displayed WITH a slash -- Stata orders them after the constant:
        #
        #   x1 / _cons / /lnsig2u / sigma_u / rho
        auxiliary = [
            index
            for index, name in enumerate(names)
            if name.startswith("/")
            or name.startswith("cut")
            or name.startswith("var(e.")
            or name in ("lnsig2u", "lnalpha", "lnsigma")
        ]
        taken = set(constants) | set(auxiliary)
        slopes = [index for index in range(len(names)) if index not in taken]

        written = getattr(self, "_formula_rhs_order", None) or []
        if written:
            def _base(name: str) -> str:
                return str(name).split("[T.", 1)[0]

            ordered: list[int] = []
            remaining = list(slopes)
            for variable in written:
                for index in list(remaining):
                    if _base(names[index]) == variable:
                        ordered.append(index)
                        remaining.remove(index)
                        break
            # Anything the formula did not name plainly (interactions, I(...))
            # keeps patsy's relative order after the named terms.
            slopes = ordered + remaining

        return slopes + constants + auxiliary

    @property
    def stata_factor_terms(self) -> list[str]:
        """Factor terms whose Stata ``e(b)`` spelling cannot be reconstructed.

        Stata expands an explicit factor into one coefficient per LEVEL plus an
        explicit base row, so ``regress mpg weight i.foreign`` reports FOUR
        entries -- ``weight``, ``0b.foreign``, ``1.foreign``, ``_cons`` -- with
        the base recorded as 0 (``regress_auto_factor.json`` shows exactly that:
        ``b = [-0.00659, 0.0, -1.65003, 41.6797]``).

        A plain binary variable is different: ``regress mpg weight foreign``
        reports a single ``foreign`` row naming the level, which maps one-to-one
        onto this package's ``foreign[T.Foreign]`` column. That case is fine and
        is NOT reported here.

        The discriminating question is therefore "is this an explicit factor
        expansion, or a 0/1 variable?"  patsy spells both as ``[T.<level>]``, so
        the check uses the source the caller wrote: a ``C(...)`` term is explicit,
        a bare column whose values are all 0/1 is a dummy.

        Returns
        -------
        list[str]
            Empty when every coefficient has an unambiguous Stata spelling.

        Notes
        -----
        The discriminating question is what the CALLER wrote, not what patsy
        produced: an explicit ``C(...)`` term is a factor expansion, while a bare
        column is a variable Stata would render as one row -- even when the column
        is a string that patsy had to code as a factor (the ``auto.dta``
        ``foreign`` column is ``"Domestic"``/``"Foreign"``, yet
        ``regress_auto.json`` reports a single ``foreign`` row).  So the check is
        purely syntactic on the recorded parameter names.
        """

        offending: list[str] = []
        # Group the factor columns by their base variable and note whether the
        # caller wrote an explicit ``C(...)``.
        levels: dict[str, int] = {}
        explicit: set[str] = set()
        for name in self._param_names:
            text = str(name)
            if "[T." not in text:
                continue
            base = text.split("[T.", 1)[0].replace("C(", "").rstrip(")")
            levels[base] = levels.get(base, 0) + 1
            if text.startswith("C("):
                explicit.add(base)

        for base, count in levels.items():
            # Two independent reasons a term is ambiguous:
            #
            # * it expands into MORE THAN ONE level column.  `y ~ x1 + g` on a
            #   string column with five levels yields `g[T.b]`..`g[T.e]` with no
            #   `C(...)` in sight; treating those as bare dummies made all four
            #   collapse to the name `g`, silently giving four coefficients one
            #   label.
            #
            # * the caller wrote an explicit `C(...)`.  Even a BINARY factor has
            #   this problem: `i.foreign` reports FOUR e(b) rows -- `0b.foreign`,
            #   `1.foreign` and `_cons` plus the other regressor -- with the base
            #   at 0, while this result has one level column.  The single level
            #   count would otherwise pass the test above.
            if count > 1 or base in explicit:
                offending.append(base)
        return sorted(offending)

    @property
    def stata_param_names(self) -> list[str]:
        """Coefficient names as Stata reports them (``_cons`` last).

        Raises
        ------
        NotImplementedError
            When a factor term's Stata spelling is ambiguous -- Stata reports one
            row per level plus a zero base row, while this package has a single
            level column. See :attr:`stata_factor_terms`. Returning an invented
            name would be worse than refusing: it would look like a valid
            correspondence to Stata's ``e(b)`` row when none exists.
        """

        factors = self.stata_factor_terms
        if factors:
            raise NotImplementedError(
                "stata_param_names cannot name these factor terms unambiguously: "
                + ", ".join(factors)
                + ".  Stata reports one e(b) row per factor LEVEL plus an "
                "explicit zero base row (0b.<var>), while this result has a "
                "single level column; there is no one-to-one mapping.  Compare "
                "with `params`/`bse`, or encode the factor as dummies and refit."
            )

        order = self._stata_order()
        names = [str(self._param_names[index]) for index in order]
        renamed = []
        for name in names:
            if name.strip().lower() in self._STATA_CONSTANT_NAMES:
                renamed.append("_cons")
            elif "[T." in name:
                # A bare factor column: Stata names the row after the VARIABLE,
                # not the level -- ``regress_auto.json`` reports ``foreign`` for
                # what patsy spells ``foreign[T.Foreign]``.
                renamed.append(name.split("[T.", 1)[0])
            elif ":" in name:
                # Stata separates interaction terms with ``#``, patsy with ``:``:
                # ``x1 * x2`` yields ``x1:x2`` here and ``x1#x2`` in Stata.  The
                # ORDER is already right, only the separator differs, so swapping
                # it is a rename rather than a guess.
                renamed.append(name.replace(":", "#"))
            elif name.startswith("/"):
                # Auxiliary parameters are DISPLAYED with a leading slash by
                # Stata (``/cut1``, ``/lnalpha``) but stored WITHOUT it in e(b).
                # The fixtures are decisive: ``ologit.json``'s names are
                # ``['x1', 'x2', 'cut1', 'cut2']`` and ``nbreg.json``'s are
                # ``['x1', '_cons', 'lnalpha']`` -- no slashes anywhere.
                renamed.append(name[1:])
            else:
                renamed.append(name)
        return renamed

    @property
    def stata_params(self) -> pd.Series:
        """``b`` in Stata's ``e(b)`` order and spelling.

        Raises
        ------
        NotImplementedError
            For factor terms whose Stata spelling is ambiguous; see
            :attr:`stata_param_names`.
        """

        order = self._stata_order()
        values = np.asarray(self._params.to_numpy(), dtype=float)[order]
        return pd.Series(values, index=self.stata_param_names, name="params")

    @property
    def stata_cov_params(self) -> np.ndarray:
        """``V`` in Stata's ``e(V)`` order (the same permutation as the names).

        Raises
        ------
        NotImplementedError
            For factor terms whose Stata layout has more rows than this result;
            see :attr:`stata_param_names`.
        """

        # Surface the same refusal as the names, so a caller cannot get a matrix
        # whose rows silently do not match Stata's.
        _ = self.stata_param_names
        order = self._stata_order()
        covariance = np.asarray(self.cov_params(), dtype=float)
        return covariance[np.ix_(order, order)]

    @property
    def params(self) -> pd.Series:
        return self._params

    @property
    def bse(self) -> pd.Series:
        return self._bse

    @property
    def tvalues(self) -> pd.Series:
        return self._tvalues

    @property
    def zvalues(self) -> pd.Series:
        """Normal statistic for :math:`H_0: \\beta_j = 0`.

        Stata reports this as ``z`` for models fitted by maximum likelihood.
        It is numerically identical to :attr:`tvalues` (``params / bse``); the
        alias exists so downstream code can branch on the inference
        distribution without computing it again.
        """

        return self._tvalues

    @property
    def pvalues(self) -> pd.Series:
        return self._pvalues

    def cov_params(self) -> np.ndarray:
        """Return the covariance matrix of the parameter estimates."""

        return self._cov

    def estat_vce(self, *, correlation: bool = False) -> Any:
        """Stata's ``estat vce`` -- print the coefficient covariance matrix.

        ``correlation=True`` gives ``estat vce, correlation``.  Both are pure
        renderings of :meth:`cov_params`; nothing is recomputed.

        Returns
        -------
        VceMatrix
            ``str(...)`` prints the table; ``.to_frame()`` returns the full
            symmetric matrix.
        """

        from westata.postestimation.vce import estat_vce

        return estat_vce(self, correlation=correlation)

    def estat_gof(self, **kwargs: Any) -> Any:
        """Stata's ``estat gof`` -- Poisson goodness-of-fit tests.

        Only defined for ``poisson``: Stata has no ``estat gof`` after ``nbreg``
        (``nbreg`` sets no ``e(estat_cmd)``), so requesting it there raises
        ``NotImplementedError`` rather than returning an invented statistic.
        """

        from westata.postestimation.gof import estat_gof

        return estat_gof(self, **kwargs)

    def estat_vif(self, **kwargs: Any) -> Any:
        """Stata's ``estat vif`` -- variance inflation factors.

        Only defined for linear models: the auxiliary regressions are ordinary
        least squares by construction, and Stata reaches the command through
        ``regress_estat``.
        """

        from westata.postestimation.vif import estat_vif

        return estat_vif(self, **kwargs)

    def estat_hettest(self, **kwargs: Any) -> Any:
        """Stata's ``estat hettest`` -- Breusch-Pagan / Cook-Weisberg.

        With no ``variables`` the auxiliary regression is on the FITTED VALUES,
        which is Stata's no-varlist behaviour; ``rhs=True`` uses every regressor.
        Refuses, like Stata, after a robust or clustered VCE, on pweighted data
        and after a ``noconstant`` fit.
        """

        from westata.postestimation.hettest import estat_hettest

        return estat_hettest(self, **kwargs)

    def suest(self, *others: Any, **kwargs: Any) -> Any:
        """Stata's ``suest`` -- stack this fit with others into one covariance.

        Combines the parameter vectors and forms the simultaneous sandwich covariance
        across models (``suest.sthlp:84-88``).  The models must share an estimation
        sample.
        """

        from westata.postestimation.suest import suest

        return suest((self,) + tuple(others), **kwargs)

    def linktest(self) -> Any:
        """Stata's ``linktest`` -- Pregibon's link test for a specification error.

        Refits the same estimator with ``_hat`` and ``_hatsq`` added and reports the t
        statistic of the squared term (``linktest.ado:131-132``).
        """

        from westata.postestimation.linktest import linktest

        return linktest(self)

    def estat_moran(self, *args: Any, **kwargs: Any) -> Any:
        """Stata's ``estat moran`` -- Moran's test for spatial correlation in residuals.

        Requires a spatial weighting matrix.  This package has no ``spmatrix`` toolchain,
        so the matrix is passed explicitly; see
        :mod:`westata.postestimation.moran` for the caveat about Stata's own statistic.
        """

        from westata.postestimation.moran import estat_moran

        return estat_moran(self, *args, **kwargs)

    def estat_esize(self, **kwargs: Any) -> Any:
        """Stata's ``estat esize`` -- eta/epsilon/omega-squared effect sizes.

        ``regress_postestimation.sthlp:599-614`` restricts it to linear models; eta-squared
        is ``R^2`` and carries a noncentral-F interval.
        """

        from westata.postestimation.esize import estat_esize

        return estat_esize(self, **kwargs)

    def estat_szroeter(self, *args: Any, **kwargs: Any) -> Any:
        """Stata's ``estat szroeter`` -- Szroeter's rank test for heteroskedasticity.

        Refuses, like ``szroeter.ado:13-22``, after anything other than ``regress``, and
        on weighted or clustered fits.
        """

        from westata.postestimation.szroeter import estat_szroeter

        return estat_szroeter(self, *args, **kwargs)

    def estat_ovtest(self, **kwargs: Any) -> Any:
        """Stata's ``estat ovtest`` -- Ramsey RESET test for omitted variables."""

        from westata.postestimation.ovtest import estat_ovtest

        return estat_ovtest(self, **kwargs)

    def estat_ic(self, **kwargs: Any) -> Any:
        """Stata's ``estat ic`` -- AIC, BIC, AICc and CAIC.

        ``df`` defaults to ``rank(V)``, matching ``estimates.ado:977-978``; ``N``
        is ``e(N_ic)`` when recorded and ``e(N)`` otherwise.
        """

        from westata.postestimation.ic import estat_ic

        return estat_ic(self, **kwargs)

    def estat_firststage(self, **kwargs: Any) -> Any:
        """Stata's ``estat firststage`` -- IV first-stage diagnostics.

        Reports each endogenous regressor's first-stage R-squared, adjusted
        R-squared, partial R-squared and the joint F of the excluded
        instruments; with more than one endogenous regressor it adds Shea's
        partial R-squared.
        """

        from westata.postestimation.firststage import estat_firststage

        return estat_firststage(self, **kwargs)

    def first_stage(self) -> str:
        """Stata's ``ivregress ..., first`` -- the first-stage regression table.

        One block per endogenous regressor, headed "First-stage regressions": the F,
        R-squared, adjusted R-squared and Root MSE of that equation, then its
        coefficient table with the included exogenous variables and the excluded
        instruments.

        This is a different output from :meth:`estat_firststage`, which prints the
        DIAGNOSTICS for the same equations. Stata offers both and they are not
        substitutes, so both are here.
        """

        from westata.estimators.first_stage_table import render_first_stage

        return render_first_stage(self)

    def estat_overid(self, **kwargs: Any) -> Any:
        """Stata's ``estat overid`` -- overidentification restrictions test.

        Reports Sargan (with Basmann's finite-sample variant) for 2SLS/LIML and
        Hansen's J for GMM.
        """

        from westata.postestimation.overid import estat_overid

        return estat_overid(self, **kwargs)

    def estat_mundlak(self, **kwargs: Any) -> Any:
        """Stata's ``estat mundlak`` -- the Mundlak panel specification test.

        Adds the within-group means of the regressors to the random-effects equation and
        tests them jointly to zero: ``xtreg_postestimation.sthlp:366-370``.  A rejection
        means the unobserved effect is correlated with ``x``, so the random-effects
        estimator is inconsistent.
        """

        from westata.postestimation.mundlak import estat_mundlak

        return estat_mundlak(self, **kwargs)

    def estat_weakrobust(self, **kwargs: Any) -> Any:
        """Stata's ``estat weakrobust`` -- inference robust to weak instruments.

        Reports the Anderson-Rubin test, whose distribution does not depend on instrument
        strength.  The conditional likelihood-ratio test (the overidentified default in
        Stata) is refused with its reason; see
        :func:`westata.postestimation.weakrobust.estat_weakrobust`.
        """

        from westata.postestimation.weakrobust import estat_weakrobust

        return estat_weakrobust(self, **kwargs)

    def estat_endogenous(self, **kwargs: Any) -> Any:
        """Stata's ``estat endogenous`` -- Durbin-Wu endogeneity test."""

        from westata.postestimation.endogenous import estat_endogenous

        return estat_endogenous(self, **kwargs)

    def estat_classification(self, **kwargs: Any) -> Any:
        """Stata's ``estat classification`` -- the 2x2 classification table."""

        from westata.postestimation.classification import estat_classification

        return estat_classification(self, **kwargs)

    def estat_auc(self) -> Any:
        """Stata's ``estat auc`` -- the area under the ROC curve.

        ``logit_estat.ado:19-21`` dispatches the subcommand to ``lroc, nograph``.
        """

        from westata.postestimation.auc import estat_auc

        return estat_auc(self)

    def estat_imtest(self, **kwargs: Any) -> Any:
        """Stata's ``estat imtest`` -- Cameron & Trivedi's IM-test."""

        from westata.postestimation.imtest import estat_imtest

        return estat_imtest(self, **kwargs)

    def estat_summarize(self, **kwargs: Any) -> Any:
        """Stata's ``estat summarize`` -- describe the estimation sample."""

        from westata.postestimation.summarize import estat_summarize

        return estat_summarize(self, **kwargs)

    @property
    def level(self) -> float:
        """The confidence level as a FRACTION (0.95 unless ``level(#)`` was given)."""

        return 0.95 if self._level is None else float(self._level)

    def conf_int(self, alpha: float | None = None) -> pd.DataFrame:
        """Confidence intervals for the coefficients.

        Uses the Student-t reference distribution for ``dist="t"`` and the
        standard normal for ``dist="z"``.

        ``alpha`` defaults to ``1 - level``: a result estimated with Stata's
        ``level(90)`` reports 90% intervals unless a caller asks for something else
        explicitly. Hard-coding 0.05 here was what made ``level(#)`` unrepresentable --
        the model could not influence the interval even if it recorded a level.
        """

        if alpha is None:
            alpha_value = 1.0 - self.level
        else:
            alpha_value = float(alpha)
        if not 0.0 < alpha_value < 1.0:
            raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")

        if self._dist == "t":
            critical = float(stats.t.ppf(1.0 - alpha_value / 2.0, self._df_resid))
        else:
            critical = float(stats.norm.ppf(1.0 - alpha_value / 2.0))

        params_arr = self._params.to_numpy()
        bse_arr = self._bse.to_numpy()

        # Some estimators build the interval on a TRANSFORMED scale and map it back --
        # ``svy: proportion`` is one: Stata's interval for a proportion is computed in
        # log-odds and transformed, which is why its table carries a second "Logit"
        # header. The hooks below are optional and default to the identity, so every
        # other estimator keeps the plain linear interval.
        transform = getattr(self, "_ci_transform", None)
        inverse = getattr(self, "_ci_inverse", None)
        if transform is not None and inverse is not None:
            centre = transform(params_arr)
            scale = transform.derivative(params_arr) * bse_arr if hasattr(
                transform, "derivative"
            ) else bse_arr
            lower = inverse(centre - critical * scale)
            upper = inverse(centre + critical * scale)
        else:
            lower = params_arr - critical * bse_arr
            upper = params_arr + critical * bse_arr
        return pd.DataFrame(
            {"lower": lower, "upper": upper},
            index=pd.Index(self._param_names, dtype=object),
        )

    # ------------------------------------------------------------------
    # Sample / model metadata
    # ------------------------------------------------------------------
    @property
    def nobs(self) -> int:
        return self._nobs

    @property
    def df_resid(self) -> float:
        return self._df_resid

    @property
    def df_model(self) -> float:
        return self._df_model

    @property
    def resid(self) -> np.ndarray:
        return self._resid

    @property
    def fittedvalues(self) -> np.ndarray:
        return self._fittedvalues

    @property
    def cov(self) -> np.ndarray:
        return self._cov

    @property
    def dist(self) -> str:
        """Inference distribution: ``"t"`` or ``"z"``."""

        return self._dist

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def cov_type(self) -> str:
        return self._cov_type

    # ------------------------------------------------------------------
    # Stata e() scalars
    # ------------------------------------------------------------------
    @property
    def e(self) -> EReturn:
        """Stata ``e()`` scalars (item and attribute access).

        Two macros are synthesised when the estimator did not record them, because
        every estimation command documents both and a consumer reading ``e()``
        cannot tell "not recorded" from "not applicable":

        * ``e(depvar)`` falls back to ``response_name`` (``logit.sthlp:264`` and
          five other help files list "name of dependent variable");
        * ``e(cmd)`` falls back to ``model_name``, which already holds Stata's
          command name (``regress``, ``logit``, ``reghdfe``, ...).

        ``e(cmdline)`` is deliberately NOT synthesised: it records the command as
        the user TYPED it, including options, and this package has no such string.
        Inventing one would be a fabricated value rather than a recovered one.
        """

        if "depvar" not in self._e:
            response = getattr(self, "response_name", None)
            if response:
                self._e["depvar"] = str(response)
        if "cmd" not in self._e:
            model_name = str(getattr(self, "_model_name", "") or "")
            if model_name:
                self._e["cmd"] = model_name
        if "properties" not in self._e:
            # ``regress.sthlp:439`` and ``logit.sthlp:279`` (and the others) all
            # document e(properties) as ``b V`` -- a CONSTANT, not a per-command
            # value -- so it is safe to recover.  e(title) and e(vcetype) are
            # per-command English strings with no derivable source, so they are
            # left absent rather than guessed.
            self._e["properties"] = "b V"
        return self._e

    def _e_scalar(self, key: str) -> float:
        value = self._e.get(key)
        if value is None:
            return float("nan")
        return _as_float(value)

    @property
    def llf(self) -> float:
        """Log-likelihood from ``e(ll)``; NaN when not recorded."""

        return self._e_scalar("ll")

    @property
    def rsquared(self) -> float:
        """R-squared from ``e(r2)``; NaN when not recorded."""

        return self._e_scalar("r2")

    @property
    def rsquared_adj(self) -> float:
        """Adjusted R-squared from ``e(r2_a)``; NaN when not recorded."""

        return self._e_scalar("r2_a")

    @property
    def fvalue(self) -> float:
        """Model F statistic from ``e(F)``; NaN when not recorded."""

        return self._e_scalar("F")

    @property
    def aic(self) -> float:
        """AIC on the statsmodels scale: ``-2*llf + 2*k``.

        ``e(aic)`` stores the Gaussian-deviance form ``N log(rss/N) + 2k``, which omits the
        ``N(1 + log 2pi)`` constant that ``e(llf)`` carries. The two differ by exactly that
        constant, and the gap grows with N -- on icecream.dta it is 85.1 (-194.37519 stored
        against -109.23887 from both ``sm.OLS(...).fit().aic`` and ``estat ic``).

        ``result.aic`` was returning the stored number, so a caller consuming it with
        statsmodels semantics got a value 85 too small while ``e["aic"]`` and ``estat_ic``
        stayed self-consistent. The property now follows statsmodels (and Stata's
        ``estat ic``, which agree), and ``e["aic"]`` keeps the package's own convention for
        anyone who reads the raw macro.
        """

        stored = self._e_scalar("aic")
        # Only recompute when the stored number is the package's OWN Gaussian-deviance
        # value. An estimator that stores a genuine ``e(aic)`` (``_aic_is_stata``) -- or a
        # hand-built ``StataResults`` whose caller set it explicitly -- must keep it, since
        # a stored macro always wins over a derived default.
        if getattr(self, "e", {}).get("_aic_is_stata", None) is not False:
            return stored
        llf = self._e_scalar("ll")
        if np.isfinite(llf):
            parameters = float(self._e_scalar("df_m")) + 1.0
            if np.isfinite(parameters):
                return float(-2.0 * llf + 2.0 * parameters)
        return stored

    @property
    def bic(self) -> float:
        """BIC on the statsmodels scale: ``-2*llf + k*log(nobs)``.

        Same reasoning as :attr:`aic`: ``e(bic)`` uses the Gaussian-deviance form.
        """

        stored = self._e_scalar("bic")
        if getattr(self, "e", {}).get("_aic_is_stata", None) is not False:
            return stored
        llf = self._e_scalar("ll")
        nobs = float(self.nobs) if np.isfinite(float(self.nobs)) else float("nan")
        if np.isfinite(llf) and np.isfinite(nobs):
            parameters = float(self._e_scalar("df_m")) + 1.0
            if np.isfinite(parameters):
                return float(-2.0 * llf + parameters * np.log(nobs))
        return stored

    @property
    def converged(self) -> bool:
        """Whether the optimizer reported convergence.

        ``design.md:209`` lists ``converged``/``iterations``/``grad_norm`` as part of the
        result surface, and Stata stores the first as ``e(converged)`` (``1`` if it
        converged, ``0`` otherwise -- ``logit.sthlp:259``).  Absent means the estimator
        does not run an iterative optimizer at all, which is reported as ``True`` rather
        than ``False``: a closed-form fit has nothing to fail to converge.
        """

        value = self.e.get("converged")
        if value is None:
            return True
        try:
            return bool(float(value) != 0.0)
        except (TypeError, ValueError):
            return bool(value)

    @property
    def iterations(self) -> float:
        """Number of optimizer iterations, from ``e(ic)`` (``logit.sthlp:257``).

        NaN when the estimator is not iterative or the backend did not report a count.
        """

        return self._e_scalar("ic")

    @property
    def grad_norm(self) -> float:
        """Norm of the gradient at the reported estimate; NaN when not recorded.

        A small value is the numeric counterpart of ``converged``: the optimizer stops
        when this is near zero.  It is exposed as NaN rather than ``0.0`` where the
        backend does not report it, so a caller cannot mistake "unavailable" for "exactly
        converged".
        """

        return self._e_scalar("gradient_norm")

    # ------------------------------------------------------------------
    # Postestimation
    # ------------------------------------------------------------------
    def test(self, *specs: Any, **kwargs: Any) -> Any:
        """Wald test of linear hypotheses (Stata ``test``).

        See :meth:`westata.postestimation.tests.run_wald_test`; this thin
        forwarder exists so the results object owns its own postestimation
        surface, like a Stata estimation result.
        """

        from westata.postestimation.api import test_as_stata

        return test_as_stata(self, *specs, **kwargs)

    def lincom(self, expression: str, **kwargs: Any) -> Any:
        """Linear combination of parameters (Stata ``lincom``)."""

        from westata.postestimation.api import lincom_as_stata

        return lincom_as_stata(self, expression, **kwargs)

    def lrtest(self, restricted: Any) -> Any:
        """Likelihood-ratio test against a nested model (Stata `lrtest`)."""

        from westata.postestimation.modeltests import lrtest

        return lrtest(self, restricted)

    def hausman(self, efficient: Any, **kwargs: Any) -> Any:
        """Hausman specification test (Stata `hausman`).

        Keyword arguments are forwarded to
        :func:`westata.postestimation.modeltests.hausman`, so the Stata
        ``hausman ..., constant`` option is reachable from a result object too:
        ``fe.hausman(re, constant=True)``.
        """

        from westata.postestimation.modeltests import hausman

        return hausman(self, efficient, **kwargs)

    def contrast(self, term: str, method: str = "r", **kwargs: Any) -> Any:
        """ANOVA-style contrast of a factor's levels (Stata `contrast`)."""

        from westata.postestimation.contrast import contrast

        return contrast(self, term, method, **kwargs)

    def pwcompare(self, term: str, **kwargs: Any) -> Any:
        """All pairwise level differences (Stata `pwcompare`)."""

        from westata.postestimation.contrast import pwcompare

        return pwcompare(self, term, **kwargs)

    def nlcom(self, expression: str, **kwargs: Any) -> Any:
        """Nonlinear combination of parameters (Stata `nlcom`)."""

        from westata.postestimation.nonlinear import nlcom

        return nlcom(self, expression, **kwargs)

    def testnl(self, expression: str) -> Any:
        """Nonlinear Wald test (Stata `testnl`)."""

        from westata.postestimation.nonlinear import testnl

        return testnl(self, expression)

    def margins(self, **kwargs: Any) -> Any:
        """Marginal effects or predictive margins (Stata ``margins``).

        The design matrix is taken from :attr:`design_matrix` when the
        estimator stored one; otherwise pass ``exog=`` and ``param_names=``.
        """

        from westata.postestimation.margins import margins

        return margins(self, **kwargs)

    def predict(self, exog: Any = None, kind: str | None = None, **kwargs: Any) -> Any:
        """Typed prediction (Stata ``predict, <type>``).

        Parameters
        ----------
        exog:
            Design matrix; defaults to the stored estimation design.
        kind:
            ``"xb"``, ``"pr"``, ``"n"``, ``"e"``, ``"ystar"``, ``"residuals"``,
            depending on the model.  ``None`` (the default) means "whatever Stata
            assumes", resolved by
            :func:`westata.postestimation.predict.default_predict_kind`: ``xb`` for
            linear models, ``pr`` for logit/probit/ordinal, ``n`` for count models.

            It used to default to ``"xb"`` for every estimator, so a bare
            ``logit_result.predict()`` returned log-odds -- neither Stata's meaning
            (Stata assumes ``pr`` and says so) nor statsmodels' (``sm.Logit.predict``
            returns probabilities). Pass ``kind="xb"`` for the old behaviour.
        """

        from westata.postestimation.predict import default_predict_kind, predict

        if kind is None:
            kind = default_predict_kind(str(getattr(self, "model_name", "")))

        if exog is None:
            stored = getattr(self, "design_matrix", None)
            if stored is None:
                raise ConsistencyError(
                    self._predict_unavailable_message()
                )
            exog = stored[0]
            # Under ``fweight`` the design matrix holds the COLLAPSED rows while ``resid``
            # and ``fittedvalues`` hold the EXPANDED sample -- 24 against 2201 on the
            # titanic logit. That mismatch made ``estat classification`` refuse with
            # "the residual and prediction vectors do not align", and it would silently
            # mis-align any caller that pairs ``predict()`` with the residuals (C11-07).
            #
            # Expanding here restores the invariant that every per-observation vector on
            # the result has the same length.
            _freq = getattr(self, "freq_weights", None)
            if _freq is not None:
                _freq = np.asarray(_freq, dtype=float).reshape(-1)
                _matrix = np.asarray(exog)
                if _freq.size == _matrix.shape[0] and np.all(np.isfinite(_freq)):
                    _repeats = np.rint(_freq).astype(int)
                    if np.all(_repeats >= 0) and int(_repeats.sum()) > 0:
                        exog = np.repeat(_matrix, _repeats, axis=0)
        else:
            exog = self._as_design_for_predict(exog)
        return predict(self, exog, kind, **kwargs)

    def _as_design_for_predict(self, exog: Any) -> Any:
        """Turn a caller-supplied ``exog`` into a design matrix, or explain why not.

        A pandas DataFrame is accepted and its columns are SELECTED BY NAME from the stored
        design. Without that, ``predict(exog=df)`` on a frame carrying extra columns -- the
        usual case, since the frame you fitted on has the response and any grouping
        columns in it -- failed deep inside ``np.matmul``:

            ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0

        which says nothing about what was expected and sends the reader looking at their
        data rather than at the shape.

        Any other input is validated against the coefficient count, which is the check that
        turns the same confusing failure into a message naming both numbers.
        """

        import numpy as _np
        import pandas as _pd

        stored = getattr(self, "design_matrix", None)

        if isinstance(exog, _pd.DataFrame):
            if stored is None:
                raise ConsistencyError(
                    "predict(exog=DataFrame) needs the estimation design to know which "
                    "columns the model used, and this result did not retain it. Pass the "
                    "design as an array instead."
                )
            _design, design_names = stored
            wanted = [
                name
                for name in design_names
                if name in exog.columns
            ]
            missing = [
                name
                for name in design_names
                if name not in exog.columns
                and str(name).lower() not in ("intercept", "const", "_cons", "intercept_")
            ]
            if missing:
                raise ValueError(
                    f"predict(exog=DataFrame): the frame is missing the model's "
                    f"regressor column(s) {missing}; it has {list(exog.columns)}"
                )
            built = _np.ones((len(exog), len(design_names)), dtype=float)
            for position, name in enumerate(design_names):
                if name in exog.columns:
                    built[:, position] = _np.asarray(exog[name], dtype=float)
                # An intercept column stays 1.0 by construction.
            return built

        # The array path is deliberately left as it was. Validating its width against the
        # coefficient count looked like an improvement and was not: ordered models and the
        # limited-response estimators carry extra PARAMETERS (cut points, sigma) that are
        # not design columns, so ``exog`` is legitimately narrower than ``params``. A
        # width check there rejected valid internal calls -- three tests caught it -- and
        # the "clearer message" would have been attached to a wrong rule.
        return _np.asarray(exog, dtype=float)

    def _predict_unavailable_message(self) -> str:
        """Explain WHY this result cannot predict, per estimator.

        A single generic message ("the design matrix was not retained") is
        accurate but unhelpful, and it hid a real gap: an audit of the full
        predict matrix showed EVERY type failing for IV2SLS while every per-type
        test passed, because each one was "correctly" refusing.  Naming the
        estimator and the reason makes the gap visible.

        For the absorbed/panel estimators the reason is semantic, not missing
        data: ``predict, xb`` after ``reghdfe``/``xtreg, fe`` must decide whether
        the absorbed effects are added back, and guessing either way would be
        silently wrong for half the callers.
        """

        name = str(getattr(self, "model_name", "") or "").strip().lower()
        if "reghdfe" in name:
            return (
                "predict after reghdfe is not implemented: the fitted value "
                "depends on whether the absorbed effects are included, which "
                "Stata exposes as separate options.  Returning one of the two "
                "without saying which would be silently wrong, so no type is "
                "offered."
            )
        if name.startswith("xtreg"):
            return (
                "predict after xtreg: the design matrix was not retained on this "
                "result, so the linear prediction cannot be formed.  Stata's "
                "default is xb = a + xb (WITHOUT the panel effect, "
                "xtreg_postestimation.sthlp:128); the variant that adds it is "
                "xbu."
            )
        return (
            "predict: the design matrix was not retained on this result; "
            "pass exog=... explicitly"
        )

    # ------------------------------------------------------------------
    # statsmodels ecosystem surface (v2 report section 6)
    # ------------------------------------------------------------------
    @property
    def model(self) -> Any:
        """The fitted model, as statsmodels exposes it.

        ``statsmodels.stats.anova.anova_lm`` reads ``res.model`` and re-fits from it, so its
        absence made ``anova_lm(westata_result)`` raise AttributeError. The object returned
        here carries ``endog``, ``exog``, ``exog_names`` and a ``fit()`` that returns this
        same result -- enough for anova_lm's contract without pretending to BE a
        statsmodels model.
        """

        from westata.results.ecosystem import ResultModel

        return ResultModel(self)

    def t_test(self, r_matrix: Any, cov_p: Any = None, use_t: bool | None = None) -> Any:
        """statsmodels' ``t_test``: a single restriction reported as a t statistic."""

        from westata.results.ecosystem import build_contrast

        return build_contrast(self, r_matrix, use_t=True if use_t is None else use_t)

    def f_test(
        self, r_matrix: Any, cov_p: Any = None, invcov: Any = None
    ) -> Any:
        """statsmodels' ``f_test``: one or more restrictions reported jointly as F."""

        from westata.results.ecosystem import build_contrast

        return build_contrast(self, r_matrix, use_t=False)

    def get_influence(self) -> Any:
        """Influence measures, named as statsmodels' ``OLSInfluence`` names them."""

        from westata.results.ecosystem import influence_measures

        return influence_measures(self)

    def wald_test(self, R: Any, q: Any = None, scalar: bool = True) -> Any:
        """statsmodels-compatible joint Wald test.

        Mirrors ``statsmodels``' ``wald_test(R, q, scalar)``: returns an object
        with ``statistic``, ``df_denom``, ``df_num`` and ``pvalue``.  When
        ``scalar`` is False a chi2 test is returned instead of the F test, which
        matches statsmodels' documented behaviour.
        """

        from westata.postestimation.api import statsmodels_wald_test

        return statsmodels_wald_test(self, R, q, scalar)

    # ------------------------------------------------------------------
    # Presentation
    # ------------------------------------------------------------------
    @property
    def odds_ratios(self) -> pd.Series:
        """``exp(params)`` -- the odds ratios ``logit, or`` prints.

        Only meaningful for a model whose link is log-odds; other families raise rather than
        returning a number that has no interpretation. ``conf_int()`` can be exponentiated
        alongside it to get Stata's odds-ratio interval.
        """

        from westata.errors import UnsupportedError

        name = str(getattr(self, "model_name", "") or "").lower()
        if name not in {"logit", "ologit", "mlogit", "clogit", "xtlogit", "exlogistic"}:
            raise UnsupportedError(
                f"odds_ratios exists only for log-odds models; {name!r} is not one"
            )
        return pd.Series(
            np.exp(np.asarray(self.params, dtype=float)),
            index=self.params.index,
            name="odds_ratio",
        )

    def summary(self, style: str = "statsmodels", *, odds_ratio: bool = False) -> str:
        """Render a text summary.

        Parameters
        ----------
        style:
            ``"statsmodels"`` (default) or ``"stata"``.
        odds_ratio:
            Exponentiate the coefficients, their standard errors and both interval
            endpoints, and relabel the column "Odds ratio" -- Stata's ``logit ..., or``.
            z and P>|z| are unaffected, since they come from the untransformed scale.
        """

        if odds_ratio:
            # A shallow copy carries the flag without mutating the fit, so two summaries of
            # the same result can differ. ``copy.copy`` keeps the shared arrays shared.
            import copy as _copy

            clone = _copy.copy(self)
            clone._odds_ratio = True
            return render_summary(clone, style)
        return render_summary(self, style)

    def __repr__(self) -> str:
        name = self._model_name or "westata"
        return (
            f"<StataResults {name}: nobs={self._nobs}, "
            f"k={len(self._param_names)}, dist={self._dist}, "
            f"cov_type={self._cov_type}>"
        )
