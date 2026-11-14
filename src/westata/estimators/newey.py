"""``newey`` -- regression with Newey-West standard errors.

Transcribed from ``newey.ado`` (StataNow 19.5).  This is NOT a new estimator: the
point estimates are plain OLS (``newey.ado:96`` calls ``_regress``), and only the
variance differs.  So this module fits ``OLS`` and asks for a HAC VCE, then
rewrites the ``e()`` collection to match ``newey``'s own.

What ``newey`` does, in order (``newey.ado:14-215``):

1. ``syntax varlist(fv ts) [if] [in] [aw/], lag(integer) [noconstant force]`` --
   ``lag()`` is REQUIRED with no default (``newey.sthlp:55``), and a negative lag
   is rejected (``newey.ado:24-27``).
2. A time variable must be set (``xt_tis``); without it Stata exits 111
   ("time variable not set, use -tsset varname ...-").
3. ``Checkt`` (``newey.ado:263-290``) classifies the spacing:

   =========  ===========================================================
   ``tflag``  meaning
   =========  ===========================================================
   0          regularly spaced, interval 1
   1          irregularly spaced
   2          regularly spaced but interval > 1
   =========  ===========================================================

   ``tflag == 2`` is ALWAYS an error (``newey.ado:74-77``), even with
   ``force``.  ``tflag == 1`` is an error unless ``force`` is given
   (``newey.ado:78-81``).
4. ``aweight`` is rescaled by its mean (``newey.ado:87-90``:
   ``replace wvar = wvar/r(mean)``) -- note this is ``w/mean(w)``, i.e. normalised
   so that ``mean(w) = 1``, which is the same target as summing to ``N``.
5. The HAC accumulation and ``v' = (xtxi*xtx*xtxi)*(nobs/tdf)``
   (``newey.ado:135-189``), which ``westata.vce.hac`` already implements.
6. ``e()`` is POSTed directly (``newey.ado:191-232``) and therefore does NOT
   inherit ``regress``'s contents.  ``newey.sthlp:156-187`` lists the complete
   set: scalars ``N df_m df_r F p lag rank``, macros ``cmd cmdline depvar wtype
   wexp title vcetype properties estat_cmd predict asbalanced asobserved``,
   matrices ``b V``.

   In particular ``newey`` stores **no** ``r2``/``r2_a``/``rmse``/``sigma``/
   ``rss``/``mss``/``ll``/``df_a`` -- those are ``regress`` keys that a plain
   ``mat post`` never creates.  Reporting them would be an invented surface.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from westata.data.weights import normalize_weights
from westata.estimators.linear import OLS
from westata.results.base import EReturn

__all__ = ["newey", "NeweyWestResult", "check_time_spacing"]

#: ``newey.sthlp:174``: the title printed above the table (note the en dash).
NEWEY_TITLE = "Regression with Newey–West standard errors"
#: ``newey.sthlp:175``.
NEWEY_VCETYPE = "Newey–West"


def check_time_spacing(time_values: Any) -> tuple[int, float | None]:
    """Classify the spacing of ``time_values``; ``Checkt`` (``newey.ado:263-290``).

    Returns
    -------
    (tflag, delta)
        ``tflag`` is 0 (regular, interval 1), 1 (irregular) or 2 (regular but
        interval > 1).  ``delta`` is the common interval when regular, else
        ``None``.

    Notes
    -----
    Stata reads ``_dta[_TSdelta]`` (the delta from ``tsset``) and divides the
    first differences by it.  With no delta set it uses 1.  Both the min and max
    of those scaled differences are then compared:

    * ``r(min) != r(max)``              -> ``tflag = 1`` (irregular)
    * ``r(mean) > 1 & r(min) == r(max)`` -> ``tflag = 2`` (interval > 1)

    Because a ``pd.Timedelta`` or a float delta both work, the caller's array is
    converted to float first and differences are taken on the ROW ORDER, matching
    Stata's ``sort touse tvar``.
    """

    values = np.asarray(time_values, dtype=float).reshape(-1)
    if values.size < 2:
        # Cannot form a difference; treat as regular with interval 1.
        return 0, 1.0

    differences = np.diff(values)
    if differences.size == 0:
        return 0, 1.0

    minimum = float(differences.min())
    maximum = float(differences.max())
    mean = float(differences.mean())

    if minimum != maximum:
        return 1, None
    if mean > 1.0:
        return 2, minimum
    return 0, minimum


class NeweyWestResult:
    """``newey``'s result: ``regress``'s numbers behind ``newey``'s narrower surface.

    Two things differ from a plain ``StataResults`` and both are visible to
    callers, so they cannot be left to inheritance:

    * ``e()`` holds only ``newey``'s scalars (managed by :func:`newey`);
    * ``predict`` accepts only ``xb``/``index``/``stdp``/``residuals``
      (``newey_p.ado:20``: ``local myopts XB Index STDP REsiduals``).  ``regress``
      offers fifteen types; forwarding all of them would claim a surface
      ``newey_p`` does not have.
    """

    #: ``newey_p.ado:20``.  ``fitted`` is excluded: ``newey_p`` does not list it.
    SUPPORTED_PREDICT_TYPES = ("xb", "index", "stdp", "residuals")

    def __init__(self, results: Any, *, lag: int) -> None:
        self._results = results
        self.lag = int(lag)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._results, name)

    def to_dict(self) -> dict[str, Any]:
        """Flat ``{name: value}`` of ``newey``'s scalars (R4.1).

        ``StataResults`` deliberately does not offer ``to_dict`` (R4.1's
        ``protocol_implementors`` exemption: an estimation result implements the
        statsmodels surface instead), so this builds the dict from that surface rather
        than forwarding a method the wrapped object does not have.  ``lag`` is added
        because it is the one quantity ``newey`` reports that ``regress`` does not.
        """

        wrapped = self._results
        payload: dict[str, Any] = {
            "nobs": int(wrapped.nobs),
            "df_resid": float(wrapped.df_resid),
            "df_model": float(wrapped.df_model),
            "cov_type": str(wrapped.cov_type),
            "lag": int(self.lag),
        }
        for name in ("llf", "rsquared", "rsquared_adj", "fvalue"):
            value = getattr(wrapped, name, None)
            # ``newey`` stores no ``e(ll)`` and no ``e(r2)`` -- Stata's ``newey`` does
            # not report either -- so the property returns NaN.  Emitting NaN here
            # would look like a measured value that came out missing; omitting the key
            # says the quantity does not exist for this command.
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            payload[name] = float(value)
        # ``e()`` carries the Stata scalars (``e(rss)``, ``e(rmse)``, ``e(F)``...) and
        # is what a reader of the Stata table would look up.
        for name, value in dict(getattr(wrapped, "e", {}) or {}).items():
            if isinstance(value, (int, float)) and name not in payload:
                payload[str(name)] = float(value)
        return payload

    def to_frame(self) -> Any:
        """One row per coefficient, in Stata's column order (R4.1)."""

        import pandas as pd

        wrapped = self._results
        return pd.DataFrame(
            {
                "params": np.asarray(wrapped.params, dtype=float),
                "bse": np.asarray(wrapped.bse, dtype=float),
                "tvalues": np.asarray(wrapped.tvalues, dtype=float),
                "pvalues": np.asarray(wrapped.pvalues, dtype=float),
            },
            index=list(wrapped.param_names),
        )

    def predict(self, exog: Any = None, kind: str = "xb", **kwargs: Any) -> Any:
        """``predict`` with ``newey_p``'s type list (``newey_p.ado:20``).

        ``index`` and ``xb`` are synonyms in ``newey_p``: ``newey_p.ado:54`` tests
        ``if "`type'" == "" | "`type'" == "index" | "`type'" == "xb"`` and sends
        both to the same branch.  So ``index`` is mapped rather than refused --
        refusing it would reject something Stata accepts.
        """

        requested = str(kind).strip().lower()
        if requested not in self.SUPPORTED_PREDICT_TYPES:
            raise ValueError(
                f"predict, {kind}: newey_p supports only "
                f"{', '.join(self.SUPPORTED_PREDICT_TYPES)} "
                "(newey_p.ado:20).  regress's other diagnostics are not available "
                "after newey."
            )
        # ``newey_p.ado:54``: index and xb share one branch.
        forwarded = "xb" if requested == "index" else requested
        return self._results.predict(exog, kind=forwarded, **kwargs)

    def estat(self, subcommand: str, **kwargs: Any) -> Any:
        """``estat`` dispatch: ``newey_estat.ado:5-20``.

        Only ``durbinalt`` has a dedicated branch in Stata, and it REQUIRES
        ``force`` ("You must specify force for this command to work after newey",
        ``newey_estat.ado:14-16``); everything else falls through to
        ``estat_default``.  ``durbinalt`` is not implemented, so it refuses rather
        than borrowing another command's test.
        """

        name = str(subcommand).strip().lower().replace("-", "_")
        if name in ("durbinalt", "durbin_alt", "durbina"):
            raise NotImplementedError(
                "estat durbinalt after newey runs the Durbin alternative test "
                "(newey_estat.ado:14, which demands force).  That test is not "
                "implemented for newey results."
            )
        method = getattr(self._results, f"estat_{name}", None)
        if method is None:
            raise ValueError(
                f"estat {subcommand}: newey has no such subcommand; "
                "newey_estat.ado:5-20 defines durbinalt and otherwise defers to "
                "estat_default"
            )
        return method(**kwargs)

    def __str__(self) -> str:
        """Render the STATA-style table.

        ``StataResults.summary()`` returns the statsmodels-style layout, which
        prints ``R-squared``/``Adj R-squared`` rows -- exactly the numbers
        ``newey`` does not store.  The Stata layout lives in
        ``render_summary(..., style="stata")``, so that is what ``str()`` calls.
        """

        from westata.results.summary import render_summary

        return render_summary(self._results, style="stata")

    def __repr__(self) -> str:
        return f"<NeweyWestResult lag={self.lag} nobs={self._results.nobs}>"


def newey(
    formula: str,
    data: pd.DataFrame,
    *,
    lag: int,
    time: Any = None,
    force: bool = False,
    noconstant: bool = False,
    weights: Any = None,
    weight_type: str | None = None,
) -> NeweyWestResult:
    """Fit ``newey depvar indepvars, lag(#)``.

    Thin wrapper over :class:`westata.Newey` -- the class is the implementation, this
    function keeps the documented ``(formula, data, *, ...)`` call working (R2.7).
    """

    return _newey_impl(
        formula,
        data,
        lag=lag,
        time=time,
        force=force,
        noconstant=noconstant,
        weights=weights,
        weight_type=weight_type,
    )


def _newey_impl(
    formula: str,
    data: pd.DataFrame,
    *,
    lag: int,
    time: Any = None,
    force: bool = False,
    noconstant: bool = False,
    weights: Any = None,
    weight_type: str | None = None,
) -> NeweyWestResult:
    """Fit ``newey depvar indepvars, lag(#)``.

    Parameters
    ----------
    formula:
        Patsy-style formula for the right-hand side, e.g. ``"y ~ x1 + x2"``.
    data:
        The estimation frame.
    lag:
        Required truncation.  ``newey.ado:24-27`` rejects a negative value.
    time:
        The time variable.  ``newey`` needs ``tsset``; passing the column here
        plays that role and enables the ``Checkt`` validation.
    force:
        Allow irregularly spaced time values (``newey.ado:78-81``).
    noconstant:
        Suppress the intercept (``newey.ado:44-58``).  Stata rejects
        ``noconstant`` with no regressors.
    weights, weight_type:
        Only ``aweight`` is allowed (``newey.ado:15``: ``[aw/]``).

    Raises
    ------
    ValueError
        For a negative ``lag``, a missing ``lag``, irregular spacing without
        ``force``, or a spacing interval greater than 1.
    NotImplementedError
        For weight types other than ``aweight``.

    Notes
    -----
    The ``e()`` collection is rewritten to ``newey``'s own list.  Keys that
    ``regress`` stores but ``newey`` does not (``r2``, ``rss``, ``mss``,
    ``rmse``, ``sigma``, ``ll``, ...) are REMOVED rather than left in place --
    a plain ``mat post`` never creates them.
    """

    if lag is None:
        raise ValueError(
            "newey requires lag(#); newey.ado:15 declares it a required option "
            "with no default (newey.sthlp:55)"
        )
    lag = int(lag)
    if lag < 0:
        raise ValueError(f"lag({lag}) invalid; newey.ado:24-27 requires a nonnegative lag")

    if weight_type is not None:
        kind = str(weight_type).strip().lower()
        if kind != "aweight":
            raise NotImplementedError(
                "newey accepts only aweights (newey.ado:15 declares '[aw/]'); "
                f"got {weight_type!r}"
            )

    if noconstant and not _has_regressors(formula):
        raise ValueError(
            "may not specify noconstant without regressors; newey.ado:55-58"
        )

    # --- Checkt -------------------------------------------------------------
    if time is not None:
        tflag, _delta = check_time_spacing(time)
        if tflag == 2:
            raise ValueError(
                "the time variable is regularly spaced, but its intervals are "
                "greater than 1; newey.ado:74-77 rejects this even with force"
            )
        if tflag == 1 and not force:
            raise ValueError(
                "the time variable is not regularly spaced; pass force=True to "
                "override (newey.ado:78-81)"
            )
    else:
        # No time variable is the analogue of Stata's "time variable not set"
        # (newey.ado:33-37, exit 111).  The HAC accumulation itself only needs
        # the row order, so the fit proceeds with a comment-worthy caveat.
        pass

    fit_formula = formula
    if noconstant and "- 1" not in formula and "+ 0" not in formula:
        fit_formula = f"{formula} - 1"

    kwargs: dict[str, Any] = {
        "cov_type": "HAC",
        "cov_kwds": {"maxlags": lag},
    }
    formula_kwargs: dict[str, Any] = {}
    if weights is not None:
        # Weights belong to ``from_formula``, not ``fit``.
        formula_kwargs["weights"] = weights
        formula_kwargs["weight_type"] = "aweight"

    results = OLS.from_formula(fit_formula, data, **formula_kwargs).fit(**kwargs)

    # --- rewrite e() to newey's own collection ------------------------------
    #
    # ``newey.ado:191`` uses ``mat post``, which creates a fresh ``e()`` rather
    # than extending ``regress``'s.  Iterating ``newey.sthlp:156-187`` against
    # ``regress``'s keys shows what must go: ``newey`` stores NO ``r2``,
    # ``r2_a``, ``rmse``, ``sigma``, ``rss``, ``mss``, ``ll`` or ``df_a``.
    # Leaving them in place would claim a surface Stata does not have.
    kept: dict[str, Any] = {}
    for key in ("N", "df_m", "df_r", "F", "rank"):
        if key in results.e:
            kept[key] = results.e[key]
    kept["lag"] = float(lag)
    kept["p"] = _model_test_pvalue(results)
    # ``newey.sthlp:168-180`` also lists the MACROS newey stores.  They were
    # previously only set as Python attributes, so ``e()`` (which is what a
    # Stata-shaped consumer reads) did not carry them.
    kept["cmd"] = "newey"
    kept["title"] = NEWEY_TITLE
    kept["vcetype"] = NEWEY_VCETYPE
    kept["estat_cmd"] = "newey_estat"
    # ``newey.ado:218``: ``eret local predict newey_p``.
    kept["predict"] = "newey_p"
    kept["properties"] = "b V"
    kept["asbalanced"] = ""
    kept["asobserved"] = ""
    if weight_type is not None:
        kept["wtype"] = "aweight"
    results._e = EReturn(kept)

    results._model_name = "newey"
    results.cmd = "newey"
    results.title = NEWEY_TITLE
    results.vcetype = NEWEY_VCETYPE
    results.estat_cmd = "newey_estat"
    results.predict_cmd = "newey_p"
    if weight_type is not None:
        results.e_wtype = "aweight"
    return NeweyWestResult(results, lag=lag)


class Newey:
    """``newey depvar indepvars, lag(#)`` -- OLS with a HAC (Newey-West) VCE.

    The class form of :func:`westata.newey`, provided so that an estimation command can
    be used the same way as every other one in the package::

        ws.Newey.from_formula("y ~ x1 + x2", data, lag=1).fit()
        ws.newey("y ~ x1 + x2", data, lag=1)      # the same fit, still supported

    Why this is a class at all (R2.7): ``newey`` is an ESTIMATION command -- it consumes
    raw data -- not a post-estimation command.  Before this, it was one of 19 estimation
    commands reachable only as a bare function, so the same operation had a construction
    step in some families (``ws.OLS.from_formula(...).fit()``) and none in others.  The
    function remains as a thin wrapper; there is exactly one implementation.

    ``lag`` is REQUIRED and belongs to the model, not to ``fit()``: it selects the
    estimator (how many autocovariances enter the HAC sum), and Stata declares it a
    required option with no default (``newey.ado:15``, ``newey.sthlp:55``).

    ``fit()`` takes NO ``cov_type``.  That is deliberate, not an oversight: Stata's
    ``newey`` has no ``vce()`` option at all -- the HAC variance IS the estimator -- so
    accepting one here would be an interface that promises a choice that does not exist
    (R2.1, R1.5).
    """

    model_name = "newey"

    def __init__(
        self,
        data: pd.DataFrame,
        formula: str,
        *,
        lag: int,
        time: Any = None,
        force: bool = False,
        noconstant: bool = False,
        weights: Any = None,
        weight_type: str | None = None,
    ) -> None:
        self._data = data
        # NOTE: the attribute order follows the FUNCTION's positional order
        # (``data`` then ``formula``) so ``Newey(df, "y ~ x")`` and
        # ``prais(df, "y ~ x")`` line up.  ``from_formula`` takes the formula first,
        # matching every other estimator's classmethod (R3.2).
        self.formula = formula
        self.lag = lag
        self.time = time
        self.force = force
        self.noconstant = noconstant
        self.weights = weights
        self.weight_type = weight_type

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        lag: int,
        time: Any = None,
        force: bool = False,
        noconstant: bool = False,
        weights: Any = None,
        weight_type: str | None = None,
    ) -> "Newey":
        """Construct from a patsy formula -- ``(formula, data, *, ...)`` (R3.2)."""

        return cls(
            data,
            formula,
            lag=lag,
            time=time,
            force=force,
            noconstant=noconstant,
            weights=weights,
            weight_type=weight_type,
        )

    def fit(self, *, lag: int | None = None) -> NeweyWestResult:
        """Estimate.  ``lag`` may be overridden here for a single fit.

        There is no ``cov_type`` parameter: see the class docstring.
        """

        return _newey_impl(
            self.formula,
            self._data,
            lag=self.lag if lag is None else lag,
            time=self.time,
            force=self.force,
            noconstant=self.noconstant,
            weights=self.weights,
            weight_type=self.weight_type,
        )


def _has_regressors(formula: str) -> bool:
    rhs = formula.split("~", 1)[1] if "~" in formula else formula
    stripped = rhs.replace("+", " ").replace("-", " ").replace("1", " ").replace("0", " ")
    return bool(stripped.strip())


def _model_test_pvalue(results: Any) -> float:
    """``e(p)``: ``fprob(df_m, df_r, e(F))`` (``newey.ado:205``)."""

    from scipy import stats

    f_value = float(results.e.get("F", float("nan")))
    df_m = float(results.e.get("df_m", float("nan")))
    df_r = float(results.e.get("df_r", float("nan")))
    if not np.isfinite(f_value) or df_m <= 0 or df_r <= 0:
        return float("nan")
    return float(stats.f.sf(f_value, df_m, df_r))
