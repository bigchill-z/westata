"""Breusch-Pagan / Cook-Weisberg test for heteroskedasticity.

Stata's ``estat hettest``.  The algorithm is transcribed from ``hettest.ado``
(StataNow 19.5, v3.4.4), not inferred from output:

.. code-block:: stata

    qui _predict double `ui' if `touse', resid
    qui replace `ui' = `ui'^2 / (e(rss)/e(N))        // standardised squared resid
    ...
    if "`varlist'" == "" {                           // no varlist -> fitted values
        qui _predict double `z' if `touse'
        local varlist `z'
    }
    ...
    quiet _regress `ui' `varlist' `wgt' if `touse'

    if "`iid'"   { chi2 = e(N)*e(r2)  ; df = e(df_m) }
    elif "`fstat'" { F   = e(F)       ; df_m/df_r }
    else           { chi2 = e(mss)/2  ; df = e(df_m) }   // the DEFAULT

Four details decide whether the number is right, and each is a plausible way to
be wrong:

* **No varlist means the FITTED VALUES** (``hettest.ado:110-115``).  It does not
  mean "every regressor".  On ``regress mpg weight foreign`` the two differ:
  fitted values give ``chi2(1) = 2.53``, all regressors give ``chi2(2) = 7.75``.
  Both look like ordinary answers, so the mistake is silent; this module
  previously defaulted to all regressors and returned the wrong one.
* The auxiliary regression is of the **standardised squared residual**.  The
  divisor ``e(rss)/e(N)`` is the ML residual variance, and ``e(N)`` is the
  weighted observation count -- not ``rss/(N-K)`` and not the row count.
* The default statistic is ``e(mss)/2`` -- the model sum of squares of that
  auxiliary regression halved -- which equals ``N * R2 / 2`` only when the
  auxiliary regression is uncentred.  Using ``N * R2`` (the ``iid`` variant) as
  the default would be wrong by a factor of two on the same fit.
* **The estimation weights are reused** (``hettest.ado:103-105, 138``): the
  auxiliary regression is ``_regress ui varlist `wgt'``.  Dropping ``wgt``
  changes both the statistic and its degrees of freedom on any weighted fit.

``hettest.ado`` refuses four situations, all reproduced here: ``pweight``ed
data, a robust or clustered VCE, a ``noconstant`` fit, and ``normal`` combined
with ``iid``/``fstat``.

Reference values for every branch, captured from StataNow 19.5, live in
``tests/conformance/stata_fixtures/hettest_auto.json``; the test suite compares
against that file rather than against a second transcription of this ado.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["HetTestResult", "estat_hettest"]

_CONSTANT_NAMES = frozenset({"intercept", "intercept_", "_cons", "const"})

#: Methods accepted by ``estat hettest``.
_METHODS = ("normal", "iid", "fstat")


def _is_constant(name: str) -> bool:
    return str(name).strip().lower() in _CONSTANT_NAMES


class HetTestResult:
    """The Breusch-Pagan/Cook-Weisberg heteroskedasticity test.

    Attributes
    ----------
    method:
        ``"normal"`` (the default), ``"iid"`` or ``"fstat"``.
    statistic, df, pvalue:
        The reported test statistic, its degrees of freedom and the upper-tail
        p-value.  For ``method="fstat"`` ``df`` is the numerator df and
        ``df_denom`` holds the denominator.
    df_denom:
        Denominator df, only for ``method="fstat"``.
    variables:
        The auxiliary regressors used.  Empty means the fitted values were used,
        which is Stata's behaviour with no varlist.
    label:
        Stata's ``Variable:``/``Variables:`` line, e.g. ``"Fitted values of mpg"``
        or ``"All independent variables"``.
    """

    __slots__ = (
        "method",
        "statistic",
        "df",
        "pvalue",
        "df_denom",
        "variables",
        "label",
        "plural",
    )

    def __init__(
        self,
        *,
        method: str,
        statistic: float,
        df: float,
        pvalue: float,
        df_denom: float | None = None,
        variables: Sequence[str] = (),
        label: str = "",
        plural: bool = False,
    ) -> None:
        self.method = method
        self.statistic = float(statistic)
        self.df = float(df)
        self.pvalue = float(pvalue)
        self.df_denom = None if df_denom is None else float(df_denom)
        self.variables = list(variables)
        self.label = str(label)
        self.plural = bool(plural)

    def __repr__(self) -> str:
        return (
            f"<HetTestResult {self.method}: statistic={self.statistic:.6g} "
            f"df={self.df:g} p={self.pvalue:.4g}>"
        )

    def __str__(self) -> str:
        label = "F" if self.method == "fstat" else "chi2"
        lines = [
            "",
            # Stata's title line carries a TRAILING SPACE -- captured as
            # 'Breusch-Pagan/Cook-Weisberg test for heteroskedasticity ' (54 chars). Omitting it
            # made every `estat hettest` line-by-line comparison fail on this one row.
            "Breusch\u2013Pagan/Cook\u2013Weisberg test for heteroskedasticity ",
            "Assumption: "
            + ("Normal error terms" if self.method == "normal" else "i.i.d. error terms"),
        ]
        if self.label:
            # ``hettest.ado:258`` prints ``Variable`s':``.  The plural ``s`` is set
            # ONLY in the non-fitted branch and only when the varlist has more than
            # one word (``:118-120``), so the fitted-values line is always singular
            # -- Stata prints "Variable: Fitted values of mpg", not "Variables:".
            lines.append(f"{'Variables' if self.plural else 'Variable'}: {self.label}")
        lines.extend(["", "H0: Constant variance", ""])
        # Stata prints these two lines INDENTED by four spaces with no padded label column::
        #
        #         chi2(1) =   3.64
        #     Prob > chi2 = 0.0563
        #
        # so the statistic lines up under "H0:" rather than starting at column 1. Padding the
        # label to 20 characters produced "chi2(1)              =     3.64" -- same numbers,
        # visibly different table, which is exactly the kind of difference this round exists
        # to remove.
        if self.method == "fstat" and self.df_denom is not None:
            statistic_label = f"F({self.df:g}, {self.df_denom:g})"
            probability_label = "Prob > F"
        else:
            statistic_label = f"{label}({self.df:g})"
            probability_label = f"Prob > {label}"
        # The statistic is indented four spaces; the probability line is NOT. Measured:
        # "    chi2(1) =   3.64" and "Prob > chi2 = 0.0563", both 20 characters. Indenting
        # the second made it 24.
        lines.append(f"    {statistic_label} = {self.statistic:>6.2f}")
        lines.append(f"{probability_label} = {self.pvalue:>6.4f}")
        return chr(10).join(lines)

    def to_dict(self) -> dict[str, float]:
        """The ``r()`` scalars Stata returns for this variant."""

        if self.method == "fstat":
            return {
                "F": self.statistic,
                "df_m": self.df,
                "df_r": float(self.df_denom) if self.df_denom is not None else float("nan"),
                "p": self.pvalue,
            }
        return {"chi2": self.statistic, "df": self.df, "p": self.pvalue}


def _check_guards(result: Any) -> None:
    """The refusals ``hettest.ado`` applies, plus the linear-model requirement.

    ``hettest.ado:22-41`` refuses ``pweight`` and any robust/clustered VCE.  It
    does NOT check the estimator name, because in Stata the commands that reach
    this code always store the linear-regression quantities it needs.

    That is not true here: ``estat_hettest`` is a method on every result, so
    calling it after ``logit`` silently produced a number (0.00496 against OLS's
    0.00487 -- indistinguishable from a real answer) even though a logit result
    stores no ``e(rss)``/``e(mss)``, which ``hettest.ado:108,155`` require.  A
    number that looks right is worse than a refusal, so the requirement is
    enforced explicitly.
    """

    cov_type = str(getattr(result, "cov_type", "") or "").strip().lower()
    if cov_type in ("hc0", "hc1", "hc2", "hc3", "robust", "cluster"):
        raise ValueError(
            "estat hettest is not appropriate after estimation using a robust "
            f"or clustered VCE (this result used {cov_type})"
        )
    # ``e(wtype)`` pweight: westata records the weight type on the result when
    # one was used.  ``pweight`` implies a robust VCE here, so the check above
    # normally fires first; this keeps the refusal explicit for a pweighted fit
    # whose VCE was overridden.
    weight_type = getattr(result, "_weight_type", None)
    if weight_type is None:
        spec = getattr(result, "weight_spec", None)
        weight_type = getattr(spec, "weight_type", None)
    if str(weight_type or "").strip().lower() == "pweight":
        raise ValueError(
            "estat hettest is not appropriate with pweighted data"
        )

    # ``rreg.ado:214`` sets ``est local estat_cmd ""`` -- robust regression
    # deliberately offers NO estat subcommands, hettest included.  It does store
    # e(rss)/e(mss), so the least-squares check below would pass and the test
    # would return a number whose distribution is not the one hettest assumes
    # (the residuals come from an iteratively reweighted fit, not OLS).
    model_name = str(getattr(result, "model_name", "") or "").strip().lower()
    if model_name == "rreg":
        raise ValueError(
            "estat hettest is not available after rreg: rreg.ado:214 sets "
            "e(estat_cmd) to empty, so Stata offers no estat subcommands after "
            "robust regression.  The test assumes OLS residuals."
        )

    # ``xtreg_estat.ado:15-28`` allows only summarize, ic, hausman, mundlak and
    # vce; every other subcommand is "estat <sub> not valid" (exit 321).  The
    # panel estimators store e(rss)/e(mss) on their pooled regression, so the
    # least-squares check below let hettest through and it returned a number that
    # Stata refuses outright.
    if model_name.startswith("xtreg"):
        raise ValueError(
            "estat hettest is not available after xtreg: xtreg_estat.ado:15-28 "
            "permits only summarize, ic, hausman, mundlak and vce and rejects "
            "everything else with 'estat <sub> not valid'."
        )

    # ``hettest.ado:108`` scales by ``e(rss)/e(N)`` and ``:155`` reports
    # ``e(mss)/2``: both are least-squares quantities.  Without them the test has
    # no definition.
    e = getattr(result, "e", None)
    if e is not None:
        has_rss = "rss" in e
        has_mss = "mss" in e
        if not (has_rss and has_mss):
            model_name = str(getattr(result, "model_name", "this model") or "this model")
            raise ValueError(
                "estat hettest is a linear-regression test: it is defined from "
                "e(rss) and e(mss) (hettest.ado:108,155), which a "
                f"{model_name} result does not store.  Stata offers hettest after "
                "regress/areg/xtreg-style fits only."
            )

    # ``estat hettest`` after ``regress, noconstant`` is r(301), "not appropriate
    # after regress, nocons" -- the test presupposes a constant.  Without this
    # check the fit above passes the sums-of-squares guard and a finite number is
    # returned for a model Stata refuses.
    design = getattr(result, "design_matrix", None)
    if design is not None:
        names = [str(name) for name in design[1]]
        if names and not any(_is_constant(name) for name in names):
            raise ValueError(
                "estat hettest is not appropriate after a noconstant fit: Stata "
                "reports 'not appropriate after regress, nocons' (r(301)), "
                "because the test is defined from a centred decomposition."
            )


def _auxiliary_regression(
    left: np.ndarray,
    design: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[float, float, float, float]:
    """Weighted OLS of ``left`` on ``design``; returns ``(r2, mss, rss, df_m)``.

    ``hettest.ado:138`` runs ``_regress ui varlist `wgt'``, so when the fit was
    weighted the auxiliary regression is weighted too.  ``e(r2)``/``e(mss)``/
    ``e(rss)`` are then the weighted quantities, with the total sum of squares
    taken about the *weighted* mean.
    """

    with_constant = any(np.ptp(design[:, j]) == 0.0 for j in range(design.shape[1]))
    if not with_constant:
        design = np.column_stack([np.ones(left.size), design])

    if weights is None:
        coefficients, *_ = np.linalg.lstsq(design, left, rcond=None)
        resid = left - design @ coefficients
        rss = float(resid @ resid)
        centred = left - left.mean()
        tss = float(centred @ centred)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        root = np.sqrt(w)
        coefficients, *_ = np.linalg.lstsq(
            design * root[:, None], left * root, rcond=None
        )
        resid = left - design @ coefficients
        rss = float((w * resid * resid).sum())
        total_w = float(w.sum())
        mean_w = float((w * left).sum() / total_w)
        tss = float((w * (left - mean_w) ** 2).sum())

    mss = tss - rss
    r2 = (mss / tss) if tss > 0 else float("nan")
    df_m = int(design.shape[1] - 1)  # slopes only, matching e(df_m)
    return r2, mss, rss, df_m


def _weight_vector(result: Any, n_rows: int) -> np.ndarray | None:
    """The estimation weights, aligned to the residual vector.

    ``StataResults.weight_spec`` holds them; a result that did not keep a
    specification is unweighted.  A length mismatch is a bug in the result
    object, not something to paper over with ``None``, because silently dropping
    the weights reproduces exactly the defect this module used to have.
    """

    spec = getattr(result, "weight_spec", None)
    if spec is None or getattr(spec, "weights", None) is None:
        return None
    weights = np.asarray(spec.weights, dtype=float).reshape(-1)
    if weights.size != n_rows:
        raise ValueError(
            f"the estimation weights have {weights.size} entries but there are "
            f"{n_rows} residuals; the result's weight specification is not "
            "aligned with its sample"
        )
    return weights


def _effective_n(result: Any, weights: np.ndarray | None, n_rows: int) -> float:
    """``e(N)`` for a weighted fit: ``sum(w)`` for fweight/iweight, rows for aweight.

    ``e(N)`` is what ``hettest.ado:108`` divides ``e(rss)`` by and what
    ``:144`` multiplies ``e(r2)`` by, so using the row count on an fweighted fit
    is wrong by ``sum(w)/rows``.
    """

    if weights is None:
        return float(n_rows)
    spec = getattr(result, "weight_spec", None)
    effective = getattr(spec, "effective_n", None)
    if effective is not None:
        return float(effective)
    return float(weights.sum())


def _ancillary_label(
    result: Any, used_names: Sequence[str], *, rhs: bool, fitted: bool
) -> str:
    """Stata's ``Variable:``/``Variables:`` line for this auxiliary regression."""

    if rhs:
        # ``hettest.ado:121-123``: with ``rhs`` the caption is always this,
        # regardless of what else the varlist named.
        return "All independent variables"
    if fitted:
        depvar = getattr(result, "response_name", None)
        if depvar is None:
            e = getattr(result, "e", None)
            depvar = e.get("depvar") if e is not None else None
        return f"Fitted values of {depvar}" if depvar else "Fitted values"
    return " ".join(used_names)


def _source_variable_design(
    result: Any, requested: Sequence[str], names: Sequence[str]
) -> tuple[np.ndarray, list[str]] | None:
    """Build the auxiliary design from SOURCE data variables, if possible.

    This is the distinction that ``hettest.ado:138`` makes and that a purely
    design-column view gets wrong: the varlist names **data variables**, so
    ``estat hettest rep78`` after ``regress mpg weight i.rep78`` regresses ``ui``
    on the single raw 1-5 column (``e(df_m)`` = 1, chi2 = 12.77), not on the four
    treatment dummies in ``e(b)`` (chi2 = 24.27).  Verified on StataNow 19.5.

    Returns ``None`` when the model did not retain its data, or when a requested
    name is not a column of it -- the caller then falls back to design columns so
    an explicit ``e(b)`` name (``"foreign[T.Foreign]"``) keeps working.
    """

    model_data = getattr(result, "model_data", None)
    if model_data is None or not hasattr(model_data, "columns"):
        return None
    frame = model_data
    if len(frame) != np.asarray(result.resid, dtype=float).size:
        return None

    columns: list[np.ndarray] = []
    used: list[str] = []
    for wanted in requested:
        if wanted not in frame.columns:
            return None
        series = frame[wanted]
        # Check the SERIES dtype, not ``to_numpy()``: pandas turns a categorical
        # into an object array, so testing the array would miss every labelled
        # Stata variable and fall through to the string branch.
        if isinstance(series.dtype, pd.CategoricalDtype):
            # A value-labelled Stata variable is numeric with a value label;
            # ``estat hettest foreign`` therefore uses the raw 0/1 codes, not the
            # dummies.  On the auto data that is chi2(1) = 7.66936122444203594,
            # whereas ``rhs`` (the e(b) columns) gives chi2(2) = 7.7524933353.
            numeric = np.asarray(series.cat.codes.to_numpy(), dtype=float)
        elif series.dtype.kind in ("U", "S", "O"):
            # A true string variable: ``_regress ui make`` is r(109) in Stata.
            raise ValueError(
                f"estat hettest: {wanted!r} is a string variable; Stata's "
                "_regress rejects it with a type mismatch (r(109))"
            )
        else:
            numeric = np.asarray(series.to_numpy(), dtype=float)
        # Stata's `_regress ui varlist` always adds a constant, so a raw
        # continuous/numeric variable contributes exactly ONE column.
        columns.append(numeric)
        used.append(wanted)
    design = np.column_stack(columns)
    return design, used


def _stata_row_labels(result: Any, used_names: Sequence[str]) -> list[str]:
    """Row names for ``r(mtest)``, in Stata's ``e(b)`` spelling.

    ``hettest.ado:139`` takes the names from ``e(b)``'s column stripes, so a
    factor row prints as ``foreign`` where the design column here is
    ``foreign[T.Foreign]``.  ``stata_param_names`` owns that mapping -- and it
    refuses when a factor term is genuinely ambiguous, which is the right answer
    for a table whose rows are meant to match ``r(mtest)``; falling back to the
    design spelling keeps the table usable either way.
    """

    try:
        # ``stata_param_names`` is in Stata's e(b) ORDER, so it must be zipped
        # against the permuted design names (``_stata_order``), not the raw ones.
        design_names = [str(name) for name in result.design_matrix[1]]
        order = result._stata_order()
        mapping = {
            design_names[position]: stata_name
            for position, stata_name in zip(order, result.stata_param_names)
        }
    except (AttributeError, NotImplementedError, TypeError):
        return list(used_names)
    return [mapping.get(name, name) for name in used_names]


def estat_hettest(
    result: Any,
    *,
    method: str = "normal",
    variables: Sequence[str] | None = None,
    rhs: bool = False,
    mtest: str | None = None,
) -> Any:
    """Stata's ``estat hettest`` (Breusch-Pagan / Cook-Weisberg).

    Parameters
    ----------
    result:
        A fitted linear result that retained its design, residuals and weights.
    method:
        ``"normal"`` (default) reports ``e(mss)/2``; ``"iid"`` reports
        ``e(N)*e(r2)``; ``"fstat"`` reports the auxiliary regression's ``F``.
    variables:
        Design columns for the auxiliary regression, named exactly as they appear
        in ``result.design_matrix[1]`` (a factor level is e.g.
        ``"foreign[T.Foreign]"``).  ``None`` or an empty sequence regresses on
        the **fitted values**, which is Stata's behaviour with no varlist -- it
        is not "all regressors"; use ``rhs=True`` for that.
    rhs:
        Stata's ``, rhs``: add every non-constant regressor to the auxiliary
        regression (``hettest.ado:69-76``).  Combines with ``variables`` as a
        union, exactly as the ado does.
    mtest:
        Request the per-regressor multiple-testing table.  As in
        ``hettest.ado:126-128``, a single-regressor auxiliary regression drops
        ``mtest()`` silently and only the simultaneous test is returned.

    Raises
    ------
    ValueError
        For a robust/clustered VCE, pweighted data, a noconstant fit, an
        inconsistent ``method``/option combination, or when the design was not
        retained.
    """

    normalized = str(method).strip().lower()
    if normalized not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")

    # ``hettest.ado:49-61`` refuses mtest() together with iid and fstat.
    if mtest is not None:
        if normalized == "iid":
            raise ValueError(
                "option iid may not be specified together with option mtest()"
            )
        if normalized == "fstat":
            raise ValueError(
                "option fstat may not be specified together with option mtest()"
            )
        from westata.postestimation.mtest import MTEST_METHODS

        if str(mtest).strip().lower() not in MTEST_METHODS:
            raise ValueError(
                f"mtest() method {mtest!r} is not supported; Stata's _mtest "
                "accepts " + ", ".join(MTEST_METHODS)
            )

    _check_guards(result)

    design = getattr(result, "design_matrix", None)
    if design is None:
        raise ValueError(
            "estat hettest needs the estimation design, which this result did "
            "not retain; refit with a from_formula constructor"
        )
    matrix_raw, raw_names = design
    matrix = np.asarray(matrix_raw, dtype=float)
    names = [str(name) for name in raw_names]

    resid = np.asarray(result.resid, dtype=float).reshape(-1)
    nobs = resid.size
    if matrix.shape[0] != nobs:
        raise ValueError(
            f"the design has {matrix.shape[0]} rows but there are {nobs} residuals"
        )

    weights = _weight_vector(result, nobs)
    effective_n = _effective_n(result, weights, nobs)

    # ui = resid^2 / (e(rss)/e(N)) -- the ML residual variance, per hettest.ado.
    # ``e(rss)`` is the WEIGHTED residual sum of squares for a weighted fit, and
    # ``e(N)`` the weighted count, so the scale uses both.
    e = getattr(result, "e", {}) or {}
    rss_model = float(e.get("rss", float(resid @ resid)))
    scale = rss_model / effective_n
    if scale <= 0.0:
        raise ValueError("estat hettest needs a positive residual variance")
    squared = resid**2 / scale

    # ----------------------------------------------------- auxiliary regressors
    # ``variables is None`` / ``[]`` -> fitted values (hettest.ado:110-115).
    slope_columns = [j for j in range(len(names)) if not _is_constant(names[j])]
    lookup = {names[j]: j for j in range(len(names))}

    explicit: list[str] = []
    if variables is not None:
        explicit = [str(v) for v in variables]

    # ``rhs`` appends every e(b) name except ``_cons`` to the varlist, then
    # de-duplicates (``hettest.ado:69-76, 76``).
    requested = list(explicit)
    if rhs:
        for column in slope_columns:
            if names[column] not in requested:
                requested.append(names[column])

    missing = [v for v in requested if v not in lookup]
    source: tuple[np.ndarray, list[str]] | None = None
    if missing:
        # Not an e(b) name.  It may still be a SOURCE variable -- Stata's varlist
        # names data variables, so ``hettest rep78`` after ``i.rep78`` means the
        # raw column, not the dummies.  Resolve when the model kept its data.
        if not rhs:
            source = _source_variable_design(result, missing, names)
        if source is None:
            raise ValueError(
                f"estat hettest: unknown regressor(s) {missing}; available: {names}. "
                "Pass the raw data variable name (e.g. 'rep78', which Stata uses "
                "as one column) or an e(b) design column (e.g. 'foreign[T.Foreign]'); "
                "use rhs=True for every regressor."
            )

    fitted_values = not requested
    if fitted_values:
        # A DESIGNED-OUT CONSTANT MAKES THE TWO SIDES DIFFERENT WIDTHS. An absorbed-effects fit has
        # no Intercept in `params` (the effects absorb it) while its design matrix still carries the
        # column, so a positional product raised "1 coefficients for 2 design columns" on `fe` fits
        # that had nothing wrong with them.
        #
        # Align by NAME instead: a design column with no matching coefficient contributes nothing to
        # the linear prediction, which is exactly right for an absorbed intercept.
        coefficients = np.asarray(result.params, dtype=float).reshape(-1)
        parameter_names = list(getattr(result, "param_names", []) or result.params.index)
        # The design's own names live beside the matrix on the results that carry both.
        design_names: list[str] = []
        stored = getattr(result, "design_matrix", None)
        if isinstance(stored, (tuple, list)) and len(stored) >= 2:
            candidate = list(stored[1])
            if len(candidate) == matrix.shape[1]:
                design_names = [str(name) for name in candidate]

        if coefficients.size == matrix.shape[1]:
            aux_design = (matrix @ coefficients).reshape(-1, 1)
        elif design_names:
            lookup_coefficient = {
                str(name): float(coefficients[index])
                for index, name in enumerate(parameter_names)
            }
            aligned = np.array(
                [lookup_coefficient.get(str(name), 0.0) for name in design_names],
                dtype=float,
            )
            aux_design = (matrix @ aligned).reshape(-1, 1)
        else:
            raise ValueError(
                "estat hettest cannot rebuild the fitted values: "
                f"{coefficients.size} coefficients for "
                f"{matrix.shape[1]} design columns, and the result exposes no design names to "
                "align them by"
            )
        used_columns: list[int] = []
        used_names: list[str] = []
    elif source is not None:
        # Requested SOURCE variables (e.g. ["rep78"]) -> raw data columns.  Stata
        # mixes them with e(b) names in one varlist; when only source names were
        # given this is the whole design.
        source_design, source_names = source
        resolved = [v for v in requested if v in lookup]
        if resolved:
            remainder = matrix[:, [lookup[v] for v in resolved]]
            aux_design = np.column_stack([source_design, remainder])
            used_columns = [lookup[v] for v in resolved]
            used_names = list(source_names) + resolved
        else:
            aux_design = source_design
            used_columns = []
            used_names = list(source_names)
    else:
        used_columns = [lookup[v] for v in requested]
        used_names = list(requested)
        aux_design = matrix[:, used_columns]

    r2, mss, rss_aux, df_m = _auxiliary_regression(squared, aux_design, weights)
    if df_m <= 0:
        raise ValueError(
            "estat hettest needs at least one regressor in the auxiliary "
            "regression to have a non-zero degrees of freedom"
        )

    label = _ancillary_label(result, used_names, rhs=rhs, fitted=fitted_values)
    # ``hettest.ado:117-120, 258``: the plural ``s`` is set in the NON-fitted
    # branch only, and only when the varlist has more than one word.  So the
    # fitted-values line is always "Variable:", ``, rhs`` on a one-regressor model
    # is "Variable: All independent variables", and two regressors is "Variables:".
    # Verified on StataNow 19.5.
    plural_label = (not fitted_values) and df_m > 1

    # ------------------------------------------------------------------ mtest
    # ``hettest.ado:126-128`` clears ``mtmethod`` when the auxiliary regression
    # has ONE regressor.  The no-varlist default always does, so ``mtest()``
    # there is silently ignored and only the simultaneous test is reported --
    # verified against StataNow 19.5, which returns no ``r(mtest)`` at all.
    if mtest is not None and df_m >= 2:
        import pandas as pd

        from westata.postestimation.mtest import (
            MTestResult,
            adjust_pvalues,
            mtest_statistics,
        )

        per_variable = matrix[:, used_columns]
        statistics, dfs, pvalues = mtest_statistics(
            squared, per_variable, used_names, weights
        )
        method_key = str(mtest).strip().lower()
        # ``r(mtest)``'s row names come from ``e(b)``'s stripes, so label them
        # Stata's way (``weight foreign``) rather than patsy's.
        frame = pd.DataFrame(
            {"chi2": statistics, "df": dfs, "p": pvalues},
            index=pd.Index(_stata_row_labels(result, used_names), name="Variable"),
        )
        if method_key != "noadjust":
            frame["adjustp"] = adjust_pvalues(pvalues, method_key)
        return MTestResult(frame, method_key)

    if normalized == "iid":
        # ``chi2 = e(N)*e(r2)``; e(N) is the AUXILIARY regression's observation
        # count, which is the weighted count for a weighted fit.
        statistic = float(effective_n * r2)
        pvalue = float(stats.chi2.sf(statistic, df_m))
        return HetTestResult(
            method="iid",
            statistic=statistic,
            df=df_m,
            pvalue=pvalue,
            variables=used_names,
            label=label,
            plural=plural_label,
        )

    if normalized == "fstat":
        # The F form reports the auxiliary regression's own F and df_r.
        denominator = float(effective_n - df_m - 1)
        if denominator <= 0:
            raise ValueError(
                "estat hettest, fstat needs positive residual degrees of freedom"
            )
        f_stat = (mss / df_m) / (rss_aux / denominator) if rss_aux > 0 else float("inf")
        return HetTestResult(
            method="fstat",
            statistic=float(f_stat),
            df=df_m,
            pvalue=float(stats.f.sf(f_stat, df_m, denominator)),
            df_denom=denominator,
            variables=used_names,
            label=label,
            plural=plural_label,
        )

    # Default: chi2 = e(mss)/2.
    statistic = float(mss / 2.0)
    return HetTestResult(
        method="normal",
        statistic=statistic,
        df=df_m,
        pvalue=float(stats.chi2.sf(statistic, df_m)),
        variables=used_names,
        label=label,
        plural=plural_label,
    )
