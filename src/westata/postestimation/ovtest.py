"""Ramsey's RESET test for omitted variables: Stata's ``estat ovtest``.

Transcribed from ``ovtest.ado`` (StataNow 19.5, v2.7.0).  The default path
(``freset``, which Stata takes whenever ``rhs`` is not given) is:

.. code-block:: stata

    quietly _predict double `yh' if `use'
    summ `yh'
    replace `yh' = (`yh' - r(min)) / (r(max) - r(min))    // rescale to [0,1]
    gen float `yh2' = `yh'*`yh'
    gen float `yh3' = `yh'*`yh'*`yh'
    gen float `yh4' = `yh'*`yh'*`yh'*`yh'
    _regress `lhs' `varlist' `yh2' `yh3' `yh4' `wgtexp' if `use'
    test `yh2' `yh3' `yh4'

so the test is a joint F on the second, third and fourth powers of the
**rescaled** fitted values, added to the original regressors.

The rescaling to ``[0, 1]`` is not cosmetic: ``yh^4`` on a raw fitted value in
the hundreds overflows the useful range and makes the auxiliary design
ill-conditioned, while on a ``[0, 1]`` scale the four powers span a sane basis.
Dropping it changes the test statistic.

``estat ovtest, rhs`` takes the alternative path and uses powers of the *right-
hand-side variables* instead (``ovtest.ado``'s ``else`` branch): every continuous
regressor is rescaled to ``[0, 1]`` and contributes ``x^2, x^3, x^4``; indicator
variables (all values in ``{0, 1}``) contribute only the variable itself.  When
no continuous regressor remains, Stata reports "powers of all explanatory
variables collinear" and the test is not possible.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy import stats
from westata.postestimation.names import is_constant_name

#: The constant-name test lives in `names.py` (R6.1).
_is_constant = is_constant_name

__all__ = ["OvTestResult", "estat_ovtest"]


#: The RESET test always uses the 2nd, 3rd and 4th powers.
_POWERS = (2, 3, 4)



class OvTestResult:
    """Ramsey's RESET test.

    Attributes
    ----------
    statistic, df, df_denom, pvalue:
        The F statistic, its numerator and denominator degrees of freedom, and
        the upper-tail p-value.
    omitted:
        Human-readable description of what the powers were taken of, matching
        Stata's ``Omitted:`` line.
    """

    __slots__ = ("statistic", "df", "df_denom", "pvalue", "omitted", "dropped")

    def __init__(
        self,
        *,
        statistic: float,
        df: float,
        df_denom: float,
        pvalue: float,
        omitted: str,
        dropped: Sequence[str] = (),
    ) -> None:
        self.statistic = float(statistic)
        self.df = float(df)
        self.df_denom = float(df_denom)
        self.pvalue = float(pvalue)
        self.omitted = str(omitted)
        #: Columns the auxiliary regression could not identify, in the order Stata reports them.
        #: Each produces a "note: <name> omitted because of collinearity." line ABOVE the table.
        self.dropped = tuple(str(name) for name in dropped)

    def __repr__(self) -> str:
        return (
            f"<OvTestResult F({self.df:g}, {self.df_denom:g})="
            f"{self.statistic:.4g} p={self.pvalue:.4g}>"
        )

    def __str__(self) -> str:
        """Reproduce ``estat ovtest`` exactly::

            Ramsey RESET test for omitted variables
            Omitted: Powers of fitted values of consumption
            H0: Model has no omitted variables
            F(3, 23) =   3.08
            Prob > F = 0.0474

        The label column is NOT padded to a fixed width -- ``F(3, 23)`` is followed straight
        away by `` = `` and the value right-aligned in eight. Left-aligning the label in a
        20-character field produced ``F(3, 23)             =     3.08``, which is the same
        numbers in a visibly different table.
        """

        label = f"F({self.df:g}, {self.df_denom:g})"
        # A COLLINEAR COLUMN EARNS A NOTE ABOVE THE TABLE, one line each, in the order the ado
        # prints them. Measured on grilic after a user adds expr2 = expr^2:
        #
        #     note: expr2 omitted because of collinearity.
        #     note: expr2^2 omitted because of collinearity.
        #     (blank)
        #     Ramsey RESET test for omitted variables
        #
        # The default path never produces one -- it fits powers of a single fitted-value vector,
        # which cannot duplicate a regressor -- so this stays empty there.
        notes = [
            f"note: {name} omitted because of collinearity." for name in self.dropped
        ]
        return chr(10).join(
            notes
            + [
                "",
                "Ramsey RESET test for omitted variables",
                f"Omitted: Powers of {self.omitted}",
                "",
                "H0: Model has no omitted variables",
                "",
                # Both lines are 17 characters: the value is right-aligned in SIX after
                # "X = ". Seven characters made "F(3, 23) =    3.08" one column wider than
                # Stata's "F(3, 23) =   3.08".
                f"{label} = {self.statistic:>6.2f}",
                # THE Prob ROW IS AS WIDE AS THE F ROW ABOVE IT, and 'Prob' is right-aligned to that
                # width rather than indented by a fixed amount. Measured at two sample sizes:
                #
                #     n=200   'F(3, 194) =   1.63'   (18)   ' Prob > F = 0.1845'   (18)
                #     n=1500  'F(3, 1494) =   0.34'  (19)   '  Prob > F = 0.7969'  (19)
                #
                # The F row grows by one character when the degrees of freedom reach four digits, and the
                # Prob row grows with it. A fixed one-space indent is right for the first case only.
                " " * (len(f"{label} = {self.statistic:>6.2f}") - len(f"Prob > F = {self.pvalue:.4f}"))
                + f"Prob > F = {self.pvalue:.4f}",
            ]
        )

    def to_dict(self) -> dict[str, float]:
        """The ``r()`` scalars Stata returns."""

        return {
            "F": self.statistic,
            "df": self.df,
            "df_r": self.df_denom,
            "p": self.pvalue,
        }


def _rescale(values: np.ndarray) -> tuple[np.ndarray, bool]:
    """Rescale to ``[0, 1]``; second element reports whether it was constant."""

    low = float(values.min())
    high = float(values.max())
    if high == low:
        return values, True
    return (values - low) / (high - low), False


def _fit_and_test(
    y: np.ndarray, design: np.ndarray, tested: Sequence[int]
) -> tuple[float, float, float]:
    """OLS of ``y`` on ``design``; joint F on the ``tested`` columns."""

    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficients
    resid = y - fitted
    rss = float(resid @ resid)
    nobs, k = design.shape
    df_denom = float(nobs - k)
    df_num = float(len(tested))
    if df_denom <= 0:
        raise ValueError(
            "estat ovtest needs positive residual degrees of freedom; the "
            f"auxiliary regression has {nobs} observations and {k} parameters"
        )
    if rss <= 0.0:
        raise ValueError(
            "estat ovtest: the auxiliary regression fits exactly, so the F "
            "statistic is undefined"
        )
    # Wald form of the joint restriction, divided by df_num, matching Stata's
    # ``test`` after the auxiliary regression.
    cov = rss / df_denom * np.linalg.pinv(design.T @ design)
    sub = np.asarray(list(tested), dtype=int)
    b = coefficients[sub]
    v = cov[np.ix_(sub, sub)]
    wald = float(b @ np.linalg.solve(v, b))
    return wald / df_num, df_num, df_denom



def _fitted_label(result: Any) -> str:
    """``fitted values of <depvar>``, as ``estat ovtest`` prints it."""

    response = str(
        getattr(result, "response_name", None)
        or (getattr(result, "e", None) or {}).get("depvar")
        or ""
    ).strip()
    return f"fitted values of {response}" if response else "fitted values"


def estat_ovtest(
    result: Any,
    *,
    rhs: bool = False,
) -> OvTestResult:
    """Stata's ``estat ovtest`` (Ramsey RESET).

    Parameters
    ----------
    result:
        A fitted linear result that retained its design and residuals.
    rhs:
        Stata's ``, rhs``: test powers of the right-hand-side variables instead
        of powers of the fitted values.

    Raises
    ------
    ValueError
        When the design was not retained, when every regressor is an indicator
        (so no powers exist), or when the auxiliary fit has no residual df.
    """

    design = getattr(result, "design_matrix", None)
    if design is None:
        raise ValueError(
            "estat ovtest needs the estimation design, which this result did "
            "not retain; refit with a from_formula constructor"
        )

    # ``rreg.ado:214`` sets ``est local estat_cmd ""``: robust regression offers
    # no estat subcommands.  It DOES store e(rss)/e(mss), so the sums-of-squares
    # check below would let it through, but the residuals come from an
    # iteratively reweighted fit -- not the OLS residuals ovtest is derived from.
    if str(getattr(result, "model_name", "") or "").strip().lower() == "rreg":
        raise ValueError(
            "estat ovtest is not available after rreg: rreg.ado:214 sets "
            "e(estat_cmd) to empty, so Stata offers no estat subcommands after "
            "robust regression.  The test assumes OLS residuals."
        )

    # ``xtreg_estat.ado:15-28`` permits only summarize, ic, hausman, mundlak and
    # vce; everything else is "estat <sub> not valid".  Panel results carry
    # e(rss)/e(mss), so the sums-of-squares check alone let ovtest through.
    if str(getattr(result, "model_name", "") or "").strip().lower().startswith("xtreg"):
        raise ValueError(
            "estat ovtest is not available after xtreg: xtreg_estat.ado:15-28 "
            "permits only summarize, ic, hausman, mundlak and vce and rejects "
            "everything else with 'estat <sub> not valid'."
        )

    # ``ovtest`` is an OLS specification test: it regresses the residuals on
    # powers of the fitted values and compares sums of squares.  Called after a
    # non-linear estimator it still "works" numerically -- it just performs OLS on
    # whatever ``fittedvalues`` holds, which for a logit is a probability.  On the
    # audit data that produced 1.3478 for logit against 1.3508 for OLS: two nearly
    # identical numbers, the logit one meaningless.  Require the least-squares
    # quantities explicitly.
    e = getattr(result, "e", None)
    if e is not None and not ("rss" in e and "mss" in e):
        model_name = str(getattr(result, "model_name", "this model") or "this model")
        raise ValueError(
            "estat ovtest is an OLS specification test: it compares sums of "
            "squares, and a "
            f"{model_name} result stores neither e(rss) nor e(mss).  Stata offers "
            "ovtest after regress-style fits only."
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
    y = np.asarray(result.fittedvalues, dtype=float).reshape(-1) + resid

    if not rhs:
        # Default: powers of the fitted values, rescaled to [0, 1].
        fitted = np.asarray(result.fittedvalues, dtype=float).reshape(-1)
        scaled, constant = _rescale(fitted)
        if constant:
            raise ValueError(
                "estat ovtest: the fitted values are constant, so their powers "
                "are collinear with the regressors"
            )
        extra = np.column_stack([scaled**power for power in _POWERS])
        design_full = np.column_stack([matrix, extra])
        tested = list(range(matrix.shape[1], design_full.shape[1]))
        statistic, df_num, df_denom = _fit_and_test(y, design_full, tested)
        return OvTestResult(
            statistic=statistic,
            df=df_num,
            df_denom=df_denom,
            pvalue=float(stats.f.sf(statistic, df_num, df_denom)),
            # Stata names the dependent variable at the end: "Powers of fitted values of
            # consumption". Without it the line reads "Powers of fitted values", which is a
            # different sentence rather than a shortened one.
            omitted=_fitted_label(result),
        )

    # ``, rhs``: powers of each continuous regressor.
    columns: list[np.ndarray] = []
    omitted_labels: list[str] = []
    for j, name in enumerate(names):
        if _is_constant(name):
            continue
        values = matrix[:, j]
        if np.all(np.isin(values, (0.0, 1.0))):
            # Indicators are SKIPPED entirely, not re-added. Stata's ``ovtest, rhs`` leaves
            # indicator regressors out of the test set; adding them here produced a column
            # perfectly collinear with the regressor already in the design, so every dummy
            # counted as TWO extra test columns.
            #
            # Measured on grilic (n = 758):
            #   lnw s expr tenure          -> F(9, 745) = 1.70 on both sides
            #   lnw s expr tenure rns smsa -> Stata F(9, 743) = 2.03
            #                                 was   F(11, 741) = 5.21
            continue
        scaled, constant = _rescale(values)
        if constant:
            continue
        for power in _POWERS:
            columns.append(scaled**power)
            omitted_labels.append(f"{name}^{power}")

    # THE GUARD COUNTS CONTINUOUS REGRESSORS, NOT ALL OF THEM. Comparing the number of power
    # columns (3 per continuous variable) against every non-constant name made the test fire
    # whenever the design had at least three indicators -- and with factors present the names
    # INCLUDED the dummies, so `estat ovtest, rhs` raised on a perfectly ordinary model:
    #
    #     reg y x1 i.grp     Stata  F(3, 293) = 0.17   package  ValueError
    #
    # What Stata refuses is a model with NO continuous regressor at all, which is exactly
    # `ovtest.ado`'s message "powers of all explanatory variables collinear". One continuous
    # variable always contributes its three powers and is testable.
    _continuous = len(omitted_labels) // len(_POWERS)
    if _continuous == 0 or not omitted_labels:
        raise ValueError(
            "estat ovtest, rhs: powers of all explanatory variables collinear "
            "(typically because all explanatory variables are indicator variables)"
        )

    # THE AUXILIARY REGRESSION PUTS THE POWERS BEFORE THE USER'S REGRESSORS. `ovtest.ado:420`
    # is `_regress lhs zlist rhs`, and `_regress` omits any column that the earlier columns can
    # reproduce. Order therefore changes the answer, and testing the powers against the finished
    # model is wrong:
    #
    #     grilic, user adds expr2 = expr^2
    #     Stata    F(11, 741) = 1.73   drops expr2^2 among the powers, then expr2 on entering
    #     wrong    F(10, 741) = 1.27   also drops expr^2, because expr^2 is dependent on the
    #                                  finished model, which already contains expr2
    #
    # `test zlist` then tests only the powers, so the reported df counts 11 while the auxiliary
    # regression carries 17 parameters.
    _design_columns = [matrix[:, [0]]]
    _tested: list[int] = []
    #: Columns `_regress` cannot identify, in the order it meets them. Stata prints one
    #: "note: <name> omitted because of collinearity." line per entry, above the table.
    _dropped_regressors: list[str] = []
    _dropped_powers: list[str] = []
    for _label, _column in zip(omitted_labels, columns):
        _current = np.column_stack(_design_columns)
        _scale = float(np.abs(_column).max())
        if _scale == 0.0:
            _dropped.append(_label)
            continue
        _coef, *_ = np.linalg.lstsq(_current, _column, rcond=None)
        _residual = _column - _current @ _coef
        if float(np.abs(_residual).max()) <= 1e-6 * _scale:
            _dropped_powers.append(_label)
            continue
        _tested.append(len(_design_columns))
        _design_columns.append(_column[:, None])
    for _index in range(1, matrix.shape[1]):
        _column = matrix[:, _index]
        _current = np.column_stack(_design_columns)
        _scale = float(np.abs(_column).max())
        if _scale == 0.0:
            continue
        _coef, *_ = np.linalg.lstsq(_current, _column, rcond=None)
        _residual = _column - _current @ _coef
        if float(np.abs(_residual).max()) <= 1e-6 * _scale:
            _dropped_regressors.append(str(names[_index]))
            continue
        _design_columns.append(_column[:, None])
    # STATA LISTS THE USER'S REGRESSORS BEFORE THE POWERS in its notes, even though the powers
    # are what enter the model first. `ovtest.ado:310,373` builds `ckdrp` from `e(b)`'s column
    # order, so `expr2` precedes the generated `expr2^2`; the loop at 431 then prints them in
    # that order. Measured on grilic:
    #
    #     note: expr2 omitted because of collinearity.
    #     note: expr2^2 omitted because of collinearity.
    _dropped = _dropped_regressors + _dropped_powers

    if not _tested:
        raise ValueError(
            "estat ovtest, rhs: powers of all explanatory variables collinear "
            "(typically because all explanatory variables are indicator variables)"
        )
    design_full = np.column_stack(_design_columns)
    statistic, df_num, df_denom = _fit_and_test(y, design_full, _tested)
    return OvTestResult(
        statistic=statistic,
        df=df_num,
        df_denom=df_denom,
        pvalue=float(stats.f.sf(statistic, df_num, df_denom)),
        omitted="independent variables",
        dropped=_dropped,
    )
