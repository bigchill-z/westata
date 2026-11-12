"""Model-comparison tests: ``lrtest`` and ``hausman``.

``[R] lrtest`` compares two nested models by their log-likelihoods:

    LR = 2 (lnL_unrestricted - lnL_restricted)  ~  chi2(k_u - k_r)

``[R] hausman`` tests whether the efficient estimator is also consistent, using
the difference between a consistent and an efficient estimate:

    H = (b_c - b_e)' [V_c - V_e]^-1 (b_c - b_e)  ~  chi2(k)

Both tests are only meaningful under assumptions westata cannot verify from the
numbers alone, so this module checks what *is* checkable (nesting, sample size,
shared parameters) and says so explicitly when a precondition fails.

Stata conformance notes
-----------------------
``lrtest`` with ``df == 0``
    Stata refuses the command outright:
    ``df(unrestricted) = df(restricted) = 2`` / ``r(498)``.  Two models with the
    same parameter count cannot be ranked by a likelihood-ratio test, and
    ``chi2(0)`` is a point mass at 0 whose survival function is 1, so the old
    convention of "report ``p = 1.0``" turned a Stata error into a successful
    command.  This raises :class:`ConsistencyError` instead, mirroring r(498).
``lrtest`` on models with different regressors but the same parameter count
    Stata reaches the same r(498) check *before* looking at names, because its
    nesting test is the parameter count alone.  So ``regress mpg weight`` versus
    ``regress mpg foreign`` is r(498) too, and only a pair with different counts
    gets an LR statistic -- which can be **negative** when the models are not
    nested (``lrtest`` on ``mpg weight`` vs ``mpg foreign length`` prints
    ``LR chi2(1) = -2.6835, Prob > chi2 = 1.0000``).  A negative LR is therefore
    reported, warned about, and never clamped.
``hausman`` and the constant
    Stata's ``hausman`` compares the **slope** coefficients and excludes the
    constant by default (``[R] hausman`` exposes ``constant`` as an opt-in
    option precisely because it is not comparable).  The reason is not cosmetic:
    a within-transformed intercept and a GLS intercept estimate different
    population quantities, so ``V_fe - V_re`` is indefinite as soon as the
    constant enters - verified on ordinary panels in the tests.  The constant is
    therefore dropped from the comparison unless ``constant=True`` is passed,
    which reproduces Stata's ``hausman ..., constant``: the statistic is the same
    quadratic form over **all** shared parameters.  On the 40x8 panel below,
    Stata reports ``chi2(2) = .9525460961317572`` without the option and
    ``chi2(3) = .952546096131752`` with it (the constant's difference is tiny and
    its variance difference is negative, so the generalised inverse Stata uses
    leaves the statistic essentially unchanged).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from westata.errors import ConsistencyError

__all__ = ["HausmanResult", "LRTestResult", "hausman", "lrtest"]

#: Names the constant can appear under in westata's results.
_CONSTANT_NAMES = frozenset({"Intercept", "_cons", "const"})


@dataclass(frozen=True)
class LRTestResult:
    """Likelihood-ratio test between a restricted and an unrestricted model.

    Attributes
    ----------
    statistic:
        ``LR = 2 (lnL_unrestricted - lnL_restricted)``.
    df:
        ``k_unrestricted - k_restricted``; always positive, because two models
        with equal parameter counts are refused (Stata ``r(498)``).
    pvalue:
        ``P > chi2(df)``; exactly 1.0 for a negative statistic, as Stata prints.
    ll_unrestricted, ll_restricted:
        The log-likelihoods that produced the statistic, kept so the number can
        be audited without re-fitting either model.
    """

    statistic: float
    df: float
    pvalue: float
    ll_unrestricted: float
    ll_restricted: float
    #: Names for the "Assumption: <restricted> nested within <unrestricted>" line.
    #: Empty by default so positional construction keeps working.
    restricted_name: str = ""
    unrestricted_name: str = ""

    def __str__(self) -> str:
        """Reproduce ``lrtest`` exactly::

            Likelihood-ratio test
            Assumption: . nested within big
            LR chi2(1) = 949.95
            Prob > chi2 = 0.0000

        The "Assumption:" line states which model is nested within which and was missing
        entirely -- without it the output gives a statistic with no statement of what was
        compared. The two statistic lines are NOT indented; Stata starts them at column 1.
        """

        restricted = str(self.restricted_name or ".")
        unrestricted = str(self.unrestricted_name or ".")
        return chr(10).join(
            [
                "Likelihood-ratio test",
                f"Assumption: {restricted} nested within {unrestricted}",
                f"LR chi2({self.df:g}) = {self.statistic:.2f}",
                f"Prob > chi2 = {self.pvalue:.4f}",
            ]
        )

    def to_dict(self) -> dict[str, float]:
        """Flatten to the ``r()`` names ``lrtest.sthlp:216-218`` returns.

        ``r(chi2)``, ``r(df)`` and ``r(p)``.  The two log-likelihoods are kept as
        attributes but are NOT ``r()`` scalars -- Stata stores them in ``e()`` of the
        two fitted models, not in the test's own ``r()``.
        """

        return {
            "chi2": float(self.statistic),
            "df": float(self.df),
            "p": float(self.pvalue),
        }


@dataclass(frozen=True)
class HausmanResult:
    """Hausman specification test.

    Attributes
    ----------
    statistic:
        ``(b_c - b_e)' [V_c - V_e]^-1 (b_c - b_e)``.
    df:
        Number of coefficients actually compared (constants excluded).
    pvalue:
        ``P > chi2(df)``.
    excluded:
        Parameters that are not present in both models, so could not be compared.
    """

    statistic: float
    df: float
    pvalue: float
    excluded: list[str] = field(default_factory=list)
    #: Compared coefficients, in Stata's print order. Empty when the caller did not supply the
    #: two estimates, in which case only the summary block is printed.
    names: list[str] = field(default_factory=list)
    consistent: np.ndarray | None = None
    efficient: np.ndarray | None = None
    standard_errors: np.ndarray | None = None
    consistent_label: str = ""
    efficient_label: str = ""
    consistent_command: str = ""
    efficient_command: str = ""
    #: ``chi2(df)`` Stata reports is the RANK of the differenced matrix; when that rank is smaller
    #: than the number of coefficients tested, Stata opens the output with a five-line ``Note:``
    #: naming both numbers. ``None`` keeps ``df`` itself as the tested count.
    tested_count: int | None = None
    #: Stata appends ``(V_b-V_B is not positive definite)`` under the summary when the differenced
    #: covariance has a non-positive eigenvalue, which is also when it needs the generalised inverse.
    not_positive_definite: bool = False

    def __str__(self) -> str:
        """Reproduce ``hausman``'s coefficient table, as Stata prints it.

        Captured from ``hausman iv_model ols_model`` after storing a 2SLS and an OLS fit::

                         ---- Coefficients ----
                     |      (b)          (B)            (b-B)     sqrt(diag(V_b-V_B))
                     |    iv_model    ols_model      Difference       Std. err.
            -------------+----------------------------------------------------------------
                       s |    .1504642      .102643        .0478211        .0101206
                    expr |    .0502254     .0381189        .0121065        .0030987
            ------------------------------------------------------------------------------
                                  b = Consistent under H0 and Ha; obtained from ivregress.
                     B = Inconsistent under Ha, efficient under H0; obtained from regress.
            Test of H0: Difference in coefficients not systematic
                chi2(5) = (b-B)'[(V_b-V_B)^(-1)](b-B)
                        =  22.33
            Prob > chi2 = 0.0005

        Coefficients print with SEVEN significant figures and no leading zero; the standard error
        with seven decimals, or "." when the difference has no estimable variance.
        """

        if self.consistent is None or self.efficient is None or not self.names:
            text = (
                f"Hausman test%" + "\n"
                f"  chi2({self.df:g}) = {self.statistic:.2f}\n"
                f"  Prob > chi2 = {self.pvalue:.4f}"
            )
            if self.excluded:
                text += f"\n  excluded (not in both models): {self.excluded}"
            return text

        def _coefficient(value: float) -> str:
            """SEVEN DECIMAL PLACES, trailing zeros trimmed, leading zero dropped.

            Sixteen coefficients from the reference fit pin this exactly:

                .1504642   .102643   .0478211   .0101206   -.0000559   -.0840797
                .0502254   .0381189  .0121065   .0355587   .0356146    .0119159

            which is "%.7f" with the padding removed -- NOT a significant-figures rule. Two
            earlier attempts got this wrong in opposite directions: "%.6g" turned -.0840797 into
            -.08408, and gave -.0000559 in scientific notation as -5.58788e-05. Significant-digit
            formatting cannot produce both ".1504642" (seven places) and "-.0000559" (three
            significant digits) from values a decade apart.
            """

            # THE DECIMAL COUNT DEPENDS ON THE MAGNITUDE, and the split is at 1.0: values at or
            # above 1 get SIX places, values below get SEVEN. Measured on the same table:
            #
            #     3.218043   3.895172   <- 6 places
            #     .0139284   .1272224   <- 7 places
            #
            # A fixed seven places (the previous rule) printed 3.2180433 where Stata has 3.218043,
            # one character too wide for every coefficient above 1. Eight values spanning both
            # branches pin the rule; "%.6g" would have dropped trailing digits instead.
            magnitude = abs(float(value))
            places = 6 if magnitude >= 1.0 else 7
            text = f"{float(value):.{places}f}"
            if "." in text:
                text = text.rstrip("0").rstrip(".")
            if text.startswith("0."):
                text = text[1:]
            elif text.startswith("-0."):
                text = "-" + text[2:]
            return text or "0"

        consistent = np.asarray(self.consistent, dtype=float)
        efficient = np.asarray(self.efficient, dtype=float)
        differences = consistent - efficient
        errors = (
            np.asarray(self.standard_errors, dtype=float)
            if self.standard_errors is not None
            else np.full(differences.size, float("nan"))
        )

        lines = [
            # A LEADING BLANK LINE: the raw capture begins with newline-then-spaces, so Stata's
            # block opens with an empty line. Round 189 found the same omission in
            # `estat endogenous`; checking the RAW prefix here too is what surfaced it.
            "",
            "                 ---- Coefficients ----",
            "             |      (b)          (B)            (b-B)     sqrt(diag(V_b-V_B))",
            # Header built from measured END columns, not padding arithmetic: the store names end
            # at 23 and 36, "Difference" at 55, "Std. err." at 71 -- widths 11/13/19/16 with the
            # 13-character label block in front of them.
            "             |"
            + " " * (23 - 14 - len(self.consistent_label)) + self.consistent_label
            + " " * (36 - 23 - len(self.efficient_label)) + self.efficient_label
            + " " * (55 - 36 - len("Difference")) + "Difference"
            + " " * (71 - 55 - len("Std. err.")) + "Std. err.",
            "-------------+" + "-" * 64,
        ]
        for index, name in enumerate(self.names):
            # STATA LABELS THE INTERCEPT "_cons", never "Intercept". The compared names come from the
            # result's parameter list, which uses patsy's spelling, so the row printed
            # "   Intercept |" where Stata has "       _cons |" -- a different label AND a different
            # column width because the two words differ by four characters.
            if str(name).strip().lower() in ("intercept", "const", "constant"):
                name = "_cons"
            error = errors[index]
            # The standard error uses the same rule as the coefficients -- seven decimals, no
            # leading zero -- and "." when the difference has no estimable variance.
            if np.isfinite(error):
                # Stata prints ".014854" here, not ".0148540": the standard-error column drops a
                # trailing zero once the value fits. Emitting a fixed seven decimals left the cell
                # one character wider than every other row in the same column.
                error_text = f"{float(error):.7f}".rstrip("0")
                if error_text.endswith("."):
                    error_text += "0"
                if error_text.startswith("0."):
                    error_text = error_text[1:]
                elif error_text.startswith("-0."):
                    error_text = "-" + error_text[2:]
            else:
                error_text = "."
            lines.append(
                # Column widths measured from the Stata block: the three coefficient columns
                # end at 26, 39 and 55, and the standard error ends at 71. "%.7f" values are at
                # most 10 characters wide, so 12/13/16/16 land exactly there -- the earlier
                # 13/13/16/15 pushed the first three numbers one column right.
                f"{name:>12} |{_coefficient(consistent[index]):>12}"
                f"{_coefficient(efficient[index]):>13}"
                f"{_coefficient(differences[index]):>16}"
                f"{error_text:>16}"
            )
        lines.append("-" * 78)
        # Indents measured from Stata: 26 spaces before "b =", 11 before "B =", and a BLANK line
        # between the pair and the test paragraph. The previous version used 22 and 9 and omitted
        # the blank, so both label lines sat one step left of Stata's.
        # Both footnote lines are RIGHT-ALIGNED to a fixed width of 78, not indented by a fixed
        # amount: 'obtained from ivregress.' starts at 22 while 'obtained from xtreg.' starts at
        # 26, and both end at 78. Writing the indent literally matched one model and put the other
        # one step out (round 190).
        lines.append(
            ("b = Consistent under H0 and Ha;"
             f" obtained from {self.consistent_command or 'ivregress'}.").rjust(78)
        )
        lines.append(
            ("B = Inconsistent under Ha, efficient under H0;"
             f" obtained from {self.efficient_command or 'regress'}.").rjust(78)
        )
        lines.append("")
        lines.append("Test of H0: Difference in coefficients not systematic")
        # Another blank line between the null hypothesis and the chi2 formula, measured from the
        # same capture (round 190).
        lines.append("")
        lines.append(f"    chi2({self.df:g}) = (b-B)'[(V_b-V_B)^(-1)](b-B)")
        lines.append(f"            = {self.statistic:>6.2f}")
        lines.append(f"Prob > chi2 = {self.pvalue:.4f}")
        # THE RANK NOTE IS THE FIRST THING STATA PRINTS, before the coefficient table, and only when
        # the differenced matrix's rank is smaller than the number of coefficients tested. Measured
        # on `hausman iv ols, sigmamore` (rank 1 over 6 coefficients):
        #
        #     (blank)
        #     Note: the rank of the differenced variance matrix (1) does not equal the number
        #             of coefficients being tested (6); be sure this is what you expect, or
        #             there may be problems computing the test.  Examine the output of your
        #             estimators for anything unexpected and possibly consider scaling your
        #             variables so that the coefficients are on a similar scale.
        #     (blank)
        #
        # The continuation lines are indented EIGHT spaces; there are TWO spaces after "test.".
        if (
            self.tested_count is not None
            and int(self.tested_count) > int(self.df)
        ):
            lines.insert(
                0,
                chr(10).join(
                    [
                        "",
                        "Note: the rank of the differenced variance matrix "
                        f"({int(self.df)}) does not equal the number",
                        "        of coefficients being tested "
                        f"({int(self.tested_count)}); be sure this is what you expect, or",
                        "        there may be problems computing the test.  Examine the output of your",
                        "        estimators for anything unexpected and possibly consider scaling your",
                        "        variables so that the coefficients are on a similar scale.",
                    ]
                ),
            )
        # ... and the non-positive-definite marker is the LAST line, after the summary.
        if self.not_positive_definite:
            lines.append("(V_b-V_B is not positive definite)")
        return chr(10).join(lines)

    def to_dict(self) -> dict[str, float]:
        """Flatten to the ``r()`` names a Hausman test returns.

        ``hausman.sthlp`` documents ``r(chi2)``, ``r(df)`` and ``r(p)`` -- the same
        shape as ``lrtest``, because both are chi-square tests.  ``excluded`` is a
        diagnostic this package adds, not an ``r()`` scalar, so it stays an attribute.
        """

        return {
            "chi2": float(self.statistic),
            "df": float(self.df),
            "p": float(self.pvalue),
        }


def _loglike(result: Any, *, role: str) -> float:
    """Read an auditable log-likelihood, refusing to invent one."""

    value = getattr(result, "llf", None)
    if value is None or not np.isfinite(value):
        # Several estimators populate e(ll) even where the property is missing.
        e = getattr(result, "e", None)
        if e is not None:
            value = e.get("ll")
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ConsistencyError(
            f"lrtest: the {role} model does not report a log-likelihood, so no "
            "likelihood-ratio test is possible"
        ) from None
    if not np.isfinite(value):
        raise ConsistencyError(
            f"lrtest: the {role} model's log-likelihood is not finite; the model "
            "may not have converged"
        )
    return value


def lrtest(unrestricted: Any, restricted: Any) -> LRTestResult:
    """Likelihood-ratio test that ``restricted`` is adequate (``[R] lrtest``).

    Parameters
    ----------
    unrestricted:
        The model with the larger parameter set.
    restricted:
        The model with fewer parameters.  Stata only checks the parameter
        *counts*; the coefficient names are compared here too, but a mismatch is
        reported as a warning rather than a refusal, because Stata computes the
        statistic anyway (and the LR is then usually negative, which is exactly
        what a non-nested pair deserves).

    Raises
    ------
    ConsistencyError
        The models were fit on different samples, the two parameter counts are
        equal (Stata ``r(498)``: ``df(unrestricted) = df(restricted)``), or a
        log-likelihood is unavailable.

    Warns
    -----
    RuntimeWarning
        ``unrestricted`` has a lower log-likelihood than ``restricted``, which
        cannot happen for a genuinely nested pair.  The statistic is still
        computed (and is negative) rather than silently reported as 0, matching
        Stata's ``LR chi2(1) = -2.6835`` output.
    RuntimeWarning
        ``restricted`` estimates a coefficient ``unrestricted`` does not, so the
        pair is not nested even though the counts differ.  Stata would compute
        the (negative) statistic regardless; so does this, with a warning.
    """

    if int(unrestricted.nobs) != int(restricted.nobs):
        raise ConsistencyError(
            "lrtest: the two models were fit on different samples "
            f"({unrestricted.nobs} vs {restricted.nobs} observations); a "
            "likelihood-ratio test requires the same observations"
        )

    unrestricted_names = list(unrestricted.param_names)
    restricted_names = list(restricted.param_names)
    unrestricted_set = set(unrestricted_names)
    missing = [name for name in restricted_names if name not in unrestricted_set]

    k_unrestricted = len(unrestricted_names)
    k_restricted = len(restricted_names)

    if k_unrestricted == k_restricted:
        # Stata checks this before anything else, even when the regressors
        # differ: "df(unrestricted) = df(restricted) = k" / r(498).
        detail = (
            ""
            if not missing
            else (
                "; the models are also not nested (the restricted model estimates "
                f"parameter(s) the unrestricted model does not: {missing})"
            )
        )
        raise ConsistencyError(
            "lrtest: df(unrestricted) = df(restricted) = "
            f"{k_unrestricted}, so no likelihood-ratio test is defined (Stata "
            f"r(498)){detail}. Compare two models with different parameter "
            "counts, or use test/lincom for hypotheses on one model."
        )

    if k_unrestricted < k_restricted:
        # Stata accepts the arguments in either order and relabels them:
        # ``lrtest L1 B1`` prints "Assumption: L1 nested within B1" and reports
        # 2*(ll_B1 - ll_L1) with df = k_B1 - k_L1 (verified: 2.41124840273977,
        # df 1, p .12046626248080883).  No warning, because this is the
        # documented behaviour of the command, not a misuse.
        swapped = lrtest(restricted, unrestricted)
        return LRTestResult(
            statistic=float(swapped.statistic),
            df=float(swapped.df),
            pvalue=float(swapped.pvalue),
            ll_unrestricted=float(swapped.ll_unrestricted),
            ll_restricted=float(swapped.ll_restricted),
        )

    if missing:
        warnings.warn(
            "lrtest: the models are not nested - the restricted model estimates "
            f"parameter(s) the unrestricted model does not: {missing}. Stata "
            "checks only the parameter counts and reports the resulting "
            "likelihood ratio anyway; it is negative whenever the smaller model "
            "fits better, so treat it as evidence that the pair is not comparable.",
            RuntimeWarning,
            stacklevel=2,
        )

    ll_unrestricted = _loglike(unrestricted, role="unrestricted")
    ll_restricted = _loglike(restricted, role="restricted")

    if ll_unrestricted < ll_restricted:
        warnings.warn(
            "lrtest: the unrestricted model has a lower log-likelihood than the "
            "restricted one, which cannot happen for a genuinely nested pair fit "
            "on the same data; the models may not be comparable. The statistic is "
            "negative, which is reported rather than clamped to zero (Stata does "
            "the same: lrtest on non-nested models prints a negative LR chi2).",
            RuntimeWarning,
            stacklevel=2,
        )

    statistic = 2.0 * (ll_unrestricted - ll_restricted)
    df = float(k_unrestricted - k_restricted)
    # df > 0 here: equal counts raised above and fewer parameters was swapped.
    pvalue = float(stats.chi2.sf(statistic, df))

    # Stata's "Assumption: <restricted> nested within <unrestricted>" line names the two
    # models. Stata uses the name given to ``estimates store`` (or "." for the active
    # estimate); westata has no store mechanism, so the estimator name is the closest
    # equivalent and at least identifies which fit each side is.
    return LRTestResult(
        statistic=float(statistic),
        df=df,
        pvalue=pvalue,
        ll_unrestricted=float(ll_unrestricted),
        ll_restricted=float(ll_restricted),
        restricted_name=str(getattr(restricted, "model_name", "") or ""),
        unrestricted_name=str(getattr(unrestricted, "model_name", "") or ""),
    )


def _command_name(result: Any) -> str:
    """The Stata command that produced ``result``, for the Hausman footnote.

    Stata closes the table with which command each side came from:

        b = Consistent under H0 and Ha; obtained from xtreg.
        B = Inconsistent under Ha, efficient under H0; obtained from xtreg.

    Results expose that as ``stata_command_name``; when they do not, ``model_name`` holds it with
    the options appended ('xtreg, fe'), so the part before the comma is the command. The previous
    code fell back to the empty string, and the renderer then printed 'ivregress' and 'regress'
    for ANY model -- measured on a panel fit in round 174, where every label was wrong (and two of
    them named commands that were never run).
    """

    explicit = str(getattr(result, "stata_command_name", "") or "").strip()
    if explicit:
        return explicit
    model = str(getattr(result, "model_name", "") or "").strip()
    # 'ivregress 2sls' -> 'ivregress': Stata's footnote names the COMMAND, not the estimator.
    # 'xtreg, fe' -> 'xtreg': the option part is not part of the command either.
    return model.split(",")[0].split()[0] if model.split() else ""


def _side_label(result: Any) -> str:
    """The header label for one side of the Hausman table -- Stata's estimate store name.

    Stata heads the columns with whatever the two models were stored as:

        |       fe           re         Difference       Std. err.

    Results carry ``estimation_label`` for that; failing that, the option part of ``model_name``
    ('xtreg, fe' -> 'fe') is what a reader expects and what Stata shows for an unrenamed store.
    The previous code produced an empty label, which left the column header reading
    '|                               Difference' with no names over the two coefficient columns.
    """

    explicit = str(getattr(result, "estimation_label", "") or "").strip()
    if explicit:
        return explicit
    model = str(getattr(result, "model_name", "") or "").strip()
    if not model:
        return ""
    if "," in model:
        # 'xtreg, fe' -> 'fe': Stata heads the column with the estimate store name, and for an
        # unrenamed panel fit that is the option step.
        return model.split(",", 1)[1].strip()
    # An IV fit's model_name is 'ivregress 2sls' with no comma. Stata shows the STORE name the
    # user chose ('iv'); we cannot know that. The ESTIMATOR word ('2sls') is short, honest, and
    # fits the 12-character field -- 'ivregress 2sls' overflowed it and pushed 'Difference' out
    # of its column (round 190).
    return model.split()[-1] if model.split() else model


def _comparable_names(
    consistent: Any, efficient: Any, *, constant: bool = False
) -> tuple[list[str], list[str]]:
    """Return ``(comparable, excluded)`` coefficient names.

    ``comparable`` keeps the consistent model's order and, by default, drops the
    constant (Stata's default).  ``constant=True`` keeps it, reproducing
    ``hausman ..., constant``.  ``excluded`` lists every name that only one of
    the two models estimates.

    The constant is only reported as ``excluded`` when it is genuinely absent
    from one model; when it is merely dropped by the ``constant=False`` default
    it is a documented design choice, not a mismatch.
    """

    # Stata prints the comparison rows in ITS OWN order, which is not patsy's. An IV model is
    # the clearest case: param_names lists the endogenous regressor LAST, while hausman's table
    # leads with it.
    #
    #   param_names  ['Intercept', 'expr', 'tenure', 'rns', 'smsa', 's']
    #   stata_params ['s', 'expr', 'tenure', 'rns', 'smsa', '_cons']
    #
    # ``stata_params`` carries the right ORDER but also Stata's spelling of the constant
    # ("_cons" where patsy says "Intercept"), so the names cannot simply be swapped in: doing
    # that made ``consistent_names.index("_cons")`` raise "'_cons' is not in list" further down.
    # The order is therefore taken from stata_params and mapped BACK to the param_names
    # spelling, which keeps every downstream lookup working.
    def _stata_order(names: list[str], result: Any) -> list[str]:
        ordered = getattr(result, "stata_params", None)
        if ordered is None or len(ordered.index) != len(names):
            return names
        by_lower = {str(name).lower(): name for name in names}
        # The constant is spelled differently on the two lists ("_cons" against "Intercept"),
        # so it is matched through _CONSTANT_NAMES rather than by name.
        # _CONSTANT_NAMES mixes cases ("Intercept" is capitalised, "_cons" is not), so the
        # comparison has to be case-insensitive on BOTH sides -- matching only the candidate to
        # a lowercased set made every constant look absent and silently fell back to param order.
        constants_lower = {name.lower() for name in _CONSTANT_NAMES}
        constant_patsy = next(
            (name for name in names if str(name).lower() in constants_lower), None
        )
        mapped: list[str] = []
        for stata_name in ordered.index:
            key = str(stata_name).lower()
            if key in by_lower:
                mapped.append(by_lower[key])
            elif key in constants_lower and constant_patsy is not None:
                if constant_patsy not in mapped:
                    mapped.append(constant_patsy)
            else:
                return names
        return mapped if sorted(mapped) == sorted(names) else names

    consistent_names = _stata_order(list(consistent.param_names), consistent)
    efficient_names = _stata_order(list(efficient.param_names), efficient)
    efficient_set = set(efficient_names)
    consistent_set = set(consistent_names)

    comparable = [
        name
        for name in consistent_names
        if name in efficient_set and (constant or name not in _CONSTANT_NAMES)
    ]
    excluded: list[str] = []
    for name in consistent_names:
        if name not in efficient_set and name not in _CONSTANT_NAMES:
            excluded.append(name)
    for name in efficient_names:
        if name not in consistent_set and name not in _CONSTANT_NAMES:
            excluded.append(name)
    return comparable, excluded


def _hausman_sigma(result: Any) -> float | None:
    """``s2`` for ``sigmamore``/``sigmaless``: ``e(sigma)`` or, when missing, ``e(rmse)``.

    ``hausman.ado:113-116`` spells the fallback out and it is NOT squared:

        scalar s2_i = e(sigma)
        if s2_i == . scalar s2_i = e(rmse)

    The caller squares the RATIO, so using ``e(rmse)**2`` here squares twice and changes the answer.
    """

    e = getattr(result, "e", None) or {}
    for key in ("sigma", "rmse"):
        value = e.get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number) and number != 0.0:
            return number
    return None


def _stata_generalised_inverse(matrix: np.ndarray) -> tuple[np.ndarray, int]:
    """Moore-Penrose inverse with Stata's own rank tolerance (``hausman.ado:606-628``).

    Stata computes ``tol = max(singular value) * n * 1.0e-9`` and treats every singular value at or
    below it as ZERO, which is what sets ``r(rank)`` and the reported degrees of freedom. NumPy's
    ``pinv`` uses ``max(shape) * eps * max(sv)`` (about 1e-16 relative), so on a matrix whose small
    singular values are 1e-9-ish it keeps directions Stata discards -- the statistic then comes out
    over a wrongly large df.
    """

    values, weights, right = np.linalg.svd(np.asarray(matrix, dtype=float))
    count = weights.size
    empty = np.zeros_like(np.asarray(matrix, dtype=float))
    if count == 0 or not np.isfinite(weights).all():
        return empty, 0
    tolerance = float(weights.max()) * count * 1.0e-9
    keep = weights > tolerance
    if not keep.any():
        return empty, 0
    reciprocal = np.where(keep, 1.0 / np.where(keep, weights, 1.0), 0.0)
    return (right.T * reciprocal) @ values.T, int(keep.sum())


def hausman(
    consistent: Any,
    efficient: Any,
    *,
    constant: bool = False,
    sigmamore: bool = False,
    sigmaless: bool = False,
) -> HausmanResult:
    """Hausman test of the efficient estimator's consistency (``[R] hausman``).

    Parameters
    ----------
    consistent:
        Consistent under both hypotheses but inefficient, e.g. ``xtreg, fe``.
    efficient:
        Efficient under the null but inconsistent under the alternative, e.g.
        ``xtreg, re``.
    sigmamore: bool, default False
        Stata's ``hausman ..., sigmamore``: rescale the CONSISTENT estimator's
        covariance by ``(s2_efficient / s2_consistent) ** 2`` before differencing,
        where ``s2`` is ``e(sigma)`` and falls back to ``e(rmse)`` when that is
        missing. Use it when the two estimators are fit on the same sample but
        the robust VCEs are not directly comparable.
    sigmaless: bool, default False
        The mirror: rescale the EFFICIENT estimator's covariance by
        ``(s2_consistent / s2_efficient) ** 2``.
    constant:
        Include the intercept in the comparison (Stata's ``hausman ...,
        constant``).  The default ``False`` is Stata's default and excludes it,
        because a within-transformed intercept and a GLS intercept estimate
        different population quantities.  Verified on StataNow 19.5 with a
        40-entity x 8-period panel: default ``chi2(2) = .9525460961317572``;
        ``constant`` reports ``chi2(3) = .952546096131752`` over the same
        quadratic form extended by the constant.

    Raises
    ------
    ConsistencyError
        The models were fit on different samples, or share no comparable
        coefficient (with ``constant=True`` the constant alone is not enough:
        a Hausman test of the intercept across parameterisations is undefined).

    Warns
    -----
    RuntimeWarning
        ``V_consistent - V_efficient`` is not positive definite.  Stata then
        inverts with a generalised inverse and still reports, so this does the
        same with ``numpy.linalg.pinv``.

    Notes
    -----
    **Clamping the statistic at 0 is deliberate, and it is an admitted
    difference from Stata.**  When ``V_c - V_e`` has a negative eigenvalue the
    quadratic form can come out negative.  Stata prints that negative number
    (measured on the 40-entity x 8-period panel with the effect correlated with
    ``x1``: ``chi2(2) = -2.35424271822838``, ``r(p) = 1``) and warns that the
    asymptotic assumptions fail.  westata reports ``0`` instead, with the same
    ``p = 1``.  The reason is that a pseudo-inverse routinely produces a
    quadratic form on the order of ``-1e-18`` through pure round-off: keeping
    the sign would turn floating-point noise into a "genuinely negative
    statistic" every time the difference matrix is near-singular, which is a
    worse failure mode than reporting the floor.  ``df``, ``p`` and the warning
    all still match Stata; only the sign of an uninterpretable statistic
    differs.
    """

    comparable, excluded = _comparable_names(consistent, efficient, constant=constant)
    if not comparable:
        raise ConsistencyError(
            "hausman: the two models share no "
            + ("coefficients" if constant else "slope coefficients")
            + ", so they cannot be compared. The constant is "
            + ("included because constant=True was requested" if constant else
               "excluded by design (pass constant=True to include it) because a "
               "within-transformed and a GLS intercept estimate different "
               "parameters")
            + f". consistent={list(consistent.param_names)}, "
            f"efficient={list(efficient.param_names)}"
        )

    if int(consistent.nobs) != int(efficient.nobs):
        raise ConsistencyError(
            "hausman: the two models were fit on different samples "
            f"({consistent.nobs} vs {efficient.nobs} observations)"
        )

    consistent_names = list(consistent.param_names)
    efficient_names = list(efficient.param_names)
    consistent_index = [consistent_names.index(name) for name in comparable]
    efficient_index = [efficient_names.index(name) for name in comparable]

    consistent_params = np.asarray(consistent.params, dtype=float)
    efficient_params = np.asarray(efficient.params, dtype=float)
    consistent_cov = np.asarray(consistent.cov_params(), dtype=float)
    efficient_cov = np.asarray(efficient.cov_params(), dtype=float)

    consistent_slice = consistent_cov[np.ix_(consistent_index, consistent_index)]
    efficient_slice = efficient_cov[np.ix_(efficient_index, efficient_index)]

    # ``sigmamore`` / ``sigmaless`` RESCALE ONE VCE BEFORE DIFFERENCING, they do not invert differently.
    # `hausman.ado:162-173`:
    #
    #     if "sigmamore" {  matrix V1 = ((s2_2/s2_1)^2) * V1  }   // V1 = the CONSISTENT estimator
    #     else           {  matrix V2 = ((s2_1/s2_2)^2) * V2  }   // V2 = the EFFICIENT one
    #
    # and `hausman.ado:113-116` defines s2 as e(sigma) falling back to e(rmse) -- NOT e(rmse)^2.
    # Getting that fallback wrong is not subtle: on grilic it leaves the difference full rank where
    # Stata reports rank 1. With the fallback right, chi2(1) = 3.8437945826 against Stata's
    # 3.8437945825639 and p = .0499304001 against .0499304000722989.
    if sigmamore or sigmaless:
        s2_consistent = _hausman_sigma(consistent)
        s2_efficient = _hausman_sigma(efficient)
        if s2_consistent is None or s2_efficient is None:
            raise ConsistencyError(
                "hausman: sigmamore/sigmaless need e(sigma) or e(rmse) on both results, "
                "which Stata also requires ('Estimators do not save e(sigma) or e(rmse)')"
            )
        if sigmamore:
            consistent_slice = (s2_efficient / s2_consistent) ** 2 * consistent_slice
        else:
            efficient_slice = (s2_consistent / s2_efficient) ** 2 * efficient_slice

    difference = consistent_params[consistent_index] - efficient_params[efficient_index]
    difference_cov = consistent_slice - efficient_slice

    # Stata inverts with a generalised inverse when the difference is not
    # positive definite, and warns. Reproduce that rather than raising: a non-PD
    # difference is common in finite samples and Stata still reports a number.
    symmetric = (difference_cov + difference_cov.T) / 2.0
    try:
        positive_definite = bool(np.all(np.linalg.eigvalsh(symmetric) > 0.0))
    except np.linalg.LinAlgError:
        positive_definite = False

    if positive_definite:
        solved = np.linalg.solve(symmetric, difference)
        rank = len(comparable)
    else:
        warnings.warn(
            "hausman: the difference V_consistent - V_efficient is not positive "
            "definite; using a generalised inverse as Stata does, so the test "
            "statistic may be unreliable",
            RuntimeWarning,
            stacklevel=2,
        )
        inverse, rank = _stata_generalised_inverse(symmetric)
        solved = inverse @ difference

    statistic = float(difference @ solved)
    # A generalised inverse can produce a tiny negative quadratic form; see the
    # Notes section of this function's docstring for why the floor is kept even
    # though Stata would print the negative value.
    statistic = max(statistic, 0.0)
    # STATA'S DF IS THE RANK OF THE GENERALISED INVERSE, not the coefficient count. hausman.ado
    # fills r(df) from the rank MatGinv computed, so a rank-deficient difference reports fewer
    # degrees of freedom -- on grilic, sigmamore gives rank 1 over 7 coefficients, hence chi2(1).
    #
    # Rank 0 means the two estimators are numerically IDENTICAL (comparing a model with itself):
    # the quadratic form is exactly 0, so the honest reading is "no evidence against H0", i.e.
    # p = 1. Stata cannot reach this state because `hausman` refuses an estimator against itself,
    # but this package's callers can, and returning NaN for a perfectly defined zero test would be
    # worse than returning the limit.
    # A rank of 0 means the difference matrix is numerically zero: the two estimators are the same
    # model. The quadratic form is exactly 0, so p = 1 is the honest limit, and the df falls back to
    # the coefficient count because reporting chi2(0) would describe no distribution at all.
    df = float(rank) if rank > 0 else float(len(comparable))
    if rank == 0:
        return HausmanResult(
            statistic=0.0,
            df=df,
            pvalue=1.0,
            excluded=excluded,
            names=list(comparable),
            consistent=consistent_params[consistent_index],
            efficient=efficient_params[efficient_index],
            standard_errors=np.sqrt(np.abs(np.diag(symmetric))),
            consistent_label=_side_label(consistent),
            efficient_label=_side_label(efficient),
            consistent_command=_command_name(consistent),
            efficient_command=_command_name(efficient),
        )

    # The coefficient table Stata prints above the summary. ``comparable`` holds the names in the
    # order both estimates share them, so the rows line up with the printed (b)/(B)/(b-B).
    _errors = np.sqrt(np.abs(np.diag(symmetric)))
    return HausmanResult(
        statistic=statistic,
        df=df,
        pvalue=float(stats.chi2.sf(statistic, df)),
        excluded=excluded,
        names=list(comparable),
        consistent=consistent_params[consistent_index],
        efficient=efficient_params[efficient_index],
        standard_errors=_errors,
        consistent_label=_side_label(consistent),
        efficient_label=_side_label(efficient),
        consistent_command=_command_name(consistent),
        efficient_command=_command_name(efficient),
        tested_count=len(comparable),
        not_positive_definite=not positive_definite,
    )
