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

    def __str__(self) -> str:
        return (
            f"Likelihood-ratio test\n"
            f"  LR chi2({self.df:g}) = {self.statistic:.2f}\n"
            f"  Prob > chi2     = {self.pvalue:.4f}"
        )


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

    def __str__(self) -> str:
        text = (
            f"Hausman test\n"
            f"  chi2({self.df:g}) = {self.statistic:.2f}\n"
            f"  Prob > chi2 = {self.pvalue:.4f}"
        )
        if self.excluded:
            text += f"\n  excluded (not in both models): {self.excluded}"
        return text


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

    return LRTestResult(
        statistic=float(statistic),
        df=df,
        pvalue=pvalue,
        ll_unrestricted=float(ll_unrestricted),
        ll_restricted=float(ll_restricted),
    )


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

    consistent_names = list(consistent.param_names)
    efficient_names = list(efficient.param_names)
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


def hausman(
    consistent: Any, efficient: Any, *, constant: bool = False
) -> HausmanResult:
    """Hausman test of the efficient estimator's consistency (``[R] hausman``).

    Parameters
    ----------
    consistent:
        Consistent under both hypotheses but inefficient, e.g. ``xtreg, fe``.
    efficient:
        Efficient under the null but inconsistent under the alternative, e.g.
        ``xtreg, re``.
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

    difference = consistent_params[consistent_index] - efficient_params[efficient_index]
    difference_cov = (
        consistent_cov[np.ix_(consistent_index, consistent_index)]
        - efficient_cov[np.ix_(efficient_index, efficient_index)]
    )

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
    else:
        warnings.warn(
            "hausman: the difference V_consistent - V_efficient is not positive "
            "definite; using a generalised inverse as Stata does, so the test "
            "statistic may be unreliable",
            RuntimeWarning,
            stacklevel=2,
        )
        solved = np.linalg.pinv(symmetric) @ difference

    statistic = float(difference @ solved)
    # A generalised inverse can produce a tiny negative quadratic form; see the
    # Notes section of this function's docstring for why the floor is kept even
    # though Stata would print the negative value.
    statistic = max(statistic, 0.0)
    df = float(len(comparable))

    return HausmanResult(
        statistic=statistic,
        df=df,
        pvalue=float(stats.chi2.sf(statistic, df)),
        excluded=excluded,
    )
