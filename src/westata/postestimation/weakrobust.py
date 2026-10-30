"""``estat weakrobust`` -- inference that does not depend on instrument strength.

``ivregress_postestimation.sthlp:364-375``:

    estat weakrobust performs hypothesis tests on the coefficients of endogenous
    regressors that are robust to weak instruments.  For just-identified models, the
    Anderson-Rubin (1949) test is reported.  For overidentified models, if a
    homoskedastic VCE was used in estimation, the conditional likelihood-ratio (CLR)
    test of Moreira (2003) is reported.  If a heteroskedastic or cluster-robust VCE was
    used instead, an appropriate generalization of the CLR test is used (see Finlay and
    Magnusson [2009]).  When there is only one endogenous regressor, the confidence
    intervals associated with any of these tests can be requested.

Only the Anderson-Rubin test is implemented here.  Its statistic is a plain restricted
F test, fully determined by the specification and verifiable against an independent
regression; the CLR test needs Moreira's conditional likelihood ratio, whose
distribution is obtained by simulation over a nuisance-parameter grid, and the Stata
implementation is a compiled builtin with no ``ado/base`` source to read.  Refusing CLR
with that reason beats returning a number that looks like a p-value but comes from a
different test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


@dataclass
class WeakRobustResult:
    """One weak-instrument-robust test.

    Attributes
    ----------
    test:
        ``"Anderson-Rubin"``.
    statistic:
        The chi-square form of the statistic.
    f_statistic:
        The F form, which Stata prints alongside it.
    df, df_denom:
        Numerator (number of instruments tested) and denominator degrees of freedom.
    pvalue:
        Upper-tail probability of the F form.
    """

    test: str
    statistic: float
    f_statistic: float
    df: float
    df_denom: float
    pvalue: float

    def __str__(self) -> str:
        return (
            f"\nWeak-instrument-robust inference\n"
            f"{self.test} chi2({self.df:g}) = {self.statistic:9.4f}\n"
            f"{'':>{len(self.test)}} F({self.df:g},{self.df_denom:g}) = "
            f"{self.f_statistic:9.4f}   Prob > F = {self.pvalue:.4f}"
        )

    def to_dict(self) -> dict[str, float]:
        """Flatten to the ``r()`` names Stata returns for this test.

        ``ivregress_postestimation.sthlp`` documents ``r(chi2)``, ``r(F)``, ``r(df)``,
        ``r(df_r)`` and ``r(p)`` for the weak-robust tests.
        """

        return {
            "chi2": float(self.statistic),
            "F": float(self.f_statistic),
            "df": float(self.df),
            "df_r": float(self.df_denom),
            "p": float(self.pvalue),
        }


def _require_iv(result: Any) -> dict[str, Any]:
    structure = getattr(result, "iv_structure", None)
    if structure is None:
        raise NotImplementedError(
            "estat weakrobust is only defined after an instrumental-variables fit: "
            "Stata reaches it through ivregress_estat, and it needs the split between "
            "included exogenous regressors, endogenous regressors and excluded "
            f"instruments; got {getattr(result, 'model_name', 'an unnamed model')!r} "
            "without that structure"
        )
    if structure["endog"].shape[1] == 0:
        raise ValueError("estat weakrobust: model contains no endogenous variables")
    return structure


def _is_constant_column(column: np.ndarray) -> bool:
    return column.size > 0 and float(np.ptp(column)) == 0.0


def _resolve_null_hypothesis(
    structure: dict[str, Any], betanull: Any
) -> np.ndarray:
    """Turn ``betanull`` into one value per endogenous regressor.

    ``sthlp:457-485`` accepts a numlist, a list of equalities, or a matrix.  Two forms
    are supported here, and the mapping between them is the one Stata's ``copy``
    suboption marks:

    * a MAPPING -- coefficients named explicitly, unspecified ones zero (Stata's
      ``varname=#`` list, where "null values are set to zero for endogenous regressors
      you did not specify");
    * a SEQUENCE -- assigned by position (Stata's ``numlist ..., copy``).

    A sequence of the wrong length is an error rather than being padded or truncated:
    either would test a hypothesis the caller did not write.
    """

    endog_names = [str(name) for name in structure["endog_names"]]
    n_endog = len(endog_names)
    if betanull is None:
        return np.zeros(n_endog, dtype=float)

    if isinstance(betanull, Mapping):
        values = np.zeros(n_endog, dtype=float)
        unknown = [name for name in betanull if name not in endog_names]
        if unknown:
            raise ValueError(
                "estat weakrobust: betanull names "
                f"{unknown} which is not an endogenous regressor; the endogenous "
                f"regressors are {endog_names}"
            )
        for name, value in betanull.items():
            values[endog_names.index(str(name))] = float(value)
        return values

    if isinstance(betanull, (str, bytes)):
        raise TypeError(
            "estat weakrobust: betanull takes a mapping or a sequence of values, not a "
            f"string; got {betanull!r}"
        )

    sequence = np.asarray(betanull, dtype=float).reshape(-1)
    if sequence.size != n_endog:
        raise ValueError(
            "estat weakrobust: betanull gave "
            f"{sequence.size} value(s) for {n_endog} endogenous regressor(s) "
            f"({endog_names}); pass one value per regressor, or a mapping to name them"
        )
    return sequence


def _anderson_rubin(
    structure: dict[str, Any], null_values: np.ndarray
) -> WeakRobustResult:
    """The Anderson-Rubin test of ``H0: beta_endog = null_values``.

    The recipe is the one in Anderson and Rubin (1949): subtract the hypothesised
    contribution of the endogenous regressors from the response, then test whether the
    EXCLUDED INSTRUMENTS have any explanatory power for what is left, once the included
    exogenous regressors are accounted for.  Under the null that test has an exact F
    distribution, and its distribution does not depend on how strong the instruments
    are -- which is the whole point of the command.

    Written as a restricted-versus-unrestricted F so it can be checked against two plain
    OLS fits:

        F = ((RSS_restricted - RSS_unrestricted) / q) / (RSS_unrestricted / (n - k))
    """

    response = np.asarray(structure["response"], dtype=float).reshape(-1)
    exog = np.asarray(structure["exog"], dtype=float)
    endog = np.asarray(structure["endog"], dtype=float)
    instruments = np.asarray(structure["instruments"], dtype=float)
    n = response.size

    if null_values.size != endog.shape[1]:
        raise ValueError(
            "estat weakrobust: the null hypothesis has "
            f"{null_values.size} value(s) for {endog.shape[1]} endogenous regressor(s)"
        )

    # Response with the hypothesised endogenous contribution removed.
    adjusted = response - endog @ null_values

    # Included exogenous regressors, keeping at most one constant (``_regress`` supplies
    # its own, and duplicating it would make the design singular).
    keep: list[np.ndarray] = []
    seen_constant = False
    for j in range(exog.shape[1]):
        column = exog[:, j]
        if _is_constant_column(column):
            if seen_constant:
                continue
            seen_constant = True
        keep.append(column)
    blocks = list(keep)
    if not seen_constant:
        blocks.insert(0, np.ones(n))
    restricted_design = np.column_stack(blocks) if blocks else np.ones((n, 1))
    unrestricted_design = np.column_stack([restricted_design, instruments])

    restricted_coef, *_ = np.linalg.lstsq(restricted_design, adjusted, rcond=None)
    unrestricted_coef, *_ = np.linalg.lstsq(unrestricted_design, adjusted, rcond=None)
    restricted_resid = adjusted - restricted_design @ restricted_coef
    unrestricted_resid = adjusted - unrestricted_design @ unrestricted_coef
    rss_restricted = float(restricted_resid @ restricted_resid)
    rss_unrestricted = float(unrestricted_resid @ unrestricted_resid)

    q = float(instruments.shape[1])
    df_denom = float(n - unrestricted_design.shape[1])
    if rss_unrestricted <= 0.0 or df_denom <= 0.0:
        raise ValueError(
            "estat weakrobust: the Anderson-Rubin statistic is undefined here -- the "
            "unrestricted regression fits exactly or has no residual degrees of freedom"
        )

    f_statistic = ((rss_restricted - rss_unrestricted) / q) / (
        rss_unrestricted / df_denom
    )
    # The chi-square form is ``q`` times the F, which is how Stata reports both.
    chi2 = q * f_statistic
    pvalue = float(scipy_stats.f.sf(f_statistic, q, df_denom))
    return WeakRobustResult(
        test="Anderson-Rubin",
        statistic=float(chi2),
        f_statistic=float(f_statistic),
        df=q,
        df_denom=df_denom,
        pvalue=pvalue,
    )


def estat_weakrobust(
    result: Any,
    *,
    ar: bool | None = None,
    clr: bool | None = None,
    betanull: Any = None,
    level: float = 95.0,
) -> WeakRobustResult:
    """Stata's ``estat weakrobust``.

    Parameters
    ----------
    ar:
        Force the Anderson-Rubin test.  It is also the default for a just-identified
        model (``sthlp:365-367``).
    clr:
        Request the conditional likelihood-ratio test.  Accepted so the option is
        recognised, then refused with the reason -- see the module docstring.
    betanull:
        The null hypothesis on the endogenous coefficients
        (``sthlp:457-485``).  Three accepted forms, matching Stata:

        * a mapping ``{name: value}`` -- coefficients named explicitly, others zero;
        * a sequence of numbers -- assigned BY POSITION, which is what Stata's
          ``copy`` suboption means;
        * ``None`` -- every coefficient is zero, the default.

        A sequence longer or shorter than the number of endogenous regressors is
        refused, because silently padding or truncating would test a different
        hypothesis than the caller wrote.
    level:
        Confidence level, in percent.  Relevant only to the ``ci`` option, which is not
        implemented (it needs the inverted test, not the statistic).

    Notes
    -----
    ``sthlp:365-373`` makes the DEFAULT depend on identification: Anderson-Rubin when
    just identified, CLR when overidentified.  Since only AR is implemented, an
    overidentified model without ``ar`` refuses rather than silently reporting a
    different test than Stata would.
    """

    structure = _require_iv(result)
    n_instruments = int(np.asarray(structure["instruments"]).shape[1])
    n_endog = int(np.asarray(structure["endog"]).shape[1])
    just_identified = n_instruments == n_endog

    if clr and not ar:
        raise NotImplementedError(
            "estat weakrobust, clr is not implemented: the conditional likelihood-ratio "
            "test of Moreira (2003) needs its conditional distribution, obtained by "
            "simulating over a nuisance-parameter grid (sthlp:511-520 mentions the "
            "reps() option for exactly this).  Stata's implementation is a compiled "
            "builtin with no ado/base source.  ar() IS available and needs no simulation."
        )

    wants_ar = bool(ar) if ar is not None else just_identified
    if not wants_ar:
        raise NotImplementedError(
            "estat weakrobust reports the CLR test by default for an overidentified "
            "model (sthlp:367-373), and that test is not implemented.  Pass ar=True to "
            "request the Anderson-Rubin test, which is available and needs no simulation."
        )

    null_values = _resolve_null_hypothesis(structure, betanull)
    return _anderson_rubin(structure, null_values)
