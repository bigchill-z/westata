"""``estat mundlak`` -- the Mundlak specification test after ``xtreg``.

``xtreg_postestimation.sthlp:366-370``:

    estat mundlak performs a Mundlak specification test, a test that nu is uncorrelated
    with the regressors in x.  estat mundlak is for use after xtreg with the re, cre, and
    fe options but not with the absorb() option or the vce(hc2) option.

The test is the one Mundlak (1978) proposed: add the WITHIN-GROUP MEANS of the regressors
to the random-effects equation.  Those means are what carries any correlation between the
unobserved effect ``nu`` and ``x``, so testing their coefficients jointly to zero is
exactly a test of that correlation.  A rejection means the random-effects estimator is
inconsistent and the fixed-effects one is preferred -- which is why it is a specification
test rather than a diagnostic.

The coefficient and its standard error come from re-estimating the augmented equation,
not from the original fit, so the test is not a rescaling of the Hausman statistic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


@dataclass
class MundlakResult:
    """The Mundlak specification test.

    Attributes
    ----------
    statistic:
        The joint Wald statistic on the group-mean coefficients.
    df:
        Number of regressors tested (one per slope).
    pvalue:
        ``P > chi2(df)``.
    tested:
        Names of the group-mean terms.
    """

    statistic: float
    df: float
    pvalue: float
    tested: list[str]

    def __str__(self) -> str:
        names = ", ".join(self.tested)
        return (
            "\nMundlak specification test\n"
            f"H0: {names} jointly zero (nu uncorrelated with x)\n\n"
            f"  chi2({self.df:g}) = {self.statistic:9.4f}\n"
            f"  Prob > chi2 = {self.pvalue:9.4f}"
        )

    def to_dict(self) -> dict[str, float]:
        """Flatten to the ``r()`` names Stata returns for this test.

        ``xtreg_postestimation.sthlp`` documents ``r(chi2)``, ``r(df)`` and ``r(p)``.
        """

        return {
            "chi2": float(self.statistic),
            "df": float(self.df),
            "p": float(self.pvalue),
        }


def _require_panel(result: Any) -> tuple[Any, np.ndarray, list[str], str]:
    """The frame, the entity labels and the slope names of an ``xtreg`` fit.

    ``model_name`` is Stata's own spelling -- ``"xtreg, re"`` / ``"xtreg, fe"`` -- and
    ``panel_entity`` holds the GROUP LABELS as a coded array, not the name of a column.
    Both facts matter: keying off a column name put the entity vector nowhere and the
    first version refused every panel fit.
    """

    model_name = str(getattr(result, "model_name", "")).strip().lower()
    if not model_name.startswith("xtreg"):
        raise NotImplementedError(
            "estat mundlak is defined only after xtreg: Stata dispatches it through "
            "xtreg_estat.ado, which rejects non-panel results; got "
            f"{getattr(result, 'model_name', 'an unnamed model')!r}"
        )
    if "re" not in model_name and "cre" not in model_name and "fe" not in model_name:
        raise ValueError(
            "estat mundlak is for use after xtreg with the re, cre or fe options "
            f"(xtreg_postestimation.sthlp:369-370); this fit is {model_name!r}"
        )

    frame = getattr(result, "model_data", None)
    if frame is None:
        raise ValueError(
            "estat mundlak needs the estimation-sample data, which this result did not "
            "retain; refit with the formula interface"
        )

    entity_labels = getattr(result, "panel_entity", None)
    if entity_labels is None:
        raise ValueError(
            "estat mundlak needs the panel identifier; this result records no "
            "panel_entity"
        )
    labels = np.asarray(entity_labels)
    if labels.size != len(frame):
        raise ValueError(
            "estat mundlak: the panel identifier has "
            f"{labels.size} entries for {len(frame)} rows"
        )

    names = [
        str(name)
        for name in getattr(result, "param_names", [])
        if str(name) not in ("Intercept", "_cons", "const")
    ]
    return frame, labels, names, model_name


def _joint_wald(result: Any, terms: Sequence[str]) -> tuple[float, float, float]:
    """Wald statistic for ``terms`` jointly zero, from the fitted result."""

    names = list(result.param_names)
    positions = [names.index(term) for term in terms]
    params = np.asarray(result.params, dtype=float)[positions]
    covariance = np.asarray(result.cov_params(), dtype=float)
    block = covariance[np.ix_(positions, positions)]
    statistic = float(params @ np.linalg.solve(block, params))
    df = float(len(terms))
    return statistic, df, float(scipy_stats.chi2.sf(statistic, df))


def estat_mundlak(
    result: Any,
    *,
    storer: Any = None,
    entity: str | None = None,
    time: str | None = None,
) -> MundlakResult:
    """Stata's ``estat mundlak``.

    Parameters
    ----------
    result:
        A fit from :class:`westata.RandomEffects` (or any panel estimator that recorded
        ``panel_entity``).
    storer:
        Callback used to fit the augmented equation.  It receives the augmented frame
        and returns a fitted result.  Highest priority when given.
    entity, time:
        The panel identifier and time name for the AUGMENTED refit.  Required when
        ``storer`` is omitted: the fitted result keeps ``panel_entity`` (the coded group
        labels) but not the column names the estimator's ``from_formula`` needs, so the
        test cannot rebuild the equation without them.

    Raises
    ------
    NotImplementedError
        For a non-panel result.
    ValueError
        When the estimation data or the panel identifier is unavailable, when the model
        has no slopes to test, or when ``entity``/``time`` are needed but missing.

    Notes
    -----
    ``sthlp:369-370`` rules out ``absorb()`` and ``vce(hc2)``: the first has no
    group-mean interpretation and the second has no likelihood to re-estimate.
    """

    frame, labels, slopes, model_name = _require_panel(result)
    if not slopes:
        raise ValueError(
            "estat mundlak: the model has no regressors besides the constant, so there "
            "is no correlation between nu and x to test"
        )

    augmented = frame.copy()
    terms: list[str] = []
    for name in slopes:
        if name not in augmented.columns:
            raise ValueError(
                f"estat mundlak: {name!r} is not a column of the estimation data"
            )
        values = pd.Series(np.asarray(augmented[name], dtype=float), index=augmented.index)
        mean_name = f"__mundlak_{name}"
        # Group means over the CODED entity labels, which is what ``panel_entity``
        # carries -- there is no column to group by.
        augmented[mean_name] = values.groupby(labels).transform("mean").to_numpy()
        terms.append(mean_name)

    if storer is None:
        from westata.estimators.panel import RandomEffects

        if entity is None:
            raise ValueError(
                "estat mundlak: the fitted result keeps the coded group labels but not "
                "the panel COLUMN names, which the augmented refit needs.  Pass "
                "entity= (and time= if the original fit used it), or storer=."
            )
        response = str(getattr(result, "response_name", "") or "")
        if not response:
            raise ValueError(
                "estat mundlak: the result records no response name; pass storer="
            )
        formula = f"{response} ~ " + " + ".join(list(slopes) + terms)
        refit = RandomEffects.from_formula(
            formula, augmented, entity=entity, time=time
        ).fit()
    else:
        refit = storer(augmented)

    statistic, df, pvalue = _joint_wald(refit, terms)
    return MundlakResult(
        statistic=statistic, df=df, pvalue=pvalue, tested=list(slopes)
    )
