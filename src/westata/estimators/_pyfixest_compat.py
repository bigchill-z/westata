"""Read PyFixest fitted-model state through ONE version-tolerant door.

Why this module exists
----------------------
``pyproject.toml`` used to say ``pyfixest>=0.30`` with no upper bound, so CI
installed whatever was newest -- 0.70.0 -- while the development machine kept
0.60.0.  PyFixest 0.70 deleted the 0.60 private attributes and replaced them
with typed components, so the SAME source tree passed locally and failed 138
tests in CI with::

    AttributeError: `fit._collin_vars` was a private attribute in pyfixest 0.60
    and is not supported in 0.70.0

That asymmetry is the whole reason CI stayed red for nine days: no local run
could ever see the breakage.

The mapping (verified against pyfixest 0.60.0 and 0.70.0 side by side)
----------------------------------------------------------------------
=================================  ==========================================================
0.60 private attribute             0.70 typed component
=================================  ==========================================================
``fit._collin_vars``               ``fit.collinearity.dropped_coef_names``
``fit._df_t``                      ``fit.variance_covariance.df_t``
``fit._df_k``                      ``fit.variance_covariance.df_k``
``fit._k_fe``                      ``fit.fixef_counts.n_levels_by_fe`` (indexed by effect)
``fit._na_index``                  ``fit.sample_info.dropped_row_index``
``fit._vcov``                      ``fit.variance_covariance.vcov``
``fit._fml``                       ``fit.model.formula``
=================================  ==========================================================

Design rule: every helper tries the modern component FIRST and falls back to the
0.60 private name.  Reading the new API first means a future pyfixest that drops
the compatibility aliases keeps working; the fallback keeps 0.60 supported for
the pinned floor.

Type normalisation is the subtle part.  ``_k_fe`` was a ``pandas.Series``
indexed by ``__fixed_effect__(<var>)``; ``fixef_counts.n_levels_by_fe`` is a
plain tuple of counts.  ``_collin_vars`` was a list/Series; the replacement is a
tuple.  In both cases the two shapes differ enough that returning one or the
other unchanged would break the caller, so each accessor normalises to ONE
documented shape.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "absorbed_regressor_names",
    "dropped_row_index",
    "estimation_frame",
    "fixed_effect_level_counts",
    "parameter_count",
    "residual_df",
    "variance_covariance",
    "within_design",
]


def _component(model: Any, name: str) -> Any | None:
    """One typed component, or ``None`` when this PyFixest does not publish it."""

    return getattr(model, name, None)


def absorbed_regressor_names(model: Any) -> list[str]:
    """Names of regressors dropped for collinearity with the absorbed effects.

    0.60: ``fit._collin_vars`` (``[]`` or a Series/list of names).
    0.70: ``fit.collinearity.dropped_coef_names`` (a tuple).

    Always returns a list of ``str``; an empty list means nothing was dropped.
    """

    collinearity = _component(model, "collinearity")
    if collinearity is not None:
        names = getattr(collinearity, "dropped_coef_names", None)
        if names is not None:
            return [str(name) for name in names]

    legacy = getattr(model, "_collin_vars", None)
    if legacy is None:
        return []
    return [str(name) for name in legacy]


def variance_covariance(model: Any) -> np.ndarray:
    """The (small-sample-adjusted) coefficient covariance matrix."""

    component = _component(model, "variance_covariance")
    if component is not None:
        covariance = getattr(component, "vcov", None)
        if covariance is not None:
            return np.asarray(covariance, dtype=float)
    return np.asarray(model._vcov, dtype=float)


def residual_df(model: Any) -> float:
    """``df_t``: the residual degrees of freedom the reference distribution uses.

    Under clustering PyFixest reports ``G - 1`` here, which is why westata keeps it
    separate from ``nobs - rank``; both quantities are needed and they differ.
    """

    component = _component(model, "variance_covariance")
    if component is not None:
        value = getattr(component, "df_t", None)
        if value is not None:
            return float(value)
    return float(model._df_t)


def parameter_count(model: Any) -> float | None:
    """``df_k``: the counted parameter total, or ``None`` when unavailable."""

    component = _component(model, "variance_covariance")
    if component is not None:
        value = getattr(component, "df_k", None)
        if value is not None:
            return float(value)
    legacy = getattr(model, "_df_k", None)
    return None if legacy is None else float(legacy)


def fixed_effect_level_counts(model: Any) -> dict[str, int] | None:
    """Absorbed fixed-effect level counts, keyed by effect name.

    Normalises two very different 0.60/0.70 shapes into one dict:

    * 0.60 ``_k_fe`` -- a ``Series`` indexed ``__fixed_effect__(firm)``.  The
      ``__fixed_effect__(...)`` wrapper is stripped so callers can look a column
      up directly.
    * 0.70 ``fixef_counts.n_levels_by_fe`` -- a ``tuple`` of counts with no
      names, paired positionally with ``model.fixef`` (a ``+``-joined string).

    Returns ``None`` when neither shape is available, so callers keep their
    previous "information not recorded" behaviour instead of inventing numbers.
    """

    counts = _component(model, "fixef_counts")
    if counts is not None:
        levels = getattr(counts, "n_levels_by_fe", None)
        if levels is not None:
            # `model.fixef` names the absorbed effects in the same order, e.g.
            # "firm" or "firm+year". An unnamed count cannot be attributed to a
            # column, and guessing would be worse than saying "unknown".
            effects = getattr(getattr(model, "model", None), "fixef", None)
            names = [part for part in str(effects or "").split("+") if part]
            values = [int(value) for value in levels]
            if len(names) == len(values):
                return dict(zip(names, values))
            return None

    legacy = getattr(model, "_k_fe", None)
    if legacy is None:
        return None
    try:
        items = list(legacy.items()) if hasattr(legacy, "items") else list(enumerate(legacy))
    except (AttributeError, TypeError, ValueError):
        return None
    normalised: dict[str, int] = {}
    for raw_name, raw_count in items:
        name = str(raw_name)
        if name.startswith("__fixed_effect__(") and name.endswith(")"):
            name = name[len("__fixed_effect__(") : -1]
        normalised[name] = int(raw_count)
    return normalised


def dropped_row_index(model: Any) -> Any | None:
    """Positions of the rows PyFixest dropped (missing values and singletons).

    This is the authoritative description of the estimation sample: re-deriving
    it by filtering on "level has more than one row" loses rows, because
    singleton removal interacts across absorbed dimensions and is applied
    iteratively.
    """

    sample_info = _component(model, "sample_info")
    if sample_info is not None:
        index = getattr(sample_info, "dropped_row_index", None)
        if index is not None:
            return index
    return getattr(model, "_na_index", None)


def estimation_frame(model: Any) -> pd.DataFrame:
    """PyFixest's own frame for the estimation sample (``fit._data`` in both)."""

    frame = getattr(model, "_data", None)
    return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()


def within_design(model: Any) -> np.ndarray:
    """The within-transformed (fixed effects removed) design matrix.

    0.60: ``fit._X`` -- an ``ndarray`` of the demeaned regressors.
    0.70: ``fit.within_data.design``.

    Callers use this to PROVE the fixed effects were absorbed (one column per
    reported coefficient), so it must be the demeaned design and not the raw
    model matrix; substituting the latter would make that check vacuous.
    """

    within = getattr(model, "within_data", None)
    if within is not None:
        design = getattr(within, "design", None)
        if design is not None:
            return np.asarray(design, dtype=float)
    return np.asarray(model._X, dtype=float)
