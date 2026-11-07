"""Stata's ``tebalance summarize``: covariate balance after a matching fit.

Reference from a live ``teffects psmatch (re78) (treat <8 covariates>), atet`` followed by
``tebalance summarize`` on nsw_dw.dta::

    Covariate balance summary
                             Raw      Matched
    -----------------------------------------
    Number of obs =          445          370
    Treated obs   =          185          185
    Control obs   =          260          185
    -----------------------------------------
    -----------------------------------------------------------------
                    |Standardized differences          Variance ratio
                    |        Raw     Matched           Raw    Matched
    ----------------+------------------------------------------------
                age |   .1072771   -.0030807      1.027755   1.081843
          education |   .1412198    .1041542      1.551284   1.213356
              black |   .0438866   -.0769979      .9250286   1.170807
           hispanic |  -.1745611    .0476934      .5828804   1.208333
            married |   .0936407    .0278399      1.180212   1.046651
           nodegree |  -.3039864   -.1223973      1.499755   1.140232
               re74 |  -.0021599   -.0172992      .7380953   .7524242
               re75 |   .0838632   -.1507322      1.076344   .5813152
    -----------------------------------------------------------------

THE FORMULAS, from ``_tebalance_summarize.ado`` (readable, 489 lines):

    standardized difference = (m1 - m0) / sqrt((v1 + v0) / 2)
    variance ratio          = v1 / v0

where m0/v0 are the CONTROL mean and variance, m1/v1 the treated ones, and the matched
columns use the matching weights on the control side.

VERIFIED: every RAW value matches to all printed digits (8/8), and every MATCHED value
matches for 7 of the 8 covariates. ``age`` is the exception -- this module reproduces
-.0000000 against Stata's -.0030807 for the difference and 1.078052 against 1.081843 for the
variance ratio. The cause is NOT the formulas, which are pinned by the other seven: matching
on the propensity score makes the weighted control age mean exactly equal the treated mean
here (difference 0.0 to sixteen digits), while Stata's differs by 0.0219 on an unmatched gap
of 0.762. Stata must be distributing the tie weights slightly differently for this
integer-valued covariate. Reported alongside the matching values rather than hidden.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["BalanceRow", "BalanceSummary", "tebalance_summarize"]


def _no_leading_zero(value: float, decimals: int, *, truncate: bool = False) -> str:
    """Format with fixed decimals but strip the leading zero, as Stata prints these cells.

    Measured from ``tebalance summarize``: ".1072771", "-.1745611", "1.027755" -- the sign is
    kept and the zero before the point is dropped, matching the %9.0g convention used
    elsewhere in this package.
    """

    if not np.isfinite(value):
        return "."
    # SEVEN SIGNIFICANT FIGURES, not fixed decimals, and the rounding must be EXPLICIT: Python's
    # "g" presentation keeps as many digits as the float carries once they fit inside the request,
    # so f"{v:.7g}" returned "-.003080708" (9 digits) where Stata prints "-.0030807". Stata also
    # shows ".1072771" (7 decimals) beside "1.027755" (6 decimals) in the same column, which no
    # fixed-decimal rule can produce.
    from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP, localcontext

    number = Decimal(repr(float(value)))
    if number == 0:
        return "0"
    # This is Stata's field: the whole rendered cell is at most ``decimals`` CHARACTERS, counting
    # a leading minus and the decimal point. Measured on `tebalance summarize`:
    #     .1072771   (8 chars, 7 digits)      -.0030807  (9 chars, 7 digits after the point)
    #     -.1745611  (9 chars)                .0476934   (8 chars)
    # Note -.0030807 keeps SEVEN digits after the point, not five: the leading zero is dropped, so
    # the digit count is driven by the characters available, which is why counting "significant
    # figures" gave the wrong answer for small negative values.
    adjusted = number.adjusted()
    # The minus sign does NOT eat into the digit budget: '-.1745611' is ten characters while
    # '.1072771' is eight, and both carry seven digits after the point.
    budget = decimals
    if adjusted >= 0:
        places = max(budget - (adjusted + 1) - 1, 0)
    else:
        places = max(budget - 2, 0)
    quantum = Decimal(1).scaleb(-places)
    with localcontext() as context:
        context.prec = 40
        rounded = number.quantize(
            quantum, rounding=ROUND_DOWN if truncate else ROUND_HALF_UP
        )
    text = format(rounded, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text.startswith("0."):
        return text[1:]
    if text.startswith("-0."):
        return "-" + text[2:]
    return text


@dataclass(frozen=True)
class BalanceRow:
    """One covariate's line in ``tebalance summarize``."""

    name: str
    raw_std_diff: float
    matched_std_diff: float
    raw_variance_ratio: float
    matched_variance_ratio: float


@dataclass(frozen=True)
class BalanceSummary:
    """The whole ``tebalance summarize`` output."""

    rows: list[BalanceRow]
    nobs_raw: int
    nobs_matched: int
    treated_raw: int
    treated_matched: int
    control_raw: int
    control_matched: int
    exact_covariates: tuple[str, ...] = ()
    """Covariates whose MATCHED columns reproduce Stata to the printed digits."""

    def __str__(self) -> str:
        """Stata's ``tebalance summarize`` output, captured byte for byte.

        Columns measured from a live run after ``teffects psmatch ..., atet`` on nsw_dw.dta:
        the name field is 15 wide with the pipe at index 16, then the four statistics END at
        28, 40, 54 and 65 -- so every line is 65 characters. An earlier version used 16/11/12/12/12
        and printed 66-character rows with each statistic one column off.
        """

        def _field(
            value: float, places: int, end: int, current: int, *, truncate: bool = False
        ) -> str:
            text = _no_leading_zero(value, places, truncate=truncate)
            return " " * max(end - current - len(text), 0) + text

        lines = [
            "Covariate balance summary",
            f"{'':>25}{'Raw':>10}{'Matched':>8}",
            "-" * 41,
            f"Number of obs = {self.nobs_raw:>12}{self.nobs_matched:>11}",
            f"Treated obs   = {self.treated_raw:>12}{self.treated_matched:>11}",
            f"Control obs   = {self.control_raw:>12}{self.control_matched:>11}",
            "-" * 41,
            "",
            "-" * 65,
            f"{'':<16}|{'Standardized differences':>26}{'Variance ratio':>18}",
            f"{'':<16}|{'Raw':>9}{'Matched':>13}{'Raw':>11}{'Matched':>11}",
            "-" * 16 + "+" + "-" * 48,
        ]
        for row in self.rows:
            line = f"{row.name:>15} |"
            line += _field(row.raw_std_diff, 9, 28, len(line))
            line += _field(row.matched_std_diff, 9, 40, len(line))
            line += _field(row.raw_variance_ratio, 9, 54, len(line), truncate=True)
            line += _field(row.matched_variance_ratio, 9, 65, len(line), truncate=True)
            lines.append(line)
        lines.append("-" * 65)
        return "\n".join(lines)


def _weighted_mean_variance(
    values: np.ndarray, weights: np.ndarray | None
) -> tuple[float, float]:
    """Mean and variance with optional frequency weights (Stata's ``[iw=]``)."""

    values = np.asarray(values, dtype=float)
    if weights is None:
        if values.size < 2:
            return float(values.mean()) if values.size else float("nan"), float("nan")
        return float(values.mean()), float(values.var(ddof=1))
    weights = np.asarray(weights, dtype=float)
    total = float(weights.sum())
    if total <= 0.0 or values.size < 2:
        return float("nan"), float("nan")
    mean = float((weights * values).sum() / total)
    if total <= 1.0:
        return mean, float("nan")
    variance = float((weights * (values - mean) ** 2).sum() / (total - 1.0))
    return mean, variance


def tebalance_summarize(
    treatment: Any,
    covariates: Any,
    *,
    match_weights: Any = None,
    names: list[str] | None = None,
    exact: tuple[str, ...] = (),
) -> BalanceSummary:
    """Balance table comparing raw and matched means/variances.

    ``covariates`` is an ``(n, k)`` array; ``match_weights`` holds the control-side matching
    weights (one per observation, zero for untreated controls that were never used, and zero
    for all treated rows). ``names`` labels the columns.
    """

    status = np.asarray(treatment, dtype=float).reshape(-1)
    values = np.asarray(covariates, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.shape[0] != status.size:
        raise ValueError(
            f"tebalance summarize: {status.size} treatment values against "
            f"{values.shape[0]} covariate rows"
        )
    levels = np.unique(status)
    if levels.size != 2 or not np.array_equal(levels, np.array([0.0, 1.0])):
        raise ValueError(
            f"tebalance summarize: treatment must be coded 0/1; found {levels.tolist()}"
        )
    if names is None:
        names = [f"x{index + 1}" for index in range(values.shape[1])]
    if len(names) != values.shape[1]:
        raise ValueError(
            f"tebalance summarize: {len(names)} names for {values.shape[1]} covariates"
        )

    use = (status == 0.0) & ~np.isnan(values).any(axis=1)
    treated = (status == 1.0) & ~np.isnan(values).any(axis=1)
    treated_mask = np.where(treated, 1.0, 0.0)
    control_mask = np.where(use, 1.0, 0.0)

    weights = None
    if match_weights is not None:
        weights = np.asarray(match_weights, dtype=float).reshape(-1) * control_mask

    rows: list[BalanceRow] = []
    for column, name in enumerate(names):
        series = values[:, column]
        m0, v0 = _weighted_mean_variance(series[use], None)
        m1, v1 = _weighted_mean_variance(series[treated], None)
        raw_scale = math.sqrt((v1 + v0) / 2.0) if v0 + v1 > 0.0 else float("nan")
        raw_diff = (m1 - m0) / raw_scale if raw_scale else float("nan")
        raw_ratio = v1 / v0 if v0 > 0.0 else float("nan")

        if weights is not None:
            m0w, v0w = _weighted_mean_variance(series[use], weights[use])
        else:
            m0w, v0w = m0, v0
        matched_scale = math.sqrt((v1 + v0w) / 2.0) if v0w + v1 > 0.0 else float("nan")
        matched_diff = (m1 - m0w) / matched_scale if matched_scale else float("nan")
        matched_ratio = v1 / v0w if v0w > 0.0 else float("nan")

        rows.append(
            BalanceRow(
                name=str(name),
                raw_std_diff=raw_diff,
                matched_std_diff=matched_diff,
                raw_variance_ratio=raw_ratio,
                matched_variance_ratio=matched_ratio,
            )
        )

    control_matched = int(round(float(weights.sum()))) if weights is not None else int(
        control_mask.sum()
    )
    return BalanceSummary(
        rows=rows,
        nobs_raw=int(treated_mask.sum() + control_mask.sum()),
        nobs_matched=int(treated_mask.sum() + control_matched),
        treated_raw=int(treated_mask.sum()),
        treated_matched=int(treated_mask.sum()),
        control_raw=int(control_mask.sum()),
        control_matched=control_matched,
        exact_covariates=tuple(exact),
    )
