"""Propensity-score matching: Stata's ``teffects psmatch`` (ATET).

``teffects`` is builtin with no readable ado source, so the estimator was reconstructed from
what Stata's ``generate(m)`` option exposes -- and then checked digit for digit against its
printed ATET.

Reference, from a live ``teffects psmatch (re78) (treat <8 covariates>), atet`` on
nsw_dw.dta (n = 445)::

    Treatment-effects estimation                   Number of obs      =        445
    Estimator      : propensity-score matching     Matches: requested =          1
    Outcome model  : matching                                     min =          1
    Treatment model: logit                                        max =          8
    ------------------------------------------------------------------------------
                 |              AI robust
            re78 | Coefficient  std. err.      z    P>|z|     [95% conf. interval]
    -------------+----------------------------------------------------------------
    ATET         |
           treat |
       (1 vs 0)  |   2639.865   714.7758     3.69   0.000      1238.93    4040.799

and with ``nn(4)``: ATET 2330.108, SE 680.0515.

THE DETAIL THAT DECIDES BOTH ROWS: distances are measured on the estimated PROPENSITY SCORE
(``teffects_psmatch.sthlp:253``), and ALL controls tied at the minimum distance are used,
not just one. Stata's ``generate(m)`` created EIGHT variables m1..m8 for a requested
``nneighbor(1)`` -- the help says "The number of variables generated may be more than
nneighbor() because of tied distances". Using only the single nearest control gave 2569.745;
taking every tied control reproduces 2639.865 exactly. The header's "max = 8" is that count.

That is why the report's C15-01/C15-02 recorded a 13.7% gap: the earlier implementation
picked exactly one (or exactly four) neighbours and ignored ties.

Verified from Stata's own exported propensity scores AND from westata's Logit predictions
(differ by 2.75e-8 in the fifth observation), with tie tolerances from 1e-8 to 1e-6 all
giving the same two ATETs. The logit coefficients themselves agree with Stata to 1e-7, so
none of this is a model discrepancy.

NOT reproduced: the Abadie-Imbens AI-robust standard error requires a local-linear variance
estimator, not a closed form. The point estimate is exact, so ``se``/``z``/``p``/the interval
are NaN rather than a plausible guess -- the same call made for rdrobust and teffects_ra.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from westata.errors import ConsistencyError

__all__ = ["PsmatchResult", "teffects_psmatch"]


@dataclass(frozen=True)
class PsmatchResult:
    """``teffects psmatch``: ATET from propensity-score nearest-neighbour matching."""

    ate: float
    nobs: int
    treated: int
    controls: int
    nneighbor: int
    matches_min: int
    matches_max: int
    outcome: str
    caliper: float = float("nan")
    treatment_model: str = "logit"
    se: float = float("nan")
    z: float = float("nan")
    pvalue: float = float("nan")
    ci_lower: float = float("nan")
    ci_upper: float = float("nan")
    variance_matches_stata: bool = False

    @property
    def statistic(self) -> float:
        """Alias Stata uses for the ATET point estimate."""

        return self.ate

    def to_dict(self) -> dict[str, float]:
        return {
            "ate": float(self.ate),
            "N": float(self.nobs),
            "N1": float(self.treated),
            "N0": float(self.controls),
        }

    def __str__(self) -> str:
        """Stata's full table, with the AI-robust columns left blank.

        Measured from the live run above: every table line is 78 characters, the right-hand
        header block starts at column 51, and the ATET row's value fields end at fixed
        columns. The standard-error columns are printed as ``.`` because the AI-robust
        variance is not reproduced -- printing a nearly-right number there would hide a
        genuine gap.
        """

        # Geometry measured from the live table: every line is 78 characters, the right-hand
        # labels start at column 47 (Number of obs / Matches: requested) or 62 (min / max),
        # "=" sits at 66 and the value is right-aligned in the final ten columns.
        def _header(left: str, label: str, value: int, label_column: int) -> str:
            # "=" sits at column 66; the label is left-aligned in the space between the
            # left text and that column. "Number of obs" gets a 19-wide field, "min"/"max"
            # a 4-wide one, and the value is right-aligned in the final ten columns.
            label_width = 66 - label_column
            return f"{left:<{label_column}}{label:<{label_width}}= {value:>10}"

        ci_lower = f"{self.ci_lower:.4g}" if np.isfinite(self.ci_lower) else "."
        ci_upper = f"{self.ci_upper:.4g}" if np.isfinite(self.ci_upper) else "."
        se_text = f"{self.se:.4g}" if np.isfinite(self.se) else "."
        z_text = f"{self.z:.2f}" if np.isfinite(self.z) else "."
        p_text = f"{self.pvalue:.3f}" if np.isfinite(self.pvalue) else "."

        # Column endings measured from Stata's ATET row: estimate 25, SE 36, z 45, p 53,
        # lower bound 66, upper bound 78. Each field is built so its LAST character lands
        # on that column -- deriving them from relative widths kept accumulating a few
        # extra spaces (an 81-character row against Stata's 78).
        def _place(text: str, pieces: list[str], end_column: int) -> None:
            used = sum(len(piece) for piece in pieces)
            pieces.append(" " * max(end_column - used - len(text), 0) + text)

        row_pieces = ["   (1 vs 0)  |"]
        _place(f"{self.ate:.3f}", row_pieces, 25)
        _place(se_text, row_pieces, 36)
        _place(z_text, row_pieces, 45)
        _place(p_text, row_pieces, 53)
        _place(ci_lower, row_pieces, 66)
        _place(ci_upper, row_pieces, 78)
        row = "".join(row_pieces)

        lines = [
            "",
            _header("Treatment-effects estimation", "Number of obs", self.nobs, 47),
            _header(
                "Estimator      : propensity-score matching",
                "Matches: requested",
                self.nneighbor,
                47,
            ),
            _header("Outcome model  : matching", "min", self.matches_min, 62),
            _header(
                f"Treatment model: {self.treatment_model}",
                "max",
                self.matches_max,
                62,
            ),
            "-" * 78,
            f"{'':<13}|{'AI robust':>23}",
            f"{self.outcome:>12} | Coefficient  std. err.      z    P>|z|"
            f"     [95% conf. interval]",
            "-" * 13 + "+" + "-" * 64,
            "ATET         |",
            "       treat |",
            row,
            "-" * 78,
        ]
        return "\n".join(lines)


def teffects_psmatch(
    response: Any,
    treatment: Any,
    covariates: Any,
    *,
    nneighbor: int = 1,
    tie_tolerance: float = 1e-6,
    outcome: str = "",
    caliper: float | None = None,
) -> PsmatchResult:
    """ATET by propensity-score matching, as ``teffects psmatch ..., atet``.

    ``covariates`` is an ``(n, k)`` array. The propensity score comes from a logit of
    ``treatment`` on those covariates with an intercept, which is what Stata's treatment
    model does by default (verified: the logit coefficients agree to 1e-7).

    ``nneighbor`` is Stata's ``nn()``; ties at the k-th smallest distance are ALL included,
    so the realised number of matches can exceed it. ``tie_tolerance`` is the absolute
    propensity-score difference still considered a tie; 1e-8 through 1e-6 were measured to
    give identical results on the reference sample, so the default is deliberately on the
    generous side of the logit's own convergence precision.

    ``caliper`` is Stata's ``caliper(#)``: a control further than # in propensity-score
    distance is ineligible, and a treated unit with nothing inside the caliper is dropped
    from the estimate (Stata reports those as "no matches"). ``None`` means no cap, which is
    the default.

    The AI-robust standard error is NOT computed (see the module docstring), so the
    inferential fields stay NaN.
    """

    from westata.estimators.discrete import Logit

    y = np.asarray(response, dtype=float).reshape(-1)
    d = np.asarray(treatment, dtype=float).reshape(-1)
    x = np.asarray(covariates, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if not (y.size == d.size == x.shape[0]):
        raise ConsistencyError(
            f"teffects psmatch: shapes disagree -- outcome {y.size}, "
            f"treatment {d.size}, covariates {x.shape[0]}"
        )
    keep = np.isfinite(y) & np.isfinite(d) & np.all(np.isfinite(x), axis=1)
    y, d, x = y[keep], d[keep], x[keep]
    if y.size < 3:
        raise ValueError(
            f"teffects psmatch: at least 3 complete observations are needed; got {y.size}"
        )
    levels = np.unique(d)
    if levels.size != 2 or not np.array_equal(levels, np.array([0.0, 1.0])):
        raise ConsistencyError(
            f"teffects psmatch: treatment must be coded 0/1; found {levels.tolist()}"
        )
    nneighbor = int(nneighbor)
    if nneighbor < 1:
        raise ValueError(f"teffects psmatch: nneighbor must be >= 1; got {nneighbor}")
    tie_tolerance = float(tie_tolerance)
    if tie_tolerance < 0:
        raise ValueError("teffects psmatch: tie_tolerance must be non-negative")

    design = np.column_stack([np.ones(x.shape[0]), x])
    fit = Logit(d, design).fit()
    score = np.asarray(fit.predict(), dtype=float)

    controls = np.flatnonzero(d == 0.0)
    treated = np.flatnonzero(d == 1.0)
    if not controls.size or not treated.size:
        raise ConsistencyError(
            "teffects psmatch: both treatment levels must have at least one observation"
        )

    differences: list[float] = []
    match_counts: list[int] = []
    for index in treated:
        distance = np.abs(score[index] - score[controls])
        if caliper is not None:
            # Beyond the caliper a control is ineligible, so it must not be chosen even as
            # the nearest one. Marking it infinite rather than filtering keeps the index
            # correspondence with ``controls``.
            distance = np.where(distance <= caliper, distance, np.inf)
            if not np.isfinite(distance).any():
                continue
        order = np.argsort(distance)
        # The k-th smallest distance sets the threshold; every control at or below it
        # (within the tie tolerance) is a match. This is the rule Stata's m1..m8 encode.
        finite = int(np.isfinite(distance).sum())
        kth = min(nneighbor, finite)
        threshold = distance[order[kth - 1]]
        if not np.isfinite(threshold):
            continue
        selected = controls[distance <= threshold + tie_tolerance]
        # ``distance`` was capped, so a selected control is inside the caliper by
        # construction; the explicit mask guards the tie_tolerance widening above.
        selected = selected[
            np.abs(score[index] - score[selected]) <= (caliper if caliper is not None else np.inf)
            + tie_tolerance
        ]
        if selected.size == 0:  # pragma: no cover - threshold always selects at least one
            raise ConsistencyError(
                f"teffects psmatch: no control matched treated observation {index}"
            )
        differences.append(float(y[index] - np.mean(y[selected])))
        match_counts.append(int(selected.size))

    ate = float(np.mean(differences))
    return PsmatchResult(
        ate=ate,
        nobs=int(y.size),
        treated=int(treated.size),
        controls=int(controls.size),
        nneighbor=nneighbor,
        matches_min=int(min(match_counts)),
        matches_max=int(max(match_counts)),
        outcome=str(outcome or ""),
        caliper=float("nan") if caliper is None else caliper,
    )
