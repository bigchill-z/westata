"""Compare a captured Stata estimation with a westata result, loudly.

Why this module exists
----------------------
A replication harness is only worth its output if a mismatch cannot be absorbed
by accident.  The failure modes that matter are all quiet ones:

* **Positional matching.**  Stata stores ``e(b)`` in *equation order* and
  includes factor base levels as ``0b.g`` with a coefficient of exactly zero,
  while patsy reorders design columns alphabetically and drops the base level.
  Comparing ``stata_b[i]`` with ``westata_b[i]`` across a factor model compares
  different parameters and reports a nonsense difference.  Every comparison
  here is made on canonical *names*; a name that cannot be matched is either a
  hard error or an explicit, reported exclusion.
* **Length disagreement.**  If one side has five parameters and the other has
  six, any elementwise difference is meaningless.  :func:`compare_capture`
  raises instead of truncating, because a silently truncated comparison looks
  like a pass.
* **Missing values.**  Stata's missing is ``8.988e307``, so an unhandled
  missing produces an enormous "relative difference" that is then blamed on
  the estimator.  Both sides are converted to ``NaN`` before differencing.
* **Degenerate scales.**  Dividing by a coefficient of ``1e-14`` turns float
  noise into a "material difference".  The denominator is floored, and the
  absolute difference is reported alongside so a reader can see which scale
  the verdict came from.

The canonical vocabulary is shared with ``retest/lib/stata_probe.py`` (via
:func:`align`) so the harness and the existing sweep cannot drift apart.  That
module is imported, never modified.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "AlignmentError",
    "Record",
    "Row",
    "VERDICTS",
    "align_parameters",
    "compare_capture",
    "difference_row",
    "rel_diff",
    "tier",
    "verdict_for",
]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RETEST_LIB = _REPO_ROOT / "retest" / "lib"
if str(_RETEST_LIB) not in sys.path:
    sys.path.insert(0, str(_RETEST_LIB))

from stata_probe import align as _probe_align  # noqa: E402
from stata_probe import canon_stata, canon_westata  # noqa: E402
from stata_probe import is_stata_base  # noqa: E402
from stata_probe import tier_for  # noqa: E402

#: Verdict -> maximum relative difference tolerated, in ascending order.
VERDICTS: tuple[tuple[str, float], ...] = (
    ("IDENTICAL", 1e-9),
    ("NUMERICALLY_EQUAL", 1e-6),
    ("MINOR_DIFF", 1e-3),
)

#: Denominator floor.  Coefficient magnitudes in published regressions start
#: around 1e-3 and go to 1e2; a denominator below this floor means the two sides
#: are both "zero" and their difference should be judged absolutely.
_SCALE_FLOOR = 1e-12


def rel_diff(actual: float | Sequence[float], expected: float | Sequence[float]) -> float:
    """Max relative difference with a floored denominator.

    Returns ``inf`` when the two vectors have different lengths: an
    incomparable pair is the worst outcome, never a small number.
    """
    a = np.atleast_1d(np.asarray(actual, dtype=float)).reshape(-1)
    e = np.atleast_1d(np.asarray(expected, dtype=float)).reshape(-1)
    if a.size != e.size:
        return float("inf")
    if a.size == 0:
        return 0.0
    scale = np.maximum(np.abs(e), _SCALE_FLOOR)
    with np.errstate(invalid="ignore"):
        return float(np.max(np.abs(a - e) / scale))


def verdict_for(value: float) -> str:
    """Map a maximum relative difference onto a replication verdict."""
    if value != value:  # NaN: nothing comparable, which is not a pass
        return "NOT_COMPARABLE"
    if math.isinf(value):
        return "MATERIAL_DIFF"
    for label, tolerance in VERDICTS:
        if value <= tolerance:
            return label
    return "MATERIAL_DIFF"


@dataclass
class Row:
    """One parameter, on both scales, plus both standard errors."""

    name: str
    stata_b: float
    westata_b: float
    b_rel: float
    stata_se: float
    westata_se: float
    se_rel: float
    stata_name: str = ""
    westata_name: str = ""

    @property
    def b_abs(self) -> float:
        """Absolute coefficient difference, ``NaN`` when either side is missing."""
        if not _finite(self.stata_b, self.westata_b):
            return float("nan")
        return abs(self.stata_b - self.westata_b)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "stata_b": self.stata_b,
            "westata_b": self.westata_b,
            "b_rel": self.b_rel,
            "b_abs": self.b_abs,
            "stata_se": self.stata_se,
            "westata_se": self.westata_se,
            "se_rel": self.se_rel,
            "stata_name": self.stata_name,
            "westata_name": self.westata_name,
        }


@dataclass
class Record:
    """The outcome of comparing one specification.

    ``verdict`` speaks only about coefficients and standard errors -- the
    quantities both sides are supposed to reproduce bit-for-bit.  Fit
    statistics that legitimately differ by convention (``ll`` under
    ``aweight``, for instance) are reported in ``scalars`` with their own
    relative differences and never move the verdict; ``notes`` says why.
    """

    name: str
    rows: list[Row]
    max_b_rel: float
    max_se_rel: float
    stata_n: int
    westata_n: int
    verdict: str
    notes: list[str] = field(default_factory=list)
    scalars: dict[str, dict[str, float]] = field(default_factory=dict)
    stata_command: str = ""
    dropped: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def n_params(self) -> int:
        return len(self.rows)

    @property
    def n_agree(self) -> bool:
        return self.stata_n == self.westata_n

    def row(self, name: str) -> Row:
        for item in self.rows:
            if item.name == name:
                return item
        raise KeyError(f"{name!r} is not in record {self.name!r}")

    def scalar_rel(self, key: str) -> float:
        entry = self.scalars.get(key)
        return float("nan") if entry is None else entry["rel"]

    def scalar_side(self, key: str, side: str) -> float:
        """One side of a compared ``e()`` scalar (``"stata"`` or ``"westata"``)."""
        if side not in ("stata", "westata"):
            raise ValueError(f"side must be 'stata' or 'westata', got {side!r}")
        entry = self.scalars.get(key)
        if entry is None:
            return float("nan")
        return float(entry[side])

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "verdict": self.verdict,
            "max_b_rel": self.max_b_rel,
            "max_se_rel": self.max_se_rel,
            "stata_n": self.stata_n,
            "westata_n": self.westata_n,
            "n_params": self.n_params,
            "stata_command": self.stata_command,
            "notes": list(self.notes),
            "dropped": list(self.dropped),
            "rows": [row.to_dict() for row in self.rows],
            "scalars": self.scalars,
            "extra": self.extra,
        }


class AlignmentError(ValueError):
    """The two parameter vectors cannot be paired one-to-one.

    Raised instead of comparing by position.  ``unmatched_stata`` and
    ``unmatched_westata`` name both sides so the caller can decide whether the
    fix is a rename, a restricted formula, or a genuine missing estimator.
    """

    def __init__(self, message: str, *, unmatched_stata: Sequence[str] = (),
                 unmatched_westata: Sequence[str] = ()) -> None:
        self.unmatched_stata = list(unmatched_stata)
        self.unmatched_westata = list(unmatched_westata)
        super().__init__(message)


def _is_base_level(name: str) -> bool:
    """True for Stata's base (``0b.g``) and omitted (``3o.g``) factor levels."""
    return is_stata_base(name) or "__omitted__" in name


def align_parameters(
    stata_names: Sequence[str],
    westata_names: Sequence[str],
    *,
    mapping: Mapping[str, str] | None = None,
    allow_unmatched_stata: Sequence[str] = (),
) -> tuple[list[str], list[int], list[int]]:
    """Pair coefficients by canonical name; raise unless the pairing is total.

    ``mapping`` maps a *Stata* name onto a *westata* name when the two sides
    genuinely label the same parameter differently (``var(e.y)`` against
    ``/sigma``).  It is applied before canonicalisation and is never inferred.

    Dropped from Stata's side: factor base levels (``0b.g``) and omitted levels
    (``3o.g``, whose coefficient is zero).  Those are not parameters westata
    estimates at all, so excluding them is not a silent leniency.  Everything
    else must match; otherwise :class:`AlignmentError` is raised, which is the
    guard that stops a positional comparison from ever happening.

    The returned order is **Stata's**, i.e. the order the coefficients appear in
    the regression's own output, with ``_cons`` last.  That is the order a
    publication table is read in, and it is the order patsy does *not* preserve,
    so fixing it here means the report and the paper can be read side by side.

    ``allow_unmatched_stata`` names Stata coefficients the caller knows westata
    does not estimate (an ``absorb()`` set, say).  They are dropped between the
    two sides but stay excluded explicitly -- the caller had to write the name,
    and :attr:`Record.dropped` reports it.  Westata-only parameters are never
    allowed: an unexpected parameter on the *tested* side means the model
    compared is not the model intended, which is the failure this module exists
    to catch.
    """
    rename = dict(mapping or {})
    stata_names = list(stata_names)
    westata_names = list(westata_names)

    kept_stata = [(i, name) for i, name in enumerate(stata_names) if not _is_base_level(name)]
    kept_index = [i for i, _ in kept_stata]
    kept_names = [name for _, name in kept_stata]

    names, si, wi, unmatched = _probe_align(kept_names, westata_names, rename=rename)
    unmatched_westata = [item for item in unmatched if not item.startswith("stata-only:")]
    unmatched_stata = [
        item.split(":", 1)[1] for item in unmatched if item.startswith("stata-only:")
    ]

    excluded = [name for name in stata_names if _is_base_level(name)]
    declared = set(allow_unmatched_stata)
    unexpected = [name for name in unmatched_stata if name not in declared]
    if unexpected or unmatched_westata:
        details = []
        if unexpected:
            details.append("only in Stata: " + ", ".join(unexpected))
        if unmatched_westata:
            details.append("only in westata: " + ", ".join(unmatched_westata))
        raise AlignmentError(
            "cannot align the two parameter vectors ("
            + "; ".join(details)
            + "). Excluding factor base levels "
            + (", ".join(excluded) if excluded else "(none)")
            + f" leaves {len(kept_names)} Stata parameters against "
            f"{len(westata_names)} westata parameters. Pass mapping= to rename a "
            "parameter the two sides label differently, or allow_unmatched_stata= "
            "to declare a coefficient westata does not estimate.",
            unmatched_stata=unexpected,
            unmatched_westata=list(unmatched_westata),
        )
    if len(names) != len(kept_names) - len(declared & set(unmatched_stata)) or (
        len(names) != len(westata_names)
    ):
        raise AlignmentError(
            f"aligned {len(names)} parameters but Stata has {len(kept_names)} "
            f"({len(declared & set(unmatched_stata))} declared as not estimated by "
            f"westata) and westata has {len(westata_names)}; refusing to compare a "
            "partial pairing",
            unmatched_stata=unmatched_stata,
            unmatched_westata=unmatched_westata,
        )
    # ``align`` returns indices into the *filtered* Stata vector, ordered by
    # westata; map them back to the caller's original ordering so the record can
    # quote raw names, then re-sort by the Stata index so the record reads in
    # regression-output order.
    pairs = sorted(
        zip(names, (kept_index[i] for i in si), wi),
        key=lambda item: item[1],
    )
    return [item[0] for item in pairs], [item[1] for item in pairs], [item[2] for item in pairs]


def _result_arrays(result: Any) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Pull names/params/bse off a statsmodels-like result, validating shape."""
    for attribute in ("param_names", "params", "bse"):
        if not hasattr(result, attribute):
            raise TypeError(
                f"westata result has no {attribute!r}; compare_capture needs a "
                "statsmodels-like object exposing param_names/params/bse"
            )
    names = [str(name) for name in result.param_names]
    b = np.asarray(result.params, dtype=float).reshape(-1)
    se = np.asarray(result.bse, dtype=float).reshape(-1)
    if not len(names) == b.size == se.size:
        raise ValueError(
            f"westata result is internally inconsistent: {len(names)} names, "
            f"{b.size} params, {se.size} standard errors"
        )
    return names, b, se


def compare_capture(
    name: str,
    stata_cap: Any,
    westata_result: Any,
    *,
    mapping: Mapping[str, str] | None = None,
    scalars: Sequence[str] = (),
    allow_unmatched_stata: Sequence[str] = (),
) -> Record:
    """Compare one captured Stata estimation against a westata result.

    ``stata_cap`` is a :class:`replication.lib.stata_session.Capture` (or any
    object with ``b``/``V``/``names``/``scalars``/``nobs``).  ``westata_result``
    is anything statsmodels-shaped.

    ``scalars`` names ``e()`` values to report alongside the coefficients.  They
    are compared but never folded into the verdict, because several of them
    differ by documented convention rather than by error; the record's
    ``notes`` lists each one that does.

    ``allow_unmatched_stata`` declares coefficients westata genuinely does not
    estimate, so they are excluded explicitly rather than by position.  Leave it
    empty whenever the two sides are meant to estimate the same model.

    Raises
    ------
    AlignmentError
        The parameter vectors cannot be paired one-to-one after excluding
        factor base levels.  This is the loud failure that replaces a
        positional comparison.
    """
    stata_names = [str(item) for item in stata_cap.names]
    w_names, w_b, w_se = _result_arrays(westata_result)

    names, si, wi = align_parameters(
        stata_names,
        w_names,
        mapping=mapping,
        allow_unmatched_stata=allow_unmatched_stata,
    )
    declared = set(allow_unmatched_stata)
    dropped = [
        item
        for item in stata_names
        if _is_base_level(item) or item in declared
    ]

    s_b_all = np.asarray(stata_cap.b, dtype=float).reshape(-1)
    if s_b_all.size != len(stata_names):
        raise AlignmentError(
            f"Stata capture is internally inconsistent: {s_b_all.size} values "
            f"but {len(stata_names)} names"
        )
    s_se_all = np.asarray(stata_cap.se, dtype=float).reshape(-1)
    if s_se_all.size != s_b_all.size:
        raise AlignmentError(
            f"Stata capture has {s_b_all.size} coefficients but {s_se_all.size} "
            "standard errors"
        )

    rows: list[Row] = []
    for key, stata_index, westata_index in zip(names, si, wi):
        s_b = float(s_b_all[stata_index])
        w_b_i = float(w_b[westata_index])
        s_se = float(s_se_all[stata_index])
        w_se_i = float(w_se[westata_index])
        rows.append(
            Row(
                name=key,
                stata_b=s_b,
                westata_b=w_b_i,
                b_rel=_pair_rel(w_b_i, s_b),
                stata_se=s_se,
                westata_se=w_se_i,
                se_rel=_pair_rel(w_se_i, s_se),
                stata_name=stata_names[stata_index],
                westata_name=w_names[westata_index],
            )
        )

    max_b_rel = _max_rel([row.b_rel for row in rows])
    max_se_rel = _max_rel([row.se_rel for row in rows])

    stata_n = int(getattr(stata_cap, "nobs", 0) or 0)
    westata_n = _westata_nobs(westata_result)

    record = Record(
        name=name,
        rows=rows,
        max_b_rel=max_b_rel,
        max_se_rel=max_se_rel,
        stata_n=stata_n,
        westata_n=westata_n,
        verdict=verdict_for(max_b_rel),
        scalars={},
        stata_command=str(getattr(stata_cap, "command", "")),
        dropped=dropped,
    )

    for key in scalars:
        record.scalars[key] = _compare_scalar(key, stata_cap, westata_result)

    if stata_n != westata_n:
        record.notes.append(
            f"sample size differs: Stata e(N)={stata_n}, westata={westata_n}. "
            "A coefficient match on different samples is a coincidence, not a "
            "replication."
        )
    if dropped:
        record.notes.append(
            "excluded from the comparison (base levels are zero by construction; "
            "declared coefficients westata does not estimate): " + ", ".join(dropped)
        )
    if all(math.isnan(row.se_rel) for row in rows):
        record.notes.append("no e(V) in the capture: standard errors were not compared")
    for key, entry in record.scalars.items():
        rel = entry["rel"]
        if rel == rel and rel > 1e-6:
            record.notes.append(
                f"{key} differs by convention or by error: stata={entry['stata']:.10g}, "
                f"westata={entry['westata']:.10g} (rel {rel:.3e}); "
                "not part of the verdict"
            )
    return record


def _finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def _max_rel(values: Sequence[float]) -> float:
    """Maximum of a vector of relative differences, poisoned by any ``NaN``.

    Python's builtin ``max`` compares with ``>``, and every comparison against
    ``NaN`` is false, so ``max([0.0, nan])`` silently returns ``0.0``.  For this
    harness that is the worst possible bug: one missing coefficient would drop
    out of the summary and the specification would be graded on the remaining
    ones.  A single non-finite entry therefore makes the whole maximum ``NaN``,
    which the verdict reports as ``NOT_COMPARABLE`` while the per-row data still
    shows which coefficient was at fault.
    """
    values = list(values)
    if not values:
        return 0.0
    if any(value != value for value in values):
        return float("nan")
    if any(math.isinf(value) for value in values):
        return float("inf")
    return max(values)


def _pair_rel(westata_value: float, stata_value: float) -> float:
    """Relative difference of one pair, floored, ``NaN`` when not comparable.

    ``NaN`` rather than a number when either side is missing: a missing value is
    not a small difference, and letting ``NaN`` propagate keeps it out of the
    "max" in the same way Stata keeps missing out of a mean.
    """
    if not _finite(westata_value, stata_value):
        return float("nan")
    return abs(westata_value - stata_value) / max(abs(stata_value), _SCALE_FLOOR)


def _westata_nobs(result: Any) -> int:
    """Read the estimation sample size off a westata result.

    ``nobs`` is the primary attribute, but some result objects only carry it in
    their ``e()`` mapping; both are checked so a convention difference never
    masquerades as a sample difference.
    """
    for attribute in ("nobs", "nobs_total"):
        value = getattr(result, attribute, None)
        if value is None:
            continue
        try:
            return int(round(float(value)))
        except (TypeError, ValueError):
            continue
    mapping = getattr(result, "e", None)
    if mapping is not None:
        try:
            if "N" in mapping:
                return int(round(float(mapping["N"])))
        except (TypeError, ValueError):
            pass
    raise TypeError(
        "cannot read the sample size from the westata result: it has no numeric "
        "nobs attribute and no e()['N']; the N comparison would be a guess"
    )


def _compare_scalar(key: str, stata_cap: Any, result: Any) -> dict[str, float]:
    """Pull one ``e()`` scalar from both sides and difference it."""
    stata_value = None
    getter = getattr(stata_cap, "get", None)
    if callable(getter):
        stata_value = getter(key)
    if stata_value is None:
        scalars = getattr(stata_cap, "scalars", {}) or {}
        stata_value = scalars.get(key)
    westata_value = _westata_scalar(key, result)
    s_val = float(stata_value) if stata_value is not None else float("nan")
    w_val = float(westata_value) if westata_value is not None else float("nan")
    return {
        "stata": s_val,
        "westata": w_val,
        "rel": _pair_rel(w_val, s_val),
    }


def _westata_scalar(key: str, result: Any) -> float | None:
    """Look an ``e()`` scalar up on ``result.e`` first, then as an attribute."""
    mapping = getattr(result, "e", None)
    if mapping is not None:
        try:
            if key in mapping:
                return float(mapping[key])
        except (TypeError, ValueError):
            pass
    try:
        attribute = getattr(result, key)
    except AttributeError:
        return None
    try:
        return float(attribute() if callable(attribute) else attribute)
    except (TypeError, ValueError):
        return None


def difference_row(record: Record) -> list[tuple[str, str, str, str]]:
    """One record's cells for the report's difference table."""
    return [
        (
            record.name,
            f"{record.max_b_rel:.3e}" if _finite(record.max_b_rel) else "n/a",
            f"{record.max_se_rel:.3e}" if _finite(record.max_se_rel) else "n/a",
            record.verdict,
        )
    ]


def tier(record: Record) -> str:
    """``stata_probe``'s A-E tier for the coefficient difference.

    Provided so a replication log can be read next to the existing sweep, whose
    summaries are graded on the same scale.
    """
    return tier_for(record.max_b_rel)
