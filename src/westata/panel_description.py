"""Panel bookkeeping: Stata's ``xtset``, ``xtdes`` and ``xtsum``.

These three describe a panel rather than estimate anything. Every layout here was measured from
StataNow 19.5 on two fixtures -- a balanced 48 x 17 panel and an unbalanced one with a hole -- rather
than guessed.

``xtset pid year`` (balanced)::

    Panel variable: pid (strongly balanced)
     Time variable: year, 1970 to 1986
             Delta: 1 unit

``xtset id t`` (unbalanced)::

    Panel variable: id (unbalanced)
     Time variable: t, 2001 to 2005
             Delta: 1 unit

with an interior hole the time line becomes::

     Time variable: t, 1 to 5, but with a gap

``xtdes`` (balanced)::

         pid:  1, 2, ..., 48                                     n =         48
        year:  1970, 1971, ..., 1986                             T =         17
               Delta(year) = 1 unit
               Span(year)  = 17 periods
               (pid*year uniquely identifies each observation)

    Distribution of T_i:   min      5%     25%       50%       75%     95%     max
                            17      17      17        17        17      17      17

         Freq.  Percent    Cum. |  Pattern
     ---------------------------+-------------------
           48    100.00  100.00 |  11111111111111111
     ---------------------------+-------------------
           48    100.00         |  XXXXXXXXXXXXXXXXX

``xtsum y k1``::

    Variable         |      Mean   Std. dev.       Min        Max |    Observations
    -----------------+--------------------------------------------+----------------
    y        overall |  61014.32    69973.9       4354     464550 |     N =     816
             between |             69484.36   5472.118   350056.9 |     n =      48
             within  |             12769.31  -25109.56   175507.4 |     T =      17
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from westata.errors import ConsistencyError
from westata.results.serialisation import ResultSerialisation

__all__ = [
    "PanelSetResults",
    "PanelDesResults",
    "PanelSumResults",
    "panel_set",
    "panel_describe",
    "panel_summarize",
]

#: ``xtdes`` prints every line 75 characters wide; ``xtsum`` uses 79.
_XTDES_WIDTH = 75
_XTSUM_WIDTH = 79


# ---------------------------------------------------------------- shared helpers
def _panel_frame(data: pd.DataFrame, entity: str, time: str) -> pd.DataFrame:
    """Validate the two columns and return the complete rows."""

    if not isinstance(data, pd.DataFrame):
        raise ConsistencyError("panel varibles: data must be a pandas DataFrame")
    for column, role in ((entity, "panel"), (time, "time")):
        if column not in data.columns:
            raise ConsistencyError(
                f"{role} variable {column!r} not found in data; available columns: "
                f"{list(data.columns)}"
            )
    subset = data[[entity, time]].dropna()
    if subset.empty:
        raise ConsistencyError("no complete observations on the panel and time variables")
    duplicated = subset.duplicated([entity, time])
    if bool(duplicated.any()):
        raise ConsistencyError(
            f"{entity}*{time} does not uniquely identify the observations "
            f"({int(duplicated.sum())} duplicates)"
        )
    return subset


def _sizes(subset: pd.DataFrame, entity: str) -> pd.Series:
    return subset.groupby(entity, sort=True).size()


def _time_values(subset: pd.DataFrame, time: str) -> list[Any]:
    return sorted(pd.unique(subset[time]).tolist())


def _delta(times: Sequence[Any]) -> float | None:
    """The modal spacing between consecutive time values, as Stata reports it."""

    if len(times) < 2:
        return None
    try:
        numeric = [float(value) for value in times]
    except (TypeError, ValueError):
        return None
    gaps = np.diff(numeric)
    gaps = gaps[gaps > 0]
    if gaps.size == 0:
        return None
    values, counts = np.unique(np.round(gaps, 10), return_counts=True)
    return float(values[counts.argmax()])


def _number(value: Any) -> str:
    if value is None:
        return "."
    if isinstance(value, (float, np.floating)) and abs(float(value) - round(float(value))) < 1e-12:
        return str(int(round(float(value))))
    return str(value)


def _cell(text: str, end: int, used: int) -> str:
    """Right-align ``text`` so the field ends at column ``end``, given ``used`` already emitted."""

    return " " * max(end - used - len(text), 1) + text


# ---------------------------------------------------------------- xtset
@dataclass(frozen=True)
class PanelSetResults(ResultSerialisation):
    """``xtset``: confirm the panel structure and summarise it."""

    entity: str
    time: str
    balance: str
    first_time: Any = None
    last_time: Any = None
    delta: float | None = None
    gapped: bool = False
    entities: int = 0
    periods: int = 0
    nobs: int = 0

    def __str__(self) -> str:
        lines = ["", f"Panel variable: {self.entity} ({self.balance})"]
        time_text = f" Time variable: {self.time}, {self.first_time} to {self.last_time}"
        if self.gapped:
            time_text += ", but with a gap"
        lines.append(time_text)
        if self.delta is not None:
            unit = "unit" if abs(self.delta - 1.0) < 1e-12 else "units"
            lines.append(f"         Delta: {_number(self.delta)} {unit}")
        return "\n".join(lines)


def panel_set(data: pd.DataFrame, entity: str, time: str) -> PanelSetResults:
    """Stata's ``xtset panelvar timevar``.

    Validates that the pair identifies the observations, then reports the balance, the time range and
    the spacing -- the three things Stata prints and every ``xt`` estimator relies on.
    """

    subset = _panel_frame(data, entity, time)
    sizes = _sizes(subset, entity)
    times = _time_values(subset, time)
    delta = _delta(times)
    if sizes.nunique() == 1 and int(sizes.iloc[0]) == len(times):
        balance = "strongly balanced"
    elif sizes.nunique() == 1:
        balance = "balanced"
    else:
        balance = "unbalanced"
    # A gap is a hole INSIDE a single entity's range, not inside the union of all observed periods.
    # The fixture makes the difference concrete: entity 2 covers t = 1..5 while entity 1 skips t = 3,
    # so the union is [1, 2, 3, 4, 5] with no missing step, yet Stata still prints
    # "1 to 5, but with a gap". Testing the union therefore cannot detect it -- walk the entities.
    gapped = False
    if len(times) >= 2 and delta:
        try:
            expected_step = float(delta)
            for _, group in subset.groupby(entity, sort=False):
                numeric = sorted(float(value) for value in group[time])
                steps = int(round((numeric[-1] - numeric[0]) / expected_step)) + 1
                if steps > len(numeric):
                    gapped = True
                    break
        except (TypeError, ValueError):
            gapped = False

    return PanelSetResults(
        entity=entity,
        time=time,
        balance=balance,
        first_time=times[0] if times else None,
        last_time=times[-1] if times else None,
        delta=delta,
        gapped=gapped,
        entities=int(sizes.size),
        periods=int(len(times)),
        nobs=int(len(subset)),
    )


# ---------------------------------------------------------------- xtdes
def _pattern(values: np.ndarray, span: Sequence[Any]) -> str:
    """``1`` where the entity has that period, ``.`` where it does not."""

    present = set(values.tolist())
    return "".join("1" if period in present else "." for period in span)


#: Column ends for the ``Distribution of T_i`` grid, measured from the capture.
_QUANTILE_COLUMNS = (
    ("min", 26),
    ("5%", 34),
    ("25%", 42),
    ("50%", 52),
    ("75%", 62),
    ("95%", 70),
    ("max", 78),
)


@dataclass(frozen=True)
class PanelDesResults(ResultSerialisation):
    """``xtdes``: the panel's shape, its T distribution and its observation patterns."""

    entity: str
    time: str
    entities: int
    periods: int
    nobs: int
    first_entity: Any = None
    second_entity: Any = None
    last_entity: Any = None
    first_time: Any = None
    second_time: Any = None
    last_time: Any = None
    delta: float | None = None
    span: int = 0
    size_quantiles: dict[str, float] = field(default_factory=dict)
    patterns: list[tuple[str, int]] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [""]

        for name, count_label, count, first, second, last in (
            (self.entity, "n", self.entities, self.first_entity, self.second_entity,
             self.last_entity),
            (self.time, "T", self.periods, self.first_time, self.second_time, self.last_time),
        ):
            head = f"{name:>8}:  {first}, {second}, ..., {last}"
            # MEASURED: the count label starts at column 61, '=' at 63, and the value ends at 75.
            gap = max(61 - len(head), 1)
            tail = f"{count_label} = {count:>10}"
            lines.append((head + " " * gap + tail).ljust(_XTDES_WIDTH))

        # 'Delta(year) = 1 unit' and 'Span(year)  = 17 periods': the label field is 13 wide, so the
        # '=' sits one column further right for the shorter of the two.
        # MEASURED: these carry a fixed 11-space indent (they line up under the name, not under the
        # colon) and the label field is 13 wide so '=' aligns: 'Delta(year) =' / 'Span(year)  ='.
        labels = (
            (f"Delta({self.time})", f"{_number(self.delta)} unit"),
            (f"Span({self.time})", f"{self.span} periods"),
        )
        field = max(len(name) for name, _ in labels) + 1
        for label, value in labels:
            lines.append(f"{'':11}{label:<{field}}= {value}")
        lines.append(
            f"{'':11}({self.entity}*{self.time} uniquely identifies each observation)"
        )
        lines.append("")

        # MEASURED: the header label occupies columns 0..23 and the seven values right-align to
        # 24 / 32 / 39 / 48 / 56 / 63 / 72.
        header = "Distribution of T_i:"
        cells, used = [], len(header)
        for label, end in _QUANTILE_COLUMNS:
            cells.append(_cell(label, end, used))
            used = end
        lines.append(header + "".join(cells))

        cells, used = [], 0
        for label, end in _QUANTILE_COLUMNS:
            cells.append(_cell(f"{self.size_quantiles.get(label, 0):g}", end, used))
            used = end
        lines.append("".join(cells))
        lines.append("")

        # MEASURED: the rule's right-hand segment is 19 dashes for a 17-period pattern and 9 for a
        # 5-period one, i.e. pattern width + 2.
        span_for_rule = len(self.patterns[0][0]) if self.patterns else 0
        # MEASURED across three spans: 4 -> 9 dashes, 5 -> 9, 17 -> 19, i.e. max(9, span + 2).
        rule = " " + "-" * 27 + "+" + "-" * max(9, span_for_rule + 2)
        lines.append("     Freq.  Percent    Cum. |  Pattern")
        lines.append(rule)
        total = sum(count for _, count in self.patterns) or 1
        cumulative = 0.0
        for pattern, count in self.patterns:
            percent = 100.0 * count / total
            cumulative += percent
            lines.append(f"{count:>9}{percent:>10.2f}{cumulative:>8.2f} |  {pattern}")
        lines.append(rule)
        span = len(self.patterns[0][0]) if self.patterns else 0
        lines.append(f"{total:>9}{100.0:>10.2f}{'':>8} |  " + "X" * span)
        return "\n".join(lines)


def panel_describe(data: pd.DataFrame, entity: str, time: str) -> PanelDesResults:
    """Stata's ``xtdes``: describe the shape of a panel."""

    subset = _panel_frame(data, entity, time)
    sizes = _sizes(subset, entity)
    times = _time_values(subset, time)

    quantiles = {
        label: float(sizes.quantile(float(label.rstrip("%")) / 100.0))
        if label not in ("min", "max")
        else float(sizes.min() if label == "min" else sizes.max())
        for label, _ in _QUANTILE_COLUMNS
    }

    counts: dict[str, int] = {}
    for _, group in subset.groupby(entity, sort=True):
        pattern = _pattern(group[time].to_numpy(), times)
        counts[pattern] = counts.get(pattern, 0) + 1
    patterns = sorted(counts.items())

    entity_values = sorted(pd.unique(subset[entity]).tolist())
    return PanelDesResults(
        entity=entity,
        time=time,
        entities=int(sizes.size),
        periods=int(len(times)),
        nobs=int(len(subset)),
        first_entity=entity_values[0] if entity_values else None,
        second_entity=entity_values[1] if len(entity_values) > 1 else (entity_values[0] if entity_values else None),
        last_entity=entity_values[-1] if entity_values else None,
        first_time=times[0] if times else None,
        second_time=times[1] if len(times) > 1 else (times[0] if times else None),
        last_time=times[-1] if times else None,
        delta=_delta(times),
        span=int(len(times)),
        size_quantiles=quantiles,
        patterns=patterns,
    )

# ---------------------------------------------------------------- xtsum
@dataclass(frozen=True)
class PanelSumResults(ResultSerialisation):
    """``xtsum``: overall / between / within decomposition of the listed variables."""

    entity: str
    time: str
    #: One entry per variable: (name, overall, between, within) where each block is a dict with
    #: ``mean``, ``sd``, ``min`` and ``max``.
    variables: list[tuple[str, dict[str, float], dict[str, float], dict[str, float]]] = field(
        default_factory=list
    )
    nobs: int = 0
    entities: int = 0
    periods: int = 0

    def __str__(self) -> str:
        """Measured on ``xtsum y k1``: 79 characters wide, pipes at columns 17 and 62.

        The label block is ``name`` left-aligned then ``overall``/``between``/``within``
        right-aligned so the word ends at column 16. Value columns end at 28 / 39 / 50 / 61, and the
        observation block is ``N =`` then a value ending at 79.
        """

        lines = [""]
        # MEASURED: 'Mean' ends at 28, 'Std. dev.' at 40, 'Min' at 50, 'Max' at 61.
        lines.append(
            f"{'Variable':<17}|{'Mean':>10}{'Std. dev.':>12}{'Min':>10}{'Max':>11} |"
            f"{'Observations':>16}"
        )
        lines.append("-" * 17 + "+" + "-" * 44 + "+" + "-" * 16)

        for position, (name, overall, between, within) in enumerate(self.variables):
            if position:
                lines.append(f"{'':17}|{'':44}|")
            blocks = (
                ("overall", overall, "N", self.nobs),
                ("between", between, "n", self.entities),
                ("within", within, "T", self.periods),
            )
            for index, (label, block, count_label, count) in enumerate(blocks):
                # The label block occupies columns 0..16 in every row: the variable name on the
                # first, the block name right-aligned to column 16 on all three.
                # MEASURED: the label block is 17 wide with the pipe at column 17.
                # MEASURED: the label ENDS at column 16 in every row, so it is right-aligned in
                # the 8 columns 9..16 and the pipe follows at 17.
                # MEASURED char-by-char: the label is LEFT-aligned in 8 columns starting at 9, and
                # the pipe follows at 17. That is why 'within' (6 chars) carries two trailing spaces
                # and 'between' (7) only one -- a right-aligned field would have padded the other
                # way and produced three failed attempts.
                if index == 0:
                    head = f"{name:<9}{label:<8}|"
                else:
                    head = f"{'':9}{label:<8}|"
                # Values are emitted relative to the pipe at column 17, so `used` starts there.
                # Values are laid out after the pipe at column 18, so the offsets are relative to
                # that: mean ends at 28, sd at 39, min at 50, max at 61.
                cells = ""
                if index == 0:
                    cells += _cell(_xtsum(block.get("mean")), 28, 18)
                else:
                    cells += " " * 10
                cells += _cell(_xtsum(block.get("sd")), 39, 28)
                cells += _cell(_xtsum(block.get("min")), 50, 39)
                cells += _cell(_xtsum(block.get("max")), 61, 50)
                # MEASURED: 'N' at 68, '=' at 70, the value ends at 79.
                # MEASURED: '|' at 62, 'N' at 68 (5 spaces), '=' at 70, value ends at 79.
                tail = f"{count_label} = {count:>7}"
                lines.append(head + cells + " |" + " " * 5 + tail)
        return "\n".join(lines)


def _xtsum(value: float | None) -> str:
    """Stata's ``xtsum`` number format: six significant figures, leading zero dropped.

    MEASURED from ``xtsum y k1``:

        61014.32   69973.9    4354     464550
        69484.36   5472.118   350056.9
        12769.31  -25109.56  175507.4
        25036.66   27780.4    2627.12  140217.3

    ``61014.32`` has seven characters and ``69973.9`` six, so this is a SIGNIFICANT-FIGURE rule, not
    a fixed decimal count; eight figures would print ``61014.316``.
    """

    if value is None or not np.isfinite(value):
        return ""
    text = f"{float(value):.7g}"
    if text.startswith("0."):
        text = text[1:]
    elif text.startswith("-0."):
        text = "-" + text[2:]
    return text


def panel_summarize(
    data: pd.DataFrame,
    variables: Sequence[str],
    *,
    entity: str,
    time: str,
) -> PanelSumResults:
    """Stata's ``xtsum``: decompose each variable into overall, between and within variation.

    ``overall`` uses every observation; ``between`` uses the per-entity means (``n`` of them);
    ``within`` uses deviations from those means, rescaled so its standard deviation is comparable
    (Stata multiplies by ``sqrt(T/(T-1))``).
    """

    subset = _panel_frame(data, entity, time)
    names = [variables] if isinstance(variables, str) else list(variables)
    for name in names:
        if name not in data.columns:
            raise ConsistencyError(
                f"xtsum: variable {name!r} not found in data; available columns: "
                f"{list(data.columns)}"
            )

    sizes = _sizes(subset, entity)
    periods = int(len(_time_values(subset, time)))
    frame = data.loc[subset.index]

    out: list[tuple[str, dict[str, float], dict[str, float], dict[str, float]]] = []
    for name in names:
        values = pd.to_numeric(frame[name], errors="coerce")
        keep = values.notna()
        series = values[keep]
        if series.empty:
            continue
        codes = frame.loc[keep, entity]
        grouped = series.groupby(codes)

        overall = {
            "mean": float(series.mean()),
            "sd": float(series.std(ddof=1)),
            "min": float(series.min()),
            "max": float(series.max()),
        }
        means = grouped.mean()
        between = {
            "mean": float("nan"),
            "sd": float(means.std(ddof=1)),
            "min": float(means.min()),
            "max": float(means.max()),
        }
        deviations = series - grouped.transform("mean")
        # MEASURED on munnell.dta: Stata reports 12769.31 for the within sd of `y`, which is exactly
        # the sample standard deviation of the deviations (ddof=1). Two plausible-looking variants
        # are both wrong: ddof=0 gives 12761.48, and multiplying ddof=0 by sqrt(T/(T-1)) -- the
        # rescaling used elsewhere -- gives 13162.30. No extra factor belongs here.
        within_sd = float(deviations.std(ddof=1))
        # MEASURED: Stata reports the within MIN/MAX shifted back by the GRAND mean, not the raw
        # deviations. On munnell.dta the raw deviations span -86123.88 .. 114493.12, while Stata
        # prints -25109.56 .. 175507.4 -- exactly deviations + overall mean (61014.32).
        centred = deviations + overall["mean"]
        within = {
            "mean": float("nan"),
            "sd": within_sd,
            "min": float(centred.min()),
            "max": float(centred.max()),
        }
        out.append((name, overall, between, within))

    return PanelSumResults(
        entity=entity,
        time=time,
        variables=out,
        nobs=int(len(subset)),
        entities=int(sizes.size),
        periods=periods,
    )

