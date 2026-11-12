"""The time axis that Stata's ``tsset`` supplies, and what it changes in lagged residuals.

Stata's serial-correlation commands do not lag by ROW POSITION. They lag by the time
variable set with ``tsset``::

    tsset year
    reg y x
    estat bgodfrey, lags(2)

``bgodfrey.ado:59-61`` fetches the variables with ``_ts timevar panelvar if `touse', sort
onepanel`` and then ``bgodfrey.ado:163`` builds the lags with ``tsrevar l(1/`lags').`res'``.
``L.`` is defined on the time axis, so a row whose predecessor PERIOD is absent from the
sample gets a MISSING lag rather than the previous row's value.

Everything below is the empirical foundation, measured on StataNow 19.5:

* **A gap makes a lag missing, not adjacent.**
  ``icecream.dta`` with ``time == 5`` dropped (29 rows). ``L.e1`` at ``time == 6`` is
  missing, and ``corrgram`` reports ``(note: time series has 1 gap)``. Pairing the rows by
  position instead moves ``estat bgodfrey`` from 4.220 to 3.286 at lag 1.
* **The default lag step is ONE, whatever the spacing.**
  ``tsset`` reports ``Delta: 1 unit`` for periods ``1,4,7,...`` and for ``10,20,30,...`` --
  both get ``but with gaps`` -- so time values ``1,4,7`` have NO second observation at
  ``t-1`` and every lag is missing. Only an explicit ``tsset t, delta(2)`` changes that.
  Measured: with ``delta(2)`` on ``t = 2,4,6,...``, ``l.e`` at ``t == 4`` is the value at
  ``t == 2``.
* **Non-integer period values are refused** by ``tsset`` itself ("time variable must
  contain only integer values", r(451)), which is why ``delta`` here is validated the same
  way.
* **A gap count is arithmetic, not a search**: ``tsreport`` reports
  ``(end - start) / delta + 1 - n``. Times ``1..30`` minus 5 give 1; two removals give 2.

The module is deliberately separate from ``bgodfrey.py`` and ``serial.py``: three commands
share this arithmetic, and a divergent copy in each is how the same bug grows three
different wrong numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from westata.errors import ConsistencyError

__all__ = [
    "TimeAxis",
    "resolve_time_axis",
    "require_single_unit",
    "refuse_panel_fit",
    "refuse_clustered_vce",
]


@dataclass(frozen=True)
class TimeAxis:
    """The estimation sample's time axis, as ``tsset`` would have set it."""

    #: Period value of each row of the estimation sample, in row order.
    values: np.ndarray
    #: The lag step. One unless ``delta=`` says otherwise (``tsset, delta(#)``).
    delta: float
    #: ``r(N_gaps)`` from ``tsreport`` -- periods inside the sample's span with no row.
    n_gaps: int

    def lag_indices(self, depth: int) -> np.ndarray:
        """Row index of ``L{depth}.`` for each row, or ``-1`` where that period is absent.

        This is what makes a gap produce a MISSING lag instead of the neighbouring row:
        ``t - depth * delta`` either names a row or it does not.
        """

        wanted = self.values - depth * self.delta
        # Exact matching on the period values: the periods are integers by `tsset`'s own
        # rule, so a float key is unambiguous here and avoids a tolerance to tune.
        lookup = {float(value): index for index, value in enumerate(self.values)}
        return np.array(
            [lookup.get(float(value), -1) for value in wanted], dtype=int
        )


def refuse_panel_fit(result: Any, command: str) -> None:
    """Refuse a fit that spans several panels, as ``_ts ..., onepanel`` does.

    Measured on StataNow 19.5 with ``xtset id yr`` -- both after ``xtreg`` (``estat
    bgodfrey not valid``, r(321)) and after a plain ``regress`` on panel data, where
    ``estat bgodfrey``, ``estat dwatson``, ``corrgram`` and ``wntestq`` all stop with
    ``sample may not include multiple panels`` (r(459)).
    """

    entity = getattr(result, "panel_entity", None)
    if entity is None:
        return
    labels = np.asarray(entity)
    if labels.ndim != 1 or labels.size == 0:
        return
    try:
        distinct = np.unique(labels.astype(str)).size
    except (TypeError, ValueError):  # pragma: no cover - exotic label dtype
        distinct = np.unique(labels).size
    if distinct > 1:
        raise ConsistencyError(
            f"{command}: sample may not include multiple panels. Stata refuses this "
            "because the lags would run across panel boundaries; estimate on one unit, "
            "or use a panel-specific serial-correlation test instead."
        )


def refuse_clustered_vce(result: Any, command: str) -> None:
    """Refuse a clustered or otherwise robust VCE, as ``bgodfrey.ado:5-10`` does.

    Stata tests ``e(vcetype) == "Robust"``, and ``vce(cluster)`` sets exactly that, so a
    clustered fit dies with ``This command does not work after regress, robust`` (r(198)).
    The check is by NAME, which is why the non-robust spelling ``nonrobust`` has to be
    excluded explicitly -- a substring test would reject every ordinary fit.
    """

    cov_type = str(getattr(result, "cov_type", "") or "").strip().lower()
    robust_names = {
        "robust",
        "cluster",
        "clustered",
        "hc0",
        "hc1",
        "hc2",
        "hc3",
        "hc4",
        "hc5",
    }
    if cov_type in robust_names or ("robust" in cov_type and not cov_type.startswith("non")):
        raise ConsistencyError(
            f"{command}: this command does not work after regress, robust "
            "(vce(cluster) is robust in Stata's eyes too, so it is refused as well)"
        )


def require_single_unit(times: Any, command: str, size: int) -> np.ndarray:
    """Validate a caller-supplied time axis and return it as floats.

    Duplicate periods mean the data hold more than one unit (a panel), which is the case
    Stata answers with ``sample may not include multiple panels``.
    """

    values = np.asarray(times, dtype=float).reshape(-1)
    if values.size != size:
        raise ConsistencyError(
            f"{command}: time has {values.size} entries against {size} observations"
        )
    if not np.isfinite(values).all():
        raise ConsistencyError(f"{command}: the time variable contains missing values")
    if np.unique(values).size != values.size:
        raise ConsistencyError(
            f"{command}: sample may not include multiple panels -- the time variable has "
            "repeated values, so the lags would run across unit boundaries"
        )
    return values


def resolve_time_axis(
    result: Any,
    time: str | Sequence[Any] | None,
    *,
    delta: float,
    command: str,
    size: int | None = None,
) -> TimeAxis | None:
    """Build the :class:`TimeAxis`, or ``None`` when the caller wants the row order.

    ``time`` may name a column of the fit's retained estimation frame (``"year"``) or be
    the period values themselves. Omitting it keeps the historical behaviour -- lags by
    row order -- so this is an additive parameter, not a reinterpretation.

    Raises
    ------
    ValueError
        For a non-positive or non-integer ``delta``: ``tsset`` refuses fractional period
        values outright ("time variable must contain only integer values", r(451)).
    ConsistencyError
        When ``time`` cannot be resolved against the estimation sample.
    """

    if abs(float(delta) - round(float(delta))) > 1e-12 or float(delta) <= 0.0:
        raise ValueError(
            f"{command}: delta must be a positive integer, as tsset requires; got {delta}"
        )
    if time is None:
        return None

    if size is None:
        resid = getattr(result, "resid", None)
        if resid is not None:
            size = int(np.asarray(resid, dtype=float).size)
        else:
            try:
                size = int(np.asarray(result, dtype=float).reshape(-1).size)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                size = 0

    if isinstance(time, str):
        frame = getattr(result, "model_data", None)
        if frame is None or time not in getattr(frame, "columns", ()):
            raise ConsistencyError(
                f"{command}: time={time!r} is not a column of the estimation frame. "
                "Pass the period values themselves, or fit from a frame that carries "
                "the time variable."
            )
        values = np.asarray(frame[time], dtype=float).reshape(-1)
    else:
        values = np.asarray(time, dtype=float).reshape(-1)

    values = require_single_unit(values, command, size)
    ordered = np.sort(values)
    steps = np.diff(ordered)
    if steps.size and np.any(steps < float(delta) * 0.5):
        raise ConsistencyError(
            f"{command}: periods are spaced more finely than delta={delta}; "
            "tsset would refuse this variable"
        )
    span = int(round((ordered[-1] - ordered[0]) / float(delta))) + 1
    n_gaps = max(span - int(values.size), 0)
    return TimeAxis(values=values, delta=float(delta), n_gaps=int(n_gaps))
