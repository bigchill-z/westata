"""Several named IRF result sets in one object -- Stata's IRF file, as `irf create` builds it.

WHY THIS EXISTS

`irf graph` draws one subgraph per **(irfname, impulse, response)** and combines them. So a Stata
session with two names in its IRF file

    quietly irf create iu, set(macro) replace
    quietly irf create iu3, set(macro) replace
    irf graph oirf, i(unem) r(dinf)

shows TWO panels with the SAME impulse/response in the title, because ``i()``/``r()`` narrow the
variables and NOT the name. Measured: ``iu`` gives ``-.005521`` at step 1 where ``iu3`` gives
``-.006555``, so the two panels are genuinely different results that merely share a label.

The name is narrowed with ``irf(irfnames)`` -- **not** ``irfname()``, which Stata rejects with
``r(198)``.

westata's ``VARResults`` carries exactly one analysis, so this class is the store that makes several
names possible:

    store = ws.IrfStore()
    store.add(result, "iu")
    store.add(other_result, "iu3")
    store.names                     # ('iu', 'iu3')
    ws.irf_graph(store, statistic="oirf", impulse="unem", response="dinf")
    ws.irf_graph(store, statistic="oirf", impulse="unem", response="dinf", irf="iu")

With no ``irf=`` selector every name is drawn, which reproduces Stata's default exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from westata.errors import ConsistencyError

__all__ = ["IrfStore"]


@dataclass
class IrfStore:
    """Named IRF result sets, mirroring Stata's IRF file.

    Stata accumulates a result set per ``irf create NAME`` until the file is cleared; drawing then
    iterates over every name unless ``irf()`` narrows it. This class does the same in memory.
    """

    #: ``(name, analysis)`` pairs in insertion order, which is the order Stata lists them.
    entries: list[tuple[str, Any]] = field(default_factory=list)

    # ---------------------------------------------------------------- building
    def add(self, result: Any, name: str) -> "IrfStore":
        """Register one result under ``name``, like ``irf create name``.

        A repeated name REPLACES the earlier one, which is Stata's ``replace`` behaviour: the file
        keeps one entry per name, not a growing list.
        """

        analysis = _as_analysis(result)
        text = str(name).strip()
        if not text:
            raise ConsistencyError("irf create: the name must not be empty")
        for index, (existing, _) in enumerate(self.entries):
            if existing == text:
                self.entries[index] = (text, analysis)
                return self
        self.entries.append((text, analysis))
        return self

    def remove(self, name: str) -> "IrfStore":
        """Drop one name, raising if it is absent so a typo cannot pass silently."""

        for index, (existing, _) in enumerate(self.entries):
            if existing == name:
                del self.entries[index]
                return self
        raise ConsistencyError(f"irf: {name!r} is not in the store; names are {self.names}")

    def clear(self) -> "IrfStore":
        """Empty the store, like ``irf clear``."""

        self.entries.clear()
        return self

    # ---------------------------------------------------------------- inspection
    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, name: object) -> bool:
        return str(name) in self.names

    def get(self, name: str) -> Any:
        for existing, analysis in self.entries:
            if existing == name:
                return analysis
        raise ConsistencyError(f"irf: {name!r} is not in the store; names are {self.names}")

    def select(self, which: str | Sequence[str] | None) -> list[tuple[str, Any]]:
        """The entries ``irf()`` selects: ``None`` means every name, as Stata draws them all."""

        if which is None:
            return list(self.entries)
        wanted = [which] if isinstance(which, str) else list(which)
        if not wanted:
            raise ConsistencyError("irf: the selector is empty")
        chosen = []
        for name in wanted:
            chosen.append((str(name), self.get(str(name))))
        return chosen

    # ---------------------------------------------------------------- drawing
    def plot(self, **kwargs: Any) -> Any:
        """``irf graph`` over this store; see :func:`westata.postestimation.irf_graph`."""

        from westata.postestimation.irf_graph import irf_graph

        return irf_graph(self, **kwargs)


def _as_analysis(result: Any) -> Any:
    """Keep the WRAPPER when given one, so each later call rebuilds at its own horizon.

    Storing a pre-built analysis froze its periods: two names drawn together produced a 9-long
    step vector against 11-long values, and ill_between raised
    "'x' has size 9, but 'y1' has an unequal size of 11". A wrapper is re-analysed per call; a bare
    analysis is accepted unchanged for callers who already built one.
    """

    if hasattr(result, "orth_irfs"):
        return result
    if callable(getattr(result, "irf", None)):
        return result
    raise ConsistencyError(
        "irf create: pass a fitted VAR (ws.VAR(...).fit()) or its irf() analysis"
    )
