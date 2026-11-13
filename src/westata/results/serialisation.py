"""Extraction contract for result objects (R4.1).

Every public result class must be readable as DATA, not only printable as a table.
Before this module eighteen of them offered ``summary()`` or ``__str__`` and nothing
else, so a caller who wanted the numbers had to parse rendered text or reach into
``__dict__`` -- exactly the failure ``docs/API_CONVENTIONS.md`` R1.4 describes.

The contract is deliberately small:

* :meth:`ResultSerialisation.to_dict` -- flat ``{name: value}`` for scalars.
* :meth:`ResultSerialisation.to_frame` -- one row per parameter/row, for tables.

Both are generic defaults read off ``dataclasses.fields`` (or ``vars`` for the two
non-dataclass results), so each class overrides only where a generic reading would be
MISLEADING (a numpy array collapsed into one dict cell, for instance).  Writing 18
near-identical methods would have created eighteen places to drift, which is the
R6.1 problem this package has already paid for.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np


def _is_scalar(value: Any) -> bool:
    """Whether a value belongs in the flat dict rather than in a column."""

    if isinstance(value, (str, bytes, bool, int, float, complex, np.generic)):
        return True
    return value is None


def _as_plain(value: Any) -> Any:
    """Convert numpy scalars/arrays into JSON-friendly Python objects."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


class ResultSerialisation:
    """Generic ``to_dict`` / ``to_frame`` for a result object.

    Mix this in BEFORE the dataclass decorator's own members matter; it reads
    ``dataclasses.fields(self)`` when the class is a dataclass and ``vars(self)``
    otherwise.  ``_serialisation_skip`` names fields that must not be flattened
    (large matrices, render caches).
    """

    #: Fields/attributes excluded from ``to_dict``/``to_frame`` by default.
    _serialisation_skip: frozenset[str] = frozenset()

    def _serialisation_items(self) -> list[tuple[str, Any]]:
        if dataclasses.is_dataclass(self) and not isinstance(self, type):
            pairs = [(f.name, getattr(self, f.name)) for f in dataclasses.fields(self)]
        else:
            pairs = [
                (name, value)
                for name, value in vars(self).items()
                if not name.startswith("_")
            ]
        return [
            (name, value)
            for name, value in pairs
            if name not in self._serialisation_skip
        ]

    def to_dict(self) -> dict[str, Any]:
        """Flat ``{name: value}`` of the scalar members.

        Array-valued members are included as ``list`` (so the dict stays
        JSON-serialisable) rather than silently dropped: a caller looking for
        ``W_weights`` must not have to guess whether the key exists.
        """

        result: dict[str, Any] = {}
        for name, value in self._serialisation_items():
            if _is_scalar(value):
                result[name] = _as_plain(value)
            elif isinstance(value, np.ndarray):
                result[name] = _as_plain(value)
            elif isinstance(value, (list, tuple)):
                result[name] = [_as_plain(item) for item in value]
            elif isinstance(value, dict):
                result[name] = {str(k): _as_plain(v) for k, v in value.items()}
            # Anything else (nested objects, models) stays out of the flat dict; it is
            # reachable as an attribute, and flattening it would produce a repr string
            # that looks like data but is not.
        return result

    def to_frame(self) -> Any:
        """One row per scalar member -- the generic table shape.

        Classes whose natural table is "one row per parameter" or "one row per test"
        override this; the default is the honest fallback for everything else, and it
        is what makes ``.to_frame()`` exist rather than raise ``AttributeError``.
        """

        import pandas as pd

        return pd.DataFrame([self.to_dict()])
