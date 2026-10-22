"""Stata ``e()`` scalar container.

Stata exposes post-estimation scalars through ``e(name)``.  westata mirrors that
with a mapping that also allows attribute access for interactive use:

>>> e = EReturn({"r2": 0.81})
>>> e["r2"], e.r2
(0.81, 0.81)

Item access keeps normal mapping semantics and raises ``KeyError`` for an
unrecorded scalar.  Attribute access is deliberately forgiving and returns
``None`` for an unrecorded scalar, matching the "this e() scalar is not stored"
case without forcing callers to write ``e.get(...)``.
"""

from __future__ import annotations

from typing import Any, Mapping


class EReturn(dict[str, Any]):
    """Mapping of Stata ``e()`` scalars with item and attribute access."""

    def __init__(
        self,
        mapping: Mapping[str, Any] | None = None,
        /,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if mapping is not None:
            self.update(mapping)
        if kwargs:
            self.update(kwargs)

    def __getattr__(self, name: str) -> Any:
        """Return ``None`` for a scalar that was not recorded.

        Dunder lookups are excluded so that copying, pickling and ``repr`` never
        recurse through ``__getattr__``.
        """

        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return self.get(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("__") and name.endswith("__"):
            super().__setattr__(name, value)
        else:
            self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __or__(self, other: Mapping[str, Any]) -> EReturn:
        merged = EReturn(self)
        merged.update(other)
        return merged

    def __ror__(self, other: Mapping[str, Any]) -> EReturn:
        merged = EReturn(other)
        merged.update(self)
        return merged
