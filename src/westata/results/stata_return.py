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

    @staticmethod
    def _bare(name: Any) -> Any:
        """Strip Stata's ``e(...)`` wrapper from a key.

        ``ereturn list`` prints names as ``e(N)``, ``e(r2)`` and so on, so a reader copying a
        name out of that listing or out of the Stata manual writes ``results.e["e(N)"]``. The
        stored key is the bare ``N``. Accepting both spellings costs one normalisation and stops
        a KeyError that says nothing about the convention.
        """

        if isinstance(name, str) and name.startswith("e(") and name.endswith(")"):
            return name[2:-1]
        return name

    def __getitem__(self, key: Any) -> Any:
        return super().__getitem__(self._bare(key))

    def __contains__(self, key: Any) -> bool:
        return super().__contains__(self._bare(key))

    def get(self, key: Any, default: Any = None) -> Any:
        return super().get(self._bare(key), default)

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
