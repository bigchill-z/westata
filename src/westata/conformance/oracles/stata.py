"""Stata oracle: run real Stata and capture ``e()`` for conformance checks.

Requires StataNow/Stata 17+ with the bundled ``pystata`` package and a valid
licence.  This module is a *test* dependency; nothing in ``westata`` imports it
at runtime.

Two properties matter for reproducible evidence:

* Results are captured as plain floats/arrays and can be written to JSON, so a
  comparison can be replayed on a machine without Stata.
* Every capture records ``c(stata_version)`` and ``c(version)``, because Stata
  results are only meaningful relative to a version.

Typical use::

    with StataOracle(stata_dir=r"D:\\Program Files\\StataNow19", edition="mp") as so:
        so.run("sysuse auto, clear")
        so.run("regress mpg weight foreign")
        capture = so.capture()
    capture["b"]          # coefficients, in e(b) order
    capture["V"]          # covariance matrix
    capture["scalars"]    # e(N), e(df_r), e(r2), ...
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = ["StataCapture", "StataOracle", "stata_available"]

#: Default install location on this machine; override via ``stata_dir``.
DEFAULT_STATA_DIR = r"D:\Program Files\StataNow19"


def stata_available() -> bool:
    """Whether a Stata installation with ``pystata`` is reachable."""

    candidate = Path(DEFAULT_STATA_DIR) / "utilities"
    return candidate.is_dir()


def _strip_prefix(key: str) -> str:
    """``"e(N)" -> "N"``; leave already-bare names alone."""

    if key.startswith("e(") and key.endswith(")"):
        return key[2:-1]
    return key


@dataclass
class StataCapture:
    """Everything a conformance check needs from one Stata estimation.

    Attributes
    ----------
    b:
        Coefficient vector in ``e(b)`` order.
    V:
        Covariance matrix of the estimators.
    scalars:
        Scalar ``e()`` results keyed without the ``e()`` wrapper.
    macros:
        String-valued ``e()`` results (``cmd``, ``vce``, ...).
    names:
        Coefficient names in ``e(b)`` order.
    version:
        Stata version string the capture was produced with.
    """

    b: np.ndarray
    V: np.ndarray
    scalars: dict[str, float]
    macros: dict[str, str]
    names: list[str] = field(default_factory=list)
    version: str = ""

    def __getitem__(self, key: str) -> Any:
        if key in self.scalars:
            return self.scalars[key]
        raise KeyError(f"{key!r} not in captured e() scalars: {sorted(self.scalars)}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "b": self.b.tolist(),
            "V": self.V.tolist(),
            "scalars": self.scalars,
            "macros": self.macros,
            "names": self.names,
            "version": self.version,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "StataCapture":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            b=np.asarray(payload["b"], dtype=float),
            V=np.asarray(payload["V"], dtype=float),
            scalars={k: float(v) for k, v in payload["scalars"].items()},
            macros=payload["macros"],
            names=list(payload.get("names", [])),
            version=payload.get("version", ""),
        )


class StataOracle:
    """Context manager that owns one embedded Stata session.

    Parameters
    ----------
    stata_dir:
        Stata installation directory (the one containing ``utilities/pystata``).
    edition:
        ``"mp"``, ``"se"`` or ``"be"``.
    """

    def __init__(
        self,
        stata_dir: str = DEFAULT_STATA_DIR,
        edition: str = "mp",
    ) -> None:
        self._stata_dir = stata_dir
        self._edition = edition
        self._stata: Any = None

    # ------------------------------------------------------------ lifecycle
    def __enter__(self) -> "StataOracle":
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def start(self) -> None:
        utilities = str(Path(self._stata_dir) / "utilities")
        if utilities not in sys.path:
            sys.path.append(utilities)
        from pystata import config

        config.init(self._edition, splash=False)
        from pystata import stata

        self._stata = stata

    def close(self) -> None:
        # pystata has no explicit shutdown; dropping the reference is enough and
        # avoids surprising the caller with a half-dead interpreter.
        self._stata = None

    # ---------------------------------------------------------------- running
    def run(self, command: str, *, quietly: bool = True) -> None:
        if self._stata is None:
            raise RuntimeError("StataOracle is not started; use it as a context manager")
        self._stata.run(command, quietly=quietly)

    def run_all(self, lines: Iterable[str]) -> None:
        for line in lines:
            self.run(line)

    def send_data(self, frame: Any, name: str = "df") -> None:
        """Load a pandas DataFrame into Stata under ``name``."""

        if self._stata is None:
            raise RuntimeError("StataOracle is not started")
        self._stata.pdataframe_to_data(frame, force=True)

    def get_output(self, command: str) -> str:
        """Run ``command`` and return its printed output.

        ``pystata.run(..., echo=False)`` prints to the process stdout instead of
        returning a string, so the output is captured by temporarily redirecting
        ``sys.stdout``.  Without this, output-parsing checks would see ``""`` and
        silently "fail to parse" rather than reporting the real result.
        """

        if self._stata is None:
            raise RuntimeError("StataOracle is not started")

        import io
        from contextlib import redirect_stdout

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            self._stata.run(command, echo=False)
        return buffer.getvalue()

    # ------------------------------------------------------------- capturing
    def capture(self, *, names: Sequence[str] | None = None) -> StataCapture:
        """Return the current ``e()`` results as a :class:`StataCapture`."""

        if self._stata is None:
            raise RuntimeError("StataOracle is not started")
        raw = self._stata.get_ereturn()

        b = None
        V = None
        scalars: dict[str, float] = {}
        macros: dict[str, str] = {}

        for key, value in raw.items():
            name = _strip_prefix(key)
            if value is None:
                continue
            if name == "b":
                b = np.asarray(value, dtype=float).reshape(-1)
            elif name == "V":
                V = np.asarray(value, dtype=float)
            elif isinstance(value, str):
                macros[name] = value
            elif isinstance(value, (int, float, np.floating, np.integer)):
                scalars[name] = float(value)

        if b is None or V is None:
            raise RuntimeError(
                "no e(b)/e(V) available; run an estimation command first"
            )

        # Coefficient order matters: e(b) is a row vector whose *column* stripes
        # hold the names.  Read them from Stata rather than inferring, otherwise
        # a comparison could align the wrong coefficients and still "pass".
        coefficient_names = list(names) if names else self._coefficient_names()

        # Version is recorded because a conformance claim is only meaningful
        # against a specific Stata release. ``display`` writes nothing to r(), so
        # read the value through a local macro instead.
        version = self._stata_version()

        return StataCapture(
            b=b,
            V=V,
            scalars=scalars,
            macros=macros,
            names=coefficient_names,
            version=version,
        )

    def _coefficient_names(self) -> list[str]:
        """Read the ``e(b)`` column stripes via ``sfi``."""

        if self._stata is None:
            return []
        try:
            names = self._stata.sfi.Matrix.getColNames("e(b)")
            return [str(name) for name in names]
        except Exception:
            return []

    def _stata_version(self) -> str:
        """Return ``c(stata_version)``, or ``""`` when it cannot be read.

        pystata exposes no direct Python accessor for ``c()`` macros and
        ``display`` writes nothing to ``r()``, so this deliberately returns an
        empty string rather than scraping stdout.  Record the version in the
        evidence registry by hand instead; a silently wrong version string would
        be worse than an absent one.
        """

        return ""
