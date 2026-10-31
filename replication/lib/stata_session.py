"""Thin, defensive wrapper around the embedded Stata (``pystata``) session.

Why this module exists
----------------------
``retest/lib/stata_probe.py`` already proves the pystata mechanics, but it is a
sweep script written to be watched by a human: it stashes one live session in a
mutable object, trusts ``e(b)``'s column names to line up with its values, and
starts Stata from whatever directory the interpreter happens to be in.  A
replication harness runs unattended and repeatedly, so each of those shortcuts
turns into a silent-wrong-answer risk that this module closes:

* ``pystata.config.init`` can be called **once per process**.  A second call
  aborts inside Stata's own startup code, so the harness owns a single
  process-wide initialisation guarded by a lock and lets every
  :class:`StataSession` share it.
* A ``profile.py`` in the current directory shadows the standard-library
  ``profile`` module and breaks the IPython import that pystata performs.
  ``%TEMP%`` on this machine still contains such a file, which is why running
  Stata from a temporary directory fails with a traceback that points at
  IPython rather than at the stray file.  :func:`_assert_profile_not_shadowed`
  refuses to start in that situation and names the offending path.
* Stata represents a missing value as ``8.98846567431158e307``, not ``NaN``.
  Every value crossing the boundary is screened, so a comparison can never
  quietly treat a missing as a huge number and still call two coefficients
  "equal".
* An estimation command drops rows with missing covariates, so the number of
  observations *in the dataset* is not the number of observations *in the
  estimation*.  ``Capture.nobs`` is ``e(N)`` for that reason and
  ``Capture.data_nobs`` keeps the raw row count for diagnostics.

The wrapper deliberately does **not** interpret Stata's parameter naming
(``0b.g`` / ``1.g``) or drop base levels: ``capture`` reports exactly what Stata
stores, and the comparison layer decides what is comparable.  Keeping the two
concerns apart means a capture can be re-inspected and re-aligned without
re-running Stata.

Typical use::

    with StataSession() as st:
        st.load_dta(DTA)
        cap = st.estimate("reg y x [aw=w], cluster(state)")

    m = ws.OLS.from_formula("y ~ x", df, weights=df["w"], weight_type="aweight")
    rec = compare_capture("spec", cap, m.fit(cov_type="cluster", cov_kwds={"groups": df["state"]}))
"""

from __future__ import annotations

import math
import re
import sys
import sysconfig
import threading
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "CapturedMissing",
    "Capture",
    "StataError",
    "StataSession",
    "STATA_HOME",
    "repo_root",
    "screen_missing",
]


class CapturedMissing:
    """Stata's missing-value sentinel and the threshold that recognises it.

    A class rather than module constants so ``CapturedMissing.SENTINEL`` reads
    as a vocabulary at the call site and stays discoverable in tab completion.
    """

    #: Stata's largest representable double, used as the missing-value marker.
    SENTINEL = 8.98846567431158e307
    #: Anything at or above half the sentinel is treated as missing.  Stata
    #: stores several distinct missing codes (``.``, ``.a`` ... ``.z``) that
    #: differ by a few ulps near the sentinel, and float32 columns round it, so
    #: an exact equality test would miss real missings.
    FLOOR = SENTINEL * 0.5


#: Backwards-compatible module-level alias for the sentinel value.
MISSING_SENTINEL = CapturedMissing.SENTINEL
_MISSING_FLOOR = CapturedMissing.FLOOR

STATA_HOME = Path(r"D:\Program Files\StataNow19")
_UTILITIES = STATA_HOME / "utilities"

_STATA_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,31}$")
_RCODE_RE = re.compile(r"r\((?P<code>\d+)\)")

_INIT_LOCK = threading.Lock()
_INITIALISED = False


def repo_root() -> Path:
    """Return the repository root (``.../statapy``), derived from this file."""
    return Path(__file__).resolve().parents[2]


class StataError(RuntimeError):
    """A Stata command failed.

    ``rcode`` carries Stata's ``r(###)`` return code when the message contains
    one and ``-1`` otherwise, so callers can special-case "no e(b) in memory"
    without parsing prose.
    """

    def __init__(self, command: str, rcode: int, message: str) -> None:
        self.command = command
        self.rcode = rcode
        self.message = message
        suffix = f" [r({rcode})]" if rcode >= 0 else ""
        super().__init__(f"{command!r} failed{suffix}: {message}")


@dataclass
class Capture:
    """Everything an estimation leaves behind that a comparison needs.

    ``nobs`` is Stata's ``e(N)`` -- the estimation sample -- because that is the
    quantity the paper and westata report.  ``data_nobs`` is the number of rows
    in memory, which differs whenever the command drops observations; keeping
    both visible makes a "why do the Ns disagree?" question answerable without
    re-running anything.

    ``V`` may be ``None`` when a command leaves no ``e(V)`` (some postestimation
    commands do); the comparison layer then reports standard-error differences
    as not available instead of inventing zeros.
    """

    b: np.ndarray
    V: np.ndarray | None
    names: list[str]
    scalars: dict[str, float]
    macros: dict[str, str]
    nobs: int
    command: str = ""
    data_nobs: int = 0
    dropped: list[str] = field(default_factory=list)

    @property
    def se(self) -> np.ndarray:
        """Standard errors from ``sqrt(diag(e(V)))``; ``NaN`` without ``e(V)``."""
        if self.V is None or self.V.size == 0:
            return np.full(self.b.size, np.nan)
        return np.sqrt(np.diag(self.V))

    def get(self, name: str) -> float | None:
        """Return one ``e()`` scalar by its bare name (``"r2"``, not ``"e(r2)"``)."""
        key = name[2:-1] if name.startswith("e(") and name.endswith(")") else name
        return self.scalars.get(key)


def screen_missing(values: Any) -> Any:
    """Replace Stata's missing sentinel with ``NaN`` in an array or DataFrame.

    Returns the same kind of object it was given.  Integer and string columns
    pass through untouched: they cannot hold the sentinel, and masking them
    would either upcast the column to float (silently changing the dtype other
    code branches on) or raise.

    ``NaN`` rather than a large number is the whole point: a comparison that
    sees ``8.988e307`` reports a colossal "difference" and blames the
    estimator, while a comparison that sees ``NaN`` can say "not comparable".
    """
    if isinstance(values, pd.DataFrame):
        out = values.copy()
        for column in out.columns:
            series = out[column]
            if pd.api.types.is_float_dtype(series):
                out[column] = series.mask(series >= _MISSING_FLOOR, np.nan)
        return out
    arr = np.array(values, dtype=float, copy=True)
    arr[arr >= _MISSING_FLOOR] = np.nan
    return arr


def _assert_profile_not_shadowed() -> None:
    """Refuse to start Stata when a stray ``profile.py`` shadows the stdlib.

    IPython (imported by pystata) does ``import profile`` during startup.  The
    interpreter searches ``sys.path`` in order, and the script/current directory
    comes *before* the standard library, so a ``profile.py`` sitting there wins
    and IPython dies with a confusing error far from the real cause.  Checking
    the effective search path -- rather than hardcoding ``%TEMP%`` -- covers the
    documented failure mode and any future copy of it.
    """
    stdlib = Path(sysconfig.get_paths()["stdlib"]).resolve()
    stdlib_index = len(sys.path)
    for index, entry in enumerate(sys.path):
        try:
            resolved = Path(entry or ".").resolve()
        except OSError:
            continue
        if resolved == stdlib:
            stdlib_index = index
            break
    for entry in sys.path[:stdlib_index]:
        candidate = Path(entry or ".") / "profile.py"
        if candidate.is_file():
            raise RuntimeError(
                "refusing to start pystata: "
                f"{candidate} shadows the standard-library 'profile' module and "
                "breaks the IPython import that pystata performs. Remove that "
                "file or run from the repository root instead."
            )
    module = sys.modules.get("profile")
    origin = getattr(module, "__file__", None) if module is not None else None
    if origin:
        try:
            Path(origin).resolve().relative_to(stdlib)
        except ValueError as exc:
            raise RuntimeError(
                "the 'profile' module was imported from "
                f"{origin}, which is not the standard library; a stray "
                "profile.py broke the import path before Stata started"
            ) from exc


def _started_module(edition: str) -> Any:
    """Return ``pystata.stata``, initialising the process-wide session once."""
    global _INITIALISED
    with _INIT_LOCK:
        utilities = str(_UTILITIES)
        if utilities not in sys.path:
            sys.path.insert(0, utilities)
        if _INITIALISED:
            from pystata import stata

            return stata
        _assert_profile_not_shadowed()
        from pystata import config

        try:
            config.init(edition, splash=False)
        except Exception as exc:  # already initialised by another harness in-process
            if "init" not in str(exc).lower():
                raise
        _INITIALISED = True
        from pystata import stata

        return stata


class StataSession:
    """One handle on the process-wide embedded Stata.

    ``__enter__`` starts Stata if this process has not started it yet and
    returns the handle.  ``__exit__`` only detaches: pystata has no shutdown
    entry point, so the dataset in memory and the last ``e()`` results survive
    the ``with`` block.  Load the data you need at the start of every block
    instead of assuming an empty Stata.
    """

    def __init__(self, edition: str = "mp") -> None:
        self.edition = edition
        self._stata: Any = None
        self.log: list[str] = []

    def __enter__(self) -> StataSession:
        self.start()
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.close()
        return False

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        """Start (or attach to) the embedded Stata."""
        self._stata = _started_module(self.edition)

    def close(self) -> None:
        """Drop this handle.  The Stata process keeps running."""
        self._stata = None

    @property
    def raw(self) -> Any:
        """The ``pystata.stata`` module, for the rare escape hatch."""
        if self._stata is None:
            raise RuntimeError("StataSession is not started; use it as a context manager")
        return self._stata

    @property
    def version(self) -> str:
        """Stata version string, e.g. ``"19.5"``."""
        return self.macro("c(stata_version)")

    # -- running -----------------------------------------------------------
    def run(self, command: str) -> None:
        """Run one command and raise :class:`StataError` if Stata fails.

        pystata raises ``SystemError`` for a Stata error, which is easy to
        mistake for an interpreter bug; converting it here keeps the Stata
        return code and the failing command together in one message.
        """
        text = command.strip()
        if not text:
            raise ValueError("refusing to run an empty Stata command")
        self.log.append(text)
        try:
            self.raw.run(text, echo=False)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            message = str(exc).strip()
            match = _RCODE_RE.search(message)
            rcode = int(match.group("code")) if match else -1
            raise StataError(text, rcode, message) from exc

    def quiet(self, command: str) -> None:
        """Run a command with output suppressed, unless it already is."""
        head = command.lstrip().split(None, 1)[0].lower() if command.strip() else ""
        if head in {"quietly", "quiet", "qui", "capture", "captu", "capt"}:
            self.run(command)
        else:
            self.run(f"quietly {command}")

    def output(self, command: str) -> str:
        """Run a command and return what it printed.

        pystata writes to ``sys.stdout`` from the C bridge, so redirecting the
        Python-level stdout is enough to capture the table.
        """
        self.log.append(command.strip())
        buffer = StringIO()
        with redirect_stdout(buffer):
            self.run(command)
        return buffer.getvalue()

    # -- capture -----------------------------------------------------------
    def capture(self, command: str = "") -> Capture:
        """Read the current ``e()`` results into a :class:`Capture`.

        Fails rather than guesses when ``e(b)`` has no column names or when
        ``e(V)`` does not match ``e(b)``: both would otherwise let a
        comparison line up the wrong numbers, which is precisely the class of
        bug the harness exists to prevent.
        """
        raw_b = self._matrix("e(b)")
        if raw_b is None:
            raise StataError(
                command or "e(b)",
                -1,
                "no estimation results in memory; run an estimation command first",
            )
        b = screen_missing(np.asarray(raw_b, dtype=float).reshape(-1))
        names = self._matrix_names("e(b)")
        if not names:
            raise StataError(
                command or "e(b)",
                -1,
                f"e(b) holds {b.size} values but Stata reports no column names",
            )
        if len(names) != b.size:
            raise StataError(
                command or "e(b)",
                -1,
                f"e(b) holds {b.size} values but {len(names)} column names; "
                "refusing to pair coefficients with the wrong labels",
            )
        V = self._matrix("e(V)")
        if V is not None:
            V = screen_missing(np.asarray(V, dtype=float))
            if V.shape != (b.size, b.size):
                raise StataError(
                    command or "e(V)",
                    -1,
                    f"e(V) has shape {V.shape} but e(b) has {b.size} entries",
                )
        scalars, macros = self._ereturn()
        data_nobs = int(self.raw.sfi.Data.getObsTotal())
        nobs = data_nobs
        if "N" in scalars and math.isfinite(scalars["N"]):
            nobs = int(round(scalars["N"]))
        return Capture(
            b=b,
            V=V,
            names=list(names),
            scalars=scalars,
            macros=macros,
            nobs=nobs,
            command=command.strip(),
            data_nobs=data_nobs,
        )

    def estimate(self, command: str, *, quiet: bool = True) -> Capture:
        """Run an estimation command and capture its results in one step.

        ``ereturn clear`` runs first on purpose.  Stata keeps the previous
        estimation's ``e()`` in memory across commands that are not estimations,
        so without the clear a failed or non-estimating command would hand back
        the *previous* model's coefficients and the harness would compare the
        wrong regression -- silently.  Clearing turns that into a loud "no
        estimation results in memory".
        """
        try:
            self.raw.run("ereturn clear", echo=False)
        except BaseException:
            pass
        self.quiet(command) if quiet else self.run(command)
        return self.capture(command)

    def _matrix(self, name: str) -> np.ndarray | None:
        try:
            return np.asarray(self.raw.sfi.Matrix.get(name), dtype=float)
        except Exception:
            return None

    def _matrix_names(self, name: str) -> list[str]:
        try:
            return [str(item) for item in self.raw.sfi.Matrix.getColNames(name)]
        except Exception:
            return []

    def _ereturn(self) -> tuple[dict[str, float], dict[str, str]]:
        """Split ``e()`` into numeric scalars and string macros."""
        scalars: dict[str, float] = {}
        macros: dict[str, str] = {}
        try:
            raw = self.raw.get_ereturn()
        except Exception:
            return scalars, macros
        for key, value in raw.items():
            name = key[2:-1] if key.startswith("e(") and key.endswith(")") else key
            if value is None or isinstance(value, (bool, np.bool_)):
                continue
            if isinstance(value, str):
                macros[name] = value
            elif isinstance(value, (int, float, np.integer, np.floating)):
                number = float(value)
                scalars[name] = math.nan if number >= _MISSING_FLOOR else number
        return scalars, macros

    def scalar(self, name: str) -> float | None:
        """Read ``e(name)``/``r(name)``/``c(name)``; ``None`` when absent."""
        try:
            value = self.raw.sfi.Scalar.getValue(name)
        except Exception:
            return None
        if value is None:
            return None
        number = float(value)
        if math.isnan(number) or number >= _MISSING_FLOOR:
            return None
        return number

    def macro(self, name: str) -> str:
        try:
            return str(self.raw.sfi.Macro.getGlobal(name))
        except Exception:
            return ""

    # -- data --------------------------------------------------------------
    def load_dta(self, path: str | Path, *, clear: bool = True) -> Path:
        """Load a ``.dta`` file, validating it exists before Stata sees it."""
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"dataset not found: {resolved}")
        # Stata accepts forward slashes on Windows; backslashes inside a quoted
        # path are fine too but do not survive the double quotes on all builds.
        self.run(f'use "{resolved.as_posix()}"{", clear" if clear else ""}')
        return resolved

    def push(self, frame: pd.DataFrame) -> None:
        """Send a DataFrame into Stata as the current dataset.

        Stata's variable-name grammar is checked here because pystata silently
        mangles or rejects illegal names, and a mangled name would make the
        later ``regress`` fail with a message about a variable the caller never
        wrote.
        """
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("push expects a pandas DataFrame")
        if frame.empty:
            raise ValueError("refusing to push an empty DataFrame")
        columns = [str(column) for column in frame.columns]
        illegal = [column for column in columns if not _STATA_NAME_RE.match(column)]
        if illegal:
            raise ValueError(
                "these column names are not legal Stata variable names: "
                + ", ".join(repr(column) for column in illegal)
            )
        if len(set(columns)) != len(columns):
            raise ValueError("DataFrame column names must be unique")
        self.raw.pdataframe_to_data(frame.reset_index(drop=True), force=True)

    def pull(self) -> pd.DataFrame:
        """Read the current Stata dataset into a DataFrame (missing -> ``NaN``)."""
        frame = self.raw.pdataframe_from_data()
        if frame is None:
            raise StataError("pdataframe_from_data", -1, "Stata returned no dataset")
        return screen_missing(frame)

    def nobs(self) -> int:
        """Number of rows currently in memory (not ``e(N)``)."""
        return int(self.raw.sfi.Data.getObsTotal())
