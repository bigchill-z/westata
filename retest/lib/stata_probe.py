"""Fresh, independent Stata-vs-westata comparison harness (2026-10-24 retest).

This module is the *only* shared contract for the retest.  It was written from
scratch for this sweep; it does not read the committed fixtures under
``tests/conformance/stata_fixtures`` and does not consult
``tests/conformance/registry.yml``.  Every number it reports comes from a live
StataNow 19.5 (MP) session started here.

Design rules
------------
* One embedded Stata session per process.  ``Stata`` is a context manager.
* Missing values cross the boundary as ``8.988e+307`` sentinels; they are
  converted to ``NaN`` on the way in and out, so a comparison never silently
  treats a missing as a huge number.
* Coefficient alignment is explicit.  Stata's ``e(b)`` names factor levels as
  ``1.g`` / ``2.g`` with the base level ``0b.g`` present and zero; patsy names
  them ``g[T.1]`` / ``g[T.2]`` and omits the base.  ``align`` maps both onto a
  canonical vocabulary and reports anything it cannot match instead of guessing.

Typical use::

    with Stata() as st:
        st.run("sysuse auto, clear")
        cap = st.estimate("regress mpg weight i.foreign")
        res = ws.OLS.from_formula("mpg ~ weight + C(foreign)", frame).fit()
        comp = compare("regress i.foreign", cap, res)

    Record("linear").add(comp)
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

STATA_HOME = Path(r"D:\Program Files\StataNow19")
STATA_VERSION_EXPECTED = "19.5"
MISSING_SENTINEL = 8.98846567431158e307

HERE = Path(__file__).resolve()
RETEST_ROOT = HERE.parents[1]
REPO_ROOT = HERE.parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# --------------------------------------------------------------------- session
class Stata:
    """One embedded Stata session plus the capture helpers the sweep needs."""

    def __init__(self, edition: str = "mp") -> None:
        self.edition = edition
        self.stata = None
        self.version = ""
        self.log: list[str] = []

    # -- lifecycle ---------------------------------------------------------
    def __enter__(self) -> "Stata":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def start(self) -> None:
        utilities = str(STATA_HOME / "utilities")
        if utilities not in sys.path:
            sys.path.insert(0, utilities)
        from pystata import config

        config.init(self.edition, splash=False)
        from pystata import stata as _stata

        self.stata = _stata
        self.version = self.macro("c(stata_version)")

    def close(self) -> None:
        self.stata = None

    # -- running -----------------------------------------------------------
    def run(self, command: str) -> None:
        """Run one command; raise on Stata error."""
        self.log.append(command)
        self.stata.run(command, echo=False)

    def quiet(self, command: str) -> None:
        self.run(command if command.startswith("quietly") else f"quietly {command}")

    def output(self, command: str) -> str:
        """Run a command and return its printed output."""
        import io
        from contextlib import redirect_stdout

        self.log.append(command)
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.stata.run(command, echo=False)
        return buf.getvalue()

    # -- r()/e() accessors -------------------------------------------------
    def scalar(self, name: str) -> float | None:
        """Read ``r(name)``/``e(name)``/``c(name)`` scalar (None if absent)."""
        try:
            value = self.stata.sfi.Scalar.getValue(name)
        except Exception:
            return None
        if value is None:
            return None
        value = float(value)
        if math.isnan(value) or value >= MISSING_SENTINEL * 0.5:
            return None
        return value

    def rscalar(self, name: str) -> float | None:
        return self.scalar(name)

    def macro(self, name: str) -> str:
        try:
            return str(self.stata.sfi.Macro.getGlobal(name))
        except Exception:
            return ""

    def matrix(self, name: str) -> np.ndarray | None:
        try:
            return np.asarray(self.stata.sfi.Matrix.get(name), dtype=float)
        except Exception:
            return None

    def matrix_names(self, name: str) -> list[str]:
        try:
            return [str(x) for x in self.stata.sfi.Matrix.getColNames(name)]
        except Exception:
            return []

    # -- data --------------------------------------------------------------
    def send(self, frame: pd.DataFrame) -> None:
        self.stata.pdataframe_to_data(frame, force=True)

    def frame(self) -> pd.DataFrame:
        out = self.stata.pdataframe_from_data()
        return out.replace(MISSING_SENTINEL, np.nan)

    def var(self, name: str, *, n: int | None = None) -> np.ndarray:
        """Read one Stata variable as a float array."""
        raw = self.stata.sfi.Data.get(name)
        arr = np.asarray(raw, dtype=float).reshape(-1)
        if n is not None and arr.size != n:
            raise ValueError(f"{name}: expected {n} values, got {arr.size}")
        arr = arr.copy()
        arr[arr >= MISSING_SENTINEL * 0.5] = np.nan
        return arr

    def var_str(self, name: str) -> list[str]:
        return [str(x) for x in self.stata.sfi.Data.get(name)]

    def nobs(self) -> int:
        return int(self.stata.sfi.Data.getObsTotal())

    def keep(self, varlist: Sequence[str]) -> None:
        self.quiet("keep " + " ".join(varlist))

    # -- estimation capture ------------------------------------------------
    def estimate(self, command: str, *, quiet: bool = True) -> "Capture":
        """Run an estimation command and capture ``e(b)``/``e(V)``/scalars."""
        self.run(("quietly " + command) if quiet else command)
        b_raw = self.matrix("e(b)")
        if b_raw is None:
            raise RuntimeError(f"{command!r} did not leave e(b)")
        V = self.matrix("e(V)")
        b = b_raw.reshape(-1)
        names = self.matrix_names("e(b)")
        if len(names) != b.size:
            names = [f"b{i}" for i in range(b.size)]
        if V is not None and V.size != b.size * b.size:
            V = None
        # Base levels appear in e(b) with value exactly 0 and name <level>b.<var>.
        keep_idx = [i for i, nm in enumerate(names) if not is_stata_base(nm)]
        b = b[keep_idx]
        names = [names[i] for i in keep_idx]
        if V is not None:
            V = V[np.ix_(keep_idx, keep_idx)]
        return Capture(
            command=command,
            b=b,
            V=V if V is not None else np.zeros((b.size, b.size)),
            names=names,
            scalars=self.ereturn_scalars(),
            macros=self.ereturn_macros(),
            nobs=self.nobs(),
            version=self.version,
        )

    def ereturn_scalars(self) -> dict[str, float]:
        out: dict[str, float] = {}
        try:
            raw = self.stata.get_ereturn()
        except Exception:
            return out
        for key, value in raw.items():
            if key in ("e(b)", "e(V)"):
                continue
            if value is None or isinstance(value, str):
                continue
            if isinstance(value, (int, float, np.floating, np.integer)):
                out[key[2:-1] if key.startswith("e(") else key] = float(value)
        return out

    def ereturn_macros(self) -> dict[str, str]:
        out: dict[str, str] = {}
        try:
            raw = self.stata.get_ereturn()
        except Exception:
            return out
        for key, value in raw.items():
            if isinstance(value, str):
                out[key[2:-1] if key.startswith("e(") else key] = value
        return out

    def test_command(self, command: str) -> dict[str, float]:
        """Run a ``test``/``lincom``-style command and capture ``r()``."""
        self.quiet(command)
        return {
            n: v
            for n in ("chi2", "df", "p", "F", "r2", "df_r")
            if (v := self.scalar(f"r({n})")) is not None
        }


# ------------------------------------------------------------------- capture
@dataclass
class Capture:
    command: str
    b: np.ndarray
    V: np.ndarray
    names: list[str]
    scalars: dict[str, float] = field(default_factory=dict)
    macros: dict[str, str] = field(default_factory=dict)
    nobs: int = 0
    version: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------- alignment
_BASE_RE = re.compile(r"^(-?\d+(?:\.\d+)?)b\.(.+)$")
_LEVEL_RE = re.compile(r"^(-?\d+(?:\.\d+)?)\.(.+)$")
_OMIT_RE = re.compile(r"^(-?\d+(?:\.\d+)?)o\.(.+)$")
_PATSY_RE = re.compile(r"^(?P<var>.+?)\[T\.(?P<lev>.+)\]$")


def is_stata_base(name: str) -> bool:
    return bool(_BASE_RE.match(name))


def _norm_level(text: str) -> str:
    """``"2.0"`` and ``"2"`` name the same factor level."""
    try:
        val = float(text)
    except (TypeError, ValueError):
        return text
    if val.is_integer():
        return str(int(val))
    return repr(val)


def canon_stata(name: str) -> str:
    if name in ("_cons", "const", "Intercept"):
        return "__const__"
    m = _LEVEL_RE.match(name)
    if m:
        return f"{m.group(2)}={_norm_level(m.group(1))}"
    m = _OMIT_RE.match(name)
    if m:
        return f"{m.group(2)}=__omitted__"
    return name


_C_WRAP_RE = re.compile(r"^C\((?P<var>[^,()]+)(?:,[^)]*)?\)$")


def canon_westata(name: str) -> str:
    if name in ("Intercept", "_cons", "const"):
        return "__const__"
    m = _PATSY_RE.match(name)
    if m:
        var = m.group("var")
        wrapped = _C_WRAP_RE.match(var)
        if wrapped:
            var = wrapped.group("var")
        lev = m.group("lev").strip("\"'")
        return f"{var}={_norm_level(lev)}"
    return name


def align(
    stata_names: Sequence[str],
    westata_names: Sequence[str],
    *,
    rename: dict[str, str] | None = None,
) -> tuple[list[str], list[int], list[int], list[str]]:
    """Match coefficient names; return (names, stata_idx, westata_idx, unmatched).

    ``rename`` maps a Stata name onto the westata name before canonicalisation,
    for the rare case where the two sides genuinely use different labels.
    """
    rename = rename or {}
    smap: dict[str, int] = {}
    for i, nm in enumerate(stata_names):
        smap.setdefault(canon_stata(rename.get(nm, nm)), i)
    # Stata stores a log-likelihood scale as ``var(e.y) = sigma^2``; westata
    # reports ``/sigma``.  Match them on a transformed scale only when the
    # caller asked for it through ``rename``; never silently here.
    wkeys = {canon_westata(w) for w in westata_names}
    names: list[str] = []
    si: list[int] = []
    wi: list[int] = []
    unmatched: list[str] = []
    used: set[int] = set()
    for j, nm in enumerate(westata_names):
        key = canon_westata(nm)
        if key in smap and smap[key] not in used:
            i = smap[key]
            used.add(i)
            names.append(key)
            si.append(i)
            wi.append(j)
        else:
            unmatched.append(nm)
    missing = [
        n
        for n in stata_names
        if canon_stata(rename.get(n, n)) not in wkeys
        and canon_stata(rename.get(n, n)) != "__omitted__"
    ]
    return names, si, wi, unmatched + [f"stata-only:{m}" for m in missing]


def rel_diff(actual: Sequence[float], expected: Sequence[float]) -> float:
    """Max relative difference, with an absolute floor for near-zero values."""
    a = np.asarray(actual, dtype=float).reshape(-1)
    e = np.asarray(expected, dtype=float).reshape(-1)
    if a.size != e.size:
        return float("inf")
    scale = np.maximum(np.abs(e), 1e-12)
    return float(np.max(np.abs(a - e) / scale))


def rel_pairs(a: float, e: float) -> float:
    if a is None or e is None:
        return float("nan")
    if not math.isfinite(a) or not math.isfinite(e):
        return float("nan")
    return abs(a - e) / max(abs(e), 1e-12)


# ---------------------------------------------------------------- comparison
TOLERANCES = {
    "A": 1e-9,    # machine precision territory
    "B": 1e-6,
    "C": 1e-3,
    "D": 1e-2,
}


def tier_for(value: float) -> str:
    if value != value:  # NaN
        return "NA"
    if value <= TOLERANCES["A"]:
        return "A"
    if value <= TOLERANCES["B"]:
        return "B"
    if value <= TOLERANCES["C"]:
        return "C"
    if value <= TOLERANCES["D"]:
        return "D"
    return "E"


@dataclass
class Comparison:
    domain: str
    name: str
    stata_command: str
    status: str                # pass | fail | error | skip
    tolerance: float = 1e-9
    param_rel: float = float("nan")
    se_rel: float = float("nan")
    n_params: int = 0
    mismatched: list[str] = field(default_factory=list)
    scalars: dict[str, dict[str, float]] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    error: str = ""
    stata_version: str = STATA_VERSION_EXPECTED
    seconds: float = 0.0

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["param_tier"] = tier_for(self.param_rel)
        d["se_tier"] = tier_for(self.se_rel)
        return d


def get_scalar(result: Any, key: str) -> float | None:
    """Read a Stata ``e()`` scalar off a westata results object."""
    e_map = getattr(result, "e", None)
    if e_map is not None:
        try:
            if key in e_map:
                return float(e_map[key])
        except Exception:
            pass
    try:
        attr = getattr(result, key)
        return float(attr() if callable(attr) else attr)
    except Exception:
        return None


def compare(
    name: str,
    cap: Capture,
    result: Any,
    *,
    domain: str = "",
    tolerance: float = 1e-9,
    se_tolerance: float | None = None,
    scalars: Iterable[str] = (),
    rename: dict[str, str] | None = None,
    stata_transform: dict[str, tuple[Any, Any]] | None = None,
    notes: Iterable[str] = (),
    elapsed: float = 0.0,
    extra: dict[str, Any] | None = None,
) -> Comparison:
    """Compare a westata results object with a captured Stata estimation.

    ``result`` only needs ``params``/``param_names``/``bse`` (a statsmodels-like
    object).  Scalars named in ``scalars`` are compared against ``result.e[x]``
    or the matching attribute.

    ``stata_transform`` handles the cases where the *same* quantity is stored in
    a different parameterisation on the two sides: a mapping from the canonical
    coefficient name to ``(value_fn, deriv_fn)``, where ``value_fn`` maps Stata's
    stored number onto westata's scale and ``deriv_fn`` is its derivative (used
    to carry the Standard Error across with the delta method).  Example: Stata's
    ``tobit`` stores ``var(e.y) = sigma^2`` while westata reports ``/sigma``::

        stata_transform={"/sigma": (np.sqrt, lambda x: 1/(2*np.sqrt(x)))}
    """
    se_tolerance = tolerance if se_tolerance is None else se_tolerance
    stata_transform = stata_transform or {}
    w_names = list(result.param_names)
    w_b = np.asarray(result.params, dtype=float)
    w_se = np.asarray(result.bse, dtype=float)
    names, si, wi, unmatched = align(cap.names, w_names, rename=rename)
    comp = Comparison(
        domain=domain,
        name=name,
        stata_command=cap.command,
        status="pass",
        tolerance=tolerance,
        n_params=len(names),
        notes=list(notes),
        extra=dict(extra or {}),
        stata_version=cap.version or STATA_VERSION_EXPECTED,
        seconds=elapsed,
    )
    hard_unmatched = [u for u in unmatched if "__omitted__" not in u]
    if hard_unmatched:
        comp.mismatched = hard_unmatched
        comp.notes.append("unmatched coefficients: " + ", ".join(hard_unmatched))
    if names:
        sb = np.asarray(cap.b, dtype=float)[si].astype(float).copy()
        sv = (
            np.sqrt(np.diag(cap.V))[si].astype(float).copy()
            if cap.V.size
            else np.full(len(si), np.nan)
        )
        for k, key in enumerate(names):
            pair = stata_transform.get(key)
            if pair is None:
                continue
            value_fn, deriv_fn = pair
            try:
                sv[k] = abs(float(deriv_fn(sb[k]))) * sv[k]
                sb[k] = float(value_fn(sb[k]))
            except Exception:  # leave the raw numbers rather than inventing one
                pass
        comp.param_rel = rel_diff(w_b[wi], sb)
        if not np.all(np.isnan(sv)):
            comp.se_rel = rel_diff(w_se[wi], sv)
    for key in scalars:
        s_val = cap.scalars.get(key)
        w_val = get_scalar(result, key)
        comp.scalars[key] = {
            "stata": float(s_val) if s_val is not None else float("nan"),
            "westata": float(w_val) if w_val is not None else float("nan"),
            "rel": rel_pairs(w_val, s_val),
        }
    bad = bool(comp.mismatched) or comp.param_rel > tolerance or comp.se_rel > se_tolerance
    for key, vals in comp.scalars.items():
        rel = vals["rel"]
        if rel == rel and rel > max(tolerance, 1e-8) and key not in ("ll", "ll_0"):
            bad = True
            comp.notes.append(f"scalar e({key}) rel diff {rel:.3e}")
    comp.status = "fail" if bad else "pass"
    return comp


# ------------------------------------------------------------------ recorder
class Record:
    """Append-only JSONL recorder for one domain."""

    def __init__(self, domain: str, outdir: str | Path | None = None) -> None:
        self.domain = domain
        self.dir = Path(outdir) if outdir else RETEST_ROOT / "domains" / domain
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "results.jsonl"
        if self.path.exists():
            self.path.unlink()
        self.items: list[dict[str, Any]] = []

    def add(self, comp: Comparison) -> Comparison:
        comp.domain = comp.domain or self.domain
        payload = comp.to_json()
        self.items.append(payload)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")
        flag = {"pass": "ok ", "fail": "FAIL", "error": "ERR ", "skip": "skip"}[comp.status]
        print(
            f"[{flag}] {comp.name}: params {comp.param_rel:.3e} "
            f"({tier_for(comp.param_rel)}) se {comp.se_rel:.3e} "
            f"({tier_for(comp.se_rel)})",
            flush=True,
        )
        return comp

    def error(
        self,
        name: str,
        exc: BaseException,
        *,
        command: str = "",
        notes: Iterable[str] = (),
    ) -> Comparison:
        comp = Comparison(
            domain=self.domain,
            name=name,
            stata_command=command,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            notes=list(notes),
        )
        return self.add(comp)

    def skip(self, name: str, reason: str, *, command: str = "") -> Comparison:
        comp = Comparison(
            domain=self.domain, name=name, stata_command=command,
            status="skip", notes=[reason],
        )
        return self.add(comp)

    def flush_summary(self) -> Path:
        counts: dict[str, int] = {}
        for item in self.items:
            counts[item["status"]] = counts.get(item["status"], 0) + 1
        vals = [
            i["param_rel"]
            for i in self.items
            if i["param_rel"] == i["param_rel"] and i["status"] not in ("skip", "error")
        ]
        summary = {
            "domain": self.domain,
            "total": len(self.items),
            "counts": counts,
            "max_param_rel": max(vals) if vals else float("nan"),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "stata_version": STATA_VERSION_EXPECTED,
            "items": [
                {
                    "name": i["name"],
                    "status": i["status"],
                    "param_rel": i["param_rel"],
                    "se_rel": i["se_rel"],
                    "tier": i.get("param_tier"),
                    "se_tier": i.get("se_tier"),
                    "command": i.get("stata_command", ""),
                    "notes": i.get("notes", []),
                    "error": i.get("error", ""),
                }
                for i in self.items
            ],
        }
        path = self.dir / "summary.json"
        path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print(f"\n{self.domain}: {counts} -> {path}", flush=True)
        return path


def fmt(value: float, digits: int = 3) -> str:
    if value is None or value != value:
        return "n/a"
    return f"{value:.{digits}e}"
