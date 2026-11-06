"""Check whether this Stata's `var lags()` can be trusted, and say so up front.

Rounds 39-47 found that `var ..., lags(k)` estimates ONLY the k-th lag here, and rounds 81-84
burned three rounds diagnosing against it. Any script that compares against a `var` reference
should establish its validity first rather than assume it.

The check is direct: fit var lags(2) on series with known structure and read e(b). A correct
VAR(2) over two variables stores 5 coefficients; a single-lag model stores 3.
"""

import contextlib
import io
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "replication")

import numpy as np
import pandas as pd

from lib.stata_session import StataSession


def coefficient_names(session: StataSession) -> list[str]:
    """The names Stata stored for e(b), via the matrix column names."""

    try:
        return session._matrix_names("e(b)")
    except Exception:  # pragma: no cover - defensive
        return []


def main() -> int:
    rng = np.random.default_rng(7)
    n = 200
    frame = pd.DataFrame(
        {
            "tt": np.arange(1, n + 1),
            "va": np.cumsum(rng.normal(size=n)),
            "vb": np.cumsum(rng.normal(size=n)),
        }
    )

    print("Stata `var` lag handling in this environment")
    print("-" * 62)
    print(f"{'command':28} {'e(b) len':>9}  lag names")
    verdict_lines: list[str] = []
    with StataSession() as session:
        session.push(frame)
        with contextlib.redirect_stdout(io.StringIO()):
            session.run("tsset tt")
        for command in ("var va vb", "var va vb, lags(1)", "var va vb, lags(2)", "var va vb, lags(3)"):
            with contextlib.redirect_stdout(io.StringIO()):
                session.run("quietly " + command)
            e = session.raw.get_ereturn()
            values = np.asarray(e.get("e(b)", []), dtype=float).ravel()
            names = coefficient_names(session)
            lags = sorted({n.split(".")[0] for n in names if n.startswith("L")})
            print(f"{command:28} {values.size:>9}  {lags}")
            verdict_lines.append(f"{command}: {values.size} coefficients, lags {lags}")

    print()
    # A VAR(k) over two variables stores 2 * (2*k + 1) coefficients: two equations, each with
    # k lags of both variables plus a constant.
    expected = {
        # No lags() option: Stata defaults to lags(2) here, which DOES work.
        "var va vb": 2 * (2 * 2 + 1),
        "var va vb, lags(1)": 2 * (2 * 1 + 1),
        "var va vb, lags(2)": 2 * (2 * 2 + 1),
        "var va vb, lags(3)": 2 * (2 * 3 + 1),
    }
    broken = False
    print("verdict")
    print("-" * 62)
    for line in verdict_lines:
        command, rest = line.split(":", 1)
        stored = int(rest.strip().split()[0])
        wanted = expected[command]
        status = "OK" if stored == wanted else "BROKEN"
        if status == "BROKEN":
            broken = True
        print(f"  {command:28} stored {stored:>2}, expected {wanted:>2}  {status}")

    print()
    if broken:
        print("WARNING: `var lags()` does NOT estimate the full lag structure here.")
        print("  A `var` reference must be rebuilt with hand-made lag columns and `regress`/`sureg`")
        print("  before it can be compared against. Rounds 39-47 and 81-84 were spent on this.")
        return 1
    print("`var lags()` behaves correctly; variable-based references are usable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
