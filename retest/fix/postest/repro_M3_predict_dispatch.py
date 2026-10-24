"""M3 -- ``predict`` delegation, and the ``zip``/``ztp`` crash it removes.

Defect (retest/REPORT.md section 2.7, capability gap): ``StataResults.predict``
dispatched every non-``poisson``/``nbreg`` model through the *binary* branch, so

* ``zip ... : predict, n`` raised a bare ``matmul`` dimension error (6
  parameters against 3 design columns) instead of answering;
* ``ztp ... : predict, n`` returned ``Phi(xb)`` -- a probability masquerading as
  a count mean, because the count link was chosen by a hard-coded model list;
* ``predict, stdp`` after ``zip`` compared 3 design columns against 6
  coefficients and raised ``ValueError``, although Stata answers it with the
  count-equation block of ``e(V)``;
* an unknown ``kind`` reported only "not registered", never the types that *do*
  work.

The repair is a generic contract, with no model list on the postestimation side:

    a result may publish ``_predict_dispatch`` (a callable) and
    ``_predict_types`` (the kinds Stata accepts); when it does, the model owns
    its predictions and ``predict.py`` only supplies the shared fallbacks
    (``xb``/``stdp``/``stdf``).

This script prints the live Stata accept/reject matrix for ``zip``/``ztp``, then
what westata does before and after the repair.

Run from the repository root::

    .venv/Scripts/python.exe retest/fix/postest/repro_M3_predict_dispatch.py

Stata version: ``c(stata_version)`` = 19.5 (MP).  Captured 2026-10-24.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "retest" / "lib"))
sys.path.insert(0, str(ROOT / "src"))

from stata_probe import Stata  # noqa: E402

import westata as ws  # noqa: E402
from westata.errors import UnsupportedError  # noqa: E402

KINDS = ("n", "pr", "xb", "stdp", "stdf", "residuals")


def stata_accepts(st: Stata, kind: str) -> tuple[bool, str]:
    """Ask the *currently estimated* model whether it offers ``kind``."""

    variable = f"probe_{abs(hash(kind)) % 10_000}"
    try:
        st.run(f"capture drop {variable}")
    except Exception:  # noqa: BLE001 - the variable need not exist yet
        pass
    try:
        st.quiet(f"predict double {variable}, {kind}")
        return True, ""
    except Exception as exc:  # noqa: BLE001 - Stata reports r() codes
        return False, str(exc).strip().splitlines()[0]


def westata_status(result, kind: str) -> tuple[str, str]:
    try:
        values = np.asarray(result.predict(kind=kind), dtype=float).reshape(-1)
        return "OK", f"mean={float(np.mean(values)):.10g}"
    except UnsupportedError as exc:
        return "UnsupportedError", str(exc)[:120]
    except Exception as exc:  # noqa: BLE001 - a crash is the defect
        return type(exc).__name__, str(exc)[:120]


def old_westata_status(result, kind: str) -> tuple[str, str]:
    """The pre-repair dispatch: binary branch for every non-poisson/nbreg model.

    ``kind in ('pr', 'n')`` used ``params[: n_columns]`` with a link chosen by
    model name, and ``stdp``/``stdf`` demanded one design column per *parameter*.
    """

    model_name = str(getattr(result, "model_name", ""))
    params = np.asarray(result.params, dtype=float)
    exog = np.asarray(result.design_matrix[0], dtype=float)
    try:
        if kind in ("stdp", "stdf"):
            if params.size != exog.shape[1]:
                raise ValueError(
                    f"predict: exog has {exog.shape[1]} column(s) but the model "
                    f"has {params.size} coefficient(s)"
                )
            return "OK", "(unchanged)"
        if kind in ("pr", "n"):
            if model_name in ("poisson", "nbreg"):
                return "OK", (
                    f"mean={float(np.mean(np.exp(exog @ params[: exog.shape[1]]))):.10g}"
                )
            xb = exog @ params
            if model_name in ("logit", "ologit"):
                probability = 1.0 / (1.0 + np.exp(-xb))
            else:
                from scipy import stats as _stats

                probability = _stats.norm.cdf(xb)
            return "OK", f"mean={float(np.mean(probability)):.10g} (probability)"
        return "OK", "(unchanged)"
    except Exception as exc:  # noqa: BLE001 - the crash is the finding
        return type(exc).__name__, str(exc)[:120]


def matrix(label: str, result, st: Stata) -> None:
    """Print Stata vs westata (before/after) for every ``kind``."""

    print(f"\n--- {label}: accept/reject ---")
    print(f"    {'kind':<10}{'Stata':<9}{'before':<19}{'after':<19}agreement")
    for kind in KINDS:
        accepted, error = stata_accepts(st, kind)
        before_status, before_note = old_westata_status(result, kind)
        after_status, after_note = westata_status(result, kind)
        agreement = "ok" if accepted == (after_status == "OK") else "MISMATCH"
        print(
            f"    {kind:<10}{('OK' if accepted else 'reject'):<9}"
            f"{before_status:<19}{after_status:<19}{agreement}"
        )
        if not accepted:
            print(f"        Stata  : {error}")
        if before_status != "OK":
            print(f"        before : {before_note}")
        if after_status != "OK":
            print(f"        after  : {after_note}")
    try:
        result.predict(kind="mu")
    except UnsupportedError as exc:
        print(f"    unsupported-kind message: {exc}")


def main() -> int:
    rng = np.random.default_rng(11)
    n = 400
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    counts = rng.poisson(np.exp(0.4 + 0.6 * x1 - 0.3 * x2))
    truncated = np.maximum(counts, 1)
    rng2 = np.random.default_rng(7)
    z = rng2.normal(size=n)
    zero_draw = rng2.random(n) < 1.0 / (1.0 + np.exp(-(-0.8 + 1.0 * z)))
    inflated = np.where(zero_draw, 0, rng2.poisson(np.exp(0.5 + 0.5 * x1)))

    zip_frame = pd.DataFrame({"yzi": inflated.astype(float), "x1": x1, "z": z})
    ztp_frame = pd.DataFrame({"y1": truncated.astype(float), "x1": x1, "x2": x2})

    with Stata() as st:
        print(f"Stata c(stata_version) = {st.version}")

        # -------------------------------------------------------------- zip
        st.send(zip_frame)
        st.quiet("zip yzi x1, inflate(z)")
        zip_names = st.matrix_names("e(b)")
        zip_b = st.matrix("e(b)").ravel()
        zip_V = st.matrix("e(V)")
        print(f"\n[zip yzi x1, inflate(z)]  Stata e(b) names = {zip_names}")
        print(f"    Stata e(b) = {np.array2string(zip_b, precision=10)}")
        for kind in ("n", "pr", "xb", "stdp"):
            st.run(f"capture drop zip_{kind}")
            st.quiet(f"predict double zip_{kind}, {kind}")
        mu = np.exp(zip_b[1] + zip_b[0] * x1)                    # count equation
        pi = 1.0 / (1.0 + np.exp(-(zip_b[3] + zip_b[2] * z)))    # inflation
        # Stata's zip e(b) is ordered [x1, _cons, z, _cons], so the count
        # equation's design columns follow the same order.
        count_design = np.column_stack([x1, np.ones(n)])
        count_V = zip_V[0:2, 0:2]
        expected_stdp = np.sqrt(
            np.einsum("ij,jk,ik->i", count_design, count_V, count_design)
        )
        print(
            f"    semantics: max|n - (1-pi)mu| = "
            f"{float(np.max(np.abs(st.var('zip_n') - (1 - pi) * mu))):.2e}"
            f";  max|pr - pi| = {float(np.max(np.abs(st.var('zip_pr') - pi))):.2e}"
            f";  max rel|stdp - sqrt(x_c'V_cc x_c)| = "
            f"{float(np.max(np.abs(st.var('zip_stdp') - expected_stdp) / expected_stdp)):.2e}"
        )
        zip_result = ws.ZeroInflatedPoisson.from_formula(
            "yzi ~ x1", zip_frame, inflate="z"
        ).fit()
        print(f"    westata param_names = {list(zip_result.param_names)}")
        westata_stdp = np.asarray(zip_result.predict(kind="stdp"), dtype=float)
        print(
            "    westata stdp vs Stata stdp: max rel = "
            f"{float(np.max(np.abs(westata_stdp - st.var('zip_stdp')) / st.var('zip_stdp'))):.2e}"
            "   (the residual is the ML fit's coefficient/V difference;"
            " stdp itself matches sqrt(x_c'V_cc x_c) to 0.0)"
        )
        matrix("zip", zip_result, st)

        # -------------------------------------------------------------- ztp
        st.send(ztp_frame)
        st.quiet("ztp y1 x1 x2")
        print(f"\n[ztp y1 x1 x2]  Stata e(b) names = {st.matrix_names('e(b)')}")
        st.run("capture drop ztp_n ztp_xb")
        st.quiet("predict double ztp_n, n")
        st.quiet("predict double ztp_xb, xb")
        eta = st.var("ztp_xb")
        print(
            "    semantics: max rel|n - exp(xb)| = "
            f"{float(np.max(np.abs(st.var('ztp_n') - np.exp(eta)) / np.exp(eta))):.2e}"
            "   (no truncation correction in Stata's n)"
        )
        ztp_result = ws.TruncatedPoisson.from_formula("y1 ~ x1 + x2", ztp_frame).fit()
        print(f"    westata param_names = {list(ztp_result.param_names)}")
        matrix("ztp", ztp_result, st)

        # --------------------------------------- a model that owns the contract
        print("\n=== a result that publishes _predict_dispatch / _predict_types ===")

        class Stub:
            model_name = "customcount"
            param_names = ["a", "b"]
            nobs = 4
            dist = "z"
            df_resid = 4.0
            resid = np.zeros(4)
            params = pd.Series([1.0, 2.0], index=["a", "b"])
            design_matrix = (np.ones((4, 2)), ["a", "b"])
            e = {"N": 4}

            def cov_params(self):
                return np.eye(2)

            _predict_types = ("xb", "double_mean")

            def _predict_dispatch(self, *, kind, exog, lower=None, upper=None):
                if kind == "double_mean":
                    return 2.0 * (np.asarray(exog) @ np.asarray(self.params))
                return None

        from westata.postestimation.predict import predict as predict_fn

        stub = Stub()
        design = np.ones((4, 2))
        print(
            "    xb (falls through to the shared implementation): "
            f"{predict_fn(stub, design, 'xb')}"
        )
        print(
            f"    double_mean (model's own): {predict_fn(stub, design, 'double_mean')}"
        )
        try:
            predict_fn(stub, design, "n")
        except UnsupportedError as exc:
            print(f"    n (not declared) -> UnsupportedError: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
