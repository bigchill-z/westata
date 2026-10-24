"""捕获 `tests/test_discrete.py` / `tests/test_counts.py` 里要断言的那几条 Stata 数字。

数据直接复用测试文件的 fixture（`_binary`/`_count`/`_ordered`/`count_frame`/…），
所以测试里写下的常量与这里打印的 StataNow 19.5（MP）输出一一对应。

运行::

    cd D:\\myCodeProject\\statapy
    .venv\\Scripts\\python.exe retest/fix/discrete/capture_test_fixtures.py

本文件只读：不改任何源码与测试，只打印可复制进测试的常量与注释。
"""

from __future__ import annotations

import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "retest", "lib"))
sys.path.insert(0, os.path.join(REPO, "tests"))

import numpy as np

from stata_probe import Stata

warnings.simplefilter("ignore")

# --- the test modules' own fixtures, imported (not re-implemented) ----------
import test_discrete as TD  # noqa: E402
import test_counts as TC  # noqa: E402

CAPTURED = "2026-10-24"


def show(st, label: str, command: str, frame, *, keys=("df_m", "rank", "k_aux",
                                                        "chi2", "p", "N")):
    """Estimate on ``frame`` and print every number the tests may assert."""

    st.send(frame)
    cap = st.estimate(command)
    se = np.sqrt(np.diag(cap.V))
    print(f"\n### {label}")
    print(f"  command : {command}")
    print(f"  captured: {CAPTURED} (StataNow {st.version}, MP)")
    print(f"  names   : {cap.names}")
    print(f"  b       : {np.round(cap.b, 10).tolist()}")
    print(f"  se      : {np.round(se, 10).tolist()}")
    print(f"  e()     : " + ", ".join(
        f"{k}={cap.scalars[k]!r}" for k in keys if k in cap.scalars
    ))
    print(f"  chi2type: {cap.macros.get('chi2type')}")
    out = st.output(command)
    for line in out.splitlines():
        if "chi2(" in line or "Prob > chi2" in line:
            print(f"  printed : {line.strip()}")
    return cap


def main() -> int:
    print("=" * 78)
    print("A. logit 默认 vs vce(robust) / vce(cluster) 的模型检验标签")
    with Stata() as st:
        st.run("set type double")
        binary = TD._binary()
        show(st, "logit 默认（Stata 打印 LR chi2）", "logit y x1 x2", binary)
        show(st, "logit vce(robust)（Stata 打印 Wald chi2）",
             "logit y x1 x2, vce(robust)", binary)

    print("\n" + "=" * 78)
    print("B. e(df_m) 不计辅助参数：nbreg / oprobit / ologit")
    with Stata() as st:
        st.run("set type double")
        counts = TD._count()
        show(st, "nbreg（/lnalpha 不计入 df_m）", "nbreg y x1", counts,
             keys=("df_m", "rank", "k_aux", "k", "chi2", "p", "ll", "alpha"))
        ordered = TD._ordered()
        show(st, "ologit（3 个 cut 不计入 df_m）", "ologit y x1", ordered,
             keys=("df_m", "rank", "k_aux", "k", "chi2", "p", "ll"))
        show(st, "oprobit（3 个 cut 不计入 df_m）", "oprobit y x1", ordered,
             keys=("df_m", "rank", "k_aux", "k", "chi2", "p", "ll"))

    print("\n" + "=" * 78)
    print("C. vce(robust) 对 probit / nbreg / ologit / oprobit：SE 与 e(chi2)")
    with Stata() as st:
        st.run("set type double")
        binary = TD._binary()
        show(st, "probit vce(robust)", "probit y x1 x2, vce(robust)", binary)
        counts = TD._count()
        show(st, "nbreg vce(robust)", "nbreg y x1, vce(robust)", counts,
             keys=("df_m", "rank", "k_aux", "chi2", "p", "ll", "alpha"))
        ordered = TD._ordered()
        show(st, "ologit vce(robust)", "ologit y x1, vce(robust)", ordered,
             keys=("df_m", "rank", "k_aux", "chi2", "p", "ll"))
        show(st, "oprobit vce(robust)", "oprobit y x1, vce(robust)", ordered,
             keys=("df_m", "rank", "k_aux", "chi2", "p", "ll"))

    print("\n" + "=" * 78)
    print("D. e(p) 的自由度约定（zip 用 rank-1，其它用 df_m）")
    with Stata() as st:
        st.run("set type double")
        frame = TC.count_frame()
        show(st, "zip（默认 VCE：e(p) 用 rank-1）",
             "zip y x1, inflate(x1)", frame,
             keys=("df_m", "rank", "k_aux", "chi2", "p", "ll", "ll_0"))
        show(st, "nc 对照 logit（e(p) 用 df_m）", "logit y x1", frame,
             keys=("df_m", "rank", "chi2", "p", "ll"))

    print("\n" + "=" * 78)
    print("E. zip / ztp 的 predict 语义（n / pr / xb）")
    with Stata() as st:
        st.run("set type double")
        frame = TC.count_frame(n=400)
        st.send(frame)
        st.quiet("zip y x1, inflate(x1)")
        b = np.asarray(st.matrix("e(b)")).reshape(-1)
        for kind in ("n", "pr", "xb"):
            st.run(f"cap drop v_{kind}")
            st.run(f"predict double v_{kind}, {kind}")
            print(f"  zip predict, {kind}: head={np.round(st.var(f'v_{kind}')[:3], 10).tolist()}")
        print(f"  zip e(b) = {np.round(b, 10).tolist()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
