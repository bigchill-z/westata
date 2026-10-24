# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) — with
the caveat that `0.0.x` explicitly carries **no stability promise**.

---

## [0.0.2] — 修正 PyPI 页面上的安装说明

**这一版没有功能变更。** 唯一的差别是项目描述里的安装说明。

0.0.1 上传之后，四份文档（英文/中文 README、英文/中文指南）仍写着「尚未发布到 PyPI，请从仓库
安装」。而 PyPI 把项目描述**冻结在上传的元数据里**，改文档不会更新已发布页面，所以 0.0.1 的
PyPI 页面上一直是那句错误的话——它会让读者去克隆仓库，而其实 `pip install westata` 就能装。

0.0.2 的内容与 0.0.1 完全一致，只把安装说明改成实际状态，并把「想改库本身」单独列为开发路径。
两条命令在上传前都实测过：

```
pip install westata            -> 0.0.1
pip install "westata[arch]"    -> arch 8.0.0 + westata 0.0.1
```

同时补了一个防回归测试 `tests/test_release_state.py`：它**直接问 PyPI**，而不是比对字符串——
如果包已发布，任何文档都不得声称未发布，且必须给出 `pip install`；pyproject 的版本号可以领先于
已发布版本，但不得落后。这样以后发新版本时测试自动跟着变。

## [Unreleased]

Nothing yet.

---

## [0.0.1] — first release

> **发布事实。** 这一版确实上传到了 PyPI 正式渠道，但**并非按计划发布**：执行者本意是发
> TestPyPI 做流程验证，而手上的 token 属于 pypi.org 而非 test.pypi.org，认证通过后 twine 直接
> 传到了正式渠道。更关键的是，首次收到 403 之后改用正式端点做"认证试探"——twine 没有 dry-run，
> 任何调用都是真实上传。
>
> **影响。** `0.0.1` 这个版本号被永久占用（PyPI 不允许删除或覆盖）。**内容本身没有问题**：
> README 顶部就是实测偏差表，许可证、依赖声明、测试与 CI 均完整。因此保留 `0.0.1`，后续修复走
> `0.0.2`。
>
> **为什么记在这里。** 发布记录应当说明"这一版是怎么出去的"。让后来看到 `0.0.1` 的人自己猜，
> 不如直接写明。同时这也是一条流程备忘：**在确认渠道之前，不要用真实 token 调用 twine。**



The first tagged version. Everything below is the initial state of the library,
not a diff against an earlier release.

### What this library is

`westata` reimplements a set of Stata estimation commands with a
statsmodels-style API, and checks each one against a real Stata installation.
The point is not "similar numbers" — it is a measured, reproducible claim about
which commands agree and how closely.

Verification: **StataNow 19.5 (MP)** via `pystata`. Captures are committed, so
the comparisons replay without a licence.

### Coverage

| | |
|---|---|
| Registry entries | 87 |
| Tests | 1434 |
| Independent comparison points | 386 (against live Stata) |
| Evidence level E3 (machine precision) | 52 |
| Evidence level E2 (bounded, measured) | 26 |
| Evidence level E1 (deviates or uncovered) | 9 |

Domains: linear regression and IV, discrete choice and count models, panel
data, limited-dependent and quantile models, time series, systems of equations
and causal inference, and post-estimation.

### Known limitations — READ THIS BEFORE RELYING ON A NUMBER

Nine registry entries measurably deviate on at least one invocation. Each is
listed with its worst measured relative difference and the reason, because a
number without a reason is not actionable.

| Command | Worst measured rel. diff. | Reason |
|---|---|---|
| `xtgls` | 3.4e+00 | **Only when the cross-equation Sigma is rank-deficient** (more equations than periods). No unique generalised inverse exists; Stata warns here too. Well-posed systems are at machine precision. |
| `teffects nnmatch` | 7.1e-02 | **Standard error only** — the point estimate is exact (3.6e-15). Relies on compiled Mata internally. |
| `frontier` | 4.3e-02 | `distribution(exponential)` only, where `sigma_u` is nearly unidentified on the test data. The likelihood at Stata's estimates matches exactly. |
| `bsqreg` | 3.0e-02 | Bootstrap VCE. Coefficients are exact (3.4e-16); a resampling estimator has not converged at any finite replicate count. |
| `permute` | 1.2e-02 | Stata's shuffle order is not reproducible from its output. |
| `arima` | 5.5e-03 | Optimiser stopping point; our likelihood is *higher* than Stata's. |
| `rreg` | 1.0e-03 | Final biweight case weights. |
| `xtlogit, pa` | 1.1e-03 | GEE working-correlation handling. |
| `heckman` | 1.9e-05 | Two-step standard errors. The ML path is a separate and exact estimate. |

**One capability is absent rather than approximate:** `teffects nnmatch,
biasadj()` raises `UnsupportedError`.

Some of these gaps are provably irreducible rather than unfinished:

- `xtgls` with a rank-deficient Sigma has no unique answer to converge to.
- `teffects nnmatch` and parts of `qreg` call compiled Stata bytecode with no
  readable source.
- `arima`, the count family and `frontier` stop where Stata stops; matching
  Stata's stopping point would mean reporting a *worse* optimum than the one
  found.
- A bootstrap VCE cannot match pointwise at any finite replicate count.

### Dependencies

- Required: `numpy`, `pandas`, `scipy`, `statsmodels`, `patsy`, `pyfixest`
  (HDFE), `linearmodels` (IV and panel), `pyyaml`.
  The last three are **not optional** — the package does not import without
  them. An earlier build declared them nowhere and installed cleanly while
  raising `ModuleNotFoundError` on import.
- Optional: `arch` for GARCH and the Phillips–Perron test. Install with
  `pip install "westata[arch]"`. Without it those commands raise
  `UnsupportedError` naming the install command, and their tests skip rather
  than fail.

Python 3.10 or newer.

### Notes on this release

- Version `0.0.1` is a placeholder in the SemVer sense: the API may change
  without notice. It exists so the packaging pipeline can be exercised
  end to end.
- The evidence levels are the project's central artefact. They are stored in
  `tests/conformance/registry.yml`, graded by an independent sweep in `retest/`,
  and a registry level that the measurement does not support causes the sweep to
  fail. A false claim is treated as a defect.
- `retest/` requires a Stata licence and does not run in CI. CI runs the test
  suite, which replays committed captures and needs no licence.
