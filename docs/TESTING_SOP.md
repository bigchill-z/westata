# westata 测试 SOP：让一个命令与 Stata 对齐

> 适用对象：任何"westata 的数值/输出与 Stata 不一致"的修复，以及任何新增命令的对标验证。
>
> 这份 SOP 来自一次真实的失败：`estat hettest` 的默认分支和 `method="iid"` 与 Stata 对不上。
> 失败的根因不是算术，而是**辅助回归用了哪些列**；而它之所以能长期潜伏，是因为
> **原来的测试拿"第二份手抄的 ado"当标准答案，而那份手抄犯了同样的误读**。

---

## 0. 一句话原则

**证据必须是 Stata 跑出来的数字，不能是对同一份 ado 的第二次阅读。**

手抄 ado 和实现犯同一个误读时，测试会通过、数字依然错。这是本项目最容易发生、也最难发现的失效方式。
唯一的解法是：参考值来自**真实 Stata 进程的实际输出**，并把它**固化入库**，让没有许可证的机器也能复现。

---

## 1. 触发条件

出现下列任一情况，走本 SOP：

- 用户报告 westata 与 Stata 结果不一致；
- `retest/` 扫描或 `scripts/scan_all_layouts.py` 报出 `SKELETON`/`DIFFERS`；
- 新增命令、新增选项、新增权重类型；
- 修改了某个 `*.ado` 对应的实现。

---

## 2. 前置条件

| 项 | 要求 | 检查方式 |
|---|---|---|
| Stata | StataNow 19.5（MP），装在 `D:\Program Files\StataNow19` | 目录存在，含 `StataMP-64.exe` 与 `ado/` |
| ado 源码 | **必须能读到** `ado/base/**/<cmd>.ado` | 读不到就不要猜，记录为 E1 并说明 |
| Python | 项目 venv | `.venv/Scripts/python.exe` |
| 数据 | `tests/data/auto.dta`（唯一入库的数据集） | 存在即可，git 已 un-ignore |

**ado 位置**：`D:/Program Files/StataNow19/ado/base/` 下按首字母分目录，如
`ado/base/h/hettest.ado`。**先读它，再谈修复。**

---

## 3. 主流程

### 第 1 步：读 ado，定位"决定数字的那一行"

不要从输出反推公式。直接读源码，并回答三个问题：

1. **统计量的定义式是什么？**（哪几个 `e()` 标量、怎么组合）
2. **输入的列是怎么选出来的？**（这是本次事故的答案：`if "`varlist'" == "" { local varlist `z' }`）
3. **有哪些拒绝/边界分支？**（`exit 498`、`r(301)` 之类，必须逐个复现）

产出：一张"ado 行号 → 行为"的对照表，写进代码注释。

> `hettest.ado` 的关键行：
> - `:103-105,138` 把 `` `wgt' `` 传给 `_regress` → 辅助回归要带权重；
> - `:110-115` 无 varlist 时用**拟合值**，不是全部回归元；
> - `:126-128` 只有一个回归元时清空 `mtmethod`；
> - `:143-158` 三个分支分别报 `N*R2`、`F`、`mss/2`。

**验收**：能逐字指出每一处影响数字的 ado 行，而不是"我觉得应该是"。

---

### 第 2 步：写捕获脚本，把 Stata 的真实输出固化成 fixture

不要手抄屏幕上的数字，也不要只用 `r(chi2)` 的默认精度。写一个可重跑的脚本：

**模板**：`scripts/capture_<cmd>_fixtures.py`（参照 `scripts/capture_hettest_fixtures.py`）

核心结构（**示意片段，不是可直接运行的脚本**；完整可运行版本见上述参考实现）：

```python
# 1. 用 marker 前缀输出全精度数值，便于解析
lines.append('di "#HT|CASE_ID|" %21.17f r(chi2) "|" %9.0f r(df) "|" %21.17f r(p)')

# 2. 批处理调用 Stata，日志由 Stata 自己写，不要用管道捕获
subprocess.run(
    [STATA_EXE, "-e", "-q", "do", str(do_file)],
    cwd=str(work),
    stdout=subprocess.DEVNULL,   # 见 §6 陷阱 4
    stderr=subprocess.DEVNULL,
    timeout=600,
)
log = (work / "capture.log").read_text(encoding="utf-8", errors="ignore")

# 3. 解析 marker → JSON，写进 tests/conformance/stata_fixtures/
```

**用例矩阵**（缺一不可）：

| 维度 | 取值 |
|---|---|
| 模型 | 至少 3 个：单回归元、多回归元、含因子变量 |
| 辅助回归选择 | 默认（拟合值）、`rhs`、显式 varlist |
| 方法变体 | 该命令的所有 `method`/分支 |
| 权重 | 无、`aweight`、`fweight`、`iweight`（`pweight` 单独验拒绝） |
| 边界 | 命令拒绝的每个情形，记录返回码 |
| 复合输出 | 若有 `r()` 矩阵（如 `r(mtest)`），逐行记录 |

**验收**：

- fixture 里的每个数字都是 `%21.17f` 全精度；
- 重跑脚本能得到**逐字节一致**的 fixture（否则说明有随机性，要固定 seed）；
- fixture 自带 `stata_version`、`ado` 版本、数据集路径。

---

### 第 3 步：修复实现

按第 1 步的对照表逐条改，一条对应一处代码注释（写 ado 行号）。

**关键纪律**：

- **一次只改一个行为**，改完立刻跑第 2 步的 fixture 比对；
- 不要"顺手"重构无关分支；
- 显示层（表头、单复数、行序）与数值层分开验证 —— 本项目已有教训：
  「布局一致不等于结果正确」（`GAP_REGISTER.md` §8）；
- 遇到"Stata 看起来像 bug"的行为（如单回归元丢弃 `mtest()`），**照抄**，不要"修正"。

**验收**：fixture 全绿，打印形如 `90/90 cases match Stata  TOTAL FAILURES: 0`。

---

### 第 4 步：改测试，让它读 fixture 而不是读 ado

**这是最容易偷懒的一步，也是本次事故的根因所在。**

- 测试**只从 fixture 取值**，不在测试里重算公式；
- 测试里可以保留独立的"关系断言"（如 `p == chi2.sf(stat, df)`、AUC 单调性），
  但**不能**用它替代与 Stata 的一致性；
- 必须加一条**回归测试**，直接针对本次的缺陷点命名，
  例如 `test_default_regresses_on_fitted_values_not_on_every_regressor`；
- 必须加一条**防空转守卫**，防止 fixture 被清空后所有用例静默通过：

```python
def test_the_fixture_is_not_vacuous(reference) -> None:
    assert len(reference["cases"]) >= 90
    assert reference["stata_version"]
```

**验收**：把实现故意改回错误版本，测试必须失败（见 §5 变异检查）。

---

### 第 5 步：登记证据并如实定级

三处都要改，缺一即视为未完成：

| 位置 | 内容 |
|---|---|
| `tests/conformance/registry.yml` | 新增/更新条目：`command`、`level`、`tolerance`、`sources`、`verified_by`、`date`、`notes` |
| `docs/GAP_REGISTER.md` | 命令表的对应行，写明验证范围与已知限制 |
| `CHANGELOG.md` | `[Unreleased]` 下的 `### Fixed`/`### Added`，**必须写出修复前后数字** |

**定级规则**（严格按设计规格 §11.1，不拔高）：

| 等级 | 何时给 | 本次 hettest 属于 |
|---|---|---|
| **E3** | 与真实 Stata 全精度输出逐值吻合，且覆盖该命令的全部分支 | ✅ 90 用例全绿 |
| **E2** | 有界、已测量的差异（如 MLE 标准误第 3–4 位） | — |
| **E1** | 未与 Stata 实跑对比，或某些调用实质不符 | 修复前 |

`registry.yml` 的 `sources.type` 只接受
`published_output` / `r_crosscheck` / `statsmodels_reference` / `invariant`
（见 `src/westata/conformance/registry.py`）。真实 Stata 捕获走 `published_output`，
`ref` 里写清脚本名、fixture 名与 Stata 版本。

---

## 4. 提交前检查清单

逐项打勾，任一未过就不提交：

- [ ] fixture 全绿，且脚本可重跑复现
- [ ] **变异检查**：把修复回退，相关测试确实失败
- [ ] 全量测试通过：`.venv/Scripts/python.exe -m pytest tests/ -q`
- [ ] 只暂存本任务的改动（`git status` 逐个确认，别把并行会话的文件卷进来）
- [ ] 提交信息写明：根因、修复前后数字、验证方式、用例数
- [ ] `CHANGELOG` / `registry.yml` / `GAP_REGISTER` 三处一致，没有一处说 E3 另一处说 E1

---

## 5. 变异检查（最容易被跳过的一步）

测试通过 ≠ 测试有效。**把实现改回错误版本，测试必须红。**

```bash
# 1. 确认全绿
.venv/Scripts/python.exe -m pytest tests/test_estat_hettest.py -q

# 2. 临时把默认改回"全部回归元"，重跑
#    期望：至少 test_default_regresses_on_fitted_values_... 失败
# 3. 还原
```

本次正是靠这一步确认：原来的测试对"默认用全部回归元"这个错误**完全不敏感**，
所以它此前一直是绿的。

---

## 6. 已知陷阱（全部是真实踩过的）

### 陷阱 1：手抄 ado 当 oracle

实现和测试犯同一个误读 → 双方互相印证 → 与 Stata 同错。
**对策**：参考值只从真实 Stata 进程取。

### 陷阱 2："默认值"想当然

`variables=None` 被读成"全部回归元"，而 Stata 的语义是**拟合值**。
两个数字都像正常统计量（7.75 vs 2.53），不报错、不显眼。

**对策**：ado 里每一个 `if "`opt'" == ""` 分支都当成独立用例捕获。

### 陷阱 3：varlist 的名字 ≠ 设计矩阵的列名

Stata 的 varlist 指**原始数据变量**：`hettest rep78` 在 `i.rep78` 模型里用的是原始 1–5 列
（`df=1`, χ²=12.77），不是 `e(b)` 的四列虚拟变量（`df=4/5`, χ²=24.30）。

**对策**：命名解析按"原始列优先，设计列兜底"，并各写一条用例。

### 陷阱 4：Windows 沙箱下用管道捕获子进程

```python
subprocess.run(..., capture_output=True)   # WinError 5 (CreatePipe)
```

**对策**：让 Stata 自己写 `capture.log`，`stdout/stderr=subprocess.DEVNULL`，再读文件。

### 陷阱 5：fixture 目录写入被 ACL 拒绝

`tests/`、`src/`、`scripts/` 的继承曾被断开，子进程写不进去（但文件工具能写）。
**对策**：捕获脚本加 `--out` 参数先写到临时目录，再由文件工具落到仓库；
若长期复现，按 `diagnose-windows-sandbox-acl` 处理。

### 陷阱 6：把"布局对齐"当成"数值正确"

`SKELETON`/`MATCH` 只描述对齐。`xtlogit, re` 曾布局全对、`e(sigma_u)` 差两个数量级。

**对策**：数值与布局分两条验收线，分别报告。

---

## 7. 附：最小骨架

```
scripts/capture_<cmd>_fixtures.py      # 跑 Stata，产出 fixture
tests/conformance/stata_fixtures/<cmd>_<data>.json
tests/test_<cmd>.py                    # 读 fixture 断言 + 回归测试 + 防空转守卫
tests/conformance/registry.yml         # 证据条目
docs/GAP_REGISTER.md                   # 命令表行
CHANGELOG.md                           # 修复前后数字
```

**仓库内可运行的完整参考实现**：

- `scripts/capture_hettest_fixtures.py`
- `tests/conformance/stata_fixtures/hettest_auto.json`
- `tests/test_estat_hettest.py`
