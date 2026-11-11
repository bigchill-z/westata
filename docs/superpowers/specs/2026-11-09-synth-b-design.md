# westata `synth` B 档设计规格

- 日期：2026-11-09
- 状态：**已实现（P0–P4 完成）**，用户通过 goal「完成设计文档的目标」授权执行
- 决策依据：`options.md` 「〇、2026-11-09 决策」——用户批准**目标档位 B**
- 前置证据：P0 已完成，见 [STATA_ALIGNMENT_SUMMARY.md](../../STATA_ALIGNMENT_SUMMARY.md)
  同日修订块、`scripts/verify_synth_mata_formulas.py`（4/4 PASS）、
  6 份 `tests/conformance/stata_fixtures/synth_*.json`（全精度、内部自洽）、
  `tests/test_stata_synth_capture.py`（30/30 PASS）
- 项目性质不变：个人自用库；函数用法以 statsmodels 为准、数值以 Stata 为准；
  一致性声明必须可审计

---

## 1. 目标与非目标

### 1.1 目标

- G1：新增 `ws.synth`（statsmodels 风格的结果对象），在**默认路径**
  （regression-based V + 内层约束 QP）上复现 Stata `synth` 的六个 `e()` 矩阵
  与处理效应路径。
- G2：证据等级 **E2**——「有界、已测量的差异」，容差由 P0 fixture 实证标定，
  每个量的实测差写入 `tests/conformance/registry.yml`。
- G3：无法保证一致的量**显式拒绝或标注 `matched=False`**，绝不静默给出
  「看起来对」的数。
- G4：保持 westata 既有纪律：结果对象不伪造系数 VCE；`e()` 键名逐字对齐 Stata；
  中英文 README/GUIDE/API 文档同步校对。

### 1.2 非目标（B 档边界；2026-11-10 扩展后重新划定）

> **2026-11-10 更新**：用户通过 goal「实现 Stata synth 除图形外的全部功能」把范围
> 扩大。下表标注了哪些 B 档排除项已在扩展轮次实现；详细实测见 §5 与
> `tests/conformance/test_stata_synth_features.py`。

| 项 | B 档（2026-11-09） | 扩展后（2026-11-10） |
|---|---|---|
| `unitnames()` | 未做 | ✅ 实现（只改 `W_weights` 行标签，数值不变） |
| `var(period)`（单期 / `&` / numlist） | 未做 | ✅ 实现（每个窗口算一个预测变量） |
| `keep()` / `replace` | ❌ 抛 `UnsupportedError` | ✅ 实现（5 列堆叠 `.dta`，非图形） |
| `margin/maxiter/sigf/bound` | 未暴露 | ✅ 公开参数（实测只影响 RMSPE，不影响 `W_weights`） |
| 缺失值 `-ignored for averaging` 警告 | 未做 | ✅ 实现（`missing_warnings`） |
| `nested` / `allopt` | ❌ 抛 `UnsupportedError` | ✅ **算法等价**实现（Powell 外层）；`allopt` 的 W 与 Stata 逐位一致、RMSPE 差 3.5e-9；`nested` 单次起点比 Stata 找到更低损失（E2，非逐位一致） |
| `synth2` | 明确不做 | **仍不做**（图形/placebo/LOO 包装器） |
| 图形 `figure` | 明确不做 | **仍不做**（本轮目标明确排除图形） |
| placebo / LOO | 明确不做 | **仍不做**（`synth2` 的机制，非 `synth` 自带） |
| 推断 / 标准误 | 明确不做 | **仍不做**（Stata `synth` 无 `e(V)`，伪造违反 G3） |
| **E3 / 全命令逐位一致** | 排除 | **仍排除为整体声明**；但 `V` 与 7/9 案例的 `W`/`X_balance`/`Y_synthetic` 已实测到机器精度，按量定级 |
| REA 逆向 | 不用 | 仍不用 |

### 1.3 成功标准

- S1：≥2 个公开数据集（Prop99 `smoking.dta`、Basque `basque.dta`），case 矩阵覆盖
  default / `customV` / `mspeperiod` / `xperiod` / `counit` / 拒绝路径；六份全精度
  fixture 已由 P0 产出并内部自洽。
- S2：实测容差满足 §5 表；**先测后写**，不得先拍死再凑。
- S3：变异测试——回退实现后对拍测试必须转红（`docs/TESTING_SOP.md:186-199`）。
- S4：全量 `pytest -q` 绿、registry schema 校验通过、四个「钉住缺席」守卫按
  §6 顺序同步更新、E 级计数三处一致。
- S5：不把未跑过的内容写成「已验证」。

---

## 2. 用户与核心流程

单一用户（作者本人）。核心流程：

```python
import pandas as pd
import westata as ws

df = pd.read_stata("smoking.dta", convert_categoricals=False)
res = ws.synth(
    data=df, unit="state", time="year",
    depvar="cigsale",
    predictors=["lnincome", "age15to24", "beer", "retprice"],
    trunit=3, trperiod=1989,
    # 可选：counit=[...], xperiod=[...], mspeperiod=[...],
    #       resultsperiod=[...], customV=[...]
)
res.V_matrix          # K x K 对角
res.W_weights         # J x 2
res.X_balance         # K x 2
res.RMSPE             # float
res.Y_treated, res.Y_synthetic, res.gap
res.summary(style="stata")
```

结果对象为**冻结 dataclass**（照 `rcm.RcmResult` / `rdrobust.RdRobustResult` 先例），
**不套 `StataResults`**：synth 无系数向量、无 `cov_params`、无 t/z 分布、无 `df_resid`。
强行套用会逼出伪造的 `params`/`bse`，违反 G3。

---

## 3. 总体架构与互斥写入范围

```
src/westata/estimators/synth.py      # 唯一实现文件（新）
src/westata/estimators/__init__.py   # barrel 导入 + __all__
src/westata/__init__.py              # 顶层导出
src/westata/results/summary.py       # 若需 Stata 风格表头，注册 model_name
tests/test_synth.py                  # 单测（新）
tests/conformance/test_stata_synth.py# fixture 对拍（新）
tests/conformance/registry.yml       # E2 条目
scripts/verify_synth_mata_formulas.py# P0 已产出，保留为公式守卫
scripts/capture_synth_fixtures.py    # P0 已产出
tests/conformance/stata_fixtures/synth_*.json  # P0 已产出（6 份）
```

**分层**：`synth.py` 内部仍分三段，互不渗透——
① 数据准备（长面板 → X/Z/Y 矩阵）；
② 数值内核（`normalize` / `regsval` / `customV` 归一化 / 内层 QP）；
③ 结果组装（六个 `e()` 等价量 + 舍入时机 + 派生轨迹）。
更换 QP 后端不影响 ①③。

**并行写入纪律**（`AGENTS.md §6.1`）：实现阶段若再委派，Lead 不写已分配文件；
共享文件（三个 `__init__` / `summary.py` / `registry.yml`）**只设一个 owner**；
每轮交付按所有权清单逐项核对。

---

## 4. 数据流契约（字段级，源码为唯一真相）

### 4.1 输入契约

| 参数 | 类型 | 语义 |
|---|---|---|
| `data` | `pd.DataFrame` | 长面板，**平衡**；`unit`/`time` 列必须存在 |
| `unit` / `time` | `str` | 面板/时间变量名；`time` 数值型 |
| `depvar` | `str` | 结果变量 |
| `predictors` | `list[str]` | 顺序**即 Stata varlist 顺序**，决定 `e(V_matrix)` 对角顺序 |
| `trunit` / `trperiod` | 底层编码 / 数值 | 必须能在 `unit`/`time` 列中取到 |
| `counit` | `list \| None` | 默认 = 全部单位减去 `trunit`（`synth.ado:62-64`） |
| `xperiod` / `mspeperiod` | `list \| None` | 默认 = 全部 `< trperiod` 的期（`:145-161`） |
| `resultsperiod` | `list \| None` | 默认 = **全部**期（`:171-174`） |
| `customV` | `list[float] \| None` | 长度必须等于 K（`:398-402`） |

### 4.2 形状契约（**最易写反**）

```
Xtr ∈ R^{K×1}   Xco ∈ R^{K×J}      # 行 = predictor，列 = 单位；treated 在第 0 列
Ztr ∈ R^{T0×1}  Zco ∈ R^{T0×J}     # 行 = 处理前时间，列 = 单位；同 X 的列序
Ytr ∈ R^{T×1}   Yco ∈ R^{T×J}      # 行 = resultsperiod
N  = J + 1                          # normalize/regsval 都含 treated 单位
```

`synth.ado:319-321` 转置 `tabstat` 输出得到上述形状。传错方向 Stata 报 r(3200)，
Python 实现必须显式校验并抛 `ValueError`（附实际/期望 shape）。

### 4.3 数据准备（可读源码逐行转写）

- predictor 在 `xperiod` 内按单位取**均值，忽略缺失**（`agmat` → `tabstat ... s(mean)`，
  `:756-775`；`missingchecker` 只警告不报错，`:727/:730`）。
- 结果变量在 `mspeperiod`/`resultsperiod` 每期每单位必须齐全，否则按 Stata
  r(198) 语义报错（`agdvar:792-803`）。
- **`.dta` 类型契约**：`pd.read_stata(..., convert_categoricals=False)` 拿底层整数编码；
  所有参与计算的数值列**显式 `astype(float64)` 后再求均值**。
  实测：不 cast 时 float32 累加使 `xtrmat` 偏 4.6e-5、V 偏 ~3e-7；
  cast 后 V 差 7.8e-14。Basque 为 double，无此问题。
- Stata 缺失值是 `8.98846567431158e307` 而非 NaN；注入 X/Z 会让 `regsval` **静默**
  返回垃圾 V（rc=0）。实现必须在入矩阵前显式 guard：任一单位某 predictor 全缺失、
  或某期结果缺失 → `ConsistencyError`。

### 4.4 数值内核（P0 已精确求解，见 `scripts/verify_synth_mata_formulas.py`）

```
Vmat = [Xtr | Xco]                                  # K x N
s[k] = std(Vmat[k,], ddof=1)                        # quadvariance = 样本方差，含 treated
Xn   = Vmat / s[:, None]

Xdm  = Xn - Xn.mean(axis=1, keepdims=True)          # 按单位去均值
Zdm  = [Ztr|Zco] - 行内按单位去均值                  # 每期一个截距
Sxx  = Xdm @ Xdm.T
B    = solve(Sxx, Xdm @ Zdm.T)                      # K x T0
raw_k= sum_t B[k,t]^2
V    = diag(raw_k / sum(raw_k))
```

`customV` 路径：`V = diag(|b| / Σ|b|)`（P0 白盒：`normweights(1,2,3,4) → .1/.2/.3/.4`）。

**精度**：满秩 `Sxx` 实测 ≤9.1e-15（11 组构造）/ ≤8.6e-14（4 组真实）。
**秩亏**：Stata `invsym()` 的广义逆 sweep 顺序未复现（唯一有界未知）；
且 `J=1` 时 Stata 自己报 r(3200)。实现按 `ConsistencyError` 拒绝
`rank(Sxx) < K`，不猜、不外推。

### 4.5 内层 QP（目标已知，`synth.ado:546-557` / `synth_ll.ado:13-27`）

```
H = Xco' · V · Xco                  # J x J
c = -(Xtr' · V · Xco)'              # J x 1
min ½ W'HW + c'W      s.t. ΣW = 1, 0 ≤ W ≤ 1
```

用 `scipy.optimize`（已核实 1.15.3 具备 SLSQP / trust-constr / `nnls` / `linprog`）。
默认参数以 **ado 源码**为准：`margin=0.005`、`maxiter=1000`、`sigf=12`、`bound=10`
（`synth.ado:35-38`）—— 帮助文件 `synth.sthlp:270/274` 的 `0.05/7` **已过时**，
照抄会错（`AGENTS.md §6.2`）。

### 4.6 结果组装（**舍入时机最容易写反**）

1. `RMSPE = sqrt( (Ztr − Zco·W_exact)'·(Ztr − Zco·W_exact) / T0 )`
   —— 用**精确 W**（`:564-566`）。
2. 随后 `W_rounded = round(W_exact, 3)`（`roundmat`，`:589-590`）。
3. `W_weights`、`X_balance`、`Y_synthetic`、`gap` 全部用 **`W_rounded`**
   （`:594/:606/:616`），因为舍入在它们之前。
4. `X_balance` 用**原始 X**（`:560-561` 恢复），不是 normalized X；优化才用 normalized。
5. `Y_treated`/`Y_synthetic` 覆盖 `resultsperiod`，`gap = Y_treated − Y_synthetic`。

实测后果：六份 fixture 中 `e(W_weights)` 列和为 **0.999 / 1.001**（非 1.0）——
三位舍入破坏归一化，**必须照抄，不得「修正」回 1**。

### 4.7 输出契约

| 属性 | 形状 | 对应 Stata |
|---|---|---|
| `V_matrix` | K×K 对角 | `e(V_matrix)`（**名字不是 `e(V)`**） |
| `W_weights` | J×2（unit, weight） | `e(W_weights)`（无条件，不需 `keep()`） |
| `X_balance` | K×2（Treated, Synthetic） | `e(X_balance)` |
| `RMSPE` | float | `e(RMSPE)` |
| `Y_treated` / `Y_synthetic` | T×1 | 同名 `e()` 矩阵 |
| `gap` | T×1 | 派生量 |

---

## 5. 证据与容差（实证标定，非拍脑袋）

> **2026-11-11 最终状态**（下表为 B 档当时记录，保留为历史）。经扩展功能 +
> 自适应内点法 QP 两轮改进，8 个可比对案例的**最终实测界**：
> `|dV| ≤ 1.4e-13`（机器精度，全部）；`|dW| = 0` 于 6/8、最差 **9e-3**；
> `|dX_balance| = 0` 于 6/8、最差 **9.3e-2**；
> `|dY_synthetic| ≤ 3e-14` 于 6/8、最差 **2.9e-1**；
> `|dRMSPE| ≤ 3.6e-4` 于良态池、最差 **6.9e-2**。
> `nested allopt` 的 W 逐位一致、RMSPE 差 3.5e-9。
> 逐量守卫与最新数字见 `tests/conformance/test_stata_synth_features.py`
> 与 `tests/test_synth_qp_refinement.py`。

| 量 | 实测差（B 档当时） | 声明容差（当时） | 依据 |
|---|---|---|---|
| `V` 对角 | ≤8.6e-14（满秩） | **≤1e-10**（相对，留 1000× 余量） | 11 构造 + 4 真实数据集 |
| `W`（逐元素） | **0**（V 偏差 ≤1e-8 时三位舍入完全吸收） | **≤1e-3** | `roundmat` 显示精度 |
| `W` 列和 | 0.999 / 1.000 / 1.001（实测） | **≤2e-3 偏离 1** | 三位舍入破坏归一化；实现须照抄，不可「修正」回 1 |
| `RMSPE` | **5.6e-4**（实测最差，smoking_default） | **≤1e-2**（绝对） | 秩亏 QP：`H` 的秩 ≤ K ≪ J，最优 W 不唯一；Stata C++ 插件以 `margin=0.005` 停在最优面外，scipy SLSQP 解得更紧 |
| `Y_*` / `X_balance` | ≤1e-3 | **≤1e-3** | 由三位舍入后的 W 派生 |

> **S2 更正（实现期实测）**：本表原写 `|dRMSPE| ≤ 1e-8`，依据是「V 偏差 ≤1e-13 时
> RMSPE 变化 ≤1e-11」。该推断**被实测证伪**——它假设 QP 解唯一，而 `H = Xco'·V·Xco`
> 秩亏（K=4、J=38 时 `cond(H)≈6e18`），不同最优面上的 W 给出不同 RMSPE。
> 六个 fixture 的实测差为 −5.6e-4 / −1.1e-6 / +1.07e-4 / −6.7e-5 / −1.9e-5 /
> +1.4e-9。按设计 S2「先测后写，不得先拍死再凑」，容差已改为实测上界 1e-2，
> 并在 registry notes、CHANGELOG、USING_WITH_STATA.md 三处一致披露。**E3 排除的
> 物理原因不变**（插件自身抖动 1.471e-11），但数值差距比原估计大 5 个数量级。

**2026-11-10 扩展功能实测更正**：`synth_features.json` 的 16 份活体捕获表明，
「两次相同运行差 1.471e-11」是**数据/设定相关**的：对同一 `cigsale lnincome
beer retprice` 默认命令连跑两次，`e(RMSPE)` 都是 `19.73684845951759`（差 **0.0**）。
改变 `sigf(7)` 会让 RMSPE 变 2.1e-3，而 `roundmat()` 让 `e(W_weights)` 在
所有调参下完全不变。因此**默认路径是确定性的**；E3 排除的真正原因只剩
「`H` 秩亏 ⇒ QP 最优 W 不唯一 + C++ 插件以 `margin=0.005` 停在最优面外」，
不是「参照源每次都不同」。先前那句 1.471e-11 的断言保留为历史，不再作为
全局依据。逐量定级见 `tests/conformance/test_stata_synth_features.py` 模块
docstring 与 registry notes。

`registry.yml` 条目形状（照 `tobit` E2 条目：`:125-138`）：

```yaml
- command: synth
  level: E2
  tolerance: 1.0e-8
  sources:
    - type: r_crosscheck
      ref: "6 fixtures captured from StataNow 19.5 MP / synth.ado 0.0.8; see tests/conformance/stata_fixtures/synth_*.json"
    - type: invariant
      ref: "V diagonal, nonnegative, sums to 1; W in simplex; RMSPE = sqrt(SSE/T0)"
  verified_by: lead
  date: 2026-11-09
  notes: "Measured bounds: |dV|<=1e-10, |dW|<=1e-3, |dRMSPE|<=1e-8. E3 excluded: synthopt.plugin is not bit-deterministic (self-jitter 1.47e-11). nested/allopt/synth2 refused."
```

---

## 6. 错误处理与降级

| 情形 | 行为 |
|---|---|
| `nested` / `allopt` / `figure` / `keep` / `synth2` / placebo | `UnsupportedError`，消息写明「Stata 有、westata 未实现 + 原因 + 等级」 |
| `rank(Sxx) < K`、`J < 2`、非平衡面板、结果缺失、全缺失 predictor | `ConsistencyError` / `ValueError`，附实测 shape 与期望 |
| `customV` 长度 ≠ K | `ValueError`（对齐 `:398-402`） |
| QP 不收敛 / 解越界 | 显式异常或 `matched=False` 标记，不静默返回 |
| 任一量超出 §5 容差 | 标注 `matched=False` + 实测差，**不隐藏** |

结果对象不提供 `params`/`bse`/`pvalues`/`conf_int()`/`df_resid`；下游 statsmodels
代码若需要这些，应显式转换，而不是由 synth 伪造。

---

## 7. 测试计划（先写测试，再写实现）

1. `scripts/verify_synth_mata_formulas.py`（**已有**，4/4 PASS）作为公式守卫，纳入回归。
2. `tests/test_synth.py`：形状契约（4 种矩阵）、cast-float64 契约（不 cast 必须失败或
   给出更大差）、缺失均值语义、raw/normalized X 切换点、`roundmat` 时机
   （W 列和 0.999/1.001 必须复现）、QP 单纯形约束、六个 `e()` 键名集合、
   `customV` 归一化、秩亏/`J<2`/非平衡/缺失的拒绝路径。
3. `tests/conformance/test_stata_synth.py`：读 6 份 fixture（**无需 Stata**），
   逐量比对 §5 容差、V 对角/非负/和 1、`cases` 与 `matrices` 自洽、
   变异测试（回退实现 → 转红）。
4. `StataCapture.capture_matrices()` 的矩阵型捕获测试（P0 已产出）。
5. 全量 `pytest -q` + registry schema 校验 + 每 20 轮修改一次全量测试（工作区规则）。

---

## 8. 文档与交付（工作区规则）

- 更新 `docs/GAP_REGISTER.md`（§3.6 行、§3.6.1、§3.6.4、§1 计数与 E 级表）、
  `README.md` / `README.zh-CN.md`（Known gaps / Evidence levels / Known deviations）、
  `docs/USING_WITH_STATA.md`、CHANGELOG（含前后计数）。
- `docs/API.md` / `docs/API.zh-CN.md` 由 `docs/render_api_reference.py` 从
  `docs/api_source/*.json` **生成**，不手写；`docs/GUIDE.md`(+zh) 同步。
- `options.md` 已记录 B 档决策，`error.md` 已记录 7 条通用教训；实现中若发现
  新的通用问题继续追加。
- 版本号提升与 PyPI / GitHub 推送**另行单独确认**，不在本设计授权内。
- 四个「钉住缺席」守卫（`test_synth_status.py`、`test_the_two_missing_features.py`、
  `test_zero_scored_but_present.py`、`test_register_p1_recheck.py`）在 `ws.synth`
  真正落地时按契约同步改写；**此前保持绿色**（缺席仍是事实）。

---

## 9. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 秩亏 `Sxx` 的广义逆行为未复现 | 显式拒绝，不外推；文档标为有界未知 |
| QP 后端与 C++ 插件活跃集不同 | 凸问题最优值唯一性 + 三位舍入吸收；容差 §5；不承诺 E3 |
| 帮助文件默认值过时 | 以 ado `:35-38` 为准，测试固定 `margin=0.005`/`sigf=12` |
| `.dta` float32 / category 陷阱 | `convert_categoricals=False` + `astype(float64)`，守卫测试覆盖 |
| 把 E2 写成 E3 | registry notes 明确写「E3 excluded」+ 抖动数字；评审时逐项核对 |
| 舍入时机写反 | 单测断言 W 列和 0.999/1.001，而非 1.0 |

---

## 10. 里程碑

| 阶段 | 产出 | 状态 |
|---|---|---|
| P0 证据 | 文档纠偏、6 份全精度 fixture、`StataCapture` 矩阵捕获、两内核公式 + 容差 | ✅ 已完成 |
| P1 设计 | 本文档 | ✅ 已完成 |
| P2 实现 | `synth.py` + 单测（TDD：RED→GREEN 已验证） | ✅ 已完成 |
| P3 对拍 | fixture 比对、容差确认、registry 登记 E2（96 条：54/27/15） | ✅ 已完成 |
| P4 交付 | 中英文文档校对、CHANGELOG、生成器重跑 | ✅ 已完成（**推送 PyPI/GitHub 另行确认**） |

---

## 11. P0 验收记录

| 检查 | 结果 |
|---|---|
| `scripts/verify_synth_mata_formulas.py` | 4/4 PASS（normalize 2.8e-14、regsval ≤8.8e-14），无需 Stata licence |
| 6 份 synth fixture | 六个矩阵键齐全、形状正确、V 对角非负和 1、`cases` 与 `matrices` 差 0、与三个已知 ground truth 差 0~1e-18 |
| `tests/test_stata_synth_capture.py` | 30 passed |
| registry schema | P0 时 95 条（54/26/15）；P3 追加 `synth` E2 后 **96 条（54/27/15）**，校验通过 |
| 文档守卫（4 个缺席守卫 + fixture 范围守卫） | 27 passed → 落地后按各自注释改写为存在性检查，全绿 |
| `tests/test_synth.py` + `tests/conformance/test_stata_synth.py` | **67 passed**（TDD RED→GREEN 已验证） |
| `scripts/verify_synth_mata_formulas.py` | 4/4 PASS，无需 Stata licence |
| 变异测试（回退 `_regsval` 为等权） | **检测到**：V 从 `[.0202,.0705,.2596,.6497]` 变 `[.25,.25,.25,.25]`，差 0.3997 |
| **全量 `pytest -q`（P0 时）** | **5866 passed, 3 skipped, exit 0** |
| **全量 `pytest -q`（P4 收尾）** | **见下方最终验收** |

P0 时 `tests/conformance/stata_fixtures/_host_probe.json` 因 `tests/` 独立 DACL
无法删除，曾降级为自解释非证据 JSON + `.gitignore` 排除。**2026-11-10 会话权限
放宽后已删除**；`.gitignore` 规则保留为防御。详见 `error.md §0.7`。

### P4 最终验收（2026-11-10）

| 检查 | 结果 |
|---|---|
| 全量 `pytest -q`（含 API/文档守卫） | 见本轮实跑输出（0 failed） |
| `collect_api_reference.py --check` | `218 prose entries against 218 names: 0 problem(s)` |
| `render_api_reference.py` 纯重渲染 | 两个 API 文件均含 `synth`/`SynthResult`/`0.05229223550829`/`vce(cluster`；无需手工回注 |
| `test_api_reference_is_in_sync.py` + `test_robust_vce_mapping.py` | 31 passed |
| `audit_registry_coverage.py` | `UNACCOUNTED FOR: 0`，exit 0 |
| 中英文 README / GUIDE / API / USING_WITH_STATA / CHANGELOG / GAP_REGISTER | 已同步校对，`test_docs_zh.py` / `test_guide.py` 全绿 |

## 12. 未决问题

- **仅剩 P4 推送 PyPI / GitHub 前的单独确认**（工作区规则：以新版本号推送，且
  推送前完成中英文文档校对）。本设计范围内 P0–P4 的实现、对拍、登记、文档
  均已执行完毕，本地未推送。
