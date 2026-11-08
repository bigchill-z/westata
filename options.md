# 目标模式：任务过程中跳过的、需要用户拍板的决策

> 本文件记录需用户决定的选项；若后续不再需要，及时更新。

## 一、2026-11-07 安装《教材必装 Stata 程序清单》时的决策

### 已关闭（2026-11-07 23:47 用户答复：「都不用」）

| 决策 | 结果 |
|---|---|
| O1 是否补装 `xtserial` | **保持现状，不补装**（本机未装，文档中为"建议装"非"必装"）|
| O2 是否补装"不必装"的 `rd`/`psmatch2`/`xtbalance` | **维持不装** |
| O3 示例数据集安放位置 | **维持现状** |
| O4 可复现产物 `_stata_pkgs\` | **保留** |
| 是否整理 SOP / 可复用 skill | **用户明确拒绝，不再询问、不创建** |

---

## 二、v3 报告修复过程中记录的技术判断（非用户决策，仅登记）

以下不是"等待用户拍板"的事项，而是**已做出并验证的技术判断**，登记在此以便追溯。

### T1. `rdrobust` 自动带宽：停在"组合规则已实现"

- **现状**：三种组合规则（`mserd`/`msesum`/`msetwo`）已从 `rdbwselect.ado:499/527/537`
  精确转写，7 个守卫覆盖算术与符号方向。
- **阻塞点**：`C[1..3]` 由 `rdrobust_bw()` 在 `l/lrdrobust.mlib` 中计算。
  `mata which` 确认位置；该文件 321 KB、**可打印字节仅 1.9%**，函数名是孤立字符串，
  前后皆二进制 —— 是编译字节码。注意这与 `synth` **不是同一类**：`synth` 的顶层算法
  是可读的 `synth.ado`，只有数值子过程在库里（见 T6）；而 `rdrobust` 的带宽选择器
  **整体**在库里，ado 一侧只有参数设置。
- **已采取**：`rdrobust` 仍拒绝自动选带宽（`UnsupportedError`），而非接入一个
  会返回 `h=0.0757`（Stata 是 `0.182`）的选择器。
- **若需推进**：需要 `rdrobust` 官方源码或论文附录的完整 `C[1..3]` 定义。

### T2. `nbreg` 的多起点选择阈值

- **现状**：阈值 `1e-3`、起点下限 `1e-9`，均已由实测确定。
- **遗留**：`alpha` 落在 `1e-6`，Stata 在 `1.03e-09`。全部起点的似然一致到 `4e-4`，
  **无法凭似然区分**"真边界"与数值噪声（另一份数据上噪声增益 `5.3e-4`，量级相同、
  方向相反）。
- **影响**：`ll` 差 `4e-4`、系数末位不同；`/lnalpha` 与 `alpha` 的 SE 已按边界规则显示 `.`。
- **若需推进**：需要 Stata 的停止准则（梯度容差 / 边界投影方式），而非继续调起点。

### T3. 未实现的条目（已确认为能力边界）

| 条目 | 状态 | 依据 |
|---|---|---|
| `synth` / `synth2` | 未实现 | **顶层算法可读**（`synth.ado` 931 行，含两步优化、V 矩阵、nested 结构）；六个数值子过程在 `lsynth_mata_subr` 库中。见 T6 |
| `xtserial` | 未实现 | 本机 Stata 未装，`r(111)` |
| `arimasoc` | 未实现 | — |
| `rdrobust` 自动带宽 | 部分 | 见 T1 |
| `teffects` AI-robust 方差 | 未实现 | 六种影响函数形式试过，均差 2.9–3.4%，见 `teffects_ra.py` 文档 |
| `teffects psmatch` 推断量 | 未实现 | 需完整 Abadie-Imbens 影响函数，docstring 已排除两种简化 |

### T4. `teffects aipw` 点估计低 2.53（阻塞点已定位到可读源码）

**重要进展**：`_teffects_gmm_aipw.ado`（1500 行）**是可读的**，核心矩条件已找到。

```stata
* L469  ipw = 1/pij             只看处理组
* L475  resj = cond(tr, ipw*(depvar-muj), 0) + muj - pop[1,j]
* L630  CMomentj, ofform=linear:  resj = cond(tr, depvar - muj, 0)
```

- **PO 矩条件**：`t*(1/p)*(y - m) + m - mu` —— 与本包的 `fitted + t*(y-fitted)/p` 形式一致；
- **条件均值矩条件**：`t*(y - m)` —— 按处理组拟合，也与本包一致。

**真正的差异在求解方式，不在公式**：

```stata
* L753  dcm = cond(tr, 1-ipw, 1)
```

`muj` 的导数含 `(1 - 1/p)`，说明 **`muj` 与 `pop` 是联合 GMM 估计的**，
而不是「先 OLS 拟合、再对 augmented 取均值」。

本包走的是后者，因此**落在另一个点**。修正项 `-6.58` / `-4.05` 不是 bug，
而是两个估计点之间的差。

- **若需推进**：把 `(pop, cm, ps)` 作为整体做 GMM，用上面三条矩条件联立求解。
  这是估计器级改动，不是修补。
- **参考**：`_teffects_gmm_aipw.ado` 在 `ado\base\_\` 下，`findfile` 可定位。

### T5. `estimate` 之外还差什么，已知清单

| 命令 | 现状 | 与 Stata 的差 |
|---|---|---|
| `teffects ra`（类） | **精确** | — |
| `teffects ipw` | 点估计精确 | SE 高 0.24% |
| `teffects aipw` | 点估计低 2.53 | 见 T4 |
| `teffects psmatch` | 点估计精确 | `se`/`z`/`p`/CI 为 NaN（需 Abadie-Imbens 影响函数）|
| `teffects nnmatch`（类） | 点估计精确 | SE 高 0.78%，输出已标注为近似 |

### T6. `synth` / `synth2`：可读与不可读的边界，已用 `findfile` 与 `mata which` 界定

前几轮把 `synth` 一句「核心在 Mata 字节码」概括为 A 类「信息不可得」。**这个判断过粗。**

**可读的部分** — `synth.ado`（`C:\Users\zjl\ado\plus\s\synth.ado`，931 行）：

```
L367  /* now go to optimization */
L371  di as res "Second Step: Run Optimization"
L386  /* Set up V matrix */
L427  /* now if the user wantes the full nested method, go and get Vstar via nested method */
L39   nested
```

即**两步优化流程、V 矩阵构造、`nested` 方法**都在可读代码里。

**不可读的部分** — 六个子过程全部位于 `lsynth_mata_subr` 库，`mata which` 逐个确认：

| 函数 | 所在 |
|---|---|
| `normalize()` | `lsynth_mata_subr` |
| `regsval()` | `lsynth_mata_subr` |
| `normweights()` | `lsynth_mata_subr` |
| `getabs()` | `lsynth_mata_subr` |
| `roottaker()` | `lsynth_mata_subr` |
| `roundmat()` | `lsynth_mata_subr` |

**这六个是数值内核**（标准化、回归得 V、权重归一、取绝对值、求根、矩阵舍入），
而损失函数的**结构**由 ado 一侧的 `loss` 变量与 `$bd`/`$marg`/`$maxit`/`$sig` 参数控制。

- **结论**：`synth` 并非「完全不可得」。重建需要按 `synth.ado` 的调用顺序实现那六个内核，
  再套上可读的两步优化骨架。工作量是**中等到大**，不是**不可能**。
- **此前记录有误**，以此条为准。

### T6 补充：lsynth_mata_subr.mlib 已实测，确实不可读

上一节把可读/不可读的边界列清了，但**没有测量库本身**。现已补测：

`
文件: lsynth_mata_subr.mlib   27068 bytes
可打印字节: 929 / 27068 = 3.4%
六个函数名: normalize / regsval / normweights / getabs / roottaker / roundmat
            全部只是孤立字符串（Contains 为 True，但前后皆二进制）
`

**3.4% 可打印率与 lrdrobust.mlib 的 1.9% 同级** —— 都是编译后的 Mata 字节码。

**因此**：

- synth 的**顶层流程**（两步优化、V 矩阵、nested）可从 synth.ado 逐行转写；
- 六个子过程的**数值内容**无法获取，只能按 Synth 论文（Abadie-Diamond-Hainmueller）
  重新实现，**无法保证与 Stata 逐位一致**。

- **结论修正**：synth 的准确状态是「骨架可转写，内核需按论文重建且无法验证到 Stata 的位级一致」，
  而**不是**「不可能」，也**不是**「可以照搬」。

### T7. xtreg, re 的 e(sigma_e) 低 1.1%（新发现，未修复）

实测 xtreg y k1 i.year, re（munnell.dta，N=816）：

| 量 | westata | Stata |
|---|---|---|
| sigma_u | 14100.88 | 14096.756 |
| sigma_e | **9463.8949** | **9567.7549** |
| 
ho | .68944081 | .68462142 |

**根因已定位**：本包的 
ss = 73085290324.19 恰好等于 **
mse² × (N−k)**
（Stata 的 
mse = 9570.035363，N−k = 798，乘积 73085290325.56）。

但 Stata 的 sigma_e² × N = 74698217705.73，**与 rss 不是同一个数**——两者相差 2.2%。

也就是说 **e(rmse) 与 e(sigma_e) 基于不同的残差**，而本包把同一个 rss 除以 N 得到
sigma_e，因此偏低。

另一个已排除的猜想：Stata 的 e(rss) 是**缺失**（.），所以无法从 e(rss) 反推，
必须按其定义（随机效应变换后的残差）实现。

- **若需推进**：需要 xtreg.ado 中 sigma_e 的确切定义，或按 GLS 变换后的残差实现并对照。

### T7 补充：`sigma_e` 的正确公式已从源码读出，接入点尚未打通（第 45 轮）

**已确认的正确公式** —— 来自 `xtreg_re.ado`：

```
L128  scalar `s_e' = sqrt(e(sse)/e(df_t))     <- 由 ffxreg2（FE 估计器）算出
L338  est scalar sigma_e = `s_e'
```

**即 `xtreg, re` 的 `sigma_e` 就是 within 回归的 rmse。** 实测验证：

| 模型 | Stata `xtreg, fe` 的 `e(rmse)` | Stata `xtreg, re` 的 `e(sigma_e)` |
|---|---|---|
| `y ~ k1 + C(year)` | `9567.754881` | `9567.754881` |
| `lny ~ lnk1 + lnk2 + lnlabor` | `0.0388216088` | `0.0388216088` |

**公式**：`sqrt(within_rss / (N - N_g - K + 1))`，`+1` 补回组内去均值消掉的截距。
手工验证：`sqrt(68747992036.4959 / (816 - 48 - 18 + 1)) = 9567.754881` ✅

**接入失败的三处原因（均实测）**：

1. `_stata_scalars` 里 `_long_response` 为 `None`（只在某些路径填充）；
2. `_data` **不包含实体列**（`st_abb` 在设计矩阵构造时被消耗）。
   回退猜列会按错误列分组 —— 实测 `entity="st_abb"` 时按 `k1` 分组，
   得到 `sigma_e = 0.0872`（应为 `0.0388`）；
3. `_panel_decomposition` 里的 `resid` 在该处**不是** within 残差
   （实测其平方和与对应 FE 的 `e(rss)` 不符）。

**正确的接入点**：在 `RandomEffects.fit` 内部显式跑一次 within 回归取 rmse，
并把结果传进 `_panel_decomposition`；或把 FE 的 `resid` 显式传进去。
两者都是小改动，但必须先确定 `_long_design` 的行序来源，**不能再靠猜列**。

**一个反向确认**：该三变量模型修复前本包 `sigma_u = 0.0818491`、`sigma_e = 0.0393108`
（与 `test_xttest0_variance_block` 注释里那两个"旧 bug 值"吻合），
而 Stata 是 `0.0818629` / `0.0388216`。说明当时的"已修复"结论并不成立。

### T8. `xtreg, re` 的 e() 表面多出 13 个 Stata 没有的键（第 49–50 轮发现，未处理）

用 `display e(key)` 逐项探测（缺失标量打印 `.`，存在则打印数值）——
**这个方法比 `ereturn list` 可靠，后者的输出是截断的**（第 49 轮先用了它，结论是错的）。

`xtreg` 三个变体的实测差值：

| 变体 | 本包键数 | Stata 存在 | 本包多出 |
|---|---|---|---|
| `fe` | 44 | 40 | 4 |
| `re` | 37 | 24 | **13** |
| `be` | 32 | 26 | 6 |

**`xtreg, re` 多出的 13 个**（Stata 全部打印 `.`）：

`F`、`corr`、`df_b`、`df_r`、`ll`、`rc`、`rss`、`sum_w`、`ui`、
`_obs_per_group`、`group`、`n_entities`、`title`

`ereturn list` 印证：`xtreg, re` 只有 `r2_o`/`r2_b`/`r2_w`、`theta`、
`chi2type`、`bf`/`VCEf` 等，**没有 `F`、`ll`、`rss`、`df_r`**。

**四个是包内命名**（`_obs_per_group`、`group`、`n_entities`、`title`）在三个变体里都多出——
它们不是 Stata 的 `e()` 键，而是包内部使用的别名（如 `title` 驱动表头）。

- **未处理原因**：移除 13 个键影响面大（`ll`、`rss` 等常被下游代码读取），
  需要先确认没有内部依赖，再按估计器分别加标志。第 48/49 轮已用同样手法处理
  `e(r2)` 与三个平方和，这条路子是验证过的。

### T8 补充：尝试移除失败，下游依赖比预想的多（第 51 轮）

第 50 轮把「影响面大」当作不移除的理由，**没有验证它**。第 51 轮验证了，结论是理由成立，
但成立的方式和设想不同。

**1. 内部代码没有直接读取这些键。**

用正则扫全库的 `["key"]` / `.get("key")` 形式，命中项绝大多数属于**其他命令**
（`hdfe`、`systems`、`newey`、`did`），而 `panel.py` 里的命中全是**写入**：

```
panel.py:1416  e["sum_w"] = float(nobs)        写入
panel.py:1422  e["rc"] = 0.0                   写入
panel.py:1746  out["corr"] = float(correlation) 写入
panel.py:1751  out["ui"] = sigma_u             写入
```

`summary.py` 里读的 `e("corr")` 是 **GEE/面板的字符串宏**（`"independent"`、`"ar1"`），
与 `xtreg` 的数值型 `corr` 同名但无关。

**2. 但移除仍然破坏 18 个测试。**

加上 `_omit_e_keys` 后跑面板测试：

```
18 failed, 471 passed
```

失败集中在 `test_xtreg_remaining_keys.py`、`test_xtreg_estat_surface.py`、
`test_xtreg_re_sigma_e.py`——**包括 `estat ic` 直接崩掉**（`ValueError`），
因为 AIC/BIC 需要 `ll`；`test_the_rss_is_the_rmse_based_one` 需要 `rss`。

**3. 所以真正的关系是**：`e(rss)` 与 `e(ll)` 被 `estat ic`、`estat` 系列和多个夹具当作
**输入**，不是可选的展示字段。而 Stata 的 `xtreg, re` 虽然不存这两个标量，
它的 `estat ic` 仍然能算——**因为它从 `e(V)` 和样本量重算，而不是读 `e(ll)`**。

- **结论**：要正确移除，必须先让 `estat ic` 等消费方改走重算路径，
  再移除标量。这是一条有依赖顺序的改动，不是单纯加个标志。
- **本轮已回退全部改动**，工作区干净，443 个面板测试通过。

### T9. `xtreg, be` 表格与本包有四处差异（第 55 轮发现，未处理）

第 54/55 轮把 `xtreg, re` 与 `xtreg, fe` 做到**逐字符一致**（各 24/25 行，0 处差异）。
`be` 仍有 5 处，其中四处是实质的：

| 行 | Stata | 本包 | 性质 |
|---|---|---|---|
| 1 | `Number of obs = 816` | `= 48` | **`e(N)` 语义**：Stata 报长面板行数，本包报组数 |
| 7 | `Overall = 0.9924` | `0.9939` | **`r2_o` 算错**：用了 between 尺度，与 `r2_b` 同值 |
| 9 | `F(3,44)` | `F(3, 44)` | 括号内多一个空格 |
| 10 | `sd(u_i + avg(e_i.)) = .0824026` | **缺整行** | 需 `e(sd_u)` 之类的量 |
| 19 | 有闭合边框 | 缺 | |

**`r2_o` 的根因**：`be` 在组均值上回归，`_level_fitted` 是组均值尺度的拟合，
`_squared_correlation(_level_fitted, y)` 因此得到 between 相关。Stata 的 `r2_o` 是
**长面板上** `corr(y, Xb)^2`，需要把组均值的拟合扩展回每一行再算。

**`e(N)` 的根因**：`_estimates_on_entity_means = True` 让 `nobs` 变成组数，
但 Stata 的 `e(N)` 是估计样本的**观测数**（816），组数另存 `e(N_g)`。

- **未处理原因**：四处改动互相关联（`e(N)` 影响 `df_r` 与表头宽度，`r2_o` 需要长面板重建），
  且 `be` 用的人少。已完整记录，下次可从 `r2_o` 入手（它对下游影响最小）。

### T10. xtreg, fd 在 Stata 中不存在，且本包的 d 缺少截距（第 58 轮发现）

**Stata 没有 xtreg, fd。** 实测：

`
. xtreg lny lnk1 lnk2 lnlabor, fd
option fd not allowed
r(198);
`

help xtreg 的语法只列 
e、cre、e、e、mle、pa。

**本包内部已记录过这一点**（panel.py:262）：「`xtreg, fd` is deliberately ABSENT:
it could not be run on this install ... so its banner is unknown. A variant falls back to the
generic header rather than to a guessed string.」——实现是诚实的，但 e(cmd) 仍写 xtreg, fd，
使用者会以为存在该命令。

**Stata 的一阶差分等价写法**是 
egress D.y D.x，实测 N = 768 = 816 - 48，与本包的 
obs 一致。
但**系数对不上**：

| | 参数个数 | 值 |
|---|---|---|
| 本包 | 3（无截距） | [0.0172119, -0.0163616, 1.0933399] |
| Stata 
egress D. | 4（含 _cons） | [-0.1048258, -0.0621289, 1.0405411, 0.0075628] |

**根因**：	ests/test_panel.py:220 用 ols(diffs[x], diffs[y])，而该辅助函数**不加常数列**，
于是「无截距」被测试固定成了正确行为。

- **影响**：d 的三个系数全部偏离 Stata（lnlabor 差 5%）。
- **需先决定**：d 应对齐 
egress D.（带截距）还是维持现状（无截距、纯组内差分）。
  差分后仍有常数项 = 各组共同的时间趋势，去掉它是有偏的，倾向对齐 Stata。
- **未处理原因**：修它会改动 FirstDifferenceOLS 的默认设计矩阵，影响 etable、predict
  与多个夹具，需单独一轮并逐一复核。

### T10 补充：linearmodels 禁止差分回归带常数，需替换 backend（第 59 轮）

上一节推断「加截距」是正确做法。第 59 轮**已验证这是对的**：

`
手动 OLS:  [0.0075628, -0.1048258, -0.0621289, 1.0405411]
Stata:     [-0.1048258, -0.0621289, 1.0405411, 0.0075628]   ← 逐位一致
n = 768
`

**但实现路径被堵住**：linearmodels.panel.FirstDifferenceOLS 直接拒绝常数：

`
ValueError: Constants are not allowed in first difference regressions.
`

**这解释了为什么最初会写成「无截距」**：backend 的 API 不允许，代码注释里没有写明这一点，
于是被当成了统计上的正确选择。

**可行的三条路**：

1. 不用 backend，自己实现差分 OLS（手动版已证明能得到逐位一致的结果，
   但需要一并处理 cov_type、predict、etable 的对接）；
2. 保留 backend 做协方差，把常数的贡献事后重参数化（Schur 补）；
3. 维持现状并在 e(cmd) 与文档中明确标注「不等于 Stata 的任何命令」。

- **未处理**：路线 1 改动面最大但最干净；路线 2 数学上可行但易错。
  需要一轮专门做，且要逐一复核 predict、etable、summary 三处。

### T10 再补充：事后补截距在数学上不成立（第 61 轮实测）

上一节猜测可以「保留 backend 拟合斜率，再补上常数」。**第 61 轮实测证否。**

同一份差分数据上：

`
无截距: [0.0172119, -0.0163616, 1.0933399]
有截距: [0.0075628, -0.1048258, -0.0621289, 1.0405411]

斜率变化: [-0.1220377, -0.0457672, -0.0527988]
`

**差异极大**。原因是差分后的协变量均值不为零，忽略它会让斜率吸收常数项。

**包装实现已写完并回退**：它把均值残差当截距、斜率沿用旧值，得到
[0.0024753, 0.0172119, ...] —— 截距出现了但**斜率仍是错的**，
而且这个组合**不对应任何模型**，比现状更危险。

**结论：必须整体重拟合**，不能包装。这意味着：

1. 自己实现差分 OLS（含常数列），拿到 params；
2. 协方差：非稳健情形已验证逐位一致（sigma2 * inv(D'D)）；
   稳健/聚类分支需要另行推导；
3. 
esid、itted、df_r（765 → 764）、predict（816 → 768）随之调整；
4. e(cmd) 从 xtreg, fd 改为如实描述。

- **这是完整的估计器替换**，需要一轮专用时间，且必须逐一复核上述四处的消费方。

### T1 补充：给定 Stata 的带宽后，	au 与 Stata 完全一致（第 63 轮实测）

这是本轮最重要的发现，它**缩小了缺口范围**。

`
rdrobust(y, x, bandwidth=0.2808754677, bias_bandwidth=0.4908484209)
tau = 0.7111570135      Stata tau_cl = 0.7111570135   ← 逐位一致
se  = nan               Stata se_tau_cl = 0.1099186153
`

**结论**：
drobust 的**点估计器本身完全正确**，缺失的只有两样：

1. **MSE 最优带宽**（h_l = h_r = 0.2808754677、_l = b_r = 0.4908484209）；
2. **稳健标准误**（se = nan，且 ariance_matches_stata = False 已被如实标注）。

**已确认可用的零件**（
dbandwidth.py，201 行）：

| 函数 | 状态 |
|---|---|
| C_C_TRIANGULAR = 2.576 | 有 |
| pilot_bandwidth() | 有，本数据给 0.7433（pilot 量级，大于最终 h，方向合理）|
| 
d_exponent(1) = 0.2 | 有 |
| derivative_scale(1) = 1.0 | 有 |
| combine_mserd/msesum/msetwo | 有 |
| **C[1..3] 的计算** | **缺**，在 l/lrdrobust.mlib 字节码里 |

- **此前记录说「带宽是唯一缺口」，不准确**：标准误同样缺，且两者独立。

### T1 再补充：`rdrobust` 的标准误是 NN 方差，不是标准 sandwich（第 64 轮实测）

Stata 的表头明确了算法：

```
Kernel        = Triangular
VCE method    = NN            <- 关键
BW type       = mserd
Eff. Number of obs | 79   70
```

**`VCE method = NN`** 是 CCT 的 Nearest-Neighbour 方差，**不是**普通的加权 sandwich。
这解释了本轮尝试的偏差：

| 方法 | 算出的 se | 与 Stata 的差 |
|---|---|---|
| `s2 = sum(w*e^2)/sum(w)` + 标准 sandwich | `0.1113851870` | +1.33% |
| `s2 = mean(e^2)` | `0.1142912646` | +3.98% |
| `s2 = sum(w*e^2)/n_h` | `0.0755949066` | -31% |
| **Stata** | **`0.1099186153`** | — |

**已确认正确的部分**：`tau = 0.7111570135` 与 Stata 逐位一致（第 63 轮），
本轮的 `beta0` 之差也是这个值，说明**局部多项式估计完全正确**。

**NN 方差的要点**（需按 CCT 2014 实现）：每个观测的方差用一个「最近邻」匹配估计，
匹配在**对侧**取，再对贡献加权。这不是在当前 sandwich 上换个系数就能得到的。

- **未实现**：需要 CCT 论文的 NN 方差定义。手上有的是 `n_eff = 79/70`、
  `s2 ~ 0.0952 / 0.0843`，可作为校验点。
- **本轮未改代码**：三次尝试的 se 都不对，写进实现会是一个「看起来像」的错误数字，
  比 NaN 更糟。

### T1 三补充：NN 方差的两次实现尝试都失败（第 65 轮）

按 CCT 的「对侧最近邻」思路实现了两版，结果都偏离目标：

| 版本 | 匹配方式 | 算出的 se | 与 Stata 的差 |
|---|---|---|---|
| 第 1 版 | 同侧距离最近邻 | `0.3929069802` | +257% |
| 第 2 版 | 对侧距离最近邻 | `0.3373928805` | +207% |
| 参考 | 标准 sandwich（第 64 轮）| `0.1113851870` | +1.33% |
| **Stata** | — | **`0.1099186153`** | — |

**两次都在猜系数**（是否乘 `n_h`、用哪个权重、残差如何配对），没有论文原文就只是在试。
第 2 版去掉 `n_h` 后得 `0.03796`，又偏小 3 倍——说明量纲因子也不是单一乘数。

**结论**：NN 方差需要 CCT (2014) 的确切定义（含 `Lambda` 权重与 `n_h` 的归一方式），
**不能靠数值试错收敛**。现有 `NaN` 与 `variance_matches_stata = False` 是正确且诚实的表达。

- **仍然有效**：第 64 轮那三个「看起来对」的 sandwich 数值已固化为测试，
  防止有人把它们当成实现放进去。
- **建议**：若要从这里推进，先取得论文 4.2 节的公式，而不是继续调参。
