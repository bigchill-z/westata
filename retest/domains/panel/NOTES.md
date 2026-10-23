# panel 复测结论

复测对象：`westata` 的 xt 族（`estimators/panel.py`、`panel_iv.py`、`panel_nonlinear.py`、
`panel_dynamic.py`）vs **StataNow 19.5 MP**（本机内嵌 pystata，`c(stata_version)=19.5`）。

- 可重跑脚本：`retest/domains/panel/run.py`
- 原始证据：`results.jsonl`（42 条，逐条含 `extra` 里的两侧全精度数字）、`summary.json`
- 判定口径：见 `retest/BRIEF.md` 2.3。闭式线性代数用 **A 档 1e-9**，迭代估计（ML/GMM/GEE）
  用 **B 档 1e-6**。标准误用 `e(V)` 对角线开方（Stata）与 `res.bse`（westata），不用打印值。
- 运行环境：Python 3.10.11（`.venv`）、linearmodels 7.1、westata 仓库现状（2026-10-24）。

## 数据（全部本脚本现造，固定 seed，不读 tests/fixtures/registry）

| 名称 | seed | 规模 | 生成规则 |
|---|---|---|---|
| 平衡面板 | 2026 | 30 × 6 = 180 | `y = 1 + 0.8x1 − 0.5x2 + u_i + e_it`，`u_i~N(0,1)`，`e~N(0,0.5²)`，`x1,x2,z~N(0,1)`；`yb=1{y+logistic>0.9}`，`yc~Poisson(exp(0.5+0.4x1−0.2x2+0.3u))`，`ytob=max(0,y)` |
| 非平衡面板 | 777 | 20 个个体，4~8 期，共 128 | 同一 DGP，`t≥3` 后逐期 35% 概率缺测；`t=0,1,2` 全观测（保证 xtpcse casewise 共同期非空） |
| 小平衡面板 | 31337 | 6 × 8 = 48 | 平衡 DGP，规模小到 `m < T`，让 `xtgls panels(correlated)` 的 Σ 满秩 |
| theta=0 面板 | 99 | 25 × 6 = 150 | `y = 1 + 0.8x1 − 0.5x2 + e`，**无个体效应**（σ_u=0） |
| 动态面板 | 5150 | 60 × 8 = 480 | `y_it = 0.5 y_i,t−1 + 0.8x1 − 0.4x2 + u_i + e_it`，`y_i0=0` |
| 单位根面板 | 4242 | 25 × 12 = 300 | `y_t = y_{t−1} + e_t`，`y_0=0`，`e~N(0,1)` |
| 平稳面板 | 4242 | 25 × 12 = 300 | `y_t = 0.4 y_{t−1} + e_t`（与上一行同 seed，只差 AR 系数） |

Stata 侧一律 `st.send(df)` 后 `xtset id t`；每条的样本量都核对过：`e(N)` 与 westata `nobs`
逐条相等（唯一例外是 `xtreg, be`，见下文，那是 e(N) 语义不同而非样本不同）。

## 覆盖矩阵

| # | Stata 命令 | westata 调用 | 数据 | 判定 | 系数相对差 | 标准误相对差 |
|---|---|---|---|---|---|---|
| 1 | `xtset id t` | —（无对应步骤） | 平衡 | pass | 0 | — |
| 2 | `xtreg y x1 x2, fe` | `PanelOLS.from_formula(...).fit()` | 平衡 | **A** | 1.27e-16 | 1.85e-16 |
| 3 | `xtreg, fe` 的 `e(r2_w)` / `e(rmse)`↔`e(sigma_e)` | 同上（读 `res.e`） | 平衡 | **A** | 0 | 0 |
| 4 | `xtreg, fe` 的 `e()` 覆盖缺口 | 同上 | 平衡 | **fail（Stata 有、westata 无）** | — | — |
| 5 | `xtreg y x1 x2, fe` | 同上 | 非平衡 | **A** | 5.26e-16 | 0 |
| 6 | `xtreg y x1 x2, fd` | — | 平衡 | **skip** | — | — |
| 7 | `regress D.y D.x1 D.x2, noconstant`（fd 的等价口径） | `FirstDifferenceOLS` | 平衡 | **A** | 4.18e-16 | 1.71e-16 |
| 8 | `xtreg y x1 x2, be` | `BetweenOLS` | 平衡 | **A** | 4.01e-15 | 4.07e-16 |
| 9 | `xtreg y x1 x2, re`（默认 GLS） | `RandomEffects` | 平衡 | **A** | 6.00e-16 | 3.15e-16（e(V) 全矩阵 2.36e-15） |
| 10 | `xtreg, re` | 同上 | 非平衡 | **A** | 2.61e-16 | 2.90e-16 |
| 11 | `xtreg, re` 的 theta→0 边界 | 同上 | theta=0 面板 | **A** | 2.64e-16 | 0 |
| 12 | `xtreg y x1 x2, fe vce(robust)` | `PanelOLS.fit(cov_type="HC1")` | 平衡 | **E** | 1.27e-16 | 17.36 |
| 13 | `xtreg y x1 x2, fe vce(cluster id)` | `PanelOLS.fit(cov_type="cluster")` | 平衡 | **E** | 1.27e-16 | 9.36e-2 |
| 14 | 归因 A：Stata `vce(cluster id) dfadj` | westata `cluster` | 平衡 | **A** | 2.54e-16 | 2.15e-15 |
| 15 | 归因 B：手工 CR1 `G/(G-1)·(N-1)/(N-K)`, K=3 | — | 平衡 | **A** | 1.27e-16 | 8.76e-15 |
| 16 | 归因 C：linearmodels `auto_df=True` | — | 平衡 | **A** | 1.27e-16 | 9.40e-15 |
| 17 | 归因 D：`areg …, absorb(id) vce(robust)` | westata `HC1` | 平衡 | **A** | 0 | 0 |
| 18 | `xtgls y x1 x2, panels(heteroskedastic)` | `XtGls(panels="heteroskedastic")` | 平衡 | **A** | 1.53e-15 | 5.68e-16 |
| 19 | `xtgls y x1 x2, panels(correlated)` | `XtGls(panels="correlated")` | 6×8 | **A** | 2.81e-15 | 6.70e-15 |
| 20 | `xtgls, panels(correlated)` | 同上 | 30×6（Σ 秩亏） | **E** | 2.12e-1 | 3.39 |
| 21 | `xtgls …, panels(heteroskedastic) corr(ar1)` | `XtGls(corr="ar1")` | 平衡 | **error**（westata 拒绝） | — | — |
| 22 | `xtpcse y x1 x2` | `XtPcse` | 平衡 | **A** | 1.10e-15 | 2.98e-15 |
| 23 | `xtpcse y x1 x2`（casewise） | `XtPcse(selection="casewise")` | 非平衡 | **E** | 6.16e-16 | 2.81e-1 |
| 24 | `xtpcse y x1 x2, pairwise` | `XtPcse(selection="pairwise")` | 非平衡 | **E** | 6.16e-16 | 4.19e-1 |
| 25 | `xtivreg y x2 (x1 = z), fe` | `XtIvreg(method="fe")` | 平衡 | **A** | 8.23e-15 | 2.04e-15 |
| 26 | `xtivreg y x2 (x1 = z), re` | `XtIvreg(method="re")` | 平衡 | **A** | 4.74e-15 | 1.23e-15 |
| 27 | `xtlogit yb x1 x2, fe` | `XtLogit(method="fe")` | 平衡 | **B/C** | 3.70e-8 | 3.40e-6 |
| 28 | `xtlogit yb x1 x2, pa` | `XtLogit(method="pa")` | 平衡 | **C/D** | 3.19e-4 | 1.06e-3 |
| 29 | `xtlogit yb x1 x2, re` | `XtLogit(method="re")` | 平衡 | **B/C** | 1.77e-7 | 1.91e-4 |
| 30 | `xtprobit yb x1 x2, re` | `XtProbit(method="re")` | 平衡 | **B/C** | 6.05e-7 | 1.16e-4 |
| 31 | `xttobit ytob x1 x2, ll(0)` | `XtTobit(lower=0)` | 平衡 | **C/C** | 2.52e-6 | 7.49e-6 |
| 32 | `xtpoisson yc x1 x2, fe` | `XtPoisson(method="fe")` | 平衡 | **B/C** | 4.42e-8 | 1.36e-5 |
| 33 | `xtpoisson yc x1 x2, re` | `XtPoisson(method="re")` | 平衡 | **B/C** | 5.29e-8 | 1.78e-4 |
| 34 | `xtnbreg yc x1 x2, fe` | `XtNbreg(method="fe")` | 平衡 | **D/E** | 1.36e-3 | 6.79e-2 |
| 35 | `xtabond y x1 x2, lags(1)` | `XtAbond(lags=1)` | 动态面板 | **D/E** | 2.03e-3 | 9.08e-1 |
| 36 | `xtabond y x1 x2, lags(2)` | `XtAbond(lags=2)` | 动态面板 | **E/E** | 3.49e-2 | 9.40e-1 |
| 37 | `xtabond …, lags(2)` 的参数命名 | 同上（读 `param_names`） | 动态面板 | **fail（命名错位）** | — | — |
| 38 | `xtabond y x1 x2, lags(1)` | `XtAbond(lags=1)` | 静态面板（弱识别） | **D/E** | 5.06e-3 | 1.99e-1 |
| 39 | `xtunitroot ips y` | `XtUnitroot(test="ips")` | 单位根面板 | **A** | 6.29e-15 | — |
| 40 | `xtunitroot llc y` | `XtUnitroot(test="llc")` | 单位根面板 | **A** | 2.72e-15 | — |
| 41 | `xtunitroot ips y` | 同上 | 平稳面板 | **A** | 1.45e-14 | — |
| 42 | `xtunitroot llc y` | 同上 | 平稳面板 | **A** | 4.83e-16 | — |

合计 **42 个对比点**：pass 22、fail 18、skip 1、error 1。
其中 6 条是**归因/诊断条目**而非"命令对命令"的数值对比：
#4（`e()` 覆盖缺口）、#14–#17（`vce(cluster/robust)` 的四步归因）、#37（`xtabond lags(2)` 命名缺陷）。
真正的命令对命令数值对比是 36 条。

## 逐项说明

### 1. 完全达 A 档的（可称"完全一致"）

`xtreg fe`（平衡/非平衡）、`xtreg be`、`xtreg re`（平衡/非平衡/theta=0）、`regress D.`
（fd 等价口径）、`xtgls heteroskedastic`、`xtgls correlated`（满秩）、`xtpcse`（平衡）、
`xtivreg fe/re`、`xtunitroot ips/llc`（有/无单位根）。

系数差都在 1e-15 ~ 1e-14，e(V) **全矩阵**（不只对角线）也在 1e-15：
- `xtreg, fe vce(cluster id)` 的归因链，见下。
- `xtreg, re`：`e(V)` 全矩阵 max rel = 2.36e-15，说明 linearmodels 的 Swamy-Arora 与
  Stata 的 GLS theta 是同一个东西（不是"接近"）。westata 的 `e(r2)=0.712731` 是它自造的
  第三个量（Stata 的 r2_w=.746877、r2_b=.116620、r2_o=.394592 都不是它），这一点在
  `e()` 覆盖缺口里另列。
- `xtivreg, fe`：Stata 的 within 变换是 `w − w̄_i + w̄`（常数项因此可估，
  `_cons=0.94023`，SE 0.0565），westata 复现到 8e-15。这一点常被写错成"within 变换会消灭常数"，
  实测证明 westata 的处理是对的。
- `xtunitroot`：Stata 是 r() 类命令，westata 用 `PanelUnitRootResult` 承载（不伪造 e(b)/e(V)），
  容器选择正确。`t-bar`、`t-tilde-bar`、`Z-t-tilde-bar`、`Adjusted t*`、`p`、LLC 的
  `sbar/Var_ep/mu_adj/sig_adj` 全部一致到 1e-15，p 值判决也一致
  （单位根面板：IPS p=0.0138、LLC p=0.0000；平稳面板：IPS p=9e-6、LLC p=0.0000 ——
  两组数据上"是否拒绝 H0"的结论都与 Stata 相同）。

### 2. `xtreg, fe vce(robust)` 与 `vce(cluster id)` —— 最严重的一条（E 档，17 倍）

**现象**

| 量 | Stata | westata | 比值 |
|---|---|---|---|
| b(x1, x2, _cons) | .785510 / −.437493 / .924831 | 同 | 1.00 |
| SE(x1) robust | .0468177 | .0422014 | 0.901 |
| SE(x2) robust | .0374282 | .0413346 | 1.104 |
| SE(_cons) robust | **.0020300** | **.0372797** | **18.36** |
| SE(x1) cluster | .0468177 | .0511995 | 1.094 |
| SE(_cons) cluster | **.0020300** | **.0022200** | 1.094 |

两个独立的问题：

**(a) `vce(robust)` 的语义映射错了。** Stata 的 `[XT] xtreg` 明确写
"Specifying `vce(robust)` is equivalent to specifying `vce(cluster panelvar)`"，
实测两者 `e(V)` 完全相同（max abs diff = **0**）。westata 把 `robust` 映射成
linearmodels 的 `cov_type="robust"` = **HC1**（观测级三明治，不聚类）。
这两者在有强个体效应时不是一回事：常数项的 SE 差 **18.36 倍**
（.0020300 vs .0372797；x1 的 SE 反而小 10%，x2 大 10%）。
证据：同一份数据上 `areg y x1 x2, absorb(id) vce(robust)` 给出
`.0422014 / .0413346 / .0372797`，与 westata 的 HC1 逐位相同（4.9e-16）——
**westata 的 HC1 实现没问题，错的是把它当成 xtreg 的 `vce(robust)` 来卖。**

**(b) `vce(cluster id)` 的小样本因子用错了 K。** westata 的 cluster SE 相对 Stata
统一偏大 **9.359%**（三条系数同一个因子，不是随机差异）。

归因链（逐条都跑过、都进了 results.jsonl）：
1. westata 的 cluster SE 与 Stata 的 **`vce(cluster id) dfadj`** 逐位相同（2.1e-15）。
   `dfadj` 的含义是"用 `(N−1)/(N−P)`，P 含被吸收的面板效应"，即 K=32（3 系数 + 29 个体效应）。
2. 手工 `G/(G-1)·(N-1)/(N-K)`，K=3，`√` 三明治 → 与 Stata 默认逐位相同（8.8e-15）。
3. 同一句 linearmodels `fit()`，只把 `auto_df` 从 westata 固定用的 `False` 改成 `True`
   → 与 Stata 逐位相同（9.4e-15）。
4. `sqrt(((N-1)/(N-3)) / ((N-1)/(N-32))) = sqrt(177/148) = 1.09359`，正是实测放大因子。

**根因在 `src/westata/estimators/panel.py::PanelOLS._extra_fit_kwargs`**：
它强制 `auto_df=False` 并注释说"`auto_df=True`（后端默认）在效应嵌套于聚类变量时会漏掉效应"。
实测相反：`auto_df=True` 时 linearmodels 的 K 只数报告的系数，正是 Stata 的口径；
`auto_df=False` 才把吸收的效应算进 K。**注释与事实相反，代码跟着注释错了。**

**另有一条只用 e() 就能看出来的不一致**：`e(df_r)`，Stata 聚类后报 **29 = G−1**，
westata 报 **148 = N−K**。这会让所有 t/z 检验的 p 值和置信区间跟着错
（westata 的 `StataResults.dist="t"` 用 `df_resid` 做参照分布）。

**是否影响结论**：系数点估计不受影响；`vce(robust)` 下常数项的检验统计量从
t=455.6（Stata）变成 t=24.8（westata），都"显著"，但置信区间宽度差 18 倍——
任何引用"常数项 ±1.96·SE"的代码都会给出完全不同的区间；
`vce(cluster)` 下 SE 系统性大 9.36%，边缘显著的结果可能翻转。**必须修。**

### 3. `xtpcse` 非平衡（casewise / pairwise）—— E 档，SE 全错

**现象**：系数逐位一致（6e-16），标准误错到第 1 位。

| 参数 | Stata casewise | Stata pairwise | westata（两种 selection 相同） |
|---|---|---|---|
| _cons | .0370215 | .0447770 | .0304356 |
| x1 | .0485098 | .0569808 | .0409808 |
| x2 | .0476406 | .0589739 | .0342562 |

westata 相对 Stata：casewise 下三个 SE 是 Stata 的 **0.82 / 0.84 / 0.72 倍**
（相对差 0.178 / 0.155 / 0.281，最大 0.281 = `se_rel` 记录值）；
pairwise 下更低（0.68 / 0.72 / 0.58，相对差 0.320 / 0.281 / 0.419，最大 0.419）。
不是统一因子，是 Σ 与 meat 的结构都不同。

**归因（读 `xtpcse.ado::getSigma` 并用 Python 独立复算验证）**：
Stata 的 `casewise` 只用"**所有面板共同期**"的残差估 Σ（本数据共同期为 `t=0,1,2`，
Stata 打印 `e(n_sigma)=3`，并警告它小于 `e(n_avg)=6.4`、建议改 `pairwise`），
但 `X'ΩX` 用**全部**观测累加。westata 的 `XtPcse._aligned_slices()` 用

```python
common = min(index.size for index in slices)
return [index[:common] for index in slices]
```

按**行号**取每个面板的前 k 行，不是按共同期取；而且它把 Σ **和** meat 都截断了。
两处都错，所以即便系数对，SE 也对不上。

**pairwise 根本没实现**：westata 接受 `selection="pairwise"`，但
`res_pw.bse` 与 `res_cw.bse` **逐位相同**（`westata_pairwise_equals_casewise=True`）。
Stata 的 pairwise 用 `T_ij` 逐对加权，给出与 casewise 不同的 SE
（x1 .056981 vs .048510，_cons .044777 vs .037021）。
这是一个"静默的同名不同量"——比报错更危险，因为用户以为换了更稳健的选项。

**是否影响结论**：不影响系数，但 PCSE 的全部价值就在 SE。TSCS 数据上
pairwise 是常用选项（Stata 自己在 `n_sigma` 太小时主动建议），westata 给出的是 casewise 的数
却挂着 pairwise 的名字。**必须修。**

### 4. `xtgls, panels(correlated)` 在 30×6 上（秩亏）—— 归因为"不可比"

Stata 打印 `results are based on a generalized inverse of a singular matrix`，
因为 30 个面板只有 6 期，Σ(30×30) 秩 ≤ 6。伪逆在零空间上不唯一，
两边给出不同答案（b 差 2.1e-1，SE 差 3.4 倍）。这不是数值精度问题，是问题本身无唯一解。
**在 6×8（m<T，Σ 满秩）的配置上 westata 是 2.8e-15 精确的**，所以实现本身没问题，
需要的是文档警告："`panels(correlated)` 只在 m<T 时有意义"。

### 5. `xtlogit, fe` —— 样本对齐是干净的，残差在数值 Hessian

**重点核对项**：`e(N)`、`e(N_g)`、`e(N_drop)`、`e(N_group_drop)` 四个数**全部相同**：

```
Stata   : N=156  N_g=26  N_drop=24  N_group_drop=4
westata : N=156  N_g=26  N_drop=24  N_group_drop=4
```

24 = 4 组 × 6 期，说明被丢弃的全 0/全 1 组是**同样的 4 个**（30 组中的 4 组），
westata 没有静默换样本。`e(ll) = −61.585071` 逐位相同（条件似然是闭式的）。

残差：系数 3.7e-8、SE 3.4e-6（B/C 档）。来源是 westata 用数值 Hessian、Stata 用解析 OIM。
这个量级不影响任何结论。

### 6. `xtlogit, pa`（GEE）—— C/D 档，westata 已知缺口，本次独立复现

系数相对差 3.2e-4（`x1` .328897 vs .329002），SE 相对差 1.1e-3。
westata 的 `alpha=0.12209` 是 Pearson 型交换相关估计；Stata 的 GEE 用另一种更新，
系数残差就来自这里。不影响显著性（p 0.00240 vs 0.00241）。

westata 不报 `e(ll)/e(ll_0)/e(aic)/e(r2_p)`——GEE 无似然，这是**正确**做法，
Stata 在这里也不报 ll。这一点 westata 比很多实现诚实。

### 7. `xtlogit, re` / `xtprobit, re` / `xttobit` / `xtpoisson`

全部在 **B/C 档**，量级 1e-7 ~ 2e-6（系数）、1e-5 ~ 1.9e-4（SE）：
- RE 的 12 点自适应 Gauss-Hermite（`mvaghermite`）复现良好：
  `sigma_u`、`rho`、`ll` 都在 1e-6 内（logit: `ll` −115.8610372 两位相同）。
- `xttobit` 的 `/sigma_u`（.953538 vs .953540）、`/sigma_e`、`rho`、`ll` 全在 1e-6 内；
  左截尾 45 个观测，westata 用 `n_left=45/n_uncensored=135` 明确标出。
  （额外做了 12/20/50 点的敏感性：系数只在小数第 7 位变动，**50 点并不比 12 点更接近 Stata**
  ——自适应求积已收敛，残差来自似然实现本身，不是点数不够。）
- `xtpoisson, fe` 系数 4.4e-8、`ll` 逐位相同（条件 Poisson 是闭式的）。
- `xtpoisson, re` 的 `/lnalpha` SE 差 1.8e-4（.50846 vs .50837）；`alpha` 两边都是 .09702。

**结论：这一族可用，但不要把 SE 报到 4 位有效数字以外。**

### 8. `xtnbreg, fe` —— D/E 档，westata 实际只做了条件 Poisson

| 命令 | Stata b(x1) | Stata SE(x1) | Stata ll |
|---|---|---|---|
| `xtpoisson yc x1 x2, fe` | .3849351 | .0659024 | −216.94487 |
| `xtnbreg yc x1 x2, fe` | .3854574 | **.0707036** | −216.43859 |
| **westata `xtnbreg, fe`** | .3849351 | **.0659018** | **−216.94487** |

westata 的 `xtnbreg, fe` 与它**自己的** `xtpoisson, fe` 给出逐位相同的系数和 SE
（.38493508 / .06590183），而 Stata 的两个命令并不相同：点估计只差 1.4e-3，
但 SE 差 **6.8%**（.07070 vs .06590，E 档）。也就是说：westata 只实现了条件 Poisson 的 OIM，
把它当作 HHG 拟似然的答案。系数上差 1.36e-3（D 档，可接受），**SE 上差 6.79e-2（E 档）**。

另外：Stata 报 `_cons = 2.6280`（SE 1.1014），那是被固定效应吸收的平坦方向；
westata 选择不报——**这个处理是正确的**，已在比对前显式剔除，不计入判定。

`ll`：westata 文档称 `xtnbreg, fe` 的 `e(ll)` 与 `xtpoisson, fe` 几乎相同
（fixture 上 −775.44069 vs −775.44068）。本数据上 westata 报 −216.94487（正是条件 Poisson 的 ll），
Stata 的 `xtnbreg, fe` 报 −216.43859，**相对差 2.3e-3**（旧说法"几乎相同"在本数据上不成立，
但差距比系数/SE 小一个量级）。

### 9. `xtabond` —— D/E 档，三个独立缺陷

在**真动态**面板（60×8，AR 真值 0.5）上：

| 量 | Stata | westata | 相对差 |
|---|---|---|---|
| L1.y | .4700769 | .4695245 | 1.2e-3 |
| x1 | .8024287 | .8040540 | 2.0e-3 |
| x2 | −.4078528 | −.4082143 | 8.9e-4 |
| SE(L1.y) | .0605331 | **.1154721** | **0.91** |
| 工具数 (`zrank`) | 24 | 24 | 对上 |
| N | 360 | 360 | 对上 |
| `e(sig2)` | .2517342 | **.5017229** | **1.993×** |

三个缺陷：

**(a) σ² 差整整 2 倍。** westata 的 `e(sigma2) = 0.501723`，Stata 的 `e(sig2) = 0.251734`，
比值 **1.993**。差分误差 `Δe` 的方差是 `2σ²`；westata 直接拿 `Δe'Δe/(N−k)` 当 σ²，
没有除以 2，于是整个 V 放大 2 倍、SE 放大 √2 ≈ 1.41 倍。实测 SE 比值 1.91（L1.y），
比 1.41 还大，说明 meat 也有问题。`lags(2)` 上同一比值 **2.025** 重复出现
（.546440 vs .269886）——系统性的常数因子，不是随机误差。

**(b) `lags(2)` 的参数名整体错位一位（比数值错误更危险）。**

```python
# panel_dynamic.py
param_names=(f"L{lags}.y",),     # lags=2 时只有 "L2.y" 一个名字
```

但设计矩阵有 L1.y 与 L2.y 两列。实测 `param_names = ['L2.y', 'x1', 'x2', 'extra_0']`，
真值是 `[L1.y, L2.y, x1, x2]`。后果：真 L1.y 被标成 L2.y、真 L2.y 被标成 x1、x2 被标成
`extra_0`。任何按名字取系数的下游代码（`test L2.y`、`lincom`、报告表）都会**取错变量且不报错**。
`lags=1` 时恰好正确，所以只有多滞后才暴露。**这条比数值偏差严重，必须修。**

**(c) 弱识别时偏差被放大。** 同一份静态面板（y 无真实动态）上 Stata L1.y=.063417、
westata L1.y=.050982（1.96e-1）；动态面板上是 1.2e-3。说明 westata 的偏差随识别强度变化，
不是常数偏移。

另：Stata 报 `Sargan=25.13`，westata 明确不报 Sargan/Hansen/AR(1)/AR(2)——**已知缺口但属诚实处理**
（近似值挂 Stata 的名字比缺失更糟）。Stata 默认 `vce(gmm)` 与 `vce(robust)` 的 SE 明显不同
（常数项 .0259 vs .1240），westata 只有一种 VCE。

### 10. e() 标量覆盖缺口（不影响系数，影响"拿 e() 做后续计算"）

独立成条的是 results.jsonl 的 **#4**（`xtreg, fe` 的 e() 覆盖缺口）与 **#3**
（`xtreg, fe` 的 `e(r2)`↔`e(r2_w)`、`e(rmse)`↔`e(sigma_e)` 数值一致）。

`#4` 的实测内容（本数据，30×6）：
```text
Stata   : r2_w=.7469091329  r2_b=.1153385025  r2_o=.3939678015
          sigma_u=.9742361557  sigma_e=.5024676468  rho=.7898870905  N_g=30  df_m=31
westata : r2=.7469091329（就是 Stata 的 r2_w）
          r2_b=None  r2_o=None  sigma_u=None  rho=None  N_g=None  df_m=2
```

| 命令 | Stata 有、westata 没有 | 说明 |
|---|---|---|
| `xtreg, fe` | `e(r2_b)`、`e(r2_o)`、`e(sigma_u)`、`e(rho)`、`e(N_g)` | 可核对的 `e(r2_w)`(=e(r2)) 与 `e(sigma_e)`(=e(rmse)) 一致到 1e-16；**名字也不同**（westata 用 `r2`/`rmse`，Stata 用 `r2`(=r2_w 是它的别名)/`r2_w`/`sigma_e`） |
| `xtreg, re` | `e(theta)`、`e(sigma_u)`、`e(rho)`、`e(r2_w/r2_b/r2_o)` | **`theta` 是最重要的缺失项**，见下；westata 自造的 `e(r2)=.712731` 与 Stata 的 r2_w/.746877、r2_b/.116620、r2_o/.394592 都不等 |
| `xtreg, be` | `e(N)` 语义不同：Stata=180（原始观测），westata=30（组均值） | 文档已声明，但同名不同义，下游拿 e(N) 算自由度会差 6 倍 |
| `xtreg, fe` | `e(df_m)` 语义不同：Stata=31（含 29 个效应列），westata=2（只数斜率） | westata 的 `e(F)` 却与 Stata 相同（218.385），说明它内部知道该用哪个 df |
| `xtivreg, fe` | `e(r2_w)` 对不上：Stata .453442、westata .510597 | westata 用水平 y 的总平方和，Stata 用 within 平方和（已用 Python 复算确认 Stata 的口径） |
| `xtivreg, fe/re` | `e(chi2)` 是**不同的检验**：Stata 336.04（df_m=32，含 29 个效应列 + 3 个系数）、westata 49.52（df=2，只对斜率） | 两边各自内部一致（用同一个 V 复算都能逐位重现自己的 chi2），但 p 值不可互换 |
| `xtreg, fe vce(cluster)` | `e(df_r)`：Stata=29(=G−1)，westata=148(=N−K) | 直接决定 t 检验的参考分布，见 §2 |
| `xtreg, re (非平衡)` | `e(theta)` 不存在（Stata 改报 `thta_min/5/50/95/max`）；westata 一个都不报 | 系数仍一致，只是少诊断量 |

`e(theta)` 缺失的实际影响：用户无法判断当前是"正常 RE"还是"theta→0 退化成 pooled"，
也无法做 `theta` 的敏感性分析。

### 11. `theta(0)` 的说明（任务书要求项）

**`theta()` 不是 xtreg 的合法选项**：实测 `xtreg y x1 x2, re theta(0)` →
`r(198): option theta() not allowed`。`theta` 在 `[XT] xtreg` 里只是一个**显示开关**
（"report theta"），不接受数值。因此"测 theta(0)"只能用数据构造：σ_u=0 的面板。

实测在该面板上 Stata `e(theta)=0`、`e(rho)=0`、`e(sigma_u)=0`，
RE 与 pooled OLS 的系数、SE **逐位相同**（2.2e-16）；westata 同样逐位相同（2.64e-16）。
westata 没有暴露 `theta`，所以从结果里看不出"现在处在 theta=0 的边界"。

### 12. `xtreg, fd` 与 `xtgls, corr(ar1)`

- `xtreg, fd`：StataNow 19.5 的 `xtreg` 只有 re/cre/be/fe/mle/pa，**没有 fd**
  （实测 `r(198): option fd not allowed`）。改用它真正的等价命令 `regress D. noconstant`
  核对：系数 4.2e-16、SE 1.7e-16、N=150、df_r=148 全对。westata 的 `FirstDifferenceOLS`
  在平衡面板上是正确的。
  （注：`xtivreg, fd` 存在，但要求有内生变量，实测 `xtivreg y x2 (x1 = z), fd`
  给出 `D.x1=.5034789`，与 `xtreg, fd` 不是同一个量，故不用它代替。）
- `xtgls, corr(ar1)`：Stata 成功且给出完全不同的估计（x1 从 .887996 变 .839671）；
  westata 抛 `UnsupportedError` 明确拒绝。**处理方式可接受**（没有把
  `corr(independent)` 的数套上 ar1 标签），但功能缺口真实：TSCS 数据上 ar1/psar1 是常用项。

### 13. `xtlogit, pa` 之外：`xtpoisson/xtnbreg` 的 `_cons`

Stata 的 `xtpoisson, fe` 不报 `_cons`，`xtnbreg, fe` 报 `_cons=2.628`（SE 1.101）。
westata 两者都不报。前者正确，后者也正确（平坦方向不是估计）。已在比对前剔除。

## 与旧结论的出入

旧证据（`tests/conformance/registry.yml`、`docs/GAP_REGISTER.md`）**只读不引用**，
以下是本次实测对它们的复核结果。

| 旧结论（出处） | 本次实测 | 一致？ |
|---|---|---|
| `xtreg, fe` E3，tol 1e-12，"Parameters 1.8e-16, SE 1.9e-16, df_r=148, r2 exact"（registry） | 1.27e-16 / 1.85e-16，df_r=148、r2 精确 | **一致** |
| `xtreg, re` E3，"Parameters 6.8e-16, SE 1.0e-15, Swamy-Arora matches exactly"（registry） | 6.00e-16 / 3.15e-16，e(V) 全矩阵 2.36e-15 | **一致** |
| `xtreg, be` E3，"Parameters 3.6e-16, SE 1.7e-16"（registry） | 4.01e-15 / 4.07e-16 | **一致**（但 e(N) 语义差异未记录） |
| `xtreg, fd` E3，tol 1e-8，"相比 regress D. noconstant，参数 7.5e-9"（registry） | 4.18e-16 / 1.71e-16；且确认 `xtreg, fd` 已从 StataNow 19 移除 | **一致**（本次数值更精确） |
| 面板线性族"全部 E3"（GAP_REGISTER 第 33 行，含 `xtgls`/`xtpcse`） | `xtgls` hetero 与 correlated(满秩) 确为 A 档；**`xtpcse` 非平衡 casewise/pairwise 为 E 档** | **不一致——旧结论高估** |
| `xtpcse` E3，tol 1e-4，"Beck-Katz PCSE with the N/(N-k)·T/(T-1) factor"（registry） | **平衡面板** A 档（1.1e-15），但 westata 源码注释明确说**不加**该因子（实测无因子才复现 Stata）；**非平衡面板 SE 全错** | **部分不一致**：平衡对、非平衡错；"有 N/(N-k)·T/(T-1) 因子"的说法与源码注释矛盾 |
| `xtgls` E3，tol 1e-4（registry） | hetero 1.5e-15、correlated(满秩) 2.8e-15，比 tol 好 10 个数量级 | **一致**（旧 tol 过松） |
| `xtlogit, pa` E3，tol 1e-5，"gaps 9.1e-6/9.7e-6/7.4e-8"（registry） | 本次测得 **3.19e-4**（x1 .328897 vs .329002）；westata 自己的文档承认"系数不匹配 Stata" | **不一致——旧结论与源码文档互相矛盾，且实测支持源码文档** |
| `xtnbreg, fe` E3，tol 1e-4，"Stata x1 .5041909; westata gap 2.1e-5"（registry） | 系数差 **1.36e-3**、**SE 差 6.79e-2**（E 档）；westata 的 xtnbreg,fe 与自己的 xtpoisson,fe 逐位相同 | **不一致——旧结论只看系数且低估** |
| `xtabond` E2，tol 2e-2，"系数差 6.0e-3 / 2.6e-3，SE 未比对"（registry） | 系数 2.0e-3（动态面板）但 **SE 差 0.91**、`e(sig2)` 差 1.99 倍、`lags(2)` 参数名错位 | **不一致——旧结论没比 SE，"E2"低估了问题** |
| `xtabond` "~3 位系数、~3 位 z/p、结论一致"（GAP_REGISTER 8.10） | 系数确实约 3 位；但 **SE 不可信**，z/p 跟着不可比 | **部分不一致** |
| 面板非线性族"全部 E3"（GAP_REGISTER 第 34 行，含 `xtnbreg, fe`） | `xtnbreg, fe` 为 D/E 档 | **不一致** |
| 面板单位根"全部 E3"（第 36 行） | ips/llc 在有/无单位根两组数据上都是 A 档（1e-15 ~ 1e-14） | **一致** |
| `xtreg, fe vce(robust)` / `vce(cluster)` | registry 里**没有**独立条目（只在 `xtreg, fe` 下隐式覆盖） | **旧证据存在系统性盲区**：vce 变体从未被单测过 |
| "StataNow 19 移除了 `xtreg, fd`"（GAP_REGISTER 第 546 行） | 实测确认 `r(198): option fd not allowed` | **一致** |

**最有价值的三条出入**：
1. **`xtpcse` 非平衡**：旧结论 E3，实测 SE 相对差 0.281（casewise）/ 0.419（pairwise）（E 档），
   且 `pairwise` 与 `casewise` 返回同一结果——这个分支从未被真正测过。
2. **`xtreg, fe vce(cluster/robust)`**：旧 registry 没有独立条目，但这是实际使用频率最高的
   组合之一，实测 SE 差 9.36%（cluster）与 18.36 倍（robust 常数项）。
3. **`xtabond` 的 `e(sig2)` 差 2 倍 + `lags(2)` 参数名错位**：旧结论"E2、系数约 3 位、
   结论一致"只看了系数，把最危险的两件事都漏掉了。

## 未覆盖

| 项目 | 原因 |
|---|---|
| `xtreg, cre`（相关随机效应） | StataNow 19.5 新增模型，westata 无对应类（`RandomEffects` 只做 re）。任务清单未列。 |
| `xtreg, mle`、`xtreg, pa`（xtreg 内的 PA） | westata 的 xtreg 族只有 fe/re/be/fd 四个适配器；`xtlogit, pa` 已单独测过。 |
| `xtregar`（带 AR(1) 扰动的面板） | 两边都没有对应实现。 |
| `xtgls` 的 `igls` | westata 明确不支持（`igls` 会改变估计量而非显示选项）；Stata 的 igls 在 30×6 上不收敛（实测迭代 34 次未收敛），没有稳定参照值。 |
| `xtgls, corr(psar1)` | 与 `corr(ar1)` 同类，westata 一并拒绝。 |
| `xtlogit, re` 的 `intmethod(ghermite)` 与 `intpoints` 网格 | 只测了默认 `mvaghermite` + 12 点。`xttobit` 做了 12/20/50 点敏感性（见 §7）。 |
| `xtpoisson, fe/nbreg` 的 `vce(robust)`、`vce(cluster)` | westata 对这两个只支持 `nonrobust`（会抛 `UnsupportedError`），无对比对象。 |
| `xtabond` 的 `vce(robust)`、Sargan/Hansen/AR(1)/AR(2) | westata 只有一种 VCE，且明确不报过度识别检验。 |
| `xtunitroot` 的 `trend`、`noconstant`、`demean`、`lags()` 指定、`breitung`/`hadri`/`fisher` | westata 只实现 ips/llc 且拒绝 trend/noconstant/demean；其余检验 Stata 有、westata 无。 |
| `xtivreg` 的 `be`/`fd`/`ec2sls` 与 `vce(robust)` | westata 明确只做 fe/re，且只支持 `nonrobust`。 |
| `xtnbreg, re` | westata 的 `XtNbreg` 只有 `fe`。 |
| 权重（`aweight`/`fweight`）与 `if`/`in` 子样本 | 面板族本轮未测；留待 linear 领域的权重覆盖。 |
| 时间序列不连续（有洞）的 `xtreg, fd` | 只用强平衡面板测过 fd；westata 的 `_period_grid` 处理有洞的 pair 的逻辑未验证。 |

## 一句话结论

**xt 线性族（fe/re/be/fd/ivreg/gls/unitroot）在系数与协方差上几乎是逐位精确的（1e-15），
问题全部集中在"协方差估计量的选择"与"面板非线性的二阶量"上：**
`vce(robust)/vce(cluster)` 的 Stata 语义映射错（E 档）、`xtpcse` 非平衡的 Σ/meat 截断错
（E 档）、`xtabond` 的 σ² 差 2 倍且 `lags≥2` 参数名错位（E 档）、`xtnbreg, fe` 用条件 Poisson
冒充 HHG（E 档）。`xtlogit fe/re`、`xtprobit re`、`xttobit`、`xtpoisson` 落在 B/C 档，可用但
SE 不应报超过 4 位有效数字。
