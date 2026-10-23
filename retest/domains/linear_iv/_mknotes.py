"""把 NOTES.md 的头部、覆盖矩阵（读 _matrix.md）、逐项说明拼成最终报告。

只拼接已存在的文件与常量文本，不产生任何新数字。
用法：.venv\\Scripts\\python.exe retest/domains/linear_iv/_mknotes.py
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
MATRIX = (HERE / "_matrix.md").read_text(encoding="utf-8").strip()

HEADER = """# linear_iv 复测结论

复测时间：2026-10-24（Asia/Shanghai）。证据来源：本进程内启动的真实 **StataNow 19.5 (MP)** 会话
（`pystata`，`config.init("mp")`）＋ 本机 `westata 0.0.1`（editable，`src/westata`）实跑。
**未引用** `tests/conformance/registry.yml`、`tests/conformance/stata_fixtures/`、
`conformance*.csv`、`docs/GAP_REGISTER.md` 的任何结论；本文所有数字均来自本次实测。

重跑：`cd D:\\myCodeProject\\statapy && .venv\\Scripts\\python.exe retest/domains/linear_iv/run.py`
（写 `results.jsonl` / `summary.json`）。`_mktable.py` 只把 JSONL 渲染成下表，不产生数字。

## 计数

**65 个记录项**：pass 46 / fail 15 / skip 4（无 error）。

| 类别 | 数量 | 说明 |
|---|---|---|
| A 档（系数与 SE 均 ≤1e-9） | 40 | 机器精度，可声称"完全一致" |
| E 档（>1e-2） | 6 | 4 项是 IV 的 VCE 语义（34/40/41/46 号）、1 项 areg 口径差（21 号，非缺陷）、1 项 GMM+cluster 点估计（47 号） |
| A*（系数/SE 是 A，专比的 e() 标量不达） | 4 | e(F) × 3、e(df_r) × 1，均是"标量语义"问题 |
| e() 覆盖度 fail / pass | 5 / 1 | Stata 存了但 westata 的 `e` 词典里没有 |
| 归因用证据条目 | 5 | 手算闭合验证，不含容差判定（7c-ter + 17 ×4） |
| skip | 4 | 边界/不支持，已在"未覆盖"说明 |

最大系数相对差 **3.6e-2**（47 号，GMM+cluster，属估计量差异，见下）；最大 SE 相对差 **1.2e-1**（同一条）。
除该条外，最大系数相对差 1.2e-13，最大 SE 相对差 4.9e-2（21 号 areg，见下，**非 westata 缺陷**）。

数据：
- `auto.dta`（Stata `sysuse auto` 的同一份，`pd.read_stata("tests/data/auto.dta", convert_categoricals=False)`）。
- **自造数据**（固定 seed `np.random.default_rng(20261024)`）：140 观测 × 14 组（`gid`）× 10 期（`t`），
  含组效应与期效应；`w` 为 1–4 的整数权重；`e1` 由 `z1 z2` 驱动（内生），`yiv` 是其结果变量。
  两侧观测数逐条核对（脚本内 `assert st.nobs() == len(panel)`）；`i.rep78` 有 5 个缺失，
  Stata `N=69` 与 westata `nobs=69` 一致，即 listwise deletion 相同。

## 覆盖矩阵

档位按 BRIEF 2.3：A ≤1e-9、B ≤1e-6、C ≤1e-3、D ≤1e-2、E >1e-2，取系数与 SE 中较差者。
`A*` = 系数/SE 都是 A，但**本条专比的某个 e() 标量**不达 A（单列并归因）。
`证据` = 手算闭合验证，不含容差判定。

| # | Stata 命令 | westata 调用 | 数据 | 判定 | 系数相对差 | 标准误相对差 |
|---|---|---|---|---|---|---|
"""

BODY = """
### 环境与已装软件（61 号）

`which reghdfe` → `C:\\Users\\zjl\\ado\\plus\\reghdfe.ado`，`*! version 6.14.1 08Jul2026`。
**本机已装 reghdfe**，所以第 8 组（26–29 号）是与真实 `reghdfe` 对拍，**不需要**用 areg 双向替代。
同时确认 `areg` 2.6.0、`ivregress` 1.6.0、`eivreg` 3.0.6、`ftools` 2.50.0 可用；`ivreg2` 未装（本领域未用）。

## 逐项说明

下面只解释 A 档以外、以及需要额外说明的条目。所有归因都用可复算的方式验证：
「把 westata 自己的 `e(V)` 代进同一个公式」或「手算同一估计量」，不靠推理。

### 21 —— areg `vce(cluster gid)` 与 FEOLS 差 4.86e-2（E），但**不是 westata 的错**

- 现象：系数一致到 1.1e-15；SE 系统性大 5.11%。
- 原始数字：`areg y x1 x2, absorb(gid) vce(cluster gid)` →
  SE `[0.05512530662889631, 0.08085366893482775]`；
  `reghdfe` 同规格 → `[0.052444696588479306, 0.07692195099971824]`；
  FEOLS 与 reghdfe 一致到 **7.94e-16**（22 号）。
- 归因（三个独立来源全部闭合）：
  - 聚类 CR1 因子是 `G/(G-1) * (N-1)/(N-K)`，两侧分歧只在 `K`。
  - `areg` 把吸收的 13 个 `gid` 虚拟变量 + 常数计入 `K` → **K=16**，因子 `1.2071960297766748`。
  - `reghdfe` 按默认 `dofadjustments(nested)` 把「嵌套于聚类变量的 FE」记为冗余 → **K=3**
    （2 斜率 + 常数；`e(df_a)=0`），因子 `1.0926445816956765`。
  - `sqrt(1.2071960297766748 / 1.0926445816956765) = 1.051113081298782`；
    实测 SE 比值 `1.0511130812987848`。
  - 23 号（闭式复算）：手算 within 变换后的 CR0 → SE `[0.05017208148834386, 0.07358864946981207]`，
    乘 `K=16` 因子后与 areg 相对差 **1.13e-15**，乘 `K=3` 因子后与 reghdfe 相对差 **9.26e-16**。
    两个候选用同一个 CR0 基准，唯一差别是 `K`。
- 反证（24 号）：把聚类变量换成**不嵌套**的 `t`（10 组），`areg` 与 `reghdfe` 的 SE 相同到
  `[0.06534372453008551, 0.1119248857201664]`（FEOLS 相对差 1.27e-15）。
  → 5.11% 只出现在「聚类变量嵌套该 FE」时，是 Stata 两个命令对小样本因子的口径差。
- **结论：这一条不构成 westata 的缺陷**，FEOLS 正确跟随了 reghdfe 口径。
  专门记下来是因为：拿 `areg` 的 SE 当基准去校验 FEOLS，会误判成 5% 的 bug。

### 34 / 40 / 46 —— IV 的 `vce(robust)`：westata 的 `HC1` 标签与 Stata 默认不一致（E，1.09e-2）

- 现象：`ivregress 2sls/liml/gmm ..., vce(robust)` 与 westata `cov_type='HC1'` 的 SE 差 **1.089e-2**，
  系数完全一致（≤2.6e-15）。
- 归因：Stata 的 `ivregress, vce(robust)` **默认不加** `N/(N-K)`；加 `small` 才加。实测
  `vce(robust) small` / `vce(robust)` 的 SE 比值 `= 1.0108896 = sqrt(140/137) = sqrt(N/(N-K))`。
  - westata `cov_type='HC0'` ↔ Stata `vce(robust)`：相对差 **6.1e-15（A）**（33/39/45 号）。
  - westata `cov_type='HC1'` ↔ Stata `vce(robust) small`：相对差 **6.0e-15（A）**（35 号）。
- 文档/代码里的错误陈述：
  - `src/westata/estimators/iv.py:586`：*"HC1: the N/(N-K) factor is always part of HC1 (as for regress)"*；
  - `iv.py:624`：*"HC1 always has N/(N-K)"*；
  - `iv.py:132-137` 的 `_LM_COV_TYPES` 与类文档表把 `HC1` 列成 `vce(robust)` 的对应项。
  但 `[R] ivregress` 的 robust 默认是 **HC0**，不是 `regress` 的 HC1。
- 影响：按文档选 `HC1` 会得到比 Stata 默认大 `sqrt(N/(N-K))` 的 SE（此处 1.0109 倍；N 小或 K 大时放大）。
  显著性判断偏保守、置信区间偏宽，系数不变。
- **判定**：实现本身没错（HC0、HC1 都给对了），错的是把 HC1 声称成 Stata 的默认 robust ——
  这是一处**语义标签错误**，属于"文档说 A、实测是 B"的典型，最值得修。

### 41 —— `ivregress liml`, `vce(cluster gid)` 与 westata `cluster` 差 4.53e-2（E）

- 现象：系数一致 1.5e-15；SE 大 4.53%。
- 归因：Stata 的 `liml`/`2sls` 在 `vce(cluster)` 下**默认是 CR0**（既不加 `G/(G-1)` 也不加
  `(N-1)/(N-K)`），`small` 才变 CR1。westata 的 `IVLIML`/`IVGMM` 走 `_backend_vce` →
  `_backend_fit`（`iv.py:625-630`）：`canonical` 非 `nonrobust`/`HC0` 时一律 `debiased=True`，
  linearmodels 的 `ClusteredCovariance` 因此乘 `n/(n-k) * (G/(G-1)) * ((n-1)/n)`，即 CR1。
- 闭合：预期 `westata/Stata` 比值 `((G/(G-1))*((N-1)/(N-3)))^0.5 = 1.045296408534764`；
  实测 westata/Stata `= 1.0452964085347631`。
- 反证（42 号）：Stata 加 `small`（即 CR1）后，与 westata `cluster` 的系数/SE 一致到 **8.7e-16**。
  → 偏差 100% 来自「Stata 默认不加小样本因子 vs westata 恒加」。
- 为什么 2SLS（36 号）没这个问题：`IV2SLS._uses_shared_vce=True`，协方差走 `compute_cov`，
  其默认 `SSC(adj=False)`（`iv.py:444`）→ CR0，与 Stata 默认一致。
  **同一个 `cov_type='cluster'`，2SLS 与 LIML/GMM 的语义不同** —— 这本身是易踩的坑。

### 47 —— `ivregress gmm`, `vce(cluster gid)` 系数差 3.58e-2（E），属**估计量选择**差异

- 现象：Stata `{'e1': -0.48536211184222755, 'x1': 2.0935721060105577, '_cons': 1.0126539516130684}`
  vs westata `{'Intercept': 1.0086674504234476, 'x1': 2.095047823235924, 'e1': -0.502753684313626}`。
- 归因：Stata 的 `ivregress gmm` 在 `vce(cluster)` 下**把 GMM 权重矩阵 W 也换成聚类形式**，
  点估计随之改变（不只是换 VCE）。westata 的 `IVGMM` 无论 `cov_type` 都用 linearmodels 默认的
  heteroskedastic W。
- 手算复现（两步 GMM，W 与 S 用不同残差集）：

  | W | S | 系数 vs Stata | SE vs Stata |
  |---|---|---|---|
  | hetero | hetero | 6.6e-3（= westata 现在的做法） | 5.1e-1 |
  | hetero | cluster | 6.6e-3 | 8.2e-2 |
  | cluster | cluster | **1.5e-15** | **1.9e-15** |

- 结论：westata 缺的是「cluster 情形下 W 也聚类」这条路径；**不是数值实现错误**，
  但聚类 GMM 的点估计与 Stata 差 3.6e-2，必须显式声明或补齐。
- 附带：47 号里 `e(r2)`/`e(rmse)` 与 Stata 差 1.5e-4/7.4e-4，是点估计不同带来的派生量差异，
  不是独立问题。

### 43 / 44 —— GMM 的 `e(V)`（BRIEF 点名的重点）：**A 档，全矩阵**

- 43 号：`ivregress gmm yiv x1 (e1 = z1 z2)` 默认（两步 GMM）与 `ws.IVGMM(...).fit()`：
  系数 2.2e-16、SE 3.7e-16。
- 44 号：不只看对角，把 `e(V)` **全部 9 个元素**逐元素比：最大相对差 **4.44e-15（A）**
  （对角 5.38e-16；对齐后矩阵元素最大绝对差 2.1e-18）。
- 结论：westata 的 `_stata_two_step_covariance()`（W 用 2SLS 残差、S 用 GMM 残差）确实等价于
  Stata 两步 GMM 的 VCE。**旧 GAP_REGISTER 的「GMM 稳健标准误差 2.9e-2」不成立**：
  按 `vce(robust)` ↔ `HC0` 的正确映射，差 8.15e-16；2.9e-2 来自把 HC1 当成 Stata 默认 robust。

### 5 / 7 / 9 / 14 / 62–65 —— e(F) 与 e(df_r) 的**标量语义**

这几条系数/SE 都是 A 档，但专比的 e() 标量不达 A，单独归因。

**e(F)**：Stata 的 `e(F)` 在任何 VCE 下都等于「用该 VCE 的 `e(V)` 算的 `b_s' V_ss^-1 b_s / k`」。
62–65 号（同一份 140 观测数据 `y x1 x2`）逐一验证：

| Stata 命令 | Stata e(F) | 把 **westata 自己的 V** 代入同一二次型 | 相对差 | westata 自己的 e(F) | 与 Stata 相对差 |
|---|---|---|---|---|---|
| `regress y x1 x2` | 296.9650238963045 | 296.9650238963045 | 0 | 296.9650238963045 | 0 |
| `regress ..., vce(robust)` | 314.61888821753223 | 314.6188882175323 | 1.8e-16 | 296.9650238963045 | **5.61e-2** |
| `regress ..., vce(cluster gid)` | 474.89190645895013 | 474.89190645895104 | 1.9e-15 | 296.9650238963045 | **3.75e-1** |
| `regress ... [pweight=w]` | 311.9681833830176 | 311.96818338301676 | 2.7e-15 | 321.0520277297091 | **2.91e-2** |

即：两侧 V 相同（把 westata 的 V 代进 Stata 的定义式，差 ≤1.9e-15），
差别纯粹是 westata `OLS` 的 `e(F)` **恒为经典 R² 型** `(r2/df_m)/((1-r2)/df_r)`，不随 cov_type 变。
`auto.dta` 上同样：robust 下 westata 69.74846262 vs Stata 73.80592553（5.50e-2）。
→ 系数、SE、p 值都不受影响，但读 `e('F')` 做整体显著性判断的下游代码会拿到偏小的 F（cluster 下小 37%）。

**e(df_r)**：Stata 在 `vce(cluster)` 下存 `e(df_r)=G-1=13`；westata `OLS` 存 `e(df_r)=N-K=137`。
**推断用的 `results.df_resid` 两侧都是 13**：8 号用 Stata 的 `test x1` 验证
（`r(F)=897.60686415`、`r(df_r)=13`、`p=2.209578718351644e-13`），
westata 的 `p=2.2095787183516284e-13`，相对差 **7.1e-15**。
所以这只影响「读 `e('df_r')` 这个数字」的代码，不影响显著性判断。

### 52–57 —— e() 覆盖度：Stata 存了、westata 的 `e` 词典里没有

| 条目 | Stata 有而 westata `e` 没有 | 影响 |
|---|---|---|
| 52 OLS | `rss`, `mss`, `sum_w`, `ll_0` | `rss`/`mss` 可从 `resid`/`fitted` 反算（50 号，相对差 0，机器精度）；`ll_0` 无法从结果对象还原，要重跑空模型 |
| 53 加权 OLS | 同上 | **`sum_w` 最要紧**：`fweight` 下 `e(N)=sum(w)=369` 而 `res.nobs=140`，下游若用 `nobs` 当 `N` 会算错自由度（westata 的 `e(N)`/`e(df_r)` 本身是对的：369/366，见 10 号） |
| 54 cluster OLS | 上面 + `N_clust` | 聚类数只能从 `results.df_resid + 1` 反推（=G-1+1），没有显式 `e(N_clust)` |
| 55 2SLS | `rss`, `mss`, `kappa`, `k_endog`, `iterations`, `tss` | `kappa`（LIML 的 k-class 参数）是有信息量的诊断量（Stata 此处 `kappa=1.0013841716`） |
| 56 GMM | 上面 + `J` | Stata 的 `e(J)` 是 Hansen J 过度识别检验统计量（此处 `0.1817596796`）；westata 不存，无法直接做 J 检验 |
| 57 eivreg | （无缺失） | westata 反而多存 `reliab`/`vce`/`p` |

westata 侧**多出**（不算缺陷，但与 Stata 不一致）：OLS 有 `aic`/`bic`/`sigma`；
IV 有 `F`/`F_first`/`ll`（Stata 的 `ivregress` 根本不存 `ll`/`F`，它存 `chi2`/`J`）。
注意 IV 的 `e(df_r)`：Stata 存缺失（本机取到 `8.99e+307` 哨兵），westata 存 `N`——
这正是 `iv.py` 文档声明的行为，本次实测一致（31/36 号 extra 里记录了哨兵判定为 true）。

### 48 —— eivreg（reliab）

- `eivreg y x1 x2, reliab(x1 .9 x2 .95)` vs `ws.Eivreg(..., reliab={'x1': .9, 'x2': .95})`：
  系数 4.50e-16、SE 6.39e-16；`e(N)=140`、`e(df_m)=2`、`e(df_r)=137`、`e(r2)=0.90051802`、
  `e(rmse)=0.66174985`、`e(F)=318.06823840` 全部 A 档。
- **为什么不用 .8/.7**：Stata `eivreg` 在 reliab 过小、`A = X'X/n - S` 病态时报
  `reliability r() too small` `r(399)`（59 号）。本数据实测：`reliab(x1 .85)` 就报错、`.9` 可用。
  westata 接受 `0 < r <= 1` 的任意值并返回数字 —— 两侧**边界行为没有对齐**，记入"未覆盖"。

### 25 —— areg `absorb(cont)`（数值存储的"连续"变量）

`cont` 以 double 存储但只有 8 个不同取值：`areg y x1 x2, absorb(cont)` 按类别吸收，
`FEOLS('y ~ x1 + x2 | cont')` 也当因子。系数 5.91e-16、SE 3.33e-16、`e(df_r)` 两侧都是 130、
`e(df_a)=7`。**注意**：若 `absorb()` 的变量每个观测唯一取值，`areg` 会饱和
（`e(df_r)=0`、系数全 0、`e(rank)=0`，见 58 号），那是模型不可识别而非数值差异；
westata `FEOLS` 遇到这种情况抛 `ConsistencyError`（不给数字），行为更安全。

### 30 —— FEOLS 双向（`gid + t`）vs `areg absorb(gid) + i.t`

系数 9.43e-16、SE 4.48e-16；`e(N)/e(df_r)/e(r2)/e(rmse)` 全 A 档（`e(df_r)` 两侧都是 115）。
`e(df_m)` 不比较：`areg` 把 9 个期虚拟变量计入（`df_m=11`），FEOLS 只报可报告斜率（`df_m=2`）——
参数化差异，不是错误。

### 10–16 —— 四种权重

全部 A 档（系数 ≤1.06e-15、SE ≤8.86e-16），且 `e(N)`/`e(df_r)` 逐项对上：
`fweight`/`iweight` → `N=369`、`df_r=366`（复制语义正确）；`aweight`/`pweight` → `N=140`、`df_r=137`。
`pweight` 隐含 `vce(robust)` 也对：westata 默认走 HC1，与 Stata `[pw=w]` 的 SE 相对差 8.86e-16。
`weight_type` 未给时 westata 抛 `WeightError`（脚本内单独验证并写进每条 `extra`），不静默猜类型：
> `WeightError: weights were supplied without weight_type; specify one of 'fweight', 'aweight', 'pweight', 'iweight' (Stata never guesses the weight type)`

`fweight` 的复制型 meat（eq.7）与 `aweight` 的精度型 meat（eq.8）都分别用 `vce(robust)` 对拍并 A 档
（15/16 号）→ `vce/robust.py` 的两分支是对的。

### 17 / 18 / 2 / 3 —— 子样本与因子变量

- `i.rep78`（5 个缺失）：Stata `N=69`、westata `nobs=69`（删同样的行），系数 1.18e-13。
  比 `i.foreign` 的 3.70e-14 略大，属设计矩阵条件数的正常浮动，仍在 A 档。
- 子样本：`if mpg>20`（系数 2.98e-14）与 `if weight<4000 & mpg>18`（3.03e-15），
  两侧观测数一致。
- 因子基线：Stata `e(b)` 里的 `0b.foreign`（值 0）被 `stata_probe` 剔除后配对，
  全部 65 条的 `mismatched` 均为空；`_cons` ↔ `Intercept` 自动配对成功。

## 与旧结论的出入

| 旧结论（registry / GAP_REGISTER） | 本次实测 | 判定 |
|---|---|---|
| `regress` E3，系数 5.1e-15、SE 6.8e-16、`e(N..ll)` 全对 | 系数 5.11e-15、SE 6.81e-16；`e(N/df_r/df_m/r2/r2_a/rmse/F/ll)` 逐项 A | **一致**（独立复核通过） |
| `regress, vce(robust)` E3，HC1 SE 4.6e-15 | SE 4.57e-15（A） | **一致**；旧结论未提 **robust 下 `e(F)` 差 5.5e-2** —— 新增发现 |
| `regress, cluster` E3，「CR1 SE 7.1e-15、`df_resid=G-1` 已确认」 | SE 1.10e-15（A）；`results.df_resid=13=G-1`，且 t 检验 p 与 Stata `test x1` 一致 7.1e-15 | **一致**；旧结论未提 **`e('df_r')` 存的是 N-K=137**（与 Stata 的 G-1 不同义）、也未提 cluster 下 `e(F)` 差 37% —— 新增发现 |
| `regress, i.factor` E3，系数 3.7e-14 | 3.70e-14（A） | **一致** |
| `areg` E3，「slopes 4.6e-16、SE 1.8e-16、`e(df_r)` 精确」 | 系数 1.13e-15、SE 1.82e-16、`e(df_r)=124` 精确 | **一致**；但旧结论只测非稳健，**没测 `vce(cluster)`**；那一项 areg 与 reghdfe 差 5.11%（口径问题，非 westata 缺陷） |
| `reghdfe` E3 + 「reghdfe 装不上（SSC 超时），多路 HDFE 仍 E1」 | **本机已装 6.14.1**；单维/双向/robust/cluster 四项全部 A（≤1.1e-15），`e(df_r)/e(df_a)/e(r2_a)/e(rmse)` 也对上 | **旧结论已过期**：多路 HDFE 由 E1 升为实测通过 |
| `ivregress 2sls` E3，「系数 7.8e-15；SE nonrobust 7.5e-15 / robust 1.6e-14 / cluster 1.9e-14」 | nonrobust 2.8e-15/3.2e-15、cluster 6.1e-15，均 A | 数值量级**一致**；但 **robust 那项**：按文档用 `HC1` 对 `vce(robust)` 差 **1.089e-2**；正确映射是 `HC0`（6.1e-15）。旧记录的 1.6e-14 只可能是按 `small`/`HC0` 比的，**与文档标签不符** |
| `ivregress liml` E3 | 恰好识别 2.4e-15、过度识别 7.2e-16，均 A | **一致**；但 `vce(cluster)` 差 4.53e-2（westata 恒加 CR1，Stata 默认 CR0）—— 新增发现 |
| `ivregress gmm` E3，「系数精确、se gap 0」 | 两步 GMM 默认：系数 2.2e-16、SE 3.7e-16、`e(V)` 全矩阵 4.4e-15 | **一致**（独立复核通过） |
| GAP_REGISTER §8：「`ivregress gmm` 稳健标准误 **2.9e-2**，GMM 权重矩阵构造与 Stata 不同」 | 按 `vce(robust)` ↔ `HC0`：SE 差 **8.15e-16**；按文档的 `HC1` 才 1.089e-2 | **旧结论不成立（推翻）**：2.9e-2 是把 HC1 当 Stata 默认 robust 的结果，与 GMM 权重矩阵无关 |
| `eivreg` E3，`reliab(x1 .7)` 系数 4.2e-8 | 用 `.9/.95`（`.7/.8/.85` 在 Stata 侧报 `r(399)`）：系数 4.50e-16、SE 6.39e-16 | **等级一致**；旧数字（4.2e-8）不可直接比（不同 reliab、不同数据），本次改用可复现的 `.9/.95` |
| GAP_REGISTER 未提及 | **GMM + `vce(cluster)` 点估计差 3.58e-2**：Stata 把 W 也聚类，westata 固定 heteroskedastic W | **新增发现（最严重的功能性缺口）** |
| GAP_REGISTER 未提及 | **OLS 的 `e(F)` 在 robust/cluster/pweight 下语义不同**（westata 恒为经典 R² 型 F，cluster 下差 37%） | **新增发现** |
| GAP_REGISTER 未提及 | **`e(df_r)` 在 cluster 下存 N-K 而非 G-1**；`e(rss)/e(mss)/e(sum_w)/e(N_clust)/e(kappa)/e(J)` 未实现 | **新增发现（e() 覆盖度缺口）** |
| GAP_REGISTER 未提及 | **IV 的 `HC1` 标签与 Stata 默认 robust 不一致**（HC1 实际对应 `vce(robust) small`） | **新增发现（语义标签，非数值错误）** |
| GAP_REGISTER 未提及 | **arEG 与 reghdfe 在嵌套聚类下差 5.11%**（Stata 内部口径差，westata 跟随 reghdfe） | **新增发现（用于防误判，不是缺陷）** |

## 未覆盖

1. **IV 加权**（`ivregress ... [pw=w]` 等）：westata 明确抛 `UnsupportedError`（60 号记录了原文），
   本次不做数值对拍。Stata 侧可用，因此这是**已知能力缺口**而非未测项。
2. **areg/reghdfe 的 `absorb()` 真连续变量**：会饱和（58 号记录 `e(df_r)=0`、`e(rank)=0`），
   两侧都不可估计，没有可比数字；改用「数值存储、8 个取值」的 `cont` 完成对比（25 号）。
3. **多路聚类**（`vce(cluster gid t)` / `reghdfe` 的 multiway）：BRIEF 未要求。westata `FEOLS`
   支持 `cov_kwds={'cluster': [a, b]}`，但本次未与 Stata 对拍（`reghdfe` 多路自由度按 `min(G)`
   取，值得单独一轮）。
4. **`reghdfe` 的 HAC/Driscoll-Kraay**：westata 明确不支持（`hdfe.py` 抛错），未测。
5. **`eivreg` 的边界 reliab**：Stata 在 `r ≤ 0.85`（本数据）报 `r(399)`，westata 不报错。
   两侧在"矩阵不可信"处的行为没对齐，未做数值对拍（59 号）。
6. **`ivregress liml` 的 `e(ll)`**：Stata 的 liml `e()` 里**没有 `ll`**（本机实测只有 2sls/`mss`/
   `rss` 等），westata 存 Gaussian 似然（`iv.py` 已文档化 LIML 集中化似然不复现）。
   定义差异仍是未验证项。
7. **`e(F_first)`（第一阶段 F）**：westata 存 `e(F_first)`（实测最小第一阶段 F = 32.576），
   Stata 的对应量要 `estat firststage` 才有，本次未对拍。
8. **`predict` 类后估计**（`predict xb/stdp/stdf`）与 `margins`：属 postest 领域，本次不做。
9. **`reghdfe` 的 `dofadjustments()` 其他取值**（`pairwise`/`continuous`/`none`）：westata
   固定 `k_fixef="nonnested"`，与 reghdfe 默认对齐；非默认口径未测。
10. **`eivreg` 的 `vce(robust)`**：Stata `eivreg` 本身没有 `vce()` 选项（只有 `reliab()`），
    故未测；westata 的 `Eivreg.fit(cov_type=...)` 存在但无可比对象。

## 附：本次复测暴露的、建议上游修复的四点（按严重度）

1. **GMM + `vce(cluster)` 没把 W 也聚类** → 点估计与 Stata 差 3.6e-2（47 号）。
   要么补齐，要么在 `IVGMM` 文档与代码里显式拒绝 `cov_type='cluster'`。
2. **IV 的 `HC1` 文档标签错**（`iv.py:586/624`）→ 按文档选 HC1 会偏大 `sqrt(N/(N-K))`（34/40/46 号）。
   建议：把 `vce(robust)` 的默认映射改成 `HC0`，并把 `HC1` 标注为 `vce(robust) small`。
3. **`OLS.e['df_r']` 在 cluster 下与 Stata 不同义**（存 N-K，Stata 存 G-1），
   而 `OLS.e['F']` 不随 cov_type 变（5/9/14 号）。`results.df_resid` 与 p 值是对的，
   所以这是"存错的 e() 标量"，最容易误导下游。
4. **e() 覆盖度缺口**：`rss`/`mss`/`sum_w`/`N_clust`/`kappa`/`J`（52–56 号）。
   其中 `sum_w` 与 `N_clust` 成本最低、收益最直接。
"""


def main() -> int:
    out = HERE / "NOTES.md"
    out.write_text(HEADER + MATRIX + "\n" + BODY, encoding="utf-8")
    print(f"-> {out} ({len(out.read_text(encoding='utf-8').splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
