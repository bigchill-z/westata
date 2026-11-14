# westata 公共 API 规范（重构依据）

> 版本：对应 `westata` 0.3.11 源码树。
> 状态：**规范已定稿，实现尚未达标**。本文件先定义「应该是什么样」，再列出
> 现在偏离到什么程度（§7 基线）与修复顺序（§8 路线）。
> 机械校验：`python scripts/check_api_conventions.py`（规则号 R1–R6 与本文件一一对应）。

---

## 0. 这份规范解决什么问题

`westata` 现有 167 个顶层导出、65 个可估类、149 个公共函数。同一个概念在不同
模块里叫不同的名字、放在不同的位置、用不同的单位，用户必须逐个记忆。已实测的
具体症状（全部在本仓库可复现，不是印象）：

| 症状 | 实测证据 |
|---|---|
| 同一动作两种写法，一种静默接受错误类型 | `ws.OLS("price ~ weight", df)` 抛 `ValueError: could not convert string to float`；`ws.FEOLS("price ~ weight", df)` **接受**并返回一个 `FEOLS` 模型对象（`__init__` 把 `formula` 当第一个参数），两者的第一个位置参数含义不同，见 §2.2 |
| 第 1 个位置参数不是 `cov_type`，`fit("HC1")` 被当成布尔值吃掉 | `PanelOLS.fit("HC1")` 返回 `cov_type='nonrobust'`，标准误与非稳健完全相同（`panel.py:535`） |
| 默认值各写各的 | `fit(maxiter=)` 出现 6 种取值（100/200/500/1000/2000/缺省），`fit(tol=)` 13 处显式 `1e-10`，51 处无此参数 |
| 参数能收下但永远不生效 | `XtUnitroot.fit(zzz=[1,2,3])` 正常返回 `PanelUnitRootResult`（`panel_iv.py:2241`）；`im_pesaran_shin(level=...)` 收下后从不使用 |
| 同名参数两种单位 | `level=0.95`（`margins`/`nlcom`/`lincom_as_stata`）与 `level=95.0`（`estat_esize`/`estat_weakrobust`）并存 |
| 同一 DataFrame 参数三种叫法 | `data`（50 个 `from_formula`）、`frame`（`prais`/`synth2`/`im_pesaran_shin`/`levin_lin_chu`）、`table`（`to_latex`/`to_markdown`） |
| 子包导出名在顶层不存在 | `from westata.results import *` 直接 `AttributeError: render_estimates_table`（`results/__init__.py:14-15`） |
| 结果对象能力不齐 | 18 个 `*Result` 没有 `to_dict()`/`to_frame()`，却 18 个都有 `summary()` |
| 同一个辅助函数复制到 11 个模块 | `_is_constant` 在 11 处各写一遍（§6.1），改一处口径不会同步 |

这些都是**接口层**问题，与数值是否正确无关。数值对拍（E1/E2/E3、registry.yml）
已经覆盖得很好；**接口一致性从来没有被机械校验过**，所以它自由生长了 38 个版本。

---

## 1. 总原则

### R1.1 规范只约束公共接口

「公共」= 出现在以下任一位置：

- `westata.__all__`（167 个名字）
- `westata.estimators.__all__` / `westata.postestimation.__all__` /
  `westata.results.__all__` / `westata.report.__all__` / `westata.data.__all__`
- 上述公共类的公共方法、公共属性

以下**不受约束**，允许保留数学记号与 Stata 缩写：

- 以下划线开头的模块级函数/类（`_is_constant`、`_absorbed_columns`…）
- `_` 前缀的内部实现类（`_DiscreteBase`、`_ZeroInflatedBase`…）
- 线性代数层 `westata.vce.*`：`X`、`Xw`、`resid`、`resid_w` 是有意为之，
  因为这一层对外的「语言」就是矩阵代数。校验器已将它们登记为例外。
- 明确标注为 statsmodels 兼容层的参数：`R`、`q`、`use_t`、`alpha`（§4.3）

**理由**：`vce` 层被 `linear`/`iv`/`panel` 三族共用，签名里的 `X` 与数学公式
`(X'X)^{-1} X'ΩX (X'X)^{-1}` 一一对应，改成 `design_matrix` 反而更难核对。

### R1.2 有 Stata 对应命令时，Python 侧的名字要能反查回 Stata

命名优先级：**Stata 命令名/选项名 → 该命令的统计术语 → 通用英文词**。

- 命令函数用 Stata 拼写的 snake_case：`estat_vce`、`estat_firststage`、
  `hausman`、`lincom`、`margins`、`predict`。已达标，保持。
- 选项名同形：`corc`、`sigmamore`、`sigmaless`、`forcenonrobust`、
  `nomiss0`、`atmeans`、`nolog`。这是**唯一**允许保留 Stata 否定式的地方（见 R3.1）。
- 例外：`estat_esize`/`estat_weakrobust` 这类既有函数名保持不动（改名收益低于成本），
  但其**参数**必须符合本规范。

### R1.3 一个概念只有一个名字

同一概念全树唯一拼写，不允许同义并存。词汇表见 §5。

### R1.4 结果对象：数值要能取出，不要只能打印

`summary()` 返回 `str`，是给人看的。凡是要给程序用的结果，必须另有
`to_dict()` / `to_frame()`。实测 18 个公共结果类中**没有一个**提供
`to_dict`/`to_frame`；其中 12 个连 `summary()` 都没有，只有 `__str__`
（`LlcResult`、`IpsResult`、`RdRobustResult`、`TefectsRaResult`、`NnmatchResult`、
`NeweyWestResult`、`UnitRootResult`、`XtTest0Result`、`VarWleResult`、
`DurbinWatsonResult`、`PortmanteauResult`、`SummarizeTable`）。调用方拿到数
只能正则解析表格或翻 `__dict__`——这正是本仓库历史脚本干过的事。

### R1.5 不能将错就错

发现现有实现在语义上错了（不只是命名不好看），按 §7 记录并修正，不允许
「为了兼容就这样」。反向亦同：**不允许为了让校验器通过而改语义**。
例：`XtUnitroot.fit(**kwargs)` 应改为拒收未知参数并抛 `TypeError`，
而不是把 `**_ignored` 改名成 `**kwargs` 继续忽略。

---

## 2. 类的形状

### 2.1 统一的构造链

每个可估命令（Stata 的一条 estimation command）提供且仅提供两条入口：

```python
# 主入口：formula 接口（首选）
Model.from_formula(formula, data, **model_options).fit(**fit_options)

# 次入口：矩阵接口（与 statsmodels 对齐，供程序化构建设计矩阵）
Model(endog, exog, *, param_names=..., **model_options).fit(**fit_options)
```

`from_formula` 是**唯一推荐**入口，写进 README/GUIDE；矩阵接口是兼容层，
文档里标为次入口。测试 `test_subset_is_uniform.py` 已经出现过一次事故：
`IV2SLS.from_formula` 收 `**kwargs`，`inspect.signature` 看不到 `subset`，
于是「功能存在」被误报成「功能缺失」——**签名必须诚实**（R2.6）。

### R2.1 `fit()` 的第一个位置参数必须是 `cov_type`

```python
def fit(self, cov_type=None, cov_kwds=None, ssc=None, level=None, *, ...):
```

理由不是风格，是**已发生的静默错算**：

```python
# 现状（panel.py:535）
PanelOLS.fit(self, entity_effects=True, time_effects=False, cov_type=None, ...)
m.fit("HC1")   # 返回 cov_type='nonrobust'，与 m.fit() 完全一致
```

`"HC1"` 被当成真值赋给 `entity_effects`，`cov_type` 得到 `None`；调用成功、
数字错误。`RandomEffects`/`RandomEffectsMLE`/`BetweenOLS`/`FirstDifferenceOLS`
同病，`Heckman`/`Etregress` 的第一个位置参数是 `method`，同类风险。

**规则**：`fit()` 的位置参数只能是 `cov_type`、`cov_kwds`、`ssc`、`level`；
其余一律 keyword-only。这样 `.fit("HC1")`、`.fit(None, {...})` 在 65 个类上同义。

### R2.2 模型选项归构造函数，估计选项归 `fit()`

| 归属 | 参数 | 依据 |
|---|---|---|
| `from_formula`/`__init__` | `formula`, `data`, `subset`, `weights`, `weight_type`, `freq_weights`, `entity`, `time`, 以及**模型定义本身**：`lower`/`upper`（tobit 族）、`method`（fe/re/ml/twostep）、`quantile`、`quantiles`、`order`、`lags`、`absorb`、`distribution`、`reliab`、`estimand` | 改变模型是什么，改了就换了模型 |
| `fit()` | `cov_type`, `cov_kwds`, `ssc`, `level`, `maxiter`, `tol`, `strict`, `reps`, `seed`, `start_params`, `intpoints`, `ml`, `iterate` | 只改变「怎么估/怎么报」，模型不变 |

**为什么这条重要**：`Tobit.from_formula(..., lower=0).fit()` 是对的，
`Tobit(...).fit(lower=0)` 不存在——用户必须能猜到。现状里 `Heckman`/`Etregress`
的 `method` 在 `fit()`，而 `XtLogit`/`XtIvreg`/`XtGls`/`XtUnitroot` 的 `method`
在 `from_formula()`，同一概念两个位置（详见 §7 的 R2.2 条目）。

**更严重的一处：两个类对「第一个位置参数」的含义根本不同。**

```python
ws.OLS("price ~ weight", df)    # ValueError: could not convert string to float
ws.FEOLS("price ~ weight", df)  # 接受，返回 FEOLS 模型对象
```

`OLS.__init__(endog, exog, ...)` 走矩阵接口，`FEOLS.__init__(formula, data, ...)`
走 formula 接口。同样写成 `Model("公式", df)`，一个报错一个静默构造，
而且 `FEOLS` 这条路径并未写进任何文档。规范：**`__init__` 恒为矩阵接口**，
formula 接口只有 `from_formula` 一条路（§2.1）。

实测 65 个可估类的 `__init__` 首个位置参数，**7 个是 formula 式**
（`Areg`、`BetweenOLS`、`FEOLS`、`FirstDifferenceOLS`、`PanelOLS`、
`RandomEffects`、`RandomEffectsMLE`，全部来自 `panel.py`/`hdfe.py`），
55 个是矩阵式，另 3 个既不是（`Intreg(lower, upper)`、`Permute(*args)`、
`XtTobit(*args)`——后两个本身就是签名缺陷）。这不是小瑕疵：`PanelOLS.from_formula(...)` 与
`PanelOLS("公式", df)` 两条路并存，而前者走 `build_design`（含因子变量展开、
`subset`、权重切片），后者未必走同一条路径——同一模型两条构造路径，
一旦发散就是「同一个类算出两个数」。

另一处顺序矛盾在 survey 族：

```python
SvyMean.__init__(self, variables, frame, design, ...)      # variables 在前
SvyRegress.__init__(self, frame, formula, design, ...)     # frame 在前
SvyRegress.from_frame(frame, formula, design)              # frame 在前
```

`from_frame` 的 `(frame, formula)` 与 R3.2 要求的 `(formula, data)` 顺序相反。
规范：`from_frame` 改为 `(formula, data, design)`，与 `from_formula` 同序。

**再拟合同一模型换稳健方差**必须是纯 `fit()` 操作：

```python
model = ws.OLS.from_formula("y ~ x1 + x2", data)
robust = model.fit(cov_type="HC1")
clustered = model.fit(cov_type="cluster", cov_kwds={"groups": data["firm"]})
```

### R2.3 DataFrame 参数一律叫 `data`

| 现状 | 计数 |
|---|---|
| `data` | 50（全部 `from_formula`）+ `newey`/`varsoc`/`estat_summarize` |
| `frame` | `prais`、`synth2`、`im_pesaran_shin`、`levin_lin_chu` |
| `table` | `to_latex`、`to_markdown` |

`table` 是正当例外（它收的是表格**结果**不是数据源）。`frame` 一律改为 `data`。
`formula` 恒为第一个位置参数（`Sureg` 用 `equations`，见 R3.2）。

### R2.4 数组与变量名分离

矩阵接口 `__init__(endog, exog, ...)` 只收 `np.ndarray`；列名走 `param_names`。
混收是 bug 温床：`Heckman.__init__` 同时收 `outcome_names`、`selection_names`、
`selected`、`model_data`，而 `XtIvreg.__init__` 收 `exog_names`/`endog_names`/
`instrument_names`/`codes`/`sizes`，`panel_dynamic.XtAbond` 又默认
`param_names=("L1.y",)`、`exog_names=("x1",)`——默认值里写着某个具体数据集
的列名，换数据必错。矩阵接口的默认参数不允许出现具体变量名。

### R2.5 公共参数名 snake_case

`customV` → `custom_v`（`synth.py:731`、`synth2.py:248` 两处），
`endog_regF_test` → `endog_regf_test`（`endog_regf.py:112`）。
大写 `X`/`R` 仅在线性代数层与 statsmodels 兼容层保留（R1.1 例外表）。

### R2.6 公共入口不接受「收下即丢」的 `**kwargs`

三条判定：

1. 有 `**kwargs` 且**转发**给一个有真实签名的被调方 → **合规**
   （`IV2SLS.from_formula` 转给 `_reject_unsupported_kwargs`，未知参数抛
   `TypeError`，实测 `from_formula(bogus=1)` → `TypeError: unexpected keyword
   argument(s): bogus`）。
2. 有 `**kwargs` 且**从不提及** → **不合规**。实测三例：
   `VECM.fit(bogus=1)`、`XtUnitroot.fit(zzz=[1,2,3])`、`MultinomialLogit.fit(bogus=1)`
   全部正常返回。用户写错选项名会得到「成功但没生效」的结果，这是最难排查的一类。
3. **禁止**「先收下、需要时再说」的占位签名。

`synth(..., **unsupported)` / `synth2(..., **unsupported)` 属于第 1 类：
它们把未知项聚合成 `UnsupportedError: synth: unsupported option(s): ['bogus']`，
明确报错，合规——但签名应改成显式列出已知选项 + 一个集中校验点，
让 `inspect.signature` 能自解释（见 §8 阶段三）。

---

## 3. 参数语义

### R3.1 布尔开关优先肯定式；**Stata 选项名本身是 R1.2 的例外**

新增 API **优先**用肯定式（`constant=False` 而非 `noconstant=True`）。但 R1.2 规定
「有 Stata 对应命令时，Python 侧的名字要能反查回 Stata」，这两条在这里冲突，
处理原则是：**能反查回 Stata 的否定式选项名保留原样，其余改为肯定式。**

实测确认这批参数**全部是 Stata 原生选项名**，各有 sthlp/ado 对照：

| 参数 | Stata 对应 | 源码证据 | 决定 |
|---|---|---|---|
| `nomiss0` | `estat bgodfrey, nomiss0` | `bgodfrey.py:36,195,207` 引用 ado 行为 | **保留** |
| `nosort` | `estat vif, nosort` | `vif.py:264-266`「Stata's `, nosort`」 | **保留** |
| `noheader` | `estat summarize, noheader` | `summarize.py:305` 引 `sthlp:93` | **保留** |
| `noconstant` | `newey ..., noconstant` / `xtunitroot ..., noconstant` | `newey.py:10` 逐字引用 syntax 行 | **保留** |
| `nowald` | `contrast ..., nowald` | `contrast.py:436` | **保留** |
| `noabsorb` | `areg` 无此选项；本包自定义（`noabsorbtest` 语义） | `hdfe.py:954` 注释「Stata: `noabsorbtest`」 | 名不符实，改名 |
| `nonrtolerance` | `synth` 内部收敛参数，非选项名 | `synth.py` | 改名 |
| `nofigure` | `synth2, nofigure` 是图形开关 | `synth2.py` | 图形选项，本包不实现图形 → 删除 |
| `noweights` | `estat summarize, noweights` | `summarize.py` | **保留** |
| `normalise_counts` | 非 Stata 选项，拼写问题 | `tebalance.py:207` | 改美式 `normalize_*` |

**真正的缺陷只有一处**：`dfgls` 的 `trend` 与 `notrend` 两个参数控制同一状态
（`dfgls(series, *, trend=True, notrend=False)`，`notrend=True` 会覆盖 `trend`）。
两个开关表达一个状态，`notrend=False` 是「看起来有意义实则无效」的调用。
`notrend` 已删除，`trend=False` 是唯一写法（保留弃用别名一个版本）。

`noabsorb` 是唯一**名不符实**的：Stata 的 `areg` 没有 `noabsorb` 选项，
本包的真实语义是「关闭被吸收变量的 F 检验」（对应 `reghdfe` 的 `noabsorbtest`）。
一个叫 `noabsorb` 的参数会让人以为它在关闭*吸收*，实际它关闭的是*检验*。

### R3.2 `from_formula` 的前两个位置参数固定

`(formula, data)`，此外无第三个位置参数。多方程用 `(equations, data)`
（`Sureg`，允许）；非 formula 型命令（DiD、teffects 族）不得叫 `from_formula`：

```python
# 现状：名字骗人
ws.Didregress.from_formula(outcome, treatment, data, group=..., time=...)
ws.RegressionAdjustment.from_formula(outcome, treatment, data, covariates=...)
```

改为 `from_variables(outcome, treatment, data, *, ...)`，并保留 `from_formula`
作弃用别名一个版本。校验器只对「叫 `from_formula` 却首参不是 formula」报错。

survey 族的 `from_frame` 同样受此约束：`SvyRegress.from_frame(frame, formula,
design)`（`survey.py:1060`）把 `frame` 放在 `formula` 前面，与 `from_formula`
相反。改为 `from_frame(formula, data, design)`；`SvyMean`/`SvyTotal`/
`SvyProportion` 的 `from_frame(variables, frame, design)` 首参不是公式，
按同一原则改名为 `from_variables(variables, data, design)`。

### R3.3 `level` 是分数，不是百分数；`alpha` 只用于 statsmodels 兼容层

| 参数 | 单位 | 使用位置 |
|---|---|---|
| `level` | **分数**，`0 < level < 1`，默认 `0.95` | 全部 |
| `alpha` | 分数，显著性水平 | 仅 `conf_int(alpha=)`（statsmodels 形状） |

现状两处违反：`estat_esize(level=95.0)`（`esize.py:134`，且 `EffectSizeResult`
自存 `level=95.0` 后在 `esize.py:72` 用 `(100 - level)/2` 反算）、
`estat_weakrobust(level=95.0)`（`weakrobust.py:235`）。
一旦有人对 `estat_esize` 传 `0.95`，`(100-0.95)/2 = 49.5%`，静默给出错区间。

`im_pesaran_shin(level=0.05)`（`xtunitroot_ips.py:180`）是第三种语义：
它收 `level` 但函数体**从未引用**，是死参数（实测 `level` 改动不改变任何输出）。
删除，或按 R3.3 赋予 `level=0.95` 的真实含义务。

### R3.4 `subset` 是行掩码，语义全树一致

`subset` 收布尔 Series / 布尔数组，语义 = 先在**整帧**上过滤再建设计矩阵，
与 `df.loc[subset]` 等价。已由 `test_subset_is_uniform.py` 固定住 41 个
`from_formula`。缺口：14 个 DataFrame 型公共函数没有 `subset`
（`synth`、`synth2`、`prais`、`newey`、`summarize`、`estat_summarize`、
`im_pesaran_shin`、`levin_lin_chu`、`varsoc`、`vecrank`…）。补齐前，
文档必须点明哪些命令没有 `subset`。

### R3.5 `weights` + `weight_type` 两参数成对出现，`weight_type` 必填

沿用 `README`「Weight semantics」与 `data/weights.py`：
`weights` 收列名字符串或数组（实测两种都可用），`weight_type ∈
{fweight, aweight, pweight, iweight}`，**不猜**。

`freq_weights`（`Logit`/`Poisson` 等离散模型）是 `weights="w",
weight_type="fweight"` 的另一套拼法（`discrete.py:507`）——同一件事两个名字。
规范：保留 `weights`/`weight_type` 为唯一入口，`freq_weights` 作为弃用别名。

### R3.6 `cov_type` 词表与 `cov_kwds` 键名集中定义

词表唯一来源 `vce.dispatch.SUPPORTED_COV_TYPES`：
`("nonrobust", "HC0", "HC1", "HC2", "HC3", "cluster", "HAC")`，
外加别名（大写、`newey`…）。各估计族**只允许在其上收窄，不允许另立词表**。

现状：`did.py:134`、`hdfe.py:84`、`discrete.py:95`、`panel.py` 各自维护一份词表，
`dispatch.py:52` 又有一份 `_normalize_cov_type`，四份实现，彼此不一致
（`hdfe` 拒绝 `robust`，`discrete` 接受，`panel` 把它解释成 cluster）。

`cov_kwds` 的键名同理：`groups`/`clusters`/`cluster` 三种拼法由
`cluster_labels_from_kwds` 统一接收（`dispatch.py`，已实现且注释说明了原因）。保持。

### R3.7 `cov_type="robust"` 是**唯一**通用拼法——用户不需要知道 HC0/HC1

**这是本节的核心目标，优先于本节其余条款。**

用户的要求（2026-11-13）：

> 我希望以后 westata 使用 cov_type 也像 Stata 一样只需要设置 robust 就行。

Stata 用户在 `regress`、`ivregress`、`logit`、`xtreg`、`reghdfe` 上写的都是
**同一个词** `vce(robust)`。Stata 在内部解析成哪一套，是 Stata 的事，用户不需要
知道。westata 目前把这份内部知识泄漏给了用户：`FEOLS` 拒绝 `robust` 并要求写
`HC1`，`IV2SLS` 的 `robust` 又悄悄等于 `HC0`。**这是 westata 独有的负担，不是
Stata 的语义。**

#### R3.7.0（最高优先级）凡是 Stata 支持 `vce(robust)` 的命令，`cov_type="robust"` 必须可用

这条压倒本节的其它细节：`robust` 可用是第一位的，映射成哪个具体值是第二位的。

`HC0`/`HC1`/`HC2`/`HC3` 降级为**高级覆盖**：给确实需要指定变体的研究场景保留，
但**不再出现在文档主路径、README 示例、GUIDE 教程里**。主路径只教 `robust`。

对 `cov_type` 的验收口径因此变成：

| 用户写法 | 要求 |
|---|---|
| `cov_type="robust"` | 在 Stata 有该选项的每个命令上都可用；结果回显**规范名**（见 R3.7.2） |
| `cov_type="nonrobust"` 或 `None` | 与 Stata 不写 `vce()` 一致 |
| `cov_type="cluster", cov_kwds={"groups": g}` | 与 `vce(cluster g)` 一致 |
| `cov_type="HC0"`/`"HC1"`… | 保留，但属高级用法 |
| 拼错或该命令没有的 VCE | 报错，**且报错信息要告诉用户改用 `robust`** |

#### R3.7.0a 拒绝 `robust` 是缺陷，不是设计——但有真假两种

实测 14 个估计族，其中 **7 个拒绝 `robust`**。必须分成两类：**假拒绝**（只是名字
没接上，改别名即可）与**真缺失**（Stata 有这个选项但 westata 没实现）。

**假拒绝——Stata 的 `robust` 在本族就等于某个已实现的值，纯粹是别名表缺项：**

| 族 | 现状 | Stata 实际 | 修法 |
|---|---|---|---|
| `FEOLS` / `Areg` | 抛错并提示"改用 HC1" | `reghdfe, robust` **就是** HC1 | `_COV_ALIASES` 加 `"ROBUST": "HC1"`，删掉 `_UNSUPPORTED_COV_TYPES["ROBUST"]` 那条 |
| `RobustRegression` | 静默吞掉 | `rreg` 无 `vce()` | 明确拒绝，且提示"rreg 不支持 vce()"，不要静默 |

`FEOLS` 这一条尤其没有道理：`_UNSUPPORTED_COV_TYPES["ROBUST"]` 的文案自己写着
"use cov_type='HC1' (Stata's vce(robust) default)"——**它承认两者是同一个东西，
却拒绝那个正确写法、要求用户写内部名字**。实测 `FEOLS(cov_type="HC1").cov_type`
返回 `'HC1'`，与 Stata 的 `robust` 数值相同，说明实现早就有了，缺的只是一个键。

**真缺失——Stata 有 `vce(robust)`，westata 没实现：**

| 族 | 现状报错 | 说明 |
|---|---|---|
| `Tobit` | `only cov_type='nonrobust' is implemented` | Stata `tobit, vce(robust)` 存在 |
| `Heckman` | `only cov_type='nonrobust' is implemented` | Stata `heckman, vce(robust)` 存在 |
| `XtIvreg` | `cov_type='robust' is not implemented ... Stata offers vce(robust)` | 报错自己承认 Stata 有 |
| `XtGls` | 同上 | 同上 |

这四类**不能靠改别名解决**，需要实现 ML/cluster 三明治。在实现之前，
报错信息必须**明确说"该命令尚未实现 robust，不是不支持"**，并给出可用的替代，
不能像现在这样让用户以为是自己写错了。这是诚实性问题（§0 总原则）。

#### R3.7.6 同一输入必须在所有族回显**同一个规范拼写**

规范值是共享词汇（R3.6），因此 `result.cov_type` 在 `OLS` 与 `Logit` 上必须是同一个
字符串。实测违反过一处：`discrete.py` 的标签映射表用大写键 `"HC1"`，而上游
`_canonical_cov` 返回的已经是小写 `"hc1"`，查表落空后直接把小写值透传——

```python
ws.OLS(..., cov_type="HC1").fit().cov_type     # 'HC1'
ws.Logit(..., cov_type="HC1").fit().cov_type   # 'hc1'   <- 修复前
```

**这个缺陷只有跨族对比才看得见**：单独看 `Logit` 的输出，`'hc1'` 并不刺眼。
校验器 `check_r3_cov_echo_case` 因此把 `HC1` 在 5 个族里逐一比对
（`PanelOLS` 例外，它按 R3.7.3 必须回显 `cluster`）。

#### 实施顺序（回答"以后只用 robust"）

1. **立刻（纯别名，零风险）**：`FEOLS`/`Areg` 接受 `robust`→`HC1`；
   修 `Areg` 报错文案写成 `FEOLS` 的 bug（R3.7.4）。
2. **立刻（标签，零风险）**：R3.7.2 的 9 处回显规范化。
3. **中期（补实现）**：`Tobit`/`Heckman`/`XtIvreg`/`XtGls` 的 `robust`。
   这四个族需要真正实现三明治，属于功能开发，不是重构。
4. **文档**：README/GUIDE 的主路径改成只写 `cov_type="robust"`，
   `HC1`/`HC0` 移到"高级用法"一节并注明"一般用不到"。

**注意 R3.7.2 与 R3.7.0 不冲突**：用户写 `robust`，结果对象**回显规范名**
（`HC1`/`HC0`/`ML_ROBUST`/`cluster`）。用户不需要写 HC1，但需要能看见系统
实际用了哪个——这正是 Stata 打印 `Robust` 表头的等价物。

#### 为什么 `robust` 的映射仍需逐族定义

Stata 的 `vce(robust)` 在数学上确实不是同一个估计量，因此"逐族映射"是
**实现细节**，不是用户接口。具体分三层：

| 估计族 | 代表类 | `"robust"` 解析为 | 与 HC 的关系 | 证据 |
|---|---|---|---|---|
| OLS/线性 | `OLS`、`newey` | `HC1` | = HC1，≠ HC0 | 实测 `robust==HC1` 为 True |
| IV | `IV2SLS`/`IVLIML`/`IVGMM` | `HC0` | = HC0，≠ HC1 | `iv.py:266-267`；实测 |
| 离散/计数（ML） | `Logit`/`Poisson`/`ologit` | **`robust`（自成一类）** | ≠ HC0 **且** ≠ HC1 | `discrete.py:154-155`：`n/(n-1)`，非 `n/(n-k)` |
| 面板 | `PanelOLS`/`RandomEffects`… | **`cluster(panelvar)`** | 无 HC 对应 | `panel.py:203,220`；实测 `robust==cluster(firm)` 为 True |
| HDFE | `FEOLS`、`Areg` | **拒绝（假拒绝，应接受→HC1）** | 要求写 `HC1` | `hdfe.py:114` |
| 系统 | `Sureg`/`Mvreg` | `robust`（`sureg` 专属三明治） | 无 HC 对应 | `systems.py:321-324` |
| 分位数 | `QuantileRegression` | `robust`（`qreg` 专属） | 无 HC 对应 | `quantile.py:483-484` |
| 分位数·自助 | `Sqreg`/`Bsqreg` | `bootstrap`（协方差来自重抽样） | 无 HC 对应 | 实测回显 `'bootstrap'` |
| 稳健回归 | `RobustRegression` | **静默降级为默认值** | — | 实测：任何 `cov_type` 都被忽略 |
| 面板 IV/GLS/PCSE | `XtIvreg`/`XtGls` | **拒绝（真缺失）** | — | `panel_iv.py:381-385` |
| 面板非线性 | `XtLogit` 等 | **拒绝（真缺失）** | — | `UnsupportedError` |
| 症状方差 | `Eivreg` | **拒绝（真缺失）** | — | 实测 `UnsupportedError` |
| Tobit | `Tobit` | **拒绝（真缺失）** | — | 实测 `UnsupportedError` |

**这张表本身就是问题**：用户必须记住 5 种不同的 `robust` 语义，其中 3 种彼此
数值不同；而在 `FEOLS`/`XtIvreg` 上同一个词直接报错。这不是「Stata 就是这样」——
Stata 里 `vce(robust)` 在每个命令上都有**唯一且已文档化**的含义，westata 只是
把这个含义各自编码了一遍，没有一个地方能回答「`robust` 到底是什么」。

更糟的两处（实测）：

```python
# 1. RobustRegression 把 cov_type 整个吞掉，数值不变
a = ws.RobustRegression.from_formula("y~x", df).fit()
b = ws.RobustRegression.from_formula("y~x", df).fit(cov_type="robust")
a.cov_type, b.cov_type        # ('nonrobust', 'nonrobust')
np.allclose(a.bse, b.bse)     # True   <-- bse 与默认逐位相同

# 2. Sqreg/Bsqreg 回显 'bootstrap'，这个词不在 SUPPORTED_COV_TYPES 里
ws.Sqreg.from_formula("y~x", df, quantiles=(0.25,0.5)).fit().cov_type   # 'bootstrap'
```

#### 规范

**R3.7.1 `robust` 是别名，不是独立值。** 任何估计族收到 `cov_type="robust"`
必须立刻解析成一个**具体**的规范值并存进结果对象。以下为规范映射表（与上表一致，
但把「自成一类」的三处显式命名）：

| 估计族 | `"robust"` → 规范值 | Stata 对应 | 公式 |
|---|---|---|---|
| 线性/OLS | `HC1` | `regress, vce(robust)` | `n/(n-k) · (X'X)⁻¹ X'ΩX (X'X)⁻¹` |
| IV | `HC0` | `ivregress, vce(robust)` | 无 `n/(n-k)`；`small` 才加 |
| 离散/计数 ML | `ML_ROBUST`（新名） | `logit ... , vce(robust)` | `n/(n-1) · H⁻¹ (Σ s s') H⁻¹` |
| 面板 | `cluster`（panelvar） | `xtreg, vce(robust)` ≡ `vce(cluster panelvar)` | CR1 |
| HDFE | 拒绝，提示改 `HC1` | `reghdfe, robust` | 同 HC1 |
| 分位数·自助 | `bootstrap` | `sqreg`/`bsqreg`（无 `vce()`） | 重抽样协方差 |
| 稳健回归 | 拒绝 | `rreg`（无 `vce()`） | — |
| 系统/分位数 | `SYSTEMS_ROBUST` / `QREG_ROBUST` | 各有专属 ado | 见各自 docstring |

**新增规范值 `ML_ROBUST`**（离散/计数的 ML 三明治）的理由：它现在是
`cov_type="robust"` 这个**字符串**，与 OLS 的 HC1 同名不同值。用户从 `Logit`
换到 `OLS`、把 `cov_type="robust"` 抄过去，得到的是两种完全不同的 VCE，
而结果对象回显的都是 `"robust"`——`OLS` 那边回显 `HC1`（已规范化），
`Logit` 这边回显 `"robust"`（未规范化），两者看起来「一样」，实则不同。

#### 这些 `robust` 在数学上是不是同一个估计量？

**不是，但性质分三层。** 这一点必须写清，否则「同名不同值」会被误解成
纯粹命名问题：

**第一层——同一个三明治，只差一个标量因子。** 数学形式都是
`bread · meat · bread`，区别只在乘的因子：

| 族 | 因子 | 说明 |
|---|---|---|
| `regress`（→ HC1） | `n/(n-k)` | 标准 OLS 小样本修正 |
| `ivregress`（→ HC0） | `1` | IV 不加修正；`small` 才加 |
| `logit`/`probit`/`poisson`/`nbreg`（→ **`ML_ROBUST`**） | **`n/(n-1)`** | ML 只修正均值，不修正参数个数 |

实测（n=200，`Logit`）：

```
无截距 k=1 :  robust/HC0 = 1.00250941   robust/HC1 = 1.00000000
有截距 k=2 :  robust/HC0 = 1.00250941   robust/HC1 = 0.99748427
理论 n/(n-1) ÷ n/(n-k) 在 k=2 时 = (n-2)/(n-1) = 0.994975，实测吻合
```

**`n/(n-1)` 与 `n/(n-k)` 在 k=1 时恰好相等**，所以只要测试模型是「无截距单参数」，
三者就分不开——这是这个差异能潜伏至今的直接原因（见 R3.7.5）。

**第二层——完全不同的估计量。** `xtreg, vce(robust)` **就是**
`vce(cluster panelvar)`（`[XT] xtreg` 明文："Specifying vce(robust) is
equivalent to specifying vce(cluster panelvar)"）。它算的是**聚类**三明治，
与异方差稳健无关。实测 `PanelOLS(robust).bse` 与
`PanelOLS(cluster, groups=panelvar).bse` 逐位相同。`xtreg` 因此不提供纯 HC
选项，`HC0` 被拒绝是正确行为（`panel_iv.py` 与 `panel.py` 都如此）。

**第三层——命令专属三明治。** `qreg`/`sqreg` 用 `tau(1-tau) J⁻¹(X'X)J⁻¹`，
`sureg` 用 `reg3` 的跨方程三明治，各自与 `regress` 的公式无关，不能互相换算。

结论：**只有第一层里的三个因子是「同公式不同缩放」，第二、三层是不同估计量。**
`ML_ROBUST` 这个新名不是命名洁癖——它是数学上确实存在的第三种缩放，
塞进 `HC0` 或 `HC1` 任何一个都会给出错误的数。

**R3.7.5 `robust` 的语义测试必须用 k≥2 的模型。**

由上面的 `k=1` 相等性直接推出：用「无截距单参数」模型对拍 `robust`，
`n/(n-1)`、`n/(n-k)` 与 `1` 中至少两对会重合，测试「通过」却不证明任何事。
校验器 `check_r3_cov_type` 的固定装置因此全部带截距且 `k≥2`
（`y ~ x` 为 k=2，`y ~ x + z` 为 k=3）。
这条与 `error.md` 已知的「只在一个参数配置下对拍等于没对拍」是同一条教训。

**R3.7.2 结果对象的 `cov_type` 必须是规范值，不能是用户输入的回声。** ✅ 已修（2026-11-13）

修复前实测 9 处违反：

| 类 | 修复前回显 | 现在回显 | 性质 |
|---|---|---|---|
| `Logit` / `Probit` / `Poisson` / `OrderedLogit` | `'robust'` | **`ML_ROBUST`** | 未规范化 |
| `PanelOLS` / `RandomEffects` | `'robust'` | **`cluster`** | 未规范化 |
| `QuantileRegression` | `'robust'` | **`QREG_ROBUST`** | 未规范化 |
| `Sureg` | `'robust'` | **`SYSTEMS_ROBUST`** | 未规范化 |
| `RobustRegression` | `'nonrobust'`（吞掉） | **明确拒绝** | 静默吞掉 |

对比达标的：`OLS` → `'HC1'`，`IV2SLS` → `'HC0'`，`Sqreg`/`Bsqreg` → `'bootstrap'`。

**这次改动踩到一个必须记住的坑**：改标签没有改任何数值，但**五个后估计模块各自
写了一份 `cov_type in ("hc0","hc1","robust",...)` 的字符串判断**，改标签后它们
全部静默失效——`estat hettest` 在 robust 拟合后不再拒绝，会打印 Stata 不给的检验，
而**任何数值对拍都发现不了**。已抽出 `postestimation/vce_labels.py`
（`is_robust` / `is_hac` / `is_clustered`），并用
`tests/test_cov_type_labels.py` 的 31 个用例钉住「所有合法标签都被认，所有守卫都触发」。

**R3.7.3 `panel` 的 `robust`/`HC1` 必须回显 `cluster`，不能回显 `robust`。** ✅ 已修（2026-11-13）

修复前：`panel.py` 把 `{"robust", "hc1"}` 都映射成内部标记 `"robust"`，再展开成
`vce(cluster panelvar)`，结果对象回显 `"robust"`：

```python
m = ws.PanelOLS.from_formula("y ~ x", panel, entity="firm", time="year")
m.fit(cov_type="HC1").cov_type   # 修复前 -> 'robust'；现在 -> 'cluster'
m.fit(cov_type="HC0")            # -> UnsupportedError（正确：xtreg 没有 HC0）
```

已核对：`robust` 与 `HC1`、与 `cluster(groups=panelvar)` 给出逐位相同的 `bse`，
所以三者确实是同一个估计量；问题只在**标签**。这违反 R1.5「不能将错就错」：
用户在 `etable` 里看到 `cov_type=robust` 无法知道那是聚类协方差，会把它与
`OLS(cov_type="robust")`（=HC1）当作可比，而 `panel.py` 自己的注释记录了
`areg` 与 `xtreg` 在同数据上常数项 SE 相差 18.4 倍。

规范：`panel` 族在 `robust`/`HC1` 下回显 `"cluster"`（并让 `cov_kwds` 的空值
有意义：标签由 panelvar 自动填入）。`HC0` 继续拒绝，这是正确行为。

**R3.7.4 报错文案必须点名调用者用的类。** `Areg(...).fit(cov_type="robust")`
报的是 `FEOLS` 错误（`hdfe.py:114` 的文案写死）：

```
ValueError: cov_type 'robust' is not supported by FEOLS: use cov_type='HC1' ...
```

用户调用的是 `Areg`，看到的是 `FEOLS`。规则：错误信息里的类名/命令名来自
被调用的实例（`self._stata_command` 或 `type(self).__name__`），不能写死。
同类问题见 R6.4。

---

## 4. 返回对象

### R4.1 两类结果，两种契约

| 契约 | 谁 | 必须提供 |
|---|---|---|
| **估计结果** | `fit()` 的返回：`StataResults` | statsmodels 面（§4.2）+ Stata 面（§4.3）+ `.e` 映射 |
| **表格结果** | `estat_*` / `*_test` / 诊断类的返回 | `to_dict()`、`to_frame()`、`render()` 三选二，且属性可直接读 |

现状：18 个结果类**无一**提供 `to_dict`/`to_frame`（§7 基线 R4.1）。
`LlcResult`、`IpsResult`、`RdRobustResult`、`TefectsRaResult`、`NnmatchResult`、
`NeweyWestResult`、`UnitRootResult`、`XtTest0Result`、`VarWleResult`、
`DurbinWatsonResult`、`PortmanteauResult`、`SummarizeTable` 连 `summary()` 都没有，
只有 `__str__`，其数值只能从打印文本里抠或读 `__dict__`。

### R4.2 估计结果必须是 statsmodels 鸭子类型

`StataResults` 已达标，`protocols.ResultsProtocol` 是契约声明：
`params`、`bse`、`tvalues`、`pvalues`、`nobs`、`df_resid`、`df_model`、`llf`、
`rsquared`、`rsquared_adj`、`cov_params()`、`conf_int()`、`summary()`、`e`。

`PraisResult` 是游离的第三个实现（`prais.py` 里自建 `params`/`bse`/
`conf_int`/`cov_params`/`summary`），且**未通过** `isinstance` 检查
（实测 `isinstance(ws.prais(df, "price ~ weight"), ResultsProtocol)` → `False`）。
`VARResults`、`VECMResults` 同样为 `False`。只有 9 个走 `StataResults` 的类
（`OLS`/`IV2SLS`/`FEOLS`/`PanelOLS`/`Logit`/`Tobit`/`QuantileRegression`/`ARIMA`…）
与 `NeweyWestResult`（转发 `_results`）为 `True`。
规范：**所有估计结果继承同一基类或显式实现 `ResultsProtocol`**，
`isinstance(result, ResultsProtocol)` 必须为真。（`ResultsProtocol` 是
`@runtime_checkable`，可以直接断言，不需要额外测试设施。）

### R4.3 `conf_int` 全树同一形状

```python
def conf_int(self, alpha: float | None = None) -> pd.DataFrame:  # index=参数名, columns=['lower','upper']
```

- `alpha` 默认 `None` = 用模型自身 `level`（`StataResults` 已如此，`base.py:887`）。
- 返回 `DataFrame`，列名 `lower`/`upper`，索引为参数名——`PraisResult` 已如此
  （`prais.py:232`），可作为模板。
- 例外：`ContrastResults`、`VECMResults` 返回 `ndarray`，因为它们是 statsmodels
  兼容层，保持 statsmodels 形状（校验器登记为例外）。

### R4.4 一个结果类只暴露一种参数名拼写

`StataResults` 同时有 `params`（公式拼写，如 `C(foreign)[T.Foreign]`）、
`stata_param_names`、`stata_params`（Stata 拼写，如 `1.foreign`）、
`stata_cov_params`。这是**有意的双拼写**，且 `README` 已经明说
「use that exact string」。规范：保留，但必须在 API 文档里成段解释，
且新增结果类不得自造第三套拼写。

### R4.5 提取方法命名固定

| 方法 | 返回 | 用在哪 |
|---|---|---|
| `to_dict()` | `dict[str, Any]` | 标量结果 |
| `to_frame()` | `pd.DataFrame` | 表格结果 |
| `to_matrix()` | `np.ndarray` | 矩阵结果（仅 `MTestResult` 有，保留） |
| `render()` | `str` | 屏幕文本，与 `summary()` 二选一 |

禁止 `by_equation()`（`varwle.py`）这类一次性访问器替代 `to_frame()`；
它可以存在，但不能**替代**。

---

## 5. 词汇表（唯一拼写）

| 概念 | 唯一名字 | 禁止 |
|---|---|---|
| 数据帧 | `data` | `frame`, `df`, `dataset`, `sample` |
| 公式 | `formula` | `fml`, `spec`, `rhs` |
| 因变量（矩阵接口） | `endog` | `y`, `response`, `depvar`, `outcome` |
| 自变量（矩阵接口） | `exog` | `X`, `design`, `regressors` |
| 参数名列表 | `param_names` | `names`, `variable_names`, `exog_names`, `outcome_names` |
| 权重 | `weights` + `weight_type` | `freq_weights`, `w`, `wvar` |
| 方差估计类型 | `cov_type` | `vce`, `vcetype`, `robust`（作为参数名） |
| 方差估计细节 | `cov_kwds` | `cov_kwargs`, `cluster_kwds` |
| 小样本修正 | `ssc` | `SSC`, `small_sample` |
| 置信水平 | `level`（分数） | `conf_level`, `confidence`, `level=95` |
| 显著性水平 | `alpha`（仅 `conf_int`） | `size`, `p_cut` |
| 行掩码 | `subset` | `if_`, `condition`, `mask` |
| 面板个体/时间 | `entity` / `time` | `id`, `group`, `panelvar`, `timevar`, `period` |
| 迭代上限 | `maxiter` | `iterations`, `max_iter`, `niterations` |
| 收敛容差 | `tol` | `tolerance`, `epsilon`, `rtol` |
| 随机种子 | `seed` | `random_state`, `rng` |
| 重抽样次数 | `reps` | `n_boot`, `B`, `nboot` |
| 未收敛时是否报错 | `strict` | `raise_on_failure`, `hard` |

`entity`/`time` 已是全树一致（`panel.py`/`panel_iv.py`/`panel_nonlinear.py`/
`panel_dynamic.py` 共用），保持。`suest(models)` 的 `models`、`etable(results)`
的 `results` 保留：它们不是数据帧。

关于 `names`：全树 58 处出现，用途分三类，必须分开命名——
参数名列用 `param_names`（27 处已如此），方程名用 `equation_names`
（`systems.py` 已如此），其余业务含义的 `names` 保留但需要 docstring 说明。

---

## 6. 模块与文件级约束

### R6.1 被 3 个及以上模块需要的私有 helper 必须抽到公共模块

实测情况（`scripts/check_api_conventions.py --rule R6`）：

| helper | 份数 | 位置 |
|---|---|---|
| `_is_constant` | **11** | `panel_iv.py` + `postestimation/{endog_regf,endog_score,endogenous,firststage,hettest,imtest,overid,ovtest,summarize,vif}.py` |
| `_stata_number` | 4 | `first_stage_table.py`, `summary_stats.py`, `vecstable.py`, `results/summary.py` |
| `_require_iv` | 4 | `endogenous.py`, `firststage.py`, `overid.py`, `weakrobust.py` |
| `_numerical_hessian` | 3 | `nonlinear.py`, `panel_nonlinear.py`, `tobit.py` |
| `_ols` | 3 | `panel_iv.py`, `resample.py`, `firststage.py` |
| `_slope_positions` | 3 | `discrete.py`, `limited.py`, `systems.py` |
| `_stars` | 3 | `report/etable.py`, `results/estimates_table.py`, `results/esttab.py` |

**一个反直觉的实测结果**：校验器按「函数体是否逐字节相同」区分后，这 7 组
**没有一组是逐字节相同的复制**——它们是「同名但实现不同」，比单纯复制更危险：

> 同一个名字在 7 个地方给出 7 种行为，调用方无法从名字判断自己拿到哪一个，
> 而修改一处不会同步到其余六处。

`_is_constant` 的 11 份是最大风险：判断「这一列是常数吗」的口径
（`std == 0`？`ptp < 1e-12`？是否忽略 `_cons`？）稍有差异，11 个后估计命令
就会对同一结果给出不同的「被省略变量」列表。抽到 `westata/postestimation/names.py`
（该模块已有 `is_constant_name`/`stata_display_name`）。

**例外：`_normalize_cov_type` 有四份，但不应合并。** `did.py` 把 `robust` 映射到
HC1，`panel.py` 映射到聚类协方差，`hdfe.py` 转发到自己的 `_COV_ALIASES`，
`vce/dispatch.py` 收 `robust_as` 参数。这是**四个不同的答案**，不是同一个函数抄了四遍；
合成一个就会重建 R3.7.0 刚刚拆掉的那个「全局默认」缺陷。校验器的
`PER_FAMILY_HELPERS` 因此豁免它，并注释说明理由——豁免不是遗漏，是结论。

### R6.2 验收证据与接口规范分离

`conformance/` 只做证据（registry、tolerances、oracles），**不得**被估计器
反向依赖来「顺手用一下」。现状 `vce/dispatch.py` 与 `conformance/registry.py`
都在做 cov_type 校验，职责重叠。

### R5.5 每个模块必须能从原始字节编译

`import westata` 成功**不能**证明文件没问题：`.pyc` 缓存会让旧的编译结果继续生效。
必须对 `read_bytes()` 的结果直接 `compile()`。

这不是假设。2026-11-14 做 R6.1 时，一个批量改写脚本把 `panel.py` 注释里的
`\x0c`（换页符）拆成了真换行，文件立即无法解析，但 `import westata` 仍然成功——
直到 15 分钟后的全量测试才暴露（47 failed + 20 errors）。校验器现在对每个模块做
这一次编译，代价不到一秒。

同时注意：本仓库源码是 **LF**，用文本模式写回会引入 CRLF 造成整文件 diff；
批量改写要用 `read_bytes`/`write_bytes`。

### R5.6 公共 classmethod 的返回注解必须指向调用者能命名的类型

`Logit.from_formula` 曾注解为 `-> "_DiscreteBase"`：一个调用者永远看不到的私有基类。
11 个估计器都是这样——注解描述了一个不可命名的对象。

规则：返回注解要么是**具体的公共类**，要么是绑定到基类的 `TypeVar`
（如 `_DiscreteT`，这是 "the concrete `cls`" 的约定写法）。指向具体私有类的注解视为违规。

### R5.7 `estat_*` 必须同时以函数和方法两种形式存在

README 的主路径是 `result.estat_vce()`，而 Stata 的写法是估计之后的 `estat vce`；
移植 Stata 代码的调用者会期望 `ws.estat_vce(result)`。实测有 6 个函数曾是
**仅方法**形式（`estat_auc`/`estat_esize`/`estat_moran`/`estat_mundlak`/
`estat_szroeter`/`estat_weakrobust`），第二种写法直接 `AttributeError`。

现在 21 个 `estat_*` 在两侧完全对齐，校验器逐名比对。

### R5.8 公共方法必须暴露真实签名，不能只有 `**kwargs`

`StataResults` 上有 **29 个**后估计方法曾写成 `def estat_vif(self, **kwargs)`，
转发到模块函数。后果与 `test_subset_is_uniform.py` 记录的
`IV2SLS.from_formula` 事故是同一类：**功能存在，但 `inspect.signature` 看不见**，
于是 `help()` 什么都不显示、自动补全拿不到参数、工具化检查把"有"报成"无"。

现在 29 个全部展开成目标函数的真实签名，`inspect.signature(res.estat_vif)`
给出 `(uncentered=False, nosort=False)`。

规则只在「除 `self` 外**一个具名参数都没有**」时触发——`test(self, *specs, **kwargs)`
这种真正变参的方法是合法的。

### R6.3 模块不得自带 BOM

`expressions/__init__.py`、`random/__init__.py` 两处带 UTF-8 BOM，导致
`ast.parse(raw_bytes)` 失败、每个读取方都要记得 `utf-8-sig`
（`test_packaging_hygiene.py` 已经为 `pyproject.toml`/`MANIFEST.in` 立过规矩，
源码文件漏了）。

---

## 7. 基线：0.3.11 实际偏离

`python scripts/check_api_conventions.py` 在 0.3.11 上的输出：

| 规则 | 条数 | 已登记例外 | 待修 | 性质 |
|---|---:|---:|---:|---|
| R2.1 `fit()` 首参不是 `cov_type` | 0 | 0 | 0 | ✅ 阶段一 |
| R2.3 DataFrame 叫 `frame` | 0 | 0 | 0 | ✅ 阶段三 |
| R2.5 非 snake_case | 14 | 14 | 0 | ✅ 阶段三（Stata 选项名与线性代数记号） |
| R2.6 `**kwargs` 收下即丢 | 0 | 0 | 0 | ✅ 阶段一 |
| R3.1 `noX` 开关 | 0 | 0 | 0 | ✅ 阶段三（Stata 选项名按 R1.2 保留） |
| R3.2 `from_formula` 名字骗人 | 5 | 5 | 0 | ✅ 阶段三（主入口改名，4 条为弃用别名） |
| R3.3 `level` 单位不一 | 0 | 0 | 0 | ✅ 阶段一 |
| R3.7 `robust` 未规范化或未拒绝 | 0 | 0 | 0 | ✅ 已完成 |
| R4.1 结果类缺 `to_dict`/`to_frame` | 4 | 4 | 0 | ✅ 阶段二 |
| R4.2 估计结果不满足 `ResultsProtocol` | 3 | 3 | 0 | ✅ 阶段二 |
| R4.3 `conf_int` 返回不是 DataFrame | 2 | 2 | 0 | ✅ 阶段二（全例外） |
| R5.1 `__all__` 名不副实 | 0 | 0 | 0 | ✅ 阶段二 |
| R5.2 子包导出未上浮 | 0 | 0 | 0 | ✅ 阶段二 |
| R5.3 BOM | 0 | 0 | 0 | ✅ 阶段二 |
| R6.1 helper 重复 ≥3 | **0** | 0 | 0 | ✅ 阶段四 |
| **合计** | **28** | **28** | **0** | |

**阶段一（会算错数/静默失效）与阶段二（可用性）已完成。** 关键实测：

| 规则 | 修复前 | 修复后 |
|---|---|---|
| R2.1 | `PanelOLS.fit("HC1")` 把字符串赋给 `entity_effects`，返回非稳健 SE `0.060844` | `0.000409`（正确生效） |
| R2.6 | `VECM.fit(bogus=1)`、`XtUnitroot.fit(zzz=[1,2,3])` 正常返回 | 抛 `TypeError` |
| R3.3 | `estat_esize(level=0.95)` 算 `(100-0.95)/2 = 49.5%` | 分数单位，正确 |
| R4.1 | 22 个结果类无提取方法，18 个只有 `__str__` | 全部提供 `to_dict`/`to_frame` |
| R5.1 | `from westata.results import *` 抛 `AttributeError` | 正常 |
| R5.2 | 35 个公共名在子包可见、顶层不可见 | 顶层导出 167 → 196 |

顺带修掉两个**预先存在**的缺陷：

1. `estat_esize().ci_lower` 恒为 `nan`：`_noncentral_f_bounds` 在 F 较小时
   `lower_gap` 全定义域为正、无零点，`brentq` 失败返回 `nan`。数学上正确下界是
   0（效应量为 0），不是缺失值。
2. `VARResults.params` 是 Stata 的**方程×变量表**（DataFrame），
   而 `ResultsProtocol` 要求一维 `params`。两者不可兼得，已登记为例外并说明
   「多方程模型没有单一系数向量」——不是遗漏，是结论。



「已登记例外」= `scripts/check_api_conventions.py` 的 `ALLOWED` 表，每条例外都写明
依据：R2.5 的 12 条（线性代数层的 `X`/`Xw`/`resid` 8 条 + statsmodels 形状的 `R` 4 条）、
R3.2 的 1 条（多方程构造 `Sureg(equations, data)`）、
R4.1 的 4 条与 R4.2/R4.3 的 4 条（`ContrastResults`/`VECMResults`/`VARResults` 是
statsmodels 兼容层，`StoredResults` 是 `e()`/`r()` 标量映射）。
例外不是豁免通道：新增例外必须在本文件对应规则下说明理由，
`tests/test_api_conventions.py::test_allowed_table_entries_name_a_rule_that_exists`
会拦下指向不存在规则的过期条目。

**R4.2 报出的两条是真缺陷**，不是兼容层问题：

```
PraisResult : 缺 df_model, e, llf
VARResults  : 缺 conf_int, df_model, df_resid, llf, pvalues, rsquared, rsquared_adj, tvalues
```

`PraisResult` 自建了 `params`/`bse`/`conf_int`/`cov_params`/`summary`，看起来像
估计结果却少了三个属性；`VARResults` 缺 8 个。用户用 `statsmodels` 习惯遍历一组
结果对象时会在这里崩——`res.pvalues` 在 `OLS` 上有、在 `VAR` 上没有。

除此之外，以下偏离**校验器覆盖不到**，但同样是规范违背，需人工核对：

| 问题 | 证据 |
|---|---|
| 同一 `fit()` 默认值五花八门 | `maxiter` 6 种取值；`tol` 13 处显式 + 51 处缺省；`strict` 33 处缺省、31 处 `False`、1 处 `True`（`Nl`） |
| 同一 `method=` 两个位置 | `Heckman`/`Etregress`/`MultinomialLogit` 在 `fit()`；`XtLogit`/`XtProbit`/`XtTobit`/`XtPoisson`/`XtNbreg`/`XtIvreg`/`XtGls`/`XtUnitroot` 在 `from_formula()` |
| 只有 5/65 个类支持 `fit(level=)` | `OLS`/`IV2SLS`/`IVLIML`/`Areg`/`FEOLS`；其余要改置信水平只能用 `conf_int(alpha=)`，而 `fit(level=)` 与 `conf_int(alpha=)` 是同一件事的两种拼法 |
| 估计结果类不带 `ResultsProtocol` | 实测 `PraisResult`/`VARResults`/`VECMResults` 三者 `isinstance(..., ResultsProtocol)` 为 `False`，其余为 `True` |
| 版本字符串双源 | `pyproject.toml` 与 `__init__.py` 的 fallback `"0.3.1"`（实际 0.3.11） |

---

## 8. 迁移路线（按风险排序，不按文件顺序）

阶段划分的依据是**错误后果**，不是改动量：会算错数的先修，只是难看的后修。

### 阶段一：会算错数或静默失效（R2.1 / R2.6 / R3.3 / R3.7）—— ✅ 全部完成

1. ✅ `PanelOLS` / `RandomEffects` / `RandomEffectsMLE` 的 `fit()` 改为
   `(cov_type, cov_kwds, ssc, *, entity_effects, time_effects)`。
   `BetweenOLS` / `FirstDifferenceOLS` 继承 `RandomEffects.fit`，一并生效。
   这会**破坏** `fit(False)` 这类旧调用——按 §0 总原则，宁可破坏也不保留会算错数的
   签名。实测修复前后：`fit("HC1")` 由「返回非稳健 SE 0.060844」变为
   「正确返回稳健 SE 0.000409」。
2. ✅ `Heckman` / `Etregress` 的 `fit()` 改为 `(cov_type, *, method, ...)`。
   `method` 没有移到 `from_formula`（与 `XtLogit` 不同）：这两个类还能从裸数组
   构造，没有 formula 调用可挂。关键是不再让 `fit("robust")` 被读成
   `method="robust"` 并报「method 未实现」——那会把用户指向错误的参数。
3. ✅ `VECM.fit` / `XtUnitroot.fit` / `MultinomialLogit.fit` 去掉吞参，
   未知参数抛 `TypeError`。`MultinomialLogit` 顺带补上了 `strict`——
   其余 ML 族都有，只有它收下即丢。
4. ✅ `estat_esize` 的 `level` 改为分数（同步修 `EffectSizeResult` 内部的
   `(1-level)*50` 显示）；`estat_weakrobust` 的 `level` **删除**而不是改单位——
   它只服务于未实现的 `ci` 选项，任何取值都被丢弃，删掉比改单位诚实。
   `im_pesaran_shin` 的死参数 `level` 已在早前删除。
   顺带修掉预先存在的 `ci_lower=nan` 缺陷（见 §7）。
5. ✅ **`cov_type` 规范化（R3.7）**
   - ✅ **5a** `Logit`/`Probit`/`Poisson`/`OrderedLogit`/零膨胀/截断族 的 `robust`
     回显为 `ML_ROBUST`；`QuantileRegression` → `QREG_ROBUST`；`Sureg` → `SYSTEMS_ROBUST`。
   - ✅ **5b** `PanelOLS` 族 的 `robust`/`HC1` 回显为 `cluster`。
   - ✅ **5b′** `FEOLS`/`Areg` **接受** `robust`（→`HC1`），删除了那条
     "use cov_type='HC1'" 的拒绝；`Areg` 报错文案不再写死 `FEOLS`。
   - ✅ **5b″** `Areg(cov_type="HC2"/"HC3")` 的 `results.cov_type` 不再停留在占位值
     `HC1`（新增 `StataResults.relabel_cov_type`）。
   - ✅ **5b‴** `RobustRegression` 不再静默吞掉 `cov_type`，改为明确拒绝并说明
     "rreg 没有 vce()"。
   - ✅ **5c（已按用户选择实施，2026-11-13）** 取消 `vce.dispatch` 的全局
     `"ROBUST": "HC1"`（用户明确选择 C①）。现在：

     ```python
     _normalize_cov_type("robust")                       # ValueError：不猜
     _normalize_cov_type("robust", robust_as="HC1")      # HC1  (regress)
     _normalize_cov_type("robust", robust_as="HC0")      # HC0  (ivregress)
     _normalize_cov_type("robust", robust_as="cluster")  # cluster (xtreg)
     ```

     族表 `ROBUST_AS = {"regress": "HC1", "ivregress": "HC0", "xtreg": "cluster"}`
     显式声明，`ml` **故意不在表内**——ML 三明治需要信息矩阵与得分，而
     `compute_cov` 只拿到 `(X, resid)`，给了表项也做不出来。这条由
     `test_robust_as_table_covers_only_the_compute_cov_families` 钉住。

     好处：将来新增估计族若忘记声明，会**立刻报错**并被告知原因，而不是静默拿到
     HC1——这正是这次重构想消除的失效模式。用户侧的 `cov_type="robust"` 不受任何
     影响（各族 `fit()` 自己解析）。
6. ✅ 重新计算受影响的对拍期望值。`level` 语义改变影响 `estat esize` 的区间；
   实测 `test_estat_esize.py` 15 项、`test_estat_weakrobust.py`、
   以及 panel/limited/tseries 相关 254 项全部通过。`cov_type` 的规范化只改标签，
   数值未动，`test_ereturn_key_spelling` / `test_vce_type_macros` /
   `test_ml_vce_label` 均通过。

### 阶段二：可用性（R4.1 / R4.2 / R5.1 / R5.2 / R5.3）—— ✅ 全部完成

1. ✅ 修 `results/__init__.py` 的 `__all__`：`render_estimates_table` 从未被 import，
   `from westata.results import *` 直接抛 `AttributeError`。现在该模块导入并导出
   `EReturn`/`StataResults`/`etable_as_stata`/`render_esttab`/`render_estimates_table`。
2. ✅ 18 个结果类补提取方法，通过新增 `results/serialisation.py` 的
   `ResultSerialisation` mixin 统一实现——写 18 份几乎相同的 `to_dict` 就是
   R6.1 已经记录过的问题。三个非 dataclass 类单独处理：
   - `PraisResult`：`to_dict` 读 12 个标量属性，`to_frame` 输出系数表；
   - `NeweyWestResult`：`StataResults` 本身不提供 `to_dict`（R4.1 例外），
     所以从 statsmodels 面自建，并补上 `lag`；`nan` 的量（`llf`/`r2`）**省略键**
     而不是输出 `NaN`，因为 Stata 的 `newey` 本来就不报这些；
   - `MarginsTable`：`list` 子类，`to_dict` 返回 `list[dict]`（每行一个），
     `to_frame` 输出多行表，不把多行压成一个 dict。
3. ✅ 35 个未上浮的公共名分类处理：
   - **上浮到 `ws.`（29 个）**：结果类型（`NnmatchResult`/`PsmatchResult`/
     `RdRobustResult`/`TefectsRaResult`/`BgTestResult`/`Correlogram`/`DFGLSResult`/
     `DurbinWatsonResult`/`KpssResult`/`LmarResult`/`MTestResult`/
     `PortmanteauResult`/`SummarizeTable`/`VarSocResult`/`VarStableResult`/
     `VarWleResult`/`VecRankResult`/`VecStableResult`/`XtTest0Result`/
     `EReturn`/`StataResults`）、命令函数（`prais_winsten`/`predict`/
     `supported_types`/`lincom_as_stata`/`test_as_stata`/`adjust_pvalues`）、
     常量（`MTEST_METHODS`）。顶层导出从 **167 增至 196**。
   - **从子包 `__all__` 移除（6 个）**：`build_contrast`/`describe_constraint`/
     `resolve_names`/`run_lincom`/`run_wald_test`/`statsmodels_wald_test`——
     它们是内部构造工具，通过命令间接使用，不该出现在公共面。
   - `SUPPORTED_TYPES`（内部 dict）移除，`supported_types()`（公共查询函数）上浮。
4. ✅ 两处 BOM 删除（`expressions/__init__.py`、`random/__init__.py`）。
5. ✅ `PraisResult` 补齐 `df_model` / `e` / `llf` 三个 `ResultsProtocol` 属性。
   `llf` 返回 `nan` 并在 docstring 说明：`prais` 是变换后的最小二乘拟合，
   Stata 的 `prais` 不报对数似然，编一个高斯似然就是造假（R1.5）。
6. ✅ `VARResults` / `VECMResults` / `ContrastResults` 登记为 R4.2 例外并写明理由：
   前两者是多方程/statsmodels 适配层，`params` 是方程×变量的**表**而不是一维向量；
   `ContrastResults` 镜像 statsmodels 的对比结果，本就不是估计结果。

### 阶段三：语义与命名 —— ✅ 完成

1. ✅ `frame` → `data`：`prais`、`im_pesaran_shin`、`levin_lin_chu` 三处改名；
   `synth2` 的 `frame=` 是 **Stata 的 `frame()` 选项名**（字符串），检查器曾按
   名字误报，已修正为按注解判断。`customV` 同样是 Stata 选项名，保留（R1.2）。
   `endog_regF_test` → `endog_regf_test`。
2. ✅ `noX` 开关：实测确认 `nomiss0`/`nosort`/`noheader`/`noconstant`/`nowald`/
   `noweights`/`nofigure`/`nonrtolerance` 全是 **Stata 原生选项名**，按 R1.2 保留
   （校验器新增 `STATA_NEGATIVE_OPTIONS` 表并逐条注明出处）。真正需要改的只有三处：
   - `noabsorb` → `absorb_test`（名不符实：关的是**检验**不是吸收）；
   - `normalise_counts` → `normalize_counts`（英式拼写，全仓库其余为美式）；
   - `dfgls` 删除 `notrend`（与 `trend` 控制同一状态）。
3. ✅ 4 个类的 `from_formula` → `from_variables`（`Didregress`/`RegressionAdjustment`/
   `IPW`/`NearestNeighbourMatching`），保留弃用别名一个版本（`retest/` 有约 180 处调用）。

顺带修掉一个**真实的静默缺陷**：`im_pesaran_shin` 与 `levin_lin_chu` 都计算了
`sample = data[[...]].dropna().sort_values(...)`，但**后续全程读取 `data` 而不是
`sample`**，所以缺失值从未被剔除。实测含 1 个缺失值的 144 行面板，修复后
`observations=131`（正确），修复前会用未清洗的序列。

### 阶段四：结构 —— ✅ 完成

1. ✅ `PraisResult` 补齐 `df_model`/`e`/`llf`，满足 `ResultsProtocol`；
   `VARResults`/`VECMResults`/`ContrastResults` 登记为例外（多方程模型的
   `params` 是方程×变量**表**，与协议的一维向量不可兼得）。
2. ✅ 7 组 helper 全部收敛。按「是否同一函数」分三类处理：
   - **真重复**（`_is_constant` 11 份、`_stata_number` 4 份、`_ols` 2 份、
     `_slope_positions` 3 份、`_numerical_hessian` 3 份、`_stars` 3 份）→ 抽公共实现；
   - **同名不同义**（`_ols` 在 `firststage` 里其实是「拟合值+R²」且参数顺序相反）
     → 改名为 `_fitted_and_r2`，避免误用；
   - **同形委托**（`_require_iv` 4 份仅命令名不同）→ 保留薄包装，校验器识别为合法形态。

   `_is_constant` 的 11 份里藏着两个真实 bug：`modeltests.py` 的集合含大写
   `"Intercept"` 却与 `.lower()` 后的名字比较（永不匹配）；`panel_nonlinear.py`
   少了 `"1"` 这个拼写。统一到 `postestimation/names.py` 的
   `CONSTANT_SPELLINGS` 后两处都修好。
3. ✅ 矩阵接口默认值里的具体变量名（R2.4）已清理：
   - `XtAbond.__init__` 的 `param_names=("L1.y",)` / `exog_names=("x1",)` → `None`，
     改为按 `lags` 和设计宽度推导；
   - `XtUnitroot.__init__` 的 `variable_name="y"` → `None`（直接构造时结果为 `''`
     而不是错误的 `y`；`from_formula` 仍传真实变量名）。

   两者都是「把首次开发所用数据集的列名写进默认值」——换数据就会得到错误的参数表。
   修 `XtAbond` 时还发现 `exog_names` 的约定是「不含常数项」（`_X_names()` 会前置
   `"Intercept"`），自动推导必须据此生成，否则 `fit` 里的位置索引会越界。

### 阶段六：补上检查器的盲区 —— ✅ 完成

阶段一到五"清零"之后复核，发现清零部分是**检查器盲区**造成的：

| 盲区 | 后果 | 规模 | 新增规则 |
|---|---|---|---|
| 只遍历 `__all__` | 未导出的结果类完全不可见 | 21 个公共结果类 | R5.2 扩展 |
| 不检查返回注解 | `from_formula` 注解指向私有基类 | 11 个估计器 | R5.6 |
| 只检查一侧 | `estat_*` 只有方法、没有函数 | 6 个命令 | R5.7 |
| 不检查原始字节 | 文件损坏被 `.pyc` 掩盖 | 1 次事故 | R5.5 |

处理结果：

- **21 个未导出的结果类**全部导出（顶层 196 → 217），其中 5 个补了提取方法
  （`RcmResult`/`VarGrangerResult`/`ContrastResult`/`PwCompareTable`，
  `VARResults`/`VECMResults` 保持例外）；
- **11 个私有基类注解**改为绑定 `TypeVar`（`_DiscreteT`/`_TruncatedBaseT` 等），
  Python 3.10 没有 `typing.Self`，所以用 `TypeVar(bound=...)`；
- **6 个 `estat_*`** 补齐函数形式，现在 21 个命令的方法与函数完全对齐。

另修正 R2.4 的两处硬编码默认值（`XtAbond` 的 `("L1.y",)`/`("x1",)`、
`XtUnitroot` 的 `variable_name="y"`），它们会把首次开发所用数据集的列名
带给任何其他数据。



阶段一到五"清零"之后，复核发现清零是**检查器的盲区**造成的，不是代码真的干净：

1. **未导出的结果类完全不可见**（R4.1/R5.2）。检查器只遍历 `__all__`，
   于是 `PanelUnitRootResult` 这类「命令会返回、但从未导出」的类逃过了两轮检查。
   实测有 **21 个**这样的公共结果类。已全部导出（顶层导出 196 → 217），
   并把规则改为**按源码扫描所有公共 `*Result/*Table/*Matrix` 类**，要求每个都被导出。
2. **结果类是否可序列化**（R4.1）。新导出的类里有 5 个没有提取方法，已补
   `RcmResult`/`VarGrangerResult`/`ContrastResult`/`PwCompareTable`；
   `VARResults`/`VECMResults` 保持例外（`params` 是方程×变量的表）。

教训：**"0 待修"必须先证明检查器没有盲区，否则它只说明"检查器没看那些地方"。**
新增的 R5.5（原始字节可编译）与本次的 R5.2 扩展都属于同一类修正。

### 阶段五：把校验器装进 CI —— ✅ 完成

待修数降到 **0** 后，`.github/workflows/ci.yml` 已切换为强制模式：

```yaml
- name: Public-API conventions (strict)
  run: python scripts/check_api_conventions.py --strict
```

`--strict` 现在实测 **exit 0**。28 条全部是已登记例外（每条在 `ALLOWED` 或
`STATA_NEGATIVE_OPTIONS` 里有出处），没有未处理项。

---

## 9. 校验器与规范的对应关系

| 规则 | 校验器 | 能否机械判定 |
|---|---|---|
| R2.1 | ✅ `--rule R2` | 是 |
| R2.2 | ❌ | 否（需人工判断参数归属） |
| R2.3 | ✅ | 是 |
| R2.4 | ⚠️ 部分 | 默认值已改为 `None` 并按数据推导；"这个默认值是否绑定到特定数据集"仍需人工判断 |
| R2.5 | ✅ | 是 |
| R2.6 | ✅（按「是否转发」判定） | 近似 |
| R3.1 | ✅ | 是 |
| R3.2 | ✅ | 是 |
| R3.3 | ✅（按默认值区间判定） | 近似 |
| R3.4 | ❌ | 否（已由 `test_subset_is_uniform.py` 覆盖） |
| R3.5 | ❌ | 否 |
| R3.6 | ❌ | 否（需跨族对比词表） |
| R3.7 | ✅（逐族实际拟合，比对 `result.cov_type`） | 是（行为级） |
| R4.1 | ✅ | 是 |
| R4.2 | ✅（比对 `ResultsProtocol` 的声明属性） | 近似 |
| R4.3 | ✅（按返回标注判定） | 近似 |
| R4.4/R4.5 | ❌ | 否 |
| R5.1 `__all__` 可解析 | ✅ | 是 |
| R5.2 导出完备 | ✅（含**未导出结果类**的源码扫描） | 是 |
| R5.3 BOM | ✅ | 是 |
| R5.5 原始字节可编译 | ✅ | 是 |
| R5.6 返回注解可命名 | ✅ | 是 |
| R5.7 `estat_*` 双向对齐 | ✅（方法集合 vs 函数集合逐名比对） | 是 |
| R5.8 方法签名可内省 | ✅（`(**kwargs)` 且无具名参数才触发） | 是 |
| R6.1 helper 重复 | ✅（按「函数体是否相同」区分复制 / 同名不同义 / 委托包装） | 是 |

规则号与检查函数**不是一一对应**：`from_formula` 的顺序规则编号是 R3.2，但由
`check_r2` 实现（它和其余构造函数规则放在一起读更自然）。因此 `--rule` 按
**Finding 自己的规则号**过滤，不按检查函数过滤；
`tests/test_api_conventions.py::test_each_rule_family_can_be_selected` 钉住这一点。

**已经发生过一次「死规则」**：R4.3 的 `conf_int` 分支最初只遍历 `__all__`，
而它要检查的两个 statsmodels 兼容类（`ContrastResults`、`VECMResults`）从未被导出，
于是该分支永远不触发，报告却照样打印「0 findings」。修正后它报出 2 条并登记为例外。
`test_every_rule_can_fire_on_a_synthetic_module` 就是为拦截这类问题写的：
每条规则都要能在一个故意违规的合成模块上触发。

「近似」的含义：校验器只能看到**声明**，看不到**行为**。因此 §7 里那些
「实测」条目（`XtUnitroot.fit(zzz=[1,2,3])` 正常返回）都附了可直接复制的
调用，人工核对时按原样执行即可。

---

## 10. 与既有文档的关系

| 文档 | 关系 |
|---|---|
| `README.md` / `README.zh-CN.md` | 用户第一入口。接口变更后必须同步（本仓库规则） |
| `docs/GUIDE.md` | 用法教程。`from_formula` 为主入口的表述已在此 |
| `docs/API.md` / `docs/API.zh-CN.md` | **自动生成**，签名来自 `docs/collect_api_reference.py`；改签名后必须重跑，`tests/test_api_reference_is_in_sync.py` 会拦 |
| `docs/api_source/WRITING_CONTRACT.md` | 文档撰写契约，与 §4.5 提取方法命名相关 |
| `tests/conformance/registry.yml` | 证据等级。§8 阶段一改 `level` 语义后必须重跑 |
| `error.md` | 通用问题记录。本次审计新增的接口层教训应写入 |
| `options.md` | 需用户拍板项。§8 阶段一的破坏性变更需记录 |
