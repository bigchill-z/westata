# 通用问题记录（避免以后再犯）

> 记录每次测试/排障中发现的、与具体任务无关的通用坑。

## 2026-11-14 打包发布前，`collect_api_reference.py --check` 是文档门禁

发布 0.3.13 前跑文档检查，发现 **52 个新导出的名字没有 prose**（工具报 49，
补上 3 个后是 52）。也就是说：新增了公共 API 而手册没跟上，读者查不到。

两个具体坑：

1. **测试套件通过 ≠ 文档完整**。`tests/test_api_reference_is_in_sync.py` 的 18 项
   全绿，但 `collect_api_reference.py --check` 报 52 个缺口——前者只校验**已有**
   条目的一致性，不校验**覆盖**。发布前必须单独跑后者。
2. **测试读的是 `api_data.json` 里内嵌的 prose**，不是 `prose_*.json` 源文件。
   改完 prose 必须重跑 `collect_api_reference.py` + `render_api_reference.py`，
   否则测试看到的是旧数据，会给出「失败但原因不明」的假象。

**通用做法**：凡是「生成型文档」，发布检查顺序固定为
`collect --check` → `collect` → `render` → 跑文档测试，缺一步就会出现
「测试绿但手册缺页」。

顺带记录本轮补文档时发现的真实缺陷：`VECMResults.conf_int()` 调用了
`self.raw.conf_int()`，而 statsmodels 的 VECM 结果**没有**这个方法
（只有分块的 `conf_int_alpha/beta/gamma/det_coef`，且返回结构化数组）。
这个包装方法从落地起就是坏的，任何调用都 `AttributeError`，
而没有任何测试覆盖它——因为没人调用过。

## 2026-11-14 「待修 0 条」可能只说明检查器没看那些地方

把规范基线降到 0 之后复核，发现**四处盲区**——每一处都让「0 条」变成假象：

| 盲区 | 后果 | 实测规模 |
|---|---|---|
| 只遍历 `__all__` | 未被导出的结果类完全不可见 | 21 个公共结果类（`PanelUnitRootResult` 等）逃过两轮检查 |
| 只看类名不看可达性 | 用户能拿到却无法命名的对象 | 同上 |
| 不检查返回注解 | `from_formula` 注解指向私有基类 | 11 个估计器 |
| 只检查一侧 | `estat_*` 只有方法形式、没有函数形式 | 6 个命令 |

**通用做法**：

1. **"清零"之后要反过来问：这条规则怎么才能不触发？** 如果答案是"换个写法就看不见"，
   规则就有盲区。本轮的 21 个未导出结果类就是这么找到的——它们明明以 `Result`
   结尾，却因为不在 `__all__` 里而躲过了"所有 Result 都要有 `to_dict`"的检查。
2. **校验器的判定范围要覆盖「用户能拿到的东西」，不是「导出表里的东西」。**
   改为扫描源码里所有公共 `*Result/*Table/*Matrix` 类，并要求每个都被某个包导出。
3. **对称性检查成本最低、收益最高**：`estat_*` 函数的集合与 `StataResults` 方法的
   集合逐名比对，一行代码就发现了 6 个缺口。凡是「同一功能两个入口」的地方都值得这么查。
4. **注解也是契约**。`-> "_DiscreteBase"` 描述了一个调用者看不到的类；机械检查
   「返回注解要么是公共类、要么是绑定 TypeVar」几乎零成本。

新增的四条规则（R5.2 扩展、R5.6、R5.7、以及此前的 R5.5）都属于同一类修正：
**先证明检查器没有盲区，再宣称达到目标。**

## 2026-11-14 批量改写源码时，`\x0c` 会被当成换行，LF 会变成 CRLF

本轮做 R6.1（消除重复 helper）时，用脚本批量改写 `panel.py`，把一段**注释里的
`\x0c`（换页符）**拆成了真正的换行：

```python
# 原文（一行，\x0c 是转义后的换页符）
# munnell.dta: \<FF>e\ gives .93913112, \<BS>e\ .99388704, e\ "." -- ...

# 被我的脚本写成了两行，且 \\ 变成行尾续行符
# munnell.dta: \
e\ gives .93913112, ...
```

结果：`SyntaxError: unexpected character after line continuation character`，
**整个 `panel.py` 无法导入**。而当时 `import westata` 仍然成功——因为
`.pyc` 缓存还在，直到一次全量测试才暴露（47 failed + 20 errors）。

**通用做法**：

1. **批量改写源码必须保留原始字节语义**：读用 `read_bytes()`，写用
   `write_bytes()`，不要经过 `read_text/write_text` 往返——文本模式会做换行
   规范化，也会把某些控制字符重新解释。
2. **改动后立刻对每个写过的文件做 `compile(raw_bytes, path, 'exec')`**，
   不要等测试。一条循环几秒钟，能立刻定位是哪个文件坏了。
3. **`import westata` 成功不代表文件没问题**：`.pyc` 会让旧的编译结果继续生效。
   判断依据是「重新编译原始字节」，不是「导入成功」。
4. 本仓库源码是 **LF**，HEAD 里 `panel.py` 的 CRLF 计数为 0；用
   `Set-Content`/`write_text` 写回会引入 CRLF，造成整文件 diff。改动要用
   `write_bytes(encode('utf-8'))`，且显式保留 `\n`。

## 2026-11-13 改「标签」会静默破坏所有按字符串判断的下游

R3.7.2 把结果对象上的 `cov_type` 从「用户输入的回声」改成「规范值」后，
`Logit(cov_type="robust").cov_type` 从 `"robust"` 变成 `"ML_ROBUST"`。
数值一个没动，但**五个后估计模块各自写了一份字符串判断**：

```python
# 改动前，五个文件各有一份
cov_type in ("hc0", "hc1", "hc2", "hc3", "robust")     # hettest / imtest / endogenous / overid
cov_type in robust_names or ("robust" in cov_type ...)  # ts_lags
```

于是 `estat hettest` 在 robust 拟合后**不再拒绝**，会打印一个 Stata 明确不给的检验。
最要命的是：**没有数值型测试能发现它**，因为数字只在错误的路径上才会产生，
而对拍 fixture 从不走那条路。

**通用做法**：

1. **凡「修改既有字段的取值域」，必须先全局搜索该字段的字符串比较**，
   而不是只改生产方。`grep -rn 'cov_type.*robust'` 一条命令就能列出全部五个点。
2. **这类谓词只能有一个定义**。本次抽到
   `postestimation/vce_labels.py`，暴露为 `is_robust` / `is_hac` / `is_clustered`；
   物理上列举所有合法标签，不用子串匹配（子串会把未来新增的 `cluster_threeway`
   漏掉或误纳）。
3. **配一个「标签全覆盖」测试**：把所有可能产生的标签逐个喂给谓词，
   再逐条验证守卫仍然触发。本次即 `tests/test_cov_type_labels.py` 的 31 个用例。
4. 顺带修掉的一处：`summary.py` 用同一个字符串判断决定打印
   `Log likelihood` 还是 `Log pseudolikelihood`，且这段逻辑**写了两遍**；
   改标签后表头静默退回 `Log likelihood`，被 `test_weighted_logit_pseudolikelihood`
   抓到。已抽成 `_PSEUDO_LIKELIHOOD_COV_TYPES` 常量。

## 2026-11-13 同一个字符串在不同族里代表不同的统计量：`cov_type="robust"`

用户指出「Stata 不同模型的 robust 会对应 westata 同种模型不同的 cov_type」。
逐族实测，比用户描述的还严重——`robust` 在 westata 里有 **5 种语义**，
其中 3 种公式不同，1 种被静默吞掉：

| 族 | `cov_type="robust"` 实际是 | 结果回显 |
|---|---|---|
| `OLS` | `HC1`（`n/(n-k)`） | `'HC1'` ✅ |
| `IV2SLS` | `HC0`（无 `n/(n-k)`） | `'HC0'` ✅ |
| `Logit`/`Poisson` | **第三种**：`n/(n-1)` 的 ML 三明治 | `'robust'` ❌ |
| `PanelOLS` | **`vce(cluster panelvar)`**（与 HC 无关） | `'robust'` ❌ |
| `QuantileRegression` | `qreg` 专属 `tau(1-tau) J⁻¹(X'X)J⁻¹` | `'robust'` ❌ |
| `RobustRegression` | **静默降级为默认值**，`bse` 逐位不变 | `'nonrobust'` ❌ |

```python
# 最坏的一种：不报错、不回显、数值也不变
a = ws.RobustRegression.from_formula("y~x", df).fit()
b = ws.RobustRegression.from_formula("y~x", df).fit(cov_type="robust")
np.allclose(a.bse, b.bse)     # True —— cov_type 被整个吞掉
```

**为什么会漏掉**：`vce/dispatch.py:48` 有一条全局别名 `"ROBUST": "HC1"`，
它对 7 个族中的 6 个都不成立，只有 `OLS` 成立。各估计族于是各自打补丁：
`iv.py:266` 覆写成 HC0，`panel.py:220` 覆写成 cluster，`discrete.py:154`
自己实现第三种，`hdfe.py:114` 直接拒绝。**没有一个地方能回答「robust 是什么」**。

**通用做法**：

1. **别名必须解析，不能透传**。`robust` 只是用户写的字面量，结果对象里存的必须是
   解析后的规范值（`HC1`/`HC0`/`ML_ROBUST`/`cluster`），否则下游无法判断用了哪种。
2. **同一词汇在跨族接口上必须显式列表**。本仓库现在有
   `scripts/check_api_conventions.py` 里的 `COV_TYPE_ROBUST_TARGETS`（逐族目标值）与
   `COV_TYPE_ROBUST_REFUSED`（拒绝名单），并由
   `test_cov_type_family_table_covers_every_estimator` 强制**每个可估类都要有决定**
   ——新增估计器若不定这一行，测试直接失败。
3. **默认别名要按「最少假设」选，不能按「最常见」选**。取 `HC1` 作全局默认，
   等于把 OLS 的语义强加给 IV 与 ML；正确做法是不设全局默认，
   让每个族自己声明。
4. **在单参数模型上对拍会掩盖公式差异**：`n/(n-1)` 与 `n/(n-k)` 在 k=1 时相等。
   验证 `robust` 语义必须用 **k≥2** 的模型，否则二者「一致」。

## 2026-11-13 接口一致性从来没有被机械校验，于是自由生长了 38 个版本

数值对拍做得再好也管不住接口。本轮把 167 个顶层导出、65 个可估类、149 个公共函数
的签名全量扫了一遍，发现 116 处接口层偏离，其中**会静默算错数**的有 5 处：

```python
# 1. fit() 第一个位置参数不是 cov_type
m = ws.PanelOLS.from_formula("y ~ x", panel, entity="firm", time="year")
m.fit("HC1").cov_type      # -> 'nonrobust'，bse 与 m.fit() 完全一致
# "HC1" 被当成真值赋给 entity_effects，cov_type 收到 None。调用成功，数字错误。

# 2. **kwargs 收下即丢
ws.XtUnitroot.from_formula(...).fit(zzz=[1,2,3])   # 正常返回 PanelUnitRootResult
ws.VECM(...).fit(bogus=1)                          # 正常返回 VECMResults

# 3. level 有两套单位
ws.margins(res, level=0.95)          # 分数
estat_esize(res, level=95.0)         # 百分数，内部算 (100-level)/2
estat_esize(res, level=0.95)         # -> 49.5% 区间，静默错

# 4. 死参数
ws.im_pesaran_shin(panel, ..., level=0.5)   # level 从未被引用，输出与默认完全相同

# 5. __all__ 名不副实
from westata.results import *   # AttributeError: render_estimates_table
```

**通用做法**：凡「签名/默认值/参数单位/导出表」这类**声明性接口**，
必须有一份规范 + 一个能跑的校验器，否则它一定会漂移——数值测试只能发现
「算错了」，发现不了「参数被吃掉但结果看起来正常」。本次产出：

* 规范：`docs/API_CONVENTIONS.md`（R1–R6，每条挂实测证据与行号）
* 校验器：`python scripts/check_api_conventions.py`（`--strict` 供 CI）
* 基线：116 条，103 待修；按「会算错数 → 可用性 → 命名 → 结构」五阶段收敛

教训补两条：

1. **`**kwargs` 是接口的静默腐蚀剂**。允许它出现在公共入口的两个条件缺一不可：
   （a）转发给有真实签名的被调方；（b）未知键会抛 `TypeError`。
   `synth(..., **unsupported)` 属合规（聚合成 `UnsupportedError` 明确报错），
   `VECM.fit(**_ )` 属不合规（什么都不做）。校验器按「函数体是否提及该名字」判定。
2. **参数名 `level` 必须钉死单位**。同一生态里 `0.95` 与 `95.0` 并存时，
   错的那一方不会报错，只会给出 `49.5%` 这种看似合理的区间。

## 2026-11-12 「默认值」类规则：只在一个参数配置下对拍，等于没对拍

`corrgram` / `wntestq` 的默认滞后阶数一直是 `(n-k)//2`，且 docstring 写着「measured
against a live Stata run」。它确实测过——但只测了 icecream.dta 上 **k=4** 的那一个拟合。
而 Stata 的真规则是 `max(1, min(int(n/2)-2, 40))`（`ac.ado:71`、`corrgram.ado:18`），
**与 k 无关**且**有 40 上限**。两条公式只在 k=4 时相等，于是：

* 仓库里所有 fixture 恰好都是 4 参数，全绿；
* 换 1 个参数就偏 1 阶（n=30 时 13 vs 15 之外还会错位），
* 样本到 758 时差到 377 vs 40 —— 差两个数量级。

**通用做法**：凡「默认值/自动选择」类规则，对拍时必须**变化决定默认值的那个自变量**：
这里既要把 k 从 1 试到 6（证明与 k 无关），也要把 n 推过上限（证明 40 存在）。
只在一处相等就宣布「逐值一致」，是把巧合固化成文档。

## 2026-11-12 多命令共用一条算术时，先看 Stata 侧是不是共用同一个 ado

`estat dwatson` / `estat bgodfrey` / `wntestq` / `corrgram` 的滞后全部按行序取，同一处
缺陷在四个命令里各错成四个不同的数（3.286 / 3.286 / 4.3302 / 1.0868969）。
而 Stata 侧它们都走 `_ts ... , sort onepanel` + `tsrevar`/`L.`，缺口语义完全一样。

**通用做法**：一条语义被 N 个命令共享时，抽成单独模块（这里是
`postestimation/ts_lags.py`）并单独测试；在 N 个文件里各抄一份，就等于给同一个 bug
准备 N 个不同的错误答案。同时保留「旧行为」开关（`time=None` 仍按行序），
把「改口径」和「加功能」分开，老调用不会静默变数。

## 2026-11-11 移植包装器命令时：先确认「它自己有没有算法」

`synth2` 看起来像另一个估计器，实际是**纯包装器**：`synth2.ado:157/269/417/516`
四处 `synth` 调用承担全部优化，它自己只做「重跑 + 汇总 + 推断」。
若误以为要重写优化器，会白做大量工作；反之若直接调用底层而**忽略它的行删除语义**，
`e(T0)/e(T1)/e(J)` 会全错。

**通用做法**：移植前先在源码里搜目标命令对其它命令的调用点，确认
「算法在哪一层」；再把包装器独有的语义（这里是 `counit/preperiod/postperiod`
用**删行**实现、`ereturn clear` 清空下游 e()）单独列出并测试。

## 2026-11-11 结果矩阵的行顺序必须按参照实现的排序规则对齐

`synth2` 的 `e(mspe)` 行名是州名（按 `levelsof` 升序）。我第一次用
`df[unit].dropna().unique()` 取单位列表 —— 那是**首次出现顺序**（year-major
面板下完全不同的顺序），导致 39 行数值整体错位，`max|d|` 高达 3.5e3，
看起来像"算错了"，实际是**行对错了**。

**通用做法**：凡输出矩阵带行标签，先确认参照实现的排序来源
（Stata 多用 `levelsof` 升序），再显式排序；对拍时同时比较
`shape` 与**行名**，否则错位会被误诊为数值错误。

## 2026-11-11 「对某预测变量用全局窗口」是常见移植错误

`synth2_balance`（`:574-575`）对带 `(period)` 后缀的预测变量用**它自己的
period 窗口**，不受全局 `xperiod` 限制。我先写成了「先按 xperiod 过滤再取子集」，
`cigsale(1975)` 与 `xperiod(1980-1988)` 无交集 → 整列 NaN。

**通用做法**：解析 `var(period)` 时，每个预测变量保留自己的窗口，
取数从**完整数据**按该窗口过滤；不要先做全局子集再套用。
NaN 出现时优先怀疑「窗口交集为空」，而不是数据缺失。

## 2026-11-11 可行性检查要在归一化之后，不要拦掉可修复的数值误差

上一轮为防「静默返回欠收敛解」，在 `_solve_w` 里加了
`sum(w)-1 > 1e-4` 就抛错。本轮 `synth2` 的空间安慰剂要对 38 个假单位各重跑一次
`synth`，其中一个退化解的残差是 4.09e-4，直接抛错导致整个 placebo 失败。

那 4e-4 是 SLSQP 的收敛残差，`clip + 重新归一化` 就能修好，不是真不可行。
**通用做法**：等式约束残差先修复（clip/归一化）再判定；只在修复后仍
不可行（如 `sum=0`、非有限）时才拒绝。区分「数值误差」与「真错误」，
否则一个合理的守卫会变成脆弱的新故障点。

## 2026-11-11 `True == 1` 会让布尔选项的守卫失效

写「拒绝图形选项」时用了 `if value not in (False, None, 1)`，
结果 `figure=True` 因为 `True == 1` 被当成默认值放过 —— 测试立刻抓到
（DID NOT RAISE）。**通用做法**：布尔标志用 `is True` / `is False` 判断，
不要与 `1`/`0` 混在同一 `in (...)` 里；数值选项才用值比较。

## 2026-11-11 「堆叠布局」要先看源码是追加行还是追加列，别凭词猜

`keep()` 之前被实现成 69 行（J=38 + T=31）的**纵向拼接**：前 38 行只有
权重、后 31 行只有结果。活体 Stata 实测是 **38 行 = max(J, T) 的并排布局**：
`svmat`（`synth.ado:620-650`）是往同一个数据集**追加列**，权重块与结果块
都从第 0 行开始写，短的那块在末尾留缺失；`:650` 只删「两列都缺」的行。
「stacked」这个词在 Stata 文档里指「不同来源的列叠放在同一批行上」，
不是 Python 里 `pd.concat(axis=0)` 的意思。

**通用做法**：涉及导出文件布局时，用活体参照**逐格核对**
（shape、每列非缺失数、首行、末行、dtype），不要靠名字推断。
本次还纠正了上一轮一个错误结论：`e(Y_synthetic)[0]` 与
`keep _Y_synthetic[0]` 在本次捕获里**完全相同**（133.07500119018556），
之前写的「keep 用舍入前 W」只在另一次捕获成立，不能当普遍规律。

**第二个坑**：`_Co_Number` 若写成字符串列，`pandas.to_stata` 会把末尾的
NaN 变成空字符串，导致「非缺失行数」全错。Stata 存的是**带值标签的数值列**，
正解是写数值编码 + `to_stata(value_labels={...})`，两者都要。

## 2026-11-11 「接受参数」不等于「参数生效」，不生效的必须显式标注

`sigf`/`margin`/`bound` 会改变 Stata 的 RMSPE（实测 `sigf(7)` −2.1e-3、
`margin(0.05)` +4.6e-8、`bound(1)` +5.3e-8），`technique()` 切换 Stata
闭源 `ml` 优化器。我们的 SLSQP/Powell 没有对应机制。

试过 `ftol` 扫描想复刻 `sigf`：RMSPE 响应**非单调且混乱**
（1e-14→25.2、0.1→18.8、1.0→20.5）；单起点 Powell 扫描也从不落在
Stata `nested` 的 6.5268。这类关系无法用参数映射表达，硬做就是凑数。

**通用做法**：参数被接受时，必须在结果对象里区分
「已记录」与「数值上生效」。本轮加 `SynthResult.numerically_inactive`，
把不活跃项显式列出，并在文档/CHANGELOG 写明原因与实测依据。
**绝不静默接受一个看起来生效、其实不改变任何数字的参数**——那会让
用户以为自己调过模型。

顺带修了一个真 bug：`maxiter` 曾被直接当作 SLSQP 的迭代上限，
`maxiter=10` 会截断求解、静默返回欠收敛的 W（RMSPE 16.5 而非 19.7）。
Stata 的 `maxiter(10)` 与 `maxiter(1000)` 结果完全相同，说明它是插件的
迭代上限且插件很快收敛。**小上限绝不能让求解器欠收敛还照常返回**，
要么给安全下限，要么检测不可行并抛错。

## 2026-11-11 「两个解都对但不一样」时，要复刻求解器轨迹，不是最优值

`synth` 的两个退化案例里，westata 的 `W` 与 Stata 差 0.176/0.081，但
**QP 目标值差只有 4.6e-6/8e-7**——两边都在同一个最优面上。原因是
`H` 近秩 1（`Vmax>0.98`），最优面维度极高；Stata 的 Vanderbei/LOQO
**内点法**返回早停迭代点（一个主权重 + 铺满 1e-3 微小权重），scipy
SLSQP 收敛到稀疏顶点。

**通用做法**：
1. 先确认「是同一个最优解吗」——比较目标值与 KKT 残差，不要只看参数差。
   若目标值一致而参数差大，问题在**选点**而非求最优；
2. 找参照实现用的求解器族（这里是 `synth.sthlp:264-266` 明写内点法），
   复刻它的**轨迹**（log-barrier Newton + 对偶间隙早停），而不是给目标函数
   加正则去凑（对角正则 / 约束松弛 / 投影梯度早停全部试过，均失败）；
3. 停止判据用**有原理的量**（对偶间隙、由公开参数 `margin` 派生），并在
   至少两个数量级上验证结果稳定——否则就是「猜系数凑数」；
4. 用**数据自身的判据**（这里是 `Vmax`）在双求解器间自适应切换，并验证
   切换不伤害原本精确的案例；把所选求解器记进结果对象供审计。

本次结果：dW 0.176/0.081 → 0.009/0.003，dRMSPE 0.474/0.336 →
0.0096/0.0688，6 个原本逐位一致的案例**全部保持不变**。

**防过拟合复核**（必做）：用 `customV` 在构造面板上把 `Vmax` 从 0.900 扫到
0.995、跨过 0.98 阈值，检查输出是否连续。结果：RMSPE 差 ~1e-6、`W` 形状与
列和不变，切换点没有突变；且对偶间隙阈值在 1e-6..1e-2 两个数量级内结果
一致。若阈值一跨就产生数值跳变，说明它是对着两三个案例凑出来的，应放弃。

## 2026-11-10 `synth` 扩展轮次：三条契约级教训

**1. 长面板的单位顺序必须与 Stata `levelsof` 对齐，不能信 `unique()`。**
`e(W_weights)` 的行序是 `levelsof pvar` 的**升序**（`synth.ado:53`，
`counit` 默认为该列表减去 treated，`:62-64`）。长面板通常按「年×单位」
存储，`df[unit].unique()` 是**首次出现顺序**，与升序完全不同。实测中
V 差 ~1e-13（X 完全正确）却报 W 差 0.4~0.8，一度误判为 QP 求解器不一致；
真相是**同一个 W 被换了行序**（非零权重 0.437/0.423/0.139 逐位对得上，只是
排在不同行）。修法：`all_units` 与 `control_units` 都显式 `sorted`。
**教训：矩阵数值全对但「差很大」时，先查行/列对齐，再查算法。**

**2. 新增 fixture 必须满足仓库既有的形状守卫，否则一次带出 3 个失败。**
`test_stata_e3.py::test_fixture_files_are_valid_json` 与
`test_fixture_verification_scope.py` 都 glob `*.json`，要求顶层有
`cases`/`ado`/`stata_version`（或 `scalars["N"]`）、`dataset` 非空，且
**`cases` 的每个 key 含 `|`**（历史 `model|variant` 约定）。新捕获的
`synth_features.json` 把元数据放在 `_meta`、case key 用 `var_single_1975`，
于是 3 个守卫同时变红，看起来像大面积回归。修法：补顶层元数据键，
case key 统一前缀 `synth|`，读取端 `split("|",1)[-1]` 还原。

**3. 「两次相同运行差 X」这类抖动断言必须写清数据/设定，否则会被反例证伪。**
P0 在 `cigsale lnincome age15to24 beer retprice` 上测到 RMSPE 抖动
1.471e-11，被写进 registry/design/USING_WITH_STATA/CHANGELOG 当作
「E3 排除」的全局依据。扩展轮次对 `cigsale lnincome beer retprice`
连跑两次，差 **0.0**；改变 `sigf(7)` 才让 RMSPE 变 2.1e-3，而
`roundmat()` 让 `e(W_weights)` 在所有调参下不变。**默认路径其实是确定性的**，
E3 排除的真正原因只剩「`H` 秩亏 ⇒ QP 最优 W 不唯一」。
**教训：把「某次测量」升级为「全局性质」前，必须换数据复核；已写进多处
的断言更正时要逐处同步，不能只改一处。**

## 2026-11-10 发布前必须区分「本地提交」与「外部推送」，并先探测凭据/连通性

发布 `westata 0.3.6` 时，工作区有并发会话的未提交改动（`prais.py` / `summary.py`
半成品）。若直接在工作区 `python -m build`，会把半成品打进 wheel。

**通用做法**：
1. 用 `git archive HEAD` 导出**干净提交树**到临时目录（PowerShell 下
   `git archive | tar` 会把字节流当文本处理而报 `Damaged tar archive`，改用
   Python `tarfile.open(fileobj=io.BytesIO(subprocess.check_output(...)))`）；
2. 在导出树里构建，并**校验 wheel 内容**：版本号、`synth.py` 存在、未提交标记
   不存在、sdist 含中英文 README/CHANGELOG/LICENSE；`twine check` 两个产物；
3. 推送前先探测外部条件：`git ls-remote`（GitHub）、`Invoke-RestMethod
   https://pypi.org/pypi/<pkg>/json`（已发布版本）、`keyring` / `.pypirc` /
   `TWINE_*` / `PYPI_API_TOKEN`（凭据）。**探测失败就如实报告，不要重试到死，
   更不要把「未推送」写成「已发布」**；
4. 交付说明必须区分「本地已提交」「已构建并校验」「外部推送受阻及原因」。

**本环境事实**：GitHub 的 `github.com:443` TCP/DNS 可达，但 TLS/HTTP 层连接被
重置（`Connection was reset` / `Failed to connect ... port 443`），3 次均失败；
PyPI 上传缺 API token（`Credential not found for API token`，keyring 亦无条目）。
`git-credentials` 里的 GitHub 凭据**不能**解决传输层阻断。`dist/` 已在
`.gitignore` 中，产物不会被提交。

## 2026-11-10 生成器与已提交输出漂移：重跑生成器会「吃掉」别人手工加的章节

`docs/generate_guide.py` / `gen_guide_zh.py` 的模板**落后于**已提交的
`docs/GUIDE.md` / `GUIDE.zh-CN.md`：并发会话在 0.3.3–0.3.5 几次提交里把
`estat bgodfrey nomiss0` 与 `prais corc` 的章节**直接加进输出文件**，没有同步进
生成器模板。本轮回填 `synth` 行时我重跑了两个生成器，结果一次删掉 **303 行**
（`git diff HEAD --stat` 显示 26 insertions / 303 deletions），差点把别人的
工作覆盖掉。

**通用做法**：
1. 重跑任何生成器前，先 `git diff HEAD --stat -- <输出文件>` 看模板与输出是否已漂移；
2. 若生成器落后于输出，**不要盲目重跑**；先把输出文件里手工追加的章节补进模板，
   或改为只在输出文件上做最小增补；
3. 发现已误删，用 `git checkout HEAD -- <输出文件>` 立即恢复，再重新做最小改动；
4. 「生成器能跑通」不等于「输出无损」——`test_guide.py` / `test_docs_zh.py`
   只检查代码块与表格行数，**不检查章节是否齐全**，所以 303 行被删时 11 个测试
   依然全绿。这正是「测试绿 ≠ 内容无损」的典型案例，与本项目
   `GAP_REGISTER.md §8.4` 的教训同源。

本次处置：恢复 HEAD 的两个 guide 输出文件（保留并发会话的 prais/bgodfrey 章节），
只在中英文 guide 输出文件与两个生成器模板里各补一行 `synth`；生成器仍缺那两个
章节的事实已在此登记，**后续若要重跑生成器，必须先把章节补进模板**。

## 2026-11-10 `__version__` 读的是「已安装元数据」，不是 pyproject

`westata.__version__` 用 `importlib.metadata.version("westata")` 读取**已安装**
分发的版本（刻意避免硬编码漂移，见 `src/westata/__init__.py:172-187`）。因此当
`pyproject.toml` 被改到 0.3.5、而 `.venv` 里的 `.dist-info` 还是 0.3.4 时，
`test_version_consistency.py::test_runtime_version_matches_pyproject` 必然报
`'0.3.4' != '0.3.5'`——**这不是代码 bug，是环境元数据过期**。

**处置**：`.venv/Scripts/python.exe -m pip install -e . --no-deps --no-build-isolation`
刷新 editable 安装的元数据（纯本地、无需联网），再跑该测试即可。
不要去改 `__init__.py` 硬编码版本号，也不要改测试去迎合旧值——那会重新引入
它本来要消除的漂移。并发会话改了 `pyproject.toml` 后尤其容易踩到，因为另一边
的安装动作不会自动同步到本会话的 `.venv`。

## 2026-11-09 `synth` 可行性评估：四类可复用教训

### 0. 「源码不可得」≠「结果不可验证」；先分清哪一层拿不到

一个编译过的 Mata `.mlib` 里没有源码（实测 3.43% 可打印、零 Mata 语法关键字），
这只能推出「**公式**不可读」，**推不出「输出不可验证」**。本次实测发现：
`synth` 把 6 个 `e()` 矩阵全精度存着（`e(V_matrix)` 等）、`regsval` 是收 4 个显式
字符串参数的**确定性纯函数**（单独调用 `RC=0`、`vmat − e(V_matrix) = 全零`、
`drop _all` 后仍可调用）。⇒ 黑箱辨识可行，不需要反编译。

**通用做法**：遇到二进制内核，按三层分别问「① 算法/目标函数能从可读外壳拿到吗？
② 黑箱的**输入/输出**能从活体 oracle 采集吗？③ 只有**公式/源码**拿不到吗？」
只有第 ③ 层是真边界。把三层混成一句「信息不可得」会误杀可交付的工作。
本项目 `GAP_REGISTER.md §8.4` 早已把这类错误定性为最坏失败模式，本次是第三例
（前两例：`bgodfrey` 实为未查用户 ado 路径、`dfgls` 实为默认选项写反）。

### 0.1 断言「某 `e()` 不存在」前，用 `ereturn list` 实测，别靠记忆/拼写

`synth` 存的是 **`e(V_matrix)`**，不是 `e(V)`；`e(W_weights)` 无条件 `ereturn mat`，
不依赖 `keep()`。旧记录写「不保存 `e(V)`/`e(V_matrix)`」把两种拼写混在一起，
于是「导出 `e(V)`」被当成死路封了 100+ 轮。

**通用做法**：`ereturn list` 列全名 → `matrix list <名字>, format(%21.17f)` 取全精度；
名字逐字抄，不要按惯例补全/改写。写捕获器时用错名字会**静默拿到空矩阵**，不报错。

### 0.2 复现「函数调用失败」的实验前，先读调用点的参数与时序

第 70/71 轮把 `synth.ado:467-474` 的全局量设好去调 `regsval`，得到 0×0 空矩阵，
据此断言「依赖更深层 Mata 内部状态」。实际那些全局量定义在 `regsval` 调用（`:390`）
**之后**，只服务 `synth_ll`/`ml`；`regsval` 收的是 4 个显式字符串矩阵名。
实验测的根本不是目标函数。

**通用做法**：黑箱调用失败时，先回答「它的真实参数是什么、在哪一行被调用、
我设的状态是否在调用点之前生效」，再下「不可复现」的结论。

### 0.3 文档/帮助文件的默认值可能比 ado 源码旧，以源码为唯一真相

`synth.sthlp:270/274` 写 `margin=0.05, sigf=7`，而 2026 版 `synth.ado:35/:37` 是
`margin=0.005, sigf=12`（帮助头 `26jan2014`，ado 头 `04/29/2026`）。照帮助移植必错。
对应 `AGENTS.md §6.2`：源码是契约唯一真相。

### 0.4 区分「还原算法」与「逐位复现优化器」，别用错工具

REA 对 Stata `.mlib` **零支持**（`targetFormatSchema` 23 种格式无 mata/mlib，包内
1598 文件零命中），对 PE 插件只能给伪代码且需自带 Ghidra/IDA（本机 4/4 引擎全缺）。
但「算法」早已由 931 行可读 ado + 帮助给出；「逐位」则连公开重实现（`pysyncon`
自述权重漂到第 3 位小数）都做不到。

**通用做法**：选型前先明确目标是「知道它在算什么」还是「复现它的浮点停止点」。
前者常由可读源码/论文解决，后者即使有反编译器也不可达，应改为**声明容差**。

### 0.5 `.dta` 是 float32 且 `state` 是 category：读错类型会凭空造出 3e-7 的「数值不一致」

`synth_smoking.dta` 里 `cigsale`/`lnincome`/`beer`/... 全是 **float32**（Stata `float`），
`state` 是**带值标签的 int32 category**。用 `pd.read_stata(path)` 默认
`convert_categoricals=True` 会把 `state` 变成州名**字符串**，于是 `trunit(3)` 匹配不到
任何行，`X` 全 NaN。必须 `convert_categoricals=False` 拿到底层整数编码。

即使列选对了，若不做 `astype(float64)`，逐单位 `mean()` 的 float32 累加会让
`xtrmat` 偏 4.6e-5、V 偏 3e-7 —— 看起来像「公式差一点」，实际只是精度。
显式 cast64 后同一公式与 Stata 差 **2.8e-14 / 7.8e-14**（机器精度）。

**通用做法**：与 Stata 对拍数值前，先 `df.dtypes` 看存储类型；把参与计算的所有
数值列显式 cast 成 float64；区分「输入精度漂移」与「算法本身的不可约差异」，
否则会把一个可修复的精度问题登记成 E2 差距。本项目 `imtest` 的 float/double
差异是同一类的前例。

### 0.6 同一个 Stata 命令跑两次结果就不同时，不能声称逐位一致

`synth` 的 C++ QP 插件 `synthopt.plugin` **不是位确定性**的：两次完全相同的
`synth cigsale …` 运行的 `e(RMSPE)` 分别为 `10.500868354649425` 与
`10.500868354634719`，差 **1.471e-11**；另一次独立实测为 `10.500868354656104`。
这不是 Python 实现误差，是 Stata 自身的抖动。扰动 V 的实验表明：V 偏差 ≤1e-8
时 `e(W_weights)`（被 `roundmat` 舍入到 3 位）**完全不变**，RMSPE 变化低于该抖动。

**通用做法**：定容差前先测「oracle 自己重复一次差多少」。若参照源本身有抖动，
容差下界不能小于它，且**必须放弃 E3/逐位一致的措辞**——否则是在验证一个
不存在的确定性。本项目 `bootstrap` 的「构造性 E2」是同类的前例。

### 0.7 平台拒绝「删除」时，别为删一个临时文件申请全会话提权

`tests/` 有独立的（非继承）DACL，DSH 授予工作区根目录的 capability SID 不向下
传播；confined token 是 Low integrity、该链上无 `WRITE_DAC`/`WRITE_OWNER`。
后果：`read`/`write` 正常（host 文件后端可绕过），但 **`Remove-Item` 一律
「访问被拒绝」**，连 0 字节临时文件也删不掉。这是**预期的沙箱限制**，不是
ACL 故障 —— `diagnose-windows-sandbox-acl` 技能明确说此类无需 ACL 修复。

一个 0 字节的 `_host_probe.json` 会让所有 glob `*.json` 的 fixture 守卫崩
（`json.loads("")` → "Expecting value"），一次能带出 **5 个失败**，看起来像
大面积回归，实际是同一根因。

**处置（诚实降级，不将错就错）**：
1. 先用隔离实验确认「排除该文件后其余测试全绿」，别凭猜改守卫；
2. 用 host `write` 把它写成**自解释、明确标注「NOT a fixture / no evidence
   value」**的合法 JSON，让 glob 测试通过；
3. 在 `.gitignore` 里精确排除该文件路径，确保永不被提交；
4. 在捕获脚本里**不要**再产生它；
5. 等到会话权限放宽（后来用户切到 full access）后**立即删除**该文件——降级只是
   临时措施，不是永久状态。
不要为了删一个垃圾文件把整个会话切到「完全权限」——收益/风险比不成立，
也不要把探针伪装成真 fixture 去骗守卫（内容已自证身份）。
**2026-11-10 收尾**：权限放宽后该文件已删除，`.gitignore` 的排除规则保留为
防御（防止将来同类探针再被提交）。

### 0.8 `git add` 也可能被沙箱挡在 `.git/index.lock`，与工作区写入是两回事

文件 `read`/`write` 正常、`pytest` 全绿，但 `git add` 报
`fatal: Unable to create '.git/index.lock': Permission denied`。原因是 `.git/`
与 `tests/` 一样有独立受保护路径，confined token 无写权限；这不是代码问题，
也不留残留 `index.lock`。

**处置**：不为一次本地提交申请全会话「完全权限」。确认无残留 lock、HEAD 未变、
工作区文件齐全后，把精确的 `git add` / `git commit` 命令交给用户在合适权限下执行，
并如实说明「代码已落盘、测试已通过、提交未执行」。**不要把「已提交」写进交付说明**。

### 0.9 纠正既有结论时，「保留原文 + 追加修订」比「改写」安全

仓库里 `test_register_p1_recheck.py` 等守卫直接断言文档中的特定字符串
（「仍缺」「反编译」「源码根本不存在于安装目录中」等）。本次纠偏**只增补、不删除**
原措辞，追加「以此修订为准」的块，21 个相关守卫测试保持全绿。
若直接改写历史段落，会同时破坏可追溯性与测试。

---

## 2026-11-07 本地 Stata 安装教材必装程序（`lpdensity` / `rddensity` 等）

### 1. Stata 的 `net install ..., from(本地目录)` 要求目录里有 `stata.toc`
- 只把 SSC 上的 `.pkg` 和文件下载到本地目录，直接 `net install` 会报
  `file ...\stata.toc not found` → **r(601)**。
- 该目录必须是"Stata 下载目录"：至少要有一个 `stata.toc`，内容为
  `p <包名> <描述>`。手工离线安装时记得一并生成。
- 来源：实测 `net install lpdensity, from(D:/.../lpdensity)` 的真实报错。

### 2. `.pkg` 里小写 `f` 行不会安装辅助 `.do` / `.dta`，要用大写 `F` 行
- `net install` 按 `.pkg` 的文件清单安装；小写 `f` 只把 `.ado`/`.sthlp`/`.mo`
  等装进 PLUS 的字母子目录，**`.do`、`*.dta` 之类的辅助文件会被跳过**。
- 后果：`rddensity.ado` 内部用 `findfile rddensity_fun.do` 动态调用，缺该文件时
  命令能 `which` 到（r(0)）但一运行就报 `file "rddensity_fun.do" not found` → **r(601)**。
- 正解：把辅助文件对应的 `f xxx.do` / `f xxx.dta` 改成 `F xxx.do` / `F xxx.dta`
  （大写 F = 复制到 ancillary/当前目录），或装完手工补齐；教材强调"把 Stata 文件夹
  内**全部文件**放进 ado\plus"正是此因。
- 相关：`net get <pkg>, from(目录)` 会把辅助文件拉到当前工作目录（实测可行），
  但更省事的是 `F` 行让 `net install` 一次装全。

### 3. 光 `which` 成功（r(0)）不等于命令能跑，必须做真实运行验证
- `which rddensity` = 0 但 `rddensity ...` = 111/601，是上面第 2 条所致。
- 验收要分三层：`which`（文件在不在）→ 真实跑一个最小例子（依赖/Mata/辅助文件
  是否齐）→ 检查 `e()`/输出，不能把"已安装"写成"已验证可运行"。

### 4. 冒烟测试报 r(111) 先确认数据集变量名，别误判成命令坏
- `rddensity_senate.dta` 的变量是 `margin`，不是 `votesfor`（那是别的数据集）。
  写错变量名得到 `variable votesfor not found` → r(111)，与安装无关。
- 冒烟测试前先 `ds` 看变量名；`webuse` 联网取数失败会 r(603) 并连累后续命令
  （如 `xtoverid`），改用本地 `.dta` 即可。

### 5. Stata 安装写的是工作区外的 PLUS 目录，沙箱下会 r(603)
- `ssc install` / `net install` 默认写 `sysdir` 的 `PLUS`
  （本机 = `C:\Users\zjl\ado\plus\`），在 workspace-write 沙箱下必然
  `cannot write in directory ...` → r(603) / "访问被拒绝"。
- 这是**预期限制，不是文件权限/ACL 故障**，不要用 ACL 修复脚本去"修"。
- 稳妥流程：先把包下载/暂存到工作区（离线、无需授权）→ 在工作区内的临时 PLUS
  用 `sysdir set PLUS` 做**端到端演练**，把机制、辅助文件、运行全部验证通过 →
  最后只对"写入真实 PLUS"这一步申请一次临时完全权限，一次成功、可复现。

### 6. 区分安装来源与版本新旧
- GitHub 在本环境被墙（连接被关闭），SSC 可达；教材建议的 `findit` 旧版会报错，
  安装后应读 `.ado` 头部 `*! version` 与 `.pkg` 的 `Distribution-Date` 核对版本，
  别把旧版当成装好了。

---

## 第二部分：输出对齐与数值核对（第 12–19 轮）

### 7. 比对 Stata 输出前必须先排除迭代日志

`tobit` 曾被判定为「表格位置错位」（26 行 vs 19 行），实际是 Stata 默认输出 **7 行迭代日志**
在表头之前：`Refining starting values:` / `Grid node 0:` / `Fitting full model:`。

`nbreg` / `xtreg, mle` 打印 `Iteration n: Log likelihood = ...`。

- **正解**：捕获时加 `nolog`，或比对前过滤上述开头的行。
- **后果**：日志行数被误算成结构差异，进而去修一个根本不存在的缺陷。

### 8. 手抄的参考输出必然出错

多次验证：手工转写的 Stata 参考行，空格数几乎总是错的。**参考值必须来自实时捕获**，
写完立刻 `print(repr(line))` 回读 —— repr 看得见末尾空格，print 看不见。

### 9. 宽度断言通过 ≠ 对齐正确

`var(e.mpg)` 行的 pipe 在第 12 列而非 Stata 的第 13 列，但**整行都是 78 字符**，
所以每个 `len(line) == 78` 断言都通过。只有**逐字符**比对才能发现同一总宽内的偏移。

- **教训**：布局测试至少保留一处逐字符断言，不能只有宽度断言。

### 10. 陈旧 `__pycache__` 会让修复看起来无效

源码已改为 `:.6g`，输出仍是 `20.086`。测量前先清：

```powershell
Get-ChildItem -Recurse -Directory -Filter __pycache__ src\westata | Remove-Item -Recurse -Force
```

### 11. 测量脚本自身的 bug 会伪装成被测代码的缺陷

`nbreg` / `logit` / `probit` / `poisson` 一度测出 0.11–0.31 的行匹配率，**真实值是 1.0000**。
原因：评分脚本里写死了 `ws.OLS.from_formula`，拿 Stata 的 logit 去比本包的 OLS。

- **教训**：行匹配率异常低（< 0.35）时，先怀疑测量脚本。验证：直接 print 被测对象的
  `model_name` 与输出首行。

### 12. 优化器停在局部极值：多起点 + 合理阈值

`nbreg` 单次 BFGS 停在 α=1.05（ll = −302），Stata 到 α=1e-9（ll = −196）。
似然函数本身没错 —— 固定系数时 α=1e-9 给出 −196.129752。

两个必要约束：

1. **起点下限 1e-9**：更小时 gamma 项灾难性抵消，`1e-12` 给出**虚假更高**的 −195.96，
   `1e-14` 崩到 −206.9；
2. **改进阈值 ≥ 1e-3**：边界附近似然是平的，另一份数据上 `1e-9` 起点只多 5.3e-4 纯噪声，
   接受它会把 α 推到边界、信息矩阵奇异、SE 变 NaN。真正修复的增益是 **60 个单位**。

### 13. 边界解不能报标准误

离散度到边界（α→0）时观测信息**秩亏**（本包实测协方差有零行，最小特征值 7e-11），
Stata 把 SE 与区间印成 `.`。本包的解析信息矩阵仍会给出有限值（se = 12.69）和跨度
`[-38.7, 11.1]` 的区间 —— **数据支持不了**。

- **判据**：`exp(lnalpha) < 1e-5` 时抑制。须同时验证**内部离散度仍保留 SE**。

### 14. 同一族命令的「哪个似然」必须逐一确认

`nbreg` 表尾的 `LR test of alpha=0` 比的是 **Poisson 拟合**
（2*(ll_nb − ll_pois) = −0.0008 → Stata 印 `0.00`），不是常数模型
（那会给出 69.88，是**模型检验被贴错标签**）。而 `xtpoisson`/`tnbreg` 确实用 `e(ll_0)`。

**同一行文字，不同命令用不同对比基准。**

### 15. `e(df_m)` 扣除 `k_aux`

`nbreg` 有两个斜率时报 `e(df_m) = 1`，因为 `e(k_aux) = 1` 被扣掉，表头才写 `LR chi2(1)`。
`_slope_positions` 只排除 `/` 开头与常数项，**扣 k_aux 是 Stata 额外的一步**。

### 16. 用 write 覆盖前必须确认文件是否已存在

一次 `write` 覆盖了 `rdbandwidth.py` —— 该文件早已存在（提交 `be80a86`），含前几轮
**细致转写**的常量与算法。被 `test_rdbandwidth_helpers.py` 的导入错误拦下，已完整恢复。

- **教训**：新建文件前先 `Test-Path` 或 `git log -- <path>`。

### 17. 死代码会伪装成权威

`_render_r2_table` 里有两份连续的 `lines = [...]`，前一份被后一份立即覆盖，且两者相差
**一个前导空格** —— 正是读者会以为「有讲究」的那种差异。已删除。

- **教训**：看到「几乎相同」的重复赋值，先确认哪一份真正生效。

### 18. 新增方法前，先在**整个包内**搜索同名符号（这条我犯了两次）

| 轮次 | 经过 |
|---|---|
| 第 2 轮 | 用 `write` 覆盖了已存在的 `rdbandwidth.py`，丢失前几轮的转写成果，靠导入错误拦下后从 git 恢复 |
| 第 25 轮 | 用 `edit` 给 `VARResults` 加了 `irf_table()` —— 而 `tseries.py:1216` **早已有**同名方法，新代码覆盖了它 |

两次都是「以为缺失，其实已有」。第二次尤其隐蔽：我搜过 `irf`，但搜的是 `ws.VAR` 的方法列表，
没有搜 `tseries.py` 内部；而真正的实现在 `VARResults` 上，不在 `VAR` 上。

**强制流程**（写方法/函数前）：

```powershell
# 1. 搜整个包，不只是我以为的那个类
Select-String -Path (Get-ChildItem -Recurse -Filter *.py src\westata | Select-Object -ExpandProperty FullName) -Pattern "def irf_table"

# 2. 确认现有符号归属哪个类
# 3. 若已存在，读完整实现再决定是改它还是加新的
```

**判据**：如果同名的 `def` 已经存在，先读它——它很可能已经正确处理了问题，
而「报告说缺失」只是没有找到入口。

### 19. 报告的「功能缺失」可能只是默认参数

C13-06 被判为「`irf table oirf` 只给 1 列」。实际 `irf_table()` **存在且正确**，
只是 `impulse`/`response` 默认取第一个变量（文档写明）。

- **教训**：核实「缺失」前，先直调用一次看返回什么，再读它的默认值。

### 20. 诊断时用错了对比基准，差点去「修」一个正确的实现

排查 `teffects aipw` 的点估计差异时，第一版诊断把本包的倾向得分与**最小二乘回归**的预测值对比：

| | p 范围 |
|---|---|
| 本包 | `[0.195, 0.676]` |
| 我用的 lstsq | `[0.543, 0.663]` |

看起来像严重缺陷，实际上**线性概率回归不是倾向得分模型**。改用直接对 logistic 似然做 BFGS
优化后，本包的系数与它匹配到 `4.2e-05`——**代码是对的，诊断是错的**。

- **教训**：拿不准基线时，先用**独立实现**算一遍（这里是 `scipy.optimize`），
  而不是自己拼一个"差不多"的近似。差的那个往往是你自己的近似。

### 21. 「最显然的修法」可能就是错的

`aipw` 的修正项均值本应为 0，实测是 `-6.58`。最直接的想法是**给结果方程加权**
（权重 `t/p`）：这样第一项**恰好变成 0.0**，正交性成立。

但点估计变成 `1617.18`，**离 Stata 的 `1621.583` 更远了**。

- **教训**：修改前先验证「修完之后另一个量是否还正确」。满足一个理论条件
  （这里是正交性），可能破坏另一个（这里是拟合均值本身）。

### 22. 区分「同名的多个实现/估计量」

`teffects ra` 有三种状态并存：

| 接口 | 状态 |
|---|---|
| `RegressionAdjustment` 类 | **精确**（ATE 与 SE 都对到打印位）|
| `teffects_ra()` 便捷函数 | 打印 `(variance not reproduced)` |
| `teffects aipw` | 点估计低 2.53，未解决 |

模块 docstring 用一句话概括了「robust 标准误未复现」，掩盖了**类版本已经精确**这个事实。

- **教训**：注解里的评估结论要**指名道姓**到具体接口，否则会低估已有的覆盖。
### 23. `ereturn list` 的输出是截断的，不能用来判断键是否存在

第 49 轮用 `ereturn list` 的文本搜索判断 `xtreg, fe` 是否存某个键，
得出「`tss_within` 不存在」等结论——**方向对，但依据不可靠**：
`ereturn list` 只列出部分内容，搜不到不等于不存在。

**可靠做法**——`display e(key)`：

```
capture noisily display e(key)
```

缺失的标量打印 `.`，存在则打印数值。判据是「输出恰为单个 `.`」。

- **踩过的坑**：第一次写判定用 `first.startswith(".")`，把 `r2_c` 的 `.99238707`
  也判成缺失。必须用 `first == "."`。

- **教训**：探测型断言要选一个**值域互斥**的信号。数值与缺失点的区别在
  「是否恰为 `.`」，不在「是否以 `.` 开头」。
### 24. 「换个后端」不是局部改动：先数清消费方再动手

第 61–62 轮尝试给 `FirstDifferenceOLS` 补上 Stata 会保留的截距。算术早已验证逐位一致
（4 个系数 + 4 个标准误，`n=768`、`df=764`），但**两次实现都在测试里翻车**：

- 第 61 轮：事后补截距 → 斜率不变，得到不属于任何模型的参数向量；
- 第 62 轮：整体重拟合 → **10 个测试失败 + 4 个 collection error**。

失败清单说明了真实依赖：

```
test_panel_e_identities.py::test_fd_r2_within_and_overall_are_a_recorded_gap
test_panel_e_surface.py::test_be_and_fd_store_no_sigma_family[FirstDifferenceOLS]
test_predict_registry.py::test_every_registered_type_runs[xtreg, fd]
test_predict_registry.py::test_xtreg_fd_design_matches_its_coefficient_count
```

即 `r2_w`/`r2_o`、`e()` 表面、`predict` 注册表、设计矩阵与系数个数的对应关系，
**全部依赖「没有截距」这个前提**。

- **教训**：把「需要改的地方」列成清单（第 60 轮做了），不等于清单完整。
  先跑一次**最小改动**看测试如何失败，失败清单就是最可靠的依赖图。
- 本次两次尝试后工作区均干净回退，`fd` 相关 22 个测试保持通过。
- 结论已写入 `options.md` T10：这不是一轮能完成的改动。
### 25. 缺公式时，数值试错会收敛到「某个能对上几个数」的错误实现

第 64–65 轮实现 `rdrobust` 的 NN 方差。目标是 `se = 0.1099186153`，我试了五种公式：

| 尝试 | 结果 | 偏差 |
|---|---|---|
| sandwich + `Σw·e²/Σw` | `0.1113851870` | +1.33% |
| sandwich + `mean(e²)` | `0.1142912646` | +3.98% |
| sandwich + `Σw·e²/n_h` | `0.0755949066` | −31% |
| 同侧最近邻匹配 | `0.3929069802` | +257% |
| 对侧最近邻匹配 | `0.3373928805` | +207% |

**最接近的那个只差 1.33%** —— 如果样本换一组，它可能恰好落在 0.3% 以内，
于是被当成"实现正确"提交。这就是这类试错的危险：**没有一个能在数学上判定对错，
只能靠数值逼近，而数值逼近在别的数据上不一定成立**。

- **教训**：缺公式时，**先判断自己是在推导还是在调参**。
  只能调参的时候，正确做法是停止并记录边界（保留 `NaN`），
  而不是挑一个看起来最接近的写进去。
- 本轮最终**未改任何实现代码**，两次原型均删除，`options.md` 记录了两版的具体数值与偏差，
  下一步明确要求「先取得论文 4.2 节公式」。

### 26. 每轮都跑全量测试：违反工作区规则，且浪费大量时间

工作区规则明确写着「**每 20 轮代码修改，就要进行一次全量测试**」。
但我在第 1–85 轮的多数轮次里**每轮都跑全量**，单次耗时 9–25 分钟。

**实际代价**：约 60 轮 x 平均 12 分钟，约 **12 小时**的纯等待，
而这些时间本可以用 5–30 秒的针对性测试替代。

**正确做法**：

```powershell
# 平时：只跑受影响的模块（秒级）
.venv/Scripts/python.exe -m pytest tests/ -q -k "vec or vecm"

# 每 20 轮，或改动涉及多个共享渲染器时：才跑全量
.venv/Scripts/python.exe -m pytest tests/ -q
```

**判据**：改动只触及单个文件的渲染逻辑时用针对性测试；
改动触及 `results/base.py`、`results/summary.py` 等共享文件时扩大范围。

- **教训**：把「谨慎」等同于「每次都跑最贵的检查」，是一种伪装成勤奋的浪费。

### 27. 几何对不上两次，就读 `.ado` 源码——不要第三次反推

**实际发生**（第 187–188 轮）：`dfuller` 的表头在 `lags>0` 时与 `lags=0` 结构完全不同。
我从输出反推列规则，**连续试了 6 次实现**：

| 尝试 | 结果 |
|---|---|
| 固定左栏 42 | n=1500 时崩 |
| 固定左栏 40 | n=200 时崩 |
| 整行 `rjust(62)` | 左栏被右移 |
| 左栏 ljust + 尾部 rjust | 差 1 格 |
| 按 `Number of lags` 锚定 42 | n=900/1500 崩 |
| `%11.0gc` + 右对齐 | 差 1 格 |

**最后读了 `dfuller.ado`（`findfile dfuller.ado`），答案只有三行**：

```stata
_align maxlen "dinobs dilag" : "`T' `lags'" "11.0gc"
local tablen 62
local col = `tablen'-14-3-`maxlen'+1
```

`col = 46 - maxlen`；`_col()` **从 1 开始计**（所以 Python 索引是 `45 - maxlen`）；
`display` **直接拼接值、不补位**——**这一条解释了全部 6 次失败**。

**正确做法**：

```python
# 1. 先找源码
s.run('findfile dfuller.ado')          # -> D:\...\ado\base\d\dfuller.ado
# 2. 搜关键词，不要通读
grep -n 'Number of obs\|Number of lags\|_col' dfuller.ado
# 3. 找到 col 的定义，照抄公式
```

**为什么反推必然失败**：Stata 的 `display` + `_col()` 组合产生的是
**「列位置 + 直接拼接」**，而不是「字段宽度 + 对齐」。用字段宽度去拟合，
每次只能对上**恰好没有溢出的那个样本**。

**触发条件**（满足任一条就该去读源码）：

- 同一张表的列规则试了 **2 次**仍有一种样本对不上；
- 不同样本呈现**互相矛盾**的列位置（如 `Number` 在 40 又 42）；
- 字段宽度与直觉不符，需要"猜一个 magic number"。

**代价对比**：6 次试错 ≈ 2 小时；`findfile` + 一次 grep ≈ **2 分钟**。

- **教训**：Stata 的表格几何**不是猜出来的**，`ado` 文件里写着确切公式。
  **两次不成功就停止反推，转去读源码。**

## 28. 说明性文字的审计要扫源码，不能只靠「跑一遍看输出」

前七轮用「对命令跑一次 Stata、抓 `note:` 开头的行」来查遗漏，15 + 13 + 12 + 17 个
命令里只发现 2 个 `note:` 生产者。**这个结论是假阴性**。

第 8 轮改为读 ado 源码（3962 个文件里 53 个相关命令），抓所有含
`note:` / `warning:` / `convergence` / `collinear` 的 display 语句，一次找到
**18 个命令的条件分支**，其中大部分靠正常运行永远触发不了：

| 分支 | 触发条件 |
|---|---|
| `poisson`/`nbreg` 的 noncount note | 因变量非整数 |
| `zip`/`zinb` 的 "never equal to zero" | 因变量无零值 |
| `prais`/`xtpcse` 的 gap note | 时间序列有缺口 |
| `teffects ipw` 的 `Warning: Convergence not achieved.` | 迭代 301 次未收敛 |
| `xtreg` 的 `Warning: Convergence not achieved.` | 面板 MLE 未收敛 |

**教训**：输出探测只能证明「这个数据集下没有」，不能证明「这个命令不会」。
条件说明藏在 `if` 里，必须读源码或刻意构造边界数据。
以后查「Stata 会不会给某类提示」，第一步应是 `grep` ado 源码的 display 语句，
第二步才用构造数据实测，第三步才对拍本包。

相关：`error.md` #27（几何对不上两次就读 .ado）——本条是它在「说明文字」维度的推广。

## 29. 版本号不能按字符串比较

全量测试报 `test_documented_version_matches_pyproject` 失败：

    pyproject 声明 0.3.11，但 PyPI 上已发布 0.3.9

**PyPI 上确实是 0.3.11**，是测试本身错了。

`_pypi_versions()` 用 `sorted(payload.get("releases", {}))` 做**字符串排序**，
而 `"0.3.9" > "0.3.11"`（因为 `'9' > '1'`），所以 `versions[-1]` 报的是 0.3.9。
断言又用 `declared >= newest` 直接比字符串，`'0.3.11' >= '0.3.9'` 同样为 False。

**修复**：`_version_key()` 把 `0.3.11` 拆成元组 `(0, 3, 11)` 再比较。

```python
def _version_key(version):
    parts = []
    for piece in str(version).split("."):
        digits = re.match(r"\d+", piece)
        parts.append(int(digits.group()) if digits else 0)
    return tuple(parts)
```

**教训**：任何形如 `0.3.9` / `0.3.11` 的版本号，**永远不要当字符串排序或比较**。
补丁号一到两位数就会翻车，而且只在那一个版本上翻，极易被误判成"发布失败"。
修复时顺带发现 `re` 只在函数内局部导入，需要提到模块级。

## 30. 全量测试耗时会波动 3 倍，不能当卡死处理

同一条全量命令：

| 时间 | 耗时 |
|---|---|
| 0.3.10 | 14:33 |
| 0.3.11 第一次 | 15:03 |
| 0.3.11 第二次 | **45:51** |

第二次运行到 30 分钟时 CPU 只有 0、单线程 Running、67 线程 Wait——**看起来像卡死**，
但子进程 CPU 时间仍在增长（1798s -> 2286s），最终正常跑完。

**判断方法**：看 `.venv` 的 python **子进程** CPU 累计时间是否在涨，
不要看包装进程（`pwsh` 拉起的那个，它 CPU 恒为 0）。
用 `Get-Process -Id <子进程>` 的 `CPU` 属性前后对比，涨就是在跑。

**教训**：`pytest -q | Select-Object -Last N` 在结束前**没有任何输出**，
不能靠"没有新输出"判断卡死；必须另查进程状态。

## 31. 稳健 VCE 的尺度因子要按命令查，`HC1` 不是万能对应

用户对比 Stata `logit ..., [fweight=freq], r` 与 westata 展开权重 + `cov_type="HC1"`，标准误对不上：

| 做法 | se[child] |
|---|---|
| Stata `logit, r` | 0.2767452 |
| westata `cov_type="HC1"` | 0.2770602（偏大 1.00137 倍）|
| westata `cov_type="robust"` | 0.2767452 ✅ |

`HC1` 用 statsmodels 的 `N/(N-K)`；Stata 的 `logit, r` 用 **`N/(N-1)`**，而包的 `robust`
分支（`discrete.py:155`）**本来就实现了 `N/(N-1)`**。

**教训**：`HC0`/`HC1` 是 statsmodels 的约定，不是 Stata 的。判断"该用哪个"必须**实测**：
先算 `Stata_se / HC0_se`，平方后看它等于 `N/(N-K)`、`N/(N-1)` 还是别的。
不要因为"Stata 写的是 r，所以用 HC1"就想当然——`regress, r` 是 HC1，`ivregress, r` 是 HC0，
`logit, r` 又是 `N/(N-1)`，三个命令三种尺度。

## 32. 行数对不上时先分清"哪部分是新增的"

`logit` 的 `summary()` 在修复后多了 `Iteration 0` 行，导致
`test_likelihood_table_border` 从 13 行变 14 行。

该测试的过滤逻辑只去掉了 `Iteration` 行，**没去掉它后面的空行**。Stata 的真实结构是：

    logit y x, nolog     '' / banner / ...            13 行
    logit y x            '' / Iteration 0 / '' / banner / ...   15 行

所以新增的是**"迭代行 + 其后空行"两行**，不是一行。

**教训**：给"新增了哪些行"下结论前，把两边逐行列出来对比，别按直觉只减掉最显眼的那一行。
过滤器写的是"去掉 X"，但真实增量是"X 及其伴随物"。

## 33. 前导空行的归属：note 没有，迭代日志才有

同一轮里三种场景的 Stata 首行：

| 命令 | 第 0 行 |
|---|---|
| `logit y x` | `''`（空行）|
| `xtlogit y x, fe`（打印 note）| `note: ...`（**无空行**）|
| `xtlogit y x, fe nolog` | `note: ...`（**无空行**）|

旧代码剥掉了所有前导空行，于是：**有 note 的表碰巧正确，纯 logit 缺空行**。
这种"一半对一半错"的 bug 最容易被漏掉——因为验证时只抽了有 note 的那个案例。

**教训**：输出首行的对齐规则要**按分支分别实测**，不能用一个案例推全体。
`_preamble` 里 note 和迭代日志是两条路径，空行归属也不同。
