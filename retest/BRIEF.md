# 复测任务书（westata vs StataNow 19.5 MP）— 所有子智能体必读

## 0. 任务是什么

从头独立复测 `westata` 各功能与真实 Stata 的数值差异，**不使用**仓库里既有的任何测试、
fixture、registry 结论作为证据。旧证据（`tests/conformance/stata_fixtures/`、
`tests/conformance/registry.yml`、`conformance*.csv`、`docs/GAP_REGISTER.md`）一律**只读且不可作为结论引用**：
它们是被检验对象，必须被本次实测独立复核或推翻。

你唯一的证据来源是：**你自己在本机启动的 StataNow 19.5 (MP) 会话** + **本机 westata 的实跑结果**。

## 1. 环境（已由 Lead 验证，可直接用）

- 仓库根：`D:\myCodeProject\statapy`（所有命令都以此为 cwd）
- Python：`.venv\Scripts\python.exe`（Python 3.10.11，可 `import westata`，已 editable 安装）
- Stata：`D:\Program Files\StataNow19`，通过 `pystata` 驱动，**已可用**（`config.init("mp", splash=False)`）。
  版本 19.5 MP，每次会话约 95 MB 内存、冷启动约 2 秒；**可以同时跑多个会话**，但每个子智能体请
  控制在自己的进程内串行使用，避免一次开 3 个以上。
- 已安装：numpy 2.2.6 / pandas 2.3.3 / scipy 1.15.3 / statsmodels 0.15.0 / linearmodels 7.1 /
  pyfixest 0.60.0 / arch 8.0.0 / patsy 1.0.3。

## 2. 共享工具（已写好，直接 import，**禁止修改**）

`retest/lib/stata_probe.py` 是本次复测唯一的公共契约，Lead 已单测过：

```python
import sys; sys.path.insert(0, "retest/lib")
from stata_probe import Stata, compare, Record, rel_diff, tier_for, canon_stata, canon_westata

with Stata() as st:                       # 启动内嵌 Stata（mp）
    st.run("sysuse auto, clear")          # 出错会抛异常，不会静默
    cap = st.estimate("regress mpg weight i.foreign")   # 捕获 e(b)/e(V)/e() scalars
    cap.b, cap.V, cap.names, cap.scalars, cap.macros, cap.nobs

res = ws.OLS.from_formula("mpg ~ weight + C(foreign)", df).fit()
comp = compare("regress i.foreign", cap, res, tolerance=1e-9,
               scalars=("N", "df_r", "r2", "rmse"))
Record("linear").add(comp)                # 立刻落盘 + 打印一行
```

`Record(域名)` 会写 `retest/domains/<域名>/results.jsonl`；收尾必须调用 `rec.flush_summary()`
生成 `summary.json`。

### 2.1 你要用的辅助函数

| 需求 | 用法 |
|---|---|
| 跑命令（出错抛异常） | `st.run(cmd)` / `st.quiet(cmd)`（自动加 `quietly`） |
| 拿打印输出（解析表格） | `st.output(cmd)` |
| 读 `r()/e()/c()` 标量 | `st.scalar("r(mean)")`，缺失返回 `None` |
| 读矩阵 | `st.matrix("r(b)")`、`st.matrix_names("r(b)")`、`st.matrix("e(V)")` |
| 数据进 Stata | `st.send(df)`（`pd.DataFrame`，**列名即变量名**，字符串列 Stata 侧是 str 变量） |
| 数据出 Stata | `st.frame()`（缺失值已转 `NaN`） |
| 单个变量 | `st.var("xb")`（缺失值已转 `NaN`）、`st.var_str("name")` |
| 观测数 | `st.nobs()` |
| 估计并捕获 | `st.estimate("logit y x")` |
| test/lincom 类 | `st.test_command("test x1 x2")` → `r(F)/r(df)/r(p)` |

### 2.2 对齐规则（`compare` 已实现，仍要理解）

- Stata 的 `e(b)` 里因子变量**包含基准水平**（`0b.foreign`，值为 0），westata/patsy **不含基准**。
  `Stata.estimate()` 会剔除 `<n>b.<var>` 的基准行，`compare` 再把 `i.var` 的 `1.g` 与 patsy 的
  `C(g)[T.1]` / `g[T.1]` 按「变量名=水平」规范化后配对。数字水平 `2.0` 与 `2` 视为同一水平。
- 配不上的系数会进 `comp.mismatched` 并自动判 fail —— 这时**不要**改工具去吞掉，要查清是命名差异
  还是真的少/多解释了变量，然后在 NOTES.md 里写清楚。
- 常数项：Stata `_cons` ↔ patsy `Intercept`，已自动配对。

### 2.3 判定口径（必须统一，写进报告）

| 档位 | 相对差上限 | 含义 |
|---|---|---|
| A | 1e-9 | 机器精度，可以声称"完全一致" |
| B | 1e-6 | 数值实现差异（迭代收敛、算法不同） |
| C | 1e-3 | 可见的数值差异，系数第 4 位起不同 |
| D | 1e-2 | 明显不同，需要解释 |
| E | >1e-2 | 结论不同，必须归因 |

相对差定义：`max(|westata - stata| / max(|stata|, 1e-12))`，逐系数取最大。
标准误同样计算（用 `e(V)` 的对角线开方，**不用**打印出来的四舍五入值）。

## 3. 纪律（违反即作废）

1. **只写真作用域内的文件**：`retest/domains/<你的域名>/` 下的 `.py` / `.jsonl` / `summary.json` / `NOTES.md`。
   **绝对不要**改 `src/`、`tests/`、`scripts/`、`retest/lib/`，也不要 `git commit`。
2. **每条结论必须有实测数字**。报告里写"一致"必须附相对差；没跑过的写"未测"。
3. **不许用打印精度冒充一致性**：必须用 `e(b)/e(V)/e()` 全精度矩阵。只有 Stata 不存 e() 的量
   （例如 `summarize` 只存 `r()`、`esttab` 只打印）才允许解析输出，且要在 NOTES 里说明。
4. **样本必须真的同一样本**：Stata 侧 `sysuse auto, clear` 与 Python 侧 `pd.read_stata("tests/data/auto.dta",
   convert_categoricals=False)` 是同一份数据；自己造数据时用固定 seed，`st.send(df)` 送进去，
   并核对 `st.nobs()` 与 westata `res.nobs` 一致（不一致就是 listwise deletion 差异，要单列一条）。
5. **失败要归因，不许掩饰**：差在收敛、自由度、权重语义、因子编码、方差估计量（HC0/HC1/CR1）、
   分布假设（t vs z）、还是真的算错，给出可验证的判断依据。
6. **westata 报错/不支持的路径**，记录为 `status=error` 并附异常原文；不要为了让它跑通去改参数凑。
7. 每个领域**至少 12 个对比点**；主要命令要覆盖变体（vce、权重、因子、子样本、预测类型）。
8. 进程内串行跑 Stata；单条 3 分钟还没结束的，考虑缩小样本并在 NOTES 说明。

## 4. 交付物（每个领域三样，缺一不可）

1. `retest/domains/<域名>/run.py` —— 可重跑的脚本（`cd D:\myCodeProject\statapy && .venv\Scripts\python.exe retest/domains/<域名>/run.py`）。
2. `retest/domains/<域名>/results.jsonl` + `summary.json` —— 由 `Record` 产生。
3. `retest/domains/<域名>/NOTES.md` —— 人读的结论，结构固定：

```markdown
# <领域> 复测结论

## 覆盖矩阵
| # | Stata 命令 | westata 调用 | 数据 | 判定 | 系数相对差 | 标准误相对差 |

## 逐项说明
（每个不达 A 档的条目：现象、归因、证据、是否影响结论/显著性判断）

## 与旧结论的出入
（旧 registry/README 声称的等级 vs 本次实测；一致/不一致都要写）

## 未覆盖
（想做但没做的，及原因；不要含糊）
```

## 5. 你完成后要回给 Lead 的一段话

- 领域名、对比点总数、pass/fail/error 计数；
- **A 档（≤1e-9）的条目清单**（一行一条：命令 + 最大相对差）；
- **不达 A 档的条目清单**：命令、系数相对差、SE 相对差、你的归因结论；
- **error/skip** 条目及原因；
- 与旧结论**冲突**的地方（这是最有价值的部分）。

## 6. 已知陷阱（Lead 已踩过，别再踩）

- `from pystata import stata` 必须在 `config.init()` **之后** import，否则 `SystemError`。
- Stata 缺失值 = `8.988e+307`；`st.frame()/st.var()` 已转 `NaN`，自己用 `sfi.Data.get` 时要手动转。
- `stata.get_ereturn()` 在非估计命令后会**保留上一个估计**的结果 —— 先跑估计再 capture。
- `predict` 之后用 `st.var("xb")`，不要用 `pdataframe_from_data()` 取（慢且易错）。
- 字符串变量不能做因子变量（Stata `r(109)`）；要比较分组就用数值编码列。
- `margins` 的结果在 `r(b)/r(V)`（不是 e()），`st.matrix("r(b)")` 可取。
- `test` 的结果在 `r(F)/r(df)/r(p)`（或 `r(chi2)`），`st.test_command` 已封装。
- `sysuse` 数据集只能取 Stata 自带的那批；需要别的一律自己造 `DataFrame` 送进去。
