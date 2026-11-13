# 加权 logit 与 Stata 对齐：实测规格

> 用户报告 `logit survive child female class1 class2 class3 [fweight=freq], r`
> 与 westata 展开权重 + `cov_type="HC1"` 的输出不同。查下来是**三个独立原因**。

## 1. 稳健 VCE 的尺度：`robust` 不是 `HC1`

| 做法 | se[child] | 与 Stata 差 |
|---|---|---|
| Stata `logit ..., [fweight=freq], r` | 0.2767452 | — |
| westata `cov_type="HC1"` | 0.2770602 | 偏大 1.00137 倍 |
| westata `cov_type="robust"` | 0.2767452 | **一致（<1e-7）** |

原因：`HC1` 用 statsmodels 的 `N/(N-K)`；Stata 的 `logit, r` 用 **`N/(N-1)`**。

```python
# 正确写法
ws.Logit.from_formula(formula, expanded).fit(cov_type="robust")
```

`robust` 在 `discrete.py:155` 已经实现 `_sandwich(bread, score.T @ score, n/(n-1))`——
**代码本来就对，是我选错了参数**。

六个标准误全部命中：

```
child   0.2767452   female  0.1363096   class1  0.1475218
class2  0.1502193   class3  0.1621035   _cons   0.0798876
```

## 2. 稳健时报告的是 `Log pseudolikelihood`

稳健/聚类 VCE 下，Stata 把对数似然称为**伪似然**，banner 与迭代日志都改：

| VCE | 标签 |
|---|---|
| `nonrobust` | `Log likelihood` |
| `robust` / `cluster` / `HC0` / `HC1` | `Log pseudolikelihood` |

```
Log pseudolikelihood = -1105.0306                       Pseudo R2     = 0.2020
```

## 3. 前导空行只属于迭代日志

| 场景 | 首行 |
|---|---|
| `logit y x` | **空行**，第 1 行才是 `Iteration 0` |
| `xtlogit y x, fe`（有 note） | **note 直接开头，无空行** |
| `xtlogit y x, fe nolog` | 同上 |

旧代码剥掉了所有前导空行，导致：有 note 的表**碰巧正确**，纯 logit 缺空行。

## 4. 迭代值是 8 位有效数字，不是固定小数位

```
Iteration 0:  Log likelihood = -441.45229      <- 5 位小数
Iteration 0:  Log pseudolikelihood = -1384.7284 <- 4 位小数
```

两者都是 **8 位有效数字**。用固定 4 位小数会把 `-441.45229` 打成 `-441.4523`，少一位。

格式：`f"Iteration 0:  {label} = {value:>10.8g}  "`（值右对齐到第 37 列，末尾 2 空格）。

## 验证结果

去掉 Stata 的 `Iteration 1..4` 中间行（既有决策：statsmodels 的中间值不同，打印会造假）后：

```
S=19  W=19  差异行: []
```

**逐字符零差异。**

## 附带修正

`summary()` 的首行空行原先在有 note 时被错误加上，本轮改为按上面第 3 条的规则区分。
受影响的 5 个旧断言已更新（它们此前断言"无前导空行"，与 Stata 不符）。
