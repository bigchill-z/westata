# 通用问题记录（避免以后再犯）

> 记录每次测试/排障中发现的、与具体任务无关的通用坑。

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
