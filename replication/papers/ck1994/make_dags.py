"""Draw the causal diagrams for Card & Krueger (1994).

Card & Krueger is the canonical *natural experiment*: the argument is not
"we controlled for confounders", it is "an exogenous policy change moved one group and
not another, so the difference in their changes is the causal effect".  A diagram makes
that structure checkable -- a reader can ask "is there any path from the policy to the
outcome other than through the treatment?" and answer it by looking.

Four figures, in the order the argument needs them:

1. ``ck_00_naive``     a single cross-section: why comparing NJ with PA levels fails
2. ``ck_01_did``       the difference-in-differences: what the two differences remove
3. ``ck_02_pretrend``  the identifying assumption, and what would break it
4. ``ck_03_final``     the specification actually estimated (Table 4 column v)

Layout rules are the ones learned on the ADH figures: a fixed frame with a title band no
node may enter, arrows anchored to box edges, and detours routed explicitly rather than
hoped for.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

X_MAX, Y_MAX = 10.0, 7.0
TITLE_Y, SUBTITLE_Y = 6.82, 6.48
NODE_CEILING = 6.02

INK = "#1f2933"
PATH = "#0b6e4f"
BIAS = "#c1121f"
MUTED = "#6b7785"
BOX = "#eef2f7"
GOOD = "#eef7f1"
BAD = "#fdf0f0"


def _figure(width: float = 9.8, height: float = 6.4):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, X_MAX)
    ax.set_ylim(0, Y_MAX)
    ax.axis("off")
    return fig, ax


class Node:
    def __init__(self, ax, x, y, label, *, sub="", width=2.1, height=0.9,
                 face=BOX, edge=INK, bold=True):
        if y + height / 2 > NODE_CEILING:
            raise ValueError(f"node {label!r} intrudes into the title band")
        self.x, self.y, self.w, self.h = x, y, width, height
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x - width / 2, y - height / 2), width, height,
                boxstyle="round,pad=0.05,rounding_size=0.12",
                linewidth=1.7, edgecolor=edge, facecolor=face, zorder=3,
            )
        )
        ax.text(x, y + (0.13 if sub else 0), label, ha="center", va="center",
                fontsize=10.3, color=INK, fontweight="bold" if bold else "normal", zorder=4)
        if sub:
            ax.text(x, y - 0.19, sub, ha="center", va="center", fontsize=8.0,
                    color=MUTED, zorder=4)

    def edge(self, toward, pad=0.10):
        dx, dy = toward[0] - self.x, toward[1] - self.y
        if dx == 0 and dy == 0:
            return (self.x, self.y)
        half_w, half_h = self.w / 2 + pad, self.h / 2 + pad
        sx = half_w / abs(dx) if dx else float("inf")
        sy = half_h / abs(dy) if dy else float("inf")
        s = min(sx, sy)
        return (self.x + dx * s, self.y + dy * s)


def _arrow(ax, src, dst, *, color=INK, lw=1.9, dashed=False, rad=0.0,
           via=None, label="", label_at=None):
    points = [src.edge(via or (dst.x, dst.y))]
    if via:
        points.append(via)
        points.append(dst.edge(via))
    else:
        points.append(dst.edge((src.x, src.y)))

    if len(points) == 2:
        ax.add_patch(FancyArrowPatch(
            points[0], points[1], arrowstyle="-|>", mutation_scale=15, linewidth=lw,
            color=color, linestyle=(0, (5, 3)) if dashed else "solid",
            connectionstyle=f"arc3,rad={rad}", zorder=2,
        ))
    else:
        ax.plot([p[0] for p in points[:-1]], [p[1] for p in points[:-1]],
                color=color, lw=lw, linestyle=(0, (5, 3)) if dashed else "solid", zorder=2)
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                     mutation_scale=15, linewidth=lw, color=color, zorder=2))

    if label:
        lx, ly = label_at or ((points[0][0] + points[-1][0]) / 2,
                              (points[0][1] + points[-1][1]) / 2)
        ax.text(lx, ly, label, ha="center", va="center", fontsize=8.6, color=color,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="none", alpha=0.95), zorder=6)


def _title(ax, text, sub=""):
    ax.text(X_MAX / 2, TITLE_Y, text, ha="center", va="top", fontsize=13.5,
            fontweight="bold", color=INK)
    if sub:
        ax.text(X_MAX / 2, SUBTITLE_Y, sub, ha="center", va="top", fontsize=9.3, color=MUTED)


def _legend(ax, items, y=0.34):
    total = sum(0.62 + 0.108 * len(label) for _, _, label in items)
    x = (X_MAX - total) / 2
    for dashed, color, label in items:
        ax.plot([x, x + 0.5], [y, y], color=color, lw=2.2,
                linestyle=(0, (5, 3)) if dashed else "solid")
        ax.text(x + 0.62, y, label, va="center", fontsize=8.8, color=INK)
        x += 0.62 + 0.108 * len(label) + 0.7


def _note(ax, y, text, *, edge=MUTED, face="#f7f9fc", size=9.2):
    ax.text(X_MAX / 2, y, text, ha="center", va="center", fontsize=size, color=INK,
            bbox=dict(boxstyle="round,pad=0.42", facecolor=face, edgecolor=edge,
                      linewidth=1.1), zorder=5)


def _save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.18)
    plt.close(fig)
    return path


# --------------------------------------------------------------------- figures
def figure_naive() -> Path:
    """Figure 1: why comparing New Jersey with Pennsylvania levels cannot answer it."""

    fig, ax = _figure()
    _title(ax, "图 1：直接比较新泽西与宾夕法尼亚为什么不行",
           "两个州本来就不同，任何水平差异都可能来自这些差异，而不是最低工资")

    nj = Node(ax, 3.0, 3.30, "新泽西（提高最低工资）", sub="N = 331", width=2.8)
    pa = Node(ax, 7.0, 3.30, "宾夕法尼亚（未提高）", sub="N = 79", width=2.8)

    state = Node(ax, 5.0, 5.30, "两州本已存在的差异",
                 sub="产业结构、工资水平、连锁品牌、区域经济",
                 width=4.6, height=0.95, face=BAD, edge=BIAS, bold=False)

    _arrow(ax, state, nj, color=BIAS, dashed=True)
    _arrow(ax, state, pa, color=BIAS, dashed=True)

    out = Node(ax, 5.0, 1.55, "FTE 就业水平", sub="1992 年第二期调查", width=2.6)
    _arrow(ax, nj, out, color=INK, dashed=True, rad=-0.15)
    _arrow(ax, pa, out, color=INK, dashed=True, rad=0.15)

    _note(ax, 0.70,
          "两州在 1990 年代初期市场规模就不同：新泽西店平均 FTE 20.4，宾州 23.4。\n"
          "直接比较 21.0 vs 21.1 得到的差异，无法区分「最低工资的影响」与「两州本来就不同」。",
          edge=BIAS, face="#fff5f5")
    _legend(ax, [(True, BIAS, "会污染比较的差异"), (True, INK, "无法识别的比较")])
    return _save(fig, "ck_00_naive.png")


def figure_did() -> Path:
    """Figure 2: the two differences and what each one removes."""

    fig, ax = _figure()
    _title(ax, "图 2：双重差分——两次相减，各消掉一部分问题",
           "第一次差分消掉两州固有的水平差异，第二次差分消掉共同的宏观变化")

    nj0 = Node(ax, 2.2, 4.55, "新泽西 1992 年前", sub="FTE 20.43", width=2.5, height=0.85)
    nj1 = Node(ax, 2.2, 2.75, "新泽西 1992 年后", sub="FTE 20.90", width=2.5, height=0.85)
    pa0 = Node(ax, 7.5, 4.55, "宾州 1992 年前", sub="FTE 23.38", width=2.5, height=0.85)
    pa1 = Node(ax, 7.5, 2.75, "宾州 1992 年后", sub="FTE 21.10", width=2.5, height=0.85)

    _arrow(ax, nj0, nj1, color=PATH, lw=2.4, label="+0.47", label_at=(1.35, 3.65))
    _arrow(ax, pa0, pa1, color=MUTED, lw=2.4, label="−2.28", label_at=(8.40, 3.65))

    ax.annotate("", xy=(3.7, 2.75), xytext=(6.0, 2.75),
                arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=1.6, mutation_scale=13))
    ax.text(4.85, 2.35, "差 2.75", ha="center", fontsize=11, fontweight="bold", color=INK)

    _note(ax, 1.35,
          "第一次差分（每州自己前后相减）：消掉该州固定的水平差异。\n"
          "第二次差分（新泽西的变化 − 宾州的变化）：消掉两州共同经历的宏观冲击（如全国性衰退）。\n"
          "剩下的 2.75 FTE 就是最低工资的因果效应——前提是「若无政策，两州会同步变化」。",
          edge=PATH, face="#eef7f1")
    _legend(ax, [(False, PATH, "新泽西的变化"), (False, MUTED, "宾州的变化")])
    return _save(fig, "ck_01_did.png")


def figure_pretrend() -> Path:
    """Figure 3: the identifying assumption, and the two ways it fails."""

    fig, ax = _figure()
    _title(ax, "图 3：识别假设——「若无政策，两州会同向同步变化」",
           "这个假设无法直接检验，只能看政策前是否满足，并排除两种破坏它的情形")

    nj = Node(ax, 3.0, 4.60, "新泽西：就业变化", sub="政策前 → 政策后", width=2.9)
    pa = Node(ax, 7.0, 4.60, "宾州：就业变化", sub="同一时期", width=2.9)
    cf = Node(ax, 5.0, 2.80, "反事实", sub="新泽西若不提高最低工资会怎样", width=3.4,
              face=GOOD, edge=PATH, bold=False)

    _arrow(ax, cf, nj, color=PATH, lw=2.2, dashed=True, label="平行趋势", label_at=(3.4, 3.7))
    _arrow(ax, cf, pa, color=PATH, lw=2.2, dashed=True)

    break1 = Node(ax, 1.55, 1.85, "破坏情形 1", sub="新泽西本身有独立趋势", width=2.5,
                  height=0.85, face=BAD, edge=BIAS, bold=False)
    break2 = Node(ax, 8.45, 1.85, "破坏情形 2", sub="共同冲击对两州影响不同", width=2.7,
                  height=0.85, face=BAD, edge=BIAS, bold=False)
    _arrow(ax, break1, nj, color=BIAS, dashed=True, lw=1.7, via=(1.55, 3.9))
    _arrow(ax, break2, pa, color=BIAS, dashed=True, lw=1.7, via=(8.45, 3.9))

    # The note and the legend both want the bottom band, and in an earlier build they
    # overlapped.  The explanatory sentence lives in the figure caption in REPORT.md
    # instead, which keeps this figure to one idea: the assumption and how it fails.
    _legend(ax, [(True, PATH, "识别所依赖的路径"), (True, BIAS, "会破坏识别的情形")], y=0.60)
    return _save(fig, "ck_02_pretrend.png")


def figure_final() -> Path:
    """Figure 4: Table 4 column (v) -- the specification actually estimated."""

    fig, ax = _figure(width=10.2, height=7.0)
    _title(ax, "图 4：最终模型的因果图（表 4 第 (v) 列）",
           "OLS · 357 家门店 · 因变量为 FTE 就业变化 · 无权重 · 常规标准误")

    treat = Node(ax, 2.35, 3.60, "初始工资缺口 GAP", sub="gap = (5.05 − 起始工资)/起始工资",
                 width=3.3, height=0.95)
    out = Node(ax, 7.20, 3.60, "FTE 就业变化", sub="d_fte = 第二期 − 第一期", width=2.9, height=0.95)
    _arrow(ax, treat, out, color=PATH, lw=3.2)
    ax.text(4.78, 4.02, "11.98", ha="center", fontsize=11.5, fontweight="bold", color=PATH,
            zorder=7, bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                edgecolor="none", alpha=0.97))
    ax.text(4.78, 3.70, "(7.42)", ha="center", fontsize=8.8, color=MUTED,
            zorder=7, bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                                edgecolor="none", alpha=0.96))
    ax.text(4.78, 3.28, "论文印 11.91 (7.39)", ha="center", fontsize=8.2, color=MUTED,
            zorder=7, bbox=dict(boxstyle="round,pad=0.14", facecolor="white",
                                edgecolor="none", alpha=0.94))

    controls = Node(ax, 4.78, 5.45, "控制变量（7 个）",
                    sub="3 个连锁品牌虚拟变量 + 1 个直营虚拟变量 + 4 个区域虚拟变量",
                    width=5.4, height=0.95, face=GOOD, edge=PATH, bold=False)
    # Every control enters the same equation, so both arrows must land on the outcome;
    # the second one is drawn to the left of the first so the pairing is visible rather
    # than implied.
    _arrow(ax, controls, out, color=PATH, lw=1.8, dashed=True)
    _arrow(ax, controls, treat, color=PATH, lw=1.8, dashed=True,
           label="控制变量同时进入方程",
           label_at=(2.55, 4.42))

    ignored = Node(ax, 7.20, 1.45, "两州固有的水平差异",
                   sub="由 GAP 的定义吸收：宾州 GAP 恒为 0",
                   width=3.4, height=0.95, face=BAD, edge=BIAS, bold=False)
    _arrow(ax, ignored, out, color=BIAS, dashed=True, lw=1.7,
           label="已消除", label_at=(8.95, 2.55))

    _note(ax, 0.55,
          "17 个观测被排除后余 357 家：6 家永久关闭店计入零就业，4 家临时关闭店按缺失处理。\n"
          "因变量均值 −0.238，标准差 8.825（论文 −0.237、8.825）——样本构造正确的独立证据。",
          edge=INK, size=8.6)
    return _save(fig, "ck_03_final.png")


FIGURES = (
    ("ck_00_naive", figure_naive),
    ("ck_01_did", figure_did),
    ("ck_02_pretrend", figure_pretrend),
    ("ck_03_final", figure_final),
)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, builder in FIGURES:
        path = builder()
        print(f"  {name}: {path.relative_to(HERE)} ({path.stat().st_size // 1024} KB)")
    print(f"\n{len(FIGURES)} figures written to {OUT.relative_to(HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
