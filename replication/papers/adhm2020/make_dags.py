"""Draw the causal diagrams for Autor-Dorn-Hanson-Majlesi (2020).

Three figures, following the order the argument needs them:

1. the design -- why Chinese import competition is treated as an exogenous shock to a
   congressional district, and what the shift-share instrument does;
2. the estimated specification -- Table 3 column 5, with the two-way clustering shown as
   the thing that makes the standard error correct;
3. the clustering question itself, which is the replication's finding.

Figure 3 is deliberately about clustering rather than about the paper: a reader who only saw
figures 1-2 would not learn that the paper's standard errors cannot be reproduced here, and
that is the most important thing this replication has to say.
"""

from __future__ import annotations

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
WARM = "#fff7e6"


def _figure(width: float = 9.8, height: float = 6.4):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, X_MAX)
    ax.set_ylim(0, Y_MAX)
    ax.axis("off")
    return fig, ax


class Node:
    def __init__(self, ax, x, y, label, *, sub="", width=2.3, height=0.9,
                 face=BOX, edge=INK, bold=True):
        if y + height / 2 > NODE_CEILING:
            raise ValueError(f"node {label!r} intrudes into the title band")
        self.x, self.y, self.w, self.h = x, y, width, height
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - width / 2, y - height / 2), width, height,
            boxstyle="round,pad=0.05,rounding_size=0.12",
            linewidth=1.7, edgecolor=edge, facecolor=face, zorder=3))
        ax.text(x, y + (0.13 if sub else 0), label, ha="center", va="center",
                fontsize=10.2, color=INK, fontweight="bold" if bold else "normal", zorder=4)
        if sub:
            ax.text(x, y - 0.20, sub, ha="center", va="center", fontsize=8.0,
                    color=MUTED, zorder=4)

    def edge(self, toward, pad=0.10):
        dx, dy = toward[0] - self.x, toward[1] - self.y
        if dx == 0 and dy == 0:
            return (self.x, self.y)
        hw, hh = self.w / 2 + pad, self.h / 2 + pad
        sx = hw / abs(dx) if dx else float("inf")
        sy = hh / abs(dy) if dy else float("inf")
        s = min(sx, sy)
        return (self.x + dx * s, self.y + dy * s)


def _arrow(ax, src, dst, *, color=INK, lw=2.0, dashed=False, rad=0.0, via=None):
    pts = [src.edge(via or (dst.x, dst.y))]
    if via:
        pts += [via, dst.edge(via)]
    else:
        pts.append(dst.edge((src.x, src.y)))
    if len(pts) == 2:
        ax.add_patch(FancyArrowPatch(
            pts[0], pts[1], arrowstyle="-|>", mutation_scale=15, linewidth=lw,
            color=color, linestyle=(0, (5, 3)) if dashed else "solid",
            connectionstyle=f"arc3,rad={rad}", zorder=2))
    else:
        ax.plot([p[0] for p in pts[:-1]], [p[1] for p in pts[:-1]], color=color, lw=lw,
                linestyle=(0, (5, 3)) if dashed else "solid", zorder=2)
        ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>",
                                     mutation_scale=15, linewidth=lw, color=color, zorder=2))


def _label(ax, x, y, text, color=INK, size=8.6):
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=color, zorder=7,
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                      edgecolor="none", alpha=0.95))


def _title(ax, text, sub=""):
    ax.text(X_MAX / 2, TITLE_Y, text, ha="center", va="top", fontsize=13.5,
            fontweight="bold", color=INK)
    if sub:
        ax.text(X_MAX / 2, SUBTITLE_Y, sub, ha="center", va="top", fontsize=9.3, color=MUTED)


def _legend(ax, items, y=0.40):
    total = sum(0.62 + 0.108 * len(label) for _, _, label in items)
    x = (X_MAX - total) / 2
    for dashed, color, label in items:
        ax.plot([x, x + 0.5], [y, y], color=color, lw=2.2,
                linestyle=(0, (5, 3)) if dashed else "solid")
        ax.text(x + 0.62, y, label, va="center", fontsize=8.8, color=INK)
        x += 0.62 + 0.108 * len(label) + 0.7


def _note(ax, y, text, *, edge=MUTED, face="#f7f9fc", size=8.8):
    ax.text(X_MAX / 2, y, text, ha="center", va="center", fontsize=size, color=INK,
            bbox=dict(boxstyle="round,pad=0.42", facecolor=face, edgecolor=edge,
                      linewidth=1.1), zorder=5)


def _save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.18)
    plt.close(fig)
    return path


def figure_design() -> Path:
    """Figure 1: the exogenous shock and the shift-share instrument.

    Layout note: "China's export expansion" and "other countries' imports from China" are
    two *sources of variation* in the same regressor, so they are drawn stacked on the left
    with one arrow each.  An earlier build also drew a separate box for the shock and left
    it orphaned, and put the coefficient label on top of a border.
    """

    fig, ax = _figure(width=10.0, height=6.8)
    _title(ax, "图 1：把中国进口冲击当作选区的外生冲击",
           "用「其他国家对华进口」作工具变量，剥离美国本地需求这条旁路")

    shock = Node(ax, 2.05, 5.15, "中国出口扩张", sub="1990 年代起快速上升",
                 width=2.7, height=0.9, face=GOOD, edge=PATH, bold=False)
    other = Node(ax, 2.05, 3.45, "其他国家对华进口",
                 sub="与美国本地需求无关 → 作工具变量",
                 width=3.1, height=0.95, face=GOOD, edge=PATH, bold=False)
    exposure = Node(ax, 6.10, 5.15, "选区的进口暴露度", sub="d_imp_usch_pd",
                    width=2.9, height=0.9)
    outcome = Node(ax, 6.10, 2.05, "政治极化", sub="竞选捐款、投票率、得票份额",
                   width=3.0, height=0.9)

    _arrow(ax, shock, exposure, color=MUTED, lw=1.9, dashed=True)
    _label(ax, 4.05, 5.02, "理论上相关（有内生性）", MUTED, 8.0)
    _arrow(ax, other, exposure, color=PATH, lw=2.8, rad=-0.12)
    _label(ax, 4.10, 4.28, "工具变量：只经由暴露度起作用", PATH, 8.2)
    _arrow(ax, exposure, outcome, color=PATH, lw=3.2)
    _label(ax, 8.85, 3.62, "15.84", PATH, 11.0)

    demand = Node(ax, 1.75, 1.30, "本地需求冲击", sub="同时影响进口与政治行为",
                  width=3.0, height=0.85, face=BAD, edge=BIAS, bold=False)
    _arrow(ax, demand, outcome, color=BIAS, dashed=True, lw=1.8, via=(4.60, 1.30))
    _arrow(ax, demand, exposure, color=BIAS, dashed=True, lw=1.8, via=(4.60, 3.05))

    _legend(ax, [(False, PATH, "识别路径"), (True, BIAS, "需要切断的旁路"),
                 (True, MUTED, "有内生性的直接关系")], y=0.60)
    return _save(fig, "adhm_00_design.png")


def figure_clustering() -> Path:
    """Figure 2: who can do two-way clustering, and what the single-way fallback costs.

    The point of this figure is the comparison table, not a single number: westata is at
    parity with stock Stata on IV and behind it on OLS, and behind ``ivreg2`` on both.  An
    earlier version drew only "westata cannot do it", which overstated the gap by treating
    Stata as monolithic.
    """

    fig, ax = _figure(width=10.2, height=6.8)
    _title(ax, "图 2：谁能做双维聚类——以及单维替代要付多少代价",
           "作者用 ivreg2 得到 se=9.24；westata 与原版 Stata 在 IV 上都做不到，OLS 上 westata 落后")

    ax.text(1.55, 5.55, "路径", ha="center", fontsize=10, fontweight="bold", color=INK)
    ax.text(5.20, 5.55, "双维聚类", ha="center", fontsize=10, fontweight="bold", color=INK)
    ax.text(8.35, 5.55, "本篇数据上的 SE", ha="center", fontsize=10, fontweight="bold", color=INK)

    rows = [
        ("作者：ivreg2", "支持", 9.2369, PATH, GOOD),
        ("ivreg2（用户自写）", "支持", 9.2369, PATH, GOOD),
        ("原版 Stata regress", "支持（OLS）", 5.8256, PATH, GOOD),
        ("原版 Stata ivregress", "不支持", None, BIAS, BAD),
        ("westata FEOLS", "支持", 5.8247, PATH, GOOD),
        ("westata OLS / IV2SLS", "不支持", None, BIAS, BAD),
    ]
    y = 4.80
    for label, support, se, color, face in rows:
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.35, y - 0.26), 9.30, 0.52,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.2, edgecolor="#d5dbe3", facecolor=face, zorder=2))
        ax.text(1.55, y, label, ha="center", va="center", fontsize=9.4, color=INK, zorder=4)
        ax.text(5.20, y, support, ha="center", va="center", fontsize=9.4,
                color=color, fontweight="bold", zorder=4)
        ax.text(8.35, y, "—" if se is None else f"{se:.4f}", ha="center", va="center",
                fontsize=9.4, color=MUTED if se is None else INK, zorder=4)
        y -= 0.66

    _note(ax, 0.55, "单维替代的代价：czone 8.32、c.district 8.43，都低于双维 9.24（低 2%–11%）。方向一致地低估 —— 退而用单维会让 t 值虚高。", edge=BIAS, face="#fff5f5", size=8.4)
    return _save(fig, "adhm_01_clustering.png")


def figure_final() -> Path:
    """Figure 3: the estimated specification, Table 3 column 5."""

    fig, ax = _figure(width=10.2, height=6.8)
    _title(ax, "图 3：最终模型的因果图（表 3 第 (5) 列）",
           "2SLS · 3,772 个选区 · [aw=sh_district_2002] · 30 个参数")

    treat = Node(ax, 2.40, 3.70, "进口暴露度变化", sub="d_imp_usch_pd", width=2.9, height=0.95)
    out = Node(ax, 7.50, 3.70, "竞选捐款变化", sub="dhs2_tot_cont_2002_2010", width=3.1, height=0.95)
    _arrow(ax, treat, out, color=PATH, lw=3.2)
    ax.text(5.00, 4.42, "37.2320", ha="center", fontsize=11.5, fontweight="bold", color=PATH,
            zorder=7, bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                edgecolor="none", alpha=0.97))
    ax.text(5.00, 4.05, "(20.491 单维 / 21.046 论文双维)", ha="center", fontsize=8.4,
            color=MUTED, zorder=7,
            bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="none", alpha=0.95))

    iv = Node(ax, 2.40, 1.85, "工具变量", sub="d_imp_otch_lag_pd", width=2.6, height=0.85,
              face=GOOD, edge=PATH, bold=False)
    _arrow(ax, iv, treat, color=PATH, lw=2.2, dashed=True)

    ctrl = Node(ax, 5.05, 5.30, "28 个控制变量",
                sub="制造业/常规任务/外包 + 15 个人口结构 + 8 个普查区 + 2 个历史投票份额",
                width=7.6, height=0.95, face=GOOD, edge=PATH, bold=False)
    _arrow(ax, ctrl, out, color=PATH, lw=1.7, dashed=True)
    _arrow(ax, ctrl, treat, color=PATH, lw=1.7, dashed=True)

    _note(ax, 0.60,
          "30 个参数 = 1 个内生变量 + 28 个控制 + 常数项。五列系数全部与作者日志逐位相同；\n"
          "westata 与 Stata 的 10 个单维聚类回归一致到 3.2e-12。",
          edge=INK, size=8.5)
    return _save(fig, "adhm_02_final.png")


FIGURES = (
    ("adhm_00_design", figure_design),
    ("adhm_01_clustering", figure_clustering),
    ("adhm_02_final", figure_final),
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
