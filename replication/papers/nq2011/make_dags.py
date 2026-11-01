"""Draw the causal diagrams for Nunn & Qian (2011).

The paper's identification is a historical natural experiment, and the diagram has to carry
three things that prose makes easy to lose:

1. the potato arrived *exogenously* (from the Americas), so its adoption is not chosen;
2. the outcome is measured centuries later, so reverse causality is implausible;
3. the "flexible" specification traces the effect year by year, and that is what the
   replication compares -- eleven coefficients, not one.

Layout rules as before: fixed frame, title band enforced in code, arrows anchored to box
edges, detours routed explicitly.
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


def _arrow(ax, src, dst, *, color=INK, lw=2.0, dashed=False, rad=0.0, via=None,
           label="", label_at=None):
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
    if label:
        lx, ly = label_at or ((pts[0][0] + pts[-1][0]) / 2, (pts[0][1] + pts[-1][1]) / 2)
        ax.text(lx, ly, label, ha="center", va="center", fontsize=8.6, color=color,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="none", alpha=0.95), zorder=6)


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


def _note(ax, y, text, *, edge=MUTED, face="#f7f9fc", size=8.9):
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
    """Figure 1: why the potato is an exogenous shock, and what it is used to explain."""

    fig, ax = _figure(width=10.0, height=6.6)
    _title(ax, "图 1：为什么马铃薯可以当外生冲击",
           "它从美洲传入，传入与否不是欧洲各地自己选的；结果是几百年后的人口与城市化")

    origin = Node(ax, 1.65, 4.55, "美洲原产地", sub="哥伦布交换之前欧洲没有马铃薯",
                  width=2.7, height=0.95, face=GOOD, edge=PATH, bold=False)
    arrival = Node(ax, 4.85, 4.55, "传入时间与强度", sub="因地区而异，取决于航线与气候",
                   width=2.8, height=0.95)
    outcome = Node(ax, 8.10, 4.55, "人口与城市化", sub="1000–1900 年的长期水平",
                   width=2.6, height=0.95)

    _arrow(ax, origin, arrival, color=PATH, lw=2.6)
    ax.text(3.25, 4.98, "外生", ha="center", fontsize=8.8, color=PATH, zorder=7,
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.95))
    _arrow(ax, arrival, outcome, color=PATH, lw=3.0)
    ax.text(6.50, 4.98, "想估计的效应", ha="center", fontsize=8.8, color=PATH, zorder=7,
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.95))

    conf1 = Node(ax, 4.85, 2.55, "地理与气候", sub="纬度、海拔、热带比例", width=2.8, height=0.9,
                 face=BAD, edge=BIAS, bold=False)
    conf2 = Node(ax, 8.10, 2.55, "其他作物与技术", sub="玉米、木薯、疟疾环境", width=2.8, height=0.9,
                 face=BAD, edge=BIAS, bold=False)
    _arrow(ax, conf1, arrival, color=BIAS, dashed=True, lw=1.8)
    _arrow(ax, conf1, outcome, color=BIAS, dashed=True, lw=1.8, rad=0.15)
    _arrow(ax, conf2, outcome, color=BIAS, dashed=True, lw=1.8)

    _note(ax, 1.05,
          "论文的处理：国家固定效应 + 年份固定效应。前者吸收「某国一直人口多」这类固定差异，\n"
          "后者吸收「所有国家在这一年都增长」的共同趋势。剩下的是「某国在某年因马铃薯而多出的人口」。",
          edge=MUTED, size=8.7)
    _legend(ax, [(False, PATH, "想估计的路径"), (True, BIAS, "固定效应要吸收的差异")], y=0.30)
    return _save(fig, "nq_00_design.png")


def figure_flexible() -> Path:
    """Figure 2: what the "flexible" specification estimates -- eleven coefficients."""

    fig, ax = _figure(width=10.2, height=6.8)
    _title(ax, "图 2：论文的「灵活设定」估计的是什么",
           "不是单一系数，而是马铃薯效应随时间逐年放大——这是它区别于普通 DiD 的地方")

    base = Node(ax, 2.15, 4.30, "马铃薯适宜度", sub="ln_wpot（跨时不变的国别特征）",
                width=3.0, height=0.95, face=GOOD, edge=PATH, bold=False)
    inter = Node(ax, 5.45, 4.30, "× 年份虚拟变量", sub="得到 11 个交互项",
                 width=2.4, height=0.95)
    outs = Node(ax, 8.45, 4.30, "ln_population", sub="或 city_pop_share", width=2.4, height=0.95)
    _arrow(ax, base, inter, color=PATH, lw=2.5)
    _arrow(ax, inter, outs, color=PATH, lw=3.0, label="11 个系数", label_at=(7.10, 3.90))

    _note(ax, 2.55,
          "每列报告 11 个交互项系数（1100 年 … 1900 年）。本次复现逐个核对了其中的 1800 与 1900 两项：\n"
          "表 2 第 (1) 列：1800 年 0.0726、1900 年 0.1213 —— 与作者日志逐位相同。\n"
          "系数随时间递增，正是「马铃薯的效应需要几百年才充分显现」这一论点的证据。",
          edge=INK, size=8.7)

    fixed = Node(ax, 5.45, 1.30, "国家固定效应 + 年份固定效应", sub="130 个国家 × 12 个年份",
                 width=4.8, height=0.85, face=GOOD, edge=PATH, bold=False)
    _arrow(ax, fixed, outs, color=PATH, lw=1.7, dashed=True, via=(9.55, 1.30))
    return _save(fig, "nq_01_flexible.png")


def figure_areg_xtreg() -> Path:
    """Figure 3: the finding -- one Stata, two standard errors."""

    fig, ax = _figure(width=10.0, height=6.4)
    _title(ax, "图 3：本次复现的核心发现——同一个 Stata，两种标准误",
           "areg 与 xtreg, fe 给出完全相同的系数，标准误却差 4.5%")

    _note(ax, 5.55,
          "同一条命令链、同一份数据、同一个系数 0.1213388。",
          edge=INK, size=9.2)

    areg = Node(ax, 2.55, 4.20, "areg", sub="作者当年跑的，也是论文报告的",
                width=2.6, height=0.95, face=BAD, edge=BIAS)
    xtreg = Node(ax, 7.45, 4.20, "xtreg, fe", sub="面板估计器的等价写法",
                 width=2.6, height=0.95, face=GOOD, edge=PATH)

    ax.text(2.55, 3.28, "se = 0.0171579", ha="center", fontsize=11, fontweight="bold", color=BIAS)
    ax.text(8.55, 3.28, "se = 0.0164182", ha="center", fontsize=11, fontweight="bold", color=PATH)
    ax.annotate("", xy=(3.95, 2.72), xytext=(6.05, 2.72),
                arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=1.6, mutation_scale=13))
    ax.text(5.00, 2.30, "差 4.5%", ha="center", fontsize=12, fontweight="bold", color=INK)

    ws_node = Node(ax, 7.45, 1.15, "westata FEOLS / PanelOLS",
                   sub="两条路径都等于 xtreg", width=3.3, height=0.9, face=GOOD, edge=PATH, bold=False)
    _arrow(ax, xtreg, ws_node, color=PATH, lw=2.2)
    ax.text(9.30, 2.55, "一致到 1e-13", ha="center", fontsize=8.4, color=PATH, zorder=7,
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.95))
    _arrow(ax, areg, ws_node, color=BIAS, dashed=True, lw=1.6)
    ax.text(2.55, 1.70, "差 4.5%：约定差异，非缺陷", ha="center", fontsize=8.6, color=BIAS,
            zorder=7, bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                                edgecolor="none", alpha=0.95))

    _note(ax, 0.42,
          "原因：areg 把小样本聚类调整里的 K 计入了被吸收的 130 个国别；xtreg 不计入。两者都是 Stata 的合法约定。",
          edge=MUTED, size=8.6)
    return _save(fig, "nq_02_areg_xtreg.png")


FIGURES = (
    ("nq_00_design", figure_design),
    ("nq_01_flexible", figure_flexible),
    ("nq_02_areg_xtreg", figure_areg_xtreg),
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
