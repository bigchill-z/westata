"""Draw the causal diagrams for Autor & Dorn (2013).

The paper's argument is a two-stage chain, and the diagram should show both stages
because the interesting claim lives in the *link* between them:

1.  Routine-intensive places lost routine jobs (Table 3).
2.  Those same places gained low-skill service jobs (Table 4).
3.  Together: task displacement pushed workers into services, which is "polarization".

A reader who sees only stage 1 will read the paper as "routine jobs declined" -- true but
not the paper's point.  A reader who sees only stage 2 will read it as "services grew" --
also true and also not the point.  The figures are arranged so the chain is the thing you
notice.

Layout rules are the ones that survived the earlier figures: fixed frame, a title band no
node may enter (enforced in code), arrows anchored to box edges, detours routed explicitly.
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
    def __init__(self, ax, x, y, label, *, sub="", width=2.2, height=0.9,
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


def _note(ax, y, text, *, edge=MUTED, face="#f7f9fc", size=9.0):
    ax.text(X_MAX / 2, y, text, ha="center", va="center", fontsize=size, color=INK,
            bbox=dict(boxstyle="round,pad=0.42", facecolor=face, edgecolor=edge,
                      linewidth=1.1), zorder=5)


def _save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.18)
    plt.close(fig)
    return path


def figure_polarization() -> Path:
    """Figure 1: the whole argument -- what happened to the middle.

    Layout: the two mechanisms the paper estimates are stacked as parallel routes from the
    same regressor, with the labour-reallocation arrow joining them *below* the boxes.
    An earlier build drew the chain left-to-right and every label landed on a border.
    """

    fig, ax = _figure(width=10.0, height=6.8)
    _title(ax, "图 1：论文的核心论证——「中间」被挤掉了",
           "常规任务岗位消失、低技能服务业岗位增加：两头扩张、中间收缩")

    base = Node(ax, 5.00, 5.30, "1990 年常规任务密集度",
                sub="某 CZ 的常规岗位占比（基期，先于结果确定）",
                width=4.5, height=0.95, face=GOOD, edge=PATH, bold=False)
    rout = Node(ax, 2.05, 3.35, "常规岗位减少", sub="制造业、文书类", width=2.7)
    svc = Node(ax, 7.95, 3.35, "低技能服务业增加", sub="餐饮、清洁、看护、零售", width=3.0)

    _arrow(ax, base, rout, color=PATH, lw=2.8,
           label="表 3：−0.254", label_at=(2.60, 4.55))
    _arrow(ax, base, svc, color=PATH, lw=2.8,
           label="表 4：+0.083", label_at=(7.55, 4.55))
    _arrow(ax, rout, svc, color=PATH, lw=2.2, dashed=True,
           label="被挤出的劳动力流入", label_at=(5.00, 2.95))

    polar = Node(ax, 5.00, 1.55, "工资与就业极化",
                 sub="高技能与低技能两端扩张，中间收缩",
                 width=4.3, height=0.9, face=WARM, edge=INK, bold=False)
    _arrow(ax, rout, polar, color=INK, lw=1.7, dashed=True, rad=-0.10)
    _arrow(ax, svc, polar, color=INK, lw=1.7, dashed=True, rad=0.10)

    _note(ax, 0.62,
          "表 3 与表 4 是同一链条的两个环节；只有其中一条都只是局部事实，合起来才是「极化」。",
          edge=INK, size=8.6)
    _legend(ax, [(False, PATH, "论文估计的两个环节"), (True, PATH, "劳动力再配置"),
                 (True, INK, "加总后的极化")], y=0.18)
    return _save(fig, "ad_00_polarization.png")


def figure_identification() -> Path:
    """Figure 2: why the design can be read causally -- and where it is weakest."""

    fig, ax = _figure()
    _title(ax, "图 2：识别的逻辑与它的软肋",
           "初始的常规岗位占比是 1990 年就定下的，不会被后来的就业变化反向影响")

    base = Node(ax, 2.35, 4.20, "1990 年常规岗位占比",
                sub="基期产业结构，先于结果变量确定", width=3.2, height=0.95,
                face=GOOD, edge=PATH, bold=False)
    change = Node(ax, 7.25, 4.20, "常规岗位/服务业就业变化",
                  sub="1990-2005 的变化", width=3.0)
    _arrow(ax, base, change, color=PATH, lw=3.0, label="想估计的效应", label_at=(4.80, 3.85))

    local = Node(ax, 7.25, 2.25, "本地需求冲击",
                 sub="可能同时压低常规岗位、推高服务业", width=3.0, height=0.9,
                 face=BAD, edge=BIAS, bold=False)
    _arrow(ax, local, change, color=BIAS, dashed=True, lw=1.9)

    region = Node(ax, 7.25, 5.55, "区域长期趋势", sub="如铁锈带整体衰退",
                  width=2.6, height=0.85, face=BAD, edge=BIAS, bold=False)
    _arrow(ax, region, change, color=BIAS, dashed=True, lw=1.9)
    _arrow(ax, region, base, color=BIAS, dashed=True, lw=1.6, rad=0.18)

    _note(ax, 1.15,
          "论文的处理：加入州固定效应（48 个虚拟变量），吸收州层面不随时间变化的差异；\n"
          "并证明在 1950/1970 年（常规岗位还没大幅消失时）这个关系不成立 —— 排除长期趋势解释。\n"
          "软肋：本地需求冲击无法完全排除，论文靠「控制变量 + 时段对比」缓解而非消除。",
          edge=BIAS, face="#fff5f5", size=8.6)
    _legend(ax, [(False, PATH, "想估计的路径"), (True, BIAS, "需要关闭的旁路")], y=0.35)
    return _save(fig, "ad_01_identification.png")


def figure_final() -> Path:
    """Figure 3: the specification actually estimated -- Table 3 row 4."""

    fig, ax = _figure(width=10.2, height=7.0)
    _title(ax, "图 3：最终模型的因果图（表 3 第 (4) 行）",
           "OLS · 2,166 obs · [aw=timepwt48] · cluster(statefip) · 51 个参数")

    treat = Node(ax, 2.30, 3.65, "常规任务密集度", sub="l_sh_routine33a（1990 基期）",
                 width=3.0, height=0.95)
    out = Node(ax, 7.20, 3.65, "常规就业份额变化", sub="d_sh_routine33a", width=2.8, height=0.95)
    _arrow(ax, treat, out, color=PATH, lw=3.2)
    # Labels sit in the clear band ABOVE the arrow; below it there is nothing but box
    # borders and an earlier build put them straight across both.
    ax.text(4.80, 4.32, "−0.2541", ha="center", fontsize=11.5, fontweight="bold",
            color=PATH, zorder=7,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.97))
    ax.text(4.80, 3.98, "(0.0226)", ha="center", fontsize=9.0, color=MUTED, zorder=7,
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.96))
    ax.text(4.80, 2.62, "作者日志：−0.2540723 (0.0225534)", ha="center", fontsize=8.2,
            color=MUTED, zorder=7,
            bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="none", alpha=0.94))

    dummies = Node(ax, 4.75, 5.45, "48 个州虚拟变量",
                   sub="对应论文 do 文件的 xi i.statefip，吸收州层面固定差异",
                   width=5.2, height=0.95, face=GOOD, edge=PATH, bold=False)
    _arrow(ax, dummies, treat, color=PATH, lw=1.7, dashed=True)
    _arrow(ax, dummies, out, color=PATH, lw=1.7, dashed=True)
    ax.text(1.55, 4.62, "控制基期产业结构", ha="center", fontsize=8.2, color=PATH, zorder=7,
            bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="none", alpha=0.95))
    ax.text(7.90, 4.62, "控制州层面趋势", ha="center", fontsize=8.2, color=PATH, zorder=7,
            bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="none", alpha=0.95))

    periods = Node(ax, 8.75, 2.05, "时期虚拟变量", sub="t2、t3", width=2.2, height=0.85,
                   face=GOOD, edge=PATH, bold=False)
    _arrow(ax, periods, out, color=PATH, lw=1.6, dashed=True)

    _note(ax, 0.55,
          "51 个参数 = 1 个处理变量 + 2 个时期虚拟变量 + 48 个州虚拟变量。\n"
          "本次复现把 51 个参数全部逐个比对：最大相对差 2.9e-13。",
          edge=INK, size=8.6)
    return _save(fig, "ad_02_final.png")


FIGURES = (
    ("ad_00_polarization", figure_polarization),
    ("ad_01_identification", figure_identification),
    ("ad_02_final", figure_final),
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
