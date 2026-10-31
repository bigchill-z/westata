"""Draw the causal diagrams behind ADH (2013) so the identification is visible.

Why a diagram at all
--------------------
The paper's argument is a chain of *exclusions*: something confounds the naive
regression, something else confounds the fixed-effects regression, and the instrument is
chosen precisely because it is not downstream of either.  Prose makes a reader hold four
variables and two failure modes in their head at once; a DAG makes the same content
checkable at a glance, because "is there a path from A to B that does not pass through the
arrow I care about?" is answered by looking rather than by reasoning.

Layout rules (learned by rendering these figures and looking at them)
---------------------------------------------------------------------
* A fixed coordinate frame: x in [0, 10], y in [0, 7].  The title band occupies
  y >= 6.2 and **no node may enter it** -- an earlier version placed a node at y=5.05 in
  a 6-high frame and the sub-title rendered behind it.
* Arrows are anchored to node *edges*, not centres, and the arrow function takes the box
  size so the anchor is exact.  `shrinkA/shrinkB` alone is unreliable once boxes have
  different sizes.
* A curved arrow that would cross a box is routed with an explicit detour point instead
  of being given a larger radius and hoped for.
* Labels sit in the gap between two boxes on a white cushion; they never overlap a box
  border, because the cushion is drawn *after* the arrow and *before* the next box.

Design rules for the content
----------------------------
* **Dashed red = a path that must be closed for the estimate to be causal; solid green =
  the path the estimate uses.**  Dashed-vs-solid survives greyscale and colour-blind
  readers, so colour is never the only encoding.
* Biased coefficients appear on the figure where the argument cites them (-0.171 for
  OLS, -0.596 for 2SLS): the size of that gap *is* the paper's evidence that the
  instrument is doing work.

Run::

    .venv/Scripts/python.exe replication/papers/adh2013/make_dags.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on the replication host

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# Frame constants: every figure uses the same frame, so a node placed at y=4.6 is safe
# in all of them and the title band can never be invaded.
X_MAX, Y_MAX = 10.0, 7.0
TITLE_Y, SUBTITLE_Y = 6.82, 6.42
NODE_CEILING = 6.02  # nothing above this line (sub-title starts at 6.48)

INK = "#1f2933"
PATH = "#0b6e4f"
BIAS = "#c1121f"
MUTED = "#6b7785"
BOX = "#eef2f7"
GOOD = "#eef7f1"
BAD = "#fdf0f0"


def _figure(width: float = 9.6, height: float = 6.2):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, X_MAX)
    ax.set_ylim(0, Y_MAX)
    ax.axis("off")
    return fig, ax


class Node:
    """A rounded box that knows its own geometry, so arrows can anchor to its edge."""

    def __init__(self, ax, x, y, label, *, sub="", width=2.0, height=0.9, face=BOX, edge=INK, bold=True):
        if y + height / 2 > NODE_CEILING:
            raise ValueError(
                f"node {label!r} at y={y} (h={height}) reaches into the title band "
                f"(ceiling {NODE_CEILING}); move it down"
            )
        self.x, self.y, self.w, self.h = x, y, width, height
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x - width / 2, y - height / 2),
                width,
                height,
                boxstyle="round,pad=0.05,rounding_size=0.12",
                linewidth=1.7,
                edgecolor=edge,
                facecolor=face,
                zorder=3,
            )
        )
        ax.text(
            x, y + (0.13 if sub else 0), label,
            ha="center", va="center", fontsize=10.4, color=INK,
            fontweight="bold" if bold else "normal", zorder=4,
        )
        if sub:
            ax.text(x, y - 0.19, sub, ha="center", va="center", fontsize=8.0, color=MUTED, zorder=4)

    def edge(self, toward, pad=0.10):
        """Point on this box's border along the ray to ``toward``."""

        dx, dy = toward[0] - self.x, toward[1] - self.y
        if dx == 0 and dy == 0:
            return (self.x, self.y)
        # Scale the direction until it hits the rectangle's half-extent.
        half_w, half_h = self.w / 2 + pad, self.h / 2 + pad
        sx = half_w / abs(dx) if dx else float("inf")
        sy = half_h / abs(dy) if dy else float("inf")
        s = min(sx, sy)
        return (self.x + dx * s, self.y + dy * s)


def _arrow(ax, src: Node, dst: Node, *, color=INK, lw=1.9, dashed=False,
           via: tuple[float, float] | None = None, rad: float = 0.0,
           label="", label_at=None, label_color=None):
    """Arrow from ``src`` border to ``dst`` border, optionally through a detour point."""

    points = [src.edge(via or (dst.x, dst.y))]
    if via:
        points.append(via)
        points.append(dst.edge(via))
    else:
        points.append(dst.edge((src.x, src.y)))

    if len(points) == 2:
        patch = FancyArrowPatch(
            points[0], points[1], arrowstyle="-|>", mutation_scale=15, linewidth=lw,
            color=color, linestyle=(0, (5, 3)) if dashed else "solid",
            connectionstyle=f"arc3,rad={rad}", zorder=2,
        )
        ax.add_patch(patch)
    else:
        # Draw the two legs as a polyline, then a small head at the end.
        ax.plot(
            [p[0] for p in points[:-1]], [p[1] for p in points[:-1]],
            color=color, lw=lw, linestyle=(0, (5, 3)) if dashed else "solid", zorder=2,
            solid_capstyle="round",
        )
        ax.add_patch(
            FancyArrowPatch(
                points[-2], points[-1], arrowstyle="-|>", mutation_scale=15,
                linewidth=lw, color=color, zorder=2,
            )
        )

    if label:
        lx, ly = label_at if label_at else ((points[0][0] + points[-1][0]) / 2,
                                            (points[0][1] + points[-1][1]) / 2)
        ax.text(
            lx, ly, label, ha="center", va="center", fontsize=8.6,
            color=label_color or color,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.95),
            zorder=6,
        )


def _title(ax, text, sub=""):
    ax.text(X_MAX / 2, TITLE_Y, text, ha="center", va="top",
            fontsize=13.5, fontweight="bold", color=INK)
    if sub:
        ax.text(X_MAX / 2, SUBTITLE_Y, sub, ha="center", va="top",
                fontsize=9.3, color=MUTED)


def _legend(ax, items, y=0.34):
    """Legend row; each item is (dashed, colour, label).  Widths are measured, not guessed."""

    total = sum(0.62 + 0.108 * len(label) for _, _, label in items)
    x = (X_MAX - total) / 2
    for dashed, color, label in items:
        ax.plot([x, x + 0.5], [y, y], color=color, lw=2.2,
                linestyle=(0, (5, 3)) if dashed else "solid")
        ax.text(x + 0.62, y, label, va="center", fontsize=8.8, color=INK)
        x += 0.62 + 0.108 * len(label) + 0.7


def _note(ax, y, text, *, edge=MUTED, face="#f7f9fc", size=9.3):
    ax.text(X_MAX / 2, y, text, ha="center", va="center", fontsize=size, color=INK,
            bbox=dict(boxstyle="round,pad=0.42", facecolor=face, edgecolor=edge, linewidth=1.1),
            zorder=5)


# --------------------------------------------------------------------------- figures
def figure_naive() -> Path:
    """Figure 1: the naive regression and its two backdoors."""

    fig, ax = _figure()
    _title(ax, "图 1：朴素回归为什么不行",
           "把「本地进口暴露」直接对「就业变化」回归，会打开两条后门")

    expo = Node(ax, 3.15, 3.45, "中国进口暴露", sub="d_tradeusch_pw")
    out = Node(ax, 6.85, 3.45, "制造业就业变化", sub="d_sh_empl_mfg")

    # Classic two-confounder "bow-tie": each confounder sits directly above one of the
    # two variables, so the vertical arrows are clean and only the two crossing diagonals
    # overlap.  They are given opposite curvature so a reader cannot mistake the crossing
    # for a junction.
    demand = Node(ax, 3.15, 5.45, "本地需求冲击", sub="本地制造业本就衰退",
                  face=BAD, edge=BIAS, bold=False)
    tech = Node(ax, 6.85, 5.45, "技术进步 / 自动化", sub="全国性冲击",
                face=BAD, edge=BIAS, bold=False)

    _arrow(ax, expo, out, color=PATH, lw=2.5, label="想估计的因果效应", label_at=(5.0, 3.12))
    _arrow(ax, demand, expo, color=BIAS, dashed=True, lw=2.0,
           label="反向因果", label_at=(2.28, 4.40))
    _arrow(ax, tech, out, color=BIAS, dashed=True, lw=2.0)
    _arrow(ax, demand, out, color=BIAS, dashed=True, lw=1.8, rad=0.22)
    _arrow(ax, tech, expo, color=BIAS, dashed=True, lw=1.8, rad=0.22)

    _note(ax, 2.15,
          "两条虚线都让「暴露」与「就业」同时变动，但都不是我们想找的那条因果路径\n"
          "→ 直接回归得到的系数有偏，且偏误方向已知：被压向零（低估）",
          edge=BIAS, face="#fff5f5")

    _legend(ax, [(False, PATH, "想估计的因果路径"), (True, BIAS, "会污染估计的旁路")])
    return _save(fig, "dag_00_naive.png")


def figure_controls() -> Path:
    """Figure 2: what fixed effects plus controls close, and what stays open."""

    fig, ax = _figure()
    _title(ax, "图 2：换成 722 个通勤区 + 加控制变量，关掉了什么",
           "Table 3 从列 (1) 到列 (6)：每加一组控制，都在切断一条特定旁路")

    expo = Node(ax, 3.60, 3.95, "中国进口暴露", sub="d_tradeusch_pw")
    out = Node(ax, 7.05, 3.95, "制造业就业变化", sub="d_sh_empl_mfg")

    # Controls sit above and reach both variables with short vertical arrows: no long
    # routed lines, nothing crossing a label.
    closed = Node(ax, 5.10, 5.45, "区域趋势 / 人口结构 / 任务构成",
                  sub="Census division 虚拟变量、学历、外国出生、常规任务等",
                  width=5.0, height=0.95, face=GOOD, edge=PATH, bold=False)

    # The confounder controls cannot close is placed *below* exposure, so its arrow is a
    # short diagonal that crosses nothing.  An earlier version routed it around the frame
    # and the line cut through the coefficient label twice.
    still = Node(ax, 1.75, 2.15, "反向因果",
                 sub="本地需求「既降制造业、又推高进口」",
                 width=2.5, height=0.95, face=BAD, edge=BIAS, bold=False)

    _arrow(ax, expo, out, color=PATH, lw=2.6)
    ax.text(5.30, 3.62, "−0.746 → −0.596", ha="center", va="center", fontsize=9.2,
            color=PATH, bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                  edgecolor="none", alpha=0.96), zorder=6)

    _arrow(ax, closed, expo, color=PATH, dashed=True)
    _arrow(ax, closed, out, color=PATH, dashed=True)
    ax.text(3.35, 4.72, "已切断", ha="center", fontsize=8.7, color=PATH, zorder=6,
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.96))
    ax.text(6.85, 4.72, "已切断", ha="center", fontsize=8.7, color=PATH, zorder=6,
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.96))

    _arrow(ax, still, expo, color=BIAS, dashed=True, lw=2.2)
    ax.text(1.95, 3.32, "关不掉", ha="center", fontsize=8.7, color=BIAS, zorder=6,
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.96))

    _note(ax, 1.05,
          "加控制变量让系数从 −0.746 衰减到 −0.596：其中一部分影响其实来自\n"
          "「本来制造业就多的地方自然降得多」，那并不是中国冲击。\n"
          "但反向因果控制变量关不掉 → 还必须有工具变量",
          edge=MUTED)

    _legend(ax, [(False, PATH, "主效应"), (True, PATH, "已切断的旁路"), (True, BIAS, "仍需靠 IV 处理")])
    return _save(fig, "dag_01_controls.png")


def figure_iv() -> Path:
    """Figure 3: the instrument, and which path it excludes."""

    fig, ax = _figure(height=6.6)
    _title(ax, "图 3：工具变量怎么把「中国的供给冲击」单独拎出来",
           "用「其他 8 个富国从中国进口」当工具：它们的需求与某个美国城市的就业无关")

    supply = Node(ax, 2.35, 5.40, "中国的供给冲击",
                  sub="生产率↑、入世、关税↓、基建改善",
                  width=3.1, height=1.0, face=GOOD, edge=PATH, bold=False)
    iv = Node(ax, 1.85, 3.45, "工具变量", sub="d_tradeotch_pw_lag", width=2.3)
    expo = Node(ax, 5.20, 3.45, "中国进口暴露", sub="d_tradeusch_pw", width=2.3)
    out = Node(ax, 8.55, 3.45, "制造业就业变化", sub="d_sh_empl_mfg", width=2.3)

    usdemand = Node(ax, 8.55, 5.40, "美国本地需求", sub="会同时推高进口与就业",
                    width=2.3, height=1.0, face=BAD, edge=BIAS, bold=False)

    _arrow(ax, supply, iv, color=PATH, lw=2.4)
    _arrow(ax, supply, expo, color=PATH, lw=2.4)
    ax.text(1.15, 4.45, "① 主要驱动", ha="center", fontsize=8.8, color=PATH, zorder=6,
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.96))
    ax.text(4.10, 4.55, "② 也驱动", ha="center", fontsize=8.8, color=PATH, zorder=6,
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.96))

    _arrow(ax, iv, expo, color=PATH, lw=2.8)
    _arrow(ax, expo, out, color=PATH, lw=2.8)
    ax.text(3.52, 3.05, "③ 第一阶段 0.63–0.79", ha="center", fontsize=8.8, color=PATH, zorder=6,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.96))
    ax.text(6.90, 3.05, "④ 结构方程 −0.596", ha="center", fontsize=8.8, color=PATH, zorder=6,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.96))

    # The path the instrument must exclude is US demand -> exposure (demand pushes imports
    # up independently of Chinese supply).  Drawn as a short arc through the free space
    # between the two upper boxes: routing it underneath would cross the structural
    # equation label, which an earlier version did.
    _arrow(ax, usdemand, out, color=BIAS, dashed=True, lw=2.0)
    _arrow(ax, usdemand, expo, color=BIAS, dashed=True, lw=2.0, via=(6.55, 5.40))
    ax.text(6.55, 4.98, "被排除", ha="center", fontsize=8.6, color=BIAS, zorder=6,
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.96))

    _note(ax, 1.05,
          "关键假设（排他性）：工具变量只能通过「中国进口暴露」影响美国就业。\n"
          "中国对德国的出口涨了，不是因为底特律的汽车需求变了 —— 这条旁路因此被切断。",
          edge=MUTED)

    _legend(ax, [(False, PATH, "工具变量利用的路径"), (True, BIAS, "被排除的路径")])
    return _save(fig, "dag_02_iv.png")


def figure_full() -> Path:
    """Figure 4: the whole design in one picture."""

    fig, ax = _figure(height=6.6)
    _title(ax, "图 5：整套识别策略合起来",
           "IV 负责切断反向因果，控制变量负责吸收区域趋势；两者一起才有 −0.596")

    supply = Node(ax, 1.75, 5.40, "中国的供给冲击", sub="外生：不由美国本地决定",
                  width=2.7, height=0.95, face=GOOD, edge=PATH, bold=False)
    controls = Node(ax, 5.00, 5.40, "控制变量组", sub="区域 / 人口 / 任务构成",
                    width=2.4, height=0.95, face=GOOD, edge=PATH, bold=False)
    resid = Node(ax, 8.55, 5.40, "本地需求等未观测因素", sub="被 IV 排除在外的旁路",
                 width=2.9, height=0.95, face=BAD, edge=BIAS, bold=False)

    iv = Node(ax, 1.75, 3.45, "其他 8 富国进口", sub="d_tradeotch_pw_lag", width=2.4)
    expo = Node(ax, 5.00, 3.45, "中国进口暴露", sub="d_tradeusch_pw", width=2.3)
    out = Node(ax, 8.55, 3.45, "制造业就业变化", sub="d_sh_empl_mfg", width=2.3)

    _arrow(ax, supply, iv, color=PATH, lw=2.3)
    _arrow(ax, iv, expo, color=PATH, lw=2.7)
    ax.text(3.38, 3.10, "第一阶段", ha="center", fontsize=8.8, color=PATH, zorder=6,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.96))

    _arrow(ax, expo, out, color=PATH, lw=2.8)
    # The coefficient label sits in the free band *below* the arrow (y = 2.95), clear of
    # both boxes' borders and of the routed bias path.  Above the arrow there is no room:
    # the "吸收" label and the controls' arrow are already there.
    ax.text(6.78, 2.55, "−0.596*** (0.099)", ha="center", fontsize=9.2, color=PATH, zorder=7,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="none", alpha=0.98))

    _arrow(ax, controls, expo, color=PATH, lw=1.9, dashed=True)
    _arrow(ax, controls, out, color=PATH, lw=1.9, dashed=True)
    ax.text(3.42, 4.60, "吸收", ha="center", fontsize=8.6, color=PATH, zorder=6,
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.96))
    ax.text(6.70, 4.42, "吸收", ha="center", fontsize=8.6, color=PATH, zorder=6,
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.96))

    _arrow(ax, resid, out, color=BIAS, dashed=True, lw=2.0)
    # Round the right-hand margin, level with nothing else, so the line never crosses the
    # coefficient label or the "吸收" label.
    _arrow(ax, resid, expo, color=BIAS, dashed=True, lw=1.8, via=(9.55, 1.75))
    ax.text(8.05, 1.75, "切断", ha="center", fontsize=8.6, color=BIAS, zorder=6,
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.96))

    _note(ax, 1.15,
          "结论：暴露度每上升 1,000 美元/工人，制造业就业占劳动年龄人口比例下降 0.596 个百分点。\n"
          "折算：解释了 1990s 33%、2000s 55%、全期 44% 的制造业就业下降。",
          edge=INK, size=9.4)
    return _save(fig, "dag_05_full.png")


def figure_comparison() -> Path:
    """Figure 5: the OLS/2SLS gap as replication evidence."""

    fig, ax = _figure(height=5.6)
    _title(ax, "图 4：OLS 与 2SLS 相差 3.5 倍——这正是识别策略在起作用的证据",
           "同一个模型、同一组控制变量、同一份数据，只因估计量不同")

    for cx, name, coef, se, note_text, color in (
        (2.55, "OLS", "−0.171", "(0.028)", "衰减偏误：测量误差 + 联立性", BIAS),
        (7.15, "2SLS", "−0.596", "(0.099)", "论文首选估计", PATH),
    ):
        ax.text(cx, 5.35, name, ha="center", fontsize=12, fontweight="bold", color=MUTED)
        ax.text(cx, 4.62, coef, ha="center", fontsize=30, fontweight="bold", color=color)
        ax.text(cx, 4.08, se, ha="center", fontsize=9.8, color=MUTED)
        ax.text(cx, 3.62, note_text, ha="center", fontsize=8.8, color=color)

    ax.annotate("", xy=(3.35, 4.62), xytext=(6.35, 4.62),
                arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=1.6, mutation_scale=13))
    ax.text(4.85, 4.95, "× 3.5", ha="center", fontsize=13, fontweight="bold", color=INK)

    _note(ax, 2.45,
          "为什么这对复现很重要：\n"
          "如果一个复现「不小心」用了 OLS，它不可能同时命中 −0.171 和 −0.596。\n"
          "两个数同时吻合，才说明跑的是论文指定的那个估计量，而不是碰巧接近。",
          edge="#b8860b", face="#fffaf0", size=9.4)

    ax.plot([0.8, 9.2], [1.5, 1.5], color=MUTED, lw=0.9, linestyle=(0, (2, 2)))
    ax.text(5.0, 1.05, "本次复现：westata 同时命中两者（−0.171128 与 −0.596360），pystata 亦然",
            ha="center", fontsize=9.3, color=PATH, fontweight="bold")
    return _save(fig, "dag_04_comparison.png")


def figure_final_model() -> Path:
    """Figure 6: the estimation itself -- Table 3 column 6, every variable named.

    The earlier figures explain the *idea*.  This one is the specification that was
    actually estimated, so a reader can check it line by line against ``specs.py``:
    1 endogenous regressor, 1 instrument, 15 exogenous controls in four named groups,
    ``[aw=timepwt48]``, ``cluster(statefip)``, N = 1,444.

    Layout decision, after two failed attempts: the control groups are stacked in a
    LEFT-HAND column and the causal spine sits in a RIGHT-HAND column.  Trying to place
    controls *above* the spine forced their arrows to cross the whole figure and pushed
    the wide region block into its neighbours.  A column layout keeps every arrow short
    and every box unobstructed.
    """

    fig, ax = _figure(width=10.6, height=7.6)
    ax.set_xlim(0, X_MAX)
    ax.set_ylim(0, Y_MAX)
    _title(ax, "图 6：最终模型的因果图（Table 3 第 6 列，论文首选规格）",
           "2SLS · N = 1,444（722 CZ × 2 期）· [aw=timepwt48] · cluster(statefip) · 17 个系数")

    # ---- right column: the causal spine ------------------------------------
    iv = Node(ax, 6.55, 4.55, "其他 8 富国进口", sub="d_tradeotch_pw_lag", width=2.5, height=0.92)
    expo = Node(ax, 6.55, 2.95, "中国进口暴露", sub="d_tradeusch_pw", width=2.5, height=0.92)
    out = Node(ax, 8.25, 1.42, "制造业就业变化", sub="d_sh_empl_mfg", width=2.3, height=0.92)

    supply = Node(ax, 7.55, 5.50, "中国的供给冲击", sub="外生：生产率↑ · 入世 · 关税↓",
                  width=2.6, height=0.86, face=GOOD, edge=PATH, bold=False)

    _arrow(ax, supply, iv, color=PATH, lw=2.2)
    _arrow(ax, iv, expo, color=PATH, lw=3.0)
    _arrow(ax, expo, out, color=PATH, lw=3.2)

    ax.text(5.05, 3.78, "第一阶段", ha="center", fontsize=9.4, color=PATH, fontweight="bold",
            zorder=7, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.97))
    ax.text(5.05, 3.46, "0.631  (0.091)", ha="center", fontsize=8.8, color=PATH, zorder=7,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.96))
    # The label goes into the empty band between the instrument arrow and the outcome box.
    # Every earlier attempt placed it over a box border or over a variable name; this band
    # (y 2.55-2.95, x 7.6-9.6) is bounded by the exposure box above and the outcome box
    # below, and nothing else is drawn there.
    ax.text(8.05, 3.16, "结构方程", ha="center", fontsize=9.0, color=PATH, fontweight="bold",
            zorder=7, bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                                edgecolor="none", alpha=0.98))
    ax.text(8.05, 2.86, "−0.596***  (0.099)", ha="center", fontsize=8.8, color=PATH, zorder=7,
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.98))

    # ---- left column: the 15 controls, in the four groups the paper names ---
    ax.text(2.05, 5.92, "外生控制变量（15 个，分 4 组）", ha="center", fontsize=9.6,
            color=INK, fontweight="bold")

    g_period = Node(ax, 2.05, 5.42, "① 时期虚拟变量", sub="t2",
                    width=3.3, height=0.78, face=GOOD, edge=PATH, bold=False)
    g_share = Node(ax, 2.05, 4.52, "② 基期制造业就业份额", sub="l_shind_manuf_cbp",
                   width=3.3, height=0.78, face=GOOD, edge=PATH, bold=False)
    g_region = Node(ax, 2.05, 3.42, "③ 区域趋势：8 个 Census division 虚拟变量",
                    sub="reg_midatl / encen / wncen / satl\nescen / wscen / mount / pacif",
                    width=3.9, height=0.98, face=GOOD, edge=PATH, bold=False)
    g_demo = Node(ax, 2.05, 2.28, "④ 人口结构", sub="l_sh_popedu_c　l_sh_popfborn　l_sh_empl_f",
                  width=3.9, height=0.78, face=GOOD, edge=PATH, bold=False)
    g_task = Node(ax, 2.05, 1.38, "⑤ 任务构成", sub="l_sh_routine33　l_task_outsource",
                  width=3.9, height=0.78, face=GOOD, edge=PATH, bold=False)

    # Each group reaches BOTH 暴露 and 就业: a control that only entered one side would
    # not be a control in this design.  Arrows are drawn horizontally into a common
    # vertical bus so they do not fan across the figure.
    bus_x = 4.30
    ax.plot([bus_x, bus_x], [1.38, 5.42], color=PATH, lw=1.4, linestyle=(0, (4, 3)), zorder=1)
    for box, y in ((g_period, 5.42), (g_share, 4.52), (g_region, 3.42),
                   (g_demo, 2.28), (g_task, 1.38)):
        ax.plot([2.05 + (box.w / 2 if hasattr(box, "w") else 1.95), bus_x], [y, y],
                color=PATH, lw=1.4, linestyle=(0, (4, 3)), zorder=1)
    # Two labelled exits: the bus feeds exposure and outcome separately, so a reader can
    # see that every control enters BOTH equations -- which is what "a control" means here.
    _arrow_bus(ax, (bus_x, 3.42), expo, color=PATH)
    _arrow_bus(ax, (bus_x, 1.62), out, color=PATH)
    ax.text(bus_x + 0.10, 2.02, "同一组控制同时\u8fdb入两个方程", ha="left", fontsize=7.8,
            color=PATH, zorder=7, bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                                            edgecolor="none", alpha=0.95))

    ax.text(bus_x + 0.08, 4.62, "吸收", ha="left", fontsize=8.2, color=PATH, zorder=7,
            bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="none", alpha=0.95))

    # ---- notes -------------------------------------------------------------
    ax.text(0.45, 0.62, "未画入：" + "本地需求等不可观测因素 —— 由工具变量排除（见图 3）",
            ha="left", fontsize=8.4, color=BIAS,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#fff5f5",
                      edgecolor=BIAS, linewidth=0.9))

    _note(ax, 0.30,
          "17 个系数 = 1 内生变量 + 1 工具变量（第一阶段）+ 15 外生控制　|　"
          "R² = 0.343　N = 1,444　48 个州聚类",
          edge=INK, size=8.6)
    return _save(fig, "dag_06_final_model.png")


def _arrow_bus(ax, start: tuple[float, float], dst: "Node", *, color=PATH) -> None:
    """Arrow from a point on the control bus into a node's left edge."""

    end = (dst.x - dst.w / 2 - 0.08, dst.y)
    ax.plot([start[0], end[0]], [start[1], end[1]],
            color=color, lw=1.4, linestyle=(0, (4, 3)), zorder=1)
    ax.add_patch(
        FancyArrowPatch(end, (end[0] + 0.01, end[1]), arrowstyle="-|>",
                        mutation_scale=13, linewidth=1.4, color=color, zorder=2)
    )


def _save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.18)
    plt.close(fig)
    return path


FIGURES = (
    ("dag_00_naive", figure_naive),
    ("dag_01_controls", figure_controls),
    ("dag_02_iv", figure_iv),
    ("dag_05_full", figure_full),
    ("dag_04_comparison", figure_comparison),
    ("dag_06_final_model", figure_final_model),
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
