"""A single place that builds figures for every `plot()` in the package.

THE DOUBLE-FIGURE BUG THIS SOLVES

In a Jupyter cell (including VS Code's, which uses the same inline backend), calling
``ws.ac_table(series).plot()`` showed TWO images. Two independent renderers fire:

  1. the inline backend draws every figure still REGISTERED in pyplot when the cell ends;
  2. the returned ``Figure`` is also displayed, because it is the cell's expression value.

Building with ``plt.subplots`` puts the figure in that registry, so both paths ran.

WHY THE FIX IS "DEREGISTER", NOT "BUILD A BARE Figure()"

A ``matplotlib.figure.Figure()`` constructed directly is never registered, which also fixes the
duplication -- but it comes with a ``FigureCanvasBase``, whose ``print_figure`` is not a real
renderer. The inline backend goes through the canvas, so an interactive session could end up with a
figure it cannot draw. ``plt.subplots`` gives a fully working canvas (``FigureCanvasAgg`` or the
backend's own); destroying only its *manager* removes it from the registry and leaves the canvas --
and therefore display, savefig and show -- intact.

``make_figure``/``make_axes``/``release`` centralise that, so every entry point behaves the same and a
future plotting function cannot reintroduce the bug by forgetting the step.
"""

from __future__ import annotations

from typing import Any

__all__ = ["make_axes", "make_figure", "release", "release_all"]


def _pyplot() -> Any:
    """Import pyplot lazily, so matplotlib stays an optional dependency."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - only without the extra
        raise ImportError(
            "westata plots need matplotlib: pip install 'westata[plot]'"
        ) from exc
    return plt


def release(figure: Any) -> Any:
    """Remove ``figure`` from pyplot's registry WITHOUT destroying it.

    Returns the same figure so callers can write ``return release(figure)``.
    """

    try:
        from matplotlib._pylab_helpers import Gcf
    except ImportError:  # pragma: no cover - very old matplotlib
        return figure
    manager = Gcf.get_fig_manager(figure.number) if hasattr(figure, "number") else None
    if manager is not None:
        Gcf.destroy(manager)
    return figure


def make_figure(*, figsize: Any = None) -> tuple[Any, Any]:
    """One figure plus one axis, created with a real canvas and then deregistered."""

    plt = _pyplot()
    figure, axis = plt.subplots(figsize=figsize)
    return release(figure), axis


def make_axes(rows: int, columns: int, *, figsize: Any = None) -> tuple[Any, list[Any]]:
    """A grid of axes as a flat list, deregistered the same way."""

    plt = _pyplot()
    figure, array = plt.subplots(rows, columns, figsize=figsize, squeeze=False)
    flat = [array[row][column] for row in range(rows) for column in range(columns)]
    return release(figure), flat


def release_all() -> None:
    """Close every registered figure -- for tests and for callers who want a clean slate."""

    _pyplot().close("all")
