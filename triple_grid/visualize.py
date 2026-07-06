from __future__ import annotations

import itertools
import math
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from .diagram import Diagram
from .geometry import Background, Color, Vertex

"""
Matplotlib visualization for dot diagrams on an :class:`~.geometry.Background`.

Geometric embedding
--------------------
``geometry.Background`` is purely combinatorial: a ``Vertex`` is an
integer pair ``(x, y)`` and the three circle colours are read off it as

    r(x, y) = x         mod R      (RED)
    b(x, y) = y         mod B      (BLUE)
    g(x, y) = (x - y)   mod G      (GREEN)

To draw this as an honest triangular lattice we fix the embedding

    e1 = (1, 0)
    e2 = (-1/2, sqrt(3)/2)                       angle(e1, e2) = 120 deg

    P(x, y) = x * e1 + y * e2  ,   (x, y) in R^2 .

Lines are straight and equally spaced.
    Fixing x = x0 and letting y range over R, the points P(x0, y) move
    by multiples of e2 as y increases: this is a straight line in
    direction e2. Fixing y = y0 similarly gives a straight line in
    direction e1. Fixing x - y = k, i.e. x = y + k, gives
    P(y + k, y) = k * e1 + y * (e1 + e2), a straight line in direction
    e1 + e2. So the RED, BLUE, GREEN circles of the background are
    drawn as three families of parallel straight lines, in directions
    e1, e2, e1 + e2 respectively, each family spaced one unit apart
    (immediate from linearity of P and integrality of the fibers).

Lines are at 60 degrees, i.e. this really is the triangular lattice.
    e1 . e2 = -1/2, so |e1| = |e2| = 1 and angle(e1, e2) = 120 deg.
    Also e1 + e2 = (1/2, sqrt(3)/2), so |e1 + e2|^2 = 1/4 + 3/4 = 1,
    and one checks angle(e1, e1+e2) = angle(e1+e2, e2) = 60 deg. Hence
    e1, e1 + e2, e2 point at 0 deg, 60 deg, 120 deg: three unit
    lattice directions spaced exactly 60 degrees apart. The vertices
    {P(x, y) : (x, y) in Z^2} are therefore exactly the vertices of
    the standard triangular lattice, and RED/BLUE/GREEN edges
    (x, y)-(x, y+1), (x, y)-(x+1, y), (x, y)-(x+1, y+1) are exactly
    its three families of unit edges.

Where to draw a dot.
    Putting a dot for Vertex(x, y) at the lattice point P(x, y) itself
    would place it exactly on a crossing of a red, a blue and a green
    line (P(x,y) lies on red line x=x, blue line y=y and green line
    x-y=x-y simultaneously) -- visually ambiguous, and not what was
    asked for. Instead every triangular *face* of the lattice is
    bounded by exactly one red, one blue and one green edge (a
    "red-green-blue triangle"), and the map

        Vertex(x, y)  |->  face with corners P(x,y), P(x,y+1), P(x+1,y+1)

    is a bijection from vertices to the "upward" faces of the lattice
    (edges: (x,y)-(x,y+1) on red line x=x; (x,y+1)-(x+1,y+1) on blue
    line y=y+1; (x,y)-(x+1,y+1) on green line x-y=x-y -- one edge of
    each colour, as claimed). Its centroid,

        center(x, y) = P(x + 1/3, y + 2/3) ,

    is where the dot for Vertex(x, y) is drawn. Distinct vertices give
    distinct faces, so distinct, non-overlapping dot positions.
"""

# ----------------------------------------------------------------------
# Embedding
# ----------------------------------------------------------------------

_E1 = (1.0, 0.0)
_E2 = (-0.5, math.sqrt(3.0) / 2.0)


def _lattice_point(x: float, y: float) -> tuple[float, float]:
    """P(x, y) = x * e1 + y * e2, extended to real x, y."""

    return (x * _E1[0] + y * _E2[0], x * _E1[1] + y * _E2[1])


def vertex_center(vertex: Vertex) -> tuple[float, float]:
    """
    Euclidean point at which ``vertex`` is drawn: the centroid of its
    associated red-blue-green triangular face (see module docstring).
    """

    return _lattice_point(vertex.x + 1.0 / 3.0, vertex.y + 2.0 / 3.0)


# One neighbour offset per colour: the far endpoint of the edge, of
# that colour, leaving (x, y) "forwards", so every edge is emitted once.
_NEIGHBOR_OFFSET: dict[Color, tuple[int, int]] = {
    Color.RED: (0, -1),
    Color.BLUE: (1, 0),
    Color.GREEN: (1, 1),
}

_COLOR_HEX: dict[Color, str] = {
    Color.RED: "#d62728",
    Color.BLUE: "#1f77b4",
    Color.GREEN: "#2ca02c",
}

_DOT_COLOR = "#1a1a1a"
_EMPTY_FACE_COLOR = "#bdbdbd"


# ----------------------------------------------------------------------
# Drawing primitives
# ----------------------------------------------------------------------


def _draw_grid(
    ax: Axes,
    background: Background,
    diagram: Optional[Diagram],
    *,
    highlight_complete: bool,
) -> None:
    """Draw every RED/BLUE/GREEN edge between vertices of the background."""

    lookup = background.vertex_lookup
    ids = background.vertex_circle_ids

    for v in background.vertices:
        for color, (dx, dy) in _NEIGHBOR_OFFSET.items():
            t = 1 if dy == -1 else 0
            p0 = _lattice_point(v.x + t, v.y + t)
            p1 = _lattice_point(v.x + dx + t, v.y + dy + t)
            ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                color=_COLOR_HEX[color],
                linewidth=1.0,
                alpha=1,
                solid_capstyle="round",
                zorder=1,
            )

    p0 = _lattice_point(0, 0)
    p1 = _lattice_point(0, background.m)
    ax.plot(
        [p0[0], p1[0]],
        [p0[1], p1[1]],
        color=_COLOR_HEX[Color.RED],
        linewidth=1.0,
        alpha=1,
        solid_capstyle="round",
        zorder=1,
    )
    
    p0 = _lattice_point(background.n, background.m + background.n)
    p1 = _lattice_point(background.n + background.p, background.m + background.n)
    ax.plot(
        [p0[0], p1[0]],
        [p0[1], p1[1]],
        color=_COLOR_HEX[Color.BLUE],
        linewidth=1.0,
        alpha=1,
        solid_capstyle="round",
        zorder=1,
    )
    
    p0 = _lattice_point(background.p, 0)
    p1 = _lattice_point(background.n + background.p, background.n)
    ax.plot(
        [p0[0], p1[0]],
        [p0[1], p1[1]],
        color=_COLOR_HEX[Color.GREEN],
        linewidth=1.0,
        alpha=1,
        solid_capstyle="round",
        zorder=1,
    )


def _draw_dots(ax: Axes, background: Background, diagram: Diagram) -> None:
    """Draw a filled dot at every occupied face, a faint mark elsewhere."""

    for v in background.vertices:
        cx, cy = vertex_center(v)
        if diagram.contains(v):
            ax.add_patch(
                Circle(
                    (cx + 1/2, cy - 1/3),
                    radius=0.12,
                    facecolor=_DOT_COLOR,
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=3,
                )
            )
        else:
            ax.add_patch(
                Circle(
                    (cx + 1/2, cy - 1/3),
                    radius=0.045,
                    facecolor=_EMPTY_FACE_COLOR,
                    edgecolor="none",
                    zorder=2,
                )
            )


def _legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=_COLOR_HEX[c], lw=2.2, label=c.name.capitalize())
        for c in Color
    ]


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def plot_diagram(
    diagram: Diagram,
    *,
    ax: Optional[Axes] = None,
    highlight_complete: bool = True,
    show_legend: bool = True,
    title: Optional[str] = None,
) -> Axes:
    """
    Draw a single ``Diagram`` on an A2 triangular grid.

    Parameters
    ----------
    diagram
        The diagram to draw.
    ax
        Axes to draw into. A new figure is created if omitted.
    highlight_complete
        If True, circles with exactly two dots are drawn with a
        thicker, fully opaque line (see ``Diagram.circle_is_complete``).
    show_legend
        If True, add a RED/BLUE/GREEN colour legend.
    title
        Axes title. Defaults to the dot count.
    """

    background = diagram.background

    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 6.0))

    _draw_grid(ax, background, diagram, highlight_complete=highlight_complete)
    _draw_dots(ax, background, diagram)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title if title is not None else f"{len(diagram)} dots", fontsize=11)

    if show_legend:
        ax.legend(
            handles=_legend_handles(),
            loc="upper right",
            fontsize=8,
            framealpha=0.9,
        )

    return ax


def plot_diagrams(
    diagrams: Iterable[Diagram] | Sequence[Diagram],
    *,
    max_diagrams: int = 12,
    ncols: int = 4,
    highlight_complete: bool = True,
) -> Figure:
    """
    Draw many diagrams (e.g. straight from ``enumerate_diagrams``) as a
    grid of subplots, at most ``max_diagrams`` of them.
    """

    diagrams = list(itertools.islice(diagrams, max_diagrams))
    n = len(diagrams)
    if n == 0:
        raise ValueError("No diagrams to plot.")

    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.3 * ncols, 3.3 * nrows), squeeze=False
    )

    for i, diagram in enumerate(diagrams):
        ax = axes[i // ncols][i % ncols]
        plot_diagram(
            diagram,
            ax=ax,
            highlight_complete=highlight_complete,
            show_legend=(i == 0),
            title=f"#{i}  ({len(diagram)} dots)",
        )

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.tight_layout()
    return fig