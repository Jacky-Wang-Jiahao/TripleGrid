from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from math import gcd
from typing import Dict, List


class Color(Enum):
    """The three circle families on the torus."""

    RED = auto()
    BLUE = auto()
    GREEN = auto()


@dataclass(frozen=True, slots=True)
class Vertex:
    """
    A lattice vertex in A₂ coordinates.

    Parameters
    ----------
    x
        First lattice coordinate.
    y
        Second lattice coordinate.
    """

    x: int
    y: int


@dataclass(slots=True)
class Background:
    """
    Finite quotient of the A₂ lattice determined by (m, n, p).

    The fundamental domain is

        0 <= x < n+p
        0 <= y < m+n
        x-y < p
        y-x <= m
    """

    m: int
    n: int
    p: int

    vertices: list[Vertex] = field(init=False)

    cycles: dict[Color, dict[int, list[Vertex]]] = field(init=False)

    vertex_lookup: dict[tuple[int, int], Vertex] = field(init=False)

    vertex_circle_ids: dict[Vertex, dict[Color, int]] = field(init=False)

    R: int = field(init=False)
    B: int = field(init=False)
    G: int = field(init=False)

    def __post_init__(self) -> None:

        self._validate()

        self.R = gcd(self.n, self.p)
        self.B = gcd(self.m, self.n)
        self.G = gcd(self.m, self.p)

        self._build_vertices()

        self._build_cycles()

    def _validate(self) -> None:

        if self.m < 0 or self.n < 0 or self.p < 0 or self.m * self.n + self.m * self.p + self.n * self.p <=0:
            raise ValueError("m, n, p must all be positive.")

    def _in_domain(self, x: int, y: int) -> bool:

        return (
            0 <= x < self.n + self.p
            and 0 <= y < self.m + self.n
            and x - y < self.p
            and y - x <= self.m
        )

    def _build_vertices(self) -> None:

        vertices = []
        lookup = {}

        for y in range(self.m + self.n):
            for x in range(self.n + self.p):

                if self._in_domain(x, y):

                    v = Vertex(x, y)

                    vertices.append(v)

                    lookup[(x, y)] = v

        self.vertices = vertices

        self.vertex_lookup = lookup

    def _build_cycles(self) -> None:

        cycles = {
            Color.RED: {i: [] for i in range(self.R)},
            Color.BLUE: {i: [] for i in range(self.B)},
            Color.GREEN: {i: [] for i in range(self.G)},
        }

        ids = {}

        for v in self.vertices:

            r = v.x % self.R
            b = v.y % self.B
            g = (v.x - v.y) % self.G

            cycles[Color.RED][r].append(v)
            cycles[Color.BLUE][b].append(v)
            cycles[Color.GREEN][g].append(v)

            ids[v] = {
                Color.RED: r,
                Color.BLUE: b,
                Color.GREEN: g,
            }

        self.cycles = cycles
        self.vertex_circle_ids = ids

    def circle_id(self, vertex: Vertex, color: Color) -> int:
        """Return the circle containing the given vertex."""

        return self.vertex_circle_ids[vertex][color]

    def circle(self, color: Color, circle_id: int) -> list[Vertex]:
        """Return all vertices on a given circle."""

        return self.cycles[color][circle_id]