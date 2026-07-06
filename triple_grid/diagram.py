from __future__ import annotations

from dataclasses import dataclass, field

from .geometry import Background, Color, Vertex


@dataclass(slots=True)
class Diagram:
    """
    Dot diagram on a fixed Background.

    A diagram is a subset of the vertices together with cached
    dot counts on every circle. Maintaining these counts allows
    O(1) legality checks during backtracking.
    """

    background: Background

    dots: set[Vertex] = field(default_factory=set)

    circle_counts: dict[Color, dict[int, int]] = field(init=False)

    def __post_init__(self) -> None:
        """Initialize circle counts from the initial dot set."""

        self.circle_counts = {
            color: {
                cid: 0
                for cid in self.background.cycles[color]
            }
            for color in Color
        }

        for vertex in self.dots:
            self._increment(vertex)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _increment(self, vertex: Vertex) -> None:
        """Increase the counts of the three circles containing vertex."""

        ids = self.background.vertex_circle_ids[vertex]

        for color, cid in ids.items():
            self.circle_counts[color][cid] += 1

    def _decrement(self, vertex: Vertex) -> None:
        """Decrease the counts of the three circles containing vertex."""

        ids = self.background.vertex_circle_ids[vertex]

        for color, cid in ids.items():
            self.circle_counts[color][cid] -= 1

    # ------------------------------------------------------------------
    # Basic operations
    # ------------------------------------------------------------------

    def contains(self, vertex: Vertex) -> bool:
        """Return True iff the vertex currently contains a dot."""

        return vertex in self.dots

    def add(self, vertex: Vertex) -> None:
        """
        Add a dot.

        Raises
        ------
        ValueError
            If the vertex already contains a dot.
        """

        if vertex in self.dots:
            raise ValueError("Vertex already contains a dot.")

        self.dots.add(vertex)
        self._increment(vertex)

    def remove(self, vertex: Vertex) -> None:
        """
        Remove a dot.

        Raises
        ------
        KeyError
            If the vertex is not occupied.
        """

        self.dots.remove(vertex)
        self._decrement(vertex)

    def clear(self) -> None:
        """Remove all dots."""

        self.dots.clear()

        for color in Color:
            for cid in self.circle_counts[color]:
                self.circle_counts[color][cid] = 0

    # ------------------------------------------------------------------
    # Circle queries
    # ------------------------------------------------------------------

    def count(self, color: Color, circle_id: int) -> int:
        """Return the number of dots on a circle."""

        return self.circle_counts[color][circle_id]

    def circle_is_complete(self, color: Color, circle_id: int) -> bool:
        """Return True iff the circle contains exactly two dots."""

        return self.count(color, circle_id) == 2

    def circle_is_empty(self, color: Color, circle_id: int) -> bool:
        """Return True iff the circle contains no dots."""

        return self.count(color, circle_id) == 0

    # ------------------------------------------------------------------
    # Legality
    # ------------------------------------------------------------------

    def can_add(self, vertex: Vertex) -> bool:
        """
        Return whether vertex can be added without exceeding
        two dots on any circle.
        """

        if vertex in self.dots:
            return False

        ids = self.background.vertex_circle_ids[vertex]

        return all(
            self.circle_counts[color][cid] < 2
            for color, cid in ids.items()
        )

    def is_valid(self) -> bool:
        """
        Return True iff every circle currently has at most two dots.

        This is mainly intended as a debugging check.
        """

        return all(
            count <= 2
            for color in Color
            for count in self.circle_counts[color].values()
        )

    def is_complete(self) -> bool:
        """
        Return True iff every circle contains either
        zero or exactly two dots.
        """

        return all(
            count in (0, 2)
            for color in Color
            for count in self.circle_counts[color].values()
        )

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def copy(self) -> Diagram:
        """
        Return an independent copy of the diagram.
        """

        return Diagram(
            background=self.background,
            dots=set(self.dots),
        )

    def active_circles(self, color: Color) -> set[int]:
        """
        Return the ids of circles containing exactly two dots.
        """

        return {
            cid
            for cid, count in self.circle_counts[color].items()
            if count == 2
        }

    def __len__(self) -> int:
        """Return the number of dots."""

        return len(self.dots)

    def __contains__(self, vertex: Vertex) -> bool:
        return vertex in self.dots

    def __iter__(self):
        return iter(self.dots)