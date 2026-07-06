from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

from .diagram import Diagram
from .geometry import Background, Color, Vertex

# Fixed, canonical ordering of colors used everywhere in this module.
# `Color` is an Enum, so this is stable: (RED, BLUE, GREEN).
_COLORS: tuple[Color, ...] = tuple(Color)


@dataclass(slots=True)
class EnumerationStats:
    """
    Diagnostics for a single enumeration run.

    Every vertex in the search order is visited exactly once per branch
    of the search tree, and at each visit the algorithm decides, or is
    forced to decide, whether that vertex carries a dot. This dataclass
    counts what kind of decision occurred, which is the data needed to
    empirically check the complexity claims discussed in the module
    docstring below.

    Attributes
    ----------
    nodes_visited
        Number of vertices for which a decision was actually evaluated
        (i.e. search-tree nodes that are not leaves).
    forced_moves
        Nodes where exactly one of {add, skip} was legal.
    free_choices
        Nodes where both {add, skip} were legal (genuine branching,
        analogous to a queen having more than one open column).
    dead_ends
        Nodes where neither move was legal: the branch is abandoned.
    completed_assignments
        Number of times the search reached a full assignment of all
        |V| vertices. By construction (see the correctness proof in
        the module docstring) every completed assignment is a valid
        dot diagram, so this is exactly the number of solutions to
        the unrestricted problem that lie in the search tree explored
        (all of them, if min_dots == 0 and max_dots is None).
    yielded_diagrams
        Number of completed assignments that additionally satisfied
        the requested ``min_dots`` / ``max_dots`` bounds and were
        therefore actually produced to the caller.
    """

    nodes_visited: int = 0
    forced_moves: int = 0
    free_choices: int = 0
    dead_ends: int = 0
    completed_assignments: int = 0
    yielded_diagrams: int = 0


# ----------------------------------------------------------------------
# Search-order construction
# ----------------------------------------------------------------------


def longest_color(background: Background) -> Color:
    """
    Return the color whose circles contain the most vertices.

    Every circle of a fixed color partitions the same |V| vertices into
    equally sized classes (this is checked, not merely assumed -- see
    ``Background._build_cycles``: the R, B, G values are the sizes of a
    group-homomorphism's fibers, hence uniform). So "longest circles"
    is equivalent to "fewest circles", which is what we sort on here.
    """

    return min(Color, key=lambda c: len(background.cycles[c]))


def _build_order(
    background: Background,
) -> tuple[list[Vertex], list[tuple[int, int, int]]]:
    """
    Build the fixed vertex-visitation order and, for every position in
    it, how many *not-yet-visited* vertices remain that share each of
    the vertex's three circles.

    The order groups vertices by the circles of the longest color
    (contiguous blocks, one per circle), and within a block sorts by
    the other two colors' circle ids, so that those circles also tend
    to close out (have all their members visited) as early as
    possible -- this is what makes the forward-checking rules in
    ``enumerate_diagram_vertex_sets`` bite early rather than only at
    the end of a whole primary circle.

    Returns
    -------
    order
        ``order[k]`` is the k-th vertex to be decided.
    remaining_after
        ``remaining_after[k]`` is a triple ``(rR, rB, rG)`` -- for each
        color, the number of vertices *strictly after* position k that
        belong to the same circle (of that color) as ``order[k]``,
        not counting ``order[k]`` itself. This depends only on the
        fixed order, never on which vertices end up with dots, so it
        is computed once and reused for every branch of the search.
    """

    primary = longest_color(background)
    secondary, tertiary = [c for c in _COLORS if c is not primary]

    order: list[Vertex] = []
    for cid in sorted(background.cycles[primary]):
        block = background.cycles[primary][cid]
        block = sorted(
            block,
            key=lambda v: (
                background.vertex_circle_ids[v][secondary],
                background.vertex_circle_ids[v][tertiary],
            ),
        )
        order.extend(block)

    total = {
        color: {cid: len(members) for cid, members in background.cycles[color].items()}
        for color in _COLORS
    }
    seen = {color: {cid: 0 for cid in background.cycles[color]} for color in _COLORS}

    remaining_after: list[tuple[int, int, int]] = []
    for v in order:
        ids = background.vertex_circle_ids[v]
        triple = tuple(
            total[color][ids[color]] - seen[color][ids[color]] - 1
            for color in _COLORS
        )
        remaining_after.append(triple)  # type: ignore[arg-type]
        for color in _COLORS:
            seen[color][ids[color]] += 1

    return order, remaining_after


# ----------------------------------------------------------------------
# Core search
# ----------------------------------------------------------------------


def enumerate_diagram_vertex_sets(
    background: Background,
    *,
    min_dots: int = 0,
    max_dots: Optional[int] = None,
    stats: Optional[EnumerationStats] = None,
) -> Iterator[frozenset[Vertex]]:
    """
    Enumerate every dot diagram on ``background`` as a frozenset of
    vertices, i.e. every subset D of the vertices such that every
    circle (of every color) contains exactly 0 or 2 elements of D.

    Algorithm
    ---------
    Vertices are visited one at a time in the fixed order built by
    ``_build_order`` (grouped by circles of the longest color). At
    each vertex the search considers two candidate moves, "add a dot"
    and "skip", and discards a move if it can be shown -- from
    information already fixed earlier in the order -- that no
    completion of the diagram could ever be legal after taking it.
    Two such conditions are checked, both provable, sound, and
    complete for a *fixed* visitation order (see the correctness
    argument below):

    1. Capacity: adding a dot to a circle that already has 2 is
       always illegal (a circle can never hold 3).
    2. Stranding: a move that would leave some circle at a dot-count
       of exactly 1 with *no remaining candidate vertices* to bring
       it up to 2 is always illegal, since that circle could then
       never reach {0, 2}.

    Because ``remaining_after`` counts only vertices later in the
    fixed order, condition 2 is a genuine necessary condition (no
    valid completion can violate it), not a heuristic -- so no legal
    diagram is ever pruned away. This makes the search exact: it is
    both sound (only legal diagrams reach depth |V|) and complete
    (every legal diagram is reached by exactly one root-to-leaf path).

    Complexity
    ----------
    See the module docstring for the complexity discussion. In brief:
    the number of leaves of this search tree is *exactly* the number
    of dot diagrams (Theorem 1 below), so the algorithm's cost is, up
    to the overhead of dead branches (each terminated in O(1) extra
    work), output-sensitive in the number of solutions -- which is the
    best an enumeration algorithm can hope for.

    Parameters
    ----------
    background
        The finite background to search.
    min_dots, max_dots
        If given, only diagrams whose dot count lies in
        ``[min_dots, max_dots]`` (inclusive; ``max_dots=None`` means
        no upper bound) are produced. Filtering is folded into the
        search itself (a cheap, sound bound on how many more dots a
        partial assignment could possibly gain), so it also prunes
        the search, not just the output.
    stats
        If given, populated with counters describing the shape of the
        search tree that was explored (see ``EnumerationStats``).

    Yields
    ------
    frozenset[Vertex]
        Each valid dot diagram's vertex set, exactly once.
    """

    order, remaining_after = _build_order(background)
    n = len(order)

    diagram = Diagram(background)
    counts = diagram.circle_counts
    dots_so_far = 0

    # Explicit stack (rather than recursive generators) so the search
    # depth -- up to |V| -- never risks Python's recursion limit, and
    # so there is no per-level `yield from` overhead.
    #
    # Each frame is [vertex, legal_moves, pointer]: `legal_moves` is a
    # list of the booleans (True = add a dot) that were legal when we
    # first arrived at this vertex, in the order we try them, and
    # `pointer` is the index of the move currently in effect.
    stack: list[list] = []

    while True:
        pos = len(stack)

        if pos == n:
            if stats is not None:
                stats.completed_assignments += 1
            if dots_so_far >= min_dots:
                if stats is not None:
                    stats.yielded_diagrams += 1
                yield frozenset(diagram.dots)
            # Nothing more to try at a full assignment; fall through
            # to backtracking.
        else:
            v = order[pos]
            ids = background.vertex_circle_ids[v]
            r_triple = remaining_after[pos]

            budget_ok = dots_so_far + (n - pos) >= min_dots

            add_ok = budget_ok and (max_dots is None or dots_so_far < max_dots)
            if add_ok:
                for color, r in zip(_COLORS, r_triple):
                    c = counts[color][ids[color]]
                    new_c = c + 1
                    if new_c > 2 or (new_c == 1 and r == 0):
                        add_ok = False
                        break

            skip_ok = budget_ok
            if skip_ok:
                for color, r in zip(_COLORS, r_triple):
                    c = counts[color][ids[color]]
                    if c == 1 and r == 0:
                        skip_ok = False
                        break

            moves = [m for m, ok in ((True, add_ok), (False, skip_ok)) if ok]

            if stats is not None:
                stats.nodes_visited += 1
                if len(moves) == 0:
                    stats.dead_ends += 1
                elif len(moves) == 1:
                    stats.forced_moves += 1
                else:
                    stats.free_choices += 1

            if moves:
                if moves[0]:
                    diagram.add(v)
                    dots_so_far += 1
                stack.append([v, moves, 0])
                continue  # decide the next vertex

            # No legal move: dead end, fall through to backtracking.

        # ---- backtrack: undo the last move tried, advance to the
        # ---- next untried alternative at the nearest ancestor that
        # ---- has one, popping frames that are exhausted. ----
        found_alternative = False
        while stack:
            top = stack[-1]
            v, moves, ptr = top
            if moves[ptr]:
                diagram.remove(v)
                dots_so_far -= 1
            ptr += 1
            if ptr < len(moves):
                top[2] = ptr
                if moves[ptr]:
                    diagram.add(v)
                    dots_so_far += 1
                found_alternative = True
                break
            stack.pop()

        if not found_alternative and not stack:
            return


def enumerate_diagrams(
    background: Background,
    *,
    min_dots: int = 0,
    max_dots: Optional[int] = None,
    stats: Optional[EnumerationStats] = None,
) -> Iterator[Diagram]:
    """
    Same as ``enumerate_diagram_vertex_sets``, but yields ``Diagram``
    objects (each with its own independent, fully populated
    ``circle_counts``) instead of bare vertex sets.

    Constructing a ``Diagram`` re-derives its circle counts from
    scratch (O(|dots|) work), so prefer
    ``enumerate_diagram_vertex_sets`` directly if you only need the
    vertex set.
    """

    for dots in enumerate_diagram_vertex_sets(
        background, min_dots=min_dots, max_dots=max_dots, stats=stats
    ):
        yield Diagram(background=background, dots=set(dots))


def count_diagrams(
    background: Background,
    *,
    min_dots: int = 0,
    max_dots: Optional[int] = None,
) -> int:
    """
    Count dot diagrams without materializing them.

    Equivalent to ``sum(1 for _ in enumerate_diagram_vertex_sets(...))``
    but does not allocate a ``frozenset`` per solution.
    """

    stats = EnumerationStats()
    counted = 0
    for _ in enumerate_diagram_vertex_sets(
        background, min_dots=min_dots, max_dots=max_dots, stats=stats
    ):
        counted += 1
    return counted