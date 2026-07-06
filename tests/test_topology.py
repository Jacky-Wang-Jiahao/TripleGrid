"""
Tests for ``triple_grid.topology``.

Uses only the standard library (``unittest``) -- no ``pytest``
required. Run directly:

    python tests/test_topology.py            # run the tests, verbose
    python tests/test_topology.py --report    # print the diagram report instead

Three square backgrounds are used throughout, (m, n, p) = (k, 0, k)
for k = 2, 3, 4. Setting n = 0 degenerates the hexagonal fundamental
domain to a k x k square (see ``geometry.Background._in_domain``:
with n = 0 the constraints reduce to 0 <= x < p, 0 <= y < m, and
x - y < p, y - x <= m are then automatically satisfied), so these are
exactly the classical k x k square triple grids, with
R = gcd(n, p) = p = k, B = gcd(m, n) = m = k, G = gcd(m, p) = k.

Two kinds of checks are used:

1. *Structural / theory-level invariants* that must hold for every
   diagram on every background, derived directly from the
   construction in ``topology.py`` (3-regularity, the parity of chi
   for orientable components, the range of chi for non-orientable
   components, etc). These do not depend on the specific numbers
   below and will keep catching regressions even if the enumeration
   algorithm (``enumerate.py`` / a future ``search.py``) changes.

2. *Regression / golden-value checks* that pin down the exact
   diagram counts and surface labels currently produced. These are
   tied to the present ``enumerate.py`` + ``diagram.py`` +
   ``topology.py`` implementation and were obtained by direct
   computation (see ``RegressionTests`` below); if a future refactor
   (e.g. ``search.py`` replacing ``enumerate.py``) legitimately
   changes these numbers, these specific assertions -- and only
   these -- should be updated.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

# Allow running this file directly as a script (``python tests/test_topology.py``)
# without the package being installed: a bare ``python tests/test_topology.py``
# only puts this file's own directory on sys.path, which is not enough to
# import ``triple_grid`` sitting next to ``tests/``.
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from triple_grid.diagram import Diagram
from triple_grid.enumerate import enumerate_diagrams
from triple_grid.geometry import Background, Color, Vertex
from triple_grid.topology import (
    SurfaceComponent,
    diagram_surface_components,
    format_surface,
)

# ----------------------------------------------------------------------
# Shared fixtures / helpers
# ----------------------------------------------------------------------

# The three square backgrounds requested: (m, n, p) = (k, 0, k).
SQUARE_SIZES = [7]

_LABEL_RE = re.compile(r"^#\^(\d+)(T\^2|RP\^2)$")


def square_background(k: int) -> Background:
    """The k x k square triple grid, (m, n, p) = (k, 0, k)."""

    return Background(m=k, n=0, p=k)


def all_diagrams(background: Background) -> list[Diagram]:
    """All dot diagrams on ``background``, materialized as a list.

    Only safe to call for backgrounds small enough to enumerate in
    full; the three sizes under test (|dot diagrams| = 2, 4, 49 for
    k = 2, 3, 4 respectively -- see ``RegressionTests.test_diagram_counts``)
    are.
    """

    return list(enumerate_diagrams(background))


def parse_label(label: str) -> tuple[int, str]:
    """
    Parse a single component label '#^gT^2' / '#^kRP^2' (no spaces,
    as produced by ``SurfaceComponent.label``) into (multiplicity,
    kind), asserting it matches the expected grammar.
    """

    m = _LABEL_RE.match(label)
    assert m is not None, f"label {label!r} does not match '#^<int>(T^2|RP^2)'"
    return int(m.group(1)), m.group(2)


# ----------------------------------------------------------------------
# 1. Structural invariants -- must hold for every diagram, on every
#    background, regardless of which enumeration backend produced it.
# ----------------------------------------------------------------------


class StructuralInvariantTests(unittest.TestCase):

    def test_every_diagram_is_complete(self) -> None:
        """
        Sanity check on the *inputs* to the topology computation: every
        diagram produced by the enumerator must satisfy the precondition
        of ``diagram_surface_components`` (every circle has 0 or 2 dots).
        This is what the correctness of ``enumerate.py`` promises; if it
        ever failed, ``topology._build_edges`` would raise, and this test
        isolates *which* stage is at fault.
        """

        for k in SQUARE_SIZES:
            with self.subTest(k=k):
                bg = square_background(k)
                for diagram in all_diagrams(bg):
                    self.assertTrue(diagram.is_complete())

    def test_components_partition_the_dots(self) -> None:
        """
        The connected components of the RGB graph partition the dot set:
        every dot belongs to exactly one component, so the component
        vertex counts must sum to the total number of dots.
        """

        for k in SQUARE_SIZES:
            with self.subTest(k=k):
                bg = square_background(k)
                for diagram in all_diagrams(bg):
                    components = diagram_surface_components(diagram)
                    total = sum(c.num_vertices for c in components)
                    self.assertEqual(total, len(diagram.dots))

    def test_component_is_3_regular(self) -> None:
        """
        Every component must be 3-regular: e = 3v/2 exactly (checked
        independently here, in addition to the internal assertion inside
        ``topology._num_edges``), and v must be even (3v must be even to
        equal 2e for an integer e).
        """

        for k in SQUARE_SIZES:
            with self.subTest(k=k):
                bg = square_background(k)
                for diagram in all_diagrams(bg):
                    for c in diagram_surface_components(diagram):
                        self.assertEqual(c.num_vertices % 2, 0)
                        self.assertEqual(c.num_edges, 3 * c.num_vertices // 2)

    def test_orientable_chi_is_even_and_at_most_two(self) -> None:
        """
        For a closed orientable surface, chi = 2 - 2g with g >= 0, so chi
        is even and chi <= 2 (equality iff the sphere, g = 0).
        """

        for k in SQUARE_SIZES:
            with self.subTest(k=k):
                bg = square_background(k)
                for diagram in all_diagrams(bg):
                    for c in diagram_surface_components(diagram):
                        if c.orientable:
                            chi = c.euler_characteristic
                            self.assertEqual(chi % 2, 0)
                            self.assertLessEqual(chi, 2)
                            g, kind = parse_label(c.label())
                            self.assertEqual(kind, "T^2")
                            self.assertEqual(g, 1 - chi // 2)
                            self.assertGreaterEqual(g, 0)

    def test_nonorientable_chi_is_at_most_one(self) -> None:
        """
        For a closed non-orientable surface #^k RP^2 with k >= 1,
        chi = 2 - k <= 1 (equality iff k = 1, the projective plane
        itself). This also pins down the *sign* of the crosscap formula:
        with the (incorrect) brief formula k = 1 - chi this bound would
        instead be chi <= 0, which the projective-plane case (chi = 1)
        violates -- see the discrepancy noted in ``topology.py``'s module
        docstring.
        """

        for k in SQUARE_SIZES:
            with self.subTest(k=k):
                bg = square_background(k)
                for diagram in all_diagrams(bg):
                    for c in diagram_surface_components(diagram):
                        if not c.orientable:
                            chi = c.euler_characteristic
                            self.assertLessEqual(chi, 1)
                            num_crosscaps, kind = parse_label(c.label())
                            self.assertEqual(kind, "RP^2")
                            self.assertEqual(num_crosscaps, 2 - chi)
                            self.assertGreaterEqual(num_crosscaps, 1)

    def test_format_surface_matches_manual_join(self) -> None:
        """``format_surface`` is exactly ``" u ".join(labels)``."""

        for k in SQUARE_SIZES:
            with self.subTest(k=k):
                bg = square_background(k)
                for diagram in all_diagrams(bg):
                    components = diagram_surface_components(diagram)
                    expected = " u ".join(c.label() for c in components)
                    self.assertEqual(format_surface(components), expected)

    def test_empty_diagram_has_empty_surface(self) -> None:
        """The empty diagram (no dots) has no components and no surface."""

        bg = square_background(2)
        empty = Diagram(background=bg, dots=set())
        self.assertTrue(empty.is_complete())
        self.assertEqual(diagram_surface_components(empty), [])
        self.assertEqual(format_surface(diagram_surface_components(empty)), "")

    def test_incomplete_diagram_is_rejected(self) -> None:
        """
        ``diagram_surface_components`` must refuse a diagram that is not
        complete (some circle has exactly 1 dot): the RGB graph is
        undefined for it, and silently proceeding would corrupt the
        3-regularity invariant that everything else in this module
        depends on.
        """

        bg = square_background(2)
        v = bg.vertices[0]
        broken = Diagram(background=bg, dots={v})  # a lone dot: 1 on each of its 3 circles
        self.assertFalse(broken.is_complete())
        with self.assertRaises(ValueError):
            diagram_surface_components(broken)


# ----------------------------------------------------------------------
# 2. Minimal synthetic sanity check, independent of the enumerator:
#    K4 with a proper 3-edge-colouring is the classical smallest
#    crystallization of RP^2 (v=4, e=6, f=3, chi=1). This isolates
#    the graph-theoretic core (bipartiteness check, face count, the
#    k = 2 - chi formula) from the geometry/enumeration layers.
# ----------------------------------------------------------------------


class K4SyntheticTests(unittest.TestCase):

    def test_k4_three_edge_coloring_is_the_projective_plane(self) -> None:
        from triple_grid.topology import _connected_components, _is_bipartite, _num_edges

        a, b, c, d = Vertex(0, 0), Vertex(1, 0), Vertex(2, 0), Vertex(3, 0)
        adjacency: dict[Vertex, list[tuple[Color, Vertex]]] = {
            v: [] for v in (a, b, c, d)
        }

        def add_edge(u: Vertex, v: Vertex, color: Color) -> None:
            adjacency[u].append((color, v))
            adjacency[v].append((color, u))

        # A 1-factorization of K4 into three perfect matchings.
        add_edge(a, b, Color.RED)
        add_edge(c, d, Color.RED)
        add_edge(a, c, Color.BLUE)
        add_edge(b, d, Color.BLUE)
        add_edge(a, d, Color.GREEN)
        add_edge(b, c, Color.GREEN)

        comp = {a, b, c, d}
        orientable = _is_bipartite(comp, adjacency)
        e = _num_edges(comp, adjacency)
        f = sum(
            len(_connected_components(comp, adjacency, colors=frozenset(Color) - {col}))
            for col in Color
        )
        sc = SurfaceComponent(
            num_vertices=4, num_edges=e, num_faces=f, orientable=orientable
        )

        self.assertFalse(orientable)  # K4 has a triangle -> odd cycle -> non-bipartite
        self.assertEqual(e, 6)
        self.assertEqual(f, 3)
        self.assertEqual(sc.euler_characteristic, 1)
        self.assertEqual(sc.label(), "#^1RP^2")


# ----------------------------------------------------------------------
# 3. Regression / golden values for the three requested backgrounds,
#    computed directly from the current enumerate.py + topology.py.
# ----------------------------------------------------------------------


class RegressionTests(unittest.TestCase):

    def test_diagram_counts(self) -> None:
        for k, expected_count in [(2, 2), (3, 4), (4, 49)]:
            with self.subTest(k=k):
                bg = square_background(k)
                self.assertEqual(len(all_diagrams(bg)), expected_count)

    def test_k2_square_is_the_minimal_rp2_example(self) -> None:
        """
        The unique non-empty diagram on the 2x2 square uses all 4
        vertices and its RGB graph is (up to isomorphism) exactly the K4
        example in ``K4SyntheticTests``: v=4, e=6, f=3, chi=1,
        non-orientable -> a single #^1 RP^2.
        """

        bg = square_background(2)
        nonempty = [d for d in all_diagrams(bg) if len(d.dots) > 0]
        self.assertEqual(len(nonempty), 1)
        diagram = nonempty[0]
        self.assertEqual(len(diagram.dots), 4)

        components = diagram_surface_components(diagram)
        self.assertEqual(len(components), 1)
        c = components[0]
        self.assertEqual(
            (c.num_vertices, c.num_edges, c.num_faces, c.orientable),
            (4, 6, 3, False),
        )
        self.assertEqual(format_surface(components), "#^1RP^2")

    def test_k3_square_all_nonempty_diagrams_are_tori(self) -> None:
        """
        On the 3x3 square, every one of the 3 non-empty diagrams (each
        using 6 of the 9 vertices) has a single orientable component with
        chi = 0, i.e. a torus #^1 T^2 -- no non-orientable examples arise
        at this size.
        """

        bg = square_background(3)
        nonempty = [d for d in all_diagrams(bg) if len(d.dots) > 0]
        self.assertEqual(len(nonempty), 3)
        for diagram in nonempty:
            self.assertEqual(len(diagram.dots), 6)
            components = diagram_surface_components(diagram)
            self.assertEqual(len(components), 1)
            c = components[0]
            self.assertTrue(c.orientable)
            self.assertEqual(c.euler_characteristic, 0)
            self.assertEqual(format_surface(components), "#^1T^2")

    def test_k4_square_produces_both_orientable_and_nonorientable_examples(self) -> None:
        """
        The 4x4 square is the smallest of the three requested backgrounds
        where *both* kinds of surface actually occur among genuine
        (geometrically realized) diagrams -- not just the hand-built K4
        example -- giving an end-to-end confirmation that the
        orientability test and the k = 2 - chi crosscap formula agree
        with each other on real data, not only on the synthetic case.
        """

        bg = square_background(4)
        diagrams = all_diagrams(bg)
        labels = [format_surface(diagram_surface_components(d)) for d in diagrams]

        self.assertIn("", labels)  # the empty diagram
        self.assertIn("#^1T^2", labels)
        self.assertIn("#^1RP^2", labels)
        self.assertIn("#^2RP^2", labels)
        # Every diagram here happens to have a single connected component;
        # this is a fact about these three specific backgrounds, not a
        # general theorem, so it is checked rather than assumed elsewhere.
        for diagram in diagrams:
            self.assertLessEqual(len(diagram_surface_components(diagram)), 1)


# ----------------------------------------------------------------------
# Script mode: print a full report for k = 2, 3, 4.
# ----------------------------------------------------------------------


# def _print_report() -> None:
#     for k in SQUARE_SIZES:
#         bg = square_background(k)
#         diagrams = all_diagrams(bg)
#         print(
#             f"=== square background k={k}  (m,n,p)=({k},0,{k})  "
#             f"|V|={len(bg.vertices)}  #diagrams={len(diagrams)} ==="
#         )
#         for i, diagram in enumerate(diagrams):
#             components = diagram_surface_components(diagram)
#             label = format_surface(components)
#             dots = sorted((v.x, v.y) for v in diagram.dots)
#             print(f"  #{i}: |dots|={len(dots)} dots={dots}")
#             for c in components:
#                 print(
#                     f"       v={c.num_vertices} e={c.num_edges} "
#                     f"f={c.num_faces} chi={c.euler_characteristic} "
#                     f"orientable={c.orientable} -> {c.label()}"
#                 )
#             print(f"       surface = {label!r}")
#         print()

from pathlib import Path

def _print_report(output_file: str = "report.txt") -> None:
    output_path = Path(output_file)

    with output_path.open("w", encoding="utf-8") as f:
        for k in SQUARE_SIZES:
            bg = square_background(k)
            diagrams = all_diagrams(bg)

            print(
                f"=== square background k={k}  (m,n,p)=({k},0,{k})  "
                f"|V|={len(bg.vertices)}  #diagrams={len(diagrams)} ===",
                file=f,
            )

            for i, diagram in enumerate(diagrams):
                components = diagram_surface_components(diagram)
                label = format_surface(components)
                dots = sorted((v.x, v.y) for v in diagram.dots)

                print(
                    f"  #{i}: |dots|={len(dots)} dots={dots}",
                    file=f,
                )

                for c in components:
                    print(
                        f"       v={c.num_vertices} "
                        f"e={c.num_edges} "
                        f"f={c.num_faces} "
                        f"chi={c.euler_characteristic} "
                        f"orientable={c.orientable} "
                        f"-> {c.label()}",
                        file=f,
                    )

                print(f"       surface = {label!r}", file=f)

            print(file=f)

    print(f"Report written to {output_path.resolve()}")


if __name__ == "__main__":
    if "--report" in sys.argv:
        _print_report()
    else:
        unittest.main(verbosity=2, argv=[sys.argv[0]])