from triple_grid.geometry import Background, Color
from triple_grid.enumerate import (
    longest_color,
    _build_order,
    enumerate_diagram_vertex_sets,
    count_diagrams,
    EnumerationStats,
)


def check_diagram(background, dots):
    """Verify every circle contains exactly 0 or 2 dots."""
    for color in Color:
        for circle in background.cycles[color].values():
            c = sum(v in dots for v in circle)
            assert c in (0, 2), (
                f"{color=} has illegal count {c}"
            )


def test_background(background):
    print("=" * 60)
    print(background)

    # -------------------------------------------------
    # longest_color
    # -------------------------------------------------
    lc = longest_color(background)

    num_cycles = {
        color: len(background.cycles[color])
        for color in Color
    }

    expected = min(num_cycles, key=num_cycles.get)

    assert lc == expected

    print("longest_color: PASS")

    # -------------------------------------------------
    # build_order
    # -------------------------------------------------
    order, remaining = _build_order(background)

    assert len(order) == len(background.vertices)
    assert len(set(order)) == len(order)
    assert len(remaining) == len(order)

    print("build_order: PASS")

    # -------------------------------------------------
    # enumeration
    # -------------------------------------------------
    stats = EnumerationStats()

    diagrams = list(
        enumerate_diagram_vertex_sets(
            background,
            stats=stats,
        )
    )

    print(f"{len(diagrams)} diagrams found\n")

    for i, d in enumerate(diagrams, start=1):
        print(f"Diagram {i}:")
        for v in sorted(d, key=lambda v: (v.x, v.y)):
            print(f"  ({v.x}, {v.y})")
        print()

        check_diagram(background, d)

    print("diagram validity: PASS")

    # -------------------------------------------------
    # count_diagrams
    # -------------------------------------------------
    cnt = count_diagrams(background)

    assert cnt == len(diagrams)

    print("count_diagrams: PASS")

    # -------------------------------------------------
    # stats sanity
    # -------------------------------------------------
    assert stats.completed_assignments == len(diagrams)
    assert stats.yielded_diagrams == len(diagrams)

    print("stats:")
    print(stats)

    print("ALL TESTS PASSED\n")


def main():
    # Small examples only.
    examples = [
        Background(5,0,5),
    ]

    for bg in examples:
        test_background(bg)


if __name__ == "__main__":
    main()