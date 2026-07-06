"""
Basic validation tests for geometry.py.

These tests verify the mathematical correctness of the Background
construction rather than software implementation details.
"""

from math import gcd

from triple_grid.geometry import Background, Color


# ------------------------------------------------------------
# Helper
# ------------------------------------------------------------

def check(condition: bool, message: str) -> None:
    if condition:
        print(f"[PASS] {message}")
    else:
        raise AssertionError(f"[FAIL] {message}")


# ------------------------------------------------------------
# Individual tests
# ------------------------------------------------------------

def test_vertex_count(m: int, n: int, p: int) -> None:
    bg = Background(m, n, p)

    expected = m * n + n * p + p * m

    check(
        len(bg.vertices) == expected,
        f"Vertex count ({m},{n},{p}) = {expected}",
    )


def test_circle_numbers(m: int, n: int, p: int) -> None:
    bg = Background(m, n, p)

    check(bg.R == gcd(n, p), "Number of red circles")
    check(bg.B == gcd(m, n), "Number of blue circles")
    check(bg.G == gcd(m, p), "Number of green circles")


def test_partition_of_vertices(m: int, n: int, p: int) -> None:
    """
    Every vertex should belong to exactly one circle of each color.
    """

    bg = Background(m, n, p)

    for color in Color:

        total = sum(
            len(circle)
            for circle in bg.cycles[color].values()
        )

        check(
            total == len(bg.vertices),
            f"{color.name} circles partition the vertices",
        )


def test_circle_membership(m: int, n: int, p: int) -> None:
    """
    circle_id and cycles must agree.
    """

    bg = Background(m, n, p)

    for color in Color:

        for v in bg.vertices:

            cid = bg.circle_id(v, color)

            check(
                v in bg.circle(color, cid),
                f"{color.name} membership of {v}",
            )


def test_vertex_lookup(m: int, n: int, p: int) -> None:
    bg = Background(m, n, p)

    for v in bg.vertices:

        check(
            bg.vertex_lookup[(v.x, v.y)] == v,
            f"Lookup of {v}",
        )


def test_circle_ids(m: int, n: int, p: int) -> None:
    """
    Verify the explicit formulas for circle ids.
    """

    bg = Background(m, n, p)

    for v in bg.vertices:

        check(
            bg.circle_id(v, Color.RED) == v.x % bg.R,
            f"Red id of {v}",
        )

        check(
            bg.circle_id(v, Color.BLUE) == v.y % bg.B,
            f"Blue id of {v}",
        )

        check(
            bg.circle_id(v, Color.GREEN) == (v.x - v.y) % bg.G,
            f"Green id of {v}",
        )


# ------------------------------------------------------------
# Run all tests
# ------------------------------------------------------------

def run_case(m: int, n: int, p: int) -> None:

    print()
    print("=" * 60)
    print(f"Testing Background({m}, {n}, {p})")
    print("=" * 60)

    test_vertex_count(m, n, p)
    test_circle_numbers(m, n, p)
    test_partition_of_vertices(m, n, p)
    test_circle_membership(m, n, p)
    test_vertex_lookup(m, n, p)
    test_circle_ids(m, n, p)


if __name__ == "__main__":

    test_cases = [

        (1, 1, 1),

        (2, 2, 2),

        (2, 3, 4),

        (1, 3, 5),

        (4, 6, 8),

    ]

    for case in test_cases:
        run_case(*case)

    print()
    print("=" * 60)
    print("All geometry tests passed.")
    print("=" * 60)