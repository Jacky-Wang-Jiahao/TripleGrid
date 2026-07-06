from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from triple_grid.geometry import Background, Color
from triple_grid.diagram import Diagram

if __name__ == "__main__":

    from pprint import pprint

    bg = Background(2, 2, 2)

    print("=" * 60)
    print("Background")
    print("=" * 60)

    print(f"{len(bg.vertices)=}")
    print()

    d = Diagram(bg)

    print("=" * 60)
    print("Initially")
    print("=" * 60)

    print("dots =", d.dots)
    print("is_valid =", d.is_valid())
    print("is_complete =", d.is_complete())
    print()

    v1 = bg.vertices[0]
    v2 = bg.vertices[1]
    v3 = bg.vertices[2]

    print("=" * 60)
    print("Add first vertex")
    print("=" * 60)

    d.add(v1)

    print(d.dots)

    for color in Color:
        cid = bg.circle_id(v1, color)
        print(color.name, cid, d.count(color, cid))

    print("can_add(v1) =", d.can_add(v1))
    print()

    print("=" * 60)
    print("Add second vertex")
    print("=" * 60)

    d.add(v2)

    print(d.dots)

    print("is_valid =", d.is_valid())
    print("is_complete =", d.is_complete())
    print()

    print("=" * 60)
    print("Remove first vertex")
    print("=" * 60)

    d.remove(v1)

    print(d.dots)

    for color in Color:
        cid = bg.circle_id(v2, color)
        print(color.name, cid, d.count(color, cid))

    print()

    print("=" * 60)
    print("Copy")
    print("=" * 60)

    d2 = d.copy()

    print(d2.dots == d.dots)
    print(d2 is d)
    print(d2.dots is d.dots)

    print()

    print("=" * 60)
    print("Active circles")
    print("=" * 60)

    pprint(d.active_circles(Color.RED))
    pprint(d.active_circles(Color.BLUE))
    pprint(d.active_circles(Color.GREEN))