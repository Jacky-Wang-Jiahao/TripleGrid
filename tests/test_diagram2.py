from pathlib import Path
import sys
from random import sample

sys.path.append(str(Path(__file__).resolve().parents[1]))

from triple_grid.geometry import Background
from triple_grid.diagram import Diagram


def main():

    bg = Background(4, 5, 3)

    d = Diagram(bg)

    print("=" * 60)
    print("Random insertion test")
    print("=" * 60)

    inserted = []

    for v in sample(bg.vertices, min(20, len(bg.vertices))):

        if d.can_add(v):

            d.add(v)
            inserted.append(v)

            # 如果你实现了 check_consistency()
            # 可以取消下一行注释
            #
            # d.check_consistency()

    print(f"Inserted {len(inserted)} vertices.")
    print(f"Diagram valid: {d.is_valid()}")

    assert d.is_valid()

    print()

    print("=" * 60)
    print("Removing all vertices")
    print("=" * 60)

    for v in inserted:

        d.remove(v)

        # d.check_consistency()

    assert len(d) == 0
    assert d.is_complete()

    print("All vertices removed successfully.")
    print("Diagram returned to empty state.")
    print("PASS")


if __name__ == "__main__":
    main()