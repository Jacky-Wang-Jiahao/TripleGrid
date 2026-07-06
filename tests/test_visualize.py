"""
Example usage of `visualization.py`.

Run as a module from the parent of the `dotdiagram` package, e.g.:

    python -m dotdiagram.demo
"""
from __future__ import annotations

import itertools

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from triple_grid.enumerate import enumerate_diagrams
from triple_grid.geometry import Background
from triple_grid.visualize import plot_diagram, plot_diagrams

if __name__ == "__main__":
    # Pick any (m, n, p); R, B, G and the vertex count follow from it.
    background = Background(m=2, n=3, p=4)
    print(f"{len(background.vertices)} vertices, R={background.R}, "
          f"B={background.B}, G={background.G}")

    # enumerate_diagrams is a generator -- every diagram it yields is
    # already complete (every circle has 0 or 2 dots), so take however
    # many you want with itertools.islice rather than exhausting it.
    diagrams = list(itertools.islice(enumerate_diagrams(background), 16))

    # A single diagram, full size.
    ax = plot_diagram(diagrams[-1])
    ax.figure.savefig("single_diagram.png", dpi=160, bbox_inches="tight")

    # # An overview grid of several diagrams at once.
    # fig = plot_diagrams(diagrams, max_diagrams=12, ncols=4)
    # fig.savefig("diagram_grid.png", dpi=150, bbox_inches="tight")