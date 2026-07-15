"""
This code tries to separate the three links of a TGD diagram, parse them, and pass them to SnapPy
"""
from __future__ import annotations

import itertools
from pathlib import Path
import sys

from snappy import Manifold


sys.path.append(str(Path(__file__).resolve().parents[1]))

from triple_grid.enumerate import enumerate_diagrams
from triple_grid.geometry import Background, Color
from triple_grid.topology import diagram_surface_components, format_surface

if __name__ == "__main__":
    # Pick any (m, n, p); R, B, G and the vertex count follow from it.
    m = 7
    n = 0
    p = 7
    background = Background(m, n, p)
    print(f"{len(background.vertices)} vertices, R={background.R}, "
          f"B={background.B}, G={background.G}")

    # enumerate_diagrams is a generator -- every diagram it yields is
    # already complete (every circle has 0 or 2 dots), so take however
    # many you want with itertools.islice rather than exhausting it.
    diagrams = list(itertools.islice(enumerate_diagrams(background), 1))

    diagram = diagrams[-1]
    # A single diagram, full size.

    components = diagram_surface_components(diagram)
    label = format_surface(components)
    dots = sorted((v.x, v.y) for v in diagram.dots)

    print(f"|dots|={len(dots)} dots={dots}")

    for c in components:
                    print(
                        f"       v={c.num_vertices} "
                        f"e={c.num_edges} "
                        f"f={c.num_faces} "
                        f"chi={c.euler_characteristic} "
                        f"orientable={c.orientable} "
                        f"-> {c.label()}"
                    )

                    orientable = c.orientable
    # Efficiently group occupied RED and BLUE circles into dicts mapping circle_id -> (v0, v1)
    from collections import defaultdict

    r_lines = {}
    b_lines = {}

    # cluster dots by circle id for RED and BLUE
    for v in diagram.dots:
        r = background.vertex_circle_ids[v][Color.RED]
        b = background.vertex_circle_ids[v][Color.BLUE]
        r_lines.setdefault(r, []).append(v)
        b_lines.setdefault(b, []).append(v)

    # keep only occupied circles with exactly two dots and order consistently
    def _order_red_pair(pair):
        # order by y increasing (so segment direction is consistent)
        a, b = pair
        return (a, b) if a.y <= b.y else (b, a)

    def _order_blue_pair(pair):
        # order by x increasing
        a, b = pair
        return (a, b) if a.x <= b.x else (b, a)

    r_lines = {cid: _order_red_pair(members) for cid, members in r_lines.items() if len(members) == 2}
    b_lines = {cid: _order_blue_pair(members) for cid, members in b_lines.items() if len(members) == 2}

    print(f"RED occupied circles: {len(r_lines)}, BLUE occupied circles: {len(b_lines)}")

    # Build a segment adjacency graph between RED and BLUE segments using shared vertices
    # Represent a segment as tuple (Color, circle_id)
    adjacency = defaultdict(list)  # seg -> list of neighbouring segs

    for cid, (u, w) in r_lines.items():
        # endpoints u,w each belong to a blue circle; connect red segment to those blue segments
        bu = background.vertex_circle_ids[u][Color.BLUE]
        bw = background.vertex_circle_ids[w][Color.BLUE]
        if bu in b_lines:
            adjacency[(Color.RED, cid)].append((Color.BLUE, bu))
        if bw in b_lines:
            adjacency[(Color.RED, cid)].append((Color.BLUE, bw))

    for cid, (u, w) in b_lines.items():
        ru = background.vertex_circle_ids[u][Color.RED]
        rw = background.vertex_circle_ids[w][Color.RED]
        if ru in r_lines:
            adjacency[(Color.BLUE, cid)].append((Color.RED, ru))
        if rw in r_lines:
            adjacency[(Color.BLUE, cid)].append((Color.RED, rw))

    # Now traverse alternating RED/BLUE cycles to produce oriented links
    visited_segments = set()
    oriented_links = []  # list of components; each component is list of (x,y) points in order

    def segment_endpoints(seg):
        color, cid = seg
        if color == Color.RED:
            return r_lines[cid]
        return b_lines[cid]

    for seg in list(adjacency.keys()):
        if seg in visited_segments:
            continue
        # start a new component from this segment
        comp_points = []
        cur_seg = seg
        # choose an orientation: pick the ordered pair endpoints and start at endpoint 0
        start_seg = cur_seg
        start_point = segment_endpoints(cur_seg)[0]
        cur_point = start_point
        while True:
            visited_segments.add(cur_seg)
            a, b = segment_endpoints(cur_seg)
            # determine which endpoint we're at and append next coordinate accordingly
            if cur_point == a:
                next_point = b
            elif cur_point == b:
                next_point = a
            else:
                # the current point should be one of endpoints; if not, pick a
                next_point = a
            # append the coordinate of the endpoint we traverse to (use center of face)
            comp_points.append((cur_point.x, cur_point.y))
            # move to adjacent segment across the vertex 'next_point'
            # find neighboring segments incident to next_point excluding cur_seg
            next_seg_candidates = []
            # other colour
            for neighbor in adjacency[cur_seg]:
                if neighbor not in visited_segments or True:
                    # neighbor is blue if cur_seg is red and vice versa
                    # need to check it actually shares the vertex 'next_point'
                    u, v = segment_endpoints(neighbor)
                    if next_point == u or next_point == v:
                        next_seg_candidates.append(neighbor)
            # pick the neighbor that is not cur_seg (it won't equal anyway)
            if not next_seg_candidates:
                # dead end: stop this component
                break
            # choose next segment (if multiple, pick the one not equal to cur_seg)
            next_seg = next_seg_candidates[0]
            # advance
            cur_seg = next_seg
            # set cur_point to the vertex on cur_seg that is not next_point (we enter the next segment at next_point, so current point becomes next_point)
            cur_point = next_point
            # if we returned to start segment and start point, stop
            if cur_seg == start_seg and cur_point == start_point:
                break
        if comp_points:
            oriented_links.append(comp_points)

    print(f"Found {len(oriented_links)} oriented link components")

    """
    # Try to hand these oriented link components to snappy. Different SnapPy APIs accept different input formats
    # Try the most common: giving a list of component polygons as lists of (x,y) pairs to snappy.Link
    try:
        from spherogram import Link 
        from spherogram import Crossing
        from snappy import Manifold 
        

        # Snappy's Link constructor accepts various formats; try feeding the list-of-components directly.
        L = Link()
        print("Constructed Snappy Link:", L)
        try:
            M = L.exterior()
            print("Exterior manifold created, volume:", M.volume())
        except Exception as exc:
            print("Snappy Link created but couldn't form exterior manifold:", exc)
    except Exception as exc:
        print("Could not construct a snappy.Link from oriented link data; returning oriented_links. Error:", exc)
        print(oriented_links)
    """
    # --- Added: compute crossings from oriented_links and build a PD-like structure ---
    try:
        import math
        from collections import defaultdict

        EPS = 1e-9

        def seg_intersection(p1, p2, q1, q2):
            # Parametric intersection of p1 + t*(p2-p1) and q1 + u*(q2-q1)
            (x1, y1), (x2, y2) = p1, p2
            (x3, y3), (x4, y4) = q1, q2
            dx1, dy1 = x2 - x1, y2 - y1
            dx2, dy2 = x4 - x3, y4 - y3
            denom = dx1 * dy2 - dy1 * dx2
            if abs(denom) < EPS:
                return None
            t = ((x3 - x1) * dy2 - (y3 - y1) * dx2) / denom
            u = ((x3 - x1) * dy1 - (y3 - y1) * dx1) / denom
            if -EPS <= t <= 1 + EPS and -EPS <= u <= 1 + EPS:
                ix = x1 + t * dx1
                iy = y1 + t * dy1
                return (max(0.0, min(1.0, t)), max(0.0, min(1.0, u)), (ix, iy))
            return None

        # Build segment list per component
        comp_segs = []  # list per component of segments ((p1,p2), seg_index)
        for comp in oriented_links:
            pts = comp
            segs = []
            L = len(pts)
            for i in range(L):
                p1 = pts[i]
                p2 = pts[(i + 1) % L]
                segs.append((p1, p2))
            comp_segs.append(segs)

        # Find all proper intersections between distinct segments (skip adjacent segments sharing a vertex)
        crossings = []
        for ci, segs_i in enumerate(comp_segs):
            for si, (p1, p2) in enumerate(segs_i):
                for cj, segs_j in enumerate(comp_segs):
                    # to avoid duplicate checks, only allow (ci,cj) with (ci<cj) or same but si<sj
                    for sj, (q1, q2) in enumerate(segs_j):
                        if (ci > cj) or (ci == cj and si >= sj):
                            continue
                        # skip adjacent segments in the same component (share endpoint)
                        if ci == cj:
                            if (si == sj) or (si == (sj + 1) % len(segs_j)) or (sj == (si + 1) % len(segs_i)):
                                continue
                        res = seg_intersection(p1, p2, q1, q2)
                        if res is not None:
                            t, u, pt = res
                            # ignore intersections at polygon vertices (t==0 or 1 or u==0 or 1)
                            if t <= EPS or t >= 1 - EPS or u <= EPS or u >= 1 - EPS:
                                continue
                            crossings.append({
                                'components': (ci, cj),
                                'segs': (si, sj),
                                'params': (t, u),
                                'point': pt,
                            })

        print(f"Detected {len(crossings)} geometric crossings")

        # For each component, collect the parameter values along the polygon (0..1 across each segment index)
        comp_params = []  # per comp: list of (seg_index, t, global_param)
        # We'll measure global parameter as (seg_index + t)/num_segments to sort along component
        for ci, segs in enumerate(comp_segs):
            n = len(segs)
            plist = []
            for cr in crossings:
                if cr['components'][0] == ci:
                    si = cr['segs'][0]
                    t = cr['params'][0]
                    plist.append((si, t, (si + t) / n))
                elif cr['components'][1] == ci:
                    sj = cr['segs'][1]
                    u = cr['params'][1]
                    plist.append((sj, u, (sj + u) / n))
            # always include 0 marker to make full coverage
            # sort by global_param
            plist.sort(key=lambda x: x[2])
            comp_params.append(plist)

        # Build arcs: intervals between consecutive crossing points along each component
        arc_id = 0
        comp_arcs = []  # per comp list of arcs: each arc is (start_global_param, end_global_param, arc_id)
        for ci, segs in enumerate(comp_segs):
            n = len(segs)
            pts = comp_params[ci]
            arcs = []
            if not pts:
                # no crossings on this component -> single arc
                arcs.append((0.0, 1.0, arc_id))
                arc_id += 1
            else:
                # compute ordered breakpoints (as global_param) and make intervals
                bps = [x[2] for x in pts]
                # ensure wrap-around by adding 1.0 (which is same as 0.0)
                loop = [0.0] + bps + [1.0]
                for k in range(len(loop) - 1):
                    start = loop[k]
                    end = loop[k + 1]
                    if end - start > EPS:
                        arcs.append((start, end, arc_id))
                        arc_id += 1
            comp_arcs.append(arcs)

        # helper to find arc id given component and a param (seg_index + t)/n (global)
        def find_arc(ci, seg_idx, t):
            n = len(comp_segs[ci])
            gp = (seg_idx + t) / n
            for (s, e, aid) in comp_arcs[ci]:
                if s - EPS <= gp <= e + EPS:
                    return aid
            # fallback: nearest
            best = min(comp_arcs[ci], key=lambda x: abs(((x[0]+x[1])/2) - gp))
            return best[2]

        # Build PD-like crossing tuples
        pd = []
        for cr in crossings:
            ci, cj = cr['components']
            si, sj = cr['segs']
            ti, uj = cr['params']
            xi, xj = cr['point'], cr['point']
            # For each component, incoming arc is the arc just before the crossing (we choose the arc whose end == gp),
            # outgoing arc is the arc just after the crossing. Using our arcs, incoming is the arc that contains a tiny epsilon less than gp,
            # outgoing contains a tiny epsilon more than gp.
            def arc_before_after(ci, seg_idx, t):
                n = len(comp_segs[ci])
                gp = (seg_idx + t) / n
                before_gp = gp - 1e-6
                after_gp = gp + 1e-6
                # wrap
                if before_gp < 0:
                    before_gp += 1.0
                if after_gp > 1:
                    after_gp -= 1.0
                aid_before = None
                aid_after = None
                for (s, e, aid) in comp_arcs[ci]:
                    if s - EPS <= before_gp <= e + EPS:
                        aid_before = aid
                    if s - EPS <= after_gp <= e + EPS:
                        aid_after = aid
                return aid_before if aid_before is not None else find_arc(ci, seg_idx, t), aid_after if aid_after is not None else find_arc(ci, seg_idx, t)

            a_in, a_out = arc_before_after(ci, si, ti)
            b_in, b_out = arc_before_after(cj, sj, uj)

            # decide which arc is 'over' by component index (simple consistent heuristic)
            if ci <= cj:
                over = (a_in, a_out)
                under = (b_in, b_out)
            else:
                over = (b_in, b_out)
                under = (a_in, a_out)

            pd.append((over[0], over[1], under[0], under[1]))

        print(f"Built PD guess with {len(pd)} crossings and {arc_id} arcs")

        # Attempt to construct spherogram.Link with a few common entrypoints
        try:
            from spherogram import Link
            constructed = False
            # try plain Link(pd)
            try:
                L2 = Link(pd)
                print("Constructed spherogram.Link(pd)")
                constructed = True
            except Exception:
                pass
            # try Link.from_pd
            if not constructed and hasattr(Link, 'from_pd'):
                try:
                    L2 = Link.from_pd(pd)
                    print("Constructed spherogram.Link.from_pd(pd)")
                    constructed = True
                except Exception:
                    pass
            # try alternative constructor signatures
            if not constructed:
                try:
                    L2 = Link(pd, 'PD')
                    print("Constructed spherogram.Link(pd, 'PD')")
                    constructed = True
                except Exception:
                    pass

            if constructed:
                try:
                    M2 = L2.exterior()
                    print("Exterior manifold from built Link, volume:", getattr(M2, 'volume', lambda: None)())
                except Exception as exc2:
                    print("Built Link but couldn't form exterior manifold:", exc2)
            else:
                print("Couldn't construct spherogram.Link from PD guess; printing PD for debugging:")
                print('pd =', pd)
                print('num_arcs =', arc_id)
        except Exception as exc2:
            print("spherogram not available or failed to construct Link. PD guess printed for manual use.")
            print('pd =', pd)
    except Exception as exc_all:
        print("Error computing crossings or building PD-like structure:", exc_all)


        
