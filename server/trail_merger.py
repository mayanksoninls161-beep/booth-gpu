
"""
trail_merger.py v3 — Skeletonize first, then merge lines
=========================================================
Pipeline:
  1. For each prediction polygon → skeletonize → extract ordered
     centerline with branch pruning → this is a "line segment".
  2. Each line segment has two real endpoints (skeleton tips).
  3. Color-cluster line segments by map color.
  4. Bridge/merge line segments by proximity of their skeleton
     endpoints (NOT polygon PCA endpoints). Same-side constraint.
  5. Stitch connected line segments into single polylines.
  6. Output: one polyline per merged trail.

No file I/O, no drawing.

Requires:
    pip install pillow numpy scikit-learn scikit-image
"""

import math
import colorsys
import logging
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from skimage.morphology import skeletonize

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────

_REF_DIAG = math.sqrt(2100 ** 2 + 1275 ** 2)  # ~2457 px at reference resolution

DEFAULT_CONFIG = {
    "max_color_clusters": 10,
    "color_merge_thresh": 0.25,      # L2 distance in norm-RGB; clusters closer than this are merged
    "merge_dist_px": None,           # None → auto-scaled from merge_dist_ratio
    "merge_dist_ratio": 150 / _REF_DIAG,  # ≈0.0611; calibrated at 2100×1275
    "cross_cluster_merge": True,
    "cross_merge_ratio": 0.15,
    "cross_merge_min": 10,
    "nms_iou_thresh": 1.0,
    "js_w": None,
    "js_h": None,
    "min_points": 5,
    "smooth_window": 5,
    "simplify_epsilon": 1.5,
    "skeleton_padding": 5,
    "densify_max_dist": 25,          # max gap (JS units) between consecutive output points
    # ── self-overlap removal (collapses "multiple lines over the same place") ──
    "self_overlap_lookback": 6,      # min loop size in vertices that can be excised
    "self_overlap_ratio": 0.17,      # threshold = merge_dist_px * ratio (perp. overlap radius)
    "self_overlap_min": 12.0,        # absolute floor (JS units) for the threshold
}


# ─────────────────────────────────────────────────────────────────
# GEOMETRY HELPERS
# ─────────────────────────────────────────────────────────────────

def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _point_seg_dist(p, a, b):
    """Shortest distance from point p to the line *segment* a-b (clamped).

    Used by self-overlap removal so a vertex that lands near an earlier
    *edge* of the polyline — not just near an earlier vertex — is detected.
    This is what catches near-parallel overlapping runs (the dashed
    rail-trail tangle) that pure vertex-to-vertex checks miss.
    """
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        return euclidean(p, a)
    t = ((px - ax) * dx + (py - ay) * dy) / L2
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


def js_to_px(x_js, y_js, sx, sy):
    return (int(x_js * sx), int(y_js * sy))


# ─────────────────────────────────────────────────────────────────
# COLOR SAMPLING
# ─────────────────────────────────────────────────────────────────

def sample_map_color(pts, map_img, sx, sy):
    if len(pts) < 3:
        return (180, 180, 180)

    mask = Image.new("L", map_img.size, 0)
    draw = ImageDraw.Draw(mask)
    pts_px = [js_to_px(p[0], p[1], sx, sy) for p in pts]
    draw.polygon(pts_px, fill=255)

    mask_arr = np.array(mask)
    map_arr = np.array(map_img)
    y_idx, x_idx = np.where(mask_arr == 255)

    if len(y_idx) == 0:
        return (180, 180, 180)

    pixels = map_arr[y_idx, x_idx]
    r = pixels[:, 0].astype(int)
    g = pixels[:, 1].astype(int)
    b = pixels[:, 2].astype(int)
    sat = np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])

    top_k = max(1, len(pixels) // 5)
    top_indices = np.argsort(sat)[-top_k:]
    top_pixels = pixels[top_indices]

    return (
        int(np.median(top_pixels[:, 0])),
        int(np.median(top_pixels[:, 1])),
        int(np.median(top_pixels[:, 2])),
    )


# ─────────────────────────────────────────────────────────────────
# SKELETON EXTRACTION — polygon → binary mask → skeleton → path
# ─────────────────────────────────────────────────────────────────

def _polygon_to_mask(pts_js, sx, sy, img_w, img_h, padding=5):
    pts_px = [(p[0] * sx, p[1] * sy) for p in pts_js]
    xs = [p[0] for p in pts_px]
    ys = [p[1] for p in pts_px]
    x_min = max(0, int(min(xs)) - padding)
    y_min = max(0, int(min(ys)) - padding)
    x_max = min(img_w - 1, int(max(xs)) + padding)
    y_max = min(img_h - 1, int(max(ys)) + padding)

    crop_w = x_max - x_min + 1
    crop_h = y_max - y_min + 1
    if crop_w < 1 or crop_h < 1:
        return np.zeros((1, 1), dtype=bool), x_min, y_min

    mask_img = Image.new('L', (crop_w, crop_h), 0)
    d = ImageDraw.Draw(mask_img)
    local_pts = [(p[0] - x_min, p[1] - y_min) for p in pts_px]
    if len(local_pts) >= 3:
        d.polygon(local_pts, fill=255)
    return np.array(mask_img) > 0, x_min, y_min


def _skeleton_to_ordered_path(skel):
    """
    Skeleton → ordered pixel path with branch pruning.
    Decomposes at junctions, finds longest spine, prunes spurs.
    """
    rows, cols = np.where(skel)
    if len(rows) == 0:
        return []

    coords = list(zip(rows.tolist(), cols.tolist()))
    coord_to_idx = {c: i for i, c in enumerate(coords)}
    n_pts = len(coords)

    adj = defaultdict(set)
    for i, (r, c) in enumerate(coords):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nb = (r + dr, c + dc)
                if nb in coord_to_idx:
                    j = coord_to_idx[nb]
                    adj[i].add(j)
                    adj[j].add(i)

    degree = {i: len(adj[i]) for i in range(n_pts)}
    junctions = {i for i in range(n_pts) if degree[i] >= 3}

    # ── No junctions: simple BFS diameter ──────────────────────
    if not junctions:
        def bfs_far(start):
            vis = {start: None}
            q = deque([start])
            far = start
            while q:
                node = q.popleft()
                far = node
                for nb in adj[node]:
                    if nb not in vis:
                        vis[nb] = node
                        q.append(nb)
            return far, vis

        f1, _ = bfs_far(0)
        f2, parents = bfs_far(f1)
        path = []
        cur = f2
        while cur is not None:
            path.append(coords[cur])
            cur = parents[cur]
        path.reverse()
        return path

    # ── Decompose into branches ────────────────────────────────
    endpoints = {i for i in range(n_pts) if degree[i] == 1}
    branches = []
    visited_edges = set()
    starts = junctions | endpoints

    for s in starts:
        for nb in adj[s]:
            edge = (min(s, nb), max(s, nb))
            if edge in visited_edges:
                continue
            branch = [s]
            prev, cur = s, nb
            while cur not in junctions and cur not in endpoints:
                visited_edges.add((min(prev, cur), max(prev, cur)))
                branch.append(cur)
                nxt = [n for n in adj[cur] if n != prev]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
            visited_edges.add((min(prev, cur), max(prev, cur)))
            branch.append(cur)
            branches.append(branch)

    if not branches:
        def bfs_far(start):
            vis = {start: None}
            q = deque([start])
            far = start
            while q:
                node = q.popleft()
                far = node
                for nb in adj[node]:
                    if nb not in vis:
                        vis[nb] = node
                        q.append(nb)
            return far, vis
        f1, _ = bfs_far(0)
        f2, parents = bfs_far(f1)
        path = []
        cur = f2
        while cur is not None:
            path.append(coords[cur])
            cur = parents[cur]
        path.reverse()
        return path

    # ── Branch-level graph → longest spine ─────────────────────
    branch_adj = defaultdict(list)
    for bi, branch in enumerate(branches):
        a, b = branch[0], branch[-1]
        branch_adj[a].append((b, len(branch), bi))
        branch_adj[b].append((a, len(branch), bi))

    graph_nodes = set()
    for branch in branches:
        graph_nodes.add(branch[0])
        graph_nodes.add(branch[-1])

    br_deg = defaultdict(int)
    for node in graph_nodes:
        br_deg[node] = len(branch_adj[node])
    graph_eps = [n for n in graph_nodes if br_deg[n] == 1]
    if len(graph_eps) < 2:
        graph_eps = list(graph_nodes)

    best_branches = []
    best_len = 0

    for sn in graph_eps:
        vis = {sn: (None, None)}
        q = deque([(sn, 0)])
        far_node, far_len = sn, 0
        while q:
            node, cl = q.popleft()
            if cl > far_len:
                far_len = cl
                far_node = node
            for (other, blen, bi) in branch_adj[node]:
                if other not in vis:
                    vis[other] = (node, bi)
                    q.append((other, cl + blen))

        if far_len > best_len:
            best_len = far_len
            pb = []
            cur = far_node
            while vis[cur][0] is not None:
                parent, bi = vis[cur]
                pb.append(bi)
                cur = parent
            best_branches = pb

    if not best_branches:
        longest = max(branches, key=len)
        return [coords[i] for i in longest]

    # ── Walk spine branches in order ───────────────────────────
    spine_set = set(best_branches)
    spine_nc = defaultdict(int)
    for bi in best_branches:
        spine_nc[branches[bi][0]] += 1
        spine_nc[branches[bi][-1]] += 1
    spine_ends = [n for n, c in spine_nc.items() if c == 1]
    if not spine_ends:
        spine_ends = [branches[best_branches[0]][0]]

    spine_px = []
    used = set()
    cur_node = spine_ends[0]

    while True:
        found = False
        for (other, blen, bi) in branch_adj[cur_node]:
            if bi in spine_set and bi not in used:
                used.add(bi)
                branch = branches[bi]
                ordered = branch if branch[0] == cur_node else branch[::-1]
                if spine_px:
                    spine_px.extend([coords[i] for i in ordered[1:]])
                else:
                    spine_px.extend([coords[i] for i in ordered])
                cur_node = ordered[-1]
                found = True
                break
        if not found:
            break

    return spine_px if spine_px else [coords[i] for i in max(branches, key=len)]


def _smooth_path(pts, window=5):
    if len(pts) <= window:
        return pts
    arr = np.array(pts, dtype=float)
    smoothed = np.copy(arr)
    half = window // 2
    for i in range(half, len(arr) - half):
        smoothed[i] = arr[i - half:i + half + 1].mean(axis=0)
    smoothed[:half] = arr[:half]
    smoothed[-half:] = arr[-half:]
    return [tuple(row) for row in smoothed]


def _simplify_rdp(pts, epsilon=1.5):
    if len(pts) <= 2:
        return pts
    start = np.array(pts[0], dtype=float)
    end = np.array(pts[-1], dtype=float)
    lv = end - start
    ll = np.linalg.norm(lv)
    if ll < 1e-9:
        return [pts[0], pts[-1]]
    lu = lv / ll

    md, mi = 0.0, 0
    for i in range(1, len(pts) - 1):
        p = np.array(pts[i], dtype=float) - start
        dist = np.linalg.norm(p - lu * np.dot(p, lu))
        if dist > md:
            md, mi = dist, i

    if md > epsilon:
        left = _simplify_rdp(pts[:mi + 1], epsilon)
        right = _simplify_rdp(pts[mi:], epsilon)
        return left[:-1] + right
    return [pts[0], pts[-1]]


def _densify(pts, max_dist):
    """Insert linearly-interpolated points so no consecutive gap exceeds max_dist."""
    if len(pts) < 2 or max_dist <= 0:
        return pts
    result = [pts[0]]
    for i in range(1, len(pts)):
        p0 = np.array(pts[i - 1], dtype=float)
        p1 = np.array(pts[i], dtype=float)
        d = np.linalg.norm(p1 - p0)
        if d > max_dist:
            n_steps = int(math.ceil(d / max_dist))
            for k in range(1, n_steps):
                t = k / n_steps
                ip = p0 + t * (p1 - p0)
                result.append((round(float(ip[0]), 3), round(float(ip[1]), 3)))
        result.append(pts[i])
    return result


def extract_centerline(polygon_pts, sx, sy, img_w, img_h, cfg):
    """
    Polygon → skeleton → ordered centerline in JSON coords.
    Returns list of (x_js, y_js). Endpoints are skeleton tips.
    """
    if len(polygon_pts) < 3:
        return list(polygon_pts)

    mask, off_x, off_y = _polygon_to_mask(
        polygon_pts, sx, sy, img_w, img_h,
        padding=cfg.get("skeleton_padding", 5)
    )
    if mask.sum() < 3:
        return [polygon_pts[0], polygon_pts[len(polygon_pts) // 2]]

    skel = skeletonize(mask)
    if skel.sum() == 0:
        return [polygon_pts[0], polygon_pts[len(polygon_pts) // 2]]

    ordered_px = _skeleton_to_ordered_path(skel)
    if len(ordered_px) < 2:
        return [polygon_pts[0], polygon_pts[len(polygon_pts) // 2]]

    ordered_px = _smooth_path(ordered_px, window=cfg.get("smooth_window", 5))
    ordered_px = _simplify_rdp(ordered_px, epsilon=cfg.get("simplify_epsilon", 1.5))

    centerline_js = []
    for (row, col) in ordered_px:
        x_js = (col + off_x) / sx
        y_js = (row + off_y) / sy
        centerline_js.append((round(x_js, 3), round(y_js, 3)))

    max_gap = cfg.get("densify_max_dist", 1)
    if max_gap and max_gap > 0:
        centerline_js = _densify(centerline_js, max_gap)

    return centerline_js


# ─────────────────────────────────────────────────────────────────
# COLOR CLUSTERING — 7D features
# ─────────────────────────────────────────────────────────────────

def cluster_by_color(segments, max_clusters, color_merge_thresh=0.25):
    """
    KMeans on a 4D hue-sat-val feature space, followed by a post-merge pass
    that collapses any two clusters whose RGB centroids are within
    color_merge_thresh (L2, normalised 0-1).  This prevents silhouette-driven
    over-splitting from keeping visually identical trail colors in separate
    clusters.
    """
    rgb_arr = np.array([s["map_color"] for s in segments], dtype=float) / 255.0

    features = []
    for rn, gn, bn in rgb_arr:
        h, sat, v = colorsys.rgb_to_hsv(rn, gn, bn)
        hx = math.cos(h * 2 * math.pi) * sat
        hy = math.sin(h * 2 * math.pi) * sat
        features.append([hx, hy, sat, v])

    mat = np.array(features, dtype=float)
    best_score, best_labels = -1.0, None
    max_k = min(max_clusters, len(segments) - 1)

    if max_k >= 2:
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(mat)
            if len(set(labels)) < 2:
                continue
            try:
                score = silhouette_score(mat, labels)
                if score > best_score:
                    best_score, best_labels = score, labels.copy()
            except ValueError:
                continue

    if best_labels is None:
        best_labels = np.zeros(len(segments), dtype=int)

    # ── Post-merge: collapse clusters whose RGB centroids are too similar ──
    unique = sorted(set(best_labels.tolist()))
    if len(unique) > 1:
        centroids = {}
        for cl in unique:
            mask = best_labels == cl
            centroids[cl] = rgb_arr[mask].mean(axis=0)

        parent = {cl: cl for cl in unique}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, ci in enumerate(unique):
            for cj in unique[i + 1:]:
                if np.linalg.norm(centroids[ci] - centroids[cj]) < color_merge_thresh:
                    ri, rj = find(ci), find(cj)
                    if ri != rj:
                        parent[rj] = ri

        # Remap to contiguous 0-based labels
        canon = {cl: find(cl) for cl in unique}
        new_ids = {old: new for new, old in enumerate(sorted(set(canon.values())))}
        best_labels = np.array([new_ids[canon[lbl]] for lbl in best_labels.tolist()])

    n_final = len(set(best_labels.tolist()))
    return best_labels, n_final, None


# ─────────────────────────────────────────────────────────────────
# UNION-FIND
# ─────────────────────────────────────────────────────────────────

class UnionFind:
    def __init__(self, n):
        self._parent = list(range(n))

    def find(self, x):
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra

    def same(self, a, b):
        return self.find(a) == self.find(b)


# ─────────────────────────────────────────────────────────────────
# BRIDGING — uses SKELETON endpoints, not polygon PCA endpoints
# ─────────────────────────────────────────────────────────────────

def find_bridges(segments, merge_dist, cross_merge_dist, cross_cluster_merge):
    """
    Bridge line segments using their skeleton endpoints (ep1, ep2).
    Each segment has exactly 2 ports. Same-side constraint is enforced
    by the ep_free dict (once a port is used, it's locked).
    """
    n = len(segments)
    uf = UnionFind(n)
    ep_free = {}
    for i in range(n):
        ep_free[(i, "1")] = True
        ep_free[(i, "2")] = True

    bridges = []

    def _candidates(same_cluster, max_dist):
        cands = []
        for i in range(n):
            for j in range(i + 1, n):
                same = segments[i]["cluster"] == segments[j]["cluster"]
                if same_cluster and not same:
                    continue
                if not same_cluster and same:
                    continue
                # Try all 4 endpoint combinations
                for ea, eb, pa, pb in [
                    ("1", "1", segments[i]["ep1"], segments[j]["ep1"]),
                    ("1", "2", segments[i]["ep1"], segments[j]["ep2"]),
                    ("2", "1", segments[i]["ep2"], segments[j]["ep1"]),
                    ("2", "2", segments[i]["ep2"], segments[j]["ep2"]),
                ]:
                    d = euclidean(pa, pb)
                    if d < max_dist:
                        cands.append((d, i, j, ea, eb, pa, pb))
        cands.sort()
        return cands

    def _greedy(candidates, cross):
        count = 0
        for d, i, j, ea, eb, pa, pb in candidates:
            if not (ep_free.get((i, ea)) and ep_free.get((j, eb))):
                continue
            if uf.same(i, j):
                continue

            cl = segments[i]["cluster"]
            if cross:
                cl = (segments[i]["cluster"]
                      if segments[i]["confidence"] >= segments[j]["confidence"]
                      else segments[j]["cluster"])

            bridges.append({
                "seg_i": i, "seg_j": j,
                "ep_i": ea, "ep_j": eb,
                "pa": pa, "pb": pb,
                "gap_px": round(d, 2),
                "cluster": cl, "cross": cross,
            })
            ep_free[(i, ea)] = False
            ep_free[(j, eb)] = False
            uf.union(i, j)
            count += 1
        return count

    same_cands = _candidates(same_cluster=True, max_dist=merge_dist)
    _greedy(same_cands, cross=False)

    if cross_cluster_merge:
        cross_cands = _candidates(same_cluster=False, max_dist=cross_merge_dist)
        _greedy(cross_cands, cross=True)

    return bridges, uf


# ─────────────────────────────────────────────────────────────────
# CHAIN WALKING — walk bridge graph, stitch centerlines
# ─────────────────────────────────────────────────────────────────

def _build_adj(bridges):
    adj = defaultdict(list)
    for b in bridges:
        i, j = b["seg_i"], b["seg_j"]
        adj[i].append((j, b["ep_i"], b["ep_j"], b))
        adj[j].append((i, b["ep_j"], b["ep_i"], b))
    return adj


def _find_diameter_chain(seg_indices, adj):
    """
    Two-pass BFS (tree diameter) to find the longest path through the bridge
    graph, then return it as an ordered chain of (seg_idx, entry_port,
    exit_port, bridge) tuples — same format that stitch_group_centerline
    expects.

    At Y-junctions the shorter branches are dropped, giving a single
    non-backtracking polyline from one leaf to the farthest leaf.
    """
    if len(seg_indices) == 1:
        return [(seg_indices[0], None, None, None)]

    seg_set = set(seg_indices)

    def bfs_far(start):
        dist = {start: 0}
        # arrival[node] = (parent_seg, exit_port_of_parent, entry_port_of_node, bridge)
        arrival = {start: None}
        q = deque([start])
        farthest, far_dist = start, 0
        while q:
            node = q.popleft()
            for nbr, my_port, their_port, b in adj[node]:
                if nbr in seg_set and nbr not in dist:
                    dist[nbr] = dist[node] + 1
                    arrival[nbr] = (node, my_port, their_port, b)
                    if dist[nbr] > far_dist:
                        far_dist, farthest = dist[nbr], nbr
                    q.append(nbr)
        return farthest, arrival

    f1, _ = bfs_far(seg_indices[0])
    f2, arrival = bfs_far(f1)

    # Reconstruct path f1 → f2
    path = []
    cur = f2
    while cur is not None:
        path.append((cur, arrival[cur]))
        info = arrival[cur]
        cur = info[0] if info else None
    path.reverse()

    # Convert to (seg_idx, entry_port, exit_port, bridge) chain
    chain = []
    for i, (seg_idx, info) in enumerate(path):
        entry_port = info[2] if info else None   # their_port = port on this seg
        bridge = info[3] if info else None

        exit_port = None
        if i + 1 < len(path):
            next_seg_idx = path[i + 1][0]
            for nbr, my_port, _, _ in adj[seg_idx]:
                if nbr == next_seg_idx:
                    exit_port = my_port
                    break

        chain.append((seg_idx, entry_port, exit_port, bridge))

    return chain


def _trim_junction_overlap(full_line, cl, tol=1.0):
    """
    Remove centerline overlap at a segment junction to prevent U-shaped artifacts.

    When adjacent polygon centerlines overlap, the stitched path briefly
    backtracks: full_line ends at P, but cl starts at Q that is "behind" P
    in the direction of travel.  This trims the tail of full_line backward
    until Q is no longer behind, then trims the head of cl for any remaining
    overlap, so the junction always moves forward.
    """
    if len(full_line) < 2 or len(cl) < 2:
        return full_line, cl

    result = list(full_line)
    cl_work = list(cl)

    # Trim tail of result while cl_work[0] is behind the current tail end
    for _ in range(len(result) - 1):
        if len(result) < 2:
            break
        travel = np.array(result[-1], dtype=float) - np.array(result[-2], dtype=float)
        t_len = np.linalg.norm(travel)
        if t_len < 1e-9:
            result.pop()
            continue
        proj = np.dot(
            np.array(cl_work[0], dtype=float) - np.array(result[-1], dtype=float),
            travel / t_len,
        )
        if proj >= -tol:
            break
        result.pop()

    # Trim head of cl_work while cl_work[0] is still behind result[-1]
    if len(result) >= 2:
        travel = np.array(result[-1], dtype=float) - np.array(result[-2], dtype=float)
        t_len = np.linalg.norm(travel)
        if t_len > 1e-9:
            travel_unit = travel / t_len
            for _ in range(len(cl_work) - 1):
                if len(cl_work) < 2:
                    break
                proj = np.dot(
                    np.array(cl_work[0], dtype=float) - np.array(result[-1], dtype=float),
                    travel_unit,
                )
                if proj >= -tol:
                    break
                cl_work.pop(0)

    return result, cl_work


def stitch_group_centerline(seg_indices, segments, adj):
    """
    Walk the bridge chain and stitch centerlines into a single polyline.

    Orientation per segment:
      - First segment: exit_port determines which end connects forward.
      - Subsequent: whichever endpoint is closest to the current tail becomes
        cl[0] (handles all four bridge-endpoint combinations).

    Nearest-attachment trim:
      If cl[0] is closer to an EARLIER point in full_line than to the tail,
      the segment wants to re-connect mid-path (a loop would form).  We trim
      full_line back to that earlier attachment point so the path never
      doubles back — implementing "merge from the nearest point available".
    """
    if len(seg_indices) == 1:
        return list(segments[seg_indices[0]]["centerline"])

    chain = _find_diameter_chain(seg_indices, adj)
    full_line = []

    for step_idx, (seg_idx, entry_port, exit_port, bridge) in enumerate(chain):
        seg = segments[seg_idx]
        cl = list(seg["centerline"])

        if len(cl) < 2:
            cl = [seg["ep1"], seg["ep2"]]

        if not full_line:
            # First segment: put the exit (connecting) end last.
            need_reverse = (exit_port == "1") if exit_port is not None else False
        else:
            # Subsequent segments: nearest endpoint to current tail goes first.
            d_start = euclidean(full_line[-1], cl[0])
            d_end   = euclidean(full_line[-1], cl[-1])
            need_reverse = d_end < d_start

        if need_reverse:
            cl = cl[::-1]

        if full_line:
            # ── Nearest-attachment trim ─────────────────────────────
            # If cl[0] is very close to an EARLIER point in full_line
            # (much closer than to the current tail), trim full_line
            # back to that earlier attachment so the path never loops.
            # Conservative threshold: only trigger when the earlier
            # point is within 10 JS units AND at least 4× closer than
            # the current tail — catches genuine re-visits without
            # harming dense winding trail networks.
            tail_dist = euclidean(full_line[-1], cl[0])
            lookback  = min(80, len(full_line) - 1)
            best_idx  = len(full_line) - 1
            best_dist = tail_dist
            for k in range(len(full_line) - 2,
                           max(-1, len(full_line) - 1 - lookback), -1):
                d = euclidean(full_line[k], cl[0])
                # Must be meaningfully closer AND within a tight absolute
                # radius to avoid false triggers in dense networks.
                if d < best_dist * 0.25 and d < 12.0:
                    best_dist = d
                    best_idx  = k
            if best_idx < len(full_line) - 1:
                full_line = full_line[:best_idx + 1]

            # ── Normal overlap trim + append ────────────────────────
            full_line, cl = _trim_junction_overlap(full_line, cl)
            if not cl:
                continue
            if euclidean(full_line[-1], cl[0]) < 1.0:
                full_line.extend(cl[1:])
            else:
                full_line.extend(cl)
        else:
            full_line.extend(cl)

    return full_line


def _remove_backtracking(pts, lookback=35, threshold=15.0):
    """
    Excise loop / backtracking sections from a polyline, returning ONE clean line.

    When a later point returns within `threshold` distance of a point that is
    at least `lookback` steps earlier, the intermediate detour is removed:

        pts[0..j] + pts[i..]   (the looping section pts[j+1..i-1] is dropped)

    This is applied repeatedly until no more loops remain, yielding a single
    forward-moving polyline with no backtracking — never multiple separate
    records that would be drawn as overlapping lines.
    """
    if len(pts) < lookback + 5:
        return list(pts)

    result = list(pts)

    for _ in range(20):   # max passes — almost always done in 1–3
        if len(result) < lookback + 5:
            break
        excised = False
        for i in range(lookback, len(result)):
            for j in range(0, i - lookback):
                if euclidean(result[i], result[j]) < threshold:
                    # Excise the backtrack: stitch j directly to i
                    result = result[:j + 1] + result[i:]
                    excised = True
                    break
            if excised:
                break
        if not excised:
            break

    return result


def _remove_self_overlap(pts, lookback=6, threshold=16.0, max_passes=40):
    """Collapse 'multiple lines over the same place' into one clean polyline.

    Unlike _remove_backtracking (vertex-to-vertex), this measures the distance
    from each later vertex to each earlier *edge* (point-to-segment). When the
    path returns within `threshold` of an edge at least `lookback` vertices back,
    the detour in between is excised and the two clean ends are rejoined:

        pts[0..k]  +  pts[i..]        (the tangled run pts[k+1..i-1] is dropped)

    `threshold` is the perpendicular overlap radius. Keep it well below the
    spacing between legitimate switchback legs so real winding trails (e.g.
    Greylock) are preserved, while near-on-top-of-each-other dashed-trail
    segments collapse. `lookback` is the minimum loop size (in vertices) that
    can be removed; small enough to catch a tight knot at a single crossing,
    large enough that a straight densified run never matches itself.
    """
    if len(pts) < lookback + 3:
        return list(pts)

    result = list(pts)
    for _ in range(max_passes):
        if len(result) < lookback + 3:
            break
        cut = None
        for i in range(lookback + 1, len(result)):
            # earliest qualifying edge → removes the largest loop first
            for k in range(0, i - lookback):
                if _point_seg_dist(result[i], result[k], result[k + 1]) < threshold:
                    cut = (k, i)
                    break
            if cut:
                break
        if not cut:
            break
        k, i = cut
        result = result[:k + 1] + result[i:]

    return result


# ─────────────────────────────────────────────────────────────────
# NMS
# ─────────────────────────────────────────────────────────────────

def _bbox_from_pts(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xn, xx = min(xs), max(xs)
    yn, yx = min(ys), max(ys)
    return (xn + xx) / 2, (yn + yx) / 2, xx - xn, yx - yn


def _iou_bbox(a, b):
    ax1, ax2 = a["x"] - a["width"] / 2, a["x"] + a["width"] / 2
    ay1, ay2 = a["y"] - a["height"] / 2, a["y"] + a["height"] / 2
    bx1, bx2 = b["x"] - b["width"] / 2, b["x"] + b["width"] / 2
    by1, by2 = b["y"] - b["height"] / 2, b["y"] + b["height"] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def apply_nms(records, iou_thresh):
    sr = sorted(records, key=lambda r: r["confidence"], reverse=True)
    kept, suppressed = [], set()
    for i, rec in enumerate(sr):
        if i in suppressed:
            continue
        kept.append(rec)
        for j in range(i + 1, len(sr)):
            if j not in suppressed and _iou_bbox(rec, sr[j]) > iou_thresh:
                suppressed.add(j)
    return kept


# ─────────────────────────────────────────────────────────────────
# JS DIMENSIONS AUTO-DETECT
# ─────────────────────────────────────────────────────────────────

def detect_js_dimensions(trail_block):
    image_info = trail_block.get("image", {})
    js_w = image_info.get("width")
    js_h = image_info.get("height")
    if js_w and js_h:
        return int(js_w), int(js_h)

    predictions = trail_block.get("predictions", [])
    if predictions:
        mx, my = 0.0, 0.0
        for pred in predictions:
            for pt in pred.get("points", []):
                mx = max(mx, pt.get("x", 0))
                my = max(my, pt.get("y", 0))
        if mx > 0 and my > 0:
            return int(math.ceil(mx * 1.05)), int(math.ceil(my * 1.05))

    return 2100, 1275


# ─────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────

def merge_trails(predictions, map_image, config=None, trail_block=None):
    """
    Pipeline:
      1. Polygon → skeleton → centerline (per segment)
      2. Endpoints = skeleton tips (first/last points of centerline)
      3. Color cluster segments
      4. Bridge segments by skeleton endpoint proximity
      5. Walk chains, stitch centerlines
      6. Output: one polyline per merged trail

    Returns list of merged prediction dicts.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    # ── Resolve JS coordinate space ────────────────────────────
    js_w, js_h = cfg["js_w"], cfg["js_h"]
    if js_w is None or js_h is None:
        src = trail_block if trail_block else {"predictions": predictions}
        js_w, js_h = detect_js_dimensions(src)

    if cfg["merge_dist_px"] is None:
        diag = math.sqrt(js_w ** 2 + js_h ** 2)
        cfg["merge_dist_px"] = diag * cfg["merge_dist_ratio"]

    img_w, img_h = map_image.size
    sx = img_w / js_w
    sy = img_h / js_h

    log.info("JS: %dx%d | Image: %dx%d | Scale: sx=%.4f sy=%.4f | merge_dist_px=%.1f",
             js_w, js_h, img_w, img_h, sx, sy, cfg['merge_dist_px'])

    # ══════════════════════════════════════════════════════════════
    # STEP 1: Polygon → Skeleton → Centerline (per prediction)
    # ══════════════════════════════════════════════════════════════
    log.info("[1/5] Skeletonizing %d predictions", len(predictions))

    segments = []
    for i, pred in enumerate(predictions):
        raw_pts = pred.get("points", [])
        if not raw_pts:
            continue

        polygon_pts = [(p["x"], p["y"]) for p in raw_pts]

        # Sample color from polygon area
        map_color = sample_map_color(polygon_pts, map_image, sx, sy)

        # Extract skeleton centerline
        centerline = extract_centerline(polygon_pts, sx, sy, img_w, img_h, cfg)

        if len(centerline) < 2:
            continue

        # Endpoints ARE the skeleton tips — first and last point
        ep1 = centerline[0]
        ep2 = centerline[-1]

        segments.append({
            "seg_idx": len(segments),
            "orig_pred_idx": i,
            "confidence": pred.get("confidence", 0.0),
            "polygon_pts": polygon_pts,
            "centerline": centerline,
            "ep1": ep1,          # skeleton tip 1 (first point)
            "ep2": ep2,          # skeleton tip 2 (last point)
            "map_color": map_color,
            "cluster": None,
        })

    log.info("Segments with valid centerlines: %d", len(segments))

    if not segments:
        return []

    # ══════════════════════════════════════════════════════════════
    # STEP 2: Color clustering
    # ══════════════════════════════════════════════════════════════
    log.info("[2/5] Clustering by color")

    labels, n_clusters, _ = cluster_by_color(
        segments, cfg["max_color_clusters"], cfg["color_merge_thresh"]
    )
    for idx, seg in enumerate(segments):
        seg["cluster"] = int(labels[idx])

    log.info("Clusters: %d", n_clusters)

    # ══════════════════════════════════════════════════════════════
    # STEP 3: Bridge segments by skeleton endpoint proximity
    # ══════════════════════════════════════════════════════════════
    log.info("[3/5] Bridging by skeleton endpoints")

    cross_dist = max(
        cfg["cross_merge_min"],
        cfg["merge_dist_px"] * cfg["cross_merge_ratio"],
    )
    bridges, uf = find_bridges(
        segments,
        merge_dist=cfg["merge_dist_px"],
        cross_merge_dist=cross_dist,
        cross_cluster_merge=cfg["cross_cluster_merge"],
    )

    n_same = sum(1 for b in bridges if not b["cross"])
    n_cross = sum(1 for b in bridges if b["cross"])
    log.info("Bridges: %d (same=%d, cross=%d)", len(bridges), n_same, n_cross)

    # ══════════════════════════════════════════════════════════════
    # STEP 4: Group & stitch
    # ══════════════════════════════════════════════════════════════
    log.info("[4/5] Stitching centerlines")

    groups = defaultdict(list)
    for idx in range(len(segments)):
        groups[uf.find(idx)].append(idx)

    adj = _build_adj(bridges)

    log.info("Groups (merged trails): %d", len(groups))

    # ══════════════════════════════════════════════════════════════
    # STEP 5: Build output records
    # ══════════════════════════════════════════════════════════════
    log.info("[5/5] Building output")

    # Backtracking-removal threshold: excise any section that returns within
    # this distance of a point ≥35 steps earlier.  Tight value (~10% of the
    # merge distance) catches only near-exact coordinate repeats from stitching
    # errors while leaving legitimate winding / loop trails untouched.
    back_threshold = max(10.0, cfg["merge_dist_px"] * 0.10)

    merged_records = []
    for root, seg_indices in groups.items():
        source_preds = [predictions[segments[i]["orig_pred_idx"]]
                        for i in seg_indices]
        best_conf = max(p.get("confidence", 0.0) for p in source_preds)
        rep_pred = max(source_preds, key=lambda p: p.get("confidence", 0.0))

        # Stitch centerlines
        centerline = stitch_group_centerline(seg_indices, segments, adj)
        if not centerline:
            continue

        # Fill bridge gaps between stitched segments
        max_gap = cfg.get("densify_max_dist", 1)
        if max_gap and max_gap > 0 and len(seg_indices) > 1:
            centerline = _densify(centerline, max_gap)
            centerline = _smooth_path(centerline, window=cfg.get("smooth_window", 5))

        # Remove any backtracking loops, producing ONE clean polyline.
        # Never splits into multiple records — avoids "multiple lines" artifacts.
        centerline = _remove_backtracking(
            centerline,
            lookback=35,
            threshold=back_threshold,
        )

        # Edge-based self-overlap removal: collapses near-parallel / tangled
        # runs (e.g. the dashed rail-trail cluster at a road crossing) that the
        # vertex-only pass above leaves behind. Catches points landing near an
        # earlier *edge*, not just an earlier vertex.
        so_thresh = max(
            cfg.get("self_overlap_min", 12.0),
            cfg["merge_dist_px"] * cfg.get("self_overlap_ratio", 0.17),
        )
        centerline = _remove_self_overlap(
            centerline,
            lookback=cfg.get("self_overlap_lookback", 6),
            threshold=so_thresh,
        )
        # Re-fill the gap left by any excision so the rejoin draws smoothly.
        max_gap = cfg.get("densify_max_dist", 1)
        if max_gap and max_gap > 0 and len(centerline) > 1:
            centerline = _densify(centerline, max_gap)

        if not centerline:
            continue

        cx, cy, w, h = _bbox_from_pts(centerline)

        merged_records.append({
            "x": round(cx, 3),
            "y": round(cy, 3),
            "width": round(w, 3),
            "height": round(h, 3),
            "confidence": round(best_conf, 6),
            "class": rep_pred.get("class", "trails"),
            "class_id": rep_pred.get("class_id", 0),
            "detection_id": rep_pred.get("detection_id", f"merged_{root:04d}"),
            "parent_id": rep_pred.get("parent_id", None),
            "class_confidence": rep_pred.get("class_confidence", None),
            "points": [
                {"x": round(p[0], 3), "y": round(p[1], 3)} for p in centerline
            ],
            "_n_segments": len(seg_indices),
            "_segment_ids": seg_indices,
        })

    # Filter small trails
    min_pts = cfg.get("min_points", 5)
    before = len(merged_records)
    merged_records = [r for r in merged_records if len(r["points"]) >= min_pts]
    dropped = before - len(merged_records)
    if dropped:
        log.info("Dropped %d trails with < %d points", dropped, min_pts)

    # NMS
    output = apply_nms(merged_records, cfg["nms_iou_thresh"])

    log.info("Final: %d merged trails (from %d predictions, %d segments)",
             len(output), len(predictions), len(segments))

    return output