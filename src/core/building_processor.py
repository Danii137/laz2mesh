"""
Building reconstruction as watertight solids.

Pipelines:
1) Cluster + plane-based reconstruction from LAS class 6 points.
2) External building footprint extrusion from GeoJSON.
"""

from __future__ import annotations

import json
import re

import numpy as np
from scipy.spatial import ConvexHull, Delaunay, QhullError, cKDTree

try:
    from shapely import concave_hull as _shapely_concave_hull
    from shapely.geometry import MultiPoint as _ShapelyMultiPoint
    from shapely.geometry import Point as _ShapelyPoint
    from shapely.geometry import Polygon as _ShapelyPolygon
    from shapely.geometry import box as _shapely_box
    from shapely.ops import unary_union as _shapely_unary_union
except Exception:
    _shapely_concave_hull = None
    _ShapelyMultiPoint = None
    _ShapelyPoint = None
    _ShapelyPolygon = None
    _shapely_box = None
    _shapely_unary_union = None


def _emit_status(status_callback, message: str, pct: int | None = None, step_pct: float | None = None, step_label: str | None = None) -> None:
    if status_callback is None:
        return
    try:
        status_callback(message, pct, step_pct, step_label)
    except TypeError:
        status_callback(message, pct)


def _to_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        v = float(value)
        return v if np.isfinite(v) else None
    if isinstance(value, str):
        text = value.strip().replace(",", ".")
        if not text:
            return None
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        if not match:
            return None
        try:
            v = float(match.group(0))
        except ValueError:
            return None
        return v if np.isfinite(v) else None
    return None


def _polygon_signed_area(poly_xy: np.ndarray) -> float:
    x = poly_xy[:, 0]
    y = poly_xy[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _normalize_polygon(poly_xy: np.ndarray, *, simplify_tolerance: float = 0.0, max_vertices: int = 500) -> np.ndarray | None:
    arr = np.asarray(poly_xy, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 2:
        return None
    arr = arr[:, :2]
    arr = arr[np.isfinite(arr).all(axis=1)]
    if arr.shape[0] < 3:
        return None

    if np.linalg.norm(arr[0] - arr[-1]) <= 1e-9:
        arr = arr[:-1]
    if arr.shape[0] < 3:
        return None

    keep = np.ones(arr.shape[0], dtype=bool)
    if arr.shape[0] > 1:
        d = np.linalg.norm(np.diff(arr, axis=0), axis=1)
        keep[1:] = d > 1e-9
        arr = arr[keep]
    if arr.shape[0] < 3:
        return None

    if simplify_tolerance > 0.0 and arr.shape[0] >= 4:
        n = arr.shape[0]
        keep = np.ones(n, dtype=bool)
        for i in range(n):
            p0 = arr[(i - 1) % n]
            p1 = arr[i]
            p2 = arr[(i + 1) % n]
            edge = p2 - p0
            edge_len = float(np.linalg.norm(edge))
            if edge_len <= 1e-10:
                keep[i] = False
                continue
            rel = p1 - p0
            dist = abs(edge[0] * rel[1] - edge[1] * rel[0]) / edge_len
            if dist < simplify_tolerance:
                keep[i] = False
        arr2 = arr[keep]
        if arr2.shape[0] >= 3:
            arr = arr2

    max_vertices = max(int(max_vertices), 3)
    if arr.shape[0] > max_vertices:
        idx = np.linspace(0, arr.shape[0], num=max_vertices, endpoint=False)
        idx = np.unique(np.floor(idx).astype(np.int64))
        if idx.size >= 3:
            arr = arr[idx]

    if arr.shape[0] < 3 or abs(_polygon_signed_area(arr)) <= 1e-10:
        return None
    return arr.astype(np.float64, copy=False)


def _dedupe_xy_keep_highest(points_xyz: np.ndarray, *, cell_size: float) -> np.ndarray:
    arr = np.asarray(points_xyz, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 3:
        return np.empty((0, 3), dtype=np.float64)
    step = max(float(cell_size), 1e-3)
    quant_xy = np.round(arr[:, :2] / step).astype(np.int64)
    order = np.argsort(arr[:, 2], kind="mergesort")[::-1]
    quant_sorted = quant_xy[order]
    _, keep_idx = np.unique(quant_sorted, axis=0, return_index=True)
    keep = order[np.sort(keep_idx)]
    out = arr[keep]
    return out.astype(np.float64, copy=False)


def _densify_ring(poly_xy: np.ndarray, *, spacing: float) -> np.ndarray:
    boundary_xy, _ = _densify_ring_with_edge_ids(poly_xy, spacing=spacing)
    return boundary_xy


def _densify_ring_with_edge_ids(poly_xy: np.ndarray, *, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    ring = np.asarray(poly_xy, dtype=np.float64)
    if ring.ndim != 2 or ring.shape[0] < 3 or ring.shape[1] < 2:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.int32)
    ring = ring[:, :2]
    samples: list[np.ndarray] = []
    edge_ids: list[int] = []
    step = max(float(spacing), 0.1)
    count = ring.shape[0]
    for idx in range(count):
        p0 = ring[idx]
        p1 = ring[(idx + 1) % count]
        seg = p1 - p0
        seg_len = float(np.linalg.norm(seg))
        divisions = max(1, int(np.ceil(seg_len / step)))
        for div in range(divisions):
            t = div / divisions
            samples.append(p0 + seg * t)
            edge_ids.append(idx)
    if not samples:
        return ring, np.arange(ring.shape[0], dtype=np.int32)
    out = np.asarray(samples, dtype=np.float64)
    return out[:, :2], np.asarray(edge_ids, dtype=np.int32)


def _smooth_boundary_z_by_edges(
    boundary_xy: np.ndarray,
    boundary_z: np.ndarray,
    edge_ids: np.ndarray | None,
) -> np.ndarray:
    xy = np.asarray(boundary_xy, dtype=np.float64)
    z = np.asarray(boundary_z, dtype=np.float64)
    if xy.ndim != 2 or z.ndim != 1 or xy.shape[0] != z.shape[0] or xy.shape[0] < 3:
        return z
    if edge_ids is None:
        return z
    ids = np.asarray(edge_ids, dtype=np.int32)
    if ids.shape[0] != z.shape[0]:
        return z

    out = z.copy()
    for edge_id in np.unique(ids):
        edge_mask = ids == edge_id
        edge_idx = np.flatnonzero(edge_mask)
        if edge_idx.size <= 1:
            continue
        edge_xy = xy[edge_idx]
        edge_z = z[edge_idx]
        base = edge_xy[0]
        if edge_xy.shape[0] >= 2:
            direction = edge_xy[-1] - edge_xy[0]
        else:
            direction = np.array([1.0, 0.0], dtype=np.float64)
        dir_norm = float(np.linalg.norm(direction))
        if dir_norm <= 1e-10:
            out[edge_idx] = float(np.median(edge_z))
            continue
        direction = direction / dir_norm
        t = ((edge_xy - base[None, :]) @ direction).astype(np.float64, copy=False)
        if np.allclose(t, t[0]):
            out[edge_idx] = float(np.median(edge_z))
            continue
        try:
            coeff = np.polyfit(t, edge_z, deg=1)
            fitted = np.polyval(coeff, t)
            out[edge_idx] = fitted.astype(np.float64, copy=False)
        except Exception:
            out[edge_idx] = float(np.median(edge_z))
    return out


def _point_on_segment_2d(point_xy: np.ndarray, a_xy: np.ndarray, b_xy: np.ndarray, tol: float = 1e-8) -> bool:
    seg = b_xy - a_xy
    rel = point_xy - a_xy
    seg_len = float(np.linalg.norm(seg))
    if seg_len <= tol:
        return float(np.linalg.norm(point_xy - a_xy)) <= tol
    cross = abs(float(seg[0] * rel[1] - seg[1] * rel[0]))
    if cross > tol * max(seg_len, 1.0):
        return False
    dot = float(np.dot(rel, seg))
    return -tol <= dot <= float(np.dot(seg, seg)) + tol


def _point_in_polygon_inclusive(point_xy: np.ndarray, ring_xy: np.ndarray, tol: float = 1e-8) -> bool:
    x = float(point_xy[0])
    y = float(point_xy[1])
    ring = np.asarray(ring_xy, dtype=np.float64)
    inside = False
    count = ring.shape[0]
    for idx in range(count):
        a = ring[idx]
        b = ring[(idx + 1) % count]
        if _point_on_segment_2d(np.asarray([x, y], dtype=np.float64), a, b, tol=tol):
            return True
        yi = float(a[1])
        yj = float(b[1])
        xi = float(a[0])
        xj = float(b[0])
        intersects = ((yi > y) != (yj > y))
        if not intersects:
            continue
        denom = (yj - yi) if abs(yj - yi) > 1e-12 else (1e-12 if yj >= yi else -1e-12)
        x_cross = xi + (y - yi) * (xj - xi) / denom
        if x <= x_cross + tol:
            inside = not inside
    return inside


def _point_to_segment_distance_2d(point_xy: np.ndarray, a_xy: np.ndarray, b_xy: np.ndarray) -> float:
    point = np.asarray(point_xy, dtype=np.float64)
    a = np.asarray(a_xy, dtype=np.float64)
    b = np.asarray(b_xy, dtype=np.float64)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return float(np.linalg.norm(point - a))
    t = float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
    proj = a + t * ab
    return float(np.linalg.norm(point - proj))


def _distance_to_polygon_edges(point_xy: np.ndarray, ring_xy: np.ndarray) -> float:
    ring = np.asarray(ring_xy, dtype=np.float64)
    if ring.ndim != 2 or ring.shape[0] < 2:
        return float("inf")
    best = float("inf")
    for idx in range(ring.shape[0]):
        a = ring[idx]
        b = ring[(idx + 1) % ring.shape[0]]
        best = min(best, _point_to_segment_distance_2d(point_xy, a, b))
    return best


def _ensure_ring_covers_points(
    ring_xy: np.ndarray,
    points_xy: np.ndarray,
    *,
    tol: float = 0.06,
    max_buffer: float = 4.0,
) -> np.ndarray | None:
    """Ensure the footprint ring encloses all XY points.

    If some points fall outside, expand the polygon just enough (bounded by
    ``max_buffer``). If geometric ops are unavailable, fallback to convex hull.
    """
    ring = _normalize_polygon(np.asarray(ring_xy, dtype=np.float64), simplify_tolerance=0.0, max_vertices=320)
    pts = np.asarray(points_xy, dtype=np.float64)
    if ring is None or pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 2:
        return ring
    pts = pts[:, :2]

    inside_mask = np.array(
        [_point_in_polygon_inclusive(p, ring, tol=max(float(tol), 1e-3)) for p in pts],
        dtype=bool,
    )
    if bool(np.all(inside_mask)):
        return ring

    if _ShapelyPolygon is not None and _ShapelyPoint is not None:
        try:
            poly = _ShapelyPolygon(ring)
            if not poly.is_empty and float(poly.area) > 1e-10:
                outside = pts[~inside_mask]
                if outside.shape[0] > 0:
                    dmax = 0.0
                    for p in outside:
                        d = float(poly.exterior.distance(_ShapelyPoint(float(p[0]), float(p[1]))))
                        if np.isfinite(d):
                            dmax = max(dmax, d)
                    grow = float(np.clip(dmax + max(float(tol) * 1.5, 0.03), 0.02, max(float(max_buffer), 0.05)))
                    grown = poly.buffer(grow, join_style=2)
                    if grown is not None and (not grown.is_empty):
                        if grown.geom_type == "MultiPolygon":
                            grown = max(grown.geoms, key=lambda g: float(g.area))
                        if grown.geom_type == "Polygon" and float(grown.area) > 1e-10:
                            ring_grown = np.asarray(grown.exterior.coords, dtype=np.float64)
                            ring_grown = _normalize_polygon(
                                ring_grown,
                                simplify_tolerance=max(0.0, float(tol) * 0.3),
                                max_vertices=320,
                            )
                            if ring_grown is not None:
                                check_mask = np.array(
                                    [_point_in_polygon_inclusive(p, ring_grown, tol=max(float(tol), 1e-3)) for p in pts],
                                    dtype=bool,
                                )
                                if bool(np.all(check_mask)):
                                    return ring_grown
        except Exception:
            pass

    # Fallback: convex hull always encloses all points.
    try:
        hull = ConvexHull(pts)
        hull_ring = pts[hull.vertices]
        return _normalize_polygon(hull_ring, simplify_tolerance=max(0.0, float(tol) * 0.25), max_vertices=320)
    except Exception:
        return ring


def _estimate_nominal_spacing_xy(points_xy: np.ndarray) -> float:
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 2:
        return 1.0
    try:
        tree = cKDTree(pts[:, :2])
        dists, _ = tree.query(pts[:, :2], k=min(4, pts.shape[0]))
    except Exception:
        return 1.0
    if np.ndim(dists) == 1:
        return 1.0
    positive = np.asarray(dists[:, 1:], dtype=np.float64)
    positive = positive[positive > 1e-9]
    if positive.size == 0:
        return 1.0
    return float(np.clip(np.median(positive), 0.15, 6.0))


def _extract_dense_core_xy(
    cluster_xy: np.ndarray,
    *,
    radius_xy: float,
    density_hint: float,
) -> np.ndarray:
    arr = np.asarray(cluster_xy, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 2:
        return np.empty((0, 2), dtype=np.float64)
    arr = arr[:, :2]

    nominal_spacing = float(
        np.sqrt(1.0 / max(float(density_hint), 0.15))
    )
    inferred_spacing = _estimate_nominal_spacing_xy(arr)
    nominal_spacing = float(
        np.clip(min(nominal_spacing, inferred_spacing * 1.35), 0.15, 6.0)
    )

    quant_step = max(nominal_spacing * 0.35, float(radius_xy) * 0.14, 0.08)
    quant = np.round(arr / quant_step).astype(np.int64)
    _, unique_idx = np.unique(quant, axis=0, return_index=True)
    unique_idx_sorted = np.sort(unique_idx)
    unique_xy = arr[unique_idx_sorted]
    unique_quant = quant[unique_idx_sorted]
    if unique_xy.shape[0] < 6:
        return unique_xy.astype(np.float64, copy=False)

    # IGN-like sparse class-6 clouds (1-4 pts/m²) lose too much footprint if we prune
    # "thin" points here. Keep full XY support and clean later at polygon stage.
    if float(density_hint) <= 2.2:
        return unique_xy.astype(np.float64, copy=False)

    # Grid-neighborhood pruning: remove 1-cell thick tendrils before geometric hull.
    cell_set = {tuple(cell.tolist()) for cell in unique_quant}
    grid_keep = np.zeros(unique_quant.shape[0], dtype=bool)
    min_grid_neighbors = 3 if unique_quant.shape[0] >= 80 else 2
    for i, cell in enumerate(unique_quant):
        cx = int(cell[0])
        cy = int(cell[1])
        nb = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                if (cx + dx, cy + dy) in cell_set:
                    nb += 1
        if nb >= min_grid_neighbors:
            grid_keep[i] = True
    if int(np.count_nonzero(grid_keep)) >= 6:
        unique_xy = unique_xy[grid_keep]
        unique_quant = unique_quant[grid_keep]
        if unique_xy.shape[0] < 6:
            return unique_xy.astype(np.float64, copy=False)

    core_radius = max(float(radius_xy) * 0.58, nominal_spacing * 1.12, 0.55)
    core_radius = min(core_radius, max(float(radius_xy) * 0.98, nominal_spacing * 2.2))

    try:
        tree = cKDTree(unique_xy)
        neighbors = tree.query_ball_point(unique_xy, r=core_radius)
    except Exception:
        return unique_xy.astype(np.float64, copy=False)

    neighbor_counts = np.fromiter(
        (max(len(ids) - 1, 0) for ids in neighbors),
        dtype=np.int32,
        count=len(neighbors),
    )
    min_neighbors = 3 if unique_xy.shape[0] >= 60 else 2
    keep = neighbor_counts >= min_neighbors
    if int(np.count_nonzero(keep)) < max(8, int(unique_xy.shape[0] * 0.28)):
        keep = neighbor_counts >= max(min_neighbors - 1, 1)
    if int(np.count_nonzero(keep)) < 6:
        return unique_xy.astype(np.float64, copy=False)

    # Reject thin linear tendrils: branches may have enough neighbors but near-collinear support.
    if unique_xy.shape[0] >= 24:
        linearity_ratio = np.ones(unique_xy.shape[0], dtype=np.float64)
        for idx, ids in enumerate(neighbors):
            if len(ids) < 4:
                continue
            nb = unique_xy[np.asarray(ids, dtype=np.int64)]
            centered = nb - np.mean(nb, axis=0, keepdims=True)
            cov = centered.T @ centered / max(nb.shape[0] - 1, 1)
            try:
                evals = np.linalg.eigvalsh(cov)
            except np.linalg.LinAlgError:
                continue
            evals = np.sort(np.maximum(evals, 0.0))[::-1]
            if evals[0] <= 1e-10:
                continue
            linearity_ratio[idx] = float(evals[1] / evals[0])
        shape_keep = (linearity_ratio >= 0.065) | (neighbor_counts >= 6)
        if int(np.count_nonzero(keep & shape_keep)) >= 6:
            keep = keep & shape_keep

    keep_idx = np.flatnonzero(keep)
    visited = np.zeros(keep_idx.shape[0], dtype=bool)
    comps: list[np.ndarray] = []
    for seed in range(keep_idx.shape[0]):
        if visited[seed]:
            continue
        queue = [seed]
        comp: list[int] = []
        while queue:
            local_i = queue.pop()
            if visited[local_i]:
                continue
            visited[local_i] = True
            global_i = int(keep_idx[local_i])
            comp.append(global_i)
            for nb in neighbors[global_i]:
                nb_local = np.searchsorted(keep_idx, int(nb))
                if (
                    nb_local < keep_idx.shape[0]
                    and int(keep_idx[nb_local]) == int(nb)
                    and not visited[nb_local]
                ):
                    queue.append(int(nb_local))
        if len(comp) >= 4:
            comps.append(np.asarray(comp, dtype=np.int64))

    if not comps:
        core_xy = unique_xy[keep]
    else:
        best_comp = max(comps, key=lambda c: int(c.shape[0]))
        core_xy = unique_xy[best_comp]

    if core_xy.shape[0] < 6:
        return unique_xy.astype(np.float64, copy=False)

    centroid = np.mean(core_xy, axis=0)
    d = np.linalg.norm(core_xy - centroid[None, :], axis=1)
    dist_limit = float(np.percentile(d, 99.0))
    if np.isfinite(dist_limit) and dist_limit > 0:
        mask_dist = d <= dist_limit
        if int(np.count_nonzero(mask_dist)) >= 6:
            core_xy = core_xy[mask_dist]

    return core_xy.astype(np.float64, copy=False)


def _estimate_building_xy_cell_size(
    points_xy: np.ndarray,
    *,
    radius_xy: float,
    density_hint: float,
) -> float:
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 2:
        return max(min(float(radius_xy) * 0.35, 0.8), 0.35)
    inferred = _estimate_nominal_spacing_xy(pts[:, :2])
    nominal = float(np.sqrt(1.0 / max(float(density_hint), 0.15)))
    step = min(inferred * 0.90, nominal * 0.95, max(float(radius_xy) * 0.45, 0.85))
    return float(np.clip(step, 0.30, 1.10))


def _extract_xy_occupancy_components(
    points_xyz: np.ndarray,
    *,
    radius_xy: float,
    density_hint: float,
    min_points: int,
) -> list[dict[str, np.ndarray | set[tuple[int, int]] | float]]:
    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < max(int(min_points), 3) or pts.shape[1] < 3:
        return []

    cell_size = _estimate_building_xy_cell_size(
        pts[:, :2],
        radius_xy=radius_xy,
        density_hint=density_hint,
    )
    quant = np.round(pts[:, :2] / cell_size).astype(np.int64)
    cell_to_indices: dict[tuple[int, int], list[int]] = {}
    for idx, cell in enumerate(quant):
        key = (int(cell[0]), int(cell[1]))
        cell_to_indices.setdefault(key, []).append(idx)
    occupied = set(cell_to_indices.keys())
    if not occupied:
        return []

    # Fill tiny 1-cell holes between occupied neighbors.
    xs = [c[0] for c in occupied]
    ys = [c[1] for c in occupied]
    if xs and ys:
        for gx in range(min(xs) - 1, max(xs) + 2):
            for gy in range(min(ys) - 1, max(ys) + 2):
                key = (gx, gy)
                if key in occupied:
                    continue
                neighbors = 0
                axial = 0
                for dx, dy in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                    if (gx + dx, gy + dy) in occupied:
                        neighbors += 1
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    if (gx + dx, gy + dy) in occupied:
                        axial += 1
                if neighbors >= 5 or axial >= 3:
                    occupied.add(key)

    # Remove isolated/tendril cells but preserve compact masses.
    filtered: set[tuple[int, int]] = set()
    for gx, gy in occupied:
        pt_count = len(cell_to_indices.get((gx, gy), ()))
        neighbors = 0
        for dx, dy in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
            if (gx + dx, gy + dy) in occupied:
                neighbors += 1
        if pt_count >= 2 or neighbors >= 2:
            filtered.add((gx, gy))
    if filtered:
        occupied = filtered

    visited: set[tuple[int, int]] = set()
    components: list[dict[str, np.ndarray | set[tuple[int, int]] | float]] = []
    for seed in sorted(occupied):
        if seed in visited:
            continue
        queue = [seed]
        comp_cells: set[tuple[int, int]] = set()
        while queue:
            cell = queue.pop()
            if cell in visited or cell not in occupied:
                continue
            visited.add(cell)
            comp_cells.add(cell)
            gx, gy = cell
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nb = (gx + dx, gy + dy)
                    if nb in occupied and nb not in visited:
                        queue.append(nb)

        if not comp_cells:
            continue

        comp_idx: list[int] = []
        for cell in comp_cells:
            comp_idx.extend(cell_to_indices.get(cell, ()))
        if len(comp_idx) < max(int(min_points), 3):
            continue

        comp_idx_arr = np.unique(np.asarray(comp_idx, dtype=np.int64))
        comp_points = pts[comp_idx_arr]
        if comp_points.shape[0] < max(int(min_points), 3):
            continue

        footprint_ring: np.ndarray | None = None
        if _shapely_box is not None and _shapely_unary_union is not None and _ShapelyPolygon is not None:
            try:
                half = cell_size * 0.5
                cells_geom = [
                    _shapely_box(
                        cell[0] * cell_size - half,
                        cell[1] * cell_size - half,
                        cell[0] * cell_size + half,
                        cell[1] * cell_size + half,
                    )
                    for cell in comp_cells
                ]
                geom = _shapely_unary_union(cells_geom)
                if geom.geom_type == "MultiPolygon":
                    geom = max(geom.geoms, key=lambda g: float(g.area))
                geom = geom.buffer(cell_size * 0.18, join_style=2).buffer(-cell_size * 0.12, join_style=2)
                if geom.geom_type == "MultiPolygon":
                    geom = max(geom.geoms, key=lambda g: float(g.area))
                if geom.geom_type == "Polygon" and float(geom.area) > 1e-8:
                    footprint_ring = _normalize_polygon(
                        np.asarray(geom.exterior.coords, dtype=np.float64),
                        simplify_tolerance=max(cell_size * 0.18, 0.05),
                        max_vertices=220,
                    )
            except Exception:
                footprint_ring = None

        if footprint_ring is None:
            footprint_ring = _estimate_cluster_footprint(
                comp_points[:, :2],
                radius_xy=radius_xy,
                orthogonalize_edges=False,
                density_hint=density_hint,
            )
        if footprint_ring is None or footprint_ring.shape[0] < 3:
            continue

        components.append(
            {
                "points": comp_points.astype(np.float64, copy=False),
                "footprint": footprint_ring.astype(np.float64, copy=False),
                "cells": comp_cells,
                "cell_size": float(cell_size),
            }
        )

    return components


def _estimate_cluster_footprint(
    cluster_xy: np.ndarray,
    *,
    radius_xy: float,
    orthogonalize_edges: bool = False,
    density_hint: float = 1.0,
) -> np.ndarray | None:
    arr = np.asarray(cluster_xy, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 2:
        return None
    sparse_lidar = float(density_hint) <= 2.2
    if sparse_lidar:
        arr = arr[:, :2]
        quant_step = max(float(radius_xy) * 0.18, 0.08)
        quant = np.round(arr / quant_step).astype(np.int64)
        _, unique_idx = np.unique(quant, axis=0, return_index=True)
        unique_xy = arr[np.sort(unique_idx)]
    else:
        core_xy = _extract_dense_core_xy(
            arr[:, :2],
            radius_xy=radius_xy,
            density_hint=density_hint,
        )
        if core_xy.shape[0] >= 6:
            unique_xy = core_xy
        else:
            arr = arr[:, :2]
            quant_step = max(float(radius_xy) * 0.20, 0.10)
            quant = np.round(arr / quant_step).astype(np.int64)
            _, unique_idx = np.unique(quant, axis=0, return_index=True)
            unique_xy = arr[np.sort(unique_idx)]
    if unique_xy.shape[0] < 3:
        return None

    ring = None
    if _ShapelyMultiPoint is not None and _shapely_concave_hull is not None:
        try:
            geom = _ShapelyMultiPoint([tuple(pt) for pt in unique_xy])
            nominal_spacing = float(np.sqrt(1.0 / max(float(density_hint), 0.15)))
            if sparse_lidar:
                ratio = 0.88
            else:
                ratio = float(
                    np.clip(
                        0.34 + np.log10(max(unique_xy.shape[0], 10)) * 0.10,
                        0.30,
                        0.72,
                    )
                )
            poly = _shapely_concave_hull(geom, ratio=ratio, allow_holes=False)
            if poly is None or poly.is_empty:
                poly = geom.convex_hull
            if poly.geom_type == "MultiPolygon":
                poly = max(poly.geoms, key=lambda g: float(g.area))
            if poly.geom_type != "Polygon" or float(poly.area) <= 1e-8:
                poly = geom.convex_hull
            expand = max(float(radius_xy) * 0.16, 0.12)
            poly = poly.buffer(expand, join_style=2).buffer(-expand * 0.30, join_style=2)
            if not sparse_lidar:
                # Remove narrow spurs/bridges for dense clouds; avoid this on sparse clouds.
                open_radius = max(float(radius_xy) * 0.24, nominal_spacing * 0.55, 0.30)
                opened = poly.buffer(-open_radius, join_style=2).buffer(open_radius, join_style=2)
                if (
                    opened is not None
                    and not opened.is_empty
                    and float(opened.area) > 1e-8
                    and float(opened.area) >= float(poly.area) * 0.42
                ):
                    poly = opened
            if sparse_lidar:
                convex = poly.convex_hull
                convex_area = float(convex.area) if convex is not None else 0.0
                if convex_area > 1e-8 and float(poly.area) < convex_area * 0.58:
                    poly = convex
            if poly.geom_type == "MultiPolygon":
                poly = max(poly.geoms, key=lambda g: float(g.area))
            if poly.geom_type == "Polygon" and float(poly.area) > 1e-8:
                ring = np.asarray(poly.exterior.coords, dtype=np.float64)
        except Exception:
            ring = None

    if ring is None:
        try:
            hull = ConvexHull(unique_xy)
            ring = unique_xy[hull.vertices]
        except QhullError:
            return None

    simplify_tol = max(float(radius_xy) * 0.08, 0.05)
    ring_norm = _normalize_polygon(ring, simplify_tolerance=simplify_tol, max_vertices=240)
    if ring_norm is None:
        return None

    if _ShapelyPolygon is not None and ring_norm.shape[0] >= 4:
        try:
            poly = _ShapelyPolygon(ring_norm)
            if not poly.is_empty and float(poly.area) > 1e-8:
                perimeter = max(float(poly.length), 1e-8)
                compactness = float((4.0 * np.pi * float(poly.area)) / (perimeter * perimeter))
                rect = poly.minimum_rotated_rectangle
                rect_area = float(rect.area) if rect is not None else 0.0
                rectangularity = float(poly.area) / rect_area if rect_area > 1e-8 else 0.0
                if compactness < 0.020 and rectangularity < 0.48:
                    spur_radius = max(float(radius_xy) * 0.44, 0.55)
                    robust = poly.buffer(-spur_radius, join_style=2).buffer(spur_radius, join_style=2)
                    if (
                        robust is not None
                        and not robust.is_empty
                        and float(robust.area) > 1e-8
                        and float(robust.area) >= float(poly.area) * 0.35
                    ):
                        poly = robust
                    hull = poly.convex_hull
                    if hull is not None and (not hull.is_empty) and float(hull.area) > 1e-8:
                        hull_ring = np.asarray(hull.exterior.coords, dtype=np.float64)
                        hull_norm = _normalize_polygon(
                            hull_ring,
                            simplify_tolerance=max(simplify_tol * 0.85, 0.05),
                            max_vertices=120,
                        )
                        if hull_norm is not None:
                            ring_norm = hull_norm
        except Exception:
            pass

    if orthogonalize_edges:
        ring_ortho = _orthogonalize_footprint_ring(
            ring_norm,
            density_hint=density_hint,
            radius_xy=radius_xy,
        )
        if ring_ortho is not None:
            return ring_ortho
    return ring_norm


def _estimate_convex_footprint(
    cluster_xy: np.ndarray,
    *,
    orthogonalize_edges: bool = False,
) -> np.ndarray | None:
    arr = np.asarray(cluster_xy, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 2:
        return None
    pts = arr[:, :2]
    try:
        hull = ConvexHull(pts)
        ring = pts[hull.vertices]
    except QhullError:
        return None
    ring_norm = _normalize_polygon(
        ring,
        simplify_tolerance=0.04,
        max_vertices=240,
    )
    if ring_norm is None:
        return None
    if orthogonalize_edges:
        centroid = np.mean(ring_norm, axis=0)
        radius_xy = float(np.max(np.linalg.norm(ring_norm - centroid[None, :], axis=1)))
        ring_ortho = _orthogonalize_footprint_ring(
            ring_norm,
            density_hint=1.0,
            radius_xy=max(radius_xy, 0.8),
        )
        if ring_ortho is not None:
            return ring_ortho
    return ring_norm


def _orthogonalize_footprint_ring(
    ring_xy: np.ndarray, *, density_hint: float, radius_xy: float
) -> np.ndarray | None:
    ring = np.asarray(ring_xy, dtype=np.float64)
    if _ShapelyPolygon is None or ring.ndim != 2 or ring.shape[0] < 3:
        return _normalize_polygon(ring, simplify_tolerance=max(radius_xy * 0.08, 0.05), max_vertices=240)
    try:
        poly = _ShapelyPolygon(ring)
        if poly.is_empty or float(poly.area) <= 1e-8:
            return _normalize_polygon(ring, simplify_tolerance=max(radius_xy * 0.08, 0.05), max_vertices=240)
        rect = poly.minimum_rotated_rectangle
        rect_area = float(rect.area)
        if rect_area <= 1e-8:
            return _normalize_polygon(ring, simplify_tolerance=max(radius_xy * 0.08, 0.05), max_vertices=240)
        rectangularity = float(poly.area) / rect_area
        if rectangularity < 0.74:
            return _normalize_polygon(ring, simplify_tolerance=max(radius_xy * 0.08, 0.05), max_vertices=240)
        rect_coords = np.asarray(rect.exterior.coords, dtype=np.float64)
        simplify_tol = max(0.10, 0.35 / np.sqrt(max(float(density_hint), 0.2)), radius_xy * 0.05)
        return _normalize_polygon(rect_coords, simplify_tolerance=simplify_tol, max_vertices=16)
    except Exception:
        return _normalize_polygon(ring, simplify_tolerance=max(radius_xy * 0.08, 0.05), max_vertices=240)


def _fit_plane_lstsq(points_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 3:
        return None, None
    A = np.column_stack([pts[:, 0], pts[:, 1], np.ones(pts.shape[0], dtype=np.float64)])
    try:
        coeff, _, _, _ = np.linalg.lstsq(A, pts[:, 2], rcond=None)
    except np.linalg.LinAlgError:
        return None, None
    a, b, _ = coeff
    normal = np.asarray([-a, -b, 1.0], dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        return None, None
    normal /= norm
    if normal[2] < 0:
        normal *= -1.0
    return coeff.astype(np.float64, copy=False), normal


def _predict_plane_height(coeff: np.ndarray, xy_local: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy_local, dtype=np.float64)
    return coeff[0] * xy[:, 0] + coeff[1] * xy[:, 1] + coeff[2]


def _extract_dominant_roof_planes(
    roof_points: np.ndarray,
    *,
    residual_tol: float,
    max_planes: int,
    min_points_per_plane: int,
    max_iterations: int,
) -> list[np.ndarray]:
    pts = np.asarray(roof_points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < max(3, min_points_per_plane):
        return []
    rng = np.random.default_rng(42)
    remaining = np.arange(pts.shape[0], dtype=np.int64)
    planes: list[np.ndarray] = []
    tol = max(float(residual_tol), 0.08)
    n_iter = max(int(max_iterations), 20)
    max_planes = max(int(max_planes), 1)
    min_plane = max(int(min_points_per_plane), 3)

    while remaining.size >= min_plane and len(planes) < max_planes:
        sample = pts[remaining]
        best_inliers_mask = None
        best_coeff = None
        best_count = 0
        if sample.shape[0] < 3:
            break
        for _ in range(n_iter):
            try:
                ids = rng.choice(sample.shape[0], size=3, replace=False)
            except ValueError:
                break
            coeff, normal = _fit_plane_lstsq(sample[ids])
            if coeff is None or normal is None:
                continue
            if float(normal[2]) <= 0.18:
                continue
            pred = _predict_plane_height(coeff, sample[:, :2])
            residual = np.abs(pred - sample[:, 2])
            inliers_mask = residual <= tol
            inlier_count = int(inliers_mask.sum())
            if inlier_count >= min_plane and inlier_count > best_count:
                best_inliers_mask = inliers_mask
                best_coeff = coeff
                best_count = inlier_count
        if best_inliers_mask is None or best_coeff is None:
            break
        inlier_points = sample[best_inliers_mask]
        refined_coeff, refined_normal = _fit_plane_lstsq(inlier_points)
        if refined_coeff is None or refined_normal is None or float(refined_normal[2]) <= 0.18:
            break
        planes.append(refined_coeff)
        remaining = remaining[~best_inliers_mask]

    return planes


def _planarity_metrics(points_xyz: np.ndarray) -> tuple[float, float]:
    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 4 or pts.shape[1] < 3:
        return 0.0, 1.0
    centered = pts[:, :3] - np.mean(pts[:, :3], axis=0, keepdims=True)
    cov = centered.T @ centered / max(pts.shape[0] - 1, 1)
    try:
        evals = np.linalg.eigvalsh(cov)
    except np.linalg.LinAlgError:
        return 0.0, 1.0
    evals = np.sort(np.maximum(evals, 0.0))[::-1]
    if evals[0] <= 1e-12:
        return 0.0, 1.0
    planarity = float((evals[1] - evals[2]) / evals[0])
    scattering = float(evals[2] / evals[0])
    return planarity, scattering


def _snap_roof_heights_to_planes(
    xy: np.ndarray,
    z: np.ndarray,
    planes: list[np.ndarray],
    *,
    max_adjust: float,
) -> np.ndarray:
    if not planes:
        return np.asarray(z, dtype=np.float64)
    xy_arr = np.asarray(xy, dtype=np.float64)
    z_arr = np.asarray(z, dtype=np.float64)
    out = z_arr.copy()
    plane_preds = np.column_stack([_predict_plane_height(coeff, xy_arr) for coeff in planes]).astype(np.float64, copy=False)
    diffs = np.abs(plane_preds - z_arr[:, None])
    best_idx = np.argmin(diffs, axis=1)
    best_diff = diffs[np.arange(z_arr.shape[0]), best_idx]
    best_pred = plane_preds[np.arange(z_arr.shape[0]), best_idx]
    mask = best_diff <= max(float(max_adjust), 0.12)
    out[mask] = best_pred[mask]
    return out


def _score_roof_candidate_fit(
    observed_z: np.ndarray,
    predicted_z: np.ndarray,
    *,
    complexity_penalty: float,
) -> float:
    obs = np.asarray(observed_z, dtype=np.float64)
    pred = np.asarray(predicted_z, dtype=np.float64)
    if obs.shape != pred.shape or obs.size == 0:
        return float("inf")
    resid = np.abs(obs - pred)
    med = float(np.median(resid))
    p85 = float(np.percentile(resid, 85.0))
    mean = float(np.mean(resid))
    return med + 0.45 * p85 + 0.15 * mean + float(complexity_penalty)


def _estimate_xy_hull_area(points_xy: np.ndarray) -> float:
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 2:
        return 0.0
    pts = pts[:, :2]
    try:
        hull = ConvexHull(pts)
        area = float(getattr(hull, "volume", 0.0))
        if np.isfinite(area):
            return max(area, 0.0)
    except Exception:
        pass
    ring = _normalize_polygon(pts, simplify_tolerance=0.0, max_vertices=500)
    if ring is None:
        return 0.0
    return max(abs(_polygon_signed_area(ring)), 0.0)


def _segment_roof_planes_for_cluster(
    roof_support: np.ndarray,
    planes: list[np.ndarray],
    *,
    residual_tol: float,
    min_plane_area_m2: float,
    min_points_per_plane: int,
) -> list[dict[str, np.ndarray | float]]:
    support = np.asarray(roof_support, dtype=np.float64)
    if support.ndim != 2 or support.shape[0] < 8 or support.shape[1] < 3 or not planes:
        return []

    base_tol = max(float(residual_tol), 0.10)
    grow_tol = max(base_tol * 1.35, 0.16)
    min_points = max(int(min_points_per_plane), 6)
    min_area = max(float(min_plane_area_m2), 0.0)

    grown_planes: list[np.ndarray] = []
    for coeff_in in planes:
        coeff = np.asarray(coeff_in, dtype=np.float64)
        if coeff.shape[0] < 3:
            continue
        for _ in range(2):
            pred = _predict_plane_height(coeff, support[:, :2])
            mask = np.abs(pred - support[:, 2]) <= grow_tol
            if int(mask.sum()) < min_points:
                break
            coeff_refit, normal = _fit_plane_lstsq(support[mask])
            if coeff_refit is None or normal is None or float(normal[2]) <= 0.20:
                break
            coeff = coeff_refit
        pred_final = _predict_plane_height(coeff, support[:, :2])
        final_mask = np.abs(pred_final - support[:, 2]) <= base_tol
        if int(final_mask.sum()) < min_points:
            continue
        area = _estimate_xy_hull_area(support[final_mask, :2])
        if area < min_area:
            continue
        grown_planes.append(coeff.astype(np.float64, copy=False))

    if len(grown_planes) < 2:
        return []

    pred_mat = np.column_stack(
        [_predict_plane_height(coeff, support[:, :2]) for coeff in grown_planes]
    ).astype(np.float64, copy=False)
    resid_mat = np.abs(pred_mat - support[:, 2:3])
    best_idx = np.argmin(resid_mat, axis=1)
    best_res = resid_mat[np.arange(support.shape[0]), best_idx]
    assign = np.where(best_res <= grow_tol, best_idx, -1)

    segmented: list[dict[str, np.ndarray | float]] = []
    for plane_idx, coeff in enumerate(grown_planes):
        ids = np.flatnonzero(assign == plane_idx)
        if ids.size < min_points:
            continue
        area = _estimate_xy_hull_area(support[ids, :2])
        if area < min_area:
            continue
        coeff_refit, normal = _fit_plane_lstsq(support[ids])
        if coeff_refit is None or normal is None or float(normal[2]) <= 0.20:
            coeff_refit = coeff
        segmented.append(
            {
                "coeff": np.asarray(coeff_refit, dtype=np.float64),
                "indices": ids.astype(np.int64, copy=False),
                "area": float(area),
            }
        )

    if len(segmented) < 2:
        return []
    return segmented


def _resolve_roof_model_mode(
    roof_support: np.ndarray,
    *,
    roof_percentile: float,
    plane_tolerance: float,
    max_plane_iterations: int,
    density_hint: float,
    snap_roof_planes: bool,
    requested_mode: str,
) -> tuple[str, np.ndarray | None, list[np.ndarray]]:
    support = np.asarray(roof_support, dtype=np.float64)
    if support.ndim != 2 or support.shape[0] < 3 or support.shape[1] < 3:
        return "freeform", None, []

    density_hint = max(float(density_hint), 0.15)
    nominal_spacing = float(np.sqrt(1.0 / density_hint))
    mode = str(requested_mode or "auto").strip().lower()
    if mode in {"planar_intersection", "plane_intersection", "segmented_planes", "planes_intersection"}:
        mode = "dominant_planes"

    single_plane = None
    coeff, normal = _fit_plane_lstsq(support)
    if coeff is not None and normal is not None and float(normal[2]) > 0.45:
        single_plane = coeff

    planes: list[np.ndarray] = []
    if snap_roof_planes or mode in {"auto", "dominant_planes"}:
        planes = _extract_dominant_roof_planes(
            support,
            residual_tol=max(float(plane_tolerance) * 1.15, nominal_spacing * 0.20, 0.10),
            max_planes=3,
            min_points_per_plane=max(5, int(5 + density_hint * 1.5)),
            max_iterations=max(int(max_plane_iterations), 40),
        )

    if mode in {"flat", "single_plane", "dominant_planes", "freeform"}:
        if mode == "single_plane" and single_plane is None:
            return "freeform", single_plane, planes
        if mode == "dominant_planes" and len(planes) < 2:
            return ("single_plane" if single_plane is not None else "freeform"), single_plane, planes
        return mode, single_plane, planes

    # Auto-bias for sparse LiDAR (typical IGN): prefer interpretable planes over noisy freeform.
    if mode == "auto":
        if len(planes) >= 2 and density_hint <= 2.5:
            dom_pred = _snap_roof_heights_to_planes(
                support[:, :2],
                support[:, 2],
                planes,
                max_adjust=max(float(plane_tolerance) * 2.2, nominal_spacing * 0.60, 0.22),
            )
            dom_resid = np.abs(dom_pred - support[:, 2])
            dom_med = float(np.median(dom_resid))
            dom_p85 = float(np.percentile(dom_resid, 85.0))
            if dom_med <= max(0.28, float(plane_tolerance) * 1.6, nominal_spacing * 0.22) and dom_p85 <= max(
                0.45, float(plane_tolerance) * 2.3, nominal_spacing * 0.36
            ):
                return "dominant_planes", single_plane, planes
        if single_plane is not None and density_hint <= 2.5:
            single_resid = np.abs(_predict_plane_height(single_plane, support[:, :2]) - support[:, 2])
            if float(np.percentile(single_resid, 85.0)) <= max(0.55, float(plane_tolerance) * 2.8, nominal_spacing * 0.45):
                return "single_plane", single_plane, planes

    roof_z = support[:, 2].astype(np.float64, copy=False)
    candidates: list[dict[str, object]] = []
    flat_z = np.full(roof_z.shape[0], float(np.median(roof_z)), dtype=np.float64)
    candidates.append(
        {
            "mode": "flat",
            "rank": 0,
            "score": _score_roof_candidate_fit(roof_z, flat_z, complexity_penalty=0.00),
        }
    )
    if single_plane is not None:
        single_pred = _predict_plane_height(single_plane, support[:, :2])
        candidates.append(
            {
                "mode": "single_plane",
                "rank": 1,
                "score": _score_roof_candidate_fit(roof_z, single_pred, complexity_penalty=0.04),
            }
        )
    if len(planes) >= 2:
        dom_pred = _snap_roof_heights_to_planes(
            support[:, :2],
            roof_z,
            planes,
            max_adjust=max(float(plane_tolerance) * 2.1, nominal_spacing * 0.55, 0.18),
        )
        plane_penalty = 0.05 + max(len(planes) - 2, 0) * 0.03
        candidates.append(
            {
                "mode": "dominant_planes",
                "rank": 2,
                "score": _score_roof_candidate_fit(roof_z, dom_pred, complexity_penalty=plane_penalty),
            }
        )
    freeform_penalty = max(
        0.35,
        nominal_spacing * 0.26,
        float(plane_tolerance) * 1.05,
        (0.20 if density_hint <= 2.5 else 0.05),
    )
    candidates.append(
        {
            "mode": "freeform",
            "rank": 3,
            "score": _score_roof_candidate_fit(roof_z, roof_z, complexity_penalty=freeform_penalty),
        }
    )

    best_score = min(float(candidate["score"]) for candidate in candidates)
    simplicity_margin = max(0.20, float(plane_tolerance) * 1.05, nominal_spacing * 0.28)
    acceptable = [
        candidate
        for candidate in candidates
        if float(candidate["score"]) <= best_score + simplicity_margin
    ]
    chosen = min(acceptable, key=lambda candidate: (int(candidate["rank"]), float(candidate["score"])))
    return str(chosen["mode"]), single_plane, planes


def _build_closed_footprint_roof_solid(
    ring: np.ndarray,
    roof_support: np.ndarray,
    roof_keep_count: int,
    boundary_xy: np.ndarray,
    boundary_edge_ids: np.ndarray | None,
    top_support_xy: np.ndarray,
    top_z: np.ndarray,
    *,
    ground_tree: cKDTree | None,
    ground_points: np.ndarray | None,
    nominal_spacing: float,
    min_height: float,
    max_height: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if top_support_xy.shape[0] != top_z.shape[0] or top_support_xy.shape[0] < 3:
        return None, None

    boundary_offset = int(roof_keep_count)
    boundary_indices = np.arange(boundary_offset, top_support_xy.shape[0], dtype=np.int32)
    if boundary_indices.size != boundary_xy.shape[0]:
        return None, None

    try:
        roof_tri = Delaunay(top_support_xy)
        candidate_faces = roof_tri.simplices.astype(np.int32, copy=False)
    except QhullError:
        candidate_faces = np.empty((0, 3), dtype=np.int32)

    if candidate_faces.shape[0] == 0:
        tris = _triangulate_polygon_ear_clipping(boundary_xy)
        candidate_faces = np.asarray(tris, dtype=np.int32) if tris else np.empty((0, 3), dtype=np.int32)
    if candidate_faces.shape[0] == 0:
        return None, None

    local_tree = cKDTree(top_support_xy)
    if top_support_xy.shape[0] >= 2:
        dists, _ = local_tree.query(top_support_xy, k=min(5, top_support_xy.shape[0]))
        positive = dists[:, 1:][dists[:, 1:] > 1e-9] if dists.ndim > 1 else np.asarray([], dtype=np.float64)
        median_nn = float(np.median(positive)) if positive.size > 0 else max(nominal_spacing, 0.3)
    else:
        median_nn = max(nominal_spacing, 0.3)
    max_edge_length = max(median_nn * 4.4, nominal_spacing * 2.8, 1.0)
    top_faces = _filter_triangles_to_footprint(
        top_support_xy,
        candidate_faces,
        ring,
        max_edge_length=max_edge_length,
    )
    if top_faces.shape[0] == 0:
        return None, None

    if ground_tree is not None and ground_points is not None and ground_points.shape[0] >= 3:
        _, ground_idx = ground_tree.query(boundary_xy, k=1)
        base_boundary_z = ground_points[np.asarray(ground_idx, dtype=np.int64), 2].astype(np.float64, copy=False)
    else:
        roof_min = float(np.min(roof_support[:, 2]))
        roof_span = float(np.ptp(roof_support[:, 2]))
        base_value = roof_min - max(0.45, min(roof_span * 0.35, 4.0))
        base_boundary_z = np.full(boundary_xy.shape[0], base_value, dtype=np.float64)

    roof_z = np.asarray(top_z, dtype=np.float64).copy()
    boundary_roof_z = roof_z[boundary_indices].astype(np.float64, copy=False)
    boundary_roof_z = _smooth_boundary_z_by_edges(boundary_xy, boundary_roof_z, boundary_edge_ids)
    min_height = max(float(min_height), 0.25)
    max_height = max(float(max_height), min_height)
    boundary_roof_z = np.clip(boundary_roof_z, base_boundary_z + min_height, base_boundary_z + max_height)
    roof_z[boundary_indices] = boundary_roof_z
    base_boundary_z = np.minimum(base_boundary_z, boundary_roof_z - 0.05)
    if np.any(~np.isfinite(base_boundary_z)) or np.any(~np.isfinite(roof_z)):
        return None, None

    top_vertices = np.column_stack([top_support_xy, roof_z]).astype(np.float64, copy=False)
    # Check whether top triangulation actually uses the alero ring edges.
    # If not, forcing walls on ring edges leaves seams (open shells).
    edge_counts: dict[tuple[int, int], int] = {}
    for tri in top_faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for e0, e1 in ((a, b), (b, c), (c, a)):
            key = (min(e0, e1), max(e0, e1))
            edge_counts[key] = edge_counts.get(key, 0) + 1

    ring_edges: list[tuple[int, int]] = []
    for i in range(boundary_indices.size):
        j = (i + 1) % boundary_indices.size
        ai = int(boundary_indices[i])
        bj = int(boundary_indices[j])
        ring_edges.append((min(ai, bj), max(ai, bj)))
    ring_edge_count = max(len(ring_edges), 1)
    ring_edges_present = sum(1 for e in ring_edges if edge_counts.get(e, 0) > 0)
    ring_coverage = float(ring_edges_present) / float(ring_edge_count)

    # Fallback closure mode: extrude the actual boundary of top_faces.
    # Guarantees watertight roof-wall connection even if ring edges are not explicit.
    if ring_coverage < 0.98:
        if ground_tree is not None and ground_points is not None and ground_points.shape[0] >= 3:
            _, ground_idx_all = ground_tree.query(top_support_xy, k=1)
            base_z_all = ground_points[np.asarray(ground_idx_all, dtype=np.int64), 2].astype(np.float64, copy=False)
        else:
            base_fallback = float(np.min(base_boundary_z)) if base_boundary_z.size > 0 else float(np.min(roof_z) - 1.0)
            base_z_all = np.full(top_support_xy.shape[0], base_fallback, dtype=np.float64)

        base_z_all = np.minimum(base_z_all, roof_z - 0.05)
        if np.any(~np.isfinite(base_z_all)):
            return None, None

        base_vertices_all = np.column_stack([top_support_xy, base_z_all]).astype(np.float64, copy=False)
        base_offset_all = top_vertices.shape[0]
        bottom_faces_all = (top_faces[:, ::-1] + base_offset_all).astype(np.int32, copy=False)

        boundary_edges = [(a, b) for (a, b), count in edge_counts.items() if count == 1]
        if not boundary_edges:
            return None, None

        wall_faces_all: list[list[int]] = []
        center_xy = np.mean(top_support_xy, axis=0)
        verts_all = np.vstack([top_vertices, base_vertices_all]).astype(np.float64, copy=False)
        for a, b in boundary_edges:
            base_a = int(a + base_offset_all)
            base_b = int(b + base_offset_all)
            candidates = [[a, b, base_b], [a, base_b, base_a]]
            for face in candidates:
                va, vb, vc = verts_all[face]
                normal = np.cross(vb - va, vc - va)
                center = (va + vb + vc) / 3.0
                outward = np.array([center[0] - center_xy[0], center[1] - center_xy[1], 0.0], dtype=np.float64)
                if normal[0] * outward[0] + normal[1] * outward[1] < 0:
                    face = [face[0], face[2], face[1]]
                wall_faces_all.append(face)

        if not wall_faces_all:
            return None, None

        faces_all = np.vstack(
            [
                top_faces.astype(np.int32, copy=False),
                bottom_faces_all,
                np.asarray(wall_faces_all, dtype=np.int32),
            ]
        ).astype(np.int32, copy=False)
        return _clean_mesh(verts_all, faces_all)

    # Nominal mode: ring-driven walls on aleros.
    base_vertices = np.column_stack([boundary_xy, base_boundary_z]).astype(np.float64, copy=False)
    base_offset = top_vertices.shape[0]

    bottom_tris = _triangulate_polygon_ear_clipping(boundary_xy)
    if not bottom_tris:
        return None, None
    bottom_faces = np.asarray(
        [[base_offset + ia, base_offset + ic, base_offset + ib] for ia, ib, ic in bottom_tris],
        dtype=np.int32,
    )

    wall_faces: list[list[int]] = []
    center_xy = np.mean(boundary_xy, axis=0)
    boundary_count = boundary_xy.shape[0]
    verts = np.vstack([top_vertices, base_vertices]).astype(np.float64, copy=False)
    for i in range(boundary_count):
        j = (i + 1) % boundary_count
        top_a = int(boundary_indices[i])
        top_b = int(boundary_indices[j])
        base_a = int(base_offset + i)
        base_b = int(base_offset + j)
        candidates = [[top_a, top_b, base_b], [top_a, base_b, base_a]]
        for face in candidates:
            va, vb, vc = verts[face]
            normal = np.cross(vb - va, vc - va)
            center = (va + vb + vc) / 3.0
            outward = np.array([center[0] - center_xy[0], center[1] - center_xy[1], 0.0], dtype=np.float64)
            if normal[0] * outward[0] + normal[1] * outward[1] < 0:
                face = [face[0], face[2], face[1]]
            wall_faces.append(face)

    if not wall_faces:
        return None, None

    faces = np.vstack(
        [
            top_faces.astype(np.int32, copy=False),
            bottom_faces,
            np.asarray(wall_faces, dtype=np.int32),
        ]
    ).astype(np.int32, copy=False)
    return _clean_mesh(verts, faces)


def _split_touching_cluster(
    cluster_pts: np.ndarray,
    *,
    radius_xy: float,
    roof_percentile: float,
    min_cluster_points: int,
    density_hint: float,
) -> list[np.ndarray]:
    pts = np.asarray(cluster_pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < max(int(min_cluster_points) * 2, 12) or pts.shape[1] < 3:
        return [pts]

    nominal_spacing = float(np.sqrt(1.0 / max(float(density_hint), 0.15)))
    roof_cut = float(np.percentile(pts[:, 2], np.clip(max(55.0, roof_percentile - 20.0), 45.0, 85.0)))
    roof_core = pts[pts[:, 2] >= roof_cut]
    if roof_core.shape[0] < max(int(min_cluster_points), 8):
        return [pts]

    split_radius = min(
        max(float(radius_xy) * 0.65, nominal_spacing * 1.10, 0.85),
        max(float(radius_xy) * 0.92, 0.95),
    )
    roof_tree = cKDTree(roof_core[:, :2])
    visited = np.zeros(roof_core.shape[0], dtype=bool)
    roof_groups: list[np.ndarray] = []
    for seed_idx in range(roof_core.shape[0]):
        if visited[seed_idx]:
            continue
        queue = [seed_idx]
        component: list[int] = []
        while queue:
            idx = queue.pop()
            if visited[idx]:
                continue
            visited[idx] = True
            component.append(idx)
            for nb in roof_tree.query_ball_point(roof_core[idx, :2], r=split_radius):
                if not visited[nb]:
                    queue.append(nb)
        if len(component) >= max(4, int(min_cluster_points * 0.55)):
            roof_groups.append(roof_core[np.asarray(component, dtype=np.int64)])

    if len(roof_groups) <= 1:
        return [pts]

    centroids = np.asarray([np.mean(group[:, :2], axis=0) for group in roof_groups], dtype=np.float64)
    assign_tree = cKDTree(centroids)
    _, labels = assign_tree.query(pts[:, :2], k=1)
    labels = np.asarray(labels, dtype=np.int64)

    groups: list[np.ndarray] = []
    leftovers: list[np.ndarray] = []
    for group_idx in range(len(roof_groups)):
        group_pts = pts[labels == group_idx]
        if group_pts.shape[0] >= min_cluster_points:
            groups.append(group_pts)
        elif group_pts.shape[0] > 0:
            leftovers.append(group_pts)

    if not groups:
        return [pts]
    if leftovers:
        group_centroids = [np.mean(g[:, :2], axis=0) for g in groups]
        for extra in leftovers:
            extra_centroid = np.mean(extra[:, :2], axis=0)
            target_idx = int(np.argmin([np.linalg.norm(extra_centroid - gc) for gc in group_centroids]))
            groups[target_idx] = np.vstack([groups[target_idx], extra]).astype(np.float64, copy=False)

    if len(groups) <= 1:
        return [pts]
    return groups


def _interpolate_height_idw(query_xy: np.ndarray, support_xyz: np.ndarray, *, k: int = 8) -> np.ndarray:
    query = np.asarray(query_xy, dtype=np.float64)
    support = np.asarray(support_xyz, dtype=np.float64)
    if query.ndim != 2 or query.shape[0] == 0:
        return np.empty((0,), dtype=np.float64)
    if support.ndim != 2 or support.shape[0] == 0 or support.shape[1] < 3:
        return np.zeros((query.shape[0],), dtype=np.float64)

    n_neighbors = max(1, min(int(k), int(support.shape[0])))
    tree = cKDTree(support[:, :2])
    distances, indices = tree.query(query, k=n_neighbors)
    if n_neighbors == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    support_z = support[:, 2]
    out = np.empty((query.shape[0],), dtype=np.float64)
    for idx in range(query.shape[0]):
        d = np.asarray(distances[idx], dtype=np.float64)
        ids = np.asarray(indices[idx], dtype=np.int64)
        if np.any(d <= 1e-9):
            out[idx] = float(support_z[ids[np.argmin(d)]])
            continue
        weights = 1.0 / np.maximum(d, 1e-6) ** 2
        out[idx] = float(np.sum(weights * support_z[ids]) / np.sum(weights))
    return out


def _filter_triangles_to_footprint(
    vertices_xy: np.ndarray,
    simplices: np.ndarray,
    footprint_ring: np.ndarray,
    *,
    max_edge_length: float,
) -> np.ndarray:
    if simplices.size == 0:
        return np.empty((0, 3), dtype=np.int32)
    tris = np.asarray(simplices, dtype=np.int32)
    pts = np.asarray(vertices_xy, dtype=np.float64)
    ring = np.asarray(footprint_ring, dtype=np.float64)
    keep: list[np.ndarray] = []
    footprint_poly = _ShapelyPolygon(ring) if _ShapelyPolygon is not None else None
    max_edge = max(float(max_edge_length), 0.25)
    for tri in tris:
        tri_xy = pts[tri]
        edge_lengths = (
            float(np.linalg.norm(tri_xy[1] - tri_xy[0])),
            float(np.linalg.norm(tri_xy[2] - tri_xy[1])),
            float(np.linalg.norm(tri_xy[0] - tri_xy[2])),
        )
        if max(edge_lengths) > max_edge:
            continue
        if footprint_poly is not None:
            try:
                tri_poly = _ShapelyPolygon(tri_xy)
                if tri_poly.is_empty or float(tri_poly.area) <= 1e-10:
                    continue
                if not footprint_poly.covers(tri_poly):
                    continue
            except Exception:
                pass
        else:
            centroid = np.mean(tri_xy, axis=0)
            mids = np.asarray(
                [
                    centroid,
                    (tri_xy[0] + tri_xy[1]) * 0.5,
                    (tri_xy[1] + tri_xy[2]) * 0.5,
                    (tri_xy[2] + tri_xy[0]) * 0.5,
                ],
                dtype=np.float64,
            )
            if not all(_point_in_polygon_inclusive(p, ring) for p in mids):
                continue
        keep.append(tri)
    if not keep:
        return np.empty((0, 3), dtype=np.int32)
    return np.asarray(keep, dtype=np.int32)


def _build_cluster_roof_solid(
    cluster_pts: np.ndarray,
    *,
    radius_xy: float,
    roof_percentile: float,
    base_percentile: float,
    ground_tree: cKDTree | None,
    ground_points: np.ndarray | None,
    plane_tolerance: float,
    max_plane_iterations: int,
    density_hint: float,
    orthogonalize_edges: bool,
    snap_roof_planes: bool,
    roof_model_mode: str = "dominant_planes",
    min_plane_area_m2: float = 10.0,
    footprint_override: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    cluster_pts = np.asarray(cluster_pts, dtype=np.float64)
    if cluster_pts.ndim != 2 or cluster_pts.shape[0] < 3 or cluster_pts.shape[1] < 3:
        return None, None

    density_hint = max(float(density_hint), 0.15)
    nominal_spacing = float(np.sqrt(1.0 / density_hint))

    roof_seed_pct = float(np.clip(max(56.0, float(roof_percentile) - 26.0), 55.0, 90.0))
    roof_seed_cut = float(np.percentile(cluster_pts[:, 2], roof_seed_pct))
    roof_seed = cluster_pts[cluster_pts[:, 2] >= roof_seed_cut]
    if roof_seed.shape[0] < max(12, int(cluster_pts.shape[0] * 0.08)):
        roof_seed = cluster_pts
    seed_support = _dedupe_xy_keep_highest(
        roof_seed[:, :3],
        cell_size=max(nominal_spacing * 0.55, 0.18),
    )
    # For sparse LiDAR, footprint from all class-6 points is usually more stable than only roof seeds.
    footprint_xy = cluster_pts[:, :2]
    if seed_support.shape[0] >= 12:
        _, _, seed_planes = _resolve_roof_model_mode(
            seed_support,
            roof_percentile=roof_percentile,
            plane_tolerance=plane_tolerance,
            max_plane_iterations=max_plane_iterations,
            density_hint=density_hint,
            snap_roof_planes=snap_roof_planes,
            requested_mode="dominant_planes",
        )
        segmented_seed_planes = _segment_roof_planes_for_cluster(
            seed_support,
            seed_planes,
            residual_tol=max(float(plane_tolerance) * 1.35, nominal_spacing * 0.28, 0.14),
            min_plane_area_m2=max(float(min_plane_area_m2), 1.0),
            min_points_per_plane=max(8, int(8 + density_hint * 2.0)),
        )
        if segmented_seed_planes:
            seg_xy = np.vstack(
                [seed_support[np.asarray(seg["indices"], dtype=np.int64), :2] for seg in segmented_seed_planes]
            ).astype(np.float64, copy=False)
            if seg_xy.shape[0] >= 6:
                seg_area = _estimate_xy_hull_area(seg_xy)
                cluster_area = _estimate_xy_hull_area(cluster_pts[:, :2])
                if cluster_area <= 1e-6 or seg_area >= cluster_area * 0.75:
                    footprint_xy = seg_xy

    if footprint_override is not None:
        footprint_ring = _normalize_polygon(
            np.asarray(footprint_override, dtype=np.float64),
            simplify_tolerance=max(float(radius_xy) * 0.06, 0.04),
            max_vertices=240,
        )
    else:
        footprint_ring = _estimate_cluster_footprint(
            footprint_xy,
            radius_xy=radius_xy,
            orthogonalize_edges=orthogonalize_edges,
            density_hint=density_hint,
        )
    if footprint_ring is None or footprint_ring.shape[0] < 3:
        return None, None
    cluster_hull_area = _estimate_xy_hull_area(cluster_pts[:, :2])
    footprint_area = abs(_polygon_signed_area(footprint_ring))
    if cluster_hull_area > 1.0 and (
        footprint_area < cluster_hull_area * 0.70 or footprint_area > cluster_hull_area * 1.75
    ):
        fallback_ring = _estimate_convex_footprint(
            cluster_pts[:, :2],
            orthogonalize_edges=orthogonalize_edges,
        )
        if fallback_ring is not None:
            footprint_ring = fallback_ring

    if footprint_override is None:
        guaranteed_ring = _ensure_ring_covers_points(
            footprint_ring,
            cluster_pts[:, :2],
            tol=max(0.04, float(radius_xy) * 0.06),
            max_buffer=max(2.5, float(radius_xy) * 2.5),
        )
        if guaranteed_ring is not None and guaranteed_ring.shape[0] >= 3:
            footprint_ring = guaranteed_ring

    # Preferred path: parametric roof reconstruction from a stable footprint.
    # This is more robust for sparse IGN class-6 clouds than freeform triangulation.
    try:
        param_vertices, param_faces = _build_footprint_roof_solid(
            footprint_ring,
            cluster_pts[:, :3],
            roof_percentile=roof_percentile,
            ground_tree=ground_tree,
            ground_points=ground_points,
            plane_tolerance=plane_tolerance,
            max_plane_iterations=max_plane_iterations,
            density_hint=density_hint,
            snap_roof_planes=snap_roof_planes,
            roof_model_mode=roof_model_mode,
            min_height=2.0,
            max_height=120.0,
            min_plane_area_m2=min_plane_area_m2,
        )
        if (
            param_vertices is not None
            and param_faces is not None
            and param_vertices.shape[0] >= 3
            and param_faces.shape[0] >= 1
        ):
            return param_vertices, param_faces
    except Exception:
        # Fallback below to legacy branch.
        pass

    support_cell = max(float(radius_xy) * 0.24, nominal_spacing * 0.72, 0.20)
    roof_support = _dedupe_xy_keep_highest(cluster_pts[:, :3], cell_size=support_cell)
    if roof_support.shape[0] < 3:
        return None, None

    cutoff_pct = float(np.clip(min(float(roof_percentile), 80.0) - 20.0, 20.0, 70.0))
    z_cut = float(np.percentile(roof_support[:, 2], cutoff_pct))
    roof_support = roof_support[roof_support[:, 2] >= z_cut]
    if roof_support.shape[0] < 6:
        roof_support = _dedupe_xy_keep_highest(cluster_pts[:, :3], cell_size=support_cell)
    if roof_support.shape[0] < 3:
        return None, None

    inside_support = np.array(
        [_point_in_polygon_inclusive(pt[:2], footprint_ring) for pt in roof_support],
        dtype=bool,
    )
    if int(inside_support.sum()) >= 3:
        roof_support = roof_support[inside_support]
    if roof_support.shape[0] < 3:
        return None, None

    planes: list[np.ndarray] = []
    if snap_roof_planes and roof_support.shape[0] >= 8:
        planes = _extract_dominant_roof_planes(
            roof_support,
            residual_tol=max(float(plane_tolerance) * 1.2, nominal_spacing * 0.22, 0.12),
            max_planes=3,
            min_points_per_plane=max(6, int(6 + density_hint * 2.0)),
            max_iterations=max(int(max_plane_iterations), 30),
        )

    boundary_spacing = max(float(radius_xy) * 0.55, nominal_spacing * 0.95, 0.35)
    boundary_xy = _densify_ring(footprint_ring, spacing=boundary_spacing)
    if boundary_xy.shape[0] < 3:
        boundary_xy = footprint_ring
    boundary_z = _interpolate_height_idw(boundary_xy, roof_support, k=min(8, roof_support.shape[0]))
    boundary_support = np.column_stack([boundary_xy, boundary_z]).astype(np.float64, copy=False)

    top_support = np.vstack([roof_support, boundary_support]).astype(np.float64, copy=False)
    top_support = _dedupe_xy_keep_highest(top_support, cell_size=max(support_cell * 0.55, 0.08))
    if top_support.shape[0] < 3:
        return None, None
    if planes:
        top_support[:, 2] = _snap_roof_heights_to_planes(
            top_support[:, :2],
            top_support[:, 2],
            planes,
            max_adjust=max(float(plane_tolerance) * 2.2, nominal_spacing * 0.55, 0.22),
        )

    try:
        roof_tri = Delaunay(top_support[:, :2])
        candidate_faces = roof_tri.simplices.astype(np.int32, copy=False)
    except QhullError:
        candidate_faces = np.empty((0, 3), dtype=np.int32)

    if candidate_faces.shape[0] == 0:
        if boundary_support.shape[0] < 3:
            return None, None
        top_support = boundary_support.astype(np.float64, copy=False)
        tris = _triangulate_polygon_ear_clipping(boundary_xy)
        candidate_faces = np.asarray(tris, dtype=np.int32) if tris else np.empty((0, 3), dtype=np.int32)
    if candidate_faces.shape[0] == 0:
        return None, None

    local_tree = cKDTree(top_support[:, :2])
    if top_support.shape[0] >= 2:
        dists, _ = local_tree.query(top_support[:, :2], k=min(5, top_support.shape[0]))
        if dists.ndim == 1:
            positive = np.asarray([], dtype=np.float64)
        else:
            positive = dists[:, 1:][dists[:, 1:] > 1e-9]
        median_nn = float(np.median(positive)) if positive.size > 0 else max(radius_xy * 0.5, 0.3)
    else:
        median_nn = max(radius_xy * 0.5, 0.3)
    max_edge_length = max(float(radius_xy) * 2.2, median_nn * 4.4, nominal_spacing * 2.8, 1.0)
    top_faces = _filter_triangles_to_footprint(
        top_support[:, :2],
        candidate_faces,
        footprint_ring,
        max_edge_length=max_edge_length,
    )
    if top_faces.shape[0] == 0:
        return None, None

    if ground_tree is not None and ground_points is not None and ground_points.shape[0] >= 3:
        _, ground_idx = ground_tree.query(top_support[:, :2], k=1)
        base_z = ground_points[np.asarray(ground_idx, dtype=np.int64), 2].astype(np.float64, copy=False)
    else:
        cluster_min = float(np.min(cluster_pts[:, 2]))
        cluster_span = float(np.ptp(cluster_pts[:, 2]))
        base_value = min(
            float(np.percentile(cluster_pts[:, 2], np.clip(base_percentile, 0.0, 100.0))),
            cluster_min - max(0.45, min(cluster_span * 0.30, 3.0)),
        )
        base_z = np.full(top_support.shape[0], base_value, dtype=np.float64)

    roof_z = top_support[:, 2].astype(np.float64, copy=False)
    base_z = np.minimum(base_z, roof_z - 0.05)
    if np.any(~np.isfinite(base_z)):
        return None, None

    top_vertices = np.column_stack([top_support[:, :2], roof_z])
    base_vertices = np.column_stack([top_support[:, :2], base_z])
    vertex_count = top_vertices.shape[0]

    edge_counts: dict[tuple[int, int], int] = {}
    for tri in top_faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for e0, e1 in ((a, b), (b, c), (c, a)):
            key = (min(e0, e1), max(e0, e1))
            edge_counts[key] = edge_counts.get(key, 0) + 1

    wall_faces: list[list[int]] = []
    for (a, b), count in edge_counts.items():
        if count != 1:
            continue
        wall_faces.append([a, b, b + vertex_count])
        wall_faces.append([a, b + vertex_count, a + vertex_count])

    if not wall_faces:
        return None, None

    bottom_faces = (top_faces[:, ::-1] + vertex_count).astype(np.int32, copy=False)
    faces = np.vstack(
        [
            top_faces.astype(np.int32, copy=False),
            bottom_faces,
            np.asarray(wall_faces, dtype=np.int32),
        ]
    ).astype(np.int32, copy=False)
    verts = np.vstack([top_vertices, base_vertices]).astype(np.float64, copy=False)
    return _clean_mesh(verts, faces)


def _select_roof_points_for_footprint(
    footprint_ring: np.ndarray,
    roof_points: np.ndarray | None,
    roof_tree: cKDTree | None,
    *,
    density_hint: float,
    footprint_shape_mode: str = "catastro",
) -> np.ndarray:
    ring = np.asarray(footprint_ring, dtype=np.float64)
    roof_xyz = np.asarray(roof_points, dtype=np.float64) if roof_points is not None else None
    if (
        roof_xyz is None
        or roof_tree is None
        or roof_xyz.ndim != 2
        or roof_xyz.shape[0] < 3
        or roof_xyz.shape[1] < 3
        or ring.ndim != 2
        or ring.shape[0] < 3
    ):
        return np.empty((0, 3), dtype=np.float64)

    density_hint = max(float(density_hint), 0.15)
    nominal_spacing = float(np.sqrt(1.0 / density_hint))
    centroid = np.mean(ring[:, :2], axis=0)
    radius = float(np.max(np.linalg.norm(ring[:, :2] - centroid[None, :], axis=1)))
    mode = str(footprint_shape_mode or "catastro").strip().lower()
    adaptive_mode = mode in {"adaptive_mix", "mix", "hybrid", "laz", "roof_only"}
    margin = max(0.30, nominal_spacing * 0.60)
    expansion = max(1.40, nominal_spacing * 2.25) if adaptive_mode else 0.0

    candidate_idx = roof_tree.query_ball_point(centroid, r=radius + margin + expansion)
    if not candidate_idx:
        return np.empty((0, 3), dtype=np.float64)

    candidate = roof_xyz[np.asarray(candidate_idx, dtype=np.int64)]
    bbox_min = np.min(ring[:, :2], axis=0) - (margin + expansion)
    bbox_max = np.max(ring[:, :2], axis=0) + (margin + expansion)
    bbox_mask = (
        (candidate[:, 0] >= bbox_min[0])
        & (candidate[:, 0] <= bbox_max[0])
        & (candidate[:, 1] >= bbox_min[1])
        & (candidate[:, 1] <= bbox_max[1])
    )
    candidate = candidate[bbox_mask]
    if candidate.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)

    inside_mask = np.array(
        [_point_in_polygon_inclusive(pt[:2], ring, tol=margin * 0.35) for pt in candidate],
        dtype=bool,
    )
    selected_mask = inside_mask.copy()
    if adaptive_mode:
        near_mask = np.fromiter(
            (_distance_to_polygon_edges(pt[:2], ring) <= expansion for pt in candidate),
            dtype=bool,
            count=candidate.shape[0],
        )
        selected_mask |= near_mask
    selected = candidate[selected_mask]

    if selected.shape[0] < 3 and _ShapelyPolygon is not None and _ShapelyPoint is not None:
        try:
            poly = _ShapelyPolygon(ring)
            if not poly.is_empty and float(poly.area) > 1e-8:
                buffered = poly.buffer(margin * 0.45, join_style=2)
                selected = candidate[
                    np.fromiter(
                        (
                            buffered.covers(_ShapelyPoint(float(pt[0]), float(pt[1])))
                            for pt in candidate
                        ),
                        dtype=bool,
                        count=candidate.shape[0],
                    )
                ]
        except Exception:
            pass

    return selected.astype(np.float64, copy=False)


def _build_footprint_roof_solid(
    footprint_ring: np.ndarray,
    roof_points: np.ndarray,
    *,
    roof_percentile: float,
    ground_tree: cKDTree | None,
    ground_points: np.ndarray | None,
    plane_tolerance: float,
    max_plane_iterations: int,
    density_hint: float,
    snap_roof_planes: bool,
    roof_model_mode: str = "auto",
    min_height: float,
    max_height: float,
    min_plane_area_m2: float = 10.0,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    ring = _normalize_polygon(footprint_ring, simplify_tolerance=0.0, max_vertices=240)
    roof_xyz = np.asarray(roof_points, dtype=np.float64)
    if (
        ring is None
        or roof_xyz.ndim != 2
        or roof_xyz.shape[0] < 3
        or roof_xyz.shape[1] < 3
    ):
        return None, None

    density_hint = max(float(density_hint), 0.15)
    nominal_spacing = float(np.sqrt(1.0 / density_hint))
    support_cell = max(nominal_spacing * 0.72, 0.20)

    roof_support = _dedupe_xy_keep_highest(roof_xyz[:, :3], cell_size=support_cell)
    if roof_support.shape[0] < 3:
        return None, None

    cutoff_pct = float(np.clip(min(float(roof_percentile), 80.0) - 20.0, 20.0, 70.0))
    z_cut = float(np.percentile(roof_support[:, 2], cutoff_pct))
    roof_support = roof_support[roof_support[:, 2] >= z_cut]
    if roof_support.shape[0] < 6:
        roof_support = _dedupe_xy_keep_highest(roof_xyz[:, :3], cell_size=support_cell)
    if roof_support.shape[0] < 3:
        return None, None

    inside_support = np.array(
        [_point_in_polygon_inclusive(pt[:2], ring, tol=max(0.08, nominal_spacing * 0.22)) for pt in roof_support],
        dtype=bool,
    )
    if int(inside_support.sum()) >= 3:
        roof_support = roof_support[inside_support]
    if roof_support.shape[0] < 3:
        return None, None

    boundary_spacing = max(nominal_spacing * 0.95, 0.35)
    boundary_xy, boundary_edge_ids = _densify_ring_with_edge_ids(ring, spacing=boundary_spacing)
    if boundary_xy.shape[0] < 3:
        boundary_xy = ring
        boundary_edge_ids = np.arange(boundary_xy.shape[0], dtype=np.int32)
    boundary_z = _interpolate_height_idw(boundary_xy, roof_support, k=min(8, roof_support.shape[0]))
    boundary_support = np.column_stack([boundary_xy, boundary_z]).astype(np.float64, copy=False)

    roof_keep = roof_support
    if roof_support.shape[0] >= 3 and boundary_xy.shape[0] >= 3:
        try:
            boundary_tree = cKDTree(boundary_xy)
            dist_to_boundary, _ = boundary_tree.query(roof_support[:, :2], k=1)
            roof_keep = roof_support[dist_to_boundary > max(boundary_spacing * 0.18, 0.08)]
            if roof_keep.shape[0] < 3:
                roof_keep = roof_support
        except Exception:
            roof_keep = roof_support

    top_support = np.vstack([roof_keep, boundary_support]).astype(np.float64, copy=False)
    if top_support.shape[0] < 3:
        return None, None

    resolved_mode, single_plane, planes = _resolve_roof_model_mode(
        roof_support,
        roof_percentile=roof_percentile,
        plane_tolerance=plane_tolerance,
        max_plane_iterations=max_plane_iterations,
        density_hint=density_hint,
        snap_roof_planes=snap_roof_planes,
        requested_mode=roof_model_mode,
    )
    top_xy = top_support[:, :2]
    top_guess = top_support[:, 2].astype(np.float64, copy=False)
    if resolved_mode == "flat":
        top_z = np.full(top_guess.shape[0], float(np.median(roof_support[:, 2])), dtype=np.float64)
    elif resolved_mode == "single_plane" and single_plane is not None:
        top_z = _predict_plane_height(single_plane, top_xy)
    elif resolved_mode == "dominant_planes" and len(planes) >= 2:
        keep_count = int(roof_keep.shape[0])
        top_z = top_guess.copy()
        segmented_planes = _segment_roof_planes_for_cluster(
            roof_support,
            planes,
            residual_tol=max(float(plane_tolerance) * 1.35, nominal_spacing * 0.28, 0.14),
            min_plane_area_m2=max(float(min_plane_area_m2), 1.0),
            min_points_per_plane=max(8, int(8 + density_hint * 2.0)),
        )
        coeffs = (
            [np.asarray(item.get("coeff"), dtype=np.float64) for item in segmented_planes]
            if len(segmented_planes) >= 2
            else [np.asarray(coeff, dtype=np.float64) for coeff in planes]
        )
        coeffs = [c for c in coeffs if c.ndim == 1 and c.shape[0] >= 3]
        if len(coeffs) >= 2:
            keep_xy = top_xy[:keep_count]
            keep_guess = top_guess[:keep_count]
            pred_keep = np.column_stack(
                [_predict_plane_height(coeff, keep_xy) for coeff in coeffs]
            ).astype(np.float64, copy=False)
            dif_keep = np.abs(pred_keep - keep_guess[:, None])
            best_keep = np.argmin(dif_keep, axis=1)
            best_keep_res = dif_keep[np.arange(keep_xy.shape[0]), best_keep]
            assign_tol = max(float(plane_tolerance) * 2.4, nominal_spacing * 0.70, 0.30)
            use_plane = best_keep_res <= assign_tol
            keep_out = keep_guess.copy()
            keep_out[use_plane] = pred_keep[np.arange(keep_xy.shape[0]), best_keep][use_plane]
            top_z[:keep_count] = keep_out

            boundary_xy_local = top_xy[keep_count:]
            if boundary_xy_local.shape[0] > 0:
                boundary_guess = top_guess[keep_count:]
                pred_boundary = np.column_stack(
                    [_predict_plane_height(coeff, boundary_xy_local) for coeff in coeffs]
                ).astype(np.float64, copy=False)
                dif_b = np.abs(pred_boundary - boundary_guess[:, None])
                top_z[keep_count:] = pred_boundary[
                    np.arange(boundary_xy_local.shape[0]), np.argmin(dif_b, axis=1)
                ]
                top_z[keep_count:] = _smooth_boundary_z_by_edges(
                    boundary_xy_local,
                    top_z[keep_count:],
                    boundary_edge_ids,
                )
        else:
            top_z[:keep_count] = _snap_roof_heights_to_planes(
                top_xy[:keep_count],
                top_guess[:keep_count],
                planes,
                max_adjust=max(float(plane_tolerance) * 2.2, nominal_spacing * 0.55, 0.22),
            )
            boundary_xy_local = top_xy[keep_count:]
            if boundary_xy_local.shape[0] > 0:
                plane_preds = np.column_stack(
                    [_predict_plane_height(coeff, boundary_xy_local) for coeff in planes]
                ).astype(np.float64, copy=False)
                diffs = np.abs(plane_preds - top_guess[keep_count:, None])
                top_z[keep_count:] = plane_preds[
                    np.arange(boundary_xy_local.shape[0]), np.argmin(diffs, axis=1)
                ]
    else:
        top_z = top_guess.copy()

    return _build_closed_footprint_roof_solid(
        ring,
        roof_support,
        int(roof_keep.shape[0]),
        boundary_xy,
        boundary_edge_ids,
        top_xy,
        top_z,
        ground_tree=ground_tree,
        ground_points=ground_points,
        nominal_spacing=nominal_spacing,
        min_height=min_height,
        max_height=max_height,
    )


def _adapt_footprint_with_roof_points(
    footprint_ring: np.ndarray,
    roof_points: np.ndarray,
    *,
    density_hint: float,
    outside_area_threshold: float,
    shape_mode: str,
) -> np.ndarray:
    ring = _normalize_polygon(footprint_ring, simplify_tolerance=0.0, max_vertices=240)
    roof_xyz = np.asarray(roof_points, dtype=np.float64)
    mode = str(shape_mode or "catastro").strip().lower()
    if ring is None or roof_xyz.ndim != 2 or roof_xyz.shape[0] < 6 or roof_xyz.shape[1] < 3:
        return footprint_ring
    if mode in {"catastro", "fixed", "none"}:
        return ring

    density_hint = max(float(density_hint), 0.15)
    nominal_spacing = float(np.sqrt(1.0 / density_hint))
    inferred = _estimate_cluster_footprint(
        roof_xyz[:, :2],
        radius_xy=max(nominal_spacing * 2.1, 1.10),
        orthogonalize_edges=False,
        density_hint=density_hint,
    )
    if inferred is None:
        return ring
    if mode in {"laz", "roof_only", "roof"}:
        return inferred

    if _ShapelyPolygon is not None:
        try:
            cat_poly = _ShapelyPolygon(ring)
            laz_poly = _ShapelyPolygon(inferred)
            if not cat_poly.is_empty and not laz_poly.is_empty and float(cat_poly.area) > 1e-8 and float(laz_poly.area) > 1e-8:
                outside_area = float(laz_poly.difference(cat_poly).area)
                if outside_area <= max(float(outside_area_threshold), 0.0):
                    return ring
                smooth = max(0.22, nominal_spacing * 0.50)
                mixed = cat_poly.union(laz_poly).buffer(smooth, join_style=2).buffer(-smooth * 0.90, join_style=2)
                if mixed.geom_type == "MultiPolygon":
                    mixed = max(mixed.geoms, key=lambda g: float(g.area))
                if mixed.geom_type == "Polygon" and float(mixed.area) > 1e-8:
                    mixed_ring = np.asarray(mixed.exterior.coords, dtype=np.float64)
                    mixed_norm = _normalize_polygon(
                        mixed_ring,
                        simplify_tolerance=max(0.06, nominal_spacing * 0.14),
                        max_vertices=240,
                    )
                    if mixed_norm is not None:
                        return mixed_norm
        except Exception:
            pass

    approx_outside = int(
        np.sum(
            [
                (not _point_in_polygon_inclusive(pt[:2], ring, tol=max(0.10, nominal_spacing * 0.25)))
                for pt in roof_xyz
            ]
        )
    ) / max(density_hint, 0.15)
    if approx_outside > max(float(outside_area_threshold), 0.0):
        return inferred
    return ring


def _extract_polygon_rings(geometry: dict | None) -> list[np.ndarray]:
    if not isinstance(geometry, dict):
        return []
    gtype = str(geometry.get("type", "")).strip().lower()
    coords = geometry.get("coordinates")
    if coords is None:
        return []
    if gtype == "polygon":
        polygons = [coords]
    elif gtype == "multipolygon":
        polygons = coords
    else:
        return []

    rings: list[np.ndarray] = []
    for poly in polygons:
        if not isinstance(poly, (list, tuple)) or len(poly) == 0:
            continue
        exterior = poly[0]
        try:
            ring = np.asarray(exterior, dtype=np.float64)
        except Exception:
            continue
        if ring.ndim != 2 or ring.shape[0] < 3 or ring.shape[1] < 2:
            continue
        rings.append(ring[:, :2])
    return rings


def load_building_footprints_geojson(
    data: bytes | str,
    *,
    min_area: float = 0.0,
    simplify_tolerance: float = 0.0,
    max_vertices: int = 500,
) -> tuple[list[dict], dict]:
    if isinstance(data, bytes):
        text = data.decode("utf-8-sig", errors="replace")
    elif isinstance(data, str):
        text = data
    else:
        raise TypeError("Formato de huellas no valido. Se esperaba texto JSON.")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"GeoJSON invalido: {exc}") from exc

    if isinstance(parsed, dict) and "features" in parsed:
        features = parsed.get("features") or []
    elif isinstance(parsed, list):
        features = parsed
    else:
        raise ValueError("No se encontraron features en el GeoJSON.")

    requested_min_area = max(float(min_area), 0.0)
    footprints: list[dict] = []
    numeric_fields: dict[str, int] = {}
    total_polygons = 0
    discarded = 0

    for feature in features:
        if not isinstance(feature, dict):
            discarded += 1
            continue

        if feature.get("type", "").lower() == "feature":
            geometry = feature.get("geometry")
            properties = feature.get("properties", {})
            if not isinstance(properties, dict):
                properties = {}
        else:
            geometry = feature
            properties = {}

        rings = _extract_polygon_rings(geometry)
        if not rings:
            continue

        for ring in rings:
            total_polygons += 1
            ring = _normalize_polygon(
                ring,
                simplify_tolerance=float(simplify_tolerance),
                max_vertices=int(max_vertices),
            )
            if ring is None:
                discarded += 1
                continue

            area = abs(_polygon_signed_area(ring))
            if area <= requested_min_area:
                discarded += 1
                continue

            footprints.append({"xy": ring, "properties": properties, "area": float(area)})

            for key, value in properties.items():
                if _to_float(value) is not None:
                    k = str(key)
                    numeric_fields[k] = numeric_fields.get(k, 0) + 1

    summary = {
        "features": len(features),
        "polygons": total_polygons,
        "accepted": len(footprints),
        "discarded": discarded,
        "numeric_fields": dict(sorted(numeric_fields.items(), key=lambda x: x[0].lower())),
    }
    return footprints, summary


def _point_in_triangle_2d(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    v0 = c - a
    v1 = b - a
    v2 = p - a

    dot00 = float(np.dot(v0, v0))
    dot01 = float(np.dot(v0, v1))
    dot02 = float(np.dot(v0, v2))
    dot11 = float(np.dot(v1, v1))
    dot12 = float(np.dot(v1, v2))
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-12:
        return False

    inv = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv
    eps = 1e-9
    return (u >= -eps) and (v >= -eps) and (u + v <= 1.0 + eps)


def _triangulate_polygon_ear_clipping(poly_xy: np.ndarray) -> list[tuple[int, int, int]]:
    n = poly_xy.shape[0]
    if n < 3:
        return []
    if n == 3:
        return [(0, 1, 2)]

    x = poly_xy[:, 0]
    y = poly_xy[:, 1]
    signed_area = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    is_ccw = signed_area > 0.0

    remaining = list(range(n))
    triangles: list[tuple[int, int, int]] = []
    guard = 0
    max_guard = n * n

    while len(remaining) > 2 and guard < max_guard:
        guard += 1
        ear_found = False
        m = len(remaining)
        for idx in range(m):
            i_prev = remaining[(idx - 1) % m]
            i_curr = remaining[idx]
            i_next = remaining[(idx + 1) % m]

            p_prev = poly_xy[i_prev]
            p_curr = poly_xy[i_curr]
            p_next = poly_xy[i_next]

            cross = float(
                (p_curr[0] - p_prev[0]) * (p_next[1] - p_curr[1])
                - (p_curr[1] - p_prev[1]) * (p_next[0] - p_curr[0])
            )
            if is_ccw:
                if cross <= 1e-12:
                    continue
            else:
                if cross >= -1e-12:
                    continue

            is_ear = True
            for j in remaining:
                if j in (i_prev, i_curr, i_next):
                    continue
                if _point_in_triangle_2d(poly_xy[j], p_prev, p_curr, p_next):
                    is_ear = False
                    break
            if not is_ear:
                continue

            if is_ccw:
                triangles.append((i_prev, i_curr, i_next))
            else:
                triangles.append((i_prev, i_next, i_curr))
            remaining.pop(idx)
            ear_found = True
            break

        if not ear_found:
            break

    if len(triangles) == 0:
        for i in range(1, n - 1):
            triangles.append((0, i, i + 1))
    return triangles

def _resolve_height(
    properties: dict,
    *,
    height_mode: str,
    fixed_height: float,
    fallback_height: float,
    height_attribute: str | None,
    floors_attribute: str | None,
    floor_height: float,
    min_height: float,
    max_height: float,
) -> float:
    mode = str(height_mode).strip().lower()
    # Manual override has priority over any automatic mode.
    manual_h = _to_float(properties.get("manual_height_m")) if isinstance(properties, dict) else None
    if manual_h is not None:
        return float(np.clip(manual_h, float(min_height), float(max_height)))

    if mode in {"fixed", "fija", "constante"}:
        h = _to_float(fixed_height)
    elif mode in {"laz_class_6", "laz_roof", "roof_from_laz", "cubierta_laz"}:
        h = _to_float(fallback_height)
    elif mode in {"attribute", "altura", "height"}:
        h = _to_float(properties.get(height_attribute)) if height_attribute else None
        if h is None:
            h = _to_float(fallback_height)
    elif mode in {"floors", "plantas"}:
        floors_value = _to_float(properties.get(floors_attribute)) if floors_attribute else None
        if floors_value is None:
            h = _to_float(fallback_height)
        else:
            h = floors_value * max(float(floor_height), 1.0)
    else:
        h = _to_float(fallback_height)

    if h is None or not np.isfinite(h):
        h = max(float(fallback_height), float(min_height))
    return float(np.clip(h, float(min_height), float(max_height)))


def _clean_mesh(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.int32)

    faces = faces.astype(np.int64, copy=False)
    valid = (
        (faces[:, 0] >= 0)
        & (faces[:, 1] >= 0)
        & (faces[:, 2] >= 0)
        & (faces[:, 0] < vertices.shape[0])
        & (faces[:, 1] < vertices.shape[0])
        & (faces[:, 2] < vertices.shape[0])
    )
    faces = faces[valid]
    if faces.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.int32)

    non_degenerate = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 0] != faces[:, 2])
    )
    faces = faces[non_degenerate]
    if faces.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.int32)

    tri_pts = vertices[faces]
    areas = np.linalg.norm(np.cross(tri_pts[:, 1] - tri_pts[:, 0], tri_pts[:, 2] - tri_pts[:, 0]), axis=1)
    faces = faces[areas > 1e-10]
    if faces.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.int32)

    canonical = np.sort(faces, axis=1)
    _, unique_idx = np.unique(canonical, axis=0, return_index=True)
    unique_idx.sort()
    faces = faces[unique_idx]

    used = np.unique(faces.ravel())
    verts = vertices[used]
    remap = np.full(vertices.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.shape[0], dtype=np.int64)
    new_faces = remap[faces].astype(np.int32, copy=False)
    return verts.astype(np.float64, copy=False), new_faces


def process_buildings_from_footprints(
    footprints: list[dict],
    *,
    ground_points: np.ndarray | None = None,
    roof_points: np.ndarray | None = None,
    height_mode: str = "fixed",
    fixed_height: float = 12.0,
    fallback_height: float = 8.0,
    height_attribute: str | None = None,
    floors_attribute: str | None = None,
    floor_height: float = 3.0,
    min_height: float = 2.0,
    max_height: float = 120.0,
    min_area: float = 8.0,
    roof_percentile: float = 90.0,
    simplify_tolerance: float = 0.0,
    max_vertices_per_footprint: int = 150,
    source_epsg: int | None = None,
    target_epsg: int | None = None,
    roof_plane_tolerance: float = 0.18,
    roof_max_plane_iterations: int = 120,
    roof_density_hint: float = 1.0,
    roof_snap_planes: bool = True,
    roof_model_mode: str = "dominant_planes",
    roof_min_plane_area_m2: float = 10.0,
    footprint_shape_mode: str = "adaptive_mix",
    footprint_outside_area_threshold: float = 30.0,
    status_callback=None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Extrude external building footprints as watertight solids."""
    if not footprints:
        return None, None

    _emit_status(status_callback, "Edificios (huellas): preparando extrusion", 65, 5.0, "Edificios")

    transformer = None
    src = int(source_epsg) if source_epsg is not None else None
    dst = int(target_epsg) if target_epsg is not None else None
    if src is not None and dst is not None and src != dst:
        try:
            from pyproj import Transformer
        except ImportError as exc:
            raise ImportError("Para transformar EPSG de huellas necesitas instalar 'pyproj'.") from exc
        transformer = Transformer.from_crs(f"EPSG:{src}", f"EPSG:{dst}", always_xy=True)
        _emit_status(
            status_callback,
            f"Edificios (huellas): reproyectando EPSG:{src} -> EPSG:{dst}",
            65,
            8.0,
            "Edificios",
        )

    ground_tree = None
    global_base = 0.0
    if ground_points is not None and ground_points.shape[0] >= 3:
        ground_xyz = np.asarray(ground_points, dtype=np.float64)
        ground_tree = cKDTree(ground_xyz[:, :2])
        global_base = float(np.percentile(ground_xyz[:, 2], 15.0))
    else:
        ground_xyz = None

    roof_xyz = None
    roof_tree = None
    if roof_points is not None and np.asarray(roof_points).shape[0] >= 3:
        roof_xyz = np.asarray(roof_points, dtype=np.float64)
        roof_tree = cKDTree(roof_xyz[:, :2])

    min_area = max(float(min_area), 0.0)
    simplify_tolerance = max(float(simplify_tolerance), 0.0)
    max_vertices_per_footprint = max(int(max_vertices_per_footprint), 20)
    roof_density_hint = max(float(roof_density_hint), 0.15)
    roof_min_plane_area_m2 = max(float(roof_min_plane_area_m2), 1.0)
    laz_roof_mode = str(height_mode).strip().lower() in {
        "laz_class_6",
        "laz_roof",
        "roof_from_laz",
        "cubierta_laz",
    }

    all_vertices: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    vertex_offset = 0

    total = max(len(footprints), 1)
    accepted = 0
    skipped = 0

    for idx, item in enumerate(footprints, start=1):
        ring_raw = item.get("xy") if isinstance(item, dict) else None
        props = item.get("properties", {}) if isinstance(item, dict) else {}
        if ring_raw is None:
            skipped += 1
            continue

        ring = np.asarray(ring_raw, dtype=np.float64)
        if ring.ndim != 2 or ring.shape[0] < 3:
            skipped += 1
            continue

        if transformer is not None:
            tx, ty = transformer.transform(ring[:, 0], ring[:, 1])
            ring = np.column_stack([tx, ty]).astype(np.float64, copy=False)

        ring = _normalize_polygon(
            ring,
            simplify_tolerance=simplify_tolerance,
            max_vertices=max_vertices_per_footprint,
        )
        if ring is None:
            skipped += 1
            continue

        area = abs(_polygon_signed_area(ring))
        source_area = _to_float(item.get("area_original")) if isinstance(item, dict) else None
        if source_area is None:
            source_area = _to_float(item.get("area")) if isinstance(item, dict) else None
        area_gate = max(float(area), float(source_area)) if source_area is not None else float(area)
        if area_gate <= min_area:
            skipped += 1
            continue

        manual_h = _to_float(props.get("manual_height_m")) if isinstance(props, dict) else None
        if laz_roof_mode and manual_h is None and roof_xyz is not None and roof_tree is not None:
            roof_subset = _select_roof_points_for_footprint(
                ring,
                roof_xyz,
                roof_tree,
                density_hint=roof_density_hint,
                footprint_shape_mode=footprint_shape_mode,
            )
            roof_ring = _adapt_footprint_with_roof_points(
                ring,
                roof_subset,
                density_hint=roof_density_hint,
                outside_area_threshold=footprint_outside_area_threshold,
                shape_mode=footprint_shape_mode,
            )
            min_support_points = int(
                np.clip(np.ceil(area_gate * roof_density_hint * 0.08), 6.0, 18.0)
            )
            if roof_subset.shape[0] >= min_support_points:
                roof_vertices, roof_faces = _build_footprint_roof_solid(
                    roof_ring,
                    roof_subset,
                    roof_percentile=roof_percentile,
                    ground_tree=ground_tree,
                    ground_points=ground_xyz,
                    plane_tolerance=roof_plane_tolerance,
                    max_plane_iterations=roof_max_plane_iterations,
                    density_hint=roof_density_hint,
                    snap_roof_planes=roof_snap_planes,
                    roof_model_mode=roof_model_mode,
                    min_height=min_height,
                    max_height=max_height,
                    min_plane_area_m2=roof_min_plane_area_m2,
                )
                if roof_vertices is not None and roof_faces is not None:
                    all_vertices.append(roof_vertices)
                    all_faces.append((roof_faces + vertex_offset).astype(np.int32, copy=False))
                    vertex_offset += roof_vertices.shape[0]
                    accepted += 1
                    if idx == 1 or idx == total or idx % max(total // 20, 1) == 0:
                        progress = 10.0 + (idx / total) * 85.0
                        _emit_status(
                            status_callback,
                            f"Edificios (huellas): ajustando cubierta LAZ {idx}/{total}",
                            66,
                            progress,
                            "Edificios",
                        )
                    continue

        height_m = _resolve_height(
            props if isinstance(props, dict) else {},
            height_mode=height_mode,
            fixed_height=fixed_height,
            fallback_height=fallback_height,
            height_attribute=height_attribute,
            floors_attribute=floors_attribute,
            floor_height=floor_height,
            min_height=min_height,
            max_height=max_height,
        )
        if not np.isfinite(height_m) or height_m <= 0.05:
            skipped += 1
            continue

        if laz_roof_mode and roof_xyz is not None and roof_tree is not None:
            fallback_subset = _select_roof_points_for_footprint(
                ring,
                roof_xyz,
                roof_tree,
                density_hint=roof_density_hint,
                footprint_shape_mode=footprint_shape_mode,
            )
            ring = _adapt_footprint_with_roof_points(
                ring,
                fallback_subset,
                density_hint=roof_density_hint,
                outside_area_threshold=footprint_outside_area_threshold,
                shape_mode=footprint_shape_mode,
            )

        n = ring.shape[0]
        if ground_tree is not None and ground_xyz is not None:
            _, g_idx = ground_tree.query(ring, k=1)
            base_z_vertices = ground_xyz[np.asarray(g_idx, dtype=np.int64), 2].astype(np.float64, copy=False)
        else:
            base_z_vertices = np.full(n, global_base, dtype=np.float64)

        base_level = float(np.median(base_z_vertices))
        roof_z = max(base_level + height_m, float(np.max(base_z_vertices) + 0.15))

        top_vertices = np.column_stack([ring, np.full(n, roof_z, dtype=np.float64)])
        base_vertices = np.column_stack([ring, base_z_vertices])
        local_vertices = np.vstack([top_vertices, base_vertices]).astype(np.float64, copy=False)

        top_tris = _triangulate_polygon_ear_clipping(ring)
        if not top_tris:
            skipped += 1
            continue

        local_faces: list[list[int]] = []

        for ia, ib, ic in top_tris:
            face = [int(ia), int(ib), int(ic)]
            va, vb, vc = local_vertices[face]
            if np.cross(vb - va, vc - va)[2] < 0:
                face = [face[0], face[2], face[1]]
            local_faces.append(face)

        for ia, ib, ic in top_tris:
            face = [int(ia) + n, int(ic) + n, int(ib) + n]
            va, vb, vc = local_vertices[face]
            if np.cross(vb - va, vc - va)[2] > 0:
                face = [face[0], face[2], face[1]]
            local_faces.append(face)

        center_xy = np.mean(ring, axis=0)
        for i in range(n):
            j = (i + 1) % n
            candidates = [[i, j, j + n], [i, j + n, i + n]]
            for face in candidates:
                va, vb, vc = local_vertices[face]
                normal = np.cross(vb - va, vc - va)
                center = (va + vb + vc) / 3.0
                outward = np.array([center[0] - center_xy[0], center[1] - center_xy[1], 0.0], dtype=np.float64)
                if normal[0] * outward[0] + normal[1] * outward[1] < 0:
                    face = [face[0], face[2], face[1]]
                local_faces.append(face)

        if len(local_faces) == 0:
            skipped += 1
            continue

        faces_arr = (np.asarray(local_faces, dtype=np.int32) + vertex_offset).astype(np.int32, copy=False)
        all_vertices.append(local_vertices)
        all_faces.append(faces_arr)
        vertex_offset += local_vertices.shape[0]
        accepted += 1

        if idx == 1 or idx == total or idx % max(total // 20, 1) == 0:
            progress = 10.0 + (idx / total) * 85.0
            _emit_status(status_callback, f"Edificios (huellas): procesando {idx}/{total}", 66, progress, "Edificios")

    if not all_vertices:
        _emit_status(status_callback, "Edificios (huellas): sin geometria valida", 67, 100.0, "Edificios")
        return None, None

    vertices = np.vstack(all_vertices).astype(np.float64, copy=False)
    faces = np.vstack(all_faces).astype(np.int32, copy=False)
    vertices, faces = _clean_mesh(vertices, faces)
    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        _emit_status(status_callback, "Edificios (huellas): malla vacia", 67, 100.0, "Edificios")
        return None, None

    _emit_status(
        status_callback,
        f"Edificios (huellas): completado ({accepted} aceptados, {skipped} descartados)",
        67,
        100.0,
        "Edificios",
    )
    return vertices, faces

def process_buildings_as_prisms(
    points: np.ndarray,
    classification: np.ndarray,
    cluster_radius: float = 2.0,
    min_points: int = 8,
    roof_percentile: float = 90.0,
    base_percentile: float = 10.0,
    *,
    ground_points: np.ndarray | None = None,
    plane_tolerance: float = 0.18,
    max_plane_iterations: int = 120,
    split_touching_buildings: bool = True,
    orthogonalize_edges: bool = False,
    snap_roof_planes: bool = True,
    roof_model_mode: str = "dominant_planes",
    min_roof_plane_area_m2: float = 10.0,
    density_hint: float = 1.0,
    status_callback=None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Reconstruct buildings from class 6 points as clean watertight solids.

    Args:
        points: Input XYZ points.
        classification: LAS/LAZ class array.
        cluster_radius: Max XY gap for points belonging to one building.
        min_points: Minimum points for a building cluster.
        roof_percentile: High percentile used to isolate roof candidates.
        base_percentile: Fallback base percentile if no terrain is available.
        ground_points: Optional terrain points to anchor building bases.
        plane_tolerance: Max distance to a roof plane model.
        max_plane_iterations: RANSAC iterations per cluster.
        split_touching_buildings: Try to split adjacent roofs inside one XY cluster.
        orthogonalize_edges: Rectify near-rectangular urban footprints.
        snap_roof_planes: Snap roof heights to a few dominant planes.
        roof_model_mode: Roof model selection ('auto', 'flat', 'single_plane', 'dominant_planes', 'freeform').
        min_roof_plane_area_m2: Minimum roof plane area kept in segmentation (m²).
        density_hint: Expected roof point density in pts/m².
    """
    mask_buildings = classification == 6
    if not mask_buildings.any():
        return None, None
    _emit_status(status_callback, "Edificios: filtrando puntos clase 6", 65, 5.0, "Edificios")

    building_points = points[mask_buildings].astype(np.float64, copy=False)
    if building_points.shape[0] < max(int(min_points), 3):
        return None, None

    radius_xy = max(float(cluster_radius), 0.5)
    plane_tol = max(float(plane_tolerance), 0.05)
    min_cluster_points = max(int(min_points), 4)
    max_plane_iterations = max(int(max_plane_iterations), 10)
    density_hint = max(float(density_hint), 0.15)
    min_roof_plane_area_m2 = max(float(min_roof_plane_area_m2), 1.0)

    components = _extract_xy_occupancy_components(
        building_points,
        radius_xy=radius_xy,
        density_hint=density_hint,
        min_points=min_cluster_points,
    )
    _emit_status(
        status_callback,
        f"Edificios: ocupacion XY limpia {len(components):,} componentes base",
        66,
        10.0,
        "Edificios",
    )
    ground_tree = None
    if ground_points is not None and ground_points.shape[0] >= 3:
        ground_tree = cKDTree(ground_points[:, :2])

    if not components:
        return None, None

    clusters_data: list[dict[str, np.ndarray]] = []
    for comp in components:
        cluster_pts = np.asarray(comp["points"], dtype=np.float64)
        footprint_ring = np.asarray(comp["footprint"], dtype=np.float64)
        if cluster_pts.shape[0] < min_cluster_points or footprint_ring.shape[0] < 3:
            continue

        if bool(split_touching_buildings):
            subclusters = _split_touching_cluster(
                cluster_pts,
                radius_xy=radius_xy,
                roof_percentile=roof_percentile,
                min_cluster_points=min_cluster_points,
                density_hint=density_hint,
            )
            if subclusters:
                for sub in subclusters:
                    sub = np.asarray(sub, dtype=np.float64)
                    if sub.shape[0] < min_cluster_points:
                        continue
                    sub_ring = _estimate_cluster_footprint(
                        sub[:, :2],
                        radius_xy=radius_xy,
                        orthogonalize_edges=orthogonalize_edges,
                        density_hint=density_hint,
                    )
                    if sub_ring is None or sub_ring.shape[0] < 3:
                        sub_ring = footprint_ring
                    clusters_data.append(
                        {
                            "points": sub.astype(np.float64, copy=False),
                            "footprint": np.asarray(sub_ring, dtype=np.float64),
                        }
                    )
                continue

        clusters_data.append(
            {
                "points": cluster_pts.astype(np.float64, copy=False),
                "footprint": footprint_ring.astype(np.float64, copy=False),
            }
        )

    if not clusters_data:
        return None, None

    _emit_status(
        status_callback,
        f"Edificios: {len(clusters_data)} componentes separados en planta",
        66,
        30.0,
        "Edificios",
    )

    all_vertices: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    vertex_offset = 0

    total_clusters = max(len(clusters_data), 1)
    for cluster_idx, cluster_data in enumerate(clusters_data, start=1):
        cluster_pts = np.asarray(cluster_data["points"], dtype=np.float64)
        cluster_footprint = np.asarray(cluster_data["footprint"], dtype=np.float64)
        if cluster_pts.shape[0] < min_cluster_points:
            continue

        cluster_progress = 30.0 + (cluster_idx - 1) / total_clusters * 60.0
        _emit_status(
            status_callback,
            f"Edificios: procesando cluster {cluster_idx}/{total_clusters}",
            66,
            cluster_progress,
            "Edificios",
        )
        _emit_status(
            status_callback,
            f"Edificios: reconstruyendo cubierta cluster {cluster_idx}/{total_clusters}",
            66,
            min(cluster_progress + 8.0, 96.0),
            "Edificios",
        )

        roof_vertices, roof_faces = _build_cluster_roof_solid(
            cluster_pts,
            radius_xy=max(radius_xy, plane_tol * 2.0),
            roof_percentile=roof_percentile,
            base_percentile=base_percentile,
            ground_tree=ground_tree,
            ground_points=ground_points,
            plane_tolerance=plane_tol,
            max_plane_iterations=max_plane_iterations,
            density_hint=density_hint,
            orthogonalize_edges=bool(orthogonalize_edges),
            snap_roof_planes=bool(snap_roof_planes),
            roof_model_mode=str(roof_model_mode or "auto"),
            min_plane_area_m2=min_roof_plane_area_m2,
            footprint_override=cluster_footprint,
        )
        if roof_vertices is None or roof_faces is None or roof_faces.shape[0] == 0:
            continue

        faces_arr = (np.asarray(roof_faces, dtype=np.int32) + vertex_offset).astype(np.int32, copy=False)
        all_vertices.append(np.asarray(roof_vertices, dtype=np.float64))
        all_faces.append(faces_arr)
        vertex_offset += roof_vertices.shape[0]

    if not all_vertices:
        return None, None

    combined_vertices = np.vstack(all_vertices).astype(np.float64, copy=False)
    combined_faces = np.vstack(all_faces).astype(np.int32, copy=False)
    _emit_status(status_callback, "Edificios completados", 67, 100.0, "Edificios")
    return combined_vertices, combined_faces


def reclassify_building_vegetation_noise(
    points: np.ndarray,
    classification: np.ndarray,
    *,
    cluster_radius: float = 2.0,
    min_points: int = 12,
    roof_percentile: float = 90.0,
    plane_tolerance: float = 0.18,
    density_hint: float = 1.0,
    aggressiveness: float = 1.0,
    target_class: int = 5,
    status_callback=None,
) -> tuple[np.ndarray, int]:
    """Move class-6 canopy outliers to high vegetation before building reconstruction."""
    cls = np.asarray(classification).copy()
    mask_buildings = cls == 6
    if not mask_buildings.any():
        return cls, 0

    building_idx = np.flatnonzero(mask_buildings)
    building_points = np.asarray(points[building_idx], dtype=np.float64)
    min_cluster_points = max(int(min_points), 6)
    if building_points.shape[0] < min_cluster_points:
        return cls, 0

    radius_xy = max(float(cluster_radius), 0.75)
    plane_tol = max(float(plane_tolerance), 0.05)
    density_hint = max(float(density_hint), 0.15)
    aggressiveness = float(np.clip(aggressiveness, 0.4, 2.5))
    nominal_spacing = float(np.sqrt(1.0 / density_hint))
    local_radius = max(nominal_spacing * 1.75, 1.10)
    outlier_above = max(0.42, plane_tol * (2.8 / aggressiveness), nominal_spacing * (0.65 / aggressiveness))
    support_tol = max(0.16, plane_tol * 1.45, nominal_spacing * 0.22)
    support_ratio_min = float(np.clip(0.42 - (aggressiveness - 1.0) * 0.05, 0.30, 0.50))
    support_ratio_max = float(np.clip(0.52 - (aggressiveness - 1.0) * 0.08, 0.28, 0.55))
    scatter_min = float(np.clip(0.10 - (aggressiveness - 1.0) * 0.02, 0.05, 0.12))
    planar_max = float(np.clip(0.34 + (1.0 - aggressiveness) * 0.06, 0.22, 0.40))
    support_cell = max(nominal_spacing * 0.72, 0.18)

    _emit_status(
        status_callback,
        "Edificios: limpiando vegetacion colada en clase 6",
        64,
        5.0,
        "Edificios",
    )

    tree = cKDTree(building_points[:, :2])
    visited = np.zeros(building_points.shape[0], dtype=bool)
    flagged_local = np.zeros(building_points.shape[0], dtype=bool)
    clusters: list[np.ndarray] = []
    cluster_local_ids: list[np.ndarray] = []

    for seed_idx in range(building_points.shape[0]):
        if visited[seed_idx]:
            continue
        queue = [seed_idx]
        component: list[int] = []
        while queue:
            idx = queue.pop()
            if visited[idx]:
                continue
            visited[idx] = True
            component.append(idx)
            for nb in tree.query_ball_point(building_points[idx, :2], r=radius_xy):
                if not visited[nb]:
                    queue.append(nb)
        if len(component) >= min_cluster_points:
            comp_ids = np.asarray(component, dtype=np.int64)
            clusters.append(building_points[comp_ids])
            cluster_local_ids.append(comp_ids)

    if not clusters:
        return cls, 0

    total_clusters = max(len(clusters), 1)
    for cluster_idx, (cluster_pts, local_ids) in enumerate(zip(clusters, cluster_local_ids), start=1):
        roof_core = _dedupe_xy_keep_highest(cluster_pts[:, :3], cell_size=support_cell)
        if roof_core.shape[0] < max(6, min_cluster_points // 2):
            continue
        support_radius = max(local_radius * 0.95, nominal_spacing * 1.40, 0.85)
        if roof_core.shape[0] >= 8:
            try:
                core_tree = cKDTree(roof_core[:, :2])
                support_keep = np.ones(roof_core.shape[0], dtype=bool)
                for core_idx in range(roof_core.shape[0]):
                    nb_core = core_tree.query_ball_point(roof_core[core_idx, :2], r=support_radius)
                    if len(nb_core) < 5:
                        continue
                    neigh_core = roof_core[np.asarray(nb_core, dtype=np.int64)]
                    z_med_core = float(np.median(neigh_core[:, 2]))
                    z_iqr_core = float(np.subtract(*np.percentile(neigh_core[:, 2], [75.0, 25.0])))
                    if roof_core[core_idx, 2] > z_med_core + max(0.45, 1.8 * z_iqr_core, outlier_above * 0.95):
                        support_keep[core_idx] = False
                if int(support_keep.sum()) >= max(6, min_cluster_points // 2):
                    roof_core = roof_core[support_keep]
            except Exception:
                pass

        planes = _extract_dominant_roof_planes(
            roof_core,
            residual_tol=max(plane_tol * 1.2, nominal_spacing * 0.22, 0.12),
            max_planes=4,
            min_points_per_plane=max(5, int(5 + density_hint * 1.5)),
            max_iterations=100,
        )
        if not planes:
            coeff, normal = _fit_plane_lstsq(roof_core)
            if coeff is not None and normal is not None and float(normal[2]) > 0.55:
                planes = [coeff]
        if not planes:
            continue

        local_tree = cKDTree(cluster_pts[:, :2])
        for point_idx in range(cluster_pts.shape[0]):
            point_xy = cluster_pts[point_idx : point_idx + 1, :2]
            point_z = float(cluster_pts[point_idx, 2])
            plane_preds = np.asarray(
                [_predict_plane_height(coeff, point_xy)[0] for coeff in planes],
                dtype=np.float64,
            )
            signed_above = point_z - plane_preds
            best_plane_idx = int(np.argmin(np.abs(signed_above)))
            best_above = float(signed_above[best_plane_idx])
            if best_above <= outlier_above:
                continue

            nb_idx = local_tree.query_ball_point(cluster_pts[point_idx, :2], r=local_radius)
            if len(nb_idx) < 4:
                if best_above > outlier_above * 1.35:
                    flagged_local[local_ids[point_idx]] = True
                continue

            neigh = cluster_pts[np.asarray(nb_idx, dtype=np.int64)]
            pred_nb = _predict_plane_height(planes[best_plane_idx], neigh[:, :2])
            resid_nb = np.abs(neigh[:, 2] - pred_nb)
            support_ratio = float(np.mean(resid_nb <= support_tol))
            local_planarity, local_scattering = _planarity_metrics(neigh)
            z_median = float(np.median(neigh[:, 2]))
            z_p85 = float(np.percentile(neigh[:, 2], 85.0))
            z_iqr = float(np.subtract(*np.percentile(neigh[:, 2], [75.0, 25.0])))
            high_above_local = point_z > max(
                z_median + max(0.30, 1.35 * z_iqr),
                z_p85 + max(0.18, 0.75 * z_iqr),
            )
            canopy_shape = local_scattering >= scatter_min or local_planarity <= planar_max

            if (
                (
                    high_above_local
                    and support_ratio <= support_ratio_max
                    and (canopy_shape or best_above > outlier_above * 1.20)
                )
                or (high_above_local and best_above > outlier_above * 1.85)
            ):
                flagged_local[local_ids[point_idx]] = True

        if cluster_idx == 1 or cluster_idx == total_clusters or cluster_idx % max(total_clusters // 10, 1) == 0:
            progress = 10.0 + (cluster_idx / total_clusters) * 85.0
            _emit_status(
                status_callback,
                f"Edificios: limpiando ruido clase 6 {cluster_idx}/{total_clusters}",
                64,
                progress,
                "Edificios",
            )

    flagged_count = int(flagged_local.sum())
    if flagged_count > 0:
        cls[building_idx[flagged_local]] = int(target_class)
        _emit_status(
            status_callback,
            f"Edificios: {flagged_count:,} puntos reclasificados de clase 6 a vegetacion alta",
            64,
            100.0,
            "Edificios",
        )
    return cls, flagged_count
