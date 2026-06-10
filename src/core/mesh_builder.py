"""
Mesh construction and post-processing utilities.

This module centralizes terrain/vegetation/building fusion for STL export.
"""

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt
from scipy.spatial import ConvexHull, Delaunay, QhullError, cKDTree

from src.core.building_processor import (
    process_buildings_as_prisms,
    process_buildings_from_footprints,
    reclassify_building_vegetation_noise,
)
from src.core.terrain_processor import process_terrain_high_detail
from src.core.vegetation_processor import (
    process_vegetation_as_metaballs,
    process_vegetation_printable_mass,
    process_vegetation_poisson,
    process_vegetation_stylized,
)
from src.utils.geometry import smooth_grid


class GridTriangulationResult:
    """Lightweight result compatible with app.py usage (`simplices`, `convex_hull`)."""

    def __init__(self, simplices: np.ndarray, convex_hull: np.ndarray):
        self.simplices = simplices
        self.convex_hull = convex_hull


def _emit_status(status_callback, message: str, pct: int | None = None, step_pct: float | None = None, step_label: str | None = None) -> None:
    """Emit status updates with backward-compatible callback signature."""
    if status_callback is None:
        return
    try:
        status_callback(message, pct, step_pct, step_label)
    except TypeError:
        status_callback(message, pct)


def reassign_overlap_reserved_class(
    points: np.ndarray,
    classification: np.ndarray | None,
    *,
    overlap_class: int = 12,
    reference_classes: tuple[int, ...] = (2, 3, 4, 5, 6),
    building_class: int = 6,
    allow_building_target: bool = True,
    building_density_hint: float = 1.0,
    building_xy_factor: float = 0.85,
    building_max_z_diff: float = 0.35,
    building_support_min_neighbors: int = 4,
    max_overlap_to_building_ratio: float = 0.10,
    chunk_size: int = 250_000,
) -> tuple[np.ndarray | None, int, dict[int, int]]:
    """
    Reassign class-12 overlap/reserved points to the nearest semantic class.

    Strategy:
    - Query nearest neighbor in 3D from reference classes (2/3/4/5/6 by default)
    - Copy that class label to each overlap point
    """
    if classification is None or points is None:
        return classification, 0, {}

    pts = np.asarray(points, dtype=np.float64)
    cls = np.asarray(classification)
    if pts.ndim != 2 or pts.shape[0] == 0 or cls.shape[0] != pts.shape[0]:
        return classification, 0, {}

    overlap_mask = cls == int(overlap_class)
    overlap_idx = np.flatnonzero(overlap_mask)
    if overlap_idx.size == 0:
        return cls, 0, {}

    reference_mask = np.isin(cls, np.asarray(reference_classes, dtype=cls.dtype)) & (~overlap_mask)
    reference_idx = np.flatnonzero(reference_mask)
    if reference_idx.size == 0:
        return cls, 0, {}

    ref_points = pts[reference_idx, :3]
    ref_classes = cls[reference_idx]
    if not bool(allow_building_target):
        keep_nb = ref_classes != int(building_class)
        ref_points = ref_points[keep_nb]
        ref_classes = ref_classes[keep_nb]
    if ref_points.shape[0] == 0:
        return cls, 0, {}
    tree = cKDTree(ref_points)

    out = cls.copy()
    chunk_size = max(int(chunk_size), 10_000)
    for start in range(0, overlap_idx.size, chunk_size):
        stop = min(start + chunk_size, overlap_idx.size)
        idx_chunk = overlap_idx[start:stop]
        query_points = pts[idx_chunk, :3]
        try:
            _, nearest_idx = tree.query(query_points, k=1, workers=-1)
        except TypeError:
            _, nearest_idx = tree.query(query_points, k=1)
        out[idx_chunk] = ref_classes[np.asarray(nearest_idx, dtype=np.int64)]

    # Guard rail for building reassignment:
    # only keep overlap->building when it is spatially consistent with original class-6 support.
    bcls = int(building_class)
    reassigned_building_idx = np.flatnonzero((overlap_mask) & (out == bcls))
    original_building_idx = np.flatnonzero((cls == bcls) & (~overlap_mask))
    if bool(allow_building_target) and reassigned_building_idx.size > 0 and original_building_idx.size > 0:
        nominal_spacing = float(np.sqrt(1.0 / max(float(building_density_hint), 0.15)))
        xy_limit = max(0.9, nominal_spacing * float(building_xy_factor))
        z_limit = max(0.8, float(building_max_z_diff))
        support_radius = max(0.75, nominal_spacing * 1.25)

        b_xy = pts[original_building_idx, :2]
        b_z = pts[original_building_idx, 2]
        tree_b_xy = cKDTree(b_xy)
        q_xy = pts[reassigned_building_idx, :2]
        q_z = pts[reassigned_building_idx, 2]
        try:
            d_xy, nn_idx = tree_b_xy.query(q_xy, k=1, workers=-1)
        except TypeError:
            d_xy, nn_idx = tree_b_xy.query(q_xy, k=1)
        nn_idx = np.asarray(nn_idx, dtype=np.int64)
        d_xy = np.asarray(d_xy, dtype=np.float64)
        d_z = np.abs(q_z - b_z[nn_idx])
        neighbor_ids = tree_b_xy.query_ball_point(q_xy, r=support_radius)
        support_counts = np.fromiter((len(ids) for ids in neighbor_ids), dtype=np.int32, count=len(neighbor_ids))
        keep_building = (
            (d_xy <= xy_limit)
            & (d_z <= z_limit)
            & (support_counts >= int(max(1, building_support_min_neighbors)))
        )
        # Vertical guard rail:
        # reject overlap points too low/high relative to original class-6 envelope.
        z_low = float(np.percentile(b_z, 20.0))
        z_high = float(np.percentile(b_z, 99.0))
        keep_building &= (q_z >= (z_low - 0.25)) & (q_z <= (z_high + 0.45))

        accepted_idx = np.flatnonzero(keep_building)
        max_allowed = int(max(6, np.floor(original_building_idx.size * float(max_overlap_to_building_ratio))))
        if accepted_idx.size > max_allowed:
            score = (d_xy / max(xy_limit, 1e-6)) + (d_z / max(z_limit, 1e-6))
            accepted_sorted = accepted_idx[np.argsort(score[accepted_idx], kind="stable")]
            accepted_keep = accepted_sorted[:max_allowed]
            keep_mask_cap = np.zeros_like(keep_building, dtype=bool)
            keep_mask_cap[accepted_keep] = True
            keep_building = keep_mask_cap

        rejected = reassigned_building_idx[~keep_building]
        if rejected.size > 0:
            non_build_ref_idx = np.flatnonzero(np.isin(cls, np.asarray((2, 3, 4, 5), dtype=cls.dtype)))
            if non_build_ref_idx.size > 0:
                nb_points = pts[non_build_ref_idx, :3]
                nb_classes = cls[non_build_ref_idx]
                tree_nb = cKDTree(nb_points)
                q_nb = pts[rejected, :3]
                try:
                    _, nn_nb = tree_nb.query(q_nb, k=1, workers=-1)
                except TypeError:
                    _, nn_nb = tree_nb.query(q_nb, k=1)
                out[rejected] = nb_classes[np.asarray(nn_nb, dtype=np.int64)]
            else:
                out[rejected] = overlap_class

    moved = int(overlap_idx.size)
    unique_cls, unique_count = np.unique(out[overlap_idx], return_counts=True)
    summary = {int(k): int(v) for k, v in zip(unique_cls.tolist(), unique_count.tolist())}
    return out, moved, summary


def _clean_mesh(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove invalid/degenerate/duplicated triangles and compact vertices."""
    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        return vertices, faces

    faces = faces.astype(np.int64, copy=False)
    valid_idx = (
        (faces[:, 0] >= 0)
        & (faces[:, 1] >= 0)
        & (faces[:, 2] >= 0)
        & (faces[:, 0] < vertices.shape[0])
        & (faces[:, 1] < vertices.shape[0])
        & (faces[:, 2] < vertices.shape[0])
    )
    faces = faces[valid_idx]
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
    areas = np.linalg.norm(
        np.cross(tri_pts[:, 1] - tri_pts[:, 0], tri_pts[:, 2] - tri_pts[:, 0]),
        axis=1,
    )
    faces = faces[areas > 1e-10]
    if faces.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.int32)

    canonical = np.sort(faces, axis=1)
    _, unique_idx = np.unique(canonical, axis=0, return_index=True)
    unique_idx.sort()
    faces = faces[unique_idx]

    used = np.unique(faces.ravel())
    new_vertices = vertices[used]
    remap = np.full(vertices.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.shape[0], dtype=np.int64)
    new_faces = remap[faces].astype(np.int32, copy=False)
    return new_vertices.astype(np.float64, copy=False), new_faces


def _fill_nans_with_nearest(grid: np.ndarray) -> np.ndarray:
    """Fill NaNs from nearest valid cells."""
    if grid.size == 0:
        return grid
    nan_mask = np.isnan(grid)
    if not nan_mask.any() or nan_mask.all():
        return grid

    nearest_idx = distance_transform_edt(
        nan_mask,
        return_distances=False,
        return_indices=True,
    )
    filled = grid.copy()
    filled[nan_mask] = grid[tuple(nearest_idx)][nan_mask]
    return filled


def _rasterize_layer_to_grid(
    layer_points: np.ndarray,
    x_min: float,
    y_min: float,
    resolution: float,
    num_x: int,
    num_y: int,
    *,
    aggregation: str = "max",
    progress_callback=None,
) -> np.ndarray:
    """Fast O(N) rasterization from points to regular grid."""
    total_cells = num_x * num_y
    if layer_points.shape[0] == 0:
        return np.full((num_y, num_x), np.nan, dtype=np.float64)

    ix = np.floor((layer_points[:, 0] - x_min) / resolution).astype(np.int64)
    iy = np.floor((layer_points[:, 1] - y_min) / resolution).astype(np.int64)
    ix = np.clip(ix, 0, num_x - 1)
    iy = np.clip(iy, 0, num_y - 1)
    flat = iy * num_x + ix
    z = layer_points[:, 2].astype(np.float64, copy=False)
    n_points = layer_points.shape[0]
    chunk_size = min(max(n_points // 24, 120_000), 350_000)
    chunk_size = max(chunk_size, 50_000)

    if aggregation == "mean":
        sums = np.zeros(total_cells, dtype=np.float64)
        counts = np.zeros(total_cells, dtype=np.int32)
        for start in range(0, n_points, chunk_size):
            end = min(start + chunk_size, n_points)
            flat_chunk = flat[start:end]
            z_chunk = z[start:end]
            np.add.at(sums, flat_chunk, z_chunk)
            np.add.at(counts, flat_chunk, 1)
            if progress_callback is not None:
                progress_callback((end / n_points) * 100.0)
        grid_flat = np.full(total_cells, np.nan, dtype=np.float64)
        valid = counts > 0
        grid_flat[valid] = sums[valid] / counts[valid]
    else:
        grid_flat = np.full(total_cells, -np.inf, dtype=np.float64)
        for start in range(0, n_points, chunk_size):
            end = min(start + chunk_size, n_points)
            np.maximum.at(grid_flat, flat[start:end], z[start:end])
            if progress_callback is not None:
                progress_callback((end / n_points) * 100.0)
        grid_flat[grid_flat == -np.inf] = np.nan

    return grid_flat.reshape(num_y, num_x)


def _interpolate_layer(
    *,
    layer_name: str,
    layer_points: np.ndarray,
    grid_x_2d: np.ndarray,
    grid_y_2d: np.ndarray,
    x_min: float,
    y_min: float,
    resolution: float,
    num_x: int,
    num_y: int,
    method: str,
    notify,
    pct: int,
    large_cloud_threshold: int = 400_000,
) -> np.ndarray:
    """Interpolate one semantic layer with an adaptive strategy."""
    if layer_points.shape[0] == 0:
        return np.full((num_y, num_x), np.nan, dtype=np.float64)

    use_fast_raster = method == "nearest" or layer_points.shape[0] > large_cloud_threshold
    if use_fast_raster:
        layer_progress_text = f"Rasterizando {layer_name.lower()} en rejilla rapida ({layer_points.shape[0]:,} puntos)"
        notify(layer_progress_text, pct, 0.0, layer_name)

        def on_progress(progress_pct: float) -> None:
            overall = min(74, pct + int((progress_pct / 100.0) * 3.0))
            notify(layer_progress_text, overall, progress_pct, layer_name)

        aggregation = "max" if layer_name in {"Vegetacion", "Edificios", "Otros"} else "mean"
        grid = _rasterize_layer_to_grid(
            layer_points,
            x_min,
            y_min,
            resolution,
            num_x,
            num_y,
            aggregation=aggregation,
            progress_callback=on_progress,
        )
        nan_ratio = float(np.isnan(grid).mean()) if grid.size else 1.0
        allow_hole_fill = layer_name in {"Terreno", "Superficie base"}
        if allow_hole_fill and 0.0 < nan_ratio < 1.0 and grid.size <= 2_000_000:
            notify(
                f"Rellenando huecos de {layer_name.lower()} ({nan_ratio * 100:.1f}% vacio)",
                min(pct + 1, 74),
                85.0,
                layer_name,
            )
            grid = _fill_nans_with_nearest(grid)
            notify(f"{layer_name} rasterizado", min(pct + 2, 74), 100.0, layer_name)
        elif not allow_hole_fill:
            notify(f"{layer_name} rasterizado sin extrapolacion", min(pct + 2, 74), 100.0, layer_name)
        return grid

    notify(
        f"Interpolando {layer_name.lower()} ({layer_points.shape[0]:,} puntos)",
        pct,
        15.0,
        layer_name,
    )
    grid = griddata(
        layer_points[:, :2],
        layer_points[:, 2],
        (grid_x_2d, grid_y_2d),
        method=method,
        fill_value=np.nan,
    )
    notify(f"Interpolacion completada de {layer_name.lower()}", min(pct + 1, 74), 100.0, layer_name)

    if np.all(np.isnan(grid)) and method != "nearest":
        notify(f"Sin triangulacion valida en {layer_name.lower()}, usando nearest", min(pct + 1, 74))
        grid = griddata(
            layer_points[:, :2],
            layer_points[:, 2],
            (grid_x_2d, grid_y_2d),
            method="nearest",
            fill_value=np.nan,
        )
    return grid


def build_mesh(
    points: np.ndarray,
    classification: np.ndarray | None,
    resolution: float,
    method: str,
    smooth_values: tuple[float, float, float],
    status_callback,
    reassign_overlap_class12: bool = True,
    overlap_assign_to_buildings: bool = True,
    overlap_building_density_hint: float = 1.0,
) -> tuple[np.ndarray, GridTriangulationResult]:
    """Build a mesh using grid interpolation (simple mode)."""
    smooth_ground, smooth_vegetation, smooth_buildings = smooth_values

    def notify(
        message: str,
        pct: int | None = None,
        step_pct: float | None = None,
        step_label: str | None = None,
    ) -> None:
        _emit_status(status_callback, message, pct, step_pct, step_label)

    if classification is not None and bool(reassign_overlap_class12):
        classification, moved_overlap, overlap_summary = reassign_overlap_reserved_class(
            points,
            classification,
            allow_building_target=bool(overlap_assign_to_buildings),
            building_density_hint=float(overlap_building_density_hint),
        )
        if moved_overlap > 0:
            summary_txt = ", ".join(f"{int(k):02d}:{int(v):,}" for k, v in sorted(overlap_summary.items()))
            notify(
                f"Clase 12 redistribuida por proximidad: {moved_overlap:,} puntos ({summary_txt})",
                49,
                100.0,
                "Clases",
            )

    notify("Calculando rejilla en XY", 50, 5.0, "Rejilla")
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    if np.isclose(x_min, x_max) or np.isclose(y_min, y_max):
        raise ValueError("La nube ocupa un area demasiado pequena para generar una malla.")

    effective_resolution = float(max(resolution, 0.05))
    num_x = max(int(np.ceil((x_max - x_min) / effective_resolution)) + 1, 2)
    num_y = max(int(np.ceil((y_max - y_min) / effective_resolution)) + 1, 2)

    max_cells = 6_000_000
    total_cells = int(num_x * num_y)
    if total_cells > max_cells:
        scale_factor = float(np.sqrt(total_cells / max_cells))
        effective_resolution *= scale_factor
        num_x = max(int(np.ceil((x_max - x_min) / effective_resolution)) + 1, 2)
        num_y = max(int(np.ceil((y_max - y_min) / effective_resolution)) + 1, 2)
        total_cells = int(num_x * num_y)
        notify(
            f"Resolucion ajustada automaticamente a {effective_resolution:.3f} m para evitar bloqueo",
            51,
            35.0,
            "Rejilla",
        )

    notify(f"Rejilla objetivo: {num_x:,} x {num_y:,} ({total_cells:,} celdas)", 52, 100.0, "Rejilla")
    grid_x = np.linspace(x_min, x_max, num_x)
    grid_y = np.linspace(y_min, y_max, num_y)
    grid_x_2d, grid_y_2d = np.meshgrid(grid_x, grid_y)

    grids: list[np.ndarray] = []
    if classification is not None:
        layers = [
            ("Terreno", classification == 2, smooth_ground, method),
            ("Vegetacion", np.isin(classification, [3, 4, 5]), smooth_vegetation, method),
            ("Edificios", classification == 6, smooth_buildings, "nearest"),
            ("Otros", ~(np.isin(classification, [2, 3, 4, 5, 6])), 0.0, method),
        ]
        active_layers = [(name, mask, sigma, interp_method) for name, mask, sigma, interp_method in layers if mask.any()]
        if not active_layers:
            active_layers = [("Superficie base", np.ones(points.shape[0], dtype=bool), smooth_ground, method)]

        pct_start, pct_end = 55, 72
        pct_span = max(pct_end - pct_start, 1)

        for idx, (name, mask, sigma, interp_method) in enumerate(active_layers):
            pct = pct_start + int((idx * pct_span) / max(len(active_layers), 1))
            if not mask.any():
                continue
            layer_points = points[mask]
            grid = _interpolate_layer(
                layer_name=name,
                layer_points=layer_points,
                grid_x_2d=grid_x_2d,
                grid_y_2d=grid_y_2d,
                x_min=x_min,
                y_min=y_min,
                resolution=effective_resolution,
                num_x=num_x,
                num_y=num_y,
                method=interp_method,
                notify=notify,
                pct=pct,
            )
            grid = smooth_grid(grid, sigma)
            grids.append(grid)
    else:
        base_grid = _interpolate_layer(
            layer_name="Superficie base",
            layer_points=points,
            grid_x_2d=grid_x_2d,
            grid_y_2d=grid_y_2d,
            x_min=x_min,
            y_min=y_min,
            resolution=effective_resolution,
            num_x=num_x,
            num_y=num_y,
            method=method,
            notify=notify,
            pct=55,
        )
        grids.append(base_grid)

    valid_grids = [grid for grid in grids if grid is not None and not np.all(np.isnan(grid))]
    if not valid_grids:
        raise ValueError("La interpolacion genero solo valores nulos. Ajusta los parametros.")

    notify("Fusionando capas interpoladas", 73, 75.0, "Fusion")
    combined = np.nanmax(np.stack(valid_grids, axis=0), axis=0)
    if np.all(np.isnan(combined)):
        raise ValueError("La interpolacion genero solo valores nulos. Ajusta los parametros.")

    notify("Construyendo nube regular", 75, 90.0, "Fusion")
    valid = ~np.isnan(combined)
    if np.count_nonzero(valid) < 3:
        raise ValueError("La superficie interpolada no contiene puntos suficientes.")

    vertex_map = np.full(valid.shape, -1, dtype=np.int64)
    valid_rows, valid_cols = np.where(valid)
    mesh_points = np.column_stack(
        (
            grid_x_2d[valid_rows, valid_cols],
            grid_y_2d[valid_rows, valid_cols],
            combined[valid_rows, valid_cols],
        )
    ).astype(np.float64, copy=False)
    vertex_map[valid_rows, valid_cols] = np.arange(mesh_points.shape[0], dtype=np.int64)

    notify("Triangulando malla regular", 79, 25.0, "Triangulacion")
    cell_valid = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    ci, cj = np.where(cell_valid)
    if ci.size == 0:
        notify("Triangulacion regular vacia, aplicando Delaunay de respaldo", 80, 60.0, "Triangulacion")
        tri = Delaunay(mesh_points[:, :2])
        hull_edges = tri.convex_hull.astype(np.int32, copy=False)
        simplices = tri.simplices.astype(np.int32, copy=False)
        return mesh_points, GridTriangulationResult(simplices=simplices, convex_hull=hull_edges)

    a = vertex_map[ci, cj]
    b = vertex_map[ci, cj + 1]
    c = vertex_map[ci + 1, cj]
    d = vertex_map[ci + 1, cj + 1]
    tri1 = np.column_stack((a, b, d))
    tri2 = np.column_stack((a, d, c))
    simplices = np.vstack((tri1, tri2)).astype(np.int32, copy=False)

    notify("Calculando borde exterior", 80, 85.0, "Triangulacion")
    try:
        hull_edges = ConvexHull(mesh_points[:, :2]).simplices.astype(np.int32, copy=False)
    except QhullError:
        if mesh_points.shape[0] >= 2:
            idx = np.arange(mesh_points.shape[0] - 1, dtype=np.int32)
            hull_edges = np.column_stack((idx, idx + 1))
        else:
            hull_edges = np.empty((0, 2), dtype=np.int32)

    notify("Triangulacion completada", 80, 100.0, "Triangulacion")
    return mesh_points, GridTriangulationResult(simplices=simplices, convex_hull=hull_edges)


def build_mesh_advanced(
    points: np.ndarray,
    classification: np.ndarray | None,
    status_callback,
    return_number: np.ndarray | None = None,
    num_returns: np.ndarray | None = None,
    sphere_radius: float = 0.5,
    terrain_resolution: float = 0.5,
    max_vertical_error: float | None = None,
    max_slope_deg: float | None = 40.0,
    metaball_grid: float = 0.25,
    metaball_threshold: float = 0.5,
    metaball_max_dim: int = 100,
    building_cluster_radius: float = 2.0,
    building_min_points: int = 12,
    building_roof_percentile: float = 90.0,
    building_base_percentile: float = 10.0,
    building_plane_tolerance: float = 0.18,
    building_split_touching_clusters: bool = True,
    building_orthogonalize_edges: bool = False,
    building_snap_roof_planes: bool = True,
    building_roof_model_mode: str = "dominant_planes",
    building_min_roof_plane_area_m2: float = 10.0,
    building_density_hint: float = 1.0,
    building_clean_class6_noise: bool = False,
    building_class6_noise_aggressiveness: float = 1.0,
    building_source: str = "laz",
    building_footprints: list[dict] | None = None,
    building_footprint_height_mode: str = "fixed",
    building_footprint_height_field: str | None = None,
    building_footprint_floors_field: str | None = None,
    building_footprint_fixed_height: float = 12.0,
    building_footprint_fallback_height: float = 8.0,
    building_footprint_floor_height: float = 3.0,
    building_footprint_min_height: float = 2.0,
    building_footprint_max_height: float = 120.0,
    building_footprint_min_area: float = 8.0,
    building_footprint_simplify_tolerance: float = 0.0,
    building_footprint_max_vertices: int = 150,
    building_footprint_source_epsg: int | None = None,
    building_footprint_target_epsg: int | None = None,
    building_footprint_shape_mode: str = "adaptive_mix",
    building_footprint_outside_area_threshold: float = 30.0,
    vegetation_max_points: int = 75_000,
    vegetation_poisson_depth: int = 8,
    vegetation_density_quantile: float = 0.01,
    vegetation_mode: str = "realistic",
    vegetation_style_min_area: float = 12.0,
    vegetation_style_detail: int = 2,
    vegetation_style_include_low: bool = False,
    vegetation_style_max_clusters: int = 500,
    vegetation_style_roughness: float = 0.45,
    vegetation_style_density_response: float = 1.0,
    vegetation_style_texture_pitch: float = 2.0,
    vegetation_style_relief_cap: float = 3.0,
    vegetation_style_min_density: float = 0.04,
    vegetation_mass_cell_size: float = 0.8,
    vegetation_mass_include_low: bool = False,
    vegetation_mass_min_height: float = 0.9,
    vegetation_mass_min_density: float = 0.03,
    vegetation_mass_min_ratio: float = 0.16,
    vegetation_mass_min_patch_area: float = 12.0,
    vegetation_mass_close_radius: float = 1.2,
    vegetation_mass_open_radius: float = 0.6,
    vegetation_mass_texture_pitch: float = 1.1,
    vegetation_mass_roughness: float = 0.58,
    vegetation_mass_density_response: float = 1.0,
    vegetation_mass_relief_cap: float = 2.6,
    vegetation_mass_relief_floor: float = 0.45,
    vegetation_mass_base_embed: float = 0.18,
    vegetation_mass_edge_softness: float = 1.0,
    vegetation_mass_organic_smooth: float = 0.9,
    vegetation_mass_micro_detail: float = 0.85,
    vegetation_mass_max_cells: int = 450_000,
    building_max_plane_iterations: int = 120,
    reassign_overlap_class12: bool = True,
    overlap_assign_to_buildings: bool = True,
    overlap_building_density_hint: float = 1.0,
    return_components: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, tuple[np.ndarray | None, np.ndarray | None]]]:
    """
    Advanced workflow:
    1) terrain TIN
    2) vegetation volume
    3) building solids
    4) layer merge
    """
    if classification is None:
        raise ValueError("El modo avanzado requiere datos de clasificacion.")

    def notify(
        message: str,
        pct: int | None = None,
        step_pct: float | None = None,
        step_label: str | None = None,
    ) -> None:
        _emit_status(status_callback, message, pct, step_pct, step_label)

    if bool(reassign_overlap_class12):
        classification, moved_overlap, overlap_summary = reassign_overlap_reserved_class(
            points,
            classification,
            allow_building_target=bool(overlap_assign_to_buildings),
            building_density_hint=float(overlap_building_density_hint),
        )
        if moved_overlap > 0:
            summary_txt = ", ".join(f"{int(k):02d}:{int(v):,}" for k, v in sorted(overlap_summary.items()))
            notify(
                f"Clase 12 redistribuida por proximidad: {moved_overlap:,} puntos ({summary_txt})",
                54,
                100.0,
                "Clases",
            )

    notify("Paso 2 - Terreno TIN de alta fidelidad", 55, 0.0, "Terreno")
    terrain = process_terrain_high_detail(
        points,
        classification,
        resolution=terrain_resolution,
        return_number=return_number,
        num_returns=num_returns,
        status_callback=status_callback,
        max_vertical_error=max_vertical_error,
        max_slope_deg=max_slope_deg,
    )
    if terrain[0] is None:
        notify("Advertencia: No se pudo generar el terreno. Se usara una base plana.", 57, 100.0, "Terreno")
        try:
            tri = Delaunay(points[:, :2])
            terrain = (points.astype(np.float64, copy=True), tri.simplices.astype(np.int32))
        except (QhullError, ValueError):
            raise ValueError("No se pudo generar una malla de terreno base.")

    notify("Terreno completado", 59, 100.0, "Terreno")

    if bool(building_clean_class6_noise):
        classification, reclass_count = reclassify_building_vegetation_noise(
            points,
            classification,
            cluster_radius=building_cluster_radius,
            min_points=building_min_points,
            roof_percentile=building_roof_percentile,
            plane_tolerance=building_plane_tolerance,
            density_hint=building_density_hint,
            aggressiveness=building_class6_noise_aggressiveness,
            target_class=5,
            status_callback=status_callback,
        )
        if reclass_count > 0:
            notify(
                f"Clase 6 limpiada: {reclass_count:,} puntos movidos a vegetacion alta",
                59,
                100.0,
                "Edificios",
            )

    notify("Paso 3 - Reconstruyendo vegetacion", 60, 0.0, "Vegetacion")
    vegetation_mode_norm = str(vegetation_mode).strip().lower()
    if vegetation_mode_norm in {"printable_mass", "masa", "imprimible", "print"}:
        notify("Modo vegetacion de masa imprimible", 60, 8.0, "Vegetacion")
        vegetation = process_vegetation_printable_mass(
            points,
            classification,
            ground_points=terrain[0],
            cell_size=float(vegetation_mass_cell_size),
            include_low_vegetation=bool(vegetation_mass_include_low),
            min_height=float(vegetation_mass_min_height),
            min_density=float(vegetation_mass_min_density),
            min_ratio=float(vegetation_mass_min_ratio),
            min_patch_area=float(vegetation_mass_min_patch_area),
            close_radius=float(vegetation_mass_close_radius),
            open_radius=float(vegetation_mass_open_radius),
            texture_pitch=float(vegetation_mass_texture_pitch),
            roughness=float(vegetation_mass_roughness),
            density_response=float(vegetation_mass_density_response),
            relief_cap=float(vegetation_mass_relief_cap),
            relief_floor=float(vegetation_mass_relief_floor),
            base_embed=float(vegetation_mass_base_embed),
            edge_softness=float(vegetation_mass_edge_softness),
            organic_smooth=float(vegetation_mass_organic_smooth),
            micro_detail=float(vegetation_mass_micro_detail),
            max_cells=int(vegetation_mass_max_cells),
            status_callback=status_callback,
        )
        if vegetation[0] is None or vegetation[1] is None:
            notify("No se pudo generar la masa vegetal imprimible, omitiendo capa vegetal.", 63, 100.0, "Vegetacion")
            vegetation = (None, None)
    elif vegetation_mode_norm in {"stylized", "estilizada", "maqueta"}:
        notify("Modo vegetacion estilizada para maqueta", 60, 8.0, "Vegetacion")
        vegetation = process_vegetation_stylized(
            points,
            classification,
            ground_points=terrain[0],
            min_mass_area=float(vegetation_style_min_area),
            detail_level=int(vegetation_style_detail),
            include_low_vegetation=bool(vegetation_style_include_low),
            max_clusters_per_class=int(vegetation_style_max_clusters),
            roughness=float(vegetation_style_roughness),
            density_response=float(vegetation_style_density_response),
            texture_pitch=float(vegetation_style_texture_pitch),
            relief_cap=float(vegetation_style_relief_cap),
            min_density=float(vegetation_style_min_density),
            status_callback=status_callback,
        )
        if vegetation[0] is None or vegetation[1] is None:
            notify("No se pudo generar vegetacion estilizada, omitiendo capa vegetal.", 63, 100.0, "Vegetacion")
            vegetation = (None, None)
    else:
        vegetation = process_vegetation_poisson(
            points,
            classification,
            max_points=int(vegetation_max_points),
            poisson_depth=int(vegetation_poisson_depth),
            density_quantile=float(vegetation_density_quantile),
            status_callback=status_callback,
        )
        if vegetation[0] is None or vegetation[1] is None:
            notify("Poisson fallo o no esta disponible, intentando con metaballs...", 62, 40.0, "Vegetacion")
            try:
                vegetation = process_vegetation_as_metaballs(
                    points,
                    classification,
                    sphere_radius=sphere_radius,
                    grid_spacing=metaball_grid,
                    threshold=metaball_threshold,
                    max_grid_dim=int(metaball_max_dim),
                    status_callback=status_callback,
                )
                if vegetation[0] is not None:
                    notify("Vegetacion generada con metaballs.", 63, 100.0, "Vegetacion")
            except ImportError:
                notify("Vegetacion omitida (librerias no disponibles).", 63, 100.0, "Vegetacion")
                vegetation = (None, None)
        else:
            notify("Vegetacion reconstruida con Poisson Surface Reconstruction.", 63, 100.0, "Vegetacion")

    building_source_norm = str(building_source).strip().lower()
    if building_source_norm in {"footprints", "huellas", "externo", "external"}:
        notify("Paso 3 - Reconstruyendo edificios desde huellas", 65, 0.0, "Edificios")
        buildings = process_buildings_from_footprints(
            building_footprints or [],
            ground_points=terrain[0],
            roof_points=points[classification == 6] if classification is not None else None,
            height_mode=building_footprint_height_mode,
            fixed_height=building_footprint_fixed_height,
            fallback_height=building_footprint_fallback_height,
            height_attribute=building_footprint_height_field,
            floors_attribute=building_footprint_floors_field,
            floor_height=building_footprint_floor_height,
            min_height=building_footprint_min_height,
            max_height=building_footprint_max_height,
            min_area=building_footprint_min_area,
            roof_percentile=building_roof_percentile,
            simplify_tolerance=building_footprint_simplify_tolerance,
            max_vertices_per_footprint=building_footprint_max_vertices,
            source_epsg=building_footprint_source_epsg,
            target_epsg=building_footprint_target_epsg,
            footprint_shape_mode=building_footprint_shape_mode,
            footprint_outside_area_threshold=building_footprint_outside_area_threshold,
            roof_plane_tolerance=building_plane_tolerance,
            roof_max_plane_iterations=building_max_plane_iterations,
            roof_density_hint=building_density_hint,
            roof_snap_planes=building_snap_roof_planes,
            roof_model_mode=building_roof_model_mode,
            roof_min_plane_area_m2=building_min_roof_plane_area_m2,
            status_callback=status_callback,
        )
    else:
        notify("Paso 3 - Reconstruyendo edificios como solidos", 65, 0.0, "Edificios")
        buildings = process_buildings_as_prisms(
            points,
            classification,
            cluster_radius=building_cluster_radius,
            min_points=building_min_points,
            roof_percentile=building_roof_percentile,
            base_percentile=building_base_percentile,
            ground_points=terrain[0],
            plane_tolerance=building_plane_tolerance,
            max_plane_iterations=building_max_plane_iterations,
            split_touching_buildings=building_split_touching_clusters,
            orthogonalize_edges=building_orthogonalize_edges,
            snap_roof_planes=building_snap_roof_planes,
            roof_model_mode=building_roof_model_mode,
            min_roof_plane_area_m2=building_min_roof_plane_area_m2,
            density_hint=building_density_hint,
            status_callback=status_callback,
        )
    if buildings[0] is None:
        notify("No se detectaron o generaron edificios.", 67, 100.0, "Edificios")
    else:
        notify(f"Se generaron {len(buildings[0])} vertices de edificios.", 67, 100.0, "Edificios")

    notify("Paso 4 - Fusionando capas en una malla final", 70, 0.0, "Fusion final")
    merged_vertices, merged_faces = merge_meshes(terrain, vegetation, buildings)
    notify("Fusion final completada", 74, 100.0, "Fusion final")
    if bool(return_components):
        return merged_vertices, merged_faces, {
            "terrain": terrain,
            "vegetation": vegetation,
            "buildings": buildings,
        }
    return merged_vertices, merged_faces


def merge_meshes(
    terrain_data: tuple[np.ndarray | None, np.ndarray | None],
    vegetation_data: tuple[np.ndarray | None, np.ndarray | None],
    buildings_data: tuple[np.ndarray | None, np.ndarray | None],
) -> tuple[np.ndarray, np.ndarray]:
    """Merge terrain, vegetation and buildings into one mesh."""
    all_verts: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    vertex_offset = 0

    for data in [terrain_data, vegetation_data, buildings_data]:
        verts, faces = data
        if verts is not None and faces is not None and verts.shape[0] > 0 and faces.shape[0] > 0:
            all_verts.append(verts)
            all_faces.append(faces + vertex_offset)
            vertex_offset += verts.shape[0]

    if not all_verts:
        raise ValueError("No hay geometria valida para crear la malla final.")

    final_verts = np.vstack(all_verts).astype(np.float64, copy=False)
    final_faces = np.vstack(all_faces).astype(np.int32, copy=False)
    return _clean_mesh(final_verts, final_faces)


def simplify_mesh_with_voxel_grid(
    vertices: np.ndarray, faces: np.ndarray, voxel_size: float
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce vertex count by clustering vertices inside voxels."""
    if voxel_size <= 0 or vertices.shape[0] == 0:
        return vertices, faces

    try:
        import open3d as o3d

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        simplified_mesh = mesh.simplify_vertex_clustering(
            voxel_size=voxel_size,
            contraction=o3d.geometry.SimplificationContraction.Average,
        )
        return np.asarray(simplified_mesh.vertices), np.asarray(simplified_mesh.triangles)
    except (ImportError, Exception):
        scale = np.floor(vertices / float(voxel_size)).astype(np.int64)
        unique_keys, inverse = np.unique(scale, axis=0, return_inverse=True)
        counts = np.bincount(inverse)
        simplified_vertices = np.zeros((unique_keys.shape[0], 3), dtype=np.float64)
        np.add.at(simplified_vertices, inverse, vertices)
        simplified_vertices /= counts[:, None]

        simplified_faces = inverse[faces]
        mask = (
            (simplified_faces[:, 0] != simplified_faces[:, 1])
            & (simplified_faces[:, 1] != simplified_faces[:, 2])
            & (simplified_faces[:, 0] != simplified_faces[:, 2])
        )
        simplified_faces = simplified_faces[mask]
        return simplified_vertices, simplified_faces.astype(np.int32, copy=False)


def count_boundary_edges(faces: np.ndarray) -> int:
    """Count open boundary edges (0 means closed with respect to edge manifoldness)."""
    if faces is None or faces.size == 0:
        return 0
    edges = np.vstack(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ]
    ).astype(np.int64, copy=False)
    edges = np.sort(edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int(np.sum(counts == 1))


def repair_open_boundaries(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    max_loops: int = 20_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Attempt to close residual open boundaries by patching boundary loops.

    This is intended as a last-mile repair pass (e.g. after simplification)
    when only a few open edges remain.
    """
    if vertices is None or faces is None or vertices.size == 0 or faces.size == 0:
        return vertices, faces

    clean_vertices, clean_faces = _clean_mesh(vertices, faces)
    if clean_vertices.shape[0] == 0 or clean_faces.shape[0] == 0:
        return clean_vertices, clean_faces

    boundary_edges = _get_boundary_edges(clean_faces)
    if boundary_edges.shape[0] == 0:
        return clean_vertices, clean_faces

    loops = _build_boundary_loops(boundary_edges)
    if not loops:
        return clean_vertices, clean_faces

    loops = loops[: max(0, int(max_loops))]
    patch_vertices: list[np.ndarray] = []
    patch_faces: list[list[int]] = []
    base_index = clean_vertices.shape[0]

    for loop in loops:
        if len(loop) < 3:
            continue
        loop_idx = np.asarray(loop, dtype=np.int64)
        loop_pts = clean_vertices[loop_idx]
        center = np.mean(loop_pts, axis=0)
        center_idx = base_index + len(patch_vertices)
        patch_vertices.append(center)

        # Approximate loop normal to orient fan triangles consistently.
        normal = np.zeros(3, dtype=np.float64)
        for i in range(loop_idx.shape[0]):
            a = loop_pts[i] - center
            b = loop_pts[(i + 1) % loop_idx.shape[0]] - center
            normal += np.cross(a, b)
        normal_norm = np.linalg.norm(normal)
        if normal_norm > 1e-12:
            normal /= normal_norm
        else:
            normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        for i in range(loop_idx.shape[0]):
            a = int(loop_idx[i])
            b = int(loop_idx[(i + 1) % loop_idx.shape[0]])
            face = [a, b, center_idx]
            va, vb, vc = (
                clean_vertices[a],
                clean_vertices[b],
                center,
            )
            tri_normal = np.cross(vb - va, vc - va)
            if float(np.dot(tri_normal, normal)) < 0.0:
                face = [a, center_idx, b]
            patch_faces.append(face)

    if not patch_faces:
        return clean_vertices, clean_faces

    merged_vertices = clean_vertices
    if patch_vertices:
        merged_vertices = np.vstack([clean_vertices, np.asarray(patch_vertices, dtype=np.float64)])
    merged_faces = np.vstack([clean_faces.astype(np.int32), np.asarray(patch_faces, dtype=np.int32)])
    return _clean_mesh(merged_vertices, merged_faces)


def _get_boundary_edges(faces: np.ndarray) -> np.ndarray:
    """Return unique undirected boundary edges (each appears once in the mesh)."""
    if faces is None or faces.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    edges = np.vstack(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ]
    ).astype(np.int64, copy=False)
    edges = np.sort(edges, axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique_edges[counts == 1]
    return boundary.astype(np.int32, copy=False)


def _build_boundary_loops(boundary_edges: np.ndarray) -> list[list[int]]:
    """Build closed loops from undirected boundary edges."""
    if boundary_edges.size == 0:
        return []

    adjacency: dict[int, list[int]] = {}
    edge_set: set[tuple[int, int]] = set()
    for e0, e1 in boundary_edges:
        a = int(e0)
        b = int(e1)
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
        edge_set.add((min(a, b), max(a, b)))

    loops: list[list[int]] = []
    while edge_set:
        a, b = next(iter(edge_set))
        edge_set.remove((a, b))
        loop = [a, b]
        prev = a
        curr = b

        for _ in range(len(boundary_edges) + 2):
            if curr == loop[0]:
                break

            candidates = adjacency.get(curr, [])
            next_node = None
            for cand in candidates:
                if cand == prev:
                    continue
                key = (min(curr, cand), max(curr, cand))
                if key in edge_set:
                    next_node = cand
                    break

            if next_node is None:
                closing_key = (min(curr, loop[0]), max(curr, loop[0]))
                if closing_key in edge_set:
                    edge_set.remove(closing_key)
                    loop.append(loop[0])
                break

            key = (min(curr, next_node), max(curr, next_node))
            edge_set.remove(key)
            loop.append(next_node)
            prev, curr = curr, next_node

        if len(loop) >= 4 and loop[0] == loop[-1]:
            loops.append(loop[:-1])

    return loops


def _point_in_triangle_2d(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    """Check if point p lies inside triangle abc in 2D (including edges)."""
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
    """Triangulate a simple polygon (2D) using ear clipping."""
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
        # Fallback fan triangulation (less robust but avoids empty cap).
        for i in range(1, n - 1):
            triangles.append((0, i, i + 1))
    return triangles


def add_base_to_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    thickness: float,
    hull_edges: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Close open meshes by extruding boundary loops to a flat base plane.

    The function does not duplicate all faces, which avoids creating invalid
    internal geometry when the mesh already contains closed solids.
    """
    if vertices.shape[0] < 3 or faces.shape[0] < 1:
        return vertices, faces

    thickness = float(thickness)
    if thickness <= 0:
        return vertices, faces

    clean_vertices, clean_faces = _clean_mesh(vertices, faces)
    if clean_vertices.shape[0] < 3 or clean_faces.shape[0] < 1:
        return vertices, faces

    boundary_edges = _get_boundary_edges(clean_faces)
    if boundary_edges.shape[0] == 0:
        # Already closed with respect to boundary edges.
        return clean_vertices, clean_faces

    # Keep optional fallback support for explicit hull edges.
    if hull_edges is not None and boundary_edges.shape[0] == 0 and hull_edges.size > 0:
        boundary_edges = np.asarray(hull_edges, dtype=np.int32)

    base_z = float(np.min(clean_vertices[:, 2]) - thickness)
    boundary_vertices = np.unique(boundary_edges)
    bottom_vertices = clean_vertices[boundary_vertices].copy()
    bottom_vertices[:, 2] = base_z

    bottom_index = {
        int(top_idx): int(clean_vertices.shape[0] + local_idx)
        for local_idx, top_idx in enumerate(boundary_vertices)
    }

    combined_vertices = np.vstack([clean_vertices, bottom_vertices]).astype(np.float64, copy=False)
    side_faces: list[list[int]] = []
    for e0, e1 in boundary_edges:
        a = int(e0)
        b = int(e1)
        a_bottom = bottom_index.get(a)
        b_bottom = bottom_index.get(b)
        if a_bottom is None or b_bottom is None:
            continue
        side_faces.append([a, b, b_bottom])
        side_faces.append([a, b_bottom, a_bottom])

    bottom_faces: list[list[int]] = []
    loops = _build_boundary_loops(boundary_edges)
    if not loops and boundary_vertices.shape[0] >= 3:
        try:
            hull = ConvexHull(clean_vertices[boundary_vertices, :2])
            loops = [boundary_vertices[hull.vertices].tolist()]
        except QhullError:
            loops = []

    for loop in loops:
        if len(loop) < 3:
            continue
        poly_xy = clean_vertices[np.asarray(loop, dtype=np.int64), :2]
        tri_local = _triangulate_polygon_ear_clipping(poly_xy)
        for ia, ib, ic in tri_local:
            a = bottom_index[int(loop[ia])]
            b = bottom_index[int(loop[ib])]
            c = bottom_index[int(loop[ic])]
            face = [a, b, c]
            # Force downward normal on base cap.
            va, vb, vc = combined_vertices[face]
            normal_z = np.cross(vb - va, vc - va)[2]
            if normal_z > 0:
                face = [a, c, b]
            bottom_faces.append(face)

    all_extra = []
    if side_faces:
        all_extra.append(np.asarray(side_faces, dtype=np.int32))
    if bottom_faces:
        all_extra.append(np.asarray(bottom_faces, dtype=np.int32))

    if all_extra:
        combined_faces = np.vstack([clean_faces.astype(np.int32, copy=False), *all_extra])
    else:
        combined_faces = clean_faces.astype(np.int32, copy=False)
    combined_vertices, combined_faces = _clean_mesh(combined_vertices, combined_faces)

    # Fallback sealing pass: patch any residual open boundary with a base-center fan.
    remaining_boundary = _get_boundary_edges(combined_faces)
    if remaining_boundary.shape[0] > 0:
        rem_vertices = np.unique(remaining_boundary)
        if rem_vertices.shape[0] >= 2:
            center_xy = np.mean(combined_vertices[rem_vertices, :2], axis=0)
            center_vertex = np.array([[center_xy[0], center_xy[1], base_z]], dtype=np.float64)
            center_idx = combined_vertices.shape[0]
            combined_vertices = np.vstack([combined_vertices, center_vertex])

            patch_faces: list[list[int]] = []
            for e0, e1 in remaining_boundary:
                a = int(e0)
                b = int(e1)
                face = [a, b, center_idx]
                va, vb, vc = combined_vertices[face]
                if np.cross(vb - va, vc - va)[2] > 0:
                    face = [a, center_idx, b]
                patch_faces.append(face)

            if patch_faces:
                combined_faces = np.vstack(
                    [combined_faces.astype(np.int32, copy=False), np.asarray(patch_faces, dtype=np.int32)]
                )
                combined_vertices, combined_faces = _clean_mesh(combined_vertices, combined_faces)

    return combined_vertices, combined_faces
