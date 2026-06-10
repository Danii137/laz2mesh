"""
High-detail terrain reconstruction based on progressive TIN densification.
"""

import numpy as np
from scipy.spatial import Delaunay, QhullError

from src.utils.geometry import remove_statistical_outliers


def _emit_status(status_callback, message: str, pct: int | None = None, step_pct: float | None = None, step_label: str | None = None) -> None:
    if status_callback is None:
        return
    try:
        status_callback(message, pct, step_pct, step_label)
    except TypeError:
        status_callback(message, pct)


def process_terrain_high_detail(
    points: np.ndarray,
    classification: np.ndarray | None,
    resolution: float = 0.5,
    *,
    return_number: np.ndarray | None = None,
    num_returns: np.ndarray | None = None,
    status_callback=None,
    max_iterations: int = 7,
    max_vertical_error: float | None = None,
    max_slope_deg: float | None = 40.0,
    distance_threshold: float | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Run a terrain-only progressive TIN workflow."""

    def notify(message: str, pct: int | None = None, step_pct: float | None = None) -> None:
        _emit_status(status_callback, message, pct, step_pct, "Terreno")

    if points.size == 0:
        return None, None

    notify("Terreno: filtrando clase de terreno", 55, 5.0)
    if classification is None:
        mask_ground = np.ones(points.shape[0], dtype=bool)
    else:
        mask_ground = classification == 2
        if not mask_ground.any():
            return None, None

    ground_points = points[mask_ground].astype(np.float64, copy=True)
    if ground_points.shape[0] < 3:
        return None, None

    ground_returns = None
    ground_total_returns = None
    if (
        return_number is not None
        and num_returns is not None
        and len(return_number) == points.shape[0]
        and len(num_returns) == points.shape[0]
    ):
        ground_returns = return_number[mask_ground]
        ground_total_returns = num_returns[mask_ground]
        last_mask = ground_returns == ground_total_returns
        if last_mask.any():
            ground_points = ground_points[last_mask]
            ground_returns = ground_returns[last_mask]
            ground_total_returns = ground_total_returns[last_mask]
            notify("Terreno: usando retornos finales", 55, 15.0)
        else:
            notify("Terreno: no hay retornos finales claros, usando todos", 55, 15.0)
    else:
        notify("Terreno: sin info de retornos, usando todos", 55, 15.0)

    if ground_points.shape[0] < 3:
        return None, None

    neighbors = min(40, max(10, ground_points.shape[0] // 200))
    notify("Terreno: eliminando outliers estadisticos", 56, 25.0)
    ground_points, _, mask_stat = remove_statistical_outliers(
        ground_points, None, neighbors=neighbors, std_ratio=2.5, return_mask=True
    )
    if ground_returns is not None:
        ground_returns = ground_returns[mask_stat]
    if ground_total_returns is not None:
        ground_total_returns = ground_total_returns[mask_stat]

    if ground_points.shape[0] < 3:
        return None, None

    if ground_points.shape[0] >= 5:
        z_mean = float(np.mean(ground_points[:, 2]))
        z_std = float(np.std(ground_points[:, 2]))
        if z_std > 0:
            mask_z = np.abs(ground_points[:, 2] - z_mean) <= 3.0 * z_std
            if not mask_z.all():
                ground_points = ground_points[mask_z]
                if ground_returns is not None:
                    ground_returns = ground_returns[mask_z]
                if ground_total_returns is not None:
                    ground_total_returns = ground_total_returns[mask_z]
                notify("Terreno: eliminando elevaciones extremas", 56, 33.0)

    if ground_points.shape[0] < 3:
        return None, None

    notify("Terreno: seleccionando seeds TIN", 56, 40.0)
    grid_size = max(float(resolution), 0.5)
    if not np.isfinite(grid_size) or grid_size <= 0:
        grid_size = 1.0

    x_min = float(np.min(ground_points[:, 0]))
    x_max = float(np.max(ground_points[:, 0]))
    y_min = float(np.min(ground_points[:, 1]))
    y_max = float(np.max(ground_points[:, 1]))
    if np.isclose(x_min, x_max) or np.isclose(y_min, y_max):
        return None, None

    seed_index_map: dict[tuple[int, int], int] = {}
    for idx, pt in enumerate(ground_points):
        cell_x = int(np.floor((pt[0] - x_min) / grid_size))
        cell_y = int(np.floor((pt[1] - y_min) / grid_size))
        key = (cell_x, cell_y)
        if key not in seed_index_map or pt[2] < ground_points[seed_index_map[key], 2]:
            seed_index_map[key] = idx

    if len(seed_index_map) < 3:
        return None, None

    seed_indices = np.fromiter(seed_index_map.values(), dtype=np.int64)
    seed_points = ground_points[seed_indices]

    remaining_mask = np.ones(ground_points.shape[0], dtype=bool)
    remaining_mask[seed_indices] = False
    remaining_points = ground_points[remaining_mask]
    classified_points = seed_points.copy()

    if distance_threshold is None:
        distance_threshold = max(0.5, grid_size * 1.25)
    if max_vertical_error is None:
        max_vertical_error = max(0.2, grid_size * 0.35)
    else:
        max_vertical_error = max(float(max_vertical_error), max(0.15, grid_size * 0.25))
    slope_limit = float(max_slope_deg) if max_slope_deg is not None else None
    max_iterations = max(1, int(max_iterations))

    notify("Terreno: densificando TIN progresivo", 57, 45.0)
    for iteration in range(max_iterations):
        if remaining_points.shape[0] == 0 or classified_points.shape[0] < 3:
            break
        try:
            tri = Delaunay(classified_points[:, :2])
        except QhullError:
            break

        simplices_idx = tri.find_simplex(remaining_points[:, :2])
        valid_mask = simplices_idx >= 0
        if not np.any(valid_mask):
            break

        candidate_indices = np.where(valid_mask)[0]
        triangles = tri.simplices[simplices_idx[valid_mask]]
        triangle_pts = classified_points[triangles]

        v1 = triangle_pts[:, 1, :] - triangle_pts[:, 0, :]
        v2 = triangle_pts[:, 2, :] - triangle_pts[:, 0, :]
        normals = np.cross(v1, v2)
        norms = np.linalg.norm(normals, axis=1)
        non_degenerate = norms > 1e-12
        if not np.any(non_degenerate):
            break

        normals = normals[non_degenerate] / norms[non_degenerate][:, None]
        triangle_pts = triangle_pts[non_degenerate]
        candidate_indices = candidate_indices[non_degenerate]
        candidate_points = remaining_points[candidate_indices]

        base_vectors = candidate_points - triangle_pts[:, 0, :]
        distances = np.abs(np.einsum("ij,ij->i", base_vectors, normals))
        distance_ok = distances <= distance_threshold
        if not np.any(distance_ok):
            iter_step = 45.0 + ((iteration + 1) / max_iterations) * 40.0
            notify(f"Terreno: iteracion {iteration + 1}/{max_iterations}", 58, iter_step)
            continue

        candidate_indices = candidate_indices[distance_ok]
        candidate_points = candidate_points[distance_ok]
        triangle_pts = triangle_pts[distance_ok]
        normals = normals[distance_ok]

        plane_pts = triangle_pts[:, 0, :]
        normals_z = normals[:, 2]
        vertical_offsets = np.full(candidate_points.shape[0], np.nan, dtype=np.float64)
        valid = np.abs(normals_z) > 1e-8
        if np.any(valid):
            rel_xy = candidate_points[valid, :2] - plane_pts[valid, :2]
            plane_term = (
                -normals[valid, 0] * rel_xy[:, 0] - normals[valid, 1] * rel_xy[:, 1]
            ) / normals_z[valid]
            plane_heights = plane_pts[valid, 2] + plane_term
            vertical_offsets[valid] = candidate_points[valid, 2] - plane_heights

        horizontal = np.linalg.norm(candidate_points[:, :2] - plane_pts[:, :2], axis=1)
        horizontal = np.maximum(horizontal, 1e-6)
        vertical_abs = np.abs(vertical_offsets)
        height_ok = vertical_abs <= max_vertical_error
        slope_ok = np.ones_like(height_ok, dtype=bool)
        if slope_limit is not None:
            slope_angles = np.degrees(np.arctan2(vertical_abs, horizontal))
            slope_ok = slope_angles <= slope_limit
        finite_mask = np.isfinite(vertical_offsets)
        offset_ok = ~finite_mask | (height_ok | slope_ok)
        if not np.any(offset_ok):
            iter_step = 45.0 + ((iteration + 1) / max_iterations) * 40.0
            notify(f"Terreno: iteracion {iteration + 1}/{max_iterations}", 58, iter_step)
            continue

        selected_indices = candidate_indices[offset_ok]
        new_points = candidate_points[offset_ok]
        classified_points = np.vstack([classified_points, new_points])

        keep_mask = np.ones(remaining_points.shape[0], dtype=bool)
        keep_mask[selected_indices] = False
        remaining_points = remaining_points[keep_mask]

        iter_step = 45.0 + ((iteration + 1) / max_iterations) * 40.0
        notify(
            f"Terreno: iteracion {iteration + 1}/{max_iterations} (restantes {remaining_points.shape[0]:,})",
            58,
            iter_step,
        )

    if classified_points.shape[0] < 3:
        return None, None

    unique_points = np.unique(classified_points, axis=0)
    if unique_points.shape[0] < 3:
        return None, None

    min_expected = max(20, int(0.1 * ground_points.shape[0]))
    if unique_points.shape[0] < min_expected and ground_points.shape[0] >= 3:
        notify("Terreno: densificacion limitada, usando todos los puntos filtrados", 58, 92.0)
        unique_points = ground_points

    notify("Terreno: triangulacion final", 59, 96.0)
    try:
        tri_final = Delaunay(unique_points[:, :2])
    except QhullError:
        return None, None

    notify("Terreno completado", 59, 100.0)
    return unique_points, tri_final.simplices.astype(np.int32)
