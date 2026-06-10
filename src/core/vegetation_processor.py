"""
Vegetation reconstruction helpers.

Available strategies:
1) Poisson reconstruction (if open3d is available)
2) Metaballs + marching cubes fallback
3) Stylized vegetation for physical scale models
"""

import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    binary_opening,
    distance_transform_edt,
    gaussian_filter,
    label,
)
from scipy.spatial import ConvexHull, QhullError, cKDTree

try:
    import open3d as o3d
except ImportError:
    o3d = None

try:
    from skimage import measure
except ImportError:
    measure = None


def _emit_status(status_callback, message: str, pct: int | None = None, step_pct: float | None = None, step_label: str | None = None) -> None:
    if status_callback is None:
        return
    try:
        status_callback(message, pct, step_pct, step_label)
    except TypeError:
        status_callback(message, pct)


def _mesh_full_ellipsoid(
    center: np.ndarray,
    rx: float,
    ry: float,
    rz: float,
    lon_segments: int,
    lat_segments: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a closed ellipsoid mesh."""
    lon_segments = max(int(lon_segments), 8)
    lat_segments = max(int(lat_segments), 6)
    rx = max(float(rx), 0.05)
    ry = max(float(ry), 0.05)
    rz = max(float(rz), 0.05)

    verts: list[list[float]] = []
    faces: list[list[int]] = []

    # Poles
    south_idx = 0
    verts.append([center[0], center[1], center[2] - rz])
    north_idx = 1
    verts.append([center[0], center[1], center[2] + rz])

    ring_indices: list[list[int]] = []
    # Rings excluding poles.
    for i in range(1, lat_segments):
        phi = -0.5 * np.pi + (i / lat_segments) * np.pi
        ring_r = np.cos(phi)
        z = center[2] + rz * np.sin(phi)
        ring: list[int] = []
        for j in range(lon_segments):
            theta = (j / lon_segments) * 2.0 * np.pi
            x = center[0] + rx * ring_r * np.cos(theta)
            y = center[1] + ry * ring_r * np.sin(theta)
            ring.append(len(verts))
            verts.append([x, y, z])
        ring_indices.append(ring)

    if not ring_indices:
        return np.asarray(verts, dtype=np.float64), np.empty((0, 3), dtype=np.int32)

    first_ring = ring_indices[0]
    last_ring = ring_indices[-1]

    # South cap
    for j in range(lon_segments):
        a = first_ring[j]
        b = first_ring[(j + 1) % lon_segments]
        faces.append([south_idx, b, a])

    # Middle strips
    for ridx in range(len(ring_indices) - 1):
        lower = ring_indices[ridx]
        upper = ring_indices[ridx + 1]
        for j in range(lon_segments):
            a = lower[j]
            b = lower[(j + 1) % lon_segments]
            c = upper[j]
            d = upper[(j + 1) % lon_segments]
            faces.append([a, b, d])
            faces.append([a, d, c])

    # North cap
    for j in range(lon_segments):
        a = last_ring[j]
        b = last_ring[(j + 1) % lon_segments]
        faces.append([north_idx, a, b])

    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def _mesh_dome(
    center_xy: np.ndarray,
    base_z: float,
    rx: float,
    ry: float,
    rz: float,
    lon_segments: int,
    lat_segments: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a closed dome (hemisphere-like) with a bottom cap."""
    lon_segments = max(int(lon_segments), 8)
    lat_segments = max(int(lat_segments), 4)
    rx = max(float(rx), 0.05)
    ry = max(float(ry), 0.05)
    rz = max(float(rz), 0.05)

    verts: list[list[float]] = []
    faces: list[list[int]] = []

    ring_indices: list[list[int]] = []
    # Rings from base to just before the top.
    for i in range(lat_segments):
        phi = (i / lat_segments) * (0.5 * np.pi)
        ring_r = np.cos(phi)
        z = base_z + rz * np.sin(phi)
        ring: list[int] = []
        for j in range(lon_segments):
            theta = (j / lon_segments) * 2.0 * np.pi
            x = center_xy[0] + rx * ring_r * np.cos(theta)
            y = center_xy[1] + ry * ring_r * np.sin(theta)
            ring.append(len(verts))
            verts.append([x, y, z])
        ring_indices.append(ring)

    top_idx = len(verts)
    verts.append([center_xy[0], center_xy[1], base_z + rz])
    bottom_center_idx = len(verts)
    verts.append([center_xy[0], center_xy[1], base_z])

    # Side strips
    for ridx in range(len(ring_indices) - 1):
        lower = ring_indices[ridx]
        upper = ring_indices[ridx + 1]
        for j in range(lon_segments):
            a = lower[j]
            b = lower[(j + 1) % lon_segments]
            c = upper[j]
            d = upper[(j + 1) % lon_segments]
            faces.append([a, b, d])
            faces.append([a, d, c])

    # Top cap
    last_ring = ring_indices[-1]
    for j in range(lon_segments):
        a = last_ring[j]
        b = last_ring[(j + 1) % lon_segments]
        faces.append([top_idx, a, b])

    # Bottom cap (downward)
    base_ring = ring_indices[0]
    for j in range(lon_segments):
        a = base_ring[j]
        b = base_ring[(j + 1) % lon_segments]
        faces.append([bottom_center_idx, b, a])

    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def _mesh_frustum(
    center_xy: np.ndarray,
    z0: float,
    z1: float,
    r0: float,
    r1: float,
    segments: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a closed frustum (trunk-like)."""
    segments = max(int(segments), 8)
    r0 = max(float(r0), 0.03)
    r1 = max(float(r1), 0.02)
    if z1 <= z0 + 1e-4:
        z1 = z0 + 1e-4

    verts: list[list[float]] = []
    faces: list[list[int]] = []

    lower_idx: list[int] = []
    upper_idx: list[int] = []
    for j in range(segments):
        theta = (j / segments) * 2.0 * np.pi
        ct = np.cos(theta)
        st = np.sin(theta)
        lower_idx.append(len(verts))
        verts.append([center_xy[0] + r0 * ct, center_xy[1] + r0 * st, z0])
        upper_idx.append(len(verts))
        verts.append([center_xy[0] + r1 * ct, center_xy[1] + r1 * st, z1])

    top_center_idx = len(verts)
    verts.append([center_xy[0], center_xy[1], z1])
    bottom_center_idx = len(verts)
    verts.append([center_xy[0], center_xy[1], z0])

    for j in range(segments):
        jn = (j + 1) % segments
        a = lower_idx[j]
        b = lower_idx[jn]
        c = upper_idx[j]
        d = upper_idx[jn]
        faces.append([a, b, d])
        faces.append([a, d, c])

        # Top cap
        faces.append([top_center_idx, c, d])
        # Bottom cap (downward)
        faces.append([bottom_center_idx, b, a])

    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def _mesh_rugged_mound(
    center_xy: np.ndarray,
    base_z: float,
    rx: float,
    ry: float,
    height: float,
    *,
    lon_segments: int,
    radial_segments: int,
    roughness: float,
    texture_pitch: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a closed, low-profile and rugged mound.

    This is intended for stylized "forest mass" representation:
    more like rough moss/texture than individual trees.
    """
    lon_segments = max(int(lon_segments), 10)
    radial_segments = max(int(radial_segments), 3)
    rx = max(float(rx), 0.15)
    ry = max(float(ry), 0.15)
    height = max(float(height), 0.06)
    roughness = float(np.clip(roughness, 0.0, 1.4))
    texture_pitch = max(float(texture_pitch), 0.08)

    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    phase1 = float(rng.uniform(0.0, 2.0 * np.pi))
    phase2 = float(rng.uniform(0.0, 2.0 * np.pi))
    phase3 = float(rng.uniform(0.0, 2.0 * np.pi))

    avg_radius = max(0.5 * (rx + ry), 0.2)
    waves_main = int(np.clip((2.0 * np.pi * avg_radius) / texture_pitch, 2, 26))
    waves_micro = max(3, int(waves_main * 1.8))
    boundary_jitter = 0.05 + roughness * 0.12
    top_gain = 1.0 + 0.08 * roughness * np.sin(phase1)

    verts: list[list[float]] = []
    faces: list[list[int]] = []

    top_center_idx = 0
    verts.append([center_xy[0], center_xy[1], base_z + height * top_gain])

    rings: list[list[int]] = []
    for ridx in range(1, radial_segments + 1):
        r = ridx / radial_segments
        # Flattened mound profile with stronger curvature near edges.
        profile = float(np.clip(1.0 - r**1.28, 0.0, 1.0) ** 1.95)
        env = 0.16 + 0.84 * profile

        ring: list[int] = []
        for j in range(lon_segments):
            theta = (j / lon_segments) * 2.0 * np.pi
            edge_noise = (
                0.65 * np.sin(theta * waves_main + phase1 + r * 1.1)
                + 0.35 * np.sin(theta * (waves_main * 0.53) + phase2 - r * 2.0)
            )
            scale_xy = max(1.0 + boundary_jitter * edge_noise * r, 0.65)
            local_rx = rx * scale_xy
            local_ry = ry * scale_xy

            x = center_xy[0] + local_rx * r * np.cos(theta)
            y = center_xy[1] + local_ry * r * np.sin(theta)

            rough_wave = (
                0.55 * np.sin(theta * waves_micro + phase2 + r * 2.5)
                + 0.30 * np.cos(theta * (waves_main * 0.8) - r * 3.2 + phase3)
                + 0.15 * np.sin((r * waves_micro * 1.2) + phase1)
            )
            z = base_z + height * (profile * top_gain + 0.24 * roughness * env * rough_wave)
            z = max(z, base_z + min(height * 0.08, 0.03))

            ring.append(len(verts))
            verts.append([x, y, z])
        rings.append(ring)

    # Top fan
    first_ring = rings[0]
    for j in range(lon_segments):
        a = first_ring[j]
        b = first_ring[(j + 1) % lon_segments]
        faces.append([top_center_idx, a, b])

    # Top strips (center -> edge)
    for ridx in range(len(rings) - 1):
        inner = rings[ridx]
        outer = rings[ridx + 1]
        for j in range(lon_segments):
            jn = (j + 1) % lon_segments
            a = inner[j]
            b = inner[jn]
            c = outer[j]
            d = outer[jn]
            faces.append([a, c, d])
            faces.append([a, d, b])

    outer_top = rings[-1]
    bottom_z = base_z - float(np.clip(height * 0.18, 0.03, 0.35))
    bottom_ring: list[int] = []
    for idx in outer_top:
        vx, vy = verts[idx][0], verts[idx][1]
        bottom_ring.append(len(verts))
        verts.append([vx, vy, bottom_z])
    bottom_center_idx = len(verts)
    verts.append([center_xy[0], center_xy[1], bottom_z])

    # Side wall
    for j in range(lon_segments):
        jn = (j + 1) % lon_segments
        t0 = outer_top[j]
        t1 = outer_top[jn]
        b0 = bottom_ring[j]
        b1 = bottom_ring[jn]
        faces.append([t0, t1, b1])
        faces.append([t0, b1, b0])

    # Bottom cap (downward)
    for j in range(lon_segments):
        jn = (j + 1) % lon_segments
        b0 = bottom_ring[j]
        b1 = bottom_ring[jn]
        faces.append([bottom_center_idx, b1, b0])

    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def _merge_mesh_parts(parts: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not parts:
        return None, None
    all_v: list[np.ndarray] = []
    all_f: list[np.ndarray] = []
    offset = 0
    for v, f in parts:
        if v is None or f is None or v.shape[0] == 0 or f.shape[0] == 0:
            continue
        all_v.append(v)
        all_f.append(f + offset)
        offset += v.shape[0]
    if not all_v:
        return None, None
    return (
        np.vstack(all_v).astype(np.float64, copy=False),
        np.vstack(all_f).astype(np.int32, copy=False),
    )


def _downsample_points_xy(points: np.ndarray, cell_size: float) -> np.ndarray:
    """Grid downsample in XY while preserving max Z per cell."""
    if points.shape[0] == 0:
        return points
    cell_size = max(float(cell_size), 0.05)
    min_x = float(np.min(points[:, 0]))
    min_y = float(np.min(points[:, 1]))
    ix = np.floor((points[:, 0] - min_x) / cell_size).astype(np.int64)
    iy = np.floor((points[:, 1] - min_y) / cell_size).astype(np.int64)
    keys = np.column_stack([ix, iy])
    _, inv = np.unique(keys, axis=0, return_inverse=True)

    counts = np.bincount(inv)
    sx = np.bincount(inv, weights=points[:, 0])
    sy = np.bincount(inv, weights=points[:, 1])
    zmax = np.full(counts.shape[0], -np.inf, dtype=np.float64)
    np.maximum.at(zmax, inv, points[:, 2])
    out = np.column_stack([sx / np.maximum(counts, 1), sy / np.maximum(counts, 1), zmax])
    return out.astype(np.float64, copy=False)


def _cluster_xy_components(
    points: np.ndarray,
    radius_xy: float,
    min_points: int,
    max_clusters: int,
) -> list[np.ndarray]:
    if points.shape[0] == 0:
        return []
    tree = cKDTree(points[:, :2])
    visited = np.zeros(points.shape[0], dtype=bool)
    clusters: list[np.ndarray] = []

    for seed in range(points.shape[0]):
        if visited[seed]:
            continue
        queue = [seed]
        comp: list[int] = []
        while queue:
            idx = queue.pop()
            if visited[idx]:
                continue
            visited[idx] = True
            comp.append(idx)
            neighbors = tree.query_ball_point(points[idx, :2], r=radius_xy)
            for nb in neighbors:
                if not visited[nb]:
                    queue.append(nb)
        if len(comp) >= min_points:
            clusters.append(points[np.asarray(comp, dtype=np.int64)])
            if len(clusters) >= max_clusters:
                break
    return clusters


def _cluster_area_xy(cluster_xy: np.ndarray) -> float:
    if cluster_xy.shape[0] < 3:
        return 0.0
    try:
        hull = ConvexHull(cluster_xy)
        return float(hull.volume)  # area in 2D
    except (QhullError, ValueError):
        return 0.0


def _disk_structure(radius_cells: int) -> np.ndarray:
    radius_cells = max(int(radius_cells), 0)
    if radius_cells <= 0:
        return np.ones((1, 1), dtype=bool)
    yy, xx = np.ogrid[-radius_cells : radius_cells + 1, -radius_cells : radius_cells + 1]
    return (xx * xx + yy * yy) <= (radius_cells * radius_cells)


def _build_relief_mask_mesh(
    mask: np.ndarray,
    top_cells: np.ndarray,
    bottom_cells: np.ndarray,
    *,
    min_x: float,
    min_y: float,
    cell_size: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Build a closed mesh from a binary cell mask with per-cell top/bottom heights."""
    if mask.size == 0 or not np.any(mask):
        return None, None

    ny, nx = mask.shape
    iy_idx, ix_idx = np.nonzero(mask)
    if iy_idx.size == 0:
        return None, None

    top_vals = top_cells[iy_idx, ix_idx].astype(np.float64, copy=False)
    bottom_vals = bottom_cells[iy_idx, ix_idx].astype(np.float64, copy=False)

    corner_top_sum = np.zeros((ny + 1, nx + 1), dtype=np.float64)
    corner_bottom_sum = np.zeros((ny + 1, nx + 1), dtype=np.float64)
    corner_count = np.zeros((ny + 1, nx + 1), dtype=np.float64)

    for dy, dx in ((0, 0), (0, 1), (1, 1), (1, 0)):
        np.add.at(corner_top_sum, (iy_idx + dy, ix_idx + dx), top_vals)
        np.add.at(corner_bottom_sum, (iy_idx + dy, ix_idx + dx), bottom_vals)
        np.add.at(corner_count, (iy_idx + dy, ix_idx + dx), 1.0)

    corner_mask = corner_count > 0.0
    if not np.any(corner_mask):
        return None, None

    top_corner = corner_top_sum / np.maximum(corner_count, 1.0)
    bottom_corner = corner_bottom_sum / np.maximum(corner_count, 1.0)

    cy, cx = np.nonzero(corner_mask)
    n_corners = cy.shape[0]
    corner_x = min_x + cx.astype(np.float64) * float(cell_size)
    corner_y = min_y + cy.astype(np.float64) * float(cell_size)

    top_vertices = np.column_stack([corner_x, corner_y, top_corner[corner_mask]])
    bottom_vertices = np.column_stack([corner_x, corner_y, bottom_corner[corner_mask]])
    vertices = np.vstack([top_vertices, bottom_vertices]).astype(np.float64, copy=False)

    top_id = np.full((ny + 1, nx + 1), -1, dtype=np.int64)
    bottom_id = np.full((ny + 1, nx + 1), -1, dtype=np.int64)
    ids = np.arange(n_corners, dtype=np.int64)
    top_id[cy, cx] = ids
    bottom_id[cy, cx] = ids + n_corners

    faces: list[list[int]] = []

    for iy, ix in zip(iy_idx.tolist(), ix_idx.tolist()):
        c00t = int(top_id[iy, ix])
        c10t = int(top_id[iy, ix + 1])
        c11t = int(top_id[iy + 1, ix + 1])
        c01t = int(top_id[iy + 1, ix])
        c00b = int(bottom_id[iy, ix])
        c10b = int(bottom_id[iy, ix + 1])
        c11b = int(bottom_id[iy + 1, ix + 1])
        c01b = int(bottom_id[iy + 1, ix])

        if min(c00t, c10t, c11t, c01t, c00b, c10b, c11b, c01b) < 0:
            continue

        # Top and bottom faces.
        faces.append([c00t, c10t, c11t])
        faces.append([c00t, c11t, c01t])
        faces.append([c00b, c11b, c10b])
        faces.append([c00b, c01b, c11b])

        # Left side (-X).
        if ix == 0 or not mask[iy, ix - 1]:
            faces.append([c00t, c01t, c01b])
            faces.append([c00t, c01b, c00b])

        # Right side (+X).
        if ix == nx - 1 or not mask[iy, ix + 1]:
            faces.append([c10t, c10b, c11b])
            faces.append([c10t, c11b, c11t])

        # Bottom side (-Y).
        if iy == 0 or not mask[iy - 1, ix]:
            faces.append([c00t, c00b, c10b])
            faces.append([c00t, c10b, c10t])

        # Top side (+Y).
        if iy == ny - 1 or not mask[iy + 1, ix]:
            faces.append([c01t, c11t, c11b])
            faces.append([c01t, c11b, c01b])

    if not faces:
        return None, None

    return vertices, np.asarray(faces, dtype=np.int32)


def process_vegetation_printable_mass(
    points: np.ndarray,
    classification: np.ndarray,
    *,
    ground_points: np.ndarray | None = None,
    cell_size: float = 0.8,
    include_low_vegetation: bool = False,
    min_height: float = 0.9,
    min_density: float = 0.03,
    min_ratio: float = 0.16,
    min_patch_area: float = 12.0,
    close_radius: float = 1.2,
    open_radius: float = 0.6,
    texture_pitch: float = 1.1,
    roughness: float = 0.58,
    density_response: float = 1.0,
    relief_cap: float = 2.6,
    relief_floor: float = 0.45,
    base_embed: float = 0.18,
    edge_softness: float = 1.0,
    organic_smooth: float = 0.9,
    micro_detail: float = 0.85,
    max_cells: int = 450_000,
    status_callback=None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Generate printable vegetation as low relief "forest masses".

    The output is intentionally compact and rugged (moss-like) to print well
    in scaled models, avoiding isolated giant tree-like blobs.
    """
    if points is None or classification is None or points.shape[0] < 10:
        return None, None

    cell_size = max(float(cell_size), 0.2)
    min_height = max(float(min_height), 0.1)
    min_density = max(float(min_density), 0.001)
    min_ratio = float(np.clip(min_ratio, 0.01, 0.99))
    min_patch_area = max(float(min_patch_area), 0.5)
    close_radius = max(float(close_radius), 0.0)
    open_radius = max(float(open_radius), 0.0)
    texture_pitch = max(float(texture_pitch), 0.25)
    roughness = float(np.clip(roughness, 0.0, 1.0))
    density_response = float(np.clip(density_response, 0.0, 2.0))
    relief_cap = max(float(relief_cap), 0.12)
    relief_floor = max(float(relief_floor), 0.06)
    base_embed = max(float(base_embed), 0.03)
    edge_softness = max(float(edge_softness), 0.05)
    organic_smooth = max(float(organic_smooth), 0.0)
    micro_detail = float(np.clip(micro_detail, 0.0, 1.8))
    max_cells = max(int(max_cells), 50_000)

    _emit_status(
        status_callback,
        "Vegetacion masa imprimible: preparando datos",
        60,
        5.0,
        "Vegetacion",
    )

    veg_classes = [4, 5]
    if include_low_vegetation:
        veg_classes.append(3)
    mask_veg = np.isin(classification, veg_classes)
    if not np.any(mask_veg):
        return None, None

    x_min = float(np.min(points[:, 0]))
    x_max = float(np.max(points[:, 0]))
    y_min = float(np.min(points[:, 1]))
    y_max = float(np.max(points[:, 1]))
    width = max(x_max - x_min, cell_size)
    height = max(y_max - y_min, cell_size)

    nx = max(int(np.ceil(width / cell_size)), 1)
    ny = max(int(np.ceil(height / cell_size)), 1)
    total_cells = nx * ny
    if total_cells > max_cells:
        scale = np.sqrt(total_cells / float(max_cells))
        cell_size *= scale
        nx = max(int(np.ceil(width / cell_size)), 1)
        ny = max(int(np.ceil(height / cell_size)), 1)
        _emit_status(
            status_callback,
            f"Vegetacion masa imprimible: ajustando celda a {cell_size:.2f} m para estabilidad",
            60,
            12.0,
            "Vegetacion",
        )

    if ground_points is not None and ground_points.shape[0] >= 3:
        ground_ref = ground_points.astype(np.float64, copy=False)
    else:
        ground_mask = classification == 2
        if np.any(ground_mask):
            ground_ref = points[ground_mask].astype(np.float64, copy=False)
        else:
            ground_ref = points.astype(np.float64, copy=False)
    if ground_ref.shape[0] < 3:
        return None, None
    ground_tree = cKDTree(ground_ref[:, :2])

    _emit_status(
        status_callback,
        "Vegetacion masa imprimible: agregando celdas",
        60,
        22.0,
        "Vegetacion",
    )

    cell_area = max(cell_size * cell_size, 1e-6)
    ix_all = np.clip(np.floor((points[:, 0] - x_min) / cell_size).astype(np.int64), 0, nx - 1)
    iy_all = np.clip(np.floor((points[:, 1] - y_min) / cell_size).astype(np.int64), 0, ny - 1)
    key_all = iy_all * np.int64(nx) + ix_all
    all_keys, all_counts = np.unique(key_all, return_counts=True)

    veg_points = points[mask_veg].astype(np.float64, copy=False)
    ix_veg = np.clip(np.floor((veg_points[:, 0] - x_min) / cell_size).astype(np.int64), 0, nx - 1)
    iy_veg = np.clip(np.floor((veg_points[:, 1] - y_min) / cell_size).astype(np.int64), 0, ny - 1)
    key_veg = iy_veg * np.int64(nx) + ix_veg
    if key_veg.size == 0:
        return None, None

    order = np.argsort(key_veg, kind="mergesort")
    key_sorted = key_veg[order]
    z_sorted = veg_points[:, 2][order]
    split_idx = np.concatenate(
        ([0], np.flatnonzero(np.diff(key_sorted)) + 1, [key_sorted.shape[0]])
    )
    group_keys = key_sorted[split_idx[:-1]]
    group_counts = (split_idx[1:] - split_idx[:-1]).astype(np.int32)

    group_z95 = np.empty(group_keys.shape[0], dtype=np.float64)
    group_z50 = np.empty(group_keys.shape[0], dtype=np.float64)
    for gid in range(group_keys.shape[0]):
        start = int(split_idx[gid])
        end = int(split_idx[gid + 1])
        chunk = z_sorted[start:end]
        group_z95[gid] = float(np.percentile(chunk, 95.0))
        group_z50[gid] = float(np.percentile(chunk, 50.0))

    all_pos = np.searchsorted(all_keys, group_keys)
    safe_pos = np.clip(all_pos, 0, all_keys.shape[0] - 1)
    all_match = (all_pos < all_keys.shape[0]) & (all_keys[safe_pos] == group_keys)
    group_all_counts = np.where(all_match, all_counts[safe_pos], group_counts).astype(np.float64)

    density = group_counts.astype(np.float64) / cell_area
    ratio = group_counts.astype(np.float64) / np.maximum(group_all_counts, 1.0)
    pre_mask = (density >= min_density) & (ratio >= min_ratio)
    if not np.any(pre_mask):
        return None, None

    group_keys = group_keys[pre_mask]
    group_counts = group_counts[pre_mask]
    group_z95 = group_z95[pre_mask]
    group_z50 = group_z50[pre_mask]
    density = density[pre_mask]

    ix_group = (group_keys % np.int64(nx)).astype(np.int64)
    iy_group = (group_keys // np.int64(nx)).astype(np.int64)
    centers = np.column_stack(
        [
            x_min + (ix_group.astype(np.float64) + 0.5) * cell_size,
            y_min + (iy_group.astype(np.float64) + 0.5) * cell_size,
        ]
    )

    _emit_status(
        status_callback,
        "Vegetacion masa imprimible: estimando altura sobre terreno",
        61,
        36.0,
        "Vegetacion",
    )

    _, gidx = ground_tree.query(centers, k=1)
    gidx = np.asarray(gidx, dtype=np.int64)
    ground_z = ground_ref[gidx, 2].astype(np.float64, copy=False)

    h95 = group_z95 - ground_z
    h50 = group_z50 - ground_z
    canopy_mask = (h95 >= min_height) | (h50 >= min_height * 0.65)
    if not np.any(canopy_mask):
        return None, None

    ix_can = ix_group[canopy_mask]
    iy_can = iy_group[canopy_mask]
    ground_can = ground_z[canopy_mask]
    h95_can = h95[canopy_mask]
    density_can = density[canopy_mask]

    density_norm = np.clip(
        np.log1p(density_can / max(min_density, 1e-6)) / np.log1p(6.0),
        0.15,
        2.2,
    )
    relief = relief_floor + np.maximum(h95_can - min_height, 0.0) * 0.55
    relief *= 0.78 + 0.28 * density_norm * max(density_response, 0.0)
    relief = np.clip(relief, relief_floor * 0.75, relief_cap)

    candidate_mask = np.zeros((ny, nx), dtype=bool)
    ground_seed = np.zeros((ny, nx), dtype=np.float64)
    relief_seed = np.zeros((ny, nx), dtype=np.float64)
    density_seed = np.zeros((ny, nx), dtype=np.float64)

    candidate_mask[iy_can, ix_can] = True
    ground_seed[iy_can, ix_can] = ground_can
    relief_seed[iy_can, ix_can] = relief
    density_seed[iy_can, ix_can] = density_can

    _emit_status(
        status_callback,
        "Vegetacion masa imprimible: limpiando mascara",
        62,
        48.0,
        "Vegetacion",
    )

    mask = candidate_mask.copy()
    close_cells = int(np.round(close_radius / max(cell_size, 1e-6)))
    open_cells = int(np.round(open_radius / max(cell_size, 1e-6)))
    if close_cells > 0:
        mask = binary_closing(mask, structure=_disk_structure(close_cells))
    if open_cells > 0:
        mask = binary_opening(mask, structure=_disk_structure(open_cells))
    mask = binary_fill_holes(mask)

    min_cells = max(int(np.ceil(min_patch_area / cell_area)), 1)
    labels, n_labels = label(mask)
    if n_labels > 0:
        counts = np.bincount(labels.ravel())
        keep = counts >= min_cells
        keep[0] = False
        mask = keep[labels]
    if not np.any(mask):
        return None, None

    active_cells = np.argwhere(mask)
    candidate_cells = np.argwhere(candidate_mask)
    if candidate_cells.shape[0] == 0:
        return None, None

    cand_ground_vals = ground_seed[candidate_cells[:, 0], candidate_cells[:, 1]]
    cand_relief_vals = relief_seed[candidate_cells[:, 0], candidate_cells[:, 1]]
    cand_density_vals = density_seed[candidate_cells[:, 0], candidate_cells[:, 1]]

    _emit_status(
        status_callback,
        "Vegetacion masa imprimible: sintetizando relieve",
        62,
        64.0,
        "Vegetacion",
    )

    smooth_sigma = organic_smooth / max(cell_size, 1e-6)
    ground_smooth = None
    relief_smooth = None
    density_smooth = None
    smooth_valid = None
    if smooth_sigma > 0.15:
        weight = candidate_mask.astype(np.float64)
        den = gaussian_filter(weight, sigma=smooth_sigma, mode="nearest")
        den_safe = np.maximum(den, 1e-9)
        smooth_valid = den > 0.02

        ground_smooth = gaussian_filter(ground_seed * weight, sigma=smooth_sigma, mode="nearest") / den_safe
        relief_smooth = gaussian_filter(relief_seed * weight, sigma=smooth_sigma, mode="nearest") / den_safe

        density_sigma = max(smooth_sigma * 0.8, 0.2)
        den_d = gaussian_filter(weight, sigma=density_sigma, mode="nearest")
        den_d_safe = np.maximum(den_d, 1e-9)
        density_smooth = gaussian_filter(density_seed * weight, sigma=density_sigma, mode="nearest") / den_d_safe

    if active_cells.shape[0] == candidate_cells.shape[0] and np.array_equal(active_cells, candidate_cells):
        nearest_idx = np.arange(active_cells.shape[0], dtype=np.int64)
    else:
        cell_tree = cKDTree(candidate_cells.astype(np.float64))
        _, nearest_idx = cell_tree.query(active_cells.astype(np.float64), k=1)
        nearest_idx = np.asarray(nearest_idx, dtype=np.int64)

    active_ground = cand_ground_vals[nearest_idx]
    active_relief = cand_relief_vals[nearest_idx]
    active_density = cand_density_vals[nearest_idx]
    if ground_smooth is not None and relief_smooth is not None and density_smooth is not None and smooth_valid is not None:
        smooth_ok = smooth_valid[mask]
        active_ground = np.where(smooth_ok, ground_smooth[mask], active_ground)
        active_relief = np.where(smooth_ok, relief_smooth[mask], active_relief)
        active_density = np.where(smooth_ok, density_smooth[mask], active_density)

    active_ix = active_cells[:, 1].astype(np.float64)
    active_iy = active_cells[:, 0].astype(np.float64)
    active_x = x_min + (active_ix + 0.5) * cell_size
    active_y = y_min + (active_iy + 0.5) * cell_size

    edge_distance = distance_transform_edt(mask).astype(np.float64, copy=False) * cell_size
    edge_scale = max(edge_softness, cell_size * 0.35)
    edge_factor = np.clip(edge_distance[mask] / edge_scale, 0.22, 1.0)

    seed_base = (
        abs(int(np.floor(x_min * 10.0))) * 73856093
        + abs(int(np.floor(y_min * 10.0))) * 19349663
        + int(nx) * 83492791
        + int(ny) * 2654435761
    ) & 0xFFFFFFFF
    rng = np.random.default_rng(int(seed_base))
    rand_field = rng.standard_normal((ny, nx), dtype=np.float64)

    pitch_cells = max(texture_pitch / max(cell_size, 1e-6), 1.0)
    detail_scale = float(np.clip(0.65 + 0.55 * micro_detail, 0.6, 1.8))
    sigma_large = max((pitch_cells / detail_scale) * 0.65, 0.45)
    sigma_mid = max(sigma_large * 0.45, 0.25)
    sigma_small = max(sigma_large * 0.22, 0.12)

    n_large = gaussian_filter(rand_field, sigma=sigma_large, mode="reflect")
    n_mid = gaussian_filter(rand_field, sigma=sigma_mid, mode="reflect")
    n_small = gaussian_filter(rand_field, sigma=sigma_small, mode="reflect")
    fbm = 0.55 * n_large + 0.30 * n_mid + 0.15 * n_small
    fbm_vals = fbm[mask]
    fbm_std = float(np.std(fbm_vals))
    if fbm_std < 1e-8:
        fbm_std = 1.0
    noise_random = np.clip((fbm_vals / fbm_std) * 0.70, -1.0, 1.0)

    freq = (2.0 * np.pi) / max(texture_pitch, cell_size * 0.45)
    noise_wave = (
        0.52 * np.sin(active_x * freq + active_y * freq * 0.45 + 0.7)
        + 0.30 * np.sin(active_x * freq * 1.9 - active_y * freq * 1.25 + 1.9)
        + 0.18 * np.cos(active_x * freq * 3.5 + active_y * freq * 2.85 + 2.2)
    )
    noise = np.clip(0.65 * noise_random + 0.35 * np.clip(noise_wave, -1.0, 1.0), -1.0, 1.0)

    density_norm_active = np.clip(
        np.log1p(active_density / max(min_density, 1e-6)) / np.log1p(6.0),
        0.1,
        2.0,
    )

    rough_amp = 0.24 + 0.34 * micro_detail
    rough_gain = 1.0 + roughness * rough_amp * noise
    density_gain = 1.0 + density_response * 0.18 * (density_norm_active - 0.8)
    edge_gain = 0.58 + 0.42 * edge_factor
    relief_active = np.clip(
        active_relief * rough_gain * density_gain * edge_gain,
        relief_floor * 0.75,
        relief_cap,
    )
    base_depth_local = np.clip(base_embed + 0.10 * relief_active + (1.0 - edge_factor) * 0.04, 0.06, 2.5)

    top_values = active_ground + relief_active
    bottom_values = active_ground - base_depth_local

    top_cells = np.zeros((ny, nx), dtype=np.float64)
    bottom_cells = np.zeros((ny, nx), dtype=np.float64)
    top_cells[mask] = top_values
    bottom_cells[mask] = bottom_values

    _emit_status(
        status_callback,
        "Vegetacion masa imprimible: mallando volumen",
        62,
        82.0,
        "Vegetacion",
    )

    verts, faces = _build_relief_mask_mesh(
        mask,
        top_cells,
        bottom_cells,
        min_x=x_min,
        min_y=y_min,
        cell_size=cell_size,
    )
    if verts is None or faces is None:
        return None, None

    _emit_status(status_callback, "Vegetacion masa imprimible completada", 63, 100.0, "Vegetacion")
    return verts.astype(np.float64, copy=False), faces.astype(np.int32, copy=False)


def process_vegetation_stylized(
    points: np.ndarray,
    classification: np.ndarray,
    *,
    ground_points: np.ndarray | None = None,
    min_mass_area: float = 12.0,
    detail_level: int = 2,
    include_low_vegetation: bool = False,
    max_clusters_per_class: int = 500,
    roughness: float = 0.45,
    density_response: float = 1.0,
    texture_pitch: float = 2.0,
    relief_cap: float = 3.0,
    min_density: float = 0.04,
    status_callback=None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Generate stylized vegetation as rugged low-profile masses.

    Goal: avoid giant tree-like "mushrooms" and produce printable
    vegetation texture closer to forest masses/moss over terrain.
    """
    detail_level = int(np.clip(detail_level, 1, 3))
    min_mass_area = max(float(min_mass_area), 0.5)
    max_clusters_per_class = max(int(max_clusters_per_class), 20)
    roughness = float(np.clip(roughness, 0.0, 1.2))
    density_response = float(np.clip(density_response, 0.0, 2.5))
    texture_pitch = max(float(texture_pitch), 0.08)
    relief_cap = max(float(relief_cap), 0.12)
    min_density = max(float(min_density), 0.001)

    if detail_level == 1:
        lon_segments, radial_segments = 12, 4
    elif detail_level == 2:
        lon_segments, radial_segments = 16, 6
    else:
        lon_segments, radial_segments = 22, 8

    ground_tree = None
    if ground_points is not None and ground_points.shape[0] >= 3:
        ground_tree = cKDTree(ground_points[:, :2])

    _emit_status(status_callback, "Vegetacion estilizada: preparando clases", 60, 5.0, "Vegetacion")

    class_specs: list[tuple[int, str, float, int, float, float, float, float]] = [
        # cls, label, cluster_radius, min_pts, min_area_factor, height_factor, cap_factor, texture_factor
        (5, "alta", 4.0, 28, 1.0, 0.34, 1.00, 1.00),
        (4, "media", 3.2, 22, 0.8, 0.29, 0.82, 0.82),
    ]
    if include_low_vegetation:
        class_specs.append((3, "baja", 2.4, 16, 1.25, 0.22, 0.62, 0.68))

    parts: list[tuple[np.ndarray, np.ndarray]] = []

    for class_idx, (
        cls_id,
        cls_label,
        radius_xy,
        min_pts,
        area_factor,
        height_factor,
        cap_factor,
        texture_factor,
    ) in enumerate(class_specs, start=1):
        mask = classification == cls_id
        if not np.any(mask):
            continue
        cls_points_raw = points[mask].astype(np.float64, copy=False)
        if cls_points_raw.shape[0] < min_pts:
            continue

        _emit_status(
            status_callback,
            f"Vegetacion estilizada: clase {cls_label}, reduciendo ruido",
            60,
            10.0 + class_idx * 8.0,
            "Vegetacion",
        )
        downsample_cell = max(radius_xy * (0.37 - 0.05 * detail_level), texture_pitch * 0.45, 0.2)
        cls_points = _downsample_points_xy(cls_points_raw, downsample_cell)
        clusters = _cluster_xy_components(
            cls_points,
            radius_xy=radius_xy,
            min_points=min_pts,
            max_clusters=max_clusters_per_class,
        )

        total_clusters = max(len(clusters), 1)
        emit_step = max(total_clusters // 25, 1)
        for cluster_idx, cluster in enumerate(clusters, start=1):
            area = _cluster_area_xy(cluster[:, :2])
            if area < min_mass_area * area_factor:
                continue
            point_density = float(cluster.shape[0] / max(area, 1e-6))
            if point_density < min_density:
                continue

            center_xy = np.mean(cluster[:, :2], axis=0)
            z05, _, z95 = np.percentile(cluster[:, 2], [5, 50, 95])
            if ground_tree is not None:
                _, gidx = ground_tree.query(center_xy.reshape(1, 2), k=1)
                ground_z = float(ground_points[int(gidx[0]), 2])
            else:
                ground_z = float(z05)
            if ground_z > z95:
                ground_z = float(z05)

            x_extent = float(np.percentile(cluster[:, 0], 95) - np.percentile(cluster[:, 0], 5))
            y_extent = float(np.percentile(cluster[:, 1], 95) - np.percentile(cluster[:, 1], 5))
            rx = max(x_extent * 0.52, radius_xy * 0.42)
            ry = max(y_extent * 0.52, radius_xy * 0.42)

            canopy_span = max(float(z95 - ground_z), 0.15)
            density_norm = float(np.clip(np.log1p(point_density) / np.log1p(4.0), 0.15, 1.8))
            density_gain = float(np.clip(1.0 + density_response * (density_norm - 0.65), 0.45, 1.8))
            target_height = canopy_span * height_factor * density_gain
            target_height = float(np.clip(target_height, 0.08, max(relief_cap * cap_factor, 0.18)))

            rough_local = float(np.clip(roughness * (0.70 + 0.55 * density_norm), 0.02, 1.2))
            texture_local = float(
                np.clip(texture_pitch * texture_factor / np.clip(density_norm, 0.65, 1.35), 0.08, 50.0)
            )

            if (
                total_clusters <= 40
                or cluster_idx == 1
                or cluster_idx == total_clusters
                or cluster_idx % emit_step == 0
            ):
                progress = 20.0 + (class_idx - 1) * 25.0 + (cluster_idx / total_clusters) * 20.0
                _emit_status(
                    status_callback,
                    f"Vegetacion estilizada: modelando masa {cls_label} {cluster_idx}/{total_clusters}",
                    62,
                    min(progress, 90.0),
                    "Vegetacion",
                )

            seed = (
                abs(int(center_xy[0] * 1000.0)) * 73856093
                + abs(int(center_xy[1] * 1000.0)) * 19349663
                + cls_id * 83492791
                + cluster_idx * 2654435761
            ) & 0xFFFFFFFF
            mound = _mesh_rugged_mound(
                center_xy=center_xy,
                base_z=ground_z,
                rx=np.clip(rx, 0.35, 25.0),
                ry=np.clip(ry, 0.35, 25.0),
                height=target_height,
                lon_segments=lon_segments,
                radial_segments=radial_segments,
                roughness=rough_local,
                texture_pitch=texture_local,
                seed=int(seed),
            )
            parts.append(mound)

    verts, faces = _merge_mesh_parts(parts)
    if verts is None or faces is None:
        return None, None

    _emit_status(status_callback, "Vegetacion estilizada completada", 63, 100.0, "Vegetacion")
    return verts, faces


def process_vegetation_poisson(
    points: np.ndarray,
    classification: np.ndarray,
    max_points: int = 75_000,
    poisson_depth: int = 8,
    density_quantile: float = 0.01,
    status_callback=None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Reconstruct vegetation with Poisson to produce watertight volumes."""
    _emit_status(status_callback, "Vegetacion: preparando puntos", 60, 5.0, "Vegetacion")
    if o3d is None:
        return None, None

    mask_veg = np.isin(classification, [3, 4, 5])
    if not np.any(mask_veg):
        return None, None

    veg_points = points[mask_veg].astype(np.float64, copy=False)
    if veg_points.shape[0] < 100:
        return None, None

    max_points = max(int(max_points), 100)
    if veg_points.shape[0] > max_points:
        rng = np.random.default_rng(42)
        indices = rng.choice(veg_points.shape[0], max_points, replace=False)
        veg_points = veg_points[indices]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(veg_points)

    try:
        _emit_status(status_callback, "Vegetacion: estimando normales", 61, 25.0, "Vegetacion")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=20)
        )
        pcd.orient_normals_consistent_tangent_plane(10)
    except Exception:
        return None, None

    try:
        _emit_status(status_callback, "Vegetacion: filtrando ruido", 61, 40.0, "Vegetacion")
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    except Exception:
        return None, None
    if len(pcd.points) < 100:
        return None, None

    try:
        _emit_status(status_callback, "Vegetacion: Poisson Surface Reconstruction", 62, 60.0, "Vegetacion")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=int(np.clip(poisson_depth, 6, 11)),
            width=0,
            scale=1.1,
            linear_fit=False,
        )
    except Exception:
        return None, None

    densities_arr = np.asarray(densities)
    if densities_arr.size > 0:
        q = float(np.clip(density_quantile, 0.0, 0.2))
        density_threshold = np.quantile(densities_arr, q)
        vertices_to_remove = densities_arr < density_threshold
        if np.any(vertices_to_remove):
            mesh.remove_vertices_by_mask(vertices_to_remove)

    try:
        _emit_status(status_callback, "Vegetacion: limpiando malla", 63, 85.0, "Vegetacion")
        triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    except Exception:
        return None, None
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    if cluster_n_triangles.size == 0:
        return None, None

    largest_cluster_idx = int(np.argmax(cluster_n_triangles))
    triangles_to_remove = np.asarray(triangle_clusters) != largest_cluster_idx
    mesh.remove_triangles_by_mask(triangles_to_remove)

    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    if vertices.shape[0] < 3 or faces.shape[0] < 1:
        return None, None

    _emit_status(status_callback, "Vegetacion completada", 63, 100.0, "Vegetacion")
    return vertices.astype(np.float64, copy=False), faces.astype(np.int32, copy=False)


def process_vegetation_as_metaballs(
    points: np.ndarray,
    classification: np.ndarray,
    sphere_radius: float = 0.5,
    grid_spacing: float | None = None,
    threshold: float = 0.5,
    max_grid_dim: int = 100,
    max_points: int = 25_000,
    status_callback=None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Fallback vegetation reconstruction using metaballs + marching cubes."""
    _emit_status(status_callback, "Vegetacion: preparando metaballs", 62, 10.0, "Vegetacion")
    if measure is None:
        return None, None

    mask_veg = np.isin(classification, [3, 4, 5])
    if not np.any(mask_veg):
        return None, None

    veg_points = points[mask_veg].astype(np.float64, copy=False)
    if veg_points.shape[0] < 10:
        return None, None

    max_points = max(int(max_points), 100)
    if veg_points.shape[0] > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(veg_points.shape[0], max_points, replace=False)
        veg_points = veg_points[idx]

    sphere_radius = max(float(sphere_radius), 0.05)

    min_bound = veg_points.min(axis=0) - sphere_radius
    max_bound = veg_points.max(axis=0) + sphere_radius

    if grid_spacing is None or grid_spacing <= 0:
        grid_spacing = max(sphere_radius / 2.0, 0.1)
    else:
        grid_spacing = max(float(grid_spacing), 0.05)

    dims = (max_bound - min_bound) / grid_spacing
    nx, ny, nz = np.ceil(dims).astype(int) + 1

    max_grid_dim = max(int(max_grid_dim), 20)
    if max(nx, ny, nz) > max_grid_dim:
        scale_factor = max(nx, ny, nz) / max_grid_dim
        grid_spacing *= scale_factor
        nx, ny, nz = np.ceil((max_bound - min_bound) / grid_spacing).astype(int) + 1

    x = np.linspace(min_bound[0], max_bound[0], nx)
    y = np.linspace(min_bound[1], max_bound[1], ny)
    z = np.linspace(min_bound[2], max_bound[2], nz)
    xg, yg, zg = np.meshgrid(x, y, z, indexing="ij")
    field = np.zeros_like(xg, dtype=np.float32)

    inv_sigma = 1.0 / (2.0 * sphere_radius * sphere_radius)
    total = veg_points.shape[0]
    chunk = max(total // 20, 1)
    for idx, pt in enumerate(veg_points, start=1):
        dist_sq = (xg - pt[0]) ** 2 + (yg - pt[1]) ** 2 + (zg - pt[2]) ** 2
        field += np.exp(-dist_sq * inv_sigma).astype(np.float32, copy=False)
        if idx == 1 or idx == total or idx % chunk == 0:
            step_pct = 15.0 + (idx / total) * 70.0
            _emit_status(status_callback, "Vegetacion: acumulando volumen metaballs", 62, step_pct, "Vegetacion")

    try:
        _emit_status(status_callback, "Vegetacion: extrayendo isosuperficie", 63, 92.0, "Vegetacion")
        verts, faces, _, _ = measure.marching_cubes(
            field,
            level=float(np.clip(threshold, 0.05, 5.0)),
            spacing=(grid_spacing, grid_spacing, grid_spacing),
        )
    except Exception:
        return None, None

    if verts.shape[0] == 0 or faces.shape[0] == 0:
        return None, None

    verts += min_bound
    _emit_status(status_callback, "Vegetacion completada (metaballs)", 63, 100.0, "Vegetacion")
    return verts.astype(np.float64, copy=False), faces.astype(np.int32, copy=False)
