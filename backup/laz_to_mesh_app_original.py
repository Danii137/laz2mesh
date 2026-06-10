import os
import tempfile
import time
from typing import Iterable

import laspy
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial import Delaunay, ConvexHull, QhullError, cKDTree

st.set_page_config(page_title="LAZ to 3D Mesh Converter", layout="wide")
st.title("Conversor de nube LAZ a malla 3D")
st.markdown(
    "Sube una nube de puntos LAZ/LAS, revisa las clases y genera una malla STL lista para imprimir o modelar."
)

STATE_DEFAULTS = {
    "points": None,
    "classification": None,
    "return_number": None,
    "num_returns": None,
    "file_loaded": False,
    "step": 1,
}

for key, value in STATE_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

THEMES = {
    "oscuro": {
        "background": "#0f172a",
        "surface": "#1e293b",
        "text": "#e2e8f0",
        "accent": "#38bdf8",
        "accent_alt": "#0ea5e9",
        "shadow": "rgba(15, 23, 42, 0.6)",
    },
    "claro": {
        "background": "#f8fafc",
        "surface": "#ffffff",
        "text": "#0f172a",
        "accent": "#2563eb",
        "accent_alt": "#3b82f6",
        "shadow": "rgba(148, 163, 184, 0.35)",
    },
}

if "ui_theme" not in st.session_state:
    st.session_state.ui_theme = "oscuro"

with st.sidebar:
    st.markdown("### Apariencia")
    theme_choice = st.radio(
        "Modo de color",
        options=("Claro", "Oscuro"),
        index=0 if st.session_state.ui_theme == "claro" else 1,
        key="theme_selector",
    )

selected_key = "claro" if theme_choice == "Claro" else "oscuro"
st.session_state.ui_theme = selected_key
theme = THEMES[selected_key]

st.markdown(
    f"""
    <style>
    .stApp {{
        background: {theme['background']};
        color: {theme['text']};
    }}
    .stApp [data-testid="stSidebar"] {{
        background: {theme['surface']};
        color: {theme['text']};
    }}
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{
        color: {theme['text']} !important;
    }}
    .stButton>button {{
        background: linear-gradient(120deg, {theme['accent']}, {theme['accent_alt']});
        color: white !important;
        border: none;
        box-shadow: 0 8px 16px {theme['shadow']};
    }}
    .stButton>button:hover {{
        filter: brightness(1.05);
        transform: translateY(-1px);
    }}
    .stProgress>div>div>div {{
        background: linear-gradient(90deg, {theme['accent']}, {theme['accent_alt']}) !important;
    }}
    .stSelectbox, .stRadio, .stSlider {{
        color: {theme['text']} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

CLASS_NAMES = {
    0: "No clasificado",
    1: "No asignado",
    2: "Terreno",
    3: "Vegetacion baja",
    4: "Vegetacion media",
    5: "Vegetacion alta",
    6: "Edificios",
    7: "Ruido",
    9: "Agua",
    10: "Ferrocarril",
    11: "Carreteras",
    12: "Overlap/Reservado",
    17: "Puentes",
}

CLASS_COLOR_SEQUENCE = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#FF9DA6",
    "#9C755F",
    "#B279A2",
    "#FFBF79",
    "#8C8C8C",
]

CLASS_COLOR_MAP: dict[int, str] = {}
for idx, cls in enumerate(sorted(CLASS_NAMES.keys())):
    CLASS_COLOR_MAP[int(cls)] = CLASS_COLOR_SEQUENCE[idx % len(CLASS_COLOR_SEQUENCE)]


def get_class_color(cls: int) -> str:
    cls_int = int(cls)
    if cls_int not in CLASS_COLOR_MAP:
        color = CLASS_COLOR_SEQUENCE[len(CLASS_COLOR_MAP) % len(CLASS_COLOR_SEQUENCE)]
        CLASS_COLOR_MAP[cls_int] = color
    return CLASS_COLOR_MAP[cls_int]


def class_label(cls: int) -> str:
    cls_int = int(cls)
    name = CLASS_NAMES.get(cls_int, f"Clase {cls_int}")
    return f"{cls_int:02d} - {name}"


def ensure_class_states(classes: Iterable[int]) -> None:
    converted: list[int] = []
    for c in classes:
        try:
            converted.append(int(c))
        except (TypeError, ValueError):
            continue
    classes_sorted = sorted(set(converted))
    st.session_state.available_classes = classes_sorted

    viz_state = st.session_state.get("viz_class_visibility", {})
    proc_state = st.session_state.get("proc_class_filter", {})

    default_excluded = {1, 7}
    for cls in classes_sorted:
        if cls not in viz_state:
            viz_state[cls] = True
        else:
            viz_state[cls] = bool(viz_state[cls])

        if cls not in proc_state:
            proc_state[cls] = cls not in default_excluded
        else:
            proc_state[cls] = bool(proc_state[cls])

        st.session_state.setdefault(f"viz_class_{cls}", viz_state[cls])
        st.session_state.setdefault(f"proc_class_{cls}", proc_state[cls])

        viz_state[cls] = bool(st.session_state[f"viz_class_{cls}"])
        proc_state[cls] = bool(st.session_state[f"proc_class_{cls}"])

    for cls in list(viz_state.keys()):
        if cls not in classes_sorted:
            viz_state.pop(cls, None)
            st.session_state.pop(f"viz_class_{cls}", None)
    for cls in list(proc_state.keys()):
        if cls not in classes_sorted:
            proc_state.pop(cls, None)
            st.session_state.pop(f"proc_class_{cls}", None)

    st.session_state.viz_class_visibility = viz_state
    st.session_state.proc_class_filter = proc_state


FILAMENT_NOZZLE_MM = 0.1

SCALE_RATIOS = {
    "1:10": 10,
    "1:25": 25,
    "1:50": 50,
    "1:100": 100,
    "1:250": 250,
    "1:500": 500,
    "1:1 000": 1_000,
    "1:2 000": 2_000,
    "1:5 000": 5_000,
    "1:10 000": 10_000,
    "1:25 000": 25_000,
    "1:50 000": 50_000,
}


def _clip(value: float, min_value: float, max_value: float) -> float:
    return float(np.clip(value, min_value, max_value))


def compute_scale_preset(scale_ratio: int) -> dict[str, float]:
    """Genera parametros recomendados segun la escala y el nozzle de impresion."""
    nozzle_m = FILAMENT_NOZZLE_MM / 1000.0
    min_feature_m = max(nozzle_m * scale_ratio, nozzle_m * 6.0)

    terrain_resolution = _clip(min_feature_m * 5.0, 0.5, 20.0)
    mesh_resolution = _clip(min_feature_m * 8.0, 0.5, 25.0)
    smooth_ground = _clip(min_feature_m * 18.0, 0.0, 3.0)
    smooth_vegetation = _clip(min_feature_m * 60.0, 0.0, 5.0)
    smooth_buildings = _clip(min_feature_m * 12.0, 0.0, 3.0)
    max_vertical_error = _clip(max(min_feature_m * 3.0, terrain_resolution * 0.7), 0.3, 12.0)
    max_slope_deg = float(np.clip(50.0 - np.log10(scale_ratio) * 8.0, 18.0, 45.0))

    sphere_radius = _clip(min_feature_m * 4.0, 0.25, 2.0)
    metaball_grid = _clip(min_feature_m * 3.0, 0.15, 1.5)
    metaball_threshold = _clip(0.45 + (min_feature_m * 2.0), 0.35, 0.9)
    voxel_simplify = _clip(min_feature_m * 4.0, 0.0, 5.0)
    building_plane_tolerance = _clip(min_feature_m * 1.2, 0.08, 0.3)

    base_thickness_real = nozzle_m * 30.0  # 3 mm en modelo
    base_thickness = _clip(base_thickness_real * scale_ratio, 1.0, 40.0)

    return {
        "terrain_resolution": terrain_resolution,
        "mesh_resolution": mesh_resolution,
        "smooth_ground": smooth_ground,
        "smooth_vegetation": smooth_vegetation,
        "smooth_buildings": smooth_buildings,
        "max_vertical_error": max_vertical_error,
        "max_slope_deg": max_slope_deg,
        "sphere_radius": sphere_radius,
        "metaball_grid": metaball_grid,
        "metaball_threshold": metaball_threshold,
        "voxel_simplify": voxel_simplify,
        "building_plane_tolerance": building_plane_tolerance,
        "base_thickness": base_thickness,
    }


SCALE_PRESETS = {label: compute_scale_preset(ratio) for label, ratio in SCALE_RATIOS.items()}


def smooth_grid(grid: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0 or grid is None:
        return grid
    working = grid.copy()
    nan_mask = np.isnan(working)
    if np.all(nan_mask):
        return working
    fill = np.nanmean(working[~nan_mask])
    working[nan_mask] = fill
    smoothed = gaussian_filter(working, sigma=sigma, mode="nearest")
    smoothed[nan_mask] = np.nan
    return smoothed


def remove_statistical_outliers(
    points: np.ndarray,
    classification: np.ndarray | None,
    neighbors: int = 30,
    std_ratio: float = 3.5,
    *,
    return_mask: bool = False,
) -> tuple[np.ndarray, np.ndarray | None] | tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    if points.shape[0] <= neighbors:
        mask_all = np.ones(points.shape[0], dtype=bool)
        if classification is None:
            if return_mask:
                return points, None, mask_all
            return points, None
        if return_mask:
            return points, classification, mask_all
        return points, classification

    tree = cKDTree(points)
    distances, _ = tree.query(points, k=min(neighbors + 1, points.shape[0]))
    distances = distances[:, 1:]
    mean_dist = distances.mean(axis=1)
    threshold = mean_dist.mean() + std_ratio * mean_dist.std()
    mask = mean_dist <= threshold

    filtered_points = points[mask]
    filtered_class = classification[mask] if classification is not None else None

    if return_mask:
        return filtered_points, filtered_class, mask

    return filtered_points, filtered_class


def build_mesh(
    points: np.ndarray,
    classification: np.ndarray | None,
    resolution: float,
    method: str,
    smooth_values: tuple[float, float, float],
    status_callback,
) -> tuple[np.ndarray, Delaunay]:
    smooth_ground, smooth_vegetation, smooth_buildings = smooth_values

    def notify(message: str, pct: int | None = None) -> None:
        if status_callback:
            status_callback(message, pct)

    notify("Calculando rejilla en XY", 50)
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    if np.isclose(x_min, x_max) or np.isclose(y_min, y_max):
        raise ValueError("La nube ocupa un area demasiado peque+/-a para generar una malla.")

    num_x = max(int(np.ceil((x_max - x_min) / resolution)) + 1, 2)
    num_y = max(int(np.ceil((y_max - y_min) / resolution)) + 1, 2)
    grid_x = np.linspace(x_min, x_max, num_x)
    grid_y = np.linspace(y_min, y_max, num_y)
    grid_x_2d, grid_y_2d = np.meshgrid(grid_x, grid_y)

    notify("Interpolando superficie base", 55)
    base_grid = griddata(points[:, :2], points[:, 2], (grid_x_2d, grid_y_2d), method=method, fill_value=np.nan)

    grids: list[np.ndarray] = []
    if classification is not None:
        masks = [
            ("Terreno", classification == 2, smooth_ground, method, 60),
            ("Vegetacion", np.isin(classification, [3, 4, 5]), smooth_vegetation, method, 65),
            ("Edificios", classification == 6, smooth_buildings, "nearest", 70),
            ("Otros", ~(np.isin(classification, [2, 3, 4, 5, 6])), 0.0, method, 72),
        ]
        for name, mask, sigma, interp_method, pct in masks:
            if not mask.any():
                continue
            notify(f"Interpolando {name.lower()} ({int(mask.sum()):,} puntos)", pct)
            grid = griddata(
                points[mask, :2],
                points[mask, 2],
                (grid_x_2d, grid_y_2d),
                method=interp_method,
                fill_value=np.nan,
            )
            grid = smooth_grid(grid, sigma)
            grids.append(grid)
    else:
        grids.append(base_grid)

    if not grids:
        grids.append(base_grid)

    combined = np.nanmax(grids, axis=0)
    if np.all(np.isnan(combined)):
        raise ValueError("La interpolacion genero solo valores nulos. Ajusta los parametros.")

    notify("Construyendo nube regular", 75)
    mesh_points = np.column_stack((grid_x_2d.ravel(), grid_y_2d.ravel(), combined.ravel()))
    valid_mask = ~np.isnan(mesh_points[:, 2])
    mesh_points = mesh_points[valid_mask]
    if mesh_points.shape[0] < 3:
        raise ValueError("La superficie interpolada no contiene puntos suficientes.")

    notify("Calculando triangulacion", 80)
    tri = Delaunay(mesh_points[:, :2])
    return mesh_points, tri


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
    """Aplica un workflow profesional basado en TIN progresivo sobre puntos de clase terreno."""

    def notify(message: str) -> None:
        if status_callback:
            status_callback(message, None)

    if points.size == 0:
        return None, None

    if classification is None:
        notify("Paso 1 - No hay clasificacion, usando toda la nube como terreno")
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
            notify("Paso 1 - Filtrado a retornos finales que tocan el terreno real")
        else:
            notify("Paso 1 - Retornos finales no detectados, usando todos los puntos de terreno")
    else:
        notify("Paso 1 - Informacion de retornos no disponible, usando todos los puntos de terreno")

    if ground_points.shape[0] < 3:
        return None, None

    neighbors = min(40, max(10, ground_points.shape[0] // 200))
    notify("Paso 1 - Eliminando outliers estadisticos del terreno")
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
                notify("Paso 1 - Eliminando elevaciones extremas del terreno")

    if ground_points.shape[0] < 3:
        return None, None

    notify("Paso 2 - Seleccionando seeds de TIN (puntos mas bajos por celda)")
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

    notify("Paso 2 - Densificando TIN progresivo")
    for _ in range(max_iterations):
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
            continue

        selected_indices = candidate_indices[offset_ok]
        new_points = candidate_points[offset_ok]
        classified_points = np.vstack([classified_points, new_points])

        keep_mask = np.ones(remaining_points.shape[0], dtype=bool)
        keep_mask[selected_indices] = False
        remaining_points = remaining_points[keep_mask]

    if classified_points.shape[0] < 3:
        return None, None

    unique_points = np.unique(classified_points, axis=0)
    if unique_points.shape[0] < 3:
        return None, None

    min_expected = max(20, int(0.1 * ground_points.shape[0]))
    if unique_points.shape[0] < min_expected and ground_points.shape[0] >= 3:
        notify("Paso 2 - Densificacion limitada, triangulando con puntos filtrados completos")
        unique_points = ground_points

    try:
        tri_final = Delaunay(unique_points[:, :2])
    except QhullError:
        return None, None

    notify("Paso 2 - TIN profesional listo")
    return unique_points, tri_final.simplices.astype(np.int32)


def process_vegetation_poisson(
    points: np.ndarray,
    classification: np.ndarray,
    max_points: int = 50_000,
    neighbors: int = 12,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Reconstruye vegetacion con Poisson Surface Reconstruction o aproxima si no hay soporte."""

    mask_veg = np.isin(classification, [3, 4, 5])
    if not mask_veg.any():
        return None, None

    veg_points = points[mask_veg].astype(np.float64, copy=True)
    total_points = veg_points.shape[0]
    if total_points < 30:
        return None, None

    if total_points > max_points:
        indices = np.random.choice(total_points, max_points, replace=False)
        veg_points = veg_points[indices]
        total_points = max_points

    neighbors = max(3, min(int(neighbors), total_points - 1))
    if neighbors < 3:
        return None, None

    def fallback_mesh() -> tuple[np.ndarray | None, np.ndarray | None]:
        try:
            hull = ConvexHull(veg_points)
            faces = np.asarray(hull.simplices, dtype=np.int32)
            if faces.size == 0:
                return None, None
            return veg_points, faces
        except (QhullError, ValueError):
            try:
                tri = Delaunay(veg_points[:, :2])
            except QhullError:
                return None, None
            faces = np.asarray(tri.simplices, dtype=np.int32)
            return veg_points, faces

    tree = cKDTree(veg_points)
    _, neighbor_idx = tree.query(veg_points, k=neighbors + 1)
    neighbor_points = veg_points[neighbor_idx[:, 1:]]
    centered = neighbor_points - neighbor_points.mean(axis=1, keepdims=True)
    if centered.shape[1] < 3:
        return fallback_mesh()

    cov = np.matmul(centered.transpose(0, 2, 1), centered)
    denom = max(centered.shape[1] - 1, 1)
    cov /= denom
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return fallback_mesh()

    normals = eigenvectors[:, :, 0]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, norms, where=norms > 0, out=np.zeros_like(normals))
    downward = normals[:, 2] < 0
    normals[downward] *= -1.0

    try:
        import open3d as o3d  # type: ignore
    except ImportError:
        return fallback_mesh()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(veg_points)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    try:
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
    except Exception:
        return fallback_mesh()

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    if faces.size == 0:
        return fallback_mesh()

    extent = np.ptp(veg_points, axis=0)
    margin = max(0.2, float(np.max(extent)) * 0.02)
    min_bound = veg_points.min(axis=0) - margin
    max_bound = veg_points.max(axis=0) + margin
    inside = np.all((vertices >= min_bound) & (vertices <= max_bound), axis=1)
    if not np.any(inside):
        return fallback_mesh()

    index_map = -np.ones(vertices.shape[0], dtype=np.int32)
    index_map[inside] = np.arange(np.count_nonzero(inside), dtype=np.int32)
    valid_faces_mask = np.all(inside[faces], axis=1)
    faces = faces[valid_faces_mask]
    if faces.size == 0:
        return fallback_mesh()

    faces = index_map[faces]
    vertices = vertices[inside]
    if vertices.shape[0] < 3:
        return fallback_mesh()

    return vertices.astype(np.float64, copy=False), faces.astype(np.int32, copy=False)


def process_vegetation_as_metaballs(
    points: np.ndarray,
    classification: np.ndarray,
    sphere_radius: float = 0.5,
    grid_spacing: float | None = None,
    threshold: float = 0.5,
    max_grid_dim: int = 100,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Representa la vegetacion como volumenes suaves utilizando metaballs."""
    mask_veg = np.isin(classification, [3, 4, 5])
    if not mask_veg.any():
        return None, None

    veg_points = points[mask_veg]

    x_min, x_max = veg_points[:, 0].min() - sphere_radius, veg_points[:, 0].max() + sphere_radius
    y_min, y_max = veg_points[:, 1].min() - sphere_radius, veg_points[:, 1].max() + sphere_radius
    z_min, z_max = veg_points[:, 2].min() - sphere_radius, veg_points[:, 2].max() + sphere_radius

    if grid_spacing is None or grid_spacing <= 0:
        grid_spacing = max(sphere_radius / 2.0, 0.1)
    else:
        grid_spacing = max(grid_spacing, 0.05)
    nx = int((x_max - x_min) / grid_spacing) + 1
    ny = int((y_max - y_min) / grid_spacing) + 1
    nz = int((z_max - z_min) / grid_spacing) + 1

    max_dim = max(10, int(max_grid_dim))
    if max(nx, ny, nz) > max_dim:
        factor = max(nx, ny, nz) / max_dim
        nx = max(int(nx / factor), 5)
        ny = max(int(ny / factor), 5)
        nz = max(int(nz / factor), 5)
        grid_spacing *= factor

    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    z = np.linspace(z_min, z_max, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    field = np.zeros((nx, ny, nz), dtype=np.float32)

    chunk_size = 1000
    for i in range(0, len(veg_points), chunk_size):
        chunk = veg_points[i : i + chunk_size]
        for pt in chunk:
            dist = np.sqrt((X - pt[0]) ** 2 + (Y - pt[1]) ** 2 + (Z - pt[2]) ** 2)
            influence = np.exp(-(dist**2) / (2 * sphere_radius**2))
            field += influence.astype(np.float32)

    try:
        from skimage import measure
    except ImportError as exc:  # pragma: no cover - dependencias en runtime
        raise ImportError(
            "El modo avanzado requiere la libreria scikit-image para generar la vegetacion."
        ) from exc

    verts, faces, _, _ = measure.marching_cubes(
        field,
        level=float(np.clip(threshold, 0.05, 5.0)),
        spacing=(grid_spacing, grid_spacing, grid_spacing),
    )

    verts[:, 0] += x_min
    verts[:, 1] += y_min
    verts[:, 2] += z_min

    return verts.astype(np.float64), faces.astype(np.int32)


def process_buildings_as_prisms(
    points: np.ndarray,
    classification: np.ndarray,
    cluster_radius: float = 2.0,
    min_points: int = 10,
    roof_percentile: float = 90.0,
    base_percentile: float = 10.0,
    *,
    ground_points: np.ndarray | None = None,
    plane_tolerance: float = 0.18,
    max_plane_iterations: int = 120,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Reconstruye edificios mediante parches planos independientes extruidos hasta el terreno."""
    mask_buildings = classification == 6
    if not mask_buildings.any():
        return None, None

    building_points = points[mask_buildings]
    if building_points.shape[0] < max(int(min_points), 3):
        return None, None

    radius_xy = max(float(cluster_radius), 0.5)
    plane_tol = max(float(plane_tolerance), 0.05)
    min_cluster_points = max(int(min_points), 4)
    min_patch_points = max(int(min_points // 2), 30)

    tree = cKDTree(building_points[:, :2])
    ground_tree = None
    if ground_points is not None and ground_points.shape[0] >= 3:
        ground_tree = cKDTree(ground_points[:, :2])

    visited = np.zeros(building_points.shape[0], dtype=bool)
    clusters: list[np.ndarray] = []
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
            neighbors = tree.query_ball_point(building_points[idx, :2], r=radius_xy)
            for nb in neighbors:
                if not visited[nb]:
                    queue.append(nb)
        if len(component) >= min_cluster_points:
            clusters.append(building_points[np.array(component, dtype=int)])

    if not clusters:
        return None, None

    rng = np.random.default_rng(42)
    all_vertices: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    vertex_offset = 0

    for cluster_pts in clusters:
        if cluster_pts.shape[0] < min_cluster_points:
            continue

        remaining = np.arange(cluster_pts.shape[0], dtype=int)
        patches: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        while remaining.size >= min_patch_points:
            candidates = cluster_pts[remaining]
            if candidates.shape[0] < 3:
                break
            best_inliers: np.ndarray | None = None
            best_normal: np.ndarray | None = None
            best_point: np.ndarray | None = None
            for _ in range(max_plane_iterations):
                sample_ids = rng.choice(candidates.shape[0], size=3, replace=False)
                p0, p1, p2 = candidates[sample_ids]
                normal = np.cross(p1 - p0, p2 - p0)
                norm = np.linalg.norm(normal)
                if norm < 1e-6:
                    continue
                normal /= norm
                if normal[2] < 0:
                    normal *= -1.0
                distances = np.abs((candidates - p0) @ normal)
                inliers_mask = distances <= plane_tol
                count = int(inliers_mask.sum())
                if count >= min_patch_points and (best_inliers is None or count > best_inliers.size):
                    best_inliers = remaining[inliers_mask]
                    best_normal = normal
                    best_point = p0
            if best_inliers is None or best_inliers.size < min_patch_points:
                break
            patches.append((best_inliers, best_normal, best_point))
            remaining = np.setdiff1d(remaining, best_inliers, assume_unique=True)

        if not patches:
            centroid = np.mean(cluster_pts, axis=0)
            try:
                _, _, vh = np.linalg.svd(cluster_pts - centroid, full_matrices=False)
            except np.linalg.LinAlgError:
                continue
            normal = vh[-1]
            if normal[2] < 0:
                normal *= -1.0
            if abs(normal[2]) < 1e-6:
                normal[2] = 1e-6
            patches = [(np.arange(cluster_pts.shape[0], dtype=int), normal, centroid)]

        for patch_indices, plane_normal, plane_point in patches:
            patch_points = cluster_pts[patch_indices]
            if patch_points.shape[0] < 3:
                continue

            if plane_normal[2] <= 0.2:
                continue

            roof_mask = patch_points[:, 2] >= np.percentile(
                patch_points[:, 2], np.clip(roof_percentile, 0.0, 100.0)
            )
            roof_candidates = patch_points[roof_mask]
            if roof_candidates.shape[0] < 3:
                roof_candidates = patch_points
            if roof_candidates.shape[0] < 3:
                continue

            xy = roof_candidates[:, :2]
            quant_step = max(radius_xy / 5.0, 0.05)
            quant_xy = np.round(xy / quant_step, decimals=3)
            _, unique_idx = np.unique(quant_xy, axis=0, return_index=True)
            roof_points = roof_candidates[unique_idx]
            if roof_points.shape[0] < 3:
                continue

            try:
                roof_tri = Delaunay(roof_points[:, :2])
            except QhullError:
                continue
            if roof_tri.simplices.size == 0:
                continue

            def plane_height(xy_local: np.ndarray) -> np.ndarray:
                dx = xy_local[:, 0] - plane_point[0]
                dy = xy_local[:, 1] - plane_point[1]
                return plane_point[2] - (plane_normal[0] * dx + plane_normal[1] * dy) / plane_normal[2]

            roof_z = plane_height(roof_points[:, :2])

            if ground_tree is not None:
                _, ground_idx = ground_tree.query(roof_points[:, :2], k=1)
                base_z = ground_points[ground_idx][:, 2]
            else:
                base_value = float(
                    np.percentile(cluster_pts[:, 2], np.clip(base_percentile, 0.0, 100.0))
                )
                base_z = np.full(roof_points.shape[0], base_value, dtype=np.float64)

            base_z = np.minimum(base_z, roof_z - 0.05)
            if np.any(~np.isfinite(base_z)):
                continue

            top_vertices = np.column_stack([roof_points[:, :2], roof_z])
            base_vertices = np.column_stack([roof_points[:, :2], base_z])
            vertex_count = top_vertices.shape[0]
            if vertex_count < 3:
                continue

            top_faces = roof_tri.simplices.tolist()
            bottom_faces = (roof_tri.simplices[:, ::-1] + vertex_count).tolist()

            edge_counts: dict[tuple[int, int], int] = {}
            for tri in roof_tri.simplices:
                a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
                for e0, e1 in ((a, b), (b, c), (c, a)):
                    key = (min(e0, e1), max(e0, e1))
                    edge_counts[key] = edge_counts.get(key, 0) + 1

            wall_faces: list[list[int]] = []
            for (a, b), count in edge_counts.items():
                if count == 1:
                    wall_faces.append([a, b, b + vertex_count])
                    wall_faces.append([a, b + vertex_count, a + vertex_count])

            faces = top_faces + bottom_faces + wall_faces
            if not faces:
                continue

            verts = np.vstack([top_vertices, base_vertices])
            faces_arr = (np.asarray(faces, dtype=np.int32) + vertex_offset).astype(np.int32, copy=False)
            all_vertices.append(verts)
            all_faces.append(faces_arr)
            vertex_offset += verts.shape[0]

    if not all_vertices:
        return None, None

    combined_vertices = np.vstack(all_vertices).astype(np.float64, copy=False)
    combined_faces = np.vstack(all_faces).astype(np.int32, copy=False)
    return combined_vertices, combined_faces


def merge_meshes(
    terrain_data: tuple[np.ndarray | None, np.ndarray | None],
    vegetation_data: tuple[np.ndarray | None, np.ndarray | None],
    buildings_data: tuple[np.ndarray | None, np.ndarray | None],
) -> tuple[np.ndarray, np.ndarray]:
    """Combina terreno, vegetacion y edificios en una unica malla."""
    all_verts: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    vertex_offset = 0

    if terrain_data[0] is not None and terrain_data[1] is not None:
        all_verts.append(terrain_data[0])
        all_faces.append(terrain_data[1])
        vertex_offset += terrain_data[0].shape[0]

    if vegetation_data[0] is not None and vegetation_data[1] is not None:
        all_verts.append(vegetation_data[0])
        all_faces.append(vegetation_data[1] + vertex_offset)
        vertex_offset += vegetation_data[0].shape[0]

    if buildings_data[0] is not None and buildings_data[1] is not None:
        all_verts.append(buildings_data[0])
        all_faces.append(buildings_data[1] + vertex_offset)

    if not all_verts:
        raise ValueError("No hay geometra v!lida para crear la malla.")

    final_verts = np.vstack(all_verts).astype(np.float64, copy=False)
    final_faces = np.vstack(all_faces).astype(np.int32, copy=False)
    return final_verts, final_faces


def simplify_mesh_with_voxel_grid(
    vertices: np.ndarray, faces: np.ndarray, voxel_size: float
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce el numero de vertices agrupandolos en una rejilla tridimensional."""
    if voxel_size <= 0 or vertices.shape[0] == 0:
        return vertices, faces

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
) -> tuple[np.ndarray, np.ndarray]:
    """Orquesta el modo avanzado combinando terreno, vegetacion y edificios."""

    if classification is None:
        raise ValueError("El modo avanzado requiere datos de clasificacion.")

    def notify(message: str, pct: int | None = None) -> None:
        if status_callback:
            status_callback(message, pct)

    notify("Paso 2 - Terreno TIN profesional", 55)
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
        notify("Sin clase de terreno; usando toda la nube como superficie base", 57)
        tri = Delaunay(points[:, :2])
        terrain = (points.astype(np.float64, copy=True), tri.simplices.astype(np.int32))

    notify("Paso 3 - Reconstruyendo vegetacion (Poisson)", 60)
    vegetation = process_vegetation_poisson(points, classification)
    if vegetation[0] is None or vegetation[1] is None:
        notify("Paso 3 - Poisson no disponible, probando aproximacion por metaballs", 62)
        try:
            vegetation = process_vegetation_as_metaballs(
                points,
                classification,
                sphere_radius=sphere_radius,
                grid_spacing=metaball_grid,
                threshold=metaball_threshold,
                max_grid_dim=int(metaball_max_dim),
            )
        except ImportError:
            notify(
                "Sin librerias Poisson/metaballs disponibles; vegetacion omitida en la fusion",
                62,
            )
            vegetation = (None, None)
    else:
        notify("Paso 3 - Vegetacion reconstruida con Poisson Surface Reconstruction", 62)

    notify("Paso 3 - Reconstruyendo edificios a partir de planos", 65)
    buildings = process_buildings_as_prisms(
        points,
        classification,
        cluster_radius=building_cluster_radius,
        min_points=building_min_points,
        roof_percentile=building_roof_percentile,
        base_percentile=building_base_percentile,
        ground_points=terrain[0],
        plane_tolerance=building_plane_tolerance,
    )
    if buildings[0] is None:
        notify("No se detectaron edificios", 67)

    notify("Paso 4 - Fusionando capas en una malla final", 70)
    merged_vertices, merged_faces = merge_meshes(terrain, vegetation, buildings)
    return merged_vertices, merged_faces


def add_base_to_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    thickness: float,
    hull_edges: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Anade una base solida extruida hacia abajo manteniendo las paredes laterales."""
    if vertices.shape[0] < 3:
        return vertices, faces

    base_layer = vertices.copy()
    base_layer[:, 2] = base_layer[:, 2].min() - float(thickness)
    original_count = vertices.shape[0]

    if hull_edges is None:
        try:
            hull = ConvexHull(vertices[:, :2])
            hull_edges = hull.simplices
        except QhullError:
            if original_count < 2:
                hull_edges = np.empty((0, 2), dtype=np.int32)
            else:
                idx = np.arange(original_count - 1, dtype=np.int32)
                hull_edges = np.column_stack((idx, idx + 1))

    side_faces = []
    for edge in hull_edges:
        a, b = int(edge[0]), int(edge[1])
        side_faces.append([a, b, b + original_count])
        side_faces.append([a, b + original_count, a + original_count])

    base_faces = faces[:, [0, 2, 1]] + original_count
    combined_vertices = np.vstack([vertices, base_layer])
    side_faces_arr = np.asarray(side_faces, dtype=np.int32)
    if side_faces_arr.size == 0:
        side_faces_arr = side_faces_arr.reshape(0, 3)
    combined_faces = np.vstack([faces, side_faces_arr, base_faces.astype(np.int32)])
    return combined_vertices, combined_faces


def write_stl(path: str, points: np.ndarray, triangles: np.ndarray) -> None:
    with open(path, "wb") as fh:
        fh.write(b"\0" * 80)
        fh.write(np.uint32(len(triangles)).tobytes())
        for tri in triangles:
            v0, v1, v2 = points[tri[0]], points[tri[1]], points[tri[2]]
            normal = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(normal)
            if norm == 0:
                normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            else:
                normal = (normal / norm).astype(np.float32)
            fh.write(normal.astype(np.float32).tobytes())
            fh.write(v0.astype(np.float32).tobytes())
            fh.write(v1.astype(np.float32).tobytes())
            fh.write(v2.astype(np.float32).tobytes())
            fh.write(b"\0\0")


st.header("Paso 1: Cargar nube de puntos")

uploaded_file = st.file_uploader(
    "Selecciona archivo LAZ o LAS",
    type=["laz", "las"],
    help="Formatos soportados: .laz, .las",
)

if uploaded_file and not st.session_state.file_loaded:
    tmp_path = None
    with st.spinner("Cargando archivo..."):
        try:
            suffix = os.path.splitext(uploaded_file.name)[1].lower() or ".laz"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            las = laspy.read(tmp_path)
            points = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)
            classification_attr = getattr(las, "classification", None)
            if classification_attr is None:
                classification = np.ones(points.shape[0], dtype=np.uint8)
            else:
                classification = np.asarray(classification_attr, dtype=np.uint8)

            return_attr = getattr(las, "return_number", None)
            if return_attr is not None:
                return_number = np.asarray(return_attr, dtype=np.uint8)
            else:
                return_number = None

            total_return_attr = getattr(las, "num_returns", None)
            if total_return_attr is not None:
                num_returns = np.asarray(total_return_attr, dtype=np.uint8)
            else:
                num_returns = None

            st.session_state.points = points
            st.session_state.classification = classification
            st.session_state.return_number = return_number
            st.session_state.num_returns = num_returns
            ensure_class_states(np.unique(classification))
            st.session_state.file_loaded = True
            st.session_state.step = 2
            st.success(f"Archivo cargado: {uploaded_file.name}")
            st.info(f"Total de puntos: {points.shape[0]:,}")
        except Exception as exc:
            st.session_state.points = None
            st.session_state.classification = None
            st.session_state.return_number = None
            st.session_state.num_returns = None
            st.session_state.file_loaded = False
            st.session_state.available_classes = []
            st.session_state.viz_class_visibility = {}
            st.session_state.proc_class_filter = {}
            st.error(f"No se pudo leer el archivo: {exc}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        st.rerun()

if st.session_state.file_loaded and st.session_state.step >= 2:
    st.markdown("---")
    st.header("Paso 2: Revisar y clasificar puntos")

    col_plot, col_side = st.columns([2, 1])
    points = st.session_state.points
    classification = st.session_state.classification
    ensure_class_states(np.unique(classification))

    with col_side:
        st.subheader("Estadisticas de clasificacion")
        unique_classes = np.unique(classification)
        for cls in unique_classes:
            count = int(np.sum(classification == cls))
            pct = (count / classification.size) * 100 if classification.size else 0
            name = CLASS_NAMES.get(cls, f"Clase {cls}")
            st.text(f"{name}: {count:,} ({pct:.1f}%)")

        st.markdown("---")
        st.subheader("Reclasificacion rapida")
        if st.checkbox("Reclasificar por altura Z", key="height_reclass"):
            z_relative = points[:, 2] - points[:, 2].min()
            ground_max = st.slider("Altura max terreno", 0.0, 10.0, 2.0, 0.5)
            veg_low_max = st.slider("Altura max vegetacion baja", 0.0, 20.0, 5.0, 0.5)
            veg_mid_max = st.slider("Altura max vegetacion media", 0.0, 30.0, 15.0, 0.5)
            if st.button("Aplicar reclasificacion", key="apply_height_reclass"):
                new_class = np.ones_like(classification)
                new_class[z_relative <= ground_max] = 2
                mask_low = (z_relative > ground_max) & (z_relative <= veg_low_max)
                new_class[mask_low] = 3
                mask_mid = (z_relative > veg_low_max) & (z_relative <= veg_mid_max)
                new_class[mask_mid] = 4
                new_class[z_relative > veg_mid_max] = 5
            st.session_state.classification = new_class
            st.success("Reclasificacion aplicada")
            st.rerun()

        st.markdown("---")
        st.subheader("Filtro visual por clase")
        available_classes = st.session_state.available_classes
        viz_state = st.session_state.viz_class_visibility
        class_counts = {
            cls: int(np.sum(classification == cls)) for cls in available_classes
        }
        btn_viz_all, btn_viz_none = st.columns(2)
        if btn_viz_all.button("Ver todas", key="btn_viz_all"):
            for cls in available_classes:
                viz_state[cls] = True
                st.session_state[f"viz_class_{cls}"] = True
            st.session_state.viz_class_visibility = viz_state
            st.rerun()
        if btn_viz_none.button("Ocultar todas", key="btn_viz_none"):
            for cls in available_classes:
                viz_state[cls] = False
                st.session_state[f"viz_class_{cls}"] = False
            st.session_state.viz_class_visibility = viz_state
            st.rerun()

        for cls in available_classes:
            key = f"viz_class_{cls}"
            current = st.session_state.get(key, viz_state.get(cls, True))
            new_val = st.checkbox(
                f"{class_label(cls)} ({class_counts.get(cls,0):,})",
                value=current,
                key=key,
            )
            viz_state[cls] = bool(new_val)
        st.session_state.viz_class_visibility = viz_state

    with col_plot:
        st.subheader("Visualizacion de puntos")
        max_points = st.slider("Puntos a visualizar", 5_000, 100_000, 30_000, 5_000)
        color_by = st.radio("Color por", ("Clasificacion", "Altura"), horizontal=True)

        viz_state = st.session_state.viz_class_visibility
        visible_classes = [cls for cls, flag in viz_state.items() if flag]
        if visible_classes:
            mask_visible = np.isin(classification, visible_classes)
        else:
            mask_visible = np.zeros(classification.shape[0], dtype=bool)

        if not np.any(mask_visible):
            st.info("Selecciona al menos una clase para visualizar.")
        else:
            filtered_points = points[mask_visible]
            filtered_class = classification[mask_visible]

            if filtered_points.shape[0] > max_points:
                indices = np.random.choice(filtered_points.shape[0], max_points, replace=False)
                viz_points = filtered_points[indices]
                viz_class = filtered_class[indices]
            else:
                viz_points = filtered_points
                viz_class = filtered_class

            fig = go.Figure()

            if color_by == "Clasificacion":
                for cls in visible_classes:
                    mask_cls = viz_class == cls
                    if not np.any(mask_cls):
                        continue
                    cls_points = viz_points[mask_cls]
                    fig.add_trace(
                        go.Scatter3d(
                            x=cls_points[:, 0],
                            y=cls_points[:, 1],
                            z=cls_points[:, 2],
                            mode="markers",
                            marker=dict(size=1.6, color=get_class_color(cls)),
                            name=class_label(cls),
                            legendgroup=str(cls),
                        )
                    )
                if not fig.data:
                    st.info("Las clases seleccionadas no tienen puntos en el muestreo actual.")
                    fig = None
            else:
                fig.add_trace(
                    go.Scatter3d(
                        x=viz_points[:, 0],
                        y=viz_points[:, 1],
                        z=viz_points[:, 2],
                        mode="markers",
                        marker=dict(
                            size=1.6,
                            color=viz_points[:, 2],
                            colorscale="earth",
                            showscale=True,
                            colorbar=dict(title="Altura (m)"),
                        ),
                        name="Puntos",
                    )
                )

            if fig is not None:
                fig.update_layout(
                    scene=dict(
                        xaxis_title="X (m)",
                        yaxis_title="Y (m)",
                        zaxis_title="Z (m)",
                        aspectmode="data",
                    ),
                    legend=dict(orientation="h"),
                    height=600,
                    margin=dict(l=0, r=0, b=0, t=30),
                )
                st.plotly_chart(fig, use_container_width=True)

    if st.button("Continuar a procesamiento", key="go_to_step3"):
        st.session_state.step = 3
        st.rerun()

if st.session_state.file_loaded and st.session_state.step >= 3:
    st.markdown("---")
    st.header("Paso 3: Configurar y procesar malla")

    classification = st.session_state.classification
    ensure_class_states(np.unique(classification))

    st.sidebar.header("Parametros de procesamiento")
    scale_options = list(SCALE_PRESETS.keys()) + ["Personalizado"]
    default_scale = st.session_state.get("proc_scale_preset", "1:1 000")
    if default_scale not in scale_options:
        default_scale = "Personalizado"
    selected_scale = st.sidebar.selectbox(
        "Escala objetivo (modelo : terreno real)",
        options=scale_options,
        index=scale_options.index(default_scale),
    )
    st.session_state.proc_scale_preset = selected_scale
    preset_values = SCALE_PRESETS.get(selected_scale)
    scale_ratio_value = SCALE_RATIOS.get(selected_scale)
    if scale_ratio_value is not None:
        min_feature = (FILAMENT_NOZZLE_MM / 1000.0) * scale_ratio_value
        st.sidebar.caption(
            f"Detalle minimo imprimible ~ {min_feature:.3f} m sobre el terreno (boquilla {FILAMENT_NOZZLE_MM} mm)."
        )

    processing_mode = st.sidebar.selectbox(
        "Modo de generacion",
        ("Interpolacion en rejilla", "Modo avanzado (detalle maximo)"),
        index=0,
    )

    smooth_ground_default = 0.5
    smooth_vegetation_default = 2.5
    smooth_buildings_default = 0.3
    mesh_resolution_default = 1.5
    interpolation_method = "linear"
    sphere_radius_default = 0.5
    terrain_resolution_default = 5.0
    metaball_grid_default = 0.3
    metaball_threshold_default = 0.5
    metaball_max_dim = 100
    building_cluster_radius = 2.0
    building_min_points = 12
    building_roof_percentile = 90.0
    building_base_percentile = 10.0
    voxel_simplify_default = 0.0
    max_vertical_error_default = 1.0
    max_slope_default = 40.0
    building_plane_tolerance_default = 0.18
    base_thickness_default = 5.0

    if preset_values:
        smooth_ground_default = preset_values["smooth_ground"]
        smooth_vegetation_default = preset_values["smooth_vegetation"]
        smooth_buildings_default = preset_values["smooth_buildings"]
        mesh_resolution_default = preset_values["mesh_resolution"]
        terrain_resolution_default = preset_values["terrain_resolution"]
        max_vertical_error_default = preset_values["max_vertical_error"]
        max_slope_default = preset_values.get("max_slope_deg", max_slope_default)
        sphere_radius_default = preset_values["sphere_radius"]
        metaball_grid_default = preset_values["metaball_grid"]
        metaball_threshold_default = preset_values["metaball_threshold"]
        voxel_simplify_default = preset_values["voxel_simplify"]
        building_plane_tolerance_default = preset_values.get("building_plane_tolerance", building_plane_tolerance_default)
        base_thickness_default = preset_values["base_thickness"]

    if processing_mode == "Modo avanzado (detalle maximo)":
        st.info(
            "\n".join(
                [
                    "Workflow profesional activado:",
                    "- Paso 1 - Filtrado de terreno real (clase 2, retornos finales, outliers).",
                    "- Paso 2 - Progressive TIN densification que preserva detalle topografico.",
                    "- Paso 3 - Vegetacion con Poisson Surface Reconstruction y edificios extruidos.",
                    "- Paso 4 - Fusion de capas manteniendo geometria independiente.",
                ]
            )
        )

    if processing_mode == "Interpolacion en rejilla":
        st.sidebar.subheader("Suavizado por clase")
        smooth_ground = st.sidebar.slider("Suavizado terreno", 0.0, 3.0, smooth_ground_default, 0.1)
        smooth_vegetation = st.sidebar.slider(
            "Suavizado vegetacion", 0.0, 5.0, smooth_vegetation_default, 0.1
        )
        smooth_buildings = st.sidebar.slider(
            "Suavizado edificios", 0.0, 3.0, smooth_buildings_default, 0.1
        )

        st.sidebar.subheader("Generacion de malla")
        mesh_resolution = st.sidebar.slider(
            "Resolucion (m)", 0.5, 25.0, mesh_resolution_default, 0.5
        )
        interpolation_method = st.sidebar.selectbox(
            "Interpolacion", ["linear", "cubic", "nearest"], index=0
        )
        terrain_resolution = mesh_resolution
        max_vertical_error = max_vertical_error_default
        max_slope_deg = max_slope_default
        building_plane_tolerance = building_plane_tolerance_default
    else:
        st.sidebar.subheader("Parametros avanzados")
        sphere_radius = st.sidebar.slider(
            "Radio de esfera para vegetacion (m)", 0.2, 2.0, sphere_radius_default, 0.1
        )
        terrain_resolution = st.sidebar.slider(
            "Tamano celda seed TIN (m)", 0.5, 20.0, terrain_resolution_default, 0.5
        )
        max_vertical_error = st.sidebar.slider(
            "Tolerancia vertical TIN (m)", 0.1, 12.0, max_vertical_error_default, 0.1
        )
        max_slope_deg = st.sidebar.slider(
            "Pendiente maxima TIN (deg)", 10.0, 60.0, max_slope_default, 1.0
        )
        metaball_grid = st.sidebar.slider(
            "Resolucion grid vegetacion (m)", 0.1, 2.0, metaball_grid_default, 0.1
        )
        metaball_threshold = st.sidebar.slider(
            "Umbral fusion vegetacion", 0.2, 1.5, metaball_threshold_default, 0.1
        )
        metaball_max_dim = st.sidebar.slider(
            "Limite celdas por eje", 30, 200, 100, 10
        )
        building_cluster_radius = st.sidebar.slider(
            "Union maxima entre puntos de edificio (m)", 0.5, 10.0, 2.0, 0.25
        )
        building_roof_percentile = st.sidebar.slider(
            "Percentil techo edificio (%)", 60.0, 100.0, 90.0, 5.0
        )
        building_base_percentile = st.sidebar.slider(
            "Percentil base edificio (%)", 0.0, 40.0, 10.0, 5.0
        )
        building_plane_tolerance = st.sidebar.slider(
            "Tolerancia plano techo (m)", 0.05, 0.4, 0.18, 0.01
        )
        building_min_points = int(
            st.sidebar.slider("Puntos minimos por edificio", 4, 60, 12, 1)
        )
        mesh_resolution = mesh_resolution_default
        smooth_ground = smooth_ground_default
        smooth_vegetation = smooth_vegetation_default
        smooth_buildings = smooth_buildings_default

    st.sidebar.subheader("Optimizacion")
    voxel_simplify = st.sidebar.slider(
        "Simplificacion por voxel (m)", 0.0, 5.0, voxel_simplify_default, 0.1
    )

    st.sidebar.subheader("Clases incluidas")
    available_classes = st.session_state.available_classes
    proc_state = st.session_state.proc_class_filter
    proc_counts = {cls: int(np.sum(st.session_state.classification == cls)) for cls in available_classes}
    proc_all_col, proc_none_col = st.sidebar.columns(2)
    if proc_all_col.button("Activar todas", key="proc_classes_all"):
        for cls in available_classes:
            proc_state[cls] = True
            st.session_state[f"proc_class_{cls}"] = True
        st.session_state.proc_class_filter = proc_state
        st.rerun()
    if proc_none_col.button("Desactivar todas", key="proc_classes_none"):
        for cls in available_classes:
            proc_state[cls] = False
            st.session_state[f"proc_class_{cls}"] = False
        st.session_state.proc_class_filter = proc_state
        st.rerun()

    for cls in available_classes:
        key = f"proc_class_{cls}"
        current = st.session_state.get(key, proc_state.get(cls, True))
        new_val = st.sidebar.checkbox(
            f"{class_label(cls)} ({proc_counts.get(cls, 0):,})",
            value=current,
            key=key,
        )
        proc_state[cls] = bool(new_val)
    st.session_state.proc_class_filter = proc_state

    st.sidebar.subheader("Limpieza")
    remove_outliers = st.sidebar.checkbox("Eliminar outliers", value=True)
    percentile_low = st.sidebar.slider("Percentil inferior Z", 0, 5, 0)
    percentile_high = st.sidebar.slider("Percentil superior Z", 95, 100, 100)

    st.sidebar.subheader("Base")
    add_base = st.sidebar.checkbox("Anadir base solida", value=True)
    base_thickness = (
        st.sidebar.slider("Grosor base (m)", 1.0, 40.0, base_thickness_default, 0.5) if add_base else None
    )

    st.sidebar.subheader("Visualizacion")
    st.sidebar.checkbox(
        "Mostrar vista previa 3D",
        value=st.session_state.get("proc_show_preview", True),
        key="proc_show_preview",
    )

    col_process, col_back = st.columns([3, 1])

    with col_process:
        if st.button("Generar malla 3D", type="primary"):
            progress_bar = st.progress(0)
            status_container = st.container()
            with status_container:
                status_current = st.empty()
                status_log = st.empty()
            start_time = time.time()
            log_messages: list[str] = []

            def update_status(message: str, pct: int | None = None, state: str = "info") -> None:
                elapsed = time.time() - start_time
                formatted = f"{message} ({elapsed:.1f}s)"
                if state == "info":
                    status_current.info(formatted)
                elif state == "success":
                    status_current.success(formatted)
                else:
                    status_current.error(formatted)
                log_messages.append(f"{message} - {elapsed:.1f}s")
                status_log.markdown("\n".join(f"- {msg}" for msg in log_messages))
                if pct is not None:
                    progress_bar.progress(int(min(max(pct, 0), 100)))

            try:
                update_status("Preparando datos iniciales", 5)
                work_points = st.session_state.points.copy()
                work_class = (
                    st.session_state.classification.copy()
                    if st.session_state.classification is not None
                    else None
                )
                work_return_number = (
                    st.session_state.return_number.copy()
                    if st.session_state.return_number is not None
                    else None
                )
                work_num_returns = (
                    st.session_state.num_returns.copy()
                    if st.session_state.num_returns is not None
                    else None
                )

                if work_class is not None:
                    proc_state = st.session_state.proc_class_filter
                    enabled = [cls for cls, flag in proc_state.items() if flag]
                    if not enabled:
                        raise ValueError("Selecciona al menos una clase para procesar.")

                    combined_mask = np.isin(work_class, enabled)
                    kept = int(np.count_nonzero(combined_mask))
                    if kept < 3:
                        raise ValueError("No hay puntos suficientes en las clases seleccionadas.")

                    label_list = [class_label(cls) for cls in enabled]
                    update_status(
                        f"Clases activas: {', '.join(label_list)} ({kept:,} puntos)", 12
                    )
                    work_points = work_points[combined_mask]
                    work_class = work_class[combined_mask]
                    if work_return_number is not None:
                        work_return_number = work_return_number[combined_mask]
                    if work_num_returns is not None:
                        work_num_returns = work_num_returns[combined_mask]

                if percentile_low > 0 or percentile_high < 100:
                    update_status("Aplicando corte por percentiles", 20)
                    z_low = np.percentile(work_points[:, 2], percentile_low)
                    z_high = np.percentile(work_points[:, 2], percentile_high)
                    mask = (work_points[:, 2] >= z_low) & (work_points[:, 2] <= z_high)
                    work_points = work_points[mask]
                    if work_class is not None:
                        work_class = work_class[mask]
                    if work_return_number is not None:
                        work_return_number = work_return_number[mask]
                    if work_num_returns is not None:
                        work_num_returns = work_num_returns[mask]
                    update_status(f"Puntos tras percentiles: {work_points.shape[0]:,}", 25)
                    if work_points.shape[0] < 3:
                        raise ValueError("Los percentiles seleccionados eliminan demasiados puntos.")

                if remove_outliers:
                    update_status("Eliminando outliers estadisticos", 35)
                    work_points, work_class, mask = remove_statistical_outliers(
                        work_points, work_class, return_mask=True
                    )
                    if work_return_number is not None:
                        work_return_number = work_return_number[mask]
                    if work_num_returns is not None:
                        work_num_returns = work_num_returns[mask]
                    update_status(f"Puntos tras limpieza: {work_points.shape[0]:,}", 40)
                    if work_points.shape[0] < 3:
                        raise ValueError("No hay puntos suficientes tras eliminar outliers.")

                if processing_mode == "Interpolacion en rejilla":
                    mesh_points, tri = build_mesh(
                        work_points,
                        work_class,
                        mesh_resolution,
                        interpolation_method,
                        (smooth_ground, smooth_vegetation, smooth_buildings),
                        update_status,
                    )
                    triangles = np.asarray(tri.simplices, dtype=np.int32)
                    hull_edges = tri.convex_hull
                else:
                    update_status("Activando modo avanzado de fusion de mallas", 50)
                    mesh_points, triangles = build_mesh_advanced(
                        work_points,
                        work_class,
                        update_status,
                        return_number=work_return_number,
                        num_returns=work_num_returns,
                        sphere_radius=sphere_radius,
                        terrain_resolution=terrain_resolution,
                        max_vertical_error=max_vertical_error,
                        max_slope_deg=max_slope_deg,
                        metaball_grid=metaball_grid,
                        metaball_threshold=metaball_threshold,
                        metaball_max_dim=metaball_max_dim,
                        building_cluster_radius=building_cluster_radius,
                        building_min_points=building_min_points,
                        building_roof_percentile=building_roof_percentile,
                        building_base_percentile=building_base_percentile,
                    )
                    triangles = np.asarray(triangles, dtype=np.int32)
                    hull_edges = None

                points_final = mesh_points
                triangles_final = triangles

                if add_base:
                    update_status("Generando base solida", 85)
                    points_final, triangles_final = add_base_to_mesh(
                        points_final,
                        triangles_final,
                        float(base_thickness),
                        hull_edges=hull_edges,
                    )

                if voxel_simplify > 0.0:
                    update_status("Aplicando simplificacion por voxel", 88)
                    points_final, triangles_final = simplify_mesh_with_voxel_grid(
                        points_final, triangles_final, float(voxel_simplify)
                    )

                preview_requested = st.session_state.get("proc_show_preview", True)
                payload_bytes = points_final.nbytes + triangles_final.nbytes
                max_preview_bytes = 600 * 1024 * 1024  # ~600 MB ~ 500 MB usable payload
                st.subheader("Malla 3D generada")
                if preview_requested and payload_bytes <= max_preview_bytes:
                    update_status("Renderizando malla", 90)
                    fig_mesh = go.Figure(
                        data=[
                            go.Mesh3d(
                                x=points_final[:, 0],
                                y=points_final[:, 1],
                                z=points_final[:, 2],
                                i=triangles_final[:, 0],
                                j=triangles_final[:, 1],
                                k=triangles_final[:, 2],
                                intensity=points_final[:, 2],
                                colorscale="earth",
                                showscale=True,
                                colorbar=dict(title="Altura (m)"),
                                lighting=dict(ambient=0.7, diffuse=0.8, specular=0.3, roughness=0.6),
                            )
                        ]
                    )
                    fig_mesh.update_layout(
                        scene=dict(
                            xaxis_title="X (m)",
                            yaxis_title="Y (m)",
                            zaxis_title="Z (m)",
                            aspectmode="data",
                        ),
                        height=600,
                        margin=dict(l=0, r=0, b=0, t=30),
                    )
                    st.plotly_chart(fig_mesh, use_container_width=True)
                else:
                    update_status("Omitiendo vista previa 3D", 90)
                    size_mb = payload_bytes / (1024 * 1024)
                    if not preview_requested:
                        st.info("La vista previa 3D esta desactivada desde la barra lateral.")
                    else:
                        st.warning(
                            f"La vista previa se ha omitido automaticamente (aprox. {size_mb:.1f} MB de datos, por encima del limite seguro). "
                            "Descarga el STL para revisarlo o reduce resolucion/clases incluidas si deseas visualizarlo aqui."
                        )

                update_status("Exportando STL", 95)
                output_name = f"terrain_mesh_{points_final.shape[0]}_vertices.stl"
                output_path = os.path.join(tempfile.gettempdir(), output_name)
                write_stl(output_path, points_final, triangles_final)

                progress_bar.progress(100)
                update_status("Proceso completado", 100, state="success")
                st.balloons()

                st.subheader("Resumen de resultado")
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Vertices", f"{points_final.shape[0]:,}")
                col_b.metric("Triangulos", f"{triangles_final.shape[0]:,}")
                bbox = points_final.max(axis=0) - points_final.min(axis=0)
                col_c.metric("Dimensiones (m)", f"{bbox[0]:.1f} x {bbox[1]:.1f} x {bbox[2]:.1f}")

                with open(output_path, "rb") as handle:
                    st.download_button(
                        label="Descargar STL",
                        data=handle,
                        file_name=output_name,
                        mime="application/octet-stream",
                        type="primary",
                    )
            except Exception as exc:
                progress_bar.progress(0)
                update_status(f"Error: {exc}", state="error")
                st.exception(exc)

    with col_back:
        if st.button("Volver a clasificacion"):
            st.session_state.step = 2
            st.rerun()

if st.session_state.file_loaded:
    st.sidebar.markdown("---")
    if st.sidebar.button("Cargar nuevo archivo"):
        st.session_state.points = None
        st.session_state.classification = None
        st.session_state.file_loaded = False
        st.session_state.step = 1
        st.rerun()

st.markdown("---")
st.caption("Aplicacion experimental para la generacion rapida de mallas a partir de nubes LiDAR.")
