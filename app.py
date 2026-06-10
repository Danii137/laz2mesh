"""
LAZ to 3D Mesh Converter - Aplicación Principal

Aplicación Streamlit para convertir nubes de puntos LAZ/LAS a mallas 3D imprimibles.

REFACTORIZADO: Arquitectura modular escalable con todos los bugs corregidos.
"""

import os
import glob
import math
import pickle
import re
import inspect
import shutil
import subprocess
import tempfile
import time
import html
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.spatial import cKDTree

# Configuración
from config.settings import (
    MAX_POINTS_LOAD, MAX_PREVIEW_BYTES, MAX_PREVIEW_FACES, DEFAULT_EXCLUDED_CLASSES,
    SCALE_RATIOS, FILAMENT_NOZZLE_MM,
    DEFAULT_SMOOTH_GROUND, DEFAULT_SMOOTH_VEGETATION, DEFAULT_SMOOTH_BUILDINGS,
    DEFAULT_MESH_RESOLUTION, DEFAULT_INTERPOLATION_METHOD,
    DEFAULT_TERRAIN_RESOLUTION, DEFAULT_MAX_VERTICAL_ERROR, DEFAULT_MAX_SLOPE_DEG,
    DEFAULT_SPHERE_RADIUS, DEFAULT_METABALL_GRID, DEFAULT_METABALL_THRESHOLD,
    DEFAULT_METABALL_MAX_DIM, DEFAULT_BUILDING_CLUSTER_RADIUS,
    DEFAULT_BUILDING_MIN_POINTS, DEFAULT_BUILDING_ROOF_PERCENTILE,
    DEFAULT_BUILDING_BASE_PERCENTILE, DEFAULT_BUILDING_PLANE_TOLERANCE,
    DEFAULT_VOXEL_SIMPLIFY, DEFAULT_BASE_THICKNESS,
    DEFAULT_REMOVE_OUTLIERS, DEFAULT_PERCENTILE_LOW, DEFAULT_PERCENTILE_HIGH
)

# Utilidades
from src.utils.file_io import load_laz_file, write_stl, get_temp_output_path
from src.utils.catastro import infer_point_cloud_epsg, fetch_catastro_building_footprints
from src.utils.validators import (
    validate_height_parameters, validate_point_cloud_size,
    validate_class_selection, validate_percentiles, validate_mesh_parameters
)
from src.utils.geometry import remove_statistical_outliers, compute_scale_preset

# Core
from src.core.mesh_builder import (
    build_mesh,
    build_mesh_advanced,
    simplify_mesh_with_voxel_grid,
    add_base_to_mesh,
    count_boundary_edges,
    repair_open_boundaries,
)
from src.core.building_processor import load_building_footprints_geojson

# UI
from src.ui.theme import apply_theme, render_theme_selector, render_stepper
from src.ui.components import (
    get_class_color, class_label, render_class_toggle_buttons,
    render_class_checkboxes, ensure_class_states
)

# ============================================================================
# CONFIGURACIÓN DE LA APLICACIÓN
# ============================================================================

APP_DIR = Path(__file__).resolve().parent
APP_ICON_PATH = APP_DIR / "logo.png"

st.set_page_config(
    page_title="LAZ to 3D Mesh Converter",
    page_icon=str(APP_ICON_PATH) if APP_ICON_PATH.exists() else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inicializar estado
STATE_DEFAULTS = {
    "points": None,
    "classification": None,
    "return_number": None,
    "num_returns": None,
    "file_loaded": False,
    "step": 1,
    "input_filename": "",
    "catastro_footprints": None,
    "catastro_summary": None,
    "catastro_last_error": "",
    "catastro_last_signature": None,
    "proc_dashboard_layout": True,
    "building_height_overrides": {},
    "building_manual_height_enabled": False,
    "app_workflow_mode": "Nube LAZ/LAS",
    "proc_generate_from_header": False,
    "proc_new_project_request": False,
    "ext_mesh_vertices": None,
    "ext_mesh_faces": None,
    "ext_mesh_source_name": "",
    "ext_mesh_report": None,
    "ext_mesh_last_status": "",
    "ext_mesh_cc_path": "",
    "ext_mesh_cc_report": None,
    "ext_raw_preclip_enabled": False,
    "ext_raw_preclip_x_min": None,
    "ext_raw_preclip_x_max": None,
    "ext_raw_preclip_y_min": None,
    "ext_raw_preclip_y_max": None,
    "ext_raw_preclip_report": None,
    "ext_clip_shape": "Rectangulo",
    "ext_clip_center_x": None,
    "ext_clip_center_y": None,
    "ext_clip_width": None,
    "ext_clip_height": None,
    "ext_clip_radius": None,
    "ext_clip_resolution": 1.0,
    "ext_clip_base_thickness": 8.0,
    "ext_clip_vertices": None,
    "ext_clip_faces": None,
    "ext_clip_report": None,
    "ext_clip_cc_path": "",
    "ext_clip_cc_report": None,
    "ext_clip_needs_regen": False,
    "ext_clip_last_apply_msg": "",
    "ext_clip_box_x_min": None,
    "ext_clip_box_x_max": None,
    "ext_clip_box_y_min": None,
    "ext_clip_box_y_max": None,
    "ext_clip_box_z_min": None,
    "ext_clip_box_z_max": None,
    "ext_clip_box_show": True,
    "ext_clip_max_cells": 1400000,
    "ext_clip_preserve_detail": True,
    "laz_cc_path": "",
    "laz_cc_report": None,
    "proc_local_preview_pick_source": "",
}

for key, value in STATE_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Flujo por pasos: avance automatico al completar acciones claras y vuelta manual desde cabecera.
if not st.session_state.file_loaded:
    st.session_state.step = 1
    if st.session_state.get("proc_new_project_request", False):
        st.session_state.proc_new_project_request = False
else:
    current_step = int(st.session_state.get("step", 1))
    if current_step < 2:
        st.session_state.step = 2
    elif current_step > 3:
        st.session_state.step = 3

# ============================================================================
# TEMA
# ============================================================================

st.session_state.ui_theme = "oscuro"
apply_theme("oscuro")

with st.sidebar:
    st.caption("Flujo activo")
    st.radio(
        "Flujo activo",
        ("Nube LAZ/LAS", "Malla externa"),
        key="app_workflow_mode",
        label_visibility="collapsed",
        help=(
            "Nube LAZ/LAS: flujo principal LiDAR. "
            "Malla externa: importa OBJ/PLY/STL (y RDC experimental), repara y exporta STL."
        ),
    )

# Generar presets de escala
SCALE_PRESETS = {label: compute_scale_preset(ratio, FILAMENT_NOZZLE_MM) 
                 for label, ratio in SCALE_RATIOS.items()}


def recommend_easy_quality(point_count: int, scale_ratio_value: float | None) -> str:
    """Pick a sensible one-click quality preset from cloud size and scale."""
    ratio = float(scale_ratio_value) if scale_ratio_value else 1000.0
    if point_count >= 3_000_000:
        return "Rapida"
    if point_count >= 1_000_000:
        return "Equilibrada"
    if ratio <= 500:
        return "Alta"
    if point_count <= 250_000:
        return "Alta"
    return "Equilibrada"


def build_easy_profile(quality: str, point_count: int, scale_ratio_value: float | None) -> dict[str, float | str]:
    """Return multipliers for the simplified processing workflow."""
    ratio = float(scale_ratio_value) if scale_ratio_value else 1000.0
    is_large = point_count >= 1_500_000
    quality_key = str(quality or "Equilibrada").strip().lower()

    profiles: dict[str, dict[str, float | str]] = {
        "rpida": {
            "processing_mode": "Interpolación en rejilla",
            "resolution_mult": 1.65,
            "terrain_mult": 1.55,
            "voxel_mult": 1.55,
            "vertical_error_mult": 1.45,
            "slope_delta": 6.0,
            "sphere_mult": 1.15,
            "metaball_grid_mult": 1.35,
            "metaball_threshold_mult": 1.08,
            "smooth_mult": 0.9,
            "building_plane_mult": 1.2,
            "vegetation_mode": "printable_mass",
        },
        "equilibrada": {
            "processing_mode": "Modo avanzado (detalle máximo)" if not is_large else "Interpolación en rejilla",
            "resolution_mult": 1.0,
            "terrain_mult": 1.0,
            "voxel_mult": 1.0,
            "vertical_error_mult": 1.0,
            "slope_delta": 0.0,
            "sphere_mult": 1.0,
            "metaball_grid_mult": 1.0,
            "metaball_threshold_mult": 1.0,
            "smooth_mult": 1.0,
            "building_plane_mult": 1.0,
            "vegetation_mode": "printable_mass",
        },
        "alta": {
            "processing_mode": "Modo avanzado (detalle máximo)",
            "resolution_mult": 0.72 if ratio <= 1000 else 0.85,
            "terrain_mult": 0.75 if ratio <= 1000 else 0.88,
            "voxel_mult": 0.75,
            "vertical_error_mult": 0.8,
            "slope_delta": -3.0,
            "sphere_mult": 0.9,
            "metaball_grid_mult": 0.85,
            "metaball_threshold_mult": 0.95,
            "smooth_mult": 1.08,
            "building_plane_mult": 0.9,
            "vegetation_mode": "printable_mass",
        },
    }
    return dict(profiles.get(quality_key, profiles["equilibrada"]))


def snap_slider_value(value: float, min_value: float, max_value: float, step: float) -> float:
    """Clamp and snap a numeric value to a slider grid."""
    step_f = float(step) if float(step) > 0 else 1.0
    low = float(min_value)
    high = float(max_value)
    clamped = min(max(float(value), low), high)
    snapped = low + round((clamped - low) / step_f) * step_f
    snapped = min(max(snapped, low), high)
    decimals = 0
    if "." in f"{step_f}":
        decimals = max(0, len(f"{step_f}".rstrip("0").split(".")[-1]))
    return round(snapped, decimals)


def parse_epsg(value: str) -> int | None:
    """Parse EPSG codes from free text like '25830' or 'EPSG:25830'."""
    text = str(value or "").strip().upper().replace("EPSG:", "")
    if not text or not text.isdigit():
        return None
    epsg = int(text)
    return epsg if epsg > 0 else None


def _safe_float_any(value) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, (int, float, np.integer, np.floating)):
            out = float(value)
            return out if np.isfinite(out) else None
        text = str(value).strip().replace(",", ".")
        if not text:
            return None
        out = float(text)
        return out if np.isfinite(out) else None
    except Exception:
        return None


def _polygon_area_xy(ring_xy: np.ndarray) -> float:
    ring = np.asarray(ring_xy, dtype=np.float64)
    if ring.ndim != 2 or ring.shape[0] < 3 or ring.shape[1] < 2:
        return 0.0
    if np.allclose(ring[0, :2], ring[-1, :2]):
        ring = ring[:-1, :2]
    else:
        ring = ring[:, :2]
    if ring.shape[0] < 3:
        return 0.0
    x = ring[:, 0]
    y = ring[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _clip_polygon_xy_halfspace(
    polygon: list[np.ndarray], axis: int, limit: float, keep_greater: bool
) -> list[np.ndarray]:
    if not polygon:
        return []
    out: list[np.ndarray] = []
    prev = polygon[-1]
    prev_inside = bool(prev[axis] >= limit - 1e-9) if keep_greater else bool(prev[axis] <= limit + 1e-9)
    for curr in polygon:
        curr_inside = bool(curr[axis] >= limit - 1e-9) if keep_greater else bool(curr[axis] <= limit + 1e-9)
        if prev_inside and curr_inside:
            out.append(curr)
        elif prev_inside and not curr_inside:
            out.append(_segment_xy_intersection(prev, curr, axis, limit))
        elif (not prev_inside) and curr_inside:
            out.append(_segment_xy_intersection(prev, curr, axis, limit))
            out.append(curr)
        prev = curr
        prev_inside = curr_inside
    return out


def _segment_xy_intersection(p0: np.ndarray, p1: np.ndarray, axis: int, limit: float) -> np.ndarray:
    denom = float(p1[axis] - p0[axis])
    if abs(denom) < 1e-15:
        return p1.copy()
    t = float((limit - p0[axis]) / denom)
    t = float(np.clip(t, 0.0, 1.0))
    return p0 + (p1 - p0) * t


def _clip_ring_to_bbox(ring_xy: np.ndarray, x_min: float, x_max: float, y_min: float, y_max: float) -> np.ndarray:
    ring = np.asarray(ring_xy, dtype=np.float64)
    if ring.ndim != 2 or ring.shape[0] < 3 or ring.shape[1] < 2:
        return np.empty((0, 2), dtype=np.float64)
    if np.allclose(ring[0, :2], ring[-1, :2]):
        ring = ring[:-1, :2]
    else:
        ring = ring[:, :2]
    polygon = [np.array([float(x), float(y)], dtype=np.float64) for x, y in ring]
    polygon = _clip_polygon_xy_halfspace(polygon, 0, float(min(x_min, x_max)), True)
    polygon = _clip_polygon_xy_halfspace(polygon, 0, float(max(x_min, x_max)), False)
    polygon = _clip_polygon_xy_halfspace(polygon, 1, float(min(y_min, y_max)), True)
    polygon = _clip_polygon_xy_halfspace(polygon, 1, float(max(y_min, y_max)), False)
    if not polygon:
        return np.empty((0, 2), dtype=np.float64)
    out = np.asarray(polygon, dtype=np.float64)
    if out.shape[0] >= 2 and np.allclose(out[0], out[-1]):
        out = out[:-1]
    return out


def filter_footprints_to_bbox(
    footprints: list[dict] | None,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    margin: float = 0.0,
    keep_mode: str = "intersects",
    clip_edges: bool = False,
) -> list[dict]:
    if not footprints:
        return []
    x_lo = float(min(x_min, x_max) - margin)
    x_hi = float(max(x_min, x_max) + margin)
    y_lo = float(min(y_min, y_max) - margin)
    y_hi = float(max(y_min, y_max) + margin)
    mode = str(keep_mode or "intersects").strip().lower()
    selected: list[dict] = []
    for item in footprints:
        if not isinstance(item, dict):
            continue
        try:
            poly_xy = np.asarray(item.get("xy"), dtype=np.float64)
        except Exception:
            continue
        if poly_xy.ndim != 2 or poly_xy.shape[0] < 3 or poly_xy.shape[1] < 2:
            continue
        poly_xy = poly_xy[:, :2]
        if np.allclose(poly_xy[0], poly_xy[-1]):
            poly_xy = poly_xy[:-1]
        if poly_xy.shape[0] < 3:
            continue
        poly_eval = poly_xy
        was_clipped = False
        if clip_edges:
            clipped = _clip_ring_to_bbox(poly_xy, x_lo, x_hi, y_lo, y_hi)
            if clipped.shape[0] < 3:
                continue
            original_area = abs(_polygon_area_xy(poly_xy))
            clipped_area = abs(_polygon_area_xy(clipped))
            poly_eval = clipped
            was_clipped = bool(clipped_area + 1e-8 < original_area)
        fx_min = float(np.min(poly_eval[:, 0]))
        fx_max = float(np.max(poly_eval[:, 0]))
        fy_min = float(np.min(poly_eval[:, 1]))
        fy_max = float(np.max(poly_eval[:, 1]))
        intersects = not (fx_max < x_lo or fx_min > x_hi or fy_max < y_lo or fy_min > y_hi)
        centroid_inside = bool(
            x_lo <= float(np.mean(poly_eval[:, 0])) <= x_hi and y_lo <= float(np.mean(poly_eval[:, 1])) <= y_hi
        )
        fully_inside = bool(fx_min >= x_lo and fx_max <= x_hi and fy_min >= y_lo and fy_max <= y_hi)
        if mode in {"inside", "strict", "full"}:
            keep = fully_inside
        elif mode in {"centroid", "center", "centro"}:
            keep = centroid_inside
        else:
            keep = intersects
        if not keep:
            continue
        if clip_edges:
            item_out = dict(item)
            item_out["xy"] = np.asarray(poly_eval, dtype=np.float64)
            item_out["area"] = float(abs(_polygon_area_xy(item_out["xy"])))
            item_out["area_original"] = float(abs(_polygon_area_xy(poly_xy)))
            item_out["_clip_applied"] = bool(was_clipped)
            selected.append(item_out)
        else:
            selected.append(item)
    return selected


def diagnose_footprints_vs_laz_class6(
    footprints: list[dict] | None,
    points: np.ndarray,
    classification: np.ndarray | None,
    *,
    density_hint: float = 1.0,
    outside_area_threshold: float = 30.0,
    max_class6_points: int = 180_000,
) -> dict:
    """Quick diagnostic to explain missing/incomplete footprint reconstruction."""
    if not footprints:
        return {"ok": False, "message": "No hay huellas para diagnosticar."}
    if classification is None or points is None:
        return {"ok": False, "message": "No hay clasificacin disponible."}

    pts = np.asarray(points, dtype=np.float64)
    cls = np.asarray(classification)
    if pts.ndim != 2 or pts.shape[0] == 0 or cls.shape[0] != pts.shape[0]:
        return {"ok": False, "message": "Datos de nube/clasificacin no vlidos."}

    mask6 = cls == 6
    if int(mask6.sum()) == 0:
        return {"ok": False, "message": "No hay puntos clase 6 en la nube."}

    roof_xyz = pts[mask6, :3].astype(np.float64, copy=False)
    if roof_xyz.shape[0] > int(max_class6_points):
        rng = np.random.default_rng(42)
        pick = rng.choice(roof_xyz.shape[0], size=int(max_class6_points), replace=False)
        roof_xyz = roof_xyz[pick]

    density = max(float(density_hint), 0.15)
    nominal_spacing = float(np.sqrt(1.0 / density))
    near_margin = max(1.5, nominal_spacing * 2.2)
    tree = cKDTree(roof_xyz[:, :2])

    rows: list[dict] = []
    total = 0
    with_support = 0
    likely_sparse = 0
    likely_outdated = 0

    for idx, item in enumerate(footprints, start=1):
        if not isinstance(item, dict):
            continue
        try:
            ring = np.asarray(item.get("xy"), dtype=np.float64)
        except Exception:
            continue
        if ring.ndim != 2 or ring.shape[0] < 3 or ring.shape[1] < 2:
            continue
        ring = ring[:, :2]
        if np.allclose(ring[0], ring[-1]):
            ring = ring[:-1]
        if ring.shape[0] < 3:
            continue

        total += 1
        area = float(abs(_polygon_area_xy(ring)))
        centroid = np.mean(ring, axis=0)
        radius = float(np.max(np.linalg.norm(ring - centroid[None, :], axis=1)))
        cand_idx = tree.query_ball_point(centroid, r=radius + near_margin)
        if not cand_idx:
            rows.append(
                {
                    "idx": idx,
                    "area_m": round(area, 1),
                    "pts_in": 0,
                    "pts_out_near": 0,
                    "out_area_est_m": 0.0,
                    "min_pts_req": int(np.clip(np.ceil(area * density * 0.08), 6.0, 18.0)),
                    "status": "sin_soporte",
                }
            )
            likely_sparse += 1
            continue

        cand = roof_xyz[np.asarray(cand_idx, dtype=np.int64)]
        bbox_min = np.min(ring, axis=0) - near_margin
        bbox_max = np.max(ring, axis=0) + near_margin
        bbox_mask = (
            (cand[:, 0] >= bbox_min[0])
            & (cand[:, 0] <= bbox_max[0])
            & (cand[:, 1] >= bbox_min[1])
            & (cand[:, 1] <= bbox_max[1])
        )
        cand = cand[bbox_mask]
        if cand.shape[0] == 0:
            likely_sparse += 1
            continue

        inside_mask = np.fromiter(
            (_point_in_polygon_inclusive(pt[:2], ring, tol=max(0.10, nominal_spacing * 0.22)) for pt in cand),
            dtype=bool,
            count=cand.shape[0],
        )
        pts_in = int(np.sum(inside_mask))
        pts_out = int(cand.shape[0] - pts_in)
        min_support = int(np.clip(np.ceil(area * density * 0.08), 6.0, 18.0))
        out_area_est = float(pts_out / density)
        status = "ok"
        if pts_in < min_support:
            status = "soporte_bajo"
            likely_sparse += 1
        if out_area_est > float(outside_area_threshold):
            status = "catastro_desactualizado"
            likely_outdated += 1
        if pts_in >= min_support:
            with_support += 1
        rows.append(
            {
                "idx": idx,
                "area_m": round(area, 1),
                "pts_in": pts_in,
                "pts_out_near": pts_out,
                "out_area_est_m": round(out_area_est, 1),
                "min_pts_req": min_support,
                "status": status,
            }
        )

    rows_sorted = sorted(
        rows,
        key=lambda row: (float(row.get("out_area_est_m", 0.0)), -float(row.get("pts_in", 0))),
        reverse=True,
    )
    return {
        "ok": True,
        "total": int(total),
        "with_support": int(with_support),
        "likely_sparse": int(likely_sparse),
        "likely_outdated": int(likely_outdated),
        "rows": rows_sorted[:40],
    }


def _manual_height_key_for_footprint(item: dict, idx: int) -> str:
    props = item.get("properties") if isinstance(item, dict) and isinstance(item.get("properties"), dict) else {}
    for key in ("gml_id", "id", "localId", "localid", "reference", "ref", "building_id", "entity_id"):
        raw = item.get(key) if isinstance(item, dict) else None
        if raw in (None, ""):
            raw = props.get(key)
        if raw not in (None, ""):
            return f"fp_ref_{str(raw).strip()}"
    try:
        ring = np.asarray(item.get("xy"), dtype=np.float64)
    except Exception:
        ring = np.empty((0, 2), dtype=np.float64)
    if ring.ndim == 2 and ring.shape[0] > 0:
        cx = float(np.mean(ring[:, 0]))
        cy = float(np.mean(ring[:, 1]))
        return f"fp_xy_{cx:.3f}_{cy:.3f}"
    return f"fp_{int(idx)}"


def _footprint_ref_label(item: dict, idx: int) -> str:
    props = item.get("properties") if isinstance(item, dict) and isinstance(item.get("properties"), dict) else {}
    for key in ("gml_id", "id", "localId", "localid", "reference", "ref", "building_id", "entity_id"):
        raw = item.get(key) if isinstance(item, dict) else None
        if raw in (None, ""):
            raw = props.get(key)
        if raw not in (None, ""):
            return str(raw)
    return f"Edificio {int(idx)}"


def _estimate_footprint_height(
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
    props = properties if isinstance(properties, dict) else {}
    manual = _safe_float_any(props.get("manual_height_m"))
    if manual is not None:
        return float(np.clip(manual, float(min_height), float(max_height)))
    mode = str(height_mode or "fixed").strip().lower()
    if mode in {"fixed", "fija", "constante"}:
        h = _safe_float_any(fixed_height)
    elif mode in {"attribute", "altura", "height"}:
        h = _safe_float_any(props.get(height_attribute)) if height_attribute else None
        if h is None:
            h = _safe_float_any(fallback_height)
    elif mode in {"floors", "plantas"}:
        floors_val = _safe_float_any(props.get(floors_attribute)) if floors_attribute else None
        if floors_val is None:
            h = _safe_float_any(fallback_height)
        else:
            h = floors_val * max(float(floor_height), 1.0)
    else:
        h = _safe_float_any(fallback_height)
    if h is None or not np.isfinite(h):
        h = float(fallback_height)
    return float(np.clip(float(h), float(min_height), float(max_height)))


def apply_manual_height_overrides_to_footprints(footprints: list[dict] | None, overrides: dict) -> list[dict]:
    if not footprints:
        return []
    raw = overrides if isinstance(overrides, dict) else {}
    clean_overrides: dict[str, float] = {}
    for key, value in raw.items():
        parsed = _safe_float_any(value)
        if parsed is not None:
            clean_overrides[str(key)] = float(parsed)
    out_items: list[dict] = []
    for idx, item in enumerate(footprints, start=1):
        if not isinstance(item, dict):
            continue
        item_out = dict(item)
        props = item_out.get("properties")
        props_out = dict(props) if isinstance(props, dict) else {}
        fp_key = _manual_height_key_for_footprint(item_out, idx)
        if fp_key in clean_overrides:
            props_out["manual_height_m"] = float(clean_overrides[fp_key])
        else:
            props_out.pop("manual_height_m", None)
        item_out["properties"] = props_out
        item_out["_manual_key"] = fp_key
        out_items.append(item_out)
    return out_items


def noop_status_callback(*_args, **_kwargs):
    """No-op callback used by local preview builders."""
    return None


def call_build_mesh_advanced_compat(points, classification, status_callback, **kwargs):
    """Filter kwargs unsupported by the loaded build_mesh_advanced implementation."""
    try:
        accepted = inspect.signature(build_mesh_advanced).parameters
        kwargs = {key: value for key, value in kwargs.items() if key in accepted}
    except Exception:
        pass
    return build_mesh_advanced(points, classification, status_callback, **kwargs)


@st.cache_data(show_spinner=False)
def parse_footprints_cached(payload: bytes) -> tuple[list[dict], dict]:
    """Cache parser for uploaded GeoJSON footprints."""
    return load_building_footprints_geojson(
        payload,
        min_area=0.0,
        simplify_tolerance=0.0,
        max_vertices=3000,
    )


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_catastro_cached(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    source_epsg: int,
    query_candidates: tuple[int, ...],
    type_name: str,
    max_features: int,
) -> tuple[list[dict], dict]:
    """Cache Catastro WFS downloads for repeated tuning runs."""
    return fetch_catastro_building_footprints(
        x_min=float(x_min),
        x_max=float(x_max),
        y_min=float(y_min),
        y_max=float(y_max),
        source_epsg=int(source_epsg),
        query_candidates=list(query_candidates),
        type_name=str(type_name),
        max_features=int(max_features),
    )


def _load_mesh_with_open3d(o3d, path: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Try loading one mesh file with Open3D and return vertices/faces when valid."""
    if o3d is None:
        return None
    try:
        try:
            mesh = o3d.io.read_triangle_mesh(path, enable_post_processing=True)
        except TypeError:
            mesh = o3d.io.read_triangle_mesh(path)
    except Exception:
        return None

    if mesh is None:
        return None
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[0] < 3:
        return None
    if triangles.ndim != 2 or triangles.shape[0] < 1:
        return None
    return vertices, triangles


def _load_mesh_with_trimesh(path: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Try loading one mesh file with trimesh and return vertices/faces when valid."""
    try:
        import trimesh
    except Exception:
        return None

    try:
        mesh_obj = trimesh.load(path, force="mesh", process=False)
    except Exception:
        return None

    mesh = None
    if isinstance(mesh_obj, trimesh.Scene):
        geometries = [
            geom
            for geom in mesh_obj.geometry.values()
            if isinstance(geom, trimesh.Trimesh)
            and geom.vertices is not None
            and geom.faces is not None
            and len(geom.vertices) >= 3
            and len(geom.faces) >= 1
        ]
        if geometries:
            try:
                mesh = trimesh.util.concatenate(geometries)
            except Exception:
                mesh = geometries[0]
    elif isinstance(mesh_obj, trimesh.Trimesh):
        mesh = mesh_obj

    if mesh is None:
        return None

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.faces, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[0] < 3:
        return None
    if triangles.ndim != 2 or triangles.shape[0] < 1:
        return None
    return vertices, triangles


def _load_mesh_file(path: str, *, o3d=None) -> tuple[np.ndarray, np.ndarray] | None:
    """Load a mesh with Open3D (preferred) or trimesh fallback."""
    loaded = _load_mesh_with_open3d(o3d, path)
    if loaded is not None:
        return loaded
    return _load_mesh_with_trimesh(path)


def _find_renderdoccmd() -> str | None:
    """Locate renderdoccmd executable if available."""
    env_path = os.environ.get("RENDERDOCCMD_PATH", "").strip()
    candidates = [
        env_path,
        shutil.which("renderdoccmd") or "",
        r"C:\Program Files\RenderDoc\renderdoccmd.exe",
        r"C:\Program Files (x86)\RenderDoc\renderdoccmd.exe",
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def _find_cloudcompare_exe() -> str | None:
    """Locate CloudCompare executable if available."""
    env_path = os.environ.get("CLOUDCOMPARE_PATH", "").strip()
    candidates = [
        env_path,
        shutil.which("CloudCompare") or "",
        shutil.which("CloudCompare.exe") or "",
        r"C:\Program Files\CloudCompare\CloudCompare.exe",
        r"C:\Program Files (x86)\CloudCompare\CloudCompare.exe",
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def _cloudcompare_roundtrip_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    timeout_s: int = 420,
    merge_meshes: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Safe fallback for CloudCompare post-processing.
    The app uses this hook to "vlidate/fix" STL before download.
    """
    del timeout_s, merge_meshes
    started = time.time()

    verts = np.asarray(vertices, dtype=np.float64)
    tris = np.asarray(faces, dtype=np.int32)
    if verts.ndim != 2 or verts.shape[0] < 3 or verts.shape[1] < 3:
        raise ValueError("Malla invlida: vertices insuficientes.")
    if tris.ndim != 2 or tris.shape[0] < 1 or tris.shape[1] < 3:
        raise ValueError("Malla invlida: caras insuficientes.")

    open_before = int(count_boundary_edges(verts, tris))
    repaired_verts, repaired_faces = repair_open_boundaries(verts, tris)
    open_after = int(count_boundary_edges(repaired_verts, repaired_faces))
    elapsed = float(time.time() - started)

    report = {
        "tool": "python_fallback",
        "open_before": open_before,
        "open_after": open_after,
        "elapsed_s": elapsed,
    }
    return (
        np.asarray(repaired_verts, dtype=np.float64),
        np.asarray(repaired_faces, dtype=np.int32),
        report,
    )


def _find_blender_exe() -> str | None:
    """Locate Blender executable for RDC fallback conversion."""
    env_path = os.environ.get("BLENDER_PATH", "").strip()
    candidates = [
        env_path,
        shutil.which("blender") or "",
        r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.3\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.1\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender\blender.exe",
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def _find_maps_models_importer_dir() -> str | None:
    """Locate MapsModelsImporter addon folder in Blender user scripts."""
    appdata = os.environ.get("APPDATA", "").strip()
    if not appdata:
        return None
    base = Path(appdata) / "Blender Foundation" / "Blender"
    if not base.is_dir():
        return None

    def _version_key(name: str) -> tuple[int, int, str]:
        try:
            parts = str(name).split(".")
            major = int(parts[0]) if parts else 0
            minor = int(parts[1]) if len(parts) > 1 else 0
            return (major, minor, str(name))
        except Exception:
            return (0, 0, str(name))

    candidates: list[tuple[tuple[int, int, str], str]] = []
    for child in base.iterdir():
        addon_dir = child / "scripts" / "addons" / "MapsModelsImporter"
        if addon_dir.is_dir():
            candidates.append((_version_key(child.name), str(addon_dir)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _find_blender_python_exe(blender_exe: str) -> str | None:
    """Locate Blender bundled python executable."""
    root = Path(blender_exe).resolve().parent
    candidates = [
        root / "python" / "bin" / "python.exe",
    ]
    candidates.extend(root.glob("*\\python\\bin\\python.exe"))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def _numpy_load_maps_bin(path: str) -> np.ndarray:
    """Load custom numpy binary format used by MapsModelsImporter."""
    with open(path, "rb") as handle:
        dim = np.fromfile(handle, dtype=np.int32, count=1)
        if dim.size == 0:
            raise ValueError("Cabecera de array vacia.")
        ndim = int(dim[0])
        shape = np.fromfile(handle, dtype=np.int32, count=ndim)
        if shape.size != ndim:
            raise ValueError("Shape incompleta en binario.")
        dtype_code = handle.read(2).decode("ascii", errors="ignore")
        if not dtype_code:
            raise ValueError("Tipo de dato no disponible en binario.")
        dtype = np.dtype(dtype_code)
        data = np.fromfile(handle, dtype=dtype)
    try:
        return data.reshape(tuple(int(v) for v in shape))
    except Exception as exc:
        raise ValueError(f"No se pudo reconstruir shape {tuple(shape.tolist())}: {exc}") from exc


def _matrix_from_constants(data: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
    """Build 4x4 matrix from flat shader constants using addon's convention."""
    arr = np.asarray(data, dtype=np.float64).reshape((4, 4))
    return arr.T


def _rotation_y_matrix(angle_rad: float) -> np.ndarray:
    """Create 4x4 rotation matrix around Y axis."""
    c = float(math.cos(angle_rad))
    s = float(math.sin(angle_rad))
    return np.array(
        [
            [c, 0.0, s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-s, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _extract_uniforms_maps(
    constants: dict,
    ref_matrix: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Port of MapsModelsImporter uniform extraction to numpy."""
    globals_uniforms = constants.get("$Globals", {})
    matrix = None
    post_matrix = None
    uv_offset_scale = None

    if "_w" in globals_uniforms and "_s" in globals_uniforms:
        ou, ov, su, sv = [float(v) for v in globals_uniforms["_w"]]
        ov -= 1.0 / sv
        sv = -sv
        uv_offset_scale = np.array([ou, ov, su, sv], dtype=np.float64)
        matrix = _matrix_from_constants(globals_uniforms["_s"])
    elif (
        "webgl_fa7f624db8ab37d1" in globals_uniforms
        and "webgl_3c7b7f37a9bd4c1d" in globals_uniforms
    ):
        uv_offset_scale = np.array(
            globals_uniforms["webgl_fa7f624db8ab37d1"], dtype=np.float64
        )
        matrix = _matrix_from_constants(globals_uniforms["webgl_3c7b7f37a9bd4c1d"])
    elif (
        "_webgl_fa7f624db8ab37d1" in globals_uniforms
        and "_webgl_3c7b7f37a9bd4c1d" in globals_uniforms
    ):
        ou, ov, su, sv = [float(v) for v in globals_uniforms["_webgl_fa7f624db8ab37d1"]]
        ov -= 1.0 / sv
        sv = -sv
        uv_offset_scale = np.array([ou, ov, su, sv], dtype=np.float64)
        matrix = _matrix_from_constants(globals_uniforms["_webgl_3c7b7f37a9bd4c1d"])
    elif "_uMeshToWorldMatrix" in globals_uniforms:
        uv_offset_scale = np.array([0.0, -1.0, 1.0, -1.0], dtype=np.float64)
        matrix = _matrix_from_constants(globals_uniforms["_uMeshToWorldMatrix"])
        matrix[3] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    elif "_uMV" in globals_uniforms:
        u_params = _matrix_from_constants(globals_uniforms["_uParams"])
        uv_offset_scale = np.array(
            [
                float(u_params[2][2] / u_params[0][2]),
                float((u_params[3][2] - 1.0) / u_params[1][2]),
                float(u_params[0][2]),
                float(-u_params[1][2]),
            ],
            dtype=np.float64,
        )
        matrix = _matrix_from_constants(globals_uniforms["_uMV"])
    else:
        if ref_matrix is None:
            return None, None, ref_matrix
        return None, None, ref_matrix

    if matrix is None or uv_offset_scale is None:
        return None, None, ref_matrix

    if ref_matrix is None:
        try:
            ref_matrix = _rotation_y_matrix(-math.pi / 2.0) @ np.linalg.inv(matrix)
        except Exception:
            return None, None, None
    matrix = ref_matrix @ matrix
    if post_matrix is not None:
        matrix = post_matrix @ matrix
    return uv_offset_scale, matrix, ref_matrix


def _triangles_from_indices(indices: np.ndarray, topology: str) -> np.ndarray:
    """Build triangle index buffer from topology and index array."""
    idx = np.asarray(indices).reshape(-1).astype(np.int64, copy=False)
    if idx.size < 3:
        return np.empty((0, 3), dtype=np.int32)

    topo = str(topology or "").upper()
    if topo == "TRIANGLE_STRIP":
        tris: list[list[int]] = []
        for i in range(max(idx.size - 2, 0)):
            if i % 2 == 0:
                tri = [int(idx[i]), int(idx[i + 1]), int(idx[i + 2])]
            else:
                tri = [int(idx[i]), int(idx[i + 2]), int(idx[i + 1])]
            if tri[0] != tri[1] and tri[1] != tri[2] and tri[0] != tri[2]:
                tris.append(tri)
        if not tris:
            return np.empty((0, 3), dtype=np.int32)
        return np.asarray(tris, dtype=np.int32)

    usable = (idx.size // 3) * 3
    if usable < 3:
        return np.empty((0, 3), dtype=np.int32)
    return idx[:usable].reshape((-1, 3)).astype(np.int32, copy=False)


def _mapycz_positions_to_vertices(positions: np.ndarray, globals_uniforms: dict) -> np.ndarray:
    """Mapy.cz vertex transform ported from MapsModelsImporter."""
    u_params_se = _matrix_from_constants(globals_uniforms["_uParamsSE"])
    raw = np.asarray(positions, dtype=np.float64)
    out = np.zeros((raw.shape[0], 3), dtype=np.float64)
    for i in range(raw.shape[0]):
        v0 = raw[i, :3]
        r1 = np.zeros((3,), dtype=np.float64)
        r2 = np.zeros((3,), dtype=np.float64)
        r1[0] = v0[0] * u_params_se[3][0] + u_params_se[0][0]
        r1[1] = v0[1] * u_params_se[0][1] + u_params_se[1][0]
        r1[2] = (v0[2] * u_params_se[1][1] + u_params_se[2][0]) * u_params_se[3][3]
        r0_1 = float(np.linalg.norm(r1))
        r0_2 = r0_1 + 0.0001
        r0_1 = r0_1 - float(u_params_se[2][3])
        r0_2 = 1.0 / r0_2
        r1 *= r0_2
        r0_2 = min(max(r0_1, float(u_params_se[1][2])), float(u_params_se[3][2]))
        r0_2 = (r0_2 - float(u_params_se[1][2])) * float(u_params_se[0][3]) * float(u_params_se[1][3]) + float(u_params_se[2][2])
        r0_1 = r0_1 * r0_2 - r0_1
        r2[0] = v0[0] * u_params_se[3][0]
        r2[1] = v0[1] * u_params_se[0][1]
        r2[2] = v0[2] * u_params_se[1][1]
        r2 += r1 * r0_1
        out[i] = r2
    return out


def _extract_rdc_with_maps_importer(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Convert RDC to merged mesh using MapsModelsImporter extractor (no Blender UI)."""
    blender_exe = _find_blender_exe()
    if not blender_exe:
        raise ValueError("No se encontr Blender para extraer geometra desde RDC.")

    addon_dir = _find_maps_models_importer_dir()
    if not addon_dir:
        raise ValueError("No se encontr el addon MapsModelsImporter.")

    extractor_script = os.path.join(addon_dir, "google_maps_rd.py")
    if not os.path.isfile(extractor_script):
        raise ValueError("No se encontr google_maps_rd.py en MapsModelsImporter.")

    bin_dir = os.path.join(addon_dir, "bin", "win64")
    if not os.path.isdir(bin_dir):
        raise ValueError("No se encontr carpeta bin/win64 del addon MapsModelsImporter.")

    blender_python = _find_blender_python_exe(blender_exe)
    if not blender_python or not os.path.isfile(blender_python):
        raise ValueError("No se encontr Python interno de Blender.")

    with tempfile.TemporaryDirectory(prefix="rdc_maps_importer_") as tmpdir:
        rdc_path = os.path.join(tmpdir, "capture.rdc")
        out_prefix = os.path.join(tmpdir, "capture-")
        with open(rdc_path, "wb") as handle:
            handle.write(payload)

        env = os.environ.copy()
        python_home = str(Path(blender_python).resolve().parents[1])
        env["PYTHONHOME"] = python_home
        env["PYTHONPATH"] = (env.get("PYTHONPATH", "") + os.pathsep + bin_dir).strip(os.pathsep)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PATH"] = env.get("PATH", "") + os.pathsep + os.path.join(python_home, "bin") + os.pathsep + bin_dir

        cmd = [blender_python, extractor_script, rdc_path, out_prefix, "-1"]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
            timeout=900,
            env=env,
        )
        if result.returncode != 0:
            details = (result.stderr or "").strip() or (result.stdout or "").strip()
            raise ValueError(
                "MapsModelsImporter no pudo extraer el RDC. "
                f"Detalle: {details or 'sin detalle'}"
            )

        constants_files = sorted(glob.glob(f"{out_prefix}*-constants.bin"))
        if not constants_files:
            raise ValueError("No se extrajeron drawcalls de malla desde el RDC.")

        all_vertices: list[np.ndarray] = []
        all_faces: list[np.ndarray] = []
        vertex_offset = 0
        ref_matrix = None
        global_scale = 1.0 / 256.0

        for const_file in constants_files:
            match = re.search(r"(\d+)-constants\.bin$", const_file.replace("\\", "/"))
            if not match:
                continue
            draw_id = int(match.group(1))
            indices_file = f"{out_prefix}{draw_id:05d}-indices.bin"
            positions_file = f"{out_prefix}{draw_id:05d}-positions.bin"
            if not (os.path.isfile(indices_file) and os.path.isfile(positions_file)):
                continue

            try:
                with open(const_file, "rb") as handle:
                    constants = pickle.load(handle)
                indices = _numpy_load_maps_bin(indices_file)
                positions = _numpy_load_maps_bin(positions_file)
            except Exception:
                continue

            draw_meta = constants.get("DrawCall", {})
            topology = str(draw_meta.get("topology", "TRIANGLES"))
            draw_type = str(draw_meta.get("type", "Google Maps"))
            globals_uniforms = constants.get("$Globals", {})

            _, matrix, ref_matrix = _extract_uniforms_maps(constants, ref_matrix)
            if matrix is None:
                continue

            raw_positions = np.asarray(positions, dtype=np.float64)
            if raw_positions.ndim != 2 or raw_positions.shape[0] < 3 or raw_positions.shape[1] < 3:
                continue

            if draw_type == "Google Maps":
                verts = raw_positions[:, :3] * 256.0
            elif draw_type == "Mapy CZ" and "_uParamsSE" in globals_uniforms:
                verts = _mapycz_positions_to_vertices(raw_positions, globals_uniforms)
            else:
                verts = raw_positions[:, :3]

            faces_local = _triangles_from_indices(indices, topology)
            if faces_local.shape[0] == 0:
                continue
            valid_face_mask = (
                (faces_local[:, 0] >= 0)
                & (faces_local[:, 1] >= 0)
                & (faces_local[:, 2] >= 0)
                & (faces_local[:, 0] < verts.shape[0])
                & (faces_local[:, 1] < verts.shape[0])
                & (faces_local[:, 2] < verts.shape[0])
            )
            faces_local = faces_local[valid_face_mask]
            if faces_local.shape[0] == 0:
                continue

            transform = matrix * float(global_scale)
            verts_h = np.concatenate(
                [verts, np.ones((verts.shape[0], 1), dtype=np.float64)],
                axis=1,
            )
            verts_world = (transform @ verts_h.T).T[:, :3]

            all_vertices.append(verts_world)
            all_faces.append(faces_local.astype(np.int32, copy=False) + int(vertex_offset))
            vertex_offset += verts_world.shape[0]

        if not all_vertices or not all_faces:
            raise ValueError("No se pudo reconstruir geometra vlida desde el RDC.")

        merged_vertices = np.vstack(all_vertices).astype(np.float64, copy=False)
        merged_faces = np.vstack(all_faces).astype(np.int32, copy=False)
        return merged_vertices, merged_faces


def _parse_renderdoc_section_names(raw_text: str) -> list[str]:
    """Parse section names from `renderdoccmd extract --list-sections` output."""
    names: list[str] = []
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lower_line = line.lower()
        if lower_line.startswith("usage:") or lower_line.startswith("options:") or lower_line.startswith("errors:"):
            continue
        if line.startswith("--"):
            continue

        token = line
        if ":" in token and not re.match(r"^[A-Za-z]:\\", token):
            token = token.split(":", 1)[0].strip()
        token = re.split(r"\s+\(", token, 1)[0].strip()
        if token and token not in names:
            names.append(token)
    return names


def _infer_extracted_mesh_ext(section_name: str, output_path: str) -> str:
    """Guess a useful extension for extracted section payload."""
    lowered = (section_name or "").lower()
    for ext in (".obj", ".ply", ".stl", ".glb", ".gltf"):
        if ext in lowered:
            return ext

    try:
        with open(output_path, "rb") as handle:
            head = handle.read(1024)
    except Exception:
        return ".bin"

    head_l = head.lower()
    if head_l.startswith(b"ply"):
        return ".ply"
    if head_l.startswith(b"solid"):
        return ".stl"
    if b"v " in head_l and b"f " in head_l:
        return ".obj"
    return ".bin"


def _load_mesh_from_rdc_with_renderdoc(payload: bytes, o3d=None) -> tuple[np.ndarray, np.ndarray]:
    """Extract mesh-like sections from an RDC capture and load as triangle mesh."""
    renderdoccmd = _find_renderdoccmd()
    if not renderdoccmd:
        raise ValueError(
            "No se encontr 'renderdoccmd'. Instala RenderDoc o exporta la malla a OBJ/PLY/STL."
        )

    with tempfile.TemporaryDirectory(prefix="rdc_import_") as tmpdir:
        rdc_path = os.path.join(tmpdir, "capture.rdc")
        with open(rdc_path, "wb") as handle:
            handle.write(payload)

        # RenderDoc versions differ: some require --section/--file even with --list-sections.
        list_dummy = os.path.join(tmpdir, "_list_dummy.bin")
        list_cmd = [
            renderdoccmd,
            "extract",
            "--section",
            "__dummy__",
            "--file",
            list_dummy,
            "--list-sections",
            rdc_path,
        ]
        list_result = subprocess.run(
            list_cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
        )
        if list_result.returncode != 0:
            details = (list_result.stderr or list_result.stdout or "").strip()
            raise ValueError(
                "No se pudieron listar secciones del RDC. "
                f"Detalle: {details or 'sin detalle'}"
            )

        section_names = _parse_renderdoc_section_names(list_result.stdout or "")
        if not section_names:
            raise ValueError(
                "El archivo RDC no contiene secciones accesibles para extraer geometra."
            )

        scored: list[tuple[int, str]] = []
        for section in section_names:
            low = section.lower()
            score = 0
            if any(ext in low for ext in (".obj", ".ply", ".stl", ".glb", ".gltf")):
                score += 10
            if "mesh" in low or "geom" in low or "vertex" in low:
                score += 4
            scored.append((score, section))
        scored.sort(key=lambda item: (-item[0], item[1]))
        candidates = [name for _, name in scored[:50]]

        for index, section_name in enumerate(candidates):
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", section_name)[:80] or f"section_{index}"
            extracted_path = os.path.join(tmpdir, f"{index:03d}_{safe_name}.bin")
            extract_cmd = [
                renderdoccmd,
                "extract",
                "--section",
                section_name,
                "--file",
                extracted_path,
                rdc_path,
            ]
            extract_result = subprocess.run(
                extract_cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                check=False,
            )
            if extract_result.returncode != 0 or not os.path.isfile(extracted_path):
                continue
            if os.path.getsize(extracted_path) <= 0:
                continue

            guessed_ext = _infer_extracted_mesh_ext(section_name, extracted_path)
            tried_paths: list[str] = []
            if guessed_ext != ".bin":
                renamed_path = os.path.join(tmpdir, f"{index:03d}_{safe_name}{guessed_ext}")
                try:
                    shutil.copyfile(extracted_path, renamed_path)
                    tried_paths.append(renamed_path)
                except Exception:
                    pass
            tried_paths.append(extracted_path)

            for candidate_path in tried_paths:
                loaded = _load_mesh_file(candidate_path, o3d=o3d)
                if loaded is not None:
                    return loaded

        raise ValueError(
            "No se encontr una malla triangulada dentro del RDC. "
            "Exporta desde RenderDoc/Blender a OBJ/PLY/STL y subelo en esta pestana."
        )


def _load_mesh_from_rdc_with_blender(payload: bytes, o3d=None) -> tuple[np.ndarray, np.ndarray]:
    """Use Blender + MapsModelsImporter addon to convert RDC to STL and load it."""
    blender_exe = _find_blender_exe()
    if not blender_exe:
        raise ValueError(
            "No se encontr Blender. Define BLENDER_PATH o instala Blender para importar RDC."
        )

    with tempfile.TemporaryDirectory(prefix="rdc_blender_") as tmpdir:
        rdc_path = os.path.join(tmpdir, "capture.rdc")
        out_stl = os.path.join(tmpdir, "capture_from_rdc.stl")
        script_path = os.path.join(tmpdir, "_rdc_to_stl.py")
        with open(rdc_path, "wb") as handle:
            handle.write(payload)

        script_text = r'''
import os
import sys
import traceback
import bpy


def fail(msg: str, code: int = 2) -> None:
    print(f"RDC_IMPORT_ERROR: {msg}")
    sys.exit(code)


argv = sys.argv
if "--" not in argv:
    fail("faltan argumentos")
args = argv[argv.index("--") + 1 :]
if len(args) < 2:
    fail("faltan argumentos de entrada/salida")

rdc_path = args[0]
out_stl = args[1]
if not os.path.isfile(rdc_path):
    fail(f"RDC no existe: {rdc_path}")

try:
    if "MapsModelsImporter" not in bpy.context.preferences.addons:
        bpy.ops.preferences.addon_enable(module="MapsModelsImporter")
except Exception as exc:
    fail(f"No se pudo activar addon MapsModelsImporter: {exc}")

try:
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
except Exception:
    pass

for mesh in list(bpy.data.meshes):
    if mesh.users == 0:
        bpy.data.meshes.remove(mesh)

try:
    from MapsModelsImporter.google_maps import importCapture, MapsModelsImportError
    from MapsModelsImporter.preferences import getPreferences
except Exception:
    fail("No se pudo importar la API interna de MapsModelsImporter.\n" + traceback.format_exc())

try:
    pref = getPreferences(bpy.context)
    if hasattr(pref, "tmp_dir"):
        pref.tmp_dir = os.path.dirname(out_stl)
    if hasattr(pref, "debug_info"):
        pref.debug_info = False
except Exception:
    pref = None

try:
    importCapture(
        bpy.context,
        rdc_path,
        -1,
        False,
        pref if pref is not None else getPreferences(bpy.context),
    )
except MapsModelsImportError as exc:
    fail(f"Fallo al importar RDC con addon: {exc}")
except Exception:
    fail("Fallo interno al importar RDC con addon.\n" + traceback.format_exc())

mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
if not mesh_objects:
    fail("No se generaron objetos de malla desde el RDC")

for obj in bpy.context.selected_objects:
    obj.select_set(False)
for obj in mesh_objects:
    obj.select_set(True)
bpy.context.view_layer.objects.active = mesh_objects[0]

exported = False
try:
    bpy.ops.wm.stl_export(
        filepath=out_stl,
        export_selected_objects=True,
        ascii_format=False,
    )
    exported = True
except Exception:
    try:
        bpy.ops.export_mesh.stl(filepath=out_stl, use_selection=True, ascii=False)
        exported = True
    except Exception as exc:
        fail(f"No se pudo exportar STL: {exc}")

if not exported:
    fail("No se pudo exportar STL")
if (not os.path.isfile(out_stl)) or os.path.getsize(out_stl) <= 0:
    fail("STL vacio tras conversion")

print(f"RDC_IMPORT_OK: {out_stl}")
sys.exit(0)
'''
        with open(script_path, "w", encoding="utf-8") as handle:
            handle.write(script_text)

        cmd = [
            blender_exe,
            "-b",
            "--python",
            script_path,
            "--",
            rdc_path,
            out_stl,
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
            timeout=900,
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            details = stderr or stdout
            if stdout:
                lines = stdout.splitlines()
                marker_idx = -1
                for idx, line in enumerate(lines):
                    if "RDC_IMPORT_ERROR:" in line:
                        marker_idx = idx
                        break
                if marker_idx >= 0:
                    first = lines[marker_idx].split("RDC_IMPORT_ERROR:", 1)[1].strip()
                    tail = "\n".join(lines[marker_idx + 1 :]).strip()
                    details = first if not tail else f"{first}\n{tail}"
            raise ValueError(
                "Blender no pudo convertir el RDC. "
                f"Detalle: {details or 'sin detalle'}"
            )
        if not os.path.isfile(out_stl) or os.path.getsize(out_stl) <= 0:
            raise ValueError("Blender termino sin generar STL valido.")

        loaded = _load_mesh_file(out_stl, o3d=o3d)
        if loaded is None:
            raise ValueError("El STL generado por Blender no se pudo leer como malla vlida.")
        return loaded


@st.cache_data(show_spinner=False, ttl=3600)
def load_external_mesh_cached(payload: bytes, suffix: str) -> tuple[np.ndarray, np.ndarray]:
    """Load external triangle mesh (STL/OBJ/PLY/RDC...) from uploaded bytes."""
    o3d = None
    try:
        import open3d as o3d
    except Exception:
        o3d = None

    ext = str(suffix or "").strip().lower()
    if not ext.startswith("."):
        ext = f".{ext}" if ext else ".stl"

    if ext == ".rdc":
        errors: list[str] = []
        try:
            return _load_mesh_from_rdc_with_renderdoc(payload, o3d)
        except Exception as exc:
            errors.append(f"RenderDoc CLI: {exc}")

        try:
            return _extract_rdc_with_maps_importer(payload)
        except Exception as exc:
            errors.append(f"MapsModelsImporter extractor: {exc}")

        try:
            return _load_mesh_from_rdc_with_blender(payload, o3d)
        except Exception as exc:
            errors.append(f"Blender addon: {exc}")

        raise ValueError(" | ".join(errors))

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(payload)
            tmp_path = tmp.name

        loaded_mesh = _load_mesh_file(tmp_path, o3d=o3d)
        if loaded_mesh is None:
            open3d_hint = (
                " Nota: no se detecto Open3D en este entorno "
                "(en Python 3.13 suele no estar disponible)."
                if o3d is None
                else ""
            )
            raise ValueError(
                "No se pudo leer el archivo como malla triangulada vlida."
                f"{open3d_hint}"
            )
        return loaded_mesh
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def clean_external_mesh_basic(
    vertices: np.ndarray,
    triangles: np.ndarray,
    *,
    merge_distance: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Basic cleanup for imported meshes using Open3D (when available)."""
    try:
        import open3d as o3d
    except Exception:
        return vertices, triangles

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32))
    if float(merge_distance) > 0.0:
        mesh.merge_close_vertices(float(merge_distance))
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    v = np.asarray(mesh.vertices, dtype=np.float64)
    t = np.asarray(mesh.triangles, dtype=np.int32)
    return v, t


def downsample_mesh_for_preview(
    vertices: np.ndarray,
    triangles: np.ndarray,
    *,
    max_faces: int = 220_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce mesh size for interactive preview without changing stored geometry."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(triangles, dtype=np.int32)
    if f.ndim != 2 or f.shape[0] <= int(max_faces):
        return v, f

    step = int(np.ceil(f.shape[0] / float(max_faces)))
    sampled_faces = f[::max(step, 1)]
    if sampled_faces.shape[0] == 0:
        sampled_faces = f[: min(f.shape[0], int(max_faces))]
    if sampled_faces.shape[0] == 0:
        return v, f

    used = np.unique(sampled_faces.ravel())
    remap = np.full(v.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.shape[0], dtype=np.int64)
    v_small = v[used]
    f_small = remap[sampled_faces].astype(np.int32, copy=False)
    return v_small, f_small


def build_topographic_section_block(
    vertices: np.ndarray,
    *,
    shape: str,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    radius: float,
    resolution: float,
    base_thickness: float,
    z_min: float | None = None,
    z_max: float | None = None,
    max_cells: int = 1_400_000,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Build a printable topographic block from a processed mesh using XY crop shape.

    Output mesh contains:
    - top sampled surface (inside shape)
    - vertical side walls on crop boundary
    - flat bottom base
    """
    pts = np.asarray(vertices, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 20 or pts.shape[1] < 3:
        raise ValueError("No hay vertices suficientes para generar un recorte topografico.")

    shape_norm = str(shape or "Rectangulo").strip().lower()
    cx = float(center_x)
    cy = float(center_y)
    req_res = float(max(resolution, 0.05))
    thickness = float(max(base_thickness, 0.1))

    if "cir" in shape_norm:
        r = float(max(radius, req_res * 2.0))
        x_lo, x_hi = cx - r, cx + r
        y_lo, y_hi = cy - r, cy + r
    else:
        w = float(max(width, req_res * 4.0))
        h = float(max(height, req_res * 4.0))
        x_lo, x_hi = cx - w * 0.5, cx + w * 0.5
        y_lo, y_hi = cy - h * 0.5, cy + h * 0.5

    span_x = max(float(x_hi - x_lo), req_res * 2.0)
    span_y = max(float(y_hi - y_lo), req_res * 2.0)
    approx_cells = int((span_x / req_res + 1.0) * (span_y / req_res + 1.0))
    used_res = req_res
    resolution_limited = False
    if approx_cells > int(max_cells):
        scale = float(np.sqrt(approx_cells / float(max_cells)))
        used_res = req_res * scale
        resolution_limited = True

    x_vals = np.arange(x_lo, x_hi + used_res * 0.5, used_res, dtype=np.float64)
    y_vals = np.arange(y_lo, y_hi + used_res * 0.5, used_res, dtype=np.float64)
    if x_vals.size < 3 or y_vals.size < 3:
        raise ValueError("El recorte es demasiado pequeo para construir una malla util.")

    gx, gy = np.meshgrid(x_vals, y_vals, indexing="xy")
    if "cir" in shape_norm:
        rr = float(max(radius, used_res * 2.0))
        mask = ((gx - cx) ** 2 + (gy - cy) ** 2) <= (rr * rr + 1e-9)
    else:
        mask = np.ones(gx.shape, dtype=bool)

    if int(np.count_nonzero(mask)) < 16:
        raise ValueError("La regin seleccionada no contiene suficientes celdas.")

    sample_xy = np.column_stack((gx[mask], gy[mask])).astype(np.float64, copy=False)
    source_xy = pts[:, :2]
    source_z = pts[:, 2]
    has_z_filter = z_min is not None or z_max is not None
    if has_z_filter:
        mask_z = np.ones(source_z.shape[0], dtype=bool)
        if z_min is not None:
            mask_z &= source_z >= float(z_min)
        if z_max is not None:
            mask_z &= source_z <= float(z_max)
        if int(np.count_nonzero(mask_z)) >= 20:
            source_xy = source_xy[mask_z]
            source_z = source_z[mask_z]

    tree = cKDTree(source_xy)
    _, nn_idx = tree.query(sample_xy, k=1)
    sample_z = source_z[np.asarray(nn_idx, dtype=np.int64)]
    if z_max is not None:
        sample_z = np.minimum(sample_z, float(z_max))
    if z_min is not None:
        sample_z = np.maximum(sample_z, float(z_min))

    top_vertices = np.column_stack((sample_xy, sample_z)).astype(np.float64, copy=False)

    idx_map = np.full(mask.shape, -1, dtype=np.int64)
    idx_map.flat[np.flatnonzero(mask.ravel())] = np.arange(top_vertices.shape[0], dtype=np.int64)

    a = idx_map[:-1, :-1].ravel()
    b = idx_map[:-1, 1:].ravel()
    c = idx_map[1:, :-1].ravel()
    d = idx_map[1:, 1:].ravel()
    valid_cell = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
    if not np.any(valid_cell):
        raise ValueError("No se pudieron generar caras en el recorte seleccionado.")

    a = a[valid_cell]
    b = b[valid_cell]
    c = c[valid_cell]
    d = d[valid_cell]
    top_faces = np.vstack(
        [
            np.column_stack((a, b, d)),
            np.column_stack((a, d, c)),
        ]
    ).astype(np.int32, copy=False)

    if z_min is not None:
        base_z = float(z_min)
    else:
        base_z = float(np.min(top_vertices[:, 2]) - thickness)
    bottom_vertices = top_vertices.copy()
    bottom_vertices[:, 2] = base_z
    bottom_offset = int(top_vertices.shape[0])

    bottom_faces = np.column_stack(
        (
            top_faces[:, 0] + bottom_offset,
            top_faces[:, 2] + bottom_offset,
            top_faces[:, 1] + bottom_offset,
        )
    ).astype(np.int32, copy=False)

    edges = np.vstack(
        [
            top_faces[:, [0, 1]],
            top_faces[:, [1, 2]],
            top_faces[:, [2, 0]],
        ]
    ).astype(np.int64, copy=False)
    edges_sorted = np.sort(edges, axis=1)
    uniq_edges, edge_counts = np.unique(edges_sorted, axis=0, return_counts=True)
    boundary_edges = uniq_edges[edge_counts == 1]
    if boundary_edges.shape[0] == 0:
        raise ValueError("No se detecto borde exterior en el recorte.")

    wall_faces_1 = np.column_stack(
        (
            boundary_edges[:, 0],
            boundary_edges[:, 1],
            boundary_edges[:, 1] + bottom_offset,
        )
    ).astype(np.int32, copy=False)
    wall_faces_2 = np.column_stack(
        (
            boundary_edges[:, 0],
            boundary_edges[:, 1] + bottom_offset,
            boundary_edges[:, 0] + bottom_offset,
        )
    ).astype(np.int32, copy=False)

    out_vertices = np.vstack((top_vertices, bottom_vertices)).astype(np.float64, copy=False)
    out_faces = np.vstack((top_faces, bottom_faces, wall_faces_1, wall_faces_2)).astype(np.int32, copy=False)

    out_vertices, out_faces = clean_external_mesh_basic(out_vertices, out_faces, merge_distance=0.0)
    open_edges = count_boundary_edges(out_faces)
    if open_edges > 0:
        rep_v, rep_f = repair_open_boundaries(out_vertices, out_faces)
        rep_open = count_boundary_edges(rep_f)
        if rep_open <= open_edges:
            out_vertices, out_faces, open_edges = rep_v, rep_f, rep_open

    report = {
        "shape": "Circulo" if "cir" in shape_norm else "Rectangulo",
        "resolution_requested": float(req_res),
        "resolution_used": float(used_res),
        "resolution_limited": bool(resolution_limited),
        "max_cells": int(max_cells),
        "cells_x": int(x_vals.size),
        "cells_y": int(y_vals.size),
        "open_edges": int(open_edges),
        "base_z": float(base_z),
        "z_min_filter": float(z_min) if z_min is not None else None,
        "z_max_filter": float(z_max) if z_max is not None else None,
        "source_points_used": int(source_xy.shape[0]),
    }
    return out_vertices, out_faces, report




def _segment_plane_intersection(
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    axis: int,
    limit: float,
) -> np.ndarray:
    """Linear intersection of segment p0->p1 with axis-aligned plane axis=value."""
    denom = float(p1[axis] - p0[axis])
    if abs(denom) < 1e-15:
        return p1.copy()
    t = float((limit - p0[axis]) / denom)
    t = float(np.clip(t, 0.0, 1.0))
    return p0 + (p1 - p0) * t


def _clip_polygon_axis_plane(
    polygon: list[np.ndarray],
    *,
    axis: int,
    limit: float,
    keep_greater: bool,
) -> list[np.ndarray]:
    """Sutherland-Hodgman clip against one axis-aligned half-space."""
    if not polygon:
        return []

    out: list[np.ndarray] = []
    prev = polygon[-1]
    prev_inside = bool(prev[axis] >= (limit - 1e-9)) if keep_greater else bool(prev[axis] <= (limit + 1e-9))
    for curr in polygon:
        curr_inside = bool(curr[axis] >= (limit - 1e-9)) if keep_greater else bool(curr[axis] <= (limit + 1e-9))
        if prev_inside and curr_inside:
            out.append(curr)
        elif prev_inside and not curr_inside:
            out.append(_segment_plane_intersection(prev, curr, axis=axis, limit=limit))
        elif (not prev_inside) and curr_inside:
            out.append(_segment_plane_intersection(prev, curr, axis=axis, limit=limit))
            out.append(curr)
        prev = curr
        prev_inside = curr_inside
    return out


def build_topographic_section_block_preserve_detail(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    base_thickness: float,
    z_min: float | None = None,
    z_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Preserve-detail topographic clipping:
    - Clips top mesh triangles by box limits (XY and optional Z)
    - Keeps top surface detail from original triangles (no grid resampling)
    - Closes with vertical walls + flat base
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int32)
    if v.ndim != 2 or v.shape[0] < 3 or v.shape[1] < 3:
        raise ValueError("No hay vertices suficientes para recorte de detalle.")
    if f.ndim != 2 or f.shape[0] < 1 or f.shape[1] < 3:
        raise ValueError("No hay caras suficientes para recorte de detalle.")

    x0, x1 = float(min(x_min, x_max)), float(max(x_min, x_max))
    y0, y1 = float(min(y_min, y_max)), float(max(y_min, y_max))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Caja XY invlida para recorte.")

    thickness = float(max(base_thickness, 0.1))

    # Choose the face orientation whose centroids are higher (top skin).
    v0 = v[f[:, 0]]
    v1 = v[f[:, 1]]
    v2 = v[f[:, 2]]
    nz = np.cross(v1 - v0, v2 - v0)[:, 2]
    cz = (v0[:, 2] + v1[:, 2] + v2[:, 2]) / 3.0
    pos_mask = nz > 1e-12
    neg_mask = nz < -1e-12
    if np.any(pos_mask):
        pos_mean = float(np.mean(cz[pos_mask]))
    else:
        pos_mean = float("-inf")
    if np.any(neg_mask):
        neg_mean = float(np.mean(cz[neg_mask]))
    else:
        neg_mean = float("-inf")
    if np.isfinite(pos_mean) or np.isfinite(neg_mean):
        use_pos = pos_mean >= neg_mean
        face_ids = np.flatnonzero(pos_mask if use_pos else neg_mask)
    else:
        face_ids = np.arange(f.shape[0], dtype=np.int64)

    if face_ids.size == 0:
        raise ValueError("No se detecto superficie superior en la malla para recortar.")

    vert_map: dict[tuple[int, int, int], int] = {}
    top_vertices_list: list[np.ndarray] = []
    top_faces_list: list[list[int]] = []

    def _vertex_index(p: np.ndarray) -> int:
        key = (
            int(np.round(float(p[0]) * 1_000_000.0)),
            int(np.round(float(p[1]) * 1_000_000.0)),
            int(np.round(float(p[2]) * 1_000_000.0)),
        )
        idx = vert_map.get(key)
        if idx is not None:
            return idx
        idx = len(top_vertices_list)
        vert_map[key] = idx
        top_vertices_list.append(np.array(p, dtype=np.float64, copy=True))
        return idx

    for fi in face_ids:
        tri = v[f[int(fi), :3]]
        polygon: list[np.ndarray] = [tri[0].copy(), tri[1].copy(), tri[2].copy()]
        polygon = _clip_polygon_axis_plane(polygon, axis=0, limit=x0, keep_greater=True)
        if len(polygon) < 3:
            continue
        polygon = _clip_polygon_axis_plane(polygon, axis=0, limit=x1, keep_greater=False)
        if len(polygon) < 3:
            continue
        polygon = _clip_polygon_axis_plane(polygon, axis=1, limit=y0, keep_greater=True)
        if len(polygon) < 3:
            continue
        polygon = _clip_polygon_axis_plane(polygon, axis=1, limit=y1, keep_greater=False)
        if len(polygon) < 3:
            continue
        if z_min is not None:
            polygon = _clip_polygon_axis_plane(polygon, axis=2, limit=float(z_min), keep_greater=True)
            if len(polygon) < 3:
                continue
        if z_max is not None:
            polygon = _clip_polygon_axis_plane(polygon, axis=2, limit=float(z_max), keep_greater=False)
            if len(polygon) < 3:
                continue

        p0 = polygon[0]
        for k in range(1, len(polygon) - 1):
            pa = p0
            pb = polygon[k]
            pc = polygon[k + 1]
            nn = np.cross(pb - pa, pc - pa)
            if abs(float(nn[2])) < 1e-12:
                continue
            # Keep all top facets oriented with positive Z normal.
            if float(nn[2]) < 0.0:
                pb, pc = pc, pb
            ia = _vertex_index(pa)
            ib = _vertex_index(pb)
            ic = _vertex_index(pc)
            if ia == ib or ib == ic or ia == ic:
                continue
            top_faces_list.append([ia, ib, ic])

    if not top_faces_list:
        raise ValueError("El recorte no intersecta suficiente superficie superior.")

    top_vertices = np.asarray(top_vertices_list, dtype=np.float64)
    top_faces = np.asarray(top_faces_list, dtype=np.int32)
    base_z = float(z_min) if z_min is not None else float(np.min(top_vertices[:, 2]) - thickness)

    bottom_vertices = top_vertices.copy()
    bottom_vertices[:, 2] = base_z
    bottom_offset = int(top_vertices.shape[0])
    bottom_faces = np.column_stack(
        (
            top_faces[:, 0] + bottom_offset,
            top_faces[:, 2] + bottom_offset,
            top_faces[:, 1] + bottom_offset,
        )
    ).astype(np.int32, copy=False)

    edges = np.vstack(
        [
            top_faces[:, [0, 1]],
            top_faces[:, [1, 2]],
            top_faces[:, [2, 0]],
        ]
    ).astype(np.int64, copy=False)
    edges_sorted = np.sort(edges, axis=1)
    uniq_edges, edge_counts = np.unique(edges_sorted, axis=0, return_counts=True)
    boundary_edges = uniq_edges[edge_counts == 1]
    if boundary_edges.shape[0] == 0:
        raise ValueError("No se detecto borde exterior para cerrar paredes.")

        if not top_faces_list:
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3), dtype=np.int32),
            )
        return (
            np.asarray(top_vertices_list, dtype=np.float64),
            np.asarray(top_faces_list, dtype=np.int32),
        )

    top_vertices, top_faces = _collect_clipped_surface(primary_face_ids)
    source_faces_used = int(primary_face_ids.size)

    all_vertices_tmp: np.ndarray | None = None
    all_faces_tmp: np.ndarray | None = None
    if primary_face_ids.size < all_face_ids.size:
        all_vertices_tmp, all_faces_tmp = _collect_clipped_surface(all_face_ids)
        if all_faces_tmp.shape[0] > 0:
            if top_faces.shape[0] == 0:
                top_vertices, top_faces = all_vertices_tmp, all_faces_tmp
                source_faces_used = int(all_face_ids.size)
                face_filter_mode = "all_faces_fallback_empty_topskin"
            else:
                top_face_ratio = float(top_faces.shape[0]) / float(max(all_faces_tmp.shape[0], 1))
                top_vert_ratio = float(top_vertices.shape[0]) / float(max(all_vertices_tmp.shape[0], 1))
                if top_face_ratio < 0.92 or top_vert_ratio < 0.92:
                    top_vertices, top_faces = all_vertices_tmp, all_faces_tmp
                    source_faces_used = int(all_face_ids.size)
                    face_filter_mode = "all_faces_fallback_detail_loss"

    if top_faces.shape[0] == 0:
        if all_faces_tmp is None or all_vertices_tmp is None:
            all_vertices_tmp, all_faces_tmp = _collect_clipped_surface(all_face_ids)
        if all_faces_tmp.shape[0] == 0:
            raise ValueError("El recorte no intersecta suficiente superficie superior.")
        top_vertices, top_faces = all_vertices_tmp, all_faces_tmp
        source_faces_used = int(all_face_ids.size)
        face_filter_mode = "all_faces_fallback_empty_result"

    top_vertices, top_faces = clean_external_mesh_basic(top_vertices, top_faces, merge_distance=0.0)
    if top_faces.shape[0] == 0:
        raise ValueError("El recorte no tiene caras vlidas tras limpiar geometra.")

    min_top_z = float(np.min(top_vertices[:, 2]))
    if base_z_override is not None:
        base_z = float(min(float(base_z_override), min_top_z - 0.05))
    elif z_min is not None:
        base_z = float(min(float(z_min), min_top_z - 0.05))
    else:
        base_z = float(min_top_z - thickness)
    base_drop = float(max(min_top_z - base_z, 0.05))
    out_vertices, out_faces = add_base_to_mesh(top_vertices, top_faces, base_drop)
    out_vertices, out_faces = clean_external_mesh_basic(out_vertices, out_faces, merge_distance=0.0)
    open_edges = int(count_boundary_edges(out_faces))
    if open_edges > 0:
        rep_v, rep_f = repair_open_boundaries(out_vertices, out_faces)
        rep_open = int(count_boundary_edges(rep_f))
        if rep_open < open_edges:
            out_vertices, out_faces, open_edges = rep_v, rep_f, rep_open

    report = {
        "shape": "Rectangulo",
        "method": "preserve_detail",
        "resolution_requested": 0.0,
        "resolution_used": 0.0,
        "resolution_limited": False,
        "max_cells": 0,
        "cells_x": 0,
        "cells_y": 0,
        "open_edges": int(open_edges),
        "base_z": float(base_z),
        "z_min_filter": float(z_min) if z_min is not None else None,
        "z_max_filter": float(z_max) if z_max is not None else None,
        "source_points_used": int(v.shape[0]),
        "source_faces_used": int(source_faces_used),
        "source_faces_total": int(all_face_ids.size),
        "face_filter_mode": str(face_filter_mode),
        "top_faces_after_clip": int(top_faces.shape[0]),
    }
    return out_vertices, out_faces, report


def clip_mesh_rect_preserve_detail(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float | None = None,
    z_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Clip mesh by rectangular XY bounds preserving triangle detail.

    Unlike topographic block generation, this function does not add walls/base.
    It is intended as a pre-cut before running the external mesh processing flow.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int32)
    if v.ndim != 2 or v.shape[0] < 3 or v.shape[1] < 3:
        raise ValueError("No hay vertices suficientes para recorte previo.")
    if f.ndim != 2 or f.shape[0] < 1 or f.shape[1] < 3:
        raise ValueError("No hay caras suficientes para recorte previo.")

    x0, x1 = float(min(x_min, x_max)), float(max(x_min, x_max))
    y0, y1 = float(min(y_min, y_max)), float(max(y_min, y_max))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Caja XY invlida para recorte previo.")

    v0 = v[f[:, 0]]
    v1 = v[f[:, 1]]
    v2 = v[f[:, 2]]
    nz = np.cross(v1 - v0, v2 - v0)[:, 2]
    cz = (v0[:, 2] + v1[:, 2] + v2[:, 2]) / 3.0
    pos_mask = nz > 1e-12
    neg_mask = nz < -1e-12
    if np.any(pos_mask):
        pos_mean = float(np.mean(cz[pos_mask]))
    else:
        pos_mean = float("-inf")
    if np.any(neg_mask):
        neg_mean = float(np.mean(cz[neg_mask]))
    else:
        neg_mean = float("-inf")

    all_face_ids = np.arange(f.shape[0], dtype=np.int64)
    if np.isfinite(pos_mean) or np.isfinite(neg_mean):
        use_pos = pos_mean >= neg_mean
        primary_face_ids = np.flatnonzero(pos_mask if use_pos else neg_mask)
        face_filter_mode = "top_skin"
        if primary_face_ids.size == 0:
            primary_face_ids = all_face_ids
            face_filter_mode = "all_faces_fallback_empty_primary"
    else:
        primary_face_ids = all_face_ids
        face_filter_mode = "all_faces_fallback"

    def _collect_clipped_surface(face_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vert_map: dict[tuple[int, int, int], int] = {}
        top_vertices_list: list[np.ndarray] = []
        top_faces_list: list[list[int]] = []

        def _vertex_index(p: np.ndarray) -> int:
            key = (
                int(np.round(float(p[0]) * 1_000_000.0)),
                int(np.round(float(p[1]) * 1_000_000.0)),
                int(np.round(float(p[2]) * 1_000_000.0)),
            )
            idx = vert_map.get(key)
            if idx is not None:
                return idx
            idx = len(top_vertices_list)
            vert_map[key] = idx
            top_vertices_list.append(np.array(p, dtype=np.float64, copy=True))
            return idx

        for fi in face_indices:
            tri = v[f[int(fi), :3]]
            polygon: list[np.ndarray] = [tri[0].copy(), tri[1].copy(), tri[2].copy()]
            polygon = _clip_polygon_axis_plane(polygon, axis=0, limit=x0, keep_greater=True)
            if len(polygon) < 3:
                continue
            polygon = _clip_polygon_axis_plane(polygon, axis=0, limit=x1, keep_greater=False)
            if len(polygon) < 3:
                continue
            polygon = _clip_polygon_axis_plane(polygon, axis=1, limit=y0, keep_greater=True)
            if len(polygon) < 3:
                continue
            polygon = _clip_polygon_axis_plane(polygon, axis=1, limit=y1, keep_greater=False)
            if len(polygon) < 3:
                continue
            if z_min is not None:
                polygon = _clip_polygon_axis_plane(polygon, axis=2, limit=float(z_min), keep_greater=True)
                if len(polygon) < 3:
                    continue
            if z_max is not None:
                polygon = _clip_polygon_axis_plane(polygon, axis=2, limit=float(z_max), keep_greater=False)
                if len(polygon) < 3:
                    continue

            p0_local = polygon[0]
            for k in range(1, len(polygon) - 1):
                pa = p0_local
                pb = polygon[k]
                pc = polygon[k + 1]
                nn = np.cross(pb - pa, pc - pa)
                if np.linalg.norm(nn) < 1e-12:
                    continue
                ia = _vertex_index(pa)
                ib = _vertex_index(pb)
                ic = _vertex_index(pc)
                if ia == ib or ib == ic or ia == ic:
                    continue
                top_faces_list.append([ia, ib, ic])

        if not top_faces_list:
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3), dtype=np.int32),
            )
        return (
            np.asarray(top_vertices_list, dtype=np.float64),
            np.asarray(top_faces_list, dtype=np.int32),
        )

    clipped_vertices, clipped_faces = _collect_clipped_surface(primary_face_ids)
    source_faces_used = int(primary_face_ids.size)

    all_vertices_tmp: np.ndarray | None = None
    all_faces_tmp: np.ndarray | None = None
    if primary_face_ids.size < all_face_ids.size:
        all_vertices_tmp, all_faces_tmp = _collect_clipped_surface(all_face_ids)
        if all_faces_tmp.shape[0] > 0:
            if clipped_faces.shape[0] == 0:
                clipped_vertices, clipped_faces = all_vertices_tmp, all_faces_tmp
                source_faces_used = int(all_face_ids.size)
                face_filter_mode = "all_faces_fallback_empty_topskin"
            else:
                ratio = float(clipped_faces.shape[0]) / float(max(all_faces_tmp.shape[0], 1))
                if ratio < 0.92:
                    clipped_vertices, clipped_faces = all_vertices_tmp, all_faces_tmp
                    source_faces_used = int(all_face_ids.size)
                    face_filter_mode = "all_faces_fallback_detail_loss"

    if clipped_faces.shape[0] == 0:
        if all_faces_tmp is None or all_vertices_tmp is None:
            all_vertices_tmp, all_faces_tmp = _collect_clipped_surface(all_face_ids)
        if all_faces_tmp.shape[0] == 0:
            raise ValueError("El recorte no intersecta suficiente superficie.")
        clipped_vertices, clipped_faces = all_vertices_tmp, all_faces_tmp
        source_faces_used = int(all_face_ids.size)
        face_filter_mode = "all_faces_fallback_empty_result"

    clipped_vertices, clipped_faces = clean_external_mesh_basic(
        clipped_vertices,
        clipped_faces,
        merge_distance=0.0,
    )
    if clipped_faces.shape[0] == 0:
        raise ValueError("El recorte no tiene caras vlidas tras limpiar geometra.")

    report = {
        "shape": "Rectangulo",
        "method": "preserve_detail_clip_only",
        "open_edges": int(count_boundary_edges(clipped_faces)),
        "source_points_used": int(v.shape[0]),
        "source_faces_used": int(source_faces_used),
        "source_faces_total": int(all_face_ids.size),
        "face_filter_mode": str(face_filter_mode),
        "faces_after_clip": int(clipped_faces.shape[0]),
    }
    return clipped_vertices, clipped_faces, report


def get_mesh_material_config(style_name: str) -> dict:
    """Map mesh style label to Plotly Mesh3d material config."""
    key = str(style_name or "").strip().lower()
    base_lighting = dict(ambient=0.42, diffuse=0.72, specular=0.22, roughness=0.72, fresnel=0.06)

    styles: dict[str, dict] = {
        "solido opaco": {
            "mode": "solid",
            "color": "#bcc3cc",
            "flatshading": True,
            "opacity": 1.0,
            "lighting": dict(base_lighting),
        },
        "altura terreno": {
            "mode": "height",
            "colorscale": "Earth",
            "flatshading": False,
            "opacity": 1.0,
            "lighting": dict(base_lighting, ambient=0.38, diffuse=0.82),
        },
        "arcilla mate": {
            "mode": "solid",
            "color": "#c98f5d",
            "flatshading": False,
            "opacity": 1.0,
            "lighting": dict(base_lighting, roughness=0.9, specular=0.08),
        },
        "hormigon tecnico": {
            "mode": "solid",
            "color": "#9ba5b3",
            "flatshading": False,
            "opacity": 1.0,
            "lighting": dict(base_lighting, roughness=0.82, specular=0.12),
        },
        "metal satinado": {
            "mode": "solid",
            "color": "#9aa8bd",
            "flatshading": False,
            "opacity": 1.0,
            "lighting": dict(base_lighting, roughness=0.38, specular=0.64, fresnel=0.2),
        },
        "topografico intenso": {
            "mode": "height",
            "colorscale": "Turbo",
            "flatshading": False,
            "opacity": 1.0,
            "lighting": dict(base_lighting, ambient=0.46, diffuse=0.88, specular=0.16, roughness=0.86),
        },
    }
    return styles.get(key, styles["solido opaco"])


def build_wireframe_trace(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    color: str = "rgba(255,255,255,0.22)",
    width: float = 1.0,
    max_edges: int = 260_000,
) -> go.Scatter3d | None:
    """Build lightweight wireframe overlay trace from triangle faces."""
    if vertices.size == 0 or faces.size == 0:
        return None

    tri = np.asarray(faces, dtype=np.int64)
    edges = np.vstack((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]))
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    if edges.shape[0] == 0:
        return None
    if edges.shape[0] > int(max_edges):
        step = int(np.ceil(edges.shape[0] / float(max_edges)))
        edges = edges[::step]

    v = np.asarray(vertices, dtype=np.float64)
    x_lines: list[float | None] = []
    y_lines: list[float | None] = []
    z_lines: list[float | None] = []
    for e in edges:
        a, b = int(e[0]), int(e[1])
        x_lines.extend([float(v[a, 0]), float(v[b, 0]), None])
        y_lines.extend([float(v[a, 1]), float(v[b, 1]), None])
        z_lines.extend([float(v[a, 2]), float(v[b, 2]), None])

    return go.Scatter3d(
        x=x_lines,
        y=y_lines,
        z=z_lines,
        mode="lines",
        line=dict(color=color, width=width),
        name="Wireframe",
        hoverinfo="skip",
        showlegend=False,
    )


def build_box_wireframe_trace(
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
    color: str = "rgba(255,80,80,0.95)",
    width: float = 3.0,
) -> go.Scatter3d:
    """Build a 3D rectangular box wireframe trace."""
    x0, x1 = float(min(x_min, x_max)), float(max(x_min, x_max))
    y0, y1 = float(min(y_min, y_max)), float(max(y_min, y_max))
    z0, z1 = float(min(z_min, z_max)), float(max(z_min, z_max))

    v = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float64,
    )
    edges = np.array(
        [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ],
        dtype=np.int32,
    )
    x_lines: list[float | None] = []
    y_lines: list[float | None] = []
    z_lines: list[float | None] = []
    for e in edges:
        a, b = int(e[0]), int(e[1])
        x_lines.extend([float(v[a, 0]), float(v[b, 0]), None])
        y_lines.extend([float(v[a, 1]), float(v[b, 1]), None])
        z_lines.extend([float(v[a, 2]), float(v[b, 2]), None])

    return go.Scatter3d(
        x=x_lines,
        y=y_lines,
        z=z_lines,
        mode="lines",
        line=dict(color=color, width=width),
        name="Caja recorte",
        hoverinfo="skip",
        showlegend=False,
    )
def build_mesh_figure(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    style_name: str,
    show_wireframe: bool,
    uirevision_id: str,
    height_px: int,
    axis_ranges: dict[str, tuple[float, float]] | None = None,
    box_bounds: dict[str, float] | None = None,
) -> go.Figure:
    """Build a Mesh3d figure with camera persistence and style presets."""
    material = get_mesh_material_config(style_name)
    mesh_kwargs: dict = {
        "x": vertices[:, 0],
        "y": vertices[:, 1],
        "z": vertices[:, 2],
        "i": faces[:, 0],
        "j": faces[:, 1],
        "k": faces[:, 2],
        "lighting": material["lighting"],
        "flatshading": material["flatshading"],
        "opacity": float(material.get("opacity", 1.0)),
        "name": "Malla",
    }
    if material["mode"] == "height":
        mesh_kwargs["intensity"] = vertices[:, 2]
        mesh_kwargs["colorscale"] = material.get("colorscale", "Earth")
        mesh_kwargs["showscale"] = True
        mesh_kwargs["colorbar"] = dict(title="Altura (m)")
    else:
        mesh_kwargs["color"] = material.get("color", "#b8bcc2")
        mesh_kwargs["showscale"] = False

    fig = go.Figure(data=[go.Mesh3d(**mesh_kwargs)])
    if show_wireframe:
        wire_trace = build_wireframe_trace(vertices, faces)
        if wire_trace is not None:
            fig.add_trace(wire_trace)
    if box_bounds:
        try:
            box_trace = build_box_wireframe_trace(
                x_min=float(box_bounds.get("x_min", 0.0)),
                x_max=float(box_bounds.get("x_max", 0.0)),
                y_min=float(box_bounds.get("y_min", 0.0)),
                y_max=float(box_bounds.get("y_max", 0.0)),
                z_min=float(box_bounds.get("z_min", 0.0)),
                z_max=float(box_bounds.get("z_max", 0.0)),
            )
            fig.add_trace(box_trace)
        except Exception:
            pass

    scene_layout: dict = {
        "xaxis": dict(title="X (m)"),
        "yaxis": dict(title="Y (m)"),
        "zaxis": dict(title="Z (m)"),
        "aspectmode": "data",
        "uirevision": uirevision_id,
    }

    if axis_ranges:
        for axis_name in ("x", "y", "z"):
            axis_range = axis_ranges.get(axis_name)
            if axis_range is None:
                continue
            low, high = float(axis_range[0]), float(axis_range[1])
            if not np.isfinite(low) or not np.isfinite(high):
                continue
            if high <= low:
                mid = 0.5 * (low + high)
                low = mid - 0.5
                high = mid + 0.5
            scene_layout[f"{axis_name}axis"] = dict(
                title=f"{axis_name.upper()} (m)",
                range=[low, high],
                autorange=False,
            )

    fig.update_layout(
        scene=scene_layout,
        uirevision=uirevision_id,
        height=height_px,
        margin=dict(l=0, r=0, b=0, t=30),
    )
    return fig


def build_layered_mesh_figure(
    components: dict[str, tuple[np.ndarray | None, np.ndarray | None]] | None,
    *,
    style_name: str,
    show_wireframe: bool,
    uirevision_id: str,
    height_px: int,
    axis_ranges: dict[str, tuple[float, float]] | None = None,
    box_bounds: dict[str, float] | None = None,
) -> go.Figure | None:
    """Build a component-colored mesh preview for terrain/vegetation/buildings."""
    if not isinstance(components, dict):
        return None

    material = get_mesh_material_config(style_name)
    lighting = material["lighting"]
    opacity = float(material.get("opacity", 1.0))
    flatshading = bool(material.get("flatshading", False))
    component_specs = [
        ("terrain", "Terreno", "#37c7a7"),
        ("vegetation", "Vegetacion", "#79d66f"),
        ("buildings", "Edificios", "#ff9f43"),
    ]

    fig = go.Figure()
    has_any = False
    for key, label, color in component_specs:
        verts, faces = components.get(key, (None, None))
        if verts is None or faces is None:
            continue
        v = np.asarray(verts, dtype=np.float64)
        f = np.asarray(faces, dtype=np.int32)
        if v.size == 0 or f.size == 0:
            continue
        has_any = True
        fig.add_trace(
            go.Mesh3d(
                x=v[:, 0],
                y=v[:, 1],
                z=v[:, 2],
                i=f[:, 0],
                j=f[:, 1],
                k=f[:, 2],
                color=color,
                name=label,
                lighting=lighting,
                flatshading=flatshading,
                opacity=opacity,
                showscale=False,
            )
        )
        if show_wireframe:
            wire_trace = build_wireframe_trace(v, f)
            if wire_trace is not None:
                wire_trace.name = f"Wireframe {label}"
                fig.add_trace(wire_trace)

    if not has_any:
        return None

    if box_bounds:
        try:
            fig.add_trace(
                build_box_wireframe_trace(
                    x_min=float(box_bounds.get("x_min", 0.0)),
                    x_max=float(box_bounds.get("x_max", 0.0)),
                    y_min=float(box_bounds.get("y_min", 0.0)),
                    y_max=float(box_bounds.get("y_max", 0.0)),
                    z_min=float(box_bounds.get("z_min", 0.0)),
                    z_max=float(box_bounds.get("z_max", 0.0)),
                )
            )
        except Exception:
            pass

    scene_layout: dict = {
        "xaxis": dict(title="X (m)"),
        "yaxis": dict(title="Y (m)"),
        "zaxis": dict(title="Z (m)"),
        "aspectmode": "data",
        "uirevision": uirevision_id,
    }
    if axis_ranges:
        for axis_name in ("x", "y", "z"):
            axis_range = axis_ranges.get(axis_name)
            if axis_range is None:
                continue
            low, high = float(axis_range[0]), float(axis_range[1])
            if not np.isfinite(low) or not np.isfinite(high):
                continue
            if high <= low:
                mid = 0.5 * (low + high)
                low = mid - 0.5
                high = mid + 0.5
            scene_layout[f"{axis_name}axis"] = dict(
                title=f"{axis_name.upper()} (m)",
                range=[low, high],
                autorange=False,
            )

    fig.update_layout(
        scene=scene_layout,
        uirevision=uirevision_id,
        height=height_px,
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(orientation="h"),
    )
    return fig


def extract_first_selected_xy(plot_state) -> tuple[float, float] | None:
    """Extract XY from the first selected Plotly point, if present."""
    points_data = _extract_selected_points(plot_state)
    if not points_data:
        return None

    first = points_data[0]
    x_val = _extract_value(first, "x")
    y_val = _extract_value(first, "y")
    if x_val is None or y_val is None:
        return None

    try:
        return float(x_val), float(y_val)
    except Exception:
        return None


def extract_first_selected_index(plot_state) -> int | None:
    """Extract point index from first selected point when x/y are not included."""
    points_data = _extract_selected_points(plot_state)
    if not points_data:
        return None

    first = points_data[0]
    for key in ("point_index", "pointIndex", "pointNumber", "point_number"):
        value = _extract_value(first, key)
        if value is None:
            continue
        try:
            idx = int(value)
        except Exception:
            continue
        if idx >= 0:
            return idx
    return None


def extract_selected_bounds(plot_state, *, min_points: int = 3) -> tuple[float, float, float, float] | None:
    """Extract XY bounds from current Plotly selection."""
    selection = _extract_value(plot_state, "selection")
    if selection is not None:
        # Prefer explicit rectangular range if available (corner-to-corner drag).
        for container_key in ("range", "box", "bbox", "selection"):
            container = _extract_value(selection, container_key)
            if container is None:
                continue
            x_pair = _extract_value(container, "x")
            y_pair = _extract_value(container, "y")
            if x_pair is None:
                x_pair = _extract_value(container, "xaxis")
            if y_pair is None:
                y_pair = _extract_value(container, "yaxis")
            try:
                if x_pair is not None and y_pair is not None and len(x_pair) >= 2 and len(y_pair) >= 2:
                    x0 = float(min(float(x_pair[0]), float(x_pair[1])))
                    x1 = float(max(float(x_pair[0]), float(x_pair[1])))
                    y0 = float(min(float(y_pair[0]), float(y_pair[1])))
                    y1 = float(max(float(y_pair[0]), float(y_pair[1])))
                    if x1 > x0 and y1 > y0:
                        return x0, x1, y0, y1
            except Exception:
                pass

        # Some payloads keep x/y ranges directly under `selection`.
        sel_x = _extract_value(selection, "x")
        sel_y = _extract_value(selection, "y")
        try:
            if sel_x is not None and sel_y is not None and len(sel_x) >= 2 and len(sel_y) >= 2:
                x0 = float(min(float(sel_x[0]), float(sel_x[1])))
                x1 = float(max(float(sel_x[0]), float(sel_x[1])))
                y0 = float(min(float(sel_y[0]), float(sel_y[1])))
                y1 = float(max(float(sel_y[0]), float(sel_y[1])))
                if x1 > x0 and y1 > y0:
                    return x0, x1, y0, y1
        except Exception:
            pass

    points_data = _extract_selected_points(plot_state)
    if not points_data:
        return None

    xs: list[float] = []
    ys: list[float] = []
    for point in points_data:
        x_val = _extract_value(point, "x")
        y_val = _extract_value(point, "y")
        if x_val is None or y_val is None:
            continue
        try:
            x_f = float(x_val)
            y_f = float(y_val)
        except Exception:
            continue
        if not np.isfinite(x_f) or not np.isfinite(y_f):
            continue
        xs.append(x_f)
        ys.append(y_f)

    if len(xs) < int(min_points):
        return None

    x_min = float(np.min(xs))
    x_max = float(np.max(xs))
    y_min = float(np.min(ys))
    y_max = float(np.max(ys))
    if x_max <= x_min or y_max <= y_min:
        return None
    return x_min, x_max, y_min, y_max


def apply_clip_rect_from_bounds(
    bounds: tuple[float, float, float, float],
    *,
    min_size: float = 0.5,
    tol: float = 1e-6,
) -> bool:
    """Apply XY rectangle bounds to clip state. Returns True if values changed."""
    x0, x1, y0, y1 = bounds
    x_min = float(min(x0, x1))
    x_max = float(max(x0, x1))
    y_min = float(min(y0, y1))
    y_max = float(max(y0, y1))

    width = float(max(x_max - x_min, float(min_size)))
    height = float(max(y_max - y_min, float(min_size)))
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    x_min = float(cx - 0.5 * width)
    x_max = float(cx + 0.5 * width)
    y_min = float(cy - 0.5 * height)
    y_max = float(cy + 0.5 * height)

    old_vals = (
        float(st.session_state.get("ext_clip_box_x_min", x_min)),
        float(st.session_state.get("ext_clip_box_x_max", x_max)),
        float(st.session_state.get("ext_clip_box_y_min", y_min)),
        float(st.session_state.get("ext_clip_box_y_max", y_max)),
        float(st.session_state.get("ext_clip_center_x", cx)),
        float(st.session_state.get("ext_clip_center_y", cy)),
        float(st.session_state.get("ext_clip_width", width)),
        float(st.session_state.get("ext_clip_height", height)),
    )
    new_vals = (x_min, x_max, y_min, y_max, cx, cy, width, height)
    changed = any(abs(a - b) > float(tol) for a, b in zip(old_vals, new_vals))

    st.session_state.ext_clip_shape = "Rectangulo"
    st.session_state.ext_clip_box_x_min = x_min
    st.session_state.ext_clip_box_x_max = x_max
    st.session_state.ext_clip_box_y_min = y_min
    st.session_state.ext_clip_box_y_max = y_max
    st.session_state.ext_clip_center_x = cx
    st.session_state.ext_clip_center_y = cy
    st.session_state.ext_clip_width = width
    st.session_state.ext_clip_height = height
    # Keep slider keys synced so next rerun doesn't revert XY.
    st.session_state.ext_clip_box_x_range_quick = (x_min, x_max)
    st.session_state.ext_clip_box_y_range_quick = (y_min, y_max)
    if changed:
        st.session_state.ext_clip_needs_regen = True
    return changed


def apply_raw_preclip_bounds(
    bounds: tuple[float, float, float, float],
    *,
    min_size: float = 0.5,
    tol: float = 1e-6,
) -> bool:
    """Apply XY rectangle bounds to raw pre-clip state. Returns True if values changed."""
    x0, x1, y0, y1 = bounds
    x_min = float(min(x0, x1))
    x_max = float(max(x0, x1))
    y_min = float(min(y0, y1))
    y_max = float(max(y0, y1))

    width = float(max(x_max - x_min, float(min_size)))
    height = float(max(y_max - y_min, float(min_size)))
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    x_min = float(cx - 0.5 * width)
    x_max = float(cx + 0.5 * width)
    y_min = float(cy - 0.5 * height)
    y_max = float(cy + 0.5 * height)

    old_vals = (
        float(st.session_state.get("ext_raw_preclip_x_min", x_min)),
        float(st.session_state.get("ext_raw_preclip_x_max", x_max)),
        float(st.session_state.get("ext_raw_preclip_y_min", y_min)),
        float(st.session_state.get("ext_raw_preclip_y_max", y_max)),
    )
    new_vals = (x_min, x_max, y_min, y_max)
    changed = any(abs(a - b) > float(tol) for a, b in zip(old_vals, new_vals))

    st.session_state.ext_raw_preclip_x_min = x_min
    st.session_state.ext_raw_preclip_x_max = x_max
    st.session_state.ext_raw_preclip_y_min = y_min
    st.session_state.ext_raw_preclip_y_max = y_max
    return changed


def _extract_value(payload, key: str):
    """Safe dict/object getter."""
    try:
        if hasattr(payload, key):
            return getattr(payload, key)
    except Exception:
        pass
    try:
        return payload.get(key)
    except Exception:
        pass
    try:
        return payload[key]
    except Exception:
        return None


def _extract_selected_points(plot_state):
    """Best-effort extraction of selected points from Streamlit Plotly state."""
    if plot_state is None:
        return None

    selection = _extract_value(plot_state, "selection")
    if selection:
        points_data = _extract_value(selection, "points")
        if points_data:
            return points_data

    points_data = _extract_value(plot_state, "points")
    if points_data:
        return points_data

    return None


def build_xy_picker_figure(
    xy_points: np.ndarray,
    z_values: np.ndarray,
    *,
    uirevision_id: str = "xy-center-picker",
    height_px: int = 360,
) -> go.Figure:
    """Build a fast top-view picker figure (2D) for center selection."""
    fig = go.Figure(
        data=[
            go.Scattergl(
                x=xy_points[:, 0],
                y=xy_points[:, 1],
                mode="markers",
                marker=dict(
                    size=4,
                    color=z_values,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Z (m)"),
                    opacity=0.85,
                ),
                name="Puntos XY",
            )
        ]
    )
    fig.update_layout(
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        yaxis_scaleanchor="x",
        yaxis_scaleratio=1,
        height=height_px,
        margin=dict(l=0, r=0, b=0, t=20),
        clickmode="event+select",
        uirevision=uirevision_id,
        dragmode="select",
        hovermode="closest",
    )
    return fig


def build_external_clip_picker_figure(
    xy_points: np.ndarray,
    z_values: np.ndarray,
    *,
    shape_mode: str,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    radius: float,
    uirevision_id: str = "external-clip-picker",
    height_px: int = 360,
) -> go.Figure:
    """Build top-view figure to draw/select rectangular crop on processed mesh."""
    xy = np.asarray(xy_points, dtype=np.float64)
    z = np.asarray(z_values, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[0] == 0 or xy.shape[1] < 2:
        return go.Figure()
    if z.ndim != 1 or z.shape[0] != xy.shape[0]:
        z = np.zeros(xy.shape[0], dtype=np.float64)

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=xy[:, 0],
            y=xy[:, 1],
            mode="markers",
            marker=dict(
                size=3,
                color=z,
                colorscale="Turbo",
                opacity=0.82,
                showscale=False,
            ),
            name="Planta malla",
            showlegend=False,
        )
    )

    mode_norm = str(shape_mode or "Rectangulo").strip().lower()
    cx = float(center_x)
    cy = float(center_y)
    if mode_norm.startswith("rect"):
        w = float(max(width, 0.1))
        h = float(max(height, 0.1))
        x0 = cx - 0.5 * w
        x1 = cx + 0.5 * w
        y0 = cy - 0.5 * h
        y1 = cy + 0.5 * h
        fig.add_trace(
            go.Scatter(
                x=[x0, x1, x1, x0, x0],
                y=[y0, y0, y1, y1, y0],
                mode="lines",
                line=dict(color="#ef4444", width=2),
                name="Recorte actual",
                showlegend=False,
                hoverinfo="skip",
            )
        )
    else:
        r = float(max(radius, 0.1))
        theta = np.linspace(0.0, 2.0 * np.pi, 160, endpoint=True)
        fig.add_trace(
            go.Scatter(
                x=(cx + r * np.cos(theta)),
                y=(cy + r * np.sin(theta)),
                mode="lines",
                line=dict(color="#ef4444", width=2),
                name="Recorte actual",
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        yaxis_scaleanchor="x",
        yaxis_scaleratio=1,
        margin=dict(l=0, r=0, b=0, t=20),
        height=height_px,
        clickmode="event+select",
        dragmode="select",
        hovermode="closest",
        uirevision=uirevision_id,
    )
    return fig


def build_footprint_selector_figure(
    editor_items: list[dict],
    selected_key: str | None,
    *,
    uirevision_id: str = "footprint-selector",
    height_px: int = 360,
) -> go.Figure:
    """2D plan view of building footprints + selectable centroids."""
    outline_x: list[float | None] = []
    outline_y: list[float | None] = []
    cx: list[float] = []
    cy: list[float] = []
    ctext: list[str] = []
    ccolor: list[str] = []

    for item in editor_items:
        ring = np.asarray(item.get("xy"), dtype=np.float64)
        if ring.ndim == 2 and ring.shape[0] >= 3 and ring.shape[1] >= 2:
            ring_xy = ring[:, :2]
            ring_closed = np.vstack([ring_xy, ring_xy[0]])
            outline_x.extend([float(v) for v in ring_closed[:, 0]])
            outline_y.extend([float(v) for v in ring_closed[:, 1]])
            outline_x.append(None)
            outline_y.append(None)

        cx.append(float(item.get("cx", 0.0)))
        cy.append(float(item.get("cy", 0.0)))
        label = str(item.get("label", "Edificio"))
        h = float(item.get("height", 0.0))
        ctext.append(f"{label}<br>Altura: {h:.2f} m")
        key = str(item.get("key", ""))
        if selected_key is not None and key == str(selected_key):
            ccolor.append("#ef4444")
        elif bool(item.get("has_override", False)):
            ccolor.append("#16a34a")
        else:
            ccolor.append("#3b82f6")

    fig = go.Figure()
    if outline_x:
        fig.add_trace(
            go.Scattergl(
                x=outline_x,
                y=outline_y,
                mode="lines",
                line=dict(color="#94a3b8", width=1),
                name="Huellas",
                hoverinfo="skip",
                showlegend=False,
            )
        )

    if cx:
        fig.add_trace(
            go.Scattergl(
                x=cx,
                y=cy,
                mode="markers",
                marker=dict(size=10, color=ccolor, line=dict(width=0.8, color="#0f172a")),
                text=ctext,
                hovertemplate="%{text}<extra></extra>",
                name="Edificios",
                showlegend=False,
            )
        )

    fig.update_layout(
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        yaxis_scaleanchor="x",
        yaxis_scaleratio=1,
        margin=dict(l=0, r=0, b=0, t=24),
        height=height_px,
        clickmode="event+select",
        dragmode="select",
        hovermode="closest",
        uirevision=uirevision_id,
    )
    return fig




def render_app_header() -> None:
    """Render mnimal sticky step navigation."""
    current_step = int(st.session_state.get("step", 1))
    loaded = bool(st.session_state.get("file_loaded", False))

    step_labels = {
        1: "Cargar",
        2: "Clasificar",
        3: "Procesar",
    }

    nav_cols = st.columns(3, gap="small")
    for idx in (1, 2, 3):
        disabled = (idx > 1 and not loaded)
        if nav_cols[idx - 1].button(
            step_labels[idx],
            key=f"header_step_btn_{idx}",
            use_container_width=True,
            disabled=disabled,
            type="primary" if idx == current_step else "secondary",
        ):
            st.session_state.step = idx
            st.rerun()


def render_external_mesh_workflow() -> None:
    """Parallel workflow: import external mesh, repair and export STL."""
    st.header("Flujo paralelo: Malla externa -> STL")
    st.caption(
        "Importa una malla (OBJ/PLY/STL/RDC), reparala para impresin 3D y exporta STL."
    )

    uploaded_mesh = st.file_uploader(
        "Archivo de malla externa",
        type=["obj", "ply", "stl", "rdc"],
        key="external_mesh_upload",
    )
    if uploaded_mesh is None:
        st.info("Sube una malla externa para comenzar este flujo.")
        return

    raw_vertices = None
    raw_triangles = None
    try:
        raw_vertices, raw_triangles = load_external_mesh_cached(
            uploaded_mesh.getvalue(),
            Path(uploaded_mesh.name).suffix.lower(),
        )
    except Exception as exc:
        st.error(f"No se pudo leer la malla: {exc}")
        return

    raw_vertices = np.asarray(raw_vertices, dtype=np.float64)
    raw_triangles = np.asarray(raw_triangles, dtype=np.int32)
    open_edges_raw = int(count_boundary_edges(raw_triangles))

    m1, m2, m3 = st.columns(3)
    m1.metric("Vertices", f"{raw_vertices.shape[0]:,}")
    m2.metric("Caras", f"{raw_triangles.shape[0]:,}")
    m3.metric("Aristas abiertas", f"{open_edges_raw:,}")

    style_name_raw = st.selectbox(
        "Acabado de superficie (origen)",
        (
            "Solido opaco",
            "Altura terreno",
            "Arcilla mate",
            "Hormigon tecnico",
            "Metal satinado",
            "Topografico intenso",
        ),
        index=0,
        key="ext_mesh_view_style_raw",
    )
    wire_raw = st.checkbox(
        "Mostrar aristas en origen (wireframe)",
        value=False,
        key="ext_mesh_wireframe_raw",
    )

    prev_v, prev_f = downsample_mesh_for_preview(
        raw_vertices,
        raw_triangles,
        max_faces=220_000,
    )
    fig_raw = build_mesh_figure(
        prev_v,
        prev_f,
        style_name=str(style_name_raw),
        show_wireframe=bool(wire_raw),
        uirevision_id="external-mesh-raw-preview",
        height_px=480,
    )
    st.plotly_chart(fig_raw, use_container_width=True, key="plot_external_mesh_raw_preview")

    c1, c2, c3 = st.columns(3)
    merge_dist = c1.slider(
        "Fusionar vertices cercanos",
        0.0,
        0.5,
        0.0,
        0.005,
        key="ext_mesh_merge_dist",
    )
    voxel_ext = c2.slider(
        "Simplificacion por voxel",
        0.0,
        1.0,
        0.0,
        0.01,
        key="ext_mesh_voxel_simplify",
        help=(
            "Agrupa vertices dentro de cubos 3D para reducir triangulos.\n\n"
            "Ejemplo: 0.00 mantiene todo el detalle; 0.08 aligera la malla pero redondea bordes finos."
        ),
    )
    max_open_tol_ext = int(
        c3.slider(
            "Tolerancia aristas abiertas",
            0,
            200,
            0,
            1,
            key="ext_mesh_open_tol",
        )
    )
    do_repair_ext = st.checkbox(
        "Intentar cierre automatico de bordes",
        value=True,
        key="ext_mesh_auto_repair",
    )

    if st.button("Procesar malla externa", type="primary", key="ext_mesh_process_btn"):
        try:
            with st.status("Procesando malla externa...", expanded=True) as status_box:
                status_box.write("1/4 - Limpieza de vertices/caras")
                proc_vertices = raw_vertices.copy()
                proc_triangles = raw_triangles.copy()
                open_before = int(count_boundary_edges(proc_triangles))

                proc_vertices, proc_triangles = clean_external_mesh_basic(
                    proc_vertices,
                    proc_triangles,
                    merge_distance=float(merge_dist),
                )

                if float(voxel_ext) > 0.0:
                    status_box.write("2/4 - Simplificacion por voxel")
                    proc_vertices, proc_triangles = simplify_mesh_with_voxel_grid(
                        proc_vertices,
                        proc_triangles,
                        float(voxel_ext),
                    )
                else:
                    status_box.write("2/4 - Simplificacion por voxel (omitida)")

                status_box.write("3/4 - Verificacion de bordes abiertos")
                open_after_clean = int(count_boundary_edges(proc_triangles))
                open_final = open_after_clean

                if bool(do_repair_ext) and open_after_clean > 0:
                    status_box.write("4/4 - Reparacion automtica de bordes")
                    rep_v, rep_t = repair_open_boundaries(proc_vertices, proc_triangles)
                    rep_open = int(count_boundary_edges(rep_t))
                    if rep_open <= open_final:
                        proc_vertices, proc_triangles = rep_v, rep_t
                        open_final = rep_open
                else:
                    status_box.write("4/4 - Reparacion automtica (omitida)")

                st.session_state.ext_mesh_vertices = proc_vertices
                st.session_state.ext_mesh_faces = proc_triangles
                st.session_state.ext_mesh_source_name = uploaded_mesh.name
                st.session_state.ext_mesh_report = {
                    "open_before": int(open_before),
                    "open_after_clean": int(open_after_clean),
                    "open_final": int(open_final),
                    "tol": int(max_open_tol_ext),
                }
                if int(open_final) <= int(max_open_tol_ext):
                    st.session_state.ext_mesh_last_status = (
                        f"Malla lista para imprimir. Aristas abiertas finales: {open_final:,} "
                        f"(tolerancia {max_open_tol_ext:,})."
                    )
                    status_box.update(label="Procesado completado", state="complete")
                    st.success(st.session_state.ext_mesh_last_status)
                else:
                    st.session_state.ext_mesh_last_status = (
                        f"Malla procesada con bordes abiertos residuales: {open_final:,} "
                        f"(tolerancia {max_open_tol_ext:,})."
                    )
                    status_box.update(label="Procesado completado con advertencias", state="error")
                    st.warning(st.session_state.ext_mesh_last_status)
        except Exception as exc:
            st.error(f"Error procesando malla externa: {exc}")

    proc_vertices = st.session_state.get("ext_mesh_vertices")
    proc_triangles = st.session_state.get("ext_mesh_faces")
    proc_report = st.session_state.get("ext_mesh_report")
    if isinstance(proc_vertices, np.ndarray) and isinstance(proc_triangles, np.ndarray):
        st.subheader("Vista previa malla externa procesada")
        style_name_ext = st.selectbox(
            "Acabado de superficie",
            (
                "Solido opaco",
                "Altura terreno",
                "Arcilla mate",
                "Hormigon tecnico",
                "Metal satinado",
                "Topografico intenso",
            ),
            index=5,
            key="ext_mesh_view_style",
        )
        wire_ext = st.checkbox(
            "Mostrar aristas (wireframe)",
            value=False,
            key="ext_mesh_wireframe",
        )
        fig_ext = build_mesh_figure(
            np.asarray(proc_vertices, dtype=np.float64),
            np.asarray(proc_triangles, dtype=np.int32),
            style_name=str(style_name_ext),
            show_wireframe=bool(wire_ext),
            uirevision_id="external-mesh-preview",
            height_px=520,
        )
        st.plotly_chart(fig_ext, use_container_width=True, key="plot_external_mesh_preview")

        if isinstance(proc_report, dict):
            st.caption(
                "Integridad: "
                f"origen={int(proc_report.get('open_before', 0)):,} | "
                f"tras limpieza={int(proc_report.get('open_after_clean', 0)):,} | "
                f"final={int(proc_report.get('open_final', 0)):,}"
            )

        output_name = (
            f"external_repaired_{int(proc_vertices.shape[0])}_vertices.stl"
            if not st.session_state.get("ext_mesh_source_name")
            else f"{Path(st.session_state.get('ext_mesh_source_name')).stem}_repaired.stl"
        )
        output_path = get_temp_output_path(output_name, ".stl")
        write_stl(output_path, proc_vertices, proc_triangles)

        with open(output_path, "rb") as handle:
            st.download_button(
                label="Descargar malla lista para imprimir (STL)",
                data=handle,
                file_name=output_name,
                mime="application/octet-stream",
                type="primary",
                key="download_external_stl_btn",
            )
# La navegación se maneja centralizadamente al inicio

if st.session_state.get("app_workflow_mode") == "Malla externa":
    render_external_mesh_workflow()
    st.stop()
# ============================================================================
# PASO 1: CARGAR NUBE DE PUNTOS
# ============================================================================

uploaded_file = None
show_step1 = (not st.session_state.file_loaded) or int(st.session_state.get("step", 1)) == 1
if show_step1:
    st.markdown(
        """
        <div class="ltm-step-intro">
            <div class="ltm-step-eyebrow">Paso 1</div>
            <div class="ltm-step-title">📥 Cargar nube de puntos</div>
            <div class="ltm-step-copy">
                Importa tu archivo LiDAR para comenzar. La aplicación te guiará automáticamente por las fases de clasificación y procesamiento.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_upload, col_help = st.columns([1.55, 1.0], gap="large")
    with col_upload:
        st.markdown(
            """
            <div class="ltm-card">
                <div class="ltm-section-label">Archivo de entrada</div>
                <div class="ltm-card-title-strong">Sube la nube en formato LAZ o LAS</div>
                <div class="ltm-card-copy">
                    Este archivo será la base para revisar clases, preparar la muestra local y generar la malla final.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Archivo LAZ o LAS",
            type=["laz", "las"],
            help="Formatos soportados: .laz, .las",
            label_visibility="collapsed",
        )
        st.caption("Formato admitido: `.laz` y `.las`.")

    with col_help:
        st.markdown(
            """
            <div class="ltm-card">
                <div class="ltm-section-label">🚀 Flujo de trabajo</div>
                <div class="ltm-card-title-strong">¿Qué haremos ahora</div>
                <ol class="ltm-card-list">
                    <li><b>Cargar</b> tu archivo LAZ o LAS.</li>
                    <li><b>Revisar</b> las clases y elegir una zona de prueba.</li>
                    <li><b>Procesar</b> y exportar tu modelo STL final.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.session_state.file_loaded:
            points_loaded = st.session_state.points
            point_count = int(points_loaded.shape[0]) if isinstance(points_loaded, np.ndarray) else 0
            active_classes = 0
            if isinstance(st.session_state.classification, np.ndarray):
                active_classes = int(np.unique(st.session_state.classification).size)
            st.markdown(
                f"""
                <div class="ltm-status-card">
                    <div class="ltm-status-kicker">Proyecto cargado</div>
                    <div class="ltm-status-title">{html.escape(st.session_state.get("input_filename", "Sin nombre"))}</div>
                    <div class="ltm-status-copy">
                        <strong>{point_count:,}</strong> puntos detectados<br>
                        <strong>{active_classes}</strong> clases disponibles
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(
                "Ir a clasificar",
                key="goto_step2_from_step1_btn",
                type="primary",
                use_container_width=True,
            ):
                st.session_state.step = 2
                st.rerun()
        else:
            st.markdown(
                """
                <div class="ltm-status-card">
                    <div class="ltm-status-kicker">Estado</div>
                    <div class="ltm-status-title">Todavia no hay una nube cargada</div>
                    <div class="ltm-status-copy">
                        Sube el archivo en la columna izquierda para activar automticamente el paso de clasificacin.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

if show_step1 and uploaded_file and not st.session_state.file_loaded:
    with st.spinner("Cargando archivo..."):
        points, classification, return_number, num_returns, error = load_laz_file(uploaded_file)

        if error:
            st.error(f"No se pudo leer el archivo: {error}")
        else:
            is_valid, msg = validate_point_cloud_size(points)
            if not is_valid:
                st.error(msg)
            else:
                st.session_state.points = points
                st.session_state.classification = classification
                st.session_state.return_number = return_number
                st.session_state.num_returns = num_returns
                st.session_state.catastro_footprints = None
                st.session_state.catastro_summary = None
                st.session_state.catastro_last_error = ""
                st.session_state.catastro_last_signature = None
                ensure_class_states(np.unique(classification), DEFAULT_EXCLUDED_CLASSES)
                st.session_state.file_loaded = True
                st.session_state.input_filename = uploaded_file.name
                st.session_state.step = 2
                st.rerun()

# ============================================================================
# PASO 2: REVISAR Y CLASIFICAR PUNTOS
# ============================================================================

if st.session_state.file_loaded and st.session_state.step == 2:
    compact_step2 = bool(st.session_state.get("proc_dashboard_layout", True))
    if not compact_step2:
        st.markdown("---")
    with st.container():
        st.markdown(
            """
            <div class="ltm-step-intro">
                <div class="ltm-step-eyebrow">Paso 2</div>
                <div class="ltm-step-title">🔍 Revisar y clasificar</div>
                <div class="ltm-step-copy">
                    Analiza la composición de la nube y selecciona el área que deseas procesar. Usa las herramientas laterales para filtrar las clases.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        points = st.session_state.points
        classification = st.session_state.classification
        ensure_class_states(np.unique(classification), DEFAULT_EXCLUDED_CLASSES)
        step2_cloud_height = 480 if compact_step2 else 620
        step2_picker_height = 420 if compact_step2 else 290
        col_tools, col_view, col_classes = st.columns([0.82, 1.80, 0.98], gap="medium")

        with col_tools:
            st.markdown("##### 🛠️ Herramientas")
            t_c1, t_c2 = st.columns(2)
            max_points = t_c1.slider("Puntos a visualizar", 5_000, 100_000, 30_000, 5_000)
            color_by = t_c2.radio("Color por", ("Clasificación", "Altura"), horizontal=True)

            t_c3, t_c4 = st.columns(2)
            t_c3.radio(
                "Modo selección",
                ("Seleccionar centro en XY/3D", "Solo revisar nube"),
                horizontal=False,
                key="step2_selection_mode_ui",
            )
            t_c4.checkbox(
                "Ocultar ruido visual (clase 7)",
                value=bool(st.session_state.get("step2_hide_noise_visual", False)),
                key="step2_hide_noise_visual",
            )
            if st.button("Limpiar centro de muestra", key="step2_clear_pick_btn"):
                st.session_state.proc_local_preview_has_pick = False
                st.session_state.proc_local_preview_center_mode = "Centro automático"
                st.session_state.proc_local_preview_pick_source = ""

        with col_classes:
            st.markdown("##### 🏷️ Clasificación")
            with st.expander("Resumen de clases", expanded=False):
                unique_classes = np.unique(classification)
                for cls in unique_classes:
                    count = int(np.sum(classification == cls))
                    pct = (count / classification.size) * 100 if classification.size else 0
                    from config.settings import CLASS_NAMES
                    name = CLASS_NAMES.get(cls, f"Clase {cls}")
                    st.text(f"{name}: {count:,} ({pct:.1f}%)")

            with st.expander("Reclasificación rápida por altura", expanded=False):
                if st.checkbox("Activar reclasificación por altura Z", key="height_reclass"):
                    z_relative = points[:, 2] - points[:, 2].min()
                    ground_max = st.slider("Altura max terreno", 0.0, 10.0, 2.0, 0.5)
                    veg_low_max = st.slider("Altura max vegetación baja", 0.0, 20.0, 5.0, 0.5)
                    veg_mid_max = st.slider("Altura max vegetación media", 0.0, 30.0, 15.0, 0.5)

                    is_valid, msg = validate_height_parameters(ground_max, veg_low_max, veg_mid_max)
                    if not is_valid:
                        st.error(msg)

                    if st.button("Aplicar reclasificación", key="apply_height_reclass"):
                        if is_valid:
                            new_class = np.ones_like(classification)
                            new_class[z_relative <= ground_max] = 2
                            mask_low = (z_relative > ground_max) & (z_relative <= veg_low_max)
                            new_class[mask_low] = 3
                            mask_mid = (z_relative > veg_low_max) & (z_relative <= veg_mid_max)
                            new_class[mask_mid] = 4
                            new_class[z_relative > veg_mid_max] = 5
                            st.session_state.classification = new_class
                            st.success("Reclasificación aplicada")
                            st.rerun()

            with st.expander("Filtro visual por clase", expanded=True):
                available_classes = st.session_state.available_classes
                viz_state = st.session_state.viz_class_visibility
                class_counts = {cls: int(np.sum(classification == cls)) for cls in available_classes}
                class_search = st.text_input(
                    "Buscar clase",
                    value=str(st.session_state.get("step2_class_search", "")),
                    key="step2_class_search",
                    placeholder="Ej: terreno, vegetación, 06...",
                )
                search_norm = class_search.strip().lower()
                if search_norm:
                    filtered_classes = [
                        cls for cls in available_classes if search_norm in class_label(cls).lower()
                    ]
                else:
                    filtered_classes = list(available_classes)
                if not filtered_classes:
                    st.info("No hay clases que coincidan con la búsqueda.")
                else:
                    render_class_toggle_buttons(
                        viz_state,
                        "viz_class",
                        filtered_classes,
                        "Activar visibles",
                        "Ocultar visibles",
                    )
                    render_class_checkboxes(viz_state, "viz_class", filtered_classes, class_counts)

        with col_view:
            st.markdown("##### 🌐 Visualización")
            viz_state = st.session_state.viz_class_visibility
            visible_classes = [cls for cls, flag in viz_state.items() if flag]
            if st.session_state.get("step2_hide_noise_visual", False):
                visible_classes = [cls for cls in visible_classes if int(cls) != 7]
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

                picker_limit = 25_000
                if viz_points.shape[0] > picker_limit:
                    picker_idx = np.random.choice(viz_points.shape[0], picker_limit, replace=False)
                    picker_points = viz_points[picker_idx]
                else:
                    picker_points = viz_points

                tab_xy, tab_3d = st.tabs(["Plano XY (selección)", "Nube 3D"])
                with tab_xy:
                    st.caption("Haz clic en XY para fijar el centro de muestra del paso 3.")
                    fig_xy_picker = build_xy_picker_figure(
                        picker_points[:, :2],
                        picker_points[:, 2],
                        uirevision_id="step2-xy-picker",
                        height_px=step2_picker_height,
                    )
                    picker_state = st.plotly_chart(
                        fig_xy_picker,
                        use_container_width=True,
                        key="plot_step2_xy_picker",
                        on_select="rerun",
                        selection_mode=("points",),
                    )
                    selected_xy = extract_first_selected_xy(picker_state)
                    if selected_xy is None:
                        selected_idx = extract_first_selected_index(picker_state)
                        if selected_idx is not None and 0 <= selected_idx < picker_points.shape[0]:
                            selected_xy = (
                                float(picker_points[selected_idx, 0]),
                                float(picker_points[selected_idx, 1]),
                            )
                    if selected_xy is not None and st.session_state.get("step2_selection_mode_ui") != "Solo revisar nube":
                        center_changed = apply_local_preview_center_pick(
                            selected_xy,
                            source_label="Plano XY (Paso 2)",
                        )
                        if center_changed:
                            picked_x, picked_y = selected_xy
                            st.success(f"Centro guardado: X={picked_x:.2f}, Y={picked_y:.2f}")
                    elif st.session_state.get("proc_local_preview_has_pick", False):
                        st.info(
                            "Centro de muestra actual: "
                            f"X={float(st.session_state.get('proc_local_preview_center_x', 0.0)):.2f}, "
                            f"Y={float(st.session_state.get('proc_local_preview_center_y', 0.0)):.2f}"
                        )

                fig = go.Figure()
                if color_by == "Clasificación":
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

                with tab_3d:
                    if fig is not None:
                        st.caption("Selecciona un punto de la nube 3D para usarlo como centro de la previsualización.")
                        fig.update_layout(
                            scene=dict(
                                xaxis_title="X (m)",
                                yaxis_title="Y (m)",
                                zaxis_title="Z (m)",
                                aspectmode="data",
                                uirevision="step2-cloud",
                            ),
                            uirevision="step2-cloud",
                            clickmode="event+select",
                            legend=dict(orientation="h"),
                            height=step2_cloud_height,
                            margin=dict(l=0, r=0, b=0, t=20),
                        )
                        cloud_state = st.plotly_chart(
                            fig,
                            use_container_width=True,
                            key="plot_step2_point_cloud",
                            on_select="rerun",
                            selection_mode=("points",),
                        )
                        selected_cloud_xy = extract_first_selected_xy(cloud_state)
                        if selected_cloud_xy is None:
                            selected_cloud_idx = extract_first_selected_index(cloud_state)
                            if selected_cloud_idx is not None and 0 <= selected_cloud_idx < viz_points.shape[0]:
                                selected_cloud_xy = (
                                    float(viz_points[selected_cloud_idx, 0]),
                                    float(viz_points[selected_cloud_idx, 1]),
                                )
                        if (
                            selected_cloud_xy is not None
                            and st.session_state.get("step2_selection_mode_ui") != "Solo revisar nube"
                        ):
                            center_changed = apply_local_preview_center_pick(
                                selected_cloud_xy,
                                source_label="Nube 3D (Paso 2)",
                            )
                            if center_changed:
                                picked_x, picked_y = selected_cloud_xy
                                st.success(
                                    f"Centro guardado desde nube 3D: X={picked_x:.2f}, Y={picked_y:.2f}"
                                )
                    else:
                        st.info("Las clases seleccionadas no tienen puntos en el muestreo actual.")
        
        # --- Navegación Paso 2 -> 3 ---
        st.markdown("---")
        nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])
        if nav_col1.button("⬅️ Volver a Cargar", use_container_width=True):
            st.session_state.step = 1
            st.rerun()
        
        if nav_col3.button("Continuar a Procesar ➡️", type="primary", use_container_width=True):
            st.session_state.step = 3
            st.rerun()

# ============================================================================
# PASO 3: CONFIGURAR Y PROCESAR MALLA
# ============================================================================

if st.session_state.file_loaded and st.session_state.step == 3:
    points = st.session_state.points
    classification = st.session_state.classification
    ensure_class_states(np.unique(classification), DEFAULT_EXCLUDED_CLASSES)
    st.session_state.proc_dashboard_layout = True
    active_proc_classes = [
        cls for cls, enabled in st.session_state.proc_class_filter.items() if enabled
    ]
    st.markdown("### Paso 3 - Procesar malla")
    st.caption("Controles a la izquierda, previsualización a la derecha. En básico solo ves lo imprescindible.")

    step3_controls_col, step3_preview_col = st.columns([1.08, 0.92], gap="medium")
    with step3_preview_col:
        st.markdown('<div class="ltm-preview-column-marker"></div>', unsafe_allow_html=True)
        preview_panel = st.container()
        cloud_panel = None

    with step3_controls_col:
        st.markdown('<div class="ltm-controls-scroll-marker"></div>', unsafe_allow_html=True)
        step3_top_mode_col, step3_top_info_col, step3_top_action_col = st.columns(
            [1.05, 1.10, 0.85],
            gap="small",
        )
        ui_mode_label = step3_top_mode_col.radio(
            "Nivel de control",
            ("Basico", "Avanzado"),
            horizontal=True,
            key="proc_ui_mode",
        )
        expert_mode_enabled = ui_mode_label == "Avanzado"
        st.session_state.proc_expert_mode = expert_mode_enabled
        workflow_mode = (
            "Avanzado (todos los controles)"
            if expert_mode_enabled
            else "Facil (3 botones)"
        )
        st.session_state.proc_workflow_mode = workflow_mode
        step3_top_info_col.markdown(
            f'''
            <div class="ltm-card">
                <div class="ltm-card-title">Estado actual</div>
                <div class="ltm-card-text">
                    {points.shape[0]:,} puntos cargados<br>
                    {len(active_proc_classes)} clases activas<br>
                    Escala actual: {html.escape(str(st.session_state.get("proc_scale_preset", "1:1 000")))}
                </div>
            </div>
            ''',
            unsafe_allow_html=True,
        )
        header_generate_requested = bool(st.session_state.get("proc_generate_from_header", False))
        generate_requested = step3_top_action_col.button(
            "Generar STL final",
            type="primary",
            use_container_width=True,
            key="proc_generate_final_top_btn",
        ) or header_generate_requested
        if header_generate_requested:
            st.session_state.proc_generate_from_header = False

        if expert_mode_enabled:
            controls_basic_host = st.expander("🛠️ General y Opciones Globales", expanded=True)
            controls_terrain_host = st.expander("⛰️ Terreno y TIN", expanded=False)
            controls_vegetation_host = st.expander("🌳 Vegetación y Acabado", expanded=False)
            controls_buildings_host = st.expander("🏢 Edificios y Catastro", expanded=False)
            controls_layers_host = st.expander("📑 Capas de Generación", expanded=False)
            controls_output_host = st.expander("🚀 Vista Previa y Exportación", expanded=False)

            controls_basic_host.caption("Decisiones globales del modelo.")
            controls_terrain_host.caption("Terreno, TIN y resolución principal.")
            controls_vegetation_host.caption("Masas vegetales y acabado superficial.")
            controls_buildings_host.caption("Edificios, Catastro y alturas manuales.")
            controls_layers_host.caption("Activa o apaga capas antes de generar.")
            controls_output_host.caption("Preview, base, limpieza y exportación.")
        else:
            controls_basic_host = st.expander("🛠️ General", expanded=True)
            controls_layers_host = st.expander("📑 Capas", expanded=False)
            controls_output_host = st.expander("🚀 Vista y salida", expanded=False)
            
            controls_terrain_host = controls_basic_host
            controls_vegetation_host = controls_basic_host
            controls_buildings_host = controls_basic_host
            controls_basic_host.info(
                "Modo básico: solo lo esencial. Para terreno, vegetación y Catastro con detalle, cambia a Avanzado."
            )

        controls_status_host = st.container()
        basic_controls = controls_basic_host.container()
    
    scale_options = list(SCALE_PRESETS.keys()) + ["Personalizado"]
    default_scale = st.session_state.get("proc_scale_preset", "1:1 000")
    if default_scale not in scale_options:
        default_scale = "Personalizado"
    selected_scale = basic_controls.selectbox(
        "Escala objetivo (modelo : terreno real)",
        options=scale_options,
        index=scale_options.index(default_scale),
    )
    st.session_state.proc_scale_preset = selected_scale
    preset_values = SCALE_PRESETS.get(selected_scale)
    scale_ratio_value = SCALE_RATIOS.get(selected_scale)
    if scale_ratio_value is not None:
        min_feature = (FILAMENT_NOZZLE_MM / 1000.0) * scale_ratio_value
        extent_x_m = float(np.ptp(points[:, 0]))
        extent_y_m = float(np.ptp(points[:, 1]))
        extent_z_m = float(np.ptp(points[:, 2]))
        scale_divisor = float(scale_ratio_value)
        model_x_mm = (extent_x_m * 1000.0) / scale_divisor
        model_y_mm = (extent_y_m * 1000.0) / scale_divisor
        model_z_mm = (extent_z_m * 1000.0) / scale_divisor
        basic_controls.caption(
            f"Detalle m?nimo imprimible ~ {min_feature:.3f} m sobre el terreno (boquilla {FILAMENT_NOZZLE_MM} mm)."
        )
        basic_controls.caption(
            "Tamano estimado del modelo: "
            f"X {model_x_mm:.1f} mm ? Y {model_y_mm:.1f} mm ? Z {model_z_mm:.1f} mm "
            "(sin base ni recortes)."
        )
        controls_basic_host.markdown(
            f"""
            <div class="ltm-card">
                <div class="ltm-card-title">Resumen de escala</div>
                <div class="ltm-card-text">
                    Escala {html.escape(str(selected_scale))}<br>
                    Pieza estimada: X {model_x_mm:.1f} mm | Y {model_y_mm:.1f} mm | Z {model_z_mm:.1f} mm<br>
                    Detalle minimo imprimible aproximado: {min_feature:.3f} m
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    

    large_cloud = points.shape[0] >= 1_500_000
    easy_profile: dict[str, float | str] | None = None
    if workflow_mode == "Facil (3 botones)":
        recommended_easy_quality = recommend_easy_quality(points.shape[0], scale_ratio_value)
        easy_quality_options = ("Rapida", "Equilibrada", "Alta")
        current_easy_quality = st.session_state.get("proc_easy_quality", recommended_easy_quality)
        if current_easy_quality not in easy_quality_options:
            current_easy_quality = recommended_easy_quality
            st.session_state.proc_easy_quality = recommended_easy_quality

        basic_controls.caption("Calidad (1 clic)")
        easy_col_a, easy_col_b, easy_col_c = basic_controls.columns(3)
        if easy_col_a.button("Rapida", use_container_width=True, key="proc_easy_quality_fast_btn"):
            st.session_state.proc_easy_quality = "Rapida"
        if easy_col_b.button("Equilibrada", use_container_width=True, key="proc_easy_quality_balanced_btn"):
            st.session_state.proc_easy_quality = "Equilibrada"
        if easy_col_c.button("Alta", use_container_width=True, key="proc_easy_quality_high_btn"):
            st.session_state.proc_easy_quality = "Alta"

        easy_quality = st.session_state.get("proc_easy_quality", recommended_easy_quality)
        basic_controls.caption(
            f"Calidad seleccionada: {easy_quality}"
            + (" (recomendada)" if easy_quality == recommended_easy_quality else "")
        )
        easy_profile = build_easy_profile(easy_quality, points.shape[0], scale_ratio_value)
        processing_mode = str(easy_profile["processing_mode"])
        basic_controls.success(f"Modo autom?tico aplicado: {processing_mode}")
    else:
        default_mode_index = 1 if large_cloud else 0
        processing_mode = basic_controls.selectbox(
            "Modo de generación",
            ("Interpolación en rejilla", "Modo avanzado (detalle máximo)"),
            index=default_mode_index,
        )
        if large_cloud:
            basic_controls.info(
                "Nube muy grande detectada: se recomienda modo avanzado o usar resoluciones más altas "
                "(menos detalle) para acelerar."
            )

    smooth_ground_default = DEFAULT_SMOOTH_GROUND
    smooth_vegetation_default = DEFAULT_SMOOTH_VEGETATION
    smooth_buildings_default = DEFAULT_SMOOTH_BUILDINGS
    mesh_resolution_default = DEFAULT_MESH_RESOLUTION
    interpolation_method = DEFAULT_INTERPOLATION_METHOD
    sphere_radius_default = DEFAULT_SPHERE_RADIUS
    terrain_resolution_default = DEFAULT_TERRAIN_RESOLUTION
    metaball_grid_default = DEFAULT_METABALL_GRID
    metaball_threshold_default = DEFAULT_METABALL_THRESHOLD
    metaball_max_dim = DEFAULT_METABALL_MAX_DIM
    building_cluster_radius = float(DEFAULT_BUILDING_CLUSTER_RADIUS)
    building_min_points = max(int(DEFAULT_BUILDING_MIN_POINTS), 8)
    building_roof_percentile = DEFAULT_BUILDING_ROOF_PERCENTILE
    building_base_percentile = DEFAULT_BUILDING_BASE_PERCENTILE
    building_split_touching_clusters = True
    building_orthogonalize_edges = False
    building_snap_roof_planes = True
    building_roof_model_mode = "dominant_planes"
    building_min_roof_plane_area_m2 = 10.0
    building_density_hint = 1.0
    voxel_simplify_default = DEFAULT_VOXEL_SIMPLIFY
    max_vertical_error_default = DEFAULT_MAX_VERTICAL_ERROR
    max_slope_default = DEFAULT_MAX_SLOPE_DEG
    building_plane_tolerance_default = DEFAULT_BUILDING_PLANE_TOLERANCE
    base_thickness_default = DEFAULT_BASE_THICKNESS
    vegetation_mode = "realistic"
    vegetation_style_min_area = 12.0
    vegetation_style_detail = 2
    vegetation_style_include_low = False
    vegetation_style_max_clusters = 500
    vegetation_style_link_to_scale = True
    vegetation_style_texture_mm = 0.9
    vegetation_style_relief_mm = 1.2
    vegetation_style_texture_pitch = 2.0
    vegetation_style_relief_cap = 3.0
    vegetation_style_roughness = 0.45
    vegetation_style_density_response = 1.0
    vegetation_style_min_density = 0.04
    vegetation_mass_cell_size = 0.8
    vegetation_mass_include_low = False
    vegetation_mass_min_height = 0.9
    vegetation_mass_min_density = 0.03
    vegetation_mass_min_ratio = 0.16
    vegetation_mass_min_patch_area = 12.0
    vegetation_mass_close_radius = 1.2
    vegetation_mass_open_radius = 0.6
    vegetation_mass_texture_pitch = 1.1
    vegetation_mass_roughness = 0.58
    vegetation_mass_density_response = 1.0
    vegetation_mass_relief_cap = 2.6
    vegetation_mass_relief_floor = 0.45
    vegetation_mass_base_embed = 0.18
    vegetation_mass_edge_softness = 1.0
    vegetation_mass_organic_smooth = 0.9
    vegetation_mass_micro_detail = 0.85
    vegetation_mass_max_cells = 450_000
    building_source = "laz"
    building_footprints: list[dict] | None = None
    building_footprint_height_mode = "fixed"
    building_footprint_height_field: str | None = None
    building_footprint_floors_field: str | None = None
    building_footprint_fixed_height = 12.0
    building_footprint_fallback_height = 8.0
    building_footprint_floor_height = 3.0
    building_footprint_min_height = 2.0
    building_footprint_max_height = 120.0
    building_footprint_min_area = 8.0
    building_footprint_simplify_tolerance = 0.0
    building_footprint_max_vertices = 150
    building_footprint_source_epsg: int | None = None
    building_footprint_target_epsg: int | None = None
    building_footprint_shape_mode = "adaptive_mix"
    building_footprint_outside_area_threshold = 30.0
    building_footprint_epsg_valid = True
    manual_height_editor_items: list[dict] = []
    manual_height_key_order: list[str] = []
    building_plane_tolerance = building_plane_tolerance_default
    building_clean_class6_noise = False
    building_class6_noise_aggressiveness = 1.0
    reassign_overlap_class12 = bool(st.session_state.get("proc_reassign_overlap_class12", True))
    overlap_assign_to_buildings = bool(st.session_state.get("proc_overlap_assign_to_buildings", False))
    max_open_edges_tolerance = 5

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
        building_plane_tolerance_default = preset_values.get(
            "building_plane_tolerance", building_plane_tolerance_default
        )
        base_thickness_default = preset_values["base_thickness"]

    if easy_profile is not None:
        mesh_resolution_default = float(mesh_resolution_default) * float(easy_profile["resolution_mult"])
        terrain_resolution_default = float(terrain_resolution_default) * float(easy_profile["terrain_mult"])
        voxel_simplify_default = float(voxel_simplify_default) * float(easy_profile["voxel_mult"])
        max_vertical_error_default = float(max_vertical_error_default) * float(easy_profile["vertical_error_mult"])
        max_slope_default = float(max_slope_default) + float(easy_profile["slope_delta"])
        sphere_radius_default = float(sphere_radius_default) * float(easy_profile["sphere_mult"])
        metaball_grid_default = float(metaball_grid_default) * float(easy_profile["metaball_grid_mult"])
        metaball_threshold_default = float(metaball_threshold_default) * float(
            easy_profile["metaball_threshold_mult"]
        )
        smooth_ground_default = float(smooth_ground_default) * float(easy_profile["smooth_mult"])
        smooth_vegetation_default = float(smooth_vegetation_default) * float(easy_profile["smooth_mult"])
        smooth_buildings_default = float(smooth_buildings_default) * float(easy_profile["smooth_mult"])
        building_plane_tolerance_default = float(building_plane_tolerance_default) * float(
            easy_profile["building_plane_mult"]
        )
        vegetation_mode = str(easy_profile.get("vegetation_mode", vegetation_mode))

    if processing_mode == "Modo avanzado (detalle máximo)" and workflow_mode == "Avanzado (todos los controles)":
        st.info(
            "\n".join(
                [
                    "Workflow profesional activado:",
                    "- Paso 1 - Filtrado de terreno real (clase 2, retornos finales, outliers).",
                    "- Paso 2 - Progressive TIN densification que preserva detalle topográfico.",
                    "- Paso 3 - Vegetación realista o estilizada para maqueta y edificios extruidos.",
                    "- Paso 4 - Fusión de capas manteniendo geometría independiente.",
                ]
            )
        )

    smooth_ground = snap_slider_value(smooth_ground_default, 0.0, 3.0, 0.1)
    smooth_vegetation = snap_slider_value(smooth_vegetation_default, 0.0, 5.0, 0.1)
    smooth_buildings = snap_slider_value(smooth_buildings_default, 0.0, 3.0, 0.1)
    mesh_resolution = snap_slider_value(mesh_resolution_default, 0.5, 25.0, 0.5)
    terrain_resolution = snap_slider_value(terrain_resolution_default, 0.5, 20.0, 0.5)
    max_vertical_error = snap_slider_value(max_vertical_error_default, 0.1, 12.0, 0.1)
    max_slope_deg = snap_slider_value(max_slope_default, 10.0, 60.0, 1.0)
    sphere_radius = snap_slider_value(sphere_radius_default, 0.2, 2.0, 0.1)
    metaball_grid = snap_slider_value(metaball_grid_default, 0.1, 2.0, 0.1)
    metaball_threshold = snap_slider_value(metaball_threshold_default, 0.2, 1.5, 0.1)
    metaball_max_dim = int(np.clip(int(round(metaball_max_dim)), 30, 200))
    building_cluster_radius = snap_slider_value(building_cluster_radius, 0.5, 10.0, 0.25)
    building_roof_percentile = snap_slider_value(building_roof_percentile, 60.0, 100.0, 5.0)
    building_base_percentile = snap_slider_value(building_base_percentile, 0.0, 40.0, 5.0)
    building_min_points = int(np.clip(int(round(building_min_points)), 4, 60))
    building_plane_tolerance = snap_slider_value(building_plane_tolerance_default, 0.05, 0.4, 0.01)

    if workflow_mode == "Facil (3 botones)":
        basic_controls.info("✨ **Modo fácil activado**: Hemos configurado todos los detalles técnicos por ti. Solo tienes que elegir el tipo de resultado que buscas.")
        if processing_mode.startswith("Interpol"):
            interpolation_method = "linear"
            terrain_resolution = mesh_resolution
        else:
            vegetation_mode = str(easy_profile.get("vegetation_mode", "printable_mass"))
        building_source = "laz"
    elif processing_mode.startswith("Interpol"):
        with controls_terrain_host.container():
            st.markdown("**Terreno**")
            st.subheader("Suavizado por clase")
            c_sm1, c_sm2, c_sm3 = st.columns(3)
            smooth_ground = c_sm1.slider(
                "Suavizado terreno",
                0.0,
                3.0,
                snap_slider_value(smooth_ground_default, 0.0, 3.0, 0.1),
                0.1,
            )
            smooth_vegetation = c_sm2.slider(
                "Suavizado vegetación",
                0.0,
                5.0,
                snap_slider_value(smooth_vegetation_default, 0.0, 5.0, 0.1),
                0.1,
            )
            smooth_buildings = c_sm3.slider(
                "Suavizado edificios",
                0.0,
                3.0,
                snap_slider_value(smooth_buildings_default, 0.0, 3.0, 0.1),
                0.1,
            )
            st.subheader("Generación de malla")
            c_res1, c_res2 = st.columns(2)
            mesh_resolution = c_res1.slider(
                "Resolución (m)",
                0.5,
                25.0,
                snap_slider_value(mesh_resolution_default, 0.5, 25.0, 0.5),
                0.5,
            )
            interpolation_options = ["linear", "cubic", "nearest"]
            interpolation_method = c_res2.selectbox(
                "Interpolación",
                interpolation_options,
                index=interpolation_options.index(DEFAULT_INTERPOLATION_METHOD)
                if DEFAULT_INTERPOLATION_METHOD in interpolation_options
                else 0,
            )
            terrain_resolution = mesh_resolution
            max_vertical_error = max_vertical_error_default
            max_slope_deg = max_slope_default
            building_plane_tolerance = building_plane_tolerance_default
        building_source = "laz"
    else:
        with controls_vegetation_host.container():
            st.markdown("**Vegetacion**")
            st.subheader("Parámetros avanzados")
            vegetation_mode_options = (
                "Masa imprimible (recomendada)",
                "Estilizada maqueta",
                "Realista (Poisson/metaballs)",
            )
            vegetation_mode_index = 0
            if vegetation_mode == "stylized":
                vegetation_mode_index = 1
            elif vegetation_mode == "realistic":
                vegetation_mode_index = 2
            vegetation_mode_label = st.selectbox(
                "Modelo de vegetación",
                vegetation_mode_options,
                index=vegetation_mode_index,
            )
            if "Masa imprimible" in vegetation_mode_label:
                vegetation_mode = "printable_mass"
            if vegetation_mode == "printable_mass":
                st.caption("Genera masas compactas tipo musgo para impresin 3D.")
                v_col1, v_col2 = st.columns(2)
                vegetation_mass_cell_size = v_col1.slider(
                    "Tamano de celda de masa (m)",
                    0.2,
                    8.0,
                    snap_slider_value(vegetation_mass_cell_size, 0.2, 8.0, 0.1),
                    0.1,
                    help="Celda más pequeña = más detalle y más tiempo.",
                )
                vegetation_mass_min_patch_area = v_col2.slider(
                    "rea mnima de masa (m)",
                    1.0,
                    400.0,
                    snap_slider_value(vegetation_mass_min_patch_area, 1.0, 400.0, 1.0),
                    1.0,
                    help="Elimina manchas sueltas pequeas.",
                )
                
                v_col3, v_col4 = st.columns(2)
                vegetation_mass_include_low = v_col3.checkbox(
                    "Incluir vegetación baja (clase 3)",
                    value=vegetation_mass_include_low,
                )
                vegetation_mass_min_height = v_col4.slider(
                    "Altura mnima sobre terreno (m)",
                    0.1,
                    6.0,
                    snap_slider_value(vegetation_mass_min_height, 0.1, 6.0, 0.1),
                    0.1,
                    help="Filtra zonas con poco relieve vegetal.",
                )
                
                v_col5, v_col6 = st.columns(2)
                vegetation_mass_min_density = v_col5.slider(
                    "Densidad mnima (pts/m)",
                    0.005,
                    0.8,
                    snap_slider_value(vegetation_mass_min_density, 0.005, 0.8, 0.005),
                    0.005,
                    help="Controla cuanto LiDAR vegetal hace falta para crear masa.",
                )
                vegetation_mass_min_ratio = v_col6.slider(
                    "Fraccion mnima de puntos vegetales",
                    0.02,
                    0.95,
                    snap_slider_value(vegetation_mass_min_ratio, 0.02, 0.95, 0.01),
                    0.01,
                    help="Exige que una celda tenga proporción suficiente de puntos de vegetación.",
                )
                
                v_col7, v_col8 = st.columns(2)
                vegetation_mass_close_radius = v_col7.slider(
                    "Cierre de huecos (m)",
                    0.0,
                    8.0,
                    snap_slider_value(vegetation_mass_close_radius, 0.0, 8.0, 0.1),
                    0.1,
                    help="Une zonas cercanas para formar masas continuas.",
                )
                vegetation_mass_open_radius = v_col8.slider(
                    "Apertura de ruido (m)",
                    0.0,
                    6.0,
                    snap_slider_value(vegetation_mass_open_radius, 0.0, 6.0, 0.1),
                    0.1,
                    help="Limpia picos o puntas aisladas.",
                )
                
                v_col9, v_col10 = st.columns(2)
                vegetation_mass_texture_pitch = v_col9.slider(
                    "Paso de textura de masa (m)",
                    0.2,
                    20.0,
                    snap_slider_value(vegetation_mass_texture_pitch, 0.2, 20.0, 0.1),
                    0.1,
                )
                vegetation_mass_roughness = v_col10.slider(
                    "Rugosidad de masa",
                    0.0,
                    1.0,
                    snap_slider_value(vegetation_mass_roughness, 0.0, 1.0, 0.05),
                    0.05,
                )
                
                v_col11, v_col12 = st.columns(2)
                vegetation_mass_density_response = v_col11.slider(
                    "Respuesta a densidad LiDAR",
                    0.0,
                    2.0,
                    snap_slider_value(vegetation_mass_density_response, 0.0, 2.0, 0.1),
                    0.1,
                )
                vegetation_mass_relief_cap = v_col12.slider(
                    "Relieve maximo de masa (m)",
                    0.1,
                    12.0,
                    snap_slider_value(vegetation_mass_relief_cap, 0.1, 12.0, 0.1),
                    0.1,
                )
                
                v_col13, v_col14 = st.columns(2)
                vegetation_mass_relief_floor = v_col13.slider(
                    "Relieve base de masa (m)",
                    0.05,
                    6.0,
                    snap_slider_value(vegetation_mass_relief_floor, 0.05, 6.0, 0.05),
                    0.05,
                )
                vegetation_mass_base_embed = v_col14.slider(
                    "Anclaje bajo terreno (m)",
                    0.02,
                    3.0,
                    snap_slider_value(vegetation_mass_base_embed, 0.02, 3.0, 0.02),
                    0.02,
                    help="Hace que la masa quede slida y bien fusinada con el terreno.",
                )
                
                v_col15, v_col16 = st.columns(2)
                vegetation_mass_edge_softness = v_col15.slider(
                    "Suavidad de borde (m)",
                    0.1,
                    6.0,
                    snap_slider_value(vegetation_mass_edge_softness, 0.1, 6.0, 0.1),
                    0.1,
                    help="Controla cuanto se redondea la transicion en el limite de la masa vegetal.",
                )
                vegetation_mass_organic_smooth = v_col16.slider(
                    "Suavizado organico interno (m)",
                    0.0,
                    6.0,
                    snap_slider_value(vegetation_mass_organic_smooth, 0.0, 6.0, 0.1),
                    0.1,
                    help="Reduce aspecto cuadriculado mezclando alturas entre celdas vecinas.",
                )
                
                v_col17, v_col18 = st.columns(2)
                vegetation_mass_micro_detail = v_col17.slider(
                    "Microdetalle de superficie",
                    0.0,
                    1.8,
                    snap_slider_value(vegetation_mass_micro_detail, 0.0, 1.8, 0.05),
                    0.05,
                    help="Añade variación fina multiescala para un aspecto más natural.",
                )
                vegetation_mass_max_cells = int(
                    v_col18.slider(
                        "Maximo celdas de mascara",
                        80_000,
                        1_000_000,
                        int(vegetation_mass_max_cells),
                        20_000,
                        help="Limita memoria/tiempo en nubes muy grandes.",
                    )
                )
            elif vegetation_mode == "stylized":
                v_st_c1, v_st_c2 = st.columns(2)
                vegetation_style_min_area = v_st_c1.slider(
                    "Área mínima masa vegetación (m²)",
                    2.0,
                    300.0,
                    snap_slider_value(vegetation_style_min_area, 2.0, 300.0, 1.0),
                    1.0,
                    help="Evita ruido creando vegetación solo en masas con área suficiente.",
                )
                vegetation_style_detail = int(
                    v_st_c2.slider(
                        "Detalle vegetación estilizada",
                        1,
                        3,
                        vegetation_style_detail,
                        1,
                        help="1 = muy rápido, 3 = más detalle.",
                    )
                )
                v_st_c3, v_st_c4 = st.columns(2)
                vegetation_style_include_low = v_st_c3.checkbox(
                    "Incluir vegetación baja estilizada",
                    value=vegetation_style_include_low,
                    help="Si se desactiva, la clase 3 no genera volumen para no ensuciar el terreno.",
                )
                vegetation_style_max_clusters = int(
                    v_st_c4.slider(
                        "Máximo masas vegetación",
                        100,
                        1500,
                        vegetation_style_max_clusters,
                        50,
                        help="Limita el número de grupos para controlar tiempo y tamaño.",
                    )
                )
                st.markdown("**Textura de masa vegetal (tipo musgo)**")
                vegetation_style_link_to_scale = st.checkbox(
                    "Vincular textura a escala del modelo",
                    value=vegetation_style_link_to_scale,
                    help="Convierte mm de maqueta a metros reales segun escala, editable con ajuste fino.",
                )
                if vegetation_style_link_to_scale:
                    active_scale_ratio = float(scale_ratio_value) if scale_ratio_value is not None else 1000.0
                    vegetation_style_texture_mm = st.slider(
                        "Paso textura en maqueta (mm)",
                        0.2,
                        3.0,
                        snap_slider_value(vegetation_style_texture_mm, 0.2, 3.0, 0.05),
                        0.05,
                        help="Más bajo = textura más fina; más alto = bulto más grande.",
                    )
                    vegetation_style_relief_mm = st.slider(
                        "Relieve de masa en maqueta (mm)",
                        0.2,
                        4.0,
                        snap_slider_value(vegetation_style_relief_mm, 0.2, 4.0, 0.05),
                        0.05,
                        help="Altura mxima de la masa vegetal impresa respecto al terreno.",
                    )
                    style_tune = st.slider(
                        "Ajuste fino de textura (x)",
                        0.5,
                        2.0,
                        1.0,
                        0.05,
                        help="Multiplicador final para afinar sin tocar mm base.",
                    )
                    vegetation_style_texture_pitch = max(
                        float((vegetation_style_texture_mm / 1000.0) * active_scale_ratio * style_tune),
                        0.08,
                    )
                    vegetation_style_relief_cap = max(
                        float((vegetation_style_relief_mm / 1000.0) * active_scale_ratio * style_tune),
                        0.12,
                    )
                    st.caption(
                        f"Equivale aprox. a paso {vegetation_style_texture_pitch:.2f} m y relieve {vegetation_style_relief_cap:.2f} m en terreno."
                    )
                else:
                    vegetation_style_texture_pitch = st.slider(
                        "Paso textura (m terreno)",
                        0.1,
                        20.0,
                        snap_slider_value(vegetation_style_texture_pitch, 0.1, 20.0, 0.1),
                        0.1,
                    )
                    vegetation_style_relief_cap = st.slider(
                        "Relieve maximo masa (m terreno)",
                        0.1,
                        12.0,
                        snap_slider_value(vegetation_style_relief_cap, 0.1, 12.0, 0.1),
                        0.1,
                    )

                v_st_c5, v_st_c6 = st.columns(2)
                vegetation_style_roughness = v_st_c5.slider(
                    "Rugosidad superficial",
                    0.0,
                    1.0,
                    snap_slider_value(vegetation_style_roughness, 0.0, 1.0, 0.05),
                    0.05,
                    help="0 = masa lisa, 1 = textura más rugosa u orgánica.",
                )
                vegetation_style_density_response = v_st_c6.slider(
                    "Respuesta a densidad LiDAR",
                    0.0,
                    2.0,
                    snap_slider_value(vegetation_style_density_response, 0.0, 2.0, 0.1),
                    0.1,
                    help="Sube este valor para que masas densas salgan más compactas y con más relieve.",
                )
                vegetation_style_min_density = st.slider(
                    "Densidad mnima de masa (pts/m²)",
                    0.005,
                    0.50,
                    snap_slider_value(vegetation_style_min_density, 0.005, 0.50, 0.005),
                    0.005,
                    help="Filtra manchas debiles para evitar 'islas' de vegetacin falsa.",
                )
        with controls_terrain_host.container():
                    st.markdown("**Terreno**")
                    m_c1, m_c2 = st.columns(2)
                    sphere_radius = m_c1.slider(
                        "Radio de esfera para vegetación (m)",
                        0.2,
                        2.0,
                        snap_slider_value(sphere_radius_default, 0.2, 2.0, 0.1),
                        0.1,
                    )
                    terrain_resolution = m_c2.slider(
                        "Tamaño celda seed TIN (m)",
                        0.5,
                        20.0,
                        snap_slider_value(terrain_resolution_default, 0.5, 20.0, 0.5),
                        0.5,
                    )
                    m_c3, m_c4 = st.columns(2)
                    max_vertical_error = m_c3.slider(
                        "Tolerancia vertical TIN (m)",
                        0.1,
                        12.0,
                        snap_slider_value(max_vertical_error_default, 0.1, 12.0, 0.1),
                        0.1,
                    )
                    max_slope_deg = m_c4.slider(
                        "Pendiente máxima TIN (deg)",
                        10.0,
                        60.0,
                        snap_slider_value(max_slope_default, 10.0, 60.0, 1.0),
                        1.0,
                    )
                    m_c5, m_c6 = st.columns(2)
                    metaball_grid = m_c5.slider(
                        "Resolución grid vegetación (m)",
                        0.1,
                        2.0,
                        snap_slider_value(metaball_grid_default, 0.1, 2.0, 0.1),
                        0.1,
                    )
                    metaball_threshold = m_c6.slider(
                        "Umbral fusión vegetación",
                        0.2,
                        1.5,
                        snap_slider_value(metaball_threshold_default, 0.2, 1.5, 0.1),
                        0.1,
                    )
                    metaball_max_dim = st.slider(
                        "Límite celdas por eje", 30, 200, 100, 10
                    )
        controls_buildings_host.markdown(
            """
            <div class="ltm-card">
                <div class="ltm-card-title">Edificios desde Catastro</div>
                <div class="ltm-card-text">
                    Para usar huellas catastrales: abre este bloque y cambia <strong>Fuente de edificios</strong>
                    a <strong>Desde huellas externas (GeoJSON/Catastro)</strong>.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with controls_buildings_host.container():
                    st.markdown("**Edificios**")
                    building_source_label = st.radio(
                        "Fuente de edificios",
                        ("Desde clase 6 del LAZ", "Desde huellas externas (GeoJSON/Catastro)"),
                        index=0,
                        help="Cambia entre reconstruccion por puntos LAZ o extrusion por huellas.",
                    )
                    building_source = "footprints" if "huellas" in building_source_label else "laz"
                    if building_source == "laz":
                        st.caption("Filtros para puntos de edificio (Clase 6).")
                        laz_tab_roof, laz_tab_definition, laz_tab_cleanup = st.tabs(
                            ["Altura y cubierta", "Definicion y union", "Limpieza"]
                        )
                        with laz_tab_roof:
                            roof_mode_options_laz = {
                                "Interseccion de planos (recomendado)": "dominant_planes",
                                "Automatico": "auto",
                                "Superficie libre": "freeform",
                                "Plano unico": "single_plane",
                                "Cubierta plana": "flat",
                            }
                            current_roof_mode_label_laz = next(
                                (
                                    label
                                    for label, value in roof_mode_options_laz.items()
                                    if value == building_roof_model_mode
                                ),
                                "Interseccion de planos (recomendado)",
                            )
                            building_roof_mode_label_laz = st.selectbox(
                                "Modelo de cubierta",
                                options=list(roof_mode_options_laz.keys()),
                                index=list(roof_mode_options_laz.keys()).index(current_roof_mode_label_laz),
                                help=(
                                    "Interseccion de planos detecta varios faldones (p. ej. cubierta a dos aguas) "
                                    "y mantiene solo los planos estables."
                                ),
                            )
                            building_roof_model_mode = roof_mode_options_laz[building_roof_mode_label_laz]
                            b_c1, b_c2 = st.columns(2)
                            building_roof_percentile = b_c1.slider(
                                "Percentil techo edificio (%)",
                                60.0,
                                100.0,
                                snap_slider_value(90.0, 60.0, 100.0, 5.0),
                                5.0,
                                help=(
                                    "Cuanto ms alto, ms se apoya el algoritmo en los puntos ms altos de la cubierta.\n\n"
                                    "Ejemplo: 90-95% va bien para tejados claros; 75-85% ayuda si la nube es rala."
                                ),
                            )
                            building_base_percentile = b_c2.slider(
                                "Percentil base edificio (%)",
                                0.0,
                                40.0,
                                snap_slider_value(10.0, 0.0, 40.0, 5.0),
                                5.0,
                                help=(
                                    "Solo se usa como respaldo cuando no hay terreno fiable bajo el edificio.\n\n"
                                    "Ejemplo: 10% suele dar una base estable; subelo si el edificio sale demasiado enterrado."
                                ),
                            )
                            b_c3, b_c4 = st.columns(2)
                            building_plane_tolerance = b_c3.slider(
                                "Tolerancia plano techo (m)",
                                0.05,
                                0.4,
                                snap_slider_value(building_plane_tolerance_default, 0.05, 0.4, 0.01),
                                0.01,
                                help=(
                                    "Controla cuanto se puede desviar un punto respecto a un plano de cubierta.\n\n"
                                    "Ejemplo: 0.10-0.16 m para cubiertas limpias; 0.20-0.30 m para nubes ms ruidosas."
                                ),
                            )
                            building_snap_roof_planes = st.checkbox(
                                "Reducir cubierta a pocos planos dominantes",
                                value=building_snap_roof_planes,
                                help=(
                                    "Agrupa la cubierta en menos planos principales para evitar dientes y prismas raros.\n\n"
                                    "Ejemplo: convierte una cubierta ruidosa en una geometra ms limpia con cumbreras ms claras."
                                ),
                            )
                            building_min_roof_plane_area_m2 = b_c4.slider(
                                "rea mnima por plano de cubierta (m)",
                                2.0,
                                80.0,
                                snap_slider_value(building_min_roof_plane_area_m2, 2.0, 80.0, 1.0),
                                1.0,
                                key="laz_min_roof_plane_area_m2",
                                help=(
                                    "Descarta planos pequeos espurios. Con nubes IGN (1-4 pts/m) normalmente 10 m funciona bien."
                                ),
                            )
                        with laz_tab_definition:
                            b_c5, b_c6 = st.columns(2)
                            building_cluster_radius = b_c5.slider(
                                "Union mxima entre puntos de edificio (m)",
                                0.5,
                                10.0,
                                snap_slider_value(building_cluster_radius, 0.5, 10.0, 0.25),
                                0.25,
                                help=(
                                    "Define cuando dos puntos de cubierta se consideran del mismo edificio.\n\n"
                                    "Ejemplo: 2.0 m une faldones cercanos de una misma nave. "
                                    "Si une edificios vecinos por error, bajalo a 1.25-1.5 m."
                                ),
                            )
                            building_min_points = int(
                                b_c6.slider(
                                    "Puntos minimos por edificio",
                                    4,
                                    60,
                                    int(np.clip(int(round(building_min_points)), 4, 60)),
                                    1,
                                    help=(
                                        "Evita reconstruir ruido como si fuera un edificio.\n\n"
                                        "Ejemplo: con 1 punto/m², valores de 8-15 suelen funcionar bien."
                                    ),
                                )
                            )
                            b_c7, b_c8 = st.columns(2)
                            building_density_hint = b_c7.slider(
                                "Densidad LiDAR esperada en cubierta (pts/m²)",
                                0.2,
                                4.0,
                                snap_slider_value(building_density_hint, 0.2, 4.0, 0.1),
                                0.1,
                                help=(
                                    "Ajusta el algoritmo a la separacion media entre puntos de tejado.\n\n"
                                    "Ejemplo: si normalmente tienes 1 punto por metro cuadrado, dejalo en 1.0."
                                ),
                            )
                            building_split_touching_clusters = b_c8.checkbox(
                                "Separar edificios contiguos",
                                value=building_split_touching_clusters,
                                help=(
                                    "Intenta dividir un cluster grande en varios edificios si detecta cubiertas separables.\n\n"
                                    "Ejemplo: util cuando dos naves cercanas te salen como una sola pieza."
                                ),
                            )
                            building_orthogonalize_edges = st.checkbox(
                                "Rectificar bordes ortogonales",
                                value=building_orthogonalize_edges,
                                help=(
                                    "Si la huella parece casi rectangular, fuerza bordes ms rectos y limpios.\n\n"
                                    "Ejemplo: mejora naves y edificios urbanos; puede simplificar demasiado formas irregulares."
                                ),
                            )
                        with laz_tab_cleanup:
                            building_clean_class6_noise = st.checkbox(
                                "Pasar a vegetacin alta los puntos de copa colados en clase 6",
                                value=bool(building_clean_class6_noise),
                                help=(
                                    "Detecta puntos de edificio que sobresalen por encima de una cubierta planar "
                                    "y los reclasifica a vegetacin alta antes de reconstruir.\n\n"
                                    "Ejemplo: copas pegadas a una nave dejan de deformar el tejado."
                                ),
                            )
                            building_class6_noise_aggressiveness = st.slider(
                                "Sensibilidad limpieza clase 6",
                                0.5,
                                2.0,
                                snap_slider_value(building_class6_noise_aggressiveness, 0.5, 2.0, 0.1),
                                0.1,
                                help=(
                                    "Sube este valor si todava quedan copas mezcladas con edificios. "
                                    "Bjalo si empieza a quitar puntos buenos de cubierta."
                                ),
                            )
                    else:
                        st.caption("Usa huellas externas o Catastro y organiza sus controles por pestañas.")
                        footprint_tab_source, footprint_tab_outline, footprint_tab_height = st.tabs(
                            ["Fuente", "Contorno", "Altura y cubierta"]
                        )
                        with footprint_tab_source:
                            st.caption("Extruye huellas georreferenciadas y las apoya sobre el terreno.")
                            footprints_origin_label = st.radio(
                                "Origen de huellas",
                                ("Catastro automatico (INSPIRE WFS)", "Archivo de huellas (GeoJSON)"),
                                index=0,
                                key="buildings_footprints_origin_mode",
                            )
                            numeric_fields: list[str] = []
                            footprint_summary: dict = {}
                            footprints_origin_mode = "catastro" if "Catastro" in footprints_origin_label else "geojson"
                            if footprints_origin_mode == "catastro":
                                infer_info = infer_point_cloud_epsg(points)
                                st.info(infer_info.get("message", ""))
    
                                catastro_layer_label = st.selectbox(
                                    "Capa catastral",
                                    ("Partes de edificio (más detalle)", "Edificio agregado"),
                                    index=0,
                                    key="catastro_layer_label",
                                )
                                catastro_type_name = (
                                    "bu:BuildingPart" if "Partes" in catastro_layer_label else "bu:Building"
                                )
                                catastro_margin = st.slider(
                                    "Margen bbox Catastro (m)",
                                    0.0,
                                    500.0,
                                    40.0,
                                    5.0,
                                    key="catastro_bbox_margin_m",
                                    help="Amplia la busqueda alrededor del borde de la nube.",
                                )
                                catastro_max_features = int(
                                    st.slider(
                                        "Max huellas a descargar",
                                        500,
                                        20000,
                                        int(st.session_state.get("catastro_max_features", 5000)),
                                        500,
                                        key="catastro_max_features",
                                    )
                                )
    
                                if infer_info.get("is_georeferenced", False):
                                    x_min = float(np.min(points[:, 0]) - catastro_margin)
                                    x_max = float(np.max(points[:, 0]) + catastro_margin)
                                    y_min = float(np.min(points[:, 1]) - catastro_margin)
                                    y_max = float(np.max(points[:, 1]) + catastro_margin)
                                    source_epsg = int(infer_info.get("source_epsg"))
                                    source_candidates = tuple(
                                        int(e) for e in (infer_info.get("source_candidates") or [source_epsg])
                                    )
                                    query_candidates = tuple(
                                        int(e) for e in (infer_info.get("query_candidates") or [source_epsg])
                                    )
                                    request_signature = (
                                        round(x_min, 3),
                                        round(x_max, 3),
                                        round(y_min, 3),
                                        round(y_max, 3),
                                        int(source_epsg),
                                        tuple(source_candidates),
                                        tuple(query_candidates),
                                        str(catastro_type_name),
                                        int(catastro_max_features),
                                    )
                                    auto_download_catastro = st.checkbox(
                                        "Descarga automtica al detectar georreferenciacin",
                                        value=True,
                                        key="catastro_auto_download",
                                        help=(
                                            "Si esta activo, se actualizan huellas al cambiar capa, "
                                            "bbox o limite de descarga."
                                        ),
                                    )
                                    manual_download_catastro = st.button(
                                        "Descargar huellas Catastro", key="download_catastro_btn"
                                    )
                                    auto_download_needed = (
                                        bool(auto_download_catastro)
                                        and st.session_state.get("catastro_last_signature") != request_signature
                                    )
                                    if manual_download_catastro or auto_download_needed:
                                        try:
                                            dl_footprints = None
                                            dl_summary = None
                                            last_errors: list[str] = []
                                            with st.spinner("Descargando huellas de Catastro..."):
                                                for source_epsg_try in source_candidates:
                                                    query_order = tuple(
                                                        [int(source_epsg_try)]
                                                        + [int(q) for q in query_candidates if int(q) != int(source_epsg_try)]
                                                    )
                                                    try:
                                                        dl_footprints, dl_summary = fetch_catastro_cached(
                                                            x_min=x_min,
                                                            x_max=x_max,
                                                            y_min=y_min,
                                                            y_max=y_max,
                                                            source_epsg=int(source_epsg_try),
                                                            query_candidates=query_order,
                                                            type_name=catastro_type_name,
                                                            max_features=catastro_max_features,
                                                        )
                                                        break
                                                    except Exception as try_exc:
                                                        last_errors.append(
                                                            f"EPSG origen {int(source_epsg_try)}: {try_exc}"
                                                        )
                                            if not dl_footprints:
                                                raise ValueError(
                                                    "No se obtuvieron huellas de Catastro. "
                                                    + " | ".join(last_errors[:3])
                                                )
                                            st.session_state.catastro_footprints = dl_footprints
                                            st.session_state.catastro_summary = dl_summary
                                            st.session_state.catastro_last_error = ""
                                        except Exception as exc:
                                            st.session_state.catastro_footprints = None
                                            st.session_state.catastro_summary = None
                                            st.session_state.catastro_last_error = str(exc)
                                        finally:
                                            st.session_state.catastro_last_signature = request_signature
    
                                    if st.button("Limpiar huellas descargadas", key="clear_catastro_btn"):
                                        st.session_state.catastro_footprints = None
                                        st.session_state.catastro_summary = None
                                        st.session_state.catastro_last_error = ""
                                        st.session_state.catastro_last_signature = request_signature
    
                                    if st.session_state.get("catastro_last_error"):
                                        st.error(
                                            f"No se pudo descargar Catastro: {st.session_state.get('catastro_last_error')}"
                                        )
    
                                    building_footprints = st.session_state.get("catastro_footprints")
                                    footprint_summary = st.session_state.get("catastro_summary") or {}
                                    if building_footprints:
                                        accepted = int(footprint_summary.get("accepted", 0))
                                        polygons = int(footprint_summary.get("polygons", 0))
                                        q_epsg = footprint_summary.get("query_epsg")
                                        st.success(
                                            f"Huellas Catastro: {accepted:,} / {polygons:,} (EPSG consulta: {q_epsg})"
                                        )
                                        if footprint_summary.get("tiled"):
                                            st.caption(
                                                "Descarga por teselas activada: "
                                                f"{int(footprint_summary.get('tiles_with_data', 0))}/"
                                                f"{int(footprint_summary.get('tiles_total', 0))} con datos, "
                                                f"lado~{float(footprint_summary.get('tile_side_m', 0.0)):.0f} m."
                                            )
                                        numeric_fields = list((footprint_summary.get("numeric_fields") or {}).keys())
    
                                        cloud_epsg = footprint_summary.get("source_epsg", infer_info.get("source_epsg"))
                                        if cloud_epsg is not None and q_epsg is not None and int(cloud_epsg) != int(q_epsg):
                                            building_footprint_source_epsg = int(q_epsg)
                                            building_footprint_target_epsg = int(cloud_epsg)
                                        else:
                                            building_footprint_source_epsg = None
                                            building_footprint_target_epsg = None
                                    else:
                                        building_footprints = None
                                else:
                                    building_footprints = None
                                    st.warning(
                                        "No se detecta georreferenciacin compatible; usa GeoJSON manual o revisa CRS."
                                    )
                            else:
                                footprints_file = st.file_uploader(
                                    "Archivo de huellas (GeoJSON)",
                                    type=["geojson", "json"],
                                    key="buildings_footprints_upload",
                                    help="Admite Polygon y MultiPolygon.",
                                )
                                if footprints_file is not None:
                                    try:
                                        building_footprints, footprint_summary = parse_footprints_cached(
                                            footprints_file.getvalue()
                                        )
                                        accepted = int(footprint_summary.get("accepted", 0))
                                        polygons = int(footprint_summary.get("polygons", 0))
                                        st.success(f"Huellas validas: {accepted:,} / {polygons:,}")
                                        numeric_fields = list((footprint_summary.get("numeric_fields") or {}).keys())
                                    except Exception as exc:
                                        building_footprints = None
                                        st.error(f"No se pudieron leer las huellas: {exc}")
    
                        with footprint_tab_outline:
                                footprint_border_mode = st.selectbox(
                                    "Huellas fuera del area de nube",
                                    (
                                        "Recortar en el borde (recomendado)",
                                        "Descartar huellas fuera",
                                        "Permitir fuera del area",
                                    ),
                                    index=0,
                                    key="footprint_border_mode",
                                    help="Controla como tratar edificios que cruzan el borde XY de la nube.",
                                )
                                if building_footprints and "Permitir" not in footprint_border_mode:
                                    original_count = int(len(building_footprints))
                                    if "Recortar" in footprint_border_mode:
                                        building_footprints = filter_footprints_to_bbox(
                                            building_footprints,
                                            float(np.min(points[:, 0])),
                                            float(np.max(points[:, 0])),
                                            float(np.min(points[:, 1])),
                                            float(np.max(points[:, 1])),
                                            margin=0.0,
                                            keep_mode="intersects",
                                            clip_edges=True,
                                        )
                                        clipped_edges = sum(
                                            1
                                            for it in building_footprints
                                            if isinstance(it, dict) and bool(it.get("_clip_applied", False))
                                        )
                                        removed_outside = original_count - int(len(building_footprints))
                                        if clipped_edges > 0 or removed_outside > 0:
                                            st.caption(
                                                "Recorte por borde de nube: "
                                                f"{clipped_edges:,} huellas recortadas, "
                                                f"{removed_outside:,} descartadas."
                                            )
                                    else:
                                        building_footprints = filter_footprints_to_bbox(
                                            building_footprints,
                                            float(np.min(points[:, 0])),
                                            float(np.max(points[:, 0])),
                                            float(np.min(points[:, 1])),
                                            float(np.max(points[:, 1])),
                                            margin=0.0,
                                            keep_mode="inside",
                                            clip_edges=False,
                                        )
                                        removed_outside = original_count - int(len(building_footprints))
                                        if removed_outside > 0:
                                            st.caption(
                                                f"Recorte al area de nube: {removed_outside:,} huellas descartadas por estar fuera."
                                            )
                                    if not building_footprints:
                                        st.warning(
                                            "Tras aplicar el borde de nube no quedan huellas; ajusta el modo o el area."
                                        )
        
                                building_footprint_min_area = st.slider(
                                    "rea mnima huella (m)",
                                    0.0,
                                    400.0,
                                    snap_slider_value(building_footprint_min_area, 0.0, 400.0, 1.0),
                                    1.0,
                                    help="Filtra construcciones muy pequeas para evitar ruido.",
                                )
                                building_footprint_simplify_tolerance = st.slider(
                                    "Simplificar contorno (m)",
                                    0.0,
                                    1.0,
                                    snap_slider_value(building_footprint_simplify_tolerance, 0.0, 1.0, 0.02),
                                    0.02,
                                    help="Reduce vertices casi colineales para aligerar el STL.",
                                )
                                building_footprint_max_vertices = int(
                                    st.slider(
                                        "Max vertices por huella",
                                        20,
                                        500,
                                        int(building_footprint_max_vertices),
                                        10,
                                    )
                                )
        
                        with footprint_tab_height:
                                height_mode_default_index = (
                                    2
                                    if footprints_origin_mode == "catastro"
                                    and "num_floors_above_ground" in numeric_fields
                                    else 0
                                )
                                height_mode_label = st.selectbox(
                                    "Altura para extrusion",
                                    (
                                        "Altura fija",
                                        "Campo de altura (m) + fallback",
                                        "Campo de plantas x altura/planta",
                                        "Huella Catastro + altura/cubierta desde LAZ clase 6",
                                    ),
                                    index=height_mode_default_index,
                                    help=(
                                        "Define como se levanta cada edificio a partir de su huella. "
                                        "La opcion de LAZ clase 6 usa la planta del Catastro y ajusta "
                                        "la altura y la forma de la cubierta con los puntos de edificio del LAZ."
                                    ),
                                )
                                if "Campo de altura" in height_mode_label:
                                    building_footprint_height_mode = "attribute"
                                    if numeric_fields:
                                        building_footprint_height_field = st.selectbox(
                                            "Campo de altura",
                                            options=numeric_fields,
                                        )
                                    else:
                                        st.info("No hay campos numericos para altura; se usara fallback.")
                                    building_footprint_fallback_height = st.slider(
                                        "Altura fallback (m)",
                                        2.0,
                                        120.0,
                                        snap_slider_value(building_footprint_fallback_height, 2.0, 120.0, 0.5),
                                        0.5,
                                        help="Se usa si el atributo de altura viene vacio o no es valido.",
                                    )
                                elif "plantas" in height_mode_label.lower():
                                    building_footprint_height_mode = "floors"
                                    if numeric_fields:
                                        building_footprint_floors_field = st.selectbox(
                                            "Campo de plantas",
                                            options=numeric_fields,
                                        )
                                    else:
                                        st.info("No hay campos numericos para plantas; se usara fallback.")
                                    building_footprint_floor_height = st.slider(
                                        "Altura por planta (m)",
                                        2.0,
                                        5.0,
                                        snap_slider_value(building_footprint_floor_height, 2.0, 5.0, 0.1),
                                        0.1,
                                        help="Ejemplo: 3.0 m por planta para residencial comun.",
                                    )
                                    building_footprint_fallback_height = st.slider(
                                        "Altura fallback (m)",
                                        2.0,
                                        120.0,
                                        snap_slider_value(building_footprint_fallback_height, 2.0, 120.0, 0.5),
                                        0.5,
                                        help="Se usa si no hay dato de plantas en la huella.",
                                    )
                                elif "LAZ clase 6" in height_mode_label:
                                    building_footprint_height_mode = "laz_class_6"
                                    st.caption(
                                        "Usa la huella como planta y ajusta la altura y la cubierta "
                                        "con los puntos de edificio (clase 6) del LAZ. "
                                        "Si una huella no tiene puntos suficientes, cae al fallback."
                                    )
                                    preset_col, diag_col = st.columns(2)
                                    if preset_col.button(
                                        "Aplicar ajuste automatico robusto",
                                        key="fp_laz6_apply_robust_defaults",
                                        use_container_width=True,
                                    ):
                                        building_roof_model_mode = "dominant_planes"
                                        building_footprint_shape_mode = "adaptive_mix"
                                        building_footprint_outside_area_threshold = 20.0
                                        building_roof_percentile = 86.0
                                        building_snap_roof_planes = True
                                        building_plane_tolerance = 0.22
                                        building_min_roof_plane_area_m2 = 10.0
                                        building_density_hint = 1.2
                                        building_clean_class6_noise = False
                                        building_class6_noise_aggressiveness = 0.8
                                        building_footprint_min_area = min(float(building_footprint_min_area), 4.0)
                                        building_footprint_simplify_tolerance = min(
                                            float(building_footprint_simplify_tolerance),
                                            0.05,
                                        )
                                        building_footprint_max_vertices = max(
                                            int(building_footprint_max_vertices),
                                            250,
                                        )
                                        st.success("Preset robusto aplicado para IGN 1-4 pts/m².")

                                    if diag_col.button(
                                        "Diagnosticar por que faltan plantas",
                                        key="fp_laz6_run_diagnostic",
                                        use_container_width=True,
                                    ):
                                        if building_footprints:
                                            st.session_state["footprint_laz_diag"] = diagnose_footprints_vs_laz_class6(
                                                building_footprints,
                                                points,
                                                classification,
                                                density_hint=building_density_hint,
                                                outside_area_threshold=building_footprint_outside_area_threshold,
                                            )
                                        else:
                                            st.warning("Carga huellas primero para ejecutar el diagnostico.")

                                    roof_mode_options = {
                                        "Automatico (mejor ajuste)": "auto",
                                        "Interseccion de planos (recomendado)": "dominant_planes",
                                        "Superficie libre": "freeform",
                                        "Plano unico": "single_plane",
                                        "Planos dominantes": "dominant_planes",
                                        "Cubierta plana": "flat",
                                    }
                                    current_roof_mode_label = next(
                                        (
                                            label
                                            for label, value in roof_mode_options.items()
                                            if value == building_roof_model_mode
                                        ),
                                        "Automatico (mejor ajuste)",
                                    )
                                    building_roof_mode_label = st.selectbox(
                                        "Modelo de cubierta",
                                        options=list(roof_mode_options.keys()),
                                        index=list(roof_mode_options.keys()).index(current_roof_mode_label),
                                        help=(
                                            "Automático prueba varios modelos y se queda con el más simple "
                                            "que sigue bien la nube. Puedes forzar un modo si quieres comparar."
                                        ),
                                    )
                                    building_roof_model_mode = roof_mode_options[building_roof_mode_label]
                                    footprint_shape_options = {
                                        "Catastro + ajuste automatico con LAZ": "adaptive_mix",
                                        "Catastro puro": "catastro",
                                        "Huella inferida desde LAZ": "laz",
                                    }
                                    current_shape_mode_label = next(
                                        (
                                            label
                                            for label, value in footprint_shape_options.items()
                                            if value == building_footprint_shape_mode
                                        ),
                                        "Catastro + ajuste automatico con LAZ",
                                    )
                                    footprint_shape_mode_label = st.selectbox(
                                        "Estrategia de huella",
                                        options=list(footprint_shape_options.keys()),
                                        index=list(footprint_shape_options.keys()).index(current_shape_mode_label),
                                        help=(
                                            "Catastro + ajuste automatico usa la huella del Catastro como base, "
                                            "pero la amplía si la nube de clase 6 demuestra que falta geometría. "
                                            "Huella inferida desde LAZ ignora Catastro y se guía solo por la nube."
                                        ),
                                    )
                                    building_footprint_shape_mode = footprint_shape_options[footprint_shape_mode_label]
                                    if building_footprint_shape_mode == "adaptive_mix":
                                        building_footprint_outside_area_threshold = st.slider(
                                            "rea mnima fuera de Catastro para corregir huella (m²)",
                                            5.0,
                                            120.0,
                                            snap_slider_value(
                                                building_footprint_outside_area_threshold,
                                                5.0,
                                                120.0,
                                                1.0,
                                            ),
                                            1.0,
                                            help=(
                                                "Si la proyección de los puntos de cubierta saca más de esta área "
                                                "fuera de la huella catastral, se mezcla Catastro con una huella "
                                                "inferida desde el LAZ."
                                            ),
                                        )
                                    building_footprint_fallback_height = st.slider(
                                        "Altura fallback (m)",
                                        2.0,
                                        120.0,
                                        snap_slider_value(building_footprint_fallback_height, 2.0, 120.0, 0.5),
                                        0.5,
                                        help=(
                                            "Solo se usa si la huella no tiene suficientes puntos clase 6 "
                                            "para reconstruir bien la cubierta."
                                        ),
                                    )
                                    b_c1, b_c2 = st.columns(2)
                                    building_roof_percentile = b_c1.slider(
                                        "Percentil de techo clase 6",
                                        70.0,
                                        99.0,
                                        snap_slider_value(building_roof_percentile, 70.0, 99.0, 1.0),
                                        1.0,
                                        help=(
                                            "Filtra puntos bajos de fachada o ruido antes de modelar la cubierta. "
                                            "Ejemplo: 90 deja solo la parte más alta del edificio como guía principal."
                                        ),
                                    )
                                    building_snap_roof_planes = st.checkbox(
                                        "Ajustar cubierta a planos dominantes",
                                        value=bool(building_snap_roof_planes),
                                        help=(
                                            "Agrupa puntos de la cubierta en pocos planos principales. "
                                            "Ejemplo: una cubierta a dos aguas sale más limpia y menos dentada."
                                        ),
                                    )
                                    b_c3, b_c4 = st.columns(2)
                                    building_plane_tolerance = b_c3.slider(
                                        "Tolerancia de ajuste de planos (m)",
                                        0.05,
                                        1.0,
                                        snap_slider_value(building_plane_tolerance, 0.05, 1.0, 0.01),
                                        0.01,
                                        help=(
                                            "Cuanto puede separarse un punto del plano ajustado para seguir "
                                            "considerandose parte de la misma cara. Sube si la nube es ruidosa."
                                        ),
                                    )
                                    building_min_roof_plane_area_m2 = b_c4.slider(
                                        "rea mnima por plano de cubierta (m²)",
                                        2.0,
                                        80.0,
                                        snap_slider_value(building_min_roof_plane_area_m2, 2.0, 80.0, 1.0),
                                        1.0,
                                        key="footprint_min_roof_plane_area_m",
                                        help=(
                                            "Descarta planos pequeos y espurios. "
                                            "Con IGN suele funcionar bien 10 m²."
                                        ),
                                    )
                                    b_c7, b_c8 = st.columns(2)
                                    building_density_hint = b_c7.slider(
                                        "Densidad LiDAR esperada en cubiertas (pts/m²)",
                                        0.2,
                                        8.0,
                                        snap_slider_value(building_density_hint, 0.2, 8.0, 0.1),
                                        0.1,
                                        help=(
                                            "Sirve para adaptar el muestreo interno. Ejemplo: si casi siempre "
                                            "tienes 1 punto/m², deja 1.0."
                                        ),
                                    )
                                    building_clean_class6_noise = st.checkbox(
                                        "Limpiar vegetación colada en clase 6",
                                        value=bool(building_clean_class6_noise),
                                        help=(
                                            "Antes de ajustar la cubierta, mueve a vegetación alta los puntos "
                                            "que sobresalen como copas respecto a los planos del tejado."
                                        ),
                                    )
                                    building_class6_noise_aggressiveness = st.slider(
                                        "Sensibilidad limpieza clase 6",
                                        0.5,
                                        2.0,
                                        snap_slider_value(building_class6_noise_aggressiveness, 0.5, 2.0, 0.1),
                                        0.1,
                                        help="1.0 suele ir bien para nubes de alrededor de 1 punto/m².",
                                    )
                                    diag_data = st.session_state.get("footprint_laz_diag")
                                    if isinstance(diag_data, dict) and diag_data.get("ok"):
                                        total = int(diag_data.get("total", 0))
                                        with_support = int(diag_data.get("with_support", 0))
                                        likely_sparse = int(diag_data.get("likely_sparse", 0))
                                        likely_outdated = int(diag_data.get("likely_outdated", 0))
                                        st.caption(
                                            f"Diagnostico: huellas={total:,} | soporte clase 6 suficiente={with_support:,} | "
                                            f"soporte bajo={likely_sparse:,} | posible Catastro desactualizado={likely_outdated:,}"
                                        )
                                        if likely_outdated > 0:
                                            st.warning(
                                                "Hay huellas donde la nube de cubierta se sale claramente del Catastro. "
                                                "Prueba Estrategia de huella = 'Catastro + ajuste automatico con LAZ' "
                                                "o 'Huella inferida desde LAZ'."
                                            )
                                        if likely_sparse > 0:
                                            st.info(
                                                "Hay huellas con poco soporte clase 6 dentro de planta. "
                                                "Baja Percentil de techo (82-88) o desactiva limpieza clase 6."
                                            )
                                        rows = diag_data.get("rows") or []
                                        if rows:
                                            st.markdown("**Huellas con mayor desajuste**")
                                            st.dataframe(rows, use_container_width=True, height=240)
                                else:
                                    building_footprint_height_mode = "fixed"
                                    building_footprint_fixed_height = st.slider(
                                        "Altura fija (m)",
                                        2.0,
                                        120.0,
                                        snap_slider_value(building_footprint_fixed_height, 2.0, 120.0, 0.5),
                                        0.5,
                                        help="Extruye todas las huellas a la misma altura.",
                                    )
        
                                building_footprint_min_height = st.slider(
                                    "Altura mnima permitida (m)",
                                    1.0,
                                    20.0,
                                    snap_slider_value(building_footprint_min_height, 1.0, 20.0, 0.5),
                                    0.5,
                                    help="Evita edificios demasiado bajos por ruido o atributos erroneos.",
                                )
                                building_footprint_max_height = st.slider(
                                    "Altura mxima permitida (m)",
                                    10.0,
                                    250.0,
                                    snap_slider_value(building_footprint_max_height, 10.0, 250.0, 1.0),
                                    1.0,
                                    help="Recorta alturas aberrantes antes de fusinar la malla final.",
                                )
                                if building_footprint_max_height < building_footprint_min_height:
                                    building_footprint_max_height = building_footprint_min_height
                                    st.info("Altura mxima ajustada para no ser menor que la mnima.")

                        if building_footprints:
                            st.markdown("**Edicion manual de alturas por edificio**")
                            st.caption(
                                "Selecciona un edificio y asigna una altura manual. "
                                "Esta altura tiene prioridad sobre la altura fija/campo."
                            )
                            current_overrides = st.session_state.get("building_height_overrides", {})
                            if not isinstance(current_overrides, dict):
                                current_overrides = {}
                            key_set = {
                                _manual_height_key_for_footprint(item, idx)
                                for idx, item in enumerate(building_footprints, start=1)
                                if isinstance(item, dict)
                            }
                            current_overrides = {
                                str(k): float(v)
                                for k, v in current_overrides.items()
                                if str(k) in key_set and _safe_float_any(v) is not None
                            }
                            st.session_state.building_height_overrides = current_overrides
                            st.caption(
                                f"Edificios cargados: {len(key_set):,} | "
                                f"con altura manual: {len(current_overrides):,}"
                            )

                            st.checkbox(
                                "Activar ajuste manual por edificio",
                                value=bool(st.session_state.get("building_manual_height_enabled", False)),
                                key="building_manual_height_enabled",
                            )
                            if st.session_state.get("building_manual_height_enabled", False):
                                options: list[str] = []
                                label_to_key: dict[str, str] = {}
                                key_to_item: dict[str, dict] = {}
                                manual_height_editor_items = []
                                manual_height_key_order = []
                                for idx, item in enumerate(building_footprints, start=1):
                                    if not isinstance(item, dict):
                                        continue
                                    fp_key = _manual_height_key_for_footprint(item, idx)
                                    ref_label = _footprint_ref_label(item, idx)
                                    area_val = _safe_float_any(item.get("area"))
                                    area_text = f"{float(area_val):.1f} m2" if area_val is not None else "s/d"
                                    base_h = _estimate_footprint_height(
                                        item.get("properties", {}),
                                        height_mode=building_footprint_height_mode,
                                        fixed_height=building_footprint_fixed_height,
                                        fallback_height=building_footprint_fallback_height,
                                        height_attribute=building_footprint_height_field,
                                        floors_attribute=building_footprint_floors_field,
                                        floor_height=building_footprint_floor_height,
                                        min_height=building_footprint_min_height,
                                        max_height=building_footprint_max_height,
                                    )
                                    shown_h = float(current_overrides.get(fp_key, base_h))
                                    label = f"{idx:04d} | {ref_label} | area {area_text} | h {shown_h:.2f} m"
                                    options.append(label)
                                    label_to_key[label] = fp_key
                                    key_to_item[fp_key] = item
                                    try:
                                        ring = np.asarray(item.get("xy"), dtype=np.float64)
                                    except Exception:
                                        ring = np.empty((0, 2), dtype=np.float64)
                                    cx = float(np.mean(ring[:, 0])) if ring.ndim == 2 and ring.shape[0] > 0 else 0.0
                                    cy = float(np.mean(ring[:, 1])) if ring.ndim == 2 and ring.shape[0] > 0 else 0.0
                                    manual_height_editor_items.append(
                                        {
                                            "key": fp_key,
                                            "label": label,
                                            "xy": ring[:, :2] if ring.ndim == 2 and ring.shape[1] >= 2 else ring,
                                            "cx": cx,
                                            "cy": cy,
                                            "height": shown_h,
                                            "has_override": fp_key in current_overrides,
                                        }
                                    )
                                    manual_height_key_order.append(fp_key)

                                if options:
                                    selected_key_state = str(
                                        st.session_state.get("manual_height_selected_key", manual_height_key_order[0])
                                    )
                                    if selected_key_state not in key_to_item:
                                        selected_key_state = manual_height_key_order[0]
                                        st.session_state.manual_height_selected_key = selected_key_state

                                    selected_label_default = next(
                                        (lab for lab, key in label_to_key.items() if key == selected_key_state),
                                        options[0],
                                    )
                                    existing_label_state = st.session_state.get("manual_height_selected_building")
                                    if (
                                        existing_label_state not in label_to_key
                                        or label_to_key.get(existing_label_state) != selected_key_state
                                    ):
                                        st.session_state.manual_height_selected_building = selected_label_default
                                    selected_label = st.selectbox(
                                        "Seleccionar edificio",
                                        options=options,
                                        index=options.index(selected_label_default),
                                        key="manual_height_selected_building",
                                    )
                                    selected_key = label_to_key[selected_label]
                                    st.session_state.manual_height_selected_key = selected_key
                                    selected_item = key_to_item[selected_key]
                                    default_h = _estimate_footprint_height(
                                        selected_item.get("properties", {}),
                                        height_mode=building_footprint_height_mode,
                                        fixed_height=building_footprint_fixed_height,
                                        fallback_height=building_footprint_fallback_height,
                                        height_attribute=building_footprint_height_field,
                                        floors_attribute=building_footprint_floors_field,
                                        floor_height=building_footprint_floor_height,
                                        min_height=building_footprint_min_height,
                                        max_height=building_footprint_max_height,
                                    )
                                    current_h = float(current_overrides.get(selected_key, default_h))
                                    manual_h = st.number_input(
                                        "Altura manual (m)",
                                        min_value=float(building_footprint_min_height),
                                        max_value=float(building_footprint_max_height),
                                        value=float(np.clip(current_h, building_footprint_min_height, building_footprint_max_height)),
                                        step=0.5,
                                        key="manual_height_value_input",
                                    )
                                    col_set, col_remove, col_clear = st.columns(3)
                                    if col_set.button("Guardar altura", key="manual_height_save_btn"):
                                        current_overrides[selected_key] = float(manual_h)
                                        st.session_state.building_height_overrides = current_overrides
                                        st.success("Altura manual guardada.")
                                        st.rerun()
                                    if col_remove.button("Quitar altura", key="manual_height_remove_btn"):
                                        if selected_key in current_overrides:
                                            current_overrides.pop(selected_key, None)
                                            st.session_state.building_height_overrides = current_overrides
                                            st.success("Altura manual eliminada para este edificio.")
                                            st.rerun()
                                    if col_clear.button("Limpiar todas", key="manual_height_clear_btn"):
                                        st.session_state.building_height_overrides = {}
                                        st.success("Se limpiaron todas las alturas manuales.")
                                        st.rerun()
                                else:
                                    st.info("No hay edificios disponibles para editar en este momento.")

                        if footprints_origin_mode == "geojson":
                            same_crs = st.checkbox(
                                "Las huellas ya estan en el mismo CRS que el LAZ",
                                value=True,
                                help="Desmarca solo si necesitas transformar entre EPSG distintos.",
                            )
                            if not same_crs:
                                src_epsg_text = st.text_input("EPSG huellas", value="25830")
                                dst_epsg_text = st.text_input("EPSG LAZ/destino", value="25830")
                                building_footprint_source_epsg = parse_epsg(src_epsg_text)
                                building_footprint_target_epsg = parse_epsg(dst_epsg_text)
                                if building_footprint_source_epsg is None or building_footprint_target_epsg is None:
                                    building_footprint_epsg_valid = False
                                    st.warning("Introduce EPSG vlidos (ej. 25830).")
                            else:
                                building_footprint_source_epsg = None
                                building_footprint_target_epsg = None

                        if building_footprints:
                            building_footprints = apply_manual_height_overrides_to_footprints(
                                building_footprints,
                                st.session_state.get("building_height_overrides", {}),
                            )
        mesh_resolution = mesh_resolution_default
        smooth_ground = smooth_ground_default
        smooth_vegetation = smooth_vegetation_default
        smooth_buildings = smooth_buildings_default
    with controls_output_host:
        voxel_simplify = st.slider(
            "Simplificacion por voxel (m)",
            0.0,
            5.0,
            snap_slider_value(voxel_simplify_default, 0.0, 5.0, 0.1),
            0.1,
            help=(
                "Reduce detalle agrupando vertices por celdas 3D (voxeles).\n\n"
                "Ejemplo: 0.0 conserva tejados y aristas; 0.3-0.6 simplifica mucho "
                "y puede borrar detalles pequeos."
            ),
        )
        max_open_edges_tolerance = int(
            st.slider(
                "Tolerancia de aristas abiertas (exportación)",
                0,
                50,
                int(max_open_edges_tolerance),
                1,
                help=(
                    "0 exige malla totalmente cerrada. "
                    "Un valor pequeo evita bloqueo en mallas muy grandes con microdefectos."
                ),
            )
        )
        remove_outliers = st.checkbox("Eliminar outliers", value=DEFAULT_REMOVE_OUTLIERS)
        percentile_low = st.slider("Percentil inferior Z", 0, 5, DEFAULT_PERCENTILE_LOW)
        percentile_high = st.slider("Percentil superior Z", 95, 100, DEFAULT_PERCENTILE_HIGH)

    available_classes = st.session_state.available_classes
    proc_state = st.session_state.proc_class_filter
    proc_counts = {cls: int(np.sum(st.session_state.classification == cls)) for cls in available_classes}

    with controls_output_host:
        add_base = st.checkbox("Anadir base slida", value=True)
        base_thickness = (
            st.slider(
                "Grosor base (m)",
                1.0,
                40.0,
                snap_slider_value(base_thickness_default, 1.0, 40.0, 0.5),
                0.5,
            )
            if add_base
            else None
        )

    with controls_output_host:
        st.info("Layout dashboard de escritorio activo.")
        st.checkbox(
            "Mostrar vista previa 3D",
            value=st.session_state.get("proc_show_preview", True),
            key="proc_show_preview",
        )
        mesh_view_style = st.selectbox(
            "Acabado de superficie",
            (
                "Altura terreno",
                "Arcilla mate",
                "Hormigon tecnico",
                "Metal satinado",
                "Topografico intenso",
            ),
            index=4,
            key="proc_mesh_view_style",
        )
        mesh_color_mode = st.selectbox(
            "Color de malla",
            ("Por capas", "Segun acabado"),
            index=0,
            key="proc_mesh_color_mode",
            help="Por capas colorea terreno, vegetación y edificios por separado cuando la reconstrucción avanzada lo permite.",
        )
        show_mesh_wireframe = st.checkbox(
            "Mostrar aristas (wireframe)",
            value=st.session_state.get("proc_mesh_wireframe", False),
            key="proc_mesh_wireframe",
        )
        st.checkbox(
            "Vista rápida por muestra local",
            value=st.session_state.get("proc_local_preview_enabled", True),
            key="proc_local_preview_enabled",
            help="Previsualiza una zona pequeña para iterar parámetros más rápido.",
        )

        bounds_x_min = float(np.min(points[:, 0]))
        bounds_x_max = float(np.max(points[:, 0]))
        bounds_y_min = float(np.min(points[:, 1]))
        bounds_y_max = float(np.max(points[:, 1]))
        default_center_x = 0.5 * (bounds_x_min + bounds_x_max)
        default_center_y = 0.5 * (bounds_y_min + bounds_y_max)

        local_preview_sample_size = int(
            st.slider(
                "Puntos muestra local",
                5_000,
                60_000,
                int(
                    snap_slider_value(
                        float(st.session_state.get("proc_local_preview_size", 20_000)),
                        5_000.0,
                        60_000.0,
                        1_000.0,
                    )
                ),
                1_000,
                key="proc_local_preview_size",
            )
        )
        st.checkbox(
            "Actualizar vista rápida automáticamente",
            value=st.session_state.get("proc_local_preview_auto", True),
            key="proc_local_preview_auto",
        )
        has_picked_center = bool(st.session_state.get("proc_local_preview_has_pick", False))
        center_mode_options = ["Centro automático", "Manual XY"]
        picked_center_label = "Punto elegido en nube (Paso 2)"
        if has_picked_center:
            center_mode_options.append(picked_center_label)
        current_center_mode = st.session_state.get(
            "proc_local_preview_center_mode", center_mode_options[0]
        )
        if current_center_mode not in center_mode_options:
            current_center_mode = center_mode_options[0]
            st.session_state.proc_local_preview_center_mode = current_center_mode
        local_preview_center_mode = current_center_mode
        local_preview_center_x = float(
            st.session_state.get("proc_local_preview_center_x", default_center_x)
        )
        local_preview_center_y = float(
            st.session_state.get("proc_local_preview_center_y", default_center_y)
        )

    use_dashboard_layout = True
    local_plot_height = 430
    final_preview_height = 540

    with controls_layers_host:
        render_class_toggle_buttons(
            proc_state,
            "proc_class",
            available_classes,
            "Activar todas",
            "Desactivar todas",
        )
        render_class_checkboxes(proc_state, "proc_class", available_classes, proc_counts)
        reassign_overlap_class12 = st.checkbox(
            "Repartir clase 12 (Overlap/Reservado) por proximidad a 2/3/4/5/6",
            value=bool(st.session_state.get("proc_reassign_overlap_class12", True)),
            key="proc_reassign_overlap_class12",
            help=(
                "Activo por defecto. Reclasifica cada punto de clase 12 al tipo más cercano en 3D "
                "(terreno, vegetación o edificio) para aprovechar mejor esa información."
            ),
        )
        overlap_assign_to_buildings = st.checkbox(
            "Permitir que clase 12 alimente edificios (clase 6)",
            value=bool(st.session_state.get("proc_overlap_assign_to_buildings", False)),
            key="proc_overlap_assign_to_buildings",
            help=(
                "Si se desactiva, la clase 12 solo reforzará terreno o vegetación. "
                "Recomendado cuando los edificios salen rotos o con plantas raras."
            ),
            disabled=not bool(reassign_overlap_class12),
        )
        if bool(reassign_overlap_class12) and bool(overlap_assign_to_buildings):
            st.caption(
                "Aviso: Permitir clase 12 en edificios puede crear puentes o paredes falsas en clase 6."
            )

    if (
        building_source == "footprints"
        and st.session_state.get("building_manual_height_enabled", False)
        and manual_height_editor_items
    ):
        with controls_buildings_host.container():
            st.markdown("**Plano de edificios (selección por clic)**")
            st.caption(
                "Azul: altura automática. Verde: con altura manual. Rojo: edificio seleccionado."
            )
            st.caption(
                "Puedes pinchar en el punto del edificio o sobre su contorno; se selecciona el edificio más cercano."
            )
            selected_map_key = st.session_state.get("manual_height_selected_key")
            fig_buildings_plan = build_footprint_selector_figure(
                manual_height_editor_items,
                str(selected_map_key) if selected_map_key is not None else None,
                uirevision_id="manual-height-footprint-plan",
                height_px=300,
            )
            plan_state = st.plotly_chart(
                fig_buildings_plan,
                use_container_width=True,
                key="plot_building_manual_plan",
                on_select="rerun",
                selection_mode=("points",),
            )
            picked_xy = extract_first_selected_xy(plan_state)
            if picked_xy is not None:
                try:
                    cx = np.asarray(
                        [float(it.get("cx", 0.0)) for it in manual_height_editor_items],
                        dtype=np.float64,
                    )
                    cy = np.asarray(
                        [float(it.get("cy", 0.0)) for it in manual_height_editor_items],
                        dtype=np.float64,
                    )
                    if cx.size > 0 and cy.size == cx.size:
                        dist2 = (cx - float(picked_xy[0])) ** 2 + (cy - float(picked_xy[1])) ** 2
                        picked_idx = int(np.argmin(dist2))
                        picked_key = str(
                            _manual_height_key_for_footprint(
                                manual_height_editor_items[picked_idx],
                                picked_idx,
                            )
                        )
                        if picked_key != str(st.session_state.get("manual_height_selected_key", "")):
                            st.session_state.manual_height_selected_key = picked_key
                            st.session_state.manual_height_selected_building = _footprint_ref_label(
                                manual_height_editor_items[picked_idx],
                                picked_idx,
                            )
                            st.rerun()
                except Exception:
                    pass

    if generate_requested:
            st.session_state.laz_cc_path = ""
            st.session_state.laz_cc_report = None
            status_container = controls_status_host.container()
            with status_container:
                st.markdown(
                    """
                    <style>
                    .ltm-progress-row { display:flex; align-items:center; gap:10px; margin: 0.3rem 0; }
                    .ltm-progress-title { width: 145px; font-size: 0.84rem; font-weight: 600; color: var(--ltm-text); }
                    .ltm-progress-track { flex: 1; background: var(--ltm-soft-bg); border-radius: 999px; height: 14px; overflow: hidden; }
                    .ltm-progress-fill { height: 100%; border-radius: 999px; transition: width 120ms linear; }
                    .ltm-progress-pct { width: 44px; text-align: right; font-size: 0.82rem; color: var(--ltm-text); font-weight: 600; }
                    .ltm-log-wrap {
                        max-height: 320px;
                        overflow-y: auto;
                        border: 1px solid var(--ltm-border);
                        border-radius: 8px;
                        padding: 8px 10px;
                        background: var(--ltm-card-alt);
                        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
                        font-size: 0.78rem;
                        line-height: 1.35;
                    }
                    .ltm-log-line { white-space: pre-wrap; color: var(--ltm-text); margin: 0 0 3px 0; }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
                status_current = st.empty()
                status_total = st.empty()
                status_step = st.empty()
                status_log = st.empty()
            start_time = time.time()
            log_messages: list[str] = []
            log_html_lines: list[str] = []
            last_message_holder = {"value": None}
            ui_state = {
                "total_pct": 0,
                "step_pct": 0,
                "step_label": "Inicializando",
                "last_refresh": 0.0,
            }

            def _color_total(value: int) -> str:
                if value < 25:
                    return "#ef4444"
                if value < 50:
                    return "#f59e0b"
                if value < 80:
                    return "#0ea5e9"
                return "#16a34a"

            def _color_step(label: str) -> str:
                text = (label or "").lower()
                if "terreno" in text:
                    return "#16a34a"
                if "vegetación" in text:
                    return "#65a30d"
                if "edific" in text:
                    return "#f59e0b"
                if "fusin" in text:
                    return "#0ea5e9"
                if "base" in text:
                    return "#8b5cf6"
                if "integridad" in text:
                    return "#ef4444"
                if "export" in text:
                    return "#0284c7"
                return "#2563eb"

            def _render_bar(holder, title: str, pct_value: int, color: str) -> None:
                pct_safe = int(min(max(pct_value, 0), 100))
                holder.markdown(
                    f"""
                    <div class="ltm-progress-row">
                        <div class="ltm-progress-title">{html.escape(title)}</div>
                        <div class="ltm-progress-track">
                            <div class="ltm-progress-fill" style="width:{pct_safe}%; background:{color};"></div>
                        </div>
                        <div class="ltm-progress-pct">{pct_safe}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            
            def update_status(
                message: str,
                pct: int | None = None,
                step_pct: float | None = None,
                step_label: str | None = None,
                state: str = "info",
            ) -> None:
                elapsed = time.time() - start_time
                formatted = f"{message} ({elapsed:.1f}s)"
                if state == "info":
                    status_current.info(formatted)
                elif state == "success":
                    status_current.success(formatted)
                else:
                    status_current.error(formatted)

                active_step = step_label if step_label else message
                if step_pct is not None:
                    ui_state["step_pct"] = int(min(max(step_pct, 0), 100))
                    ui_state["step_label"] = active_step
                else:
                    if message != last_message_holder["value"]:
                        ui_state["step_pct"] = 0
                        ui_state["step_label"] = active_step

                if message != last_message_holder["value"]:
                    step_id = len(log_messages) + 1
                    log_messages.append(f"[{step_id:02d}] {message} - {elapsed:.1f}s")
                    log_html_lines.append(
                        f'<div class="ltm-log-line">{html.escape(log_messages[-1])}</div>'
                    )
                    last_message_holder["value"] = message

                if pct is not None:
                    ui_state["total_pct"] = int(min(max(pct, 0), 100))

                now = time.time()
                force_refresh = state != "info"
                if step_pct is not None and (step_pct <= 0.01 or step_pct >= 99.99):
                    force_refresh = True
                if pct is not None and (pct in (0, 100)):
                    force_refresh = True
                if (now - ui_state["last_refresh"] >= 0.35) or force_refresh:
                    _render_bar(
                        status_total,
                        "Progreso total",
                        int(ui_state["total_pct"]),
                        _color_total(int(ui_state["total_pct"])),
                    )
                    _render_bar(
                        status_step,
                        f"Paso actual: {ui_state['step_label']}",
                        int(ui_state["step_pct"]),
                        _color_step(str(ui_state["step_label"])),
                    )
                    status_log.markdown(
                        f'<div class="ltm-log-wrap">{"".join(log_html_lines)}</div>',
                        unsafe_allow_html=True,
                    )
                    ui_state["last_refresh"] = now
            
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
                    
                    # VALIDACIÓN AGREGADA
                    is_valid, msg = validate_class_selection(work_class, enabled)
                    if not is_valid:
                        raise ValueError(msg)
                    
                    combined_mask = np.isin(work_class, enabled)
                    kept = int(np.count_nonzero(combined_mask))
                    
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
                    update_status("Eliminando outliers estadísticos", 35)
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

                if processing_mode == "Modo avanzado (detalle máximo)" and building_source == "footprints":
                    if not building_footprint_epsg_valid:
                        raise ValueError("EPSG de huellas invalido. Revisa los codigos EPSG.")
                    if building_footprints is None or len(building_footprints) == 0:
                        raise ValueError(
                            "Seleccionaste edificios por huellas, pero no hay huellas vlidas cargadas."
                        )
                    update_status(
                        f"Huellas externas cargadas: {len(building_footprints):,}",
                        46,
                        100.0,
                        "Edificios",
                    )
                
                if processing_mode == "Interpolación en rejilla":
                    mesh_points, tri = build_mesh(
                        work_points,
                        work_class,
                        mesh_resolution,
                        interpolation_method,
                        (smooth_ground, smooth_vegetation, smooth_buildings),
                        update_status,
                        reassign_overlap_class12=bool(reassign_overlap_class12),
                        overlap_assign_to_buildings=bool(overlap_assign_to_buildings),
                        overlap_building_density_hint=float(building_density_hint),
                    )
                    triangles = np.asarray(tri.simplices, dtype=np.int32)
                    hull_edges = tri.convex_hull
                else:
                    update_status("Activando modo avanzado de fusión de mallas", 50)
                    advanced_components = None
                    advanced_result = call_build_mesh_advanced_compat(
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
                        building_plane_tolerance=building_plane_tolerance,  # BUG CORREGIDO
                        building_split_touching_clusters=building_split_touching_clusters,
                        building_orthogonalize_edges=building_orthogonalize_edges,
                        building_snap_roof_planes=building_snap_roof_planes,
                        building_roof_model_mode=building_roof_model_mode,
                        building_min_roof_plane_area_m2=building_min_roof_plane_area_m2,
                        building_density_hint=building_density_hint,
                        building_clean_class6_noise=building_clean_class6_noise,
                        building_class6_noise_aggressiveness=building_class6_noise_aggressiveness,
                        building_source=building_source,
                        building_footprints=building_footprints,
                        building_footprint_height_mode=building_footprint_height_mode,
                        building_footprint_height_field=building_footprint_height_field,
                        building_footprint_floors_field=building_footprint_floors_field,
                        building_footprint_fixed_height=building_footprint_fixed_height,
                        building_footprint_fallback_height=building_footprint_fallback_height,
                        building_footprint_floor_height=building_footprint_floor_height,
                        building_footprint_min_height=building_footprint_min_height,
                        building_footprint_max_height=building_footprint_max_height,
                        building_footprint_min_area=building_footprint_min_area,
                        building_footprint_simplify_tolerance=building_footprint_simplify_tolerance,
                        building_footprint_max_vertices=building_footprint_max_vertices,
                        building_footprint_source_epsg=building_footprint_source_epsg,
                        building_footprint_target_epsg=building_footprint_target_epsg,
                        building_footprint_shape_mode=building_footprint_shape_mode,
                        building_footprint_outside_area_threshold=building_footprint_outside_area_threshold,
                        vegetation_mode=vegetation_mode,
                        vegetation_style_min_area=vegetation_style_min_area,
                        vegetation_style_detail=vegetation_style_detail,
                        vegetation_style_include_low=vegetation_style_include_low,
                        vegetation_style_max_clusters=vegetation_style_max_clusters,
                        vegetation_style_roughness=vegetation_style_roughness,
                        vegetation_style_density_response=vegetation_style_density_response,
                        vegetation_style_texture_pitch=vegetation_style_texture_pitch,
                        vegetation_style_relief_cap=vegetation_style_relief_cap,
                        vegetation_style_min_density=vegetation_style_min_density,
                        vegetation_mass_cell_size=vegetation_mass_cell_size,
                        vegetation_mass_include_low=vegetation_mass_include_low,
                        vegetation_mass_min_height=vegetation_mass_min_height,
                        vegetation_mass_min_density=vegetation_mass_min_density,
                        vegetation_mass_min_ratio=vegetation_mass_min_ratio,
                        vegetation_mass_min_patch_area=vegetation_mass_min_patch_area,
                        vegetation_mass_close_radius=vegetation_mass_close_radius,
                        vegetation_mass_open_radius=vegetation_mass_open_radius,
                        vegetation_mass_texture_pitch=vegetation_mass_texture_pitch,
                        vegetation_mass_roughness=vegetation_mass_roughness,
                        vegetation_mass_density_response=vegetation_mass_density_response,
                        vegetation_mass_relief_cap=vegetation_mass_relief_cap,
                        vegetation_mass_relief_floor=vegetation_mass_relief_floor,
                        vegetation_mass_base_embed=vegetation_mass_base_embed,
                        vegetation_mass_edge_softness=vegetation_mass_edge_softness,
                        vegetation_mass_organic_smooth=vegetation_mass_organic_smooth,
                        vegetation_mass_micro_detail=vegetation_mass_micro_detail,
                        vegetation_mass_max_cells=vegetation_mass_max_cells,
                        reassign_overlap_class12=bool(reassign_overlap_class12),
                        overlap_assign_to_buildings=bool(overlap_assign_to_buildings),
                        overlap_building_density_hint=float(building_density_hint),
                        return_components=bool(mesh_color_mode == "Por capas"),
                    )
                    if isinstance(advanced_result, tuple) and len(advanced_result) == 3:
                        mesh_points, triangles, advanced_components = advanced_result
                    else:
                        mesh_points, triangles = advanced_result
                    triangles = np.asarray(triangles, dtype=np.int32)
                    hull_edges = None
                

                points_final = mesh_points
                triangles_final = triangles

                if add_base:
                    update_status("Generando base sólida", 85, 0.0, "Base")
                    points_final, triangles_final = add_base_to_mesh(
                        points_final,
                        triangles_final,
                        float(base_thickness),
                        hull_edges=hull_edges,
                    )
                    update_status("Base sólida completada", 86, 100.0, "Base")

                if voxel_simplify > 0.0:
                    update_status("Aplicando simplificación por voxel", 88, 0.0, "Simplificación")
                    points_final, triangles_final = simplify_mesh_with_voxel_grid(
                        points_final,
                        triangles_final,
                        float(voxel_simplify),
                    )
                    update_status("Simplificación completada", 89, 100.0, "Simplificación")

                update_status("Verificando integridad de malla", 89, 0.0, "Integridad")
                open_edges = int(count_boundary_edges(triangles_final))
                final_open_edges = open_edges
                if open_edges > 0:
                    update_status(
                        f"Malla con {open_edges:,} aristas abiertas. Intentando reparaci?n autom?tica",
                        89,
                        25.0,
                        "Integridad",
                    )
                    rep_v, rep_t = repair_open_boundaries(points_final, triangles_final)
                    rep_open = int(count_boundary_edges(rep_t))
                    if rep_open <= final_open_edges:
                        points_final = np.asarray(rep_v, dtype=np.float64)
                        triangles_final = np.asarray(rep_t, dtype=np.int32)
                        final_open_edges = rep_open

                if final_open_edges == 0:
                    update_status(
                        "Reparación automática completada. Malla cerrada para impresión",
                        89,
                        100.0,
                        "Integridad",
                        state="success",
                    )
                elif final_open_edges <= int(max_open_edges_tolerance):
                    update_status(
                        f"Malla casi cerrada ({final_open_edges:,} aristas abiertas <= tolerancia {int(max_open_edges_tolerance):,}). Continuando exportaci?n",
                        89,
                        100.0,
                        "Integridad",
                        state="success",
                    )
                    st.warning(
                        "La malla mantiene un pequeño número de aristas abiertas dentro de la tolerancia configurada. "
                        "Se continuará con la exportación STL."
                    )
                else:
                    update_status(
                        f"Malla con {final_open_edges:,} aristas abiertas tras reparaci?n",
                        89,
                        100.0,
                        "Integridad",
                        state="error",
                    )
                    st.warning(
                        "La malla no quedó completamente cerrada. Prueba a bajar simplificación por voxel o subir grosor de base."
                    )
                    raise ValueError(
                        f"No se pudo garantizar una malla cerrada para impresi?n (aristas abiertas: {final_open_edges:,}). "
                        "Ajusta parámetros y vuelve a generar."
                    )

                preview_requested = st.session_state.get("proc_show_preview", True)
                payload_bytes = points_final.nbytes + triangles_final.nbytes
                face_count_final = int(np.asarray(triangles_final, dtype=np.int32).shape[0])
                preview_panel.subheader("Malla final")
                if preview_requested and payload_bytes <= MAX_PREVIEW_BYTES:
                    update_status("Renderizando malla", 90, 0.0, "Vista previa")
                    preview_plot_v, preview_plot_f = downsample_mesh_for_preview(
                        np.asarray(points_final, dtype=np.float64),
                        np.asarray(triangles_final, dtype=np.int32),
                        max_faces=int(MAX_PREVIEW_FACES),
                    )
                    final_render_mode_note = "Render activo: uniforme segun acabado."
                    fig_mesh = None
                    if mesh_color_mode == "Por capas" and advanced_components is not None and voxel_simplify <= 0.0:
                        fig_mesh = build_layered_mesh_figure(
                            advanced_components,
                            style_name=mesh_view_style,
                            show_wireframe=bool(show_mesh_wireframe),
                            uirevision_id="final-mesh-preview",
                            height_px=final_preview_height,
                        )
                        if fig_mesh is not None:
                            final_render_mode_note = "Render activo: por capas (terreno, vegetación, edificios)."
                    if fig_mesh is None:
                        fig_mesh = build_mesh_figure(
                            preview_plot_v,
                            preview_plot_f,
                            style_name=mesh_view_style,
                            show_wireframe=bool(show_mesh_wireframe),
                            uirevision_id="final-mesh-preview",
                            height_px=final_preview_height,
                        )
                        if mesh_color_mode == "Por capas":
                            if advanced_components is None:
                                final_render_mode_note = "Render activo: uniforme. Motivo: esta vista no dispone de capas separadas."
                            elif voxel_simplify > 0.0:
                                final_render_mode_note = "Render activo: uniforme. Motivo: la simplificacin por voxel mezcla las capas."
                            else:
                                final_render_mode_note = "Render activo: uniforme. Motivo: no se pudo construir la vista por capas."
                    preview_panel.plotly_chart(
                        fig_mesh,
                        use_container_width=True,
                        key="plot_final_mesh_preview",
                    )
                    preview_panel.caption(final_render_mode_note)
                    if mesh_color_mode == "Por capas" and advanced_components is not None and add_base:
                        preview_panel.caption(
                            "La vista por capas muestra terreno, vegetación y edificios por separado; la base sólida no se colorea aparte."
                        )
                    if mesh_color_mode == "Por capas" and advanced_components is not None and voxel_simplify > 0.0:
                        preview_panel.caption(
                            "Vista por capas desactivada en esta preview final porque la simplificacin por voxel mezcla las capas."
                        )
                    if preview_plot_f.shape[0] < face_count_final:
                        preview_panel.caption(
                            f"Vista previa en modo seguro: {preview_plot_f.shape[0]:,} caras mostradas de "
                            f"{face_count_final:,}. La descarga STL conserva todo el detalle."
                        )
                    update_status("Vista previa renderizada", 91, 100.0, "Vista previa")
                else:
                    update_status("Omitiendo vista previa 3D", 90, 100.0, "Vista previa")
                    size_mb = payload_bytes / (1024 * 1024)
                    if not preview_requested:
                        preview_panel.info("La vista previa 3D esta desactivada.")
                    else:
                        preview_panel.warning(
                            f"La vista previa se ha omitido autom?ticamente (aprox. {size_mb:.1f} MB de datos, por encima del l?mite seguro). "
                            "Descarga el STL para revisarlo o reduce resolución/clases incluidas si deseas visualizarlo aquí."
                        )

                update_status("Exportando STL", 95, 0.0, "Exportación")
                output_name = f"terrain_mesh_{points_final.shape[0]}_vertices.stl"
                output_path = get_temp_output_path(output_name, ".stl")
                write_stl(output_path, points_final, triangles_final)
                update_status("STL exportado", 97, 100.0, "Exportación")

                update_status("Proceso completado", 100, 100.0, "Finalizado", state="success")
                st.balloons()

                controls_status_host.subheader("Resumen de resultado")
                col_a, col_b, col_c = controls_status_host.columns(3)
                col_a.metric("Vértices", f"{points_final.shape[0]:,}")
                col_b.metric("Triángulos", f"{triangles_final.shape[0]:,}")
                bbox = points_final.max(axis=0) - points_final.min(axis=0)
                col_c.metric("Dimensiones (m)", f"{bbox[0]:.1f} x {bbox[1]:.1f} x {bbox[2]:.1f}")

                with open(output_path, "rb") as handle:
                    controls_status_host.download_button(
                        label="Descargar STL",
                        data=handle,
                        file_name=output_name,
                        mime="application/octet-stream",
                        type="primary",
                    )

                cc_exe_final = _find_cloudcompare_exe()
                if cc_exe_final:
                    if controls_status_host.button(
                        "Generar STL vlidado con CloudCompare",
                        key="laz_cc_generate_btn",
                    ):
                        try:
                            with st.spinner("Postprocesando STL final con CloudCompare..."):
                                cc_vertices, cc_faces, cc_report = _cloudcompare_roundtrip_mesh(
                                    np.asarray(points_final, dtype=np.float64),
                                    np.asarray(triangles_final, dtype=np.int32),
                                    timeout_s=420,
                                    merge_meshes=True,
                                )
                            cc_output_name = f"{Path(output_name).stem}_cloudcompare.stl"
                            cc_output_path = get_temp_output_path(Path(cc_output_name).stem, ".stl")
                            write_stl(cc_output_path, cc_vertices, cc_faces)
                            st.session_state.laz_cc_path = cc_output_path
                            st.session_state.laz_cc_report = cc_report
                            controls_status_host.success(
                                "CloudCompare completado. "
                                f"Aristas abiertas: {int(cc_report.get('open_before', 0))} -> {int(cc_report.get('open_after', 0))}."
                            )
                        except Exception as cc_exc:
                            controls_status_host.error(f"No se pudo validar el STL con CloudCompare: {cc_exc}")

                    laz_cc_path = str(st.session_state.get("laz_cc_path", "")).strip()
                    laz_cc_report = st.session_state.get("laz_cc_report")
                    if laz_cc_path and os.path.isfile(laz_cc_path):
                        with open(laz_cc_path, "rb") as cc_handle:
                            controls_status_host.download_button(
                                label="Descargar STL vlidado (CloudCompare)",
                                data=cc_handle,
                                file_name=f"{Path(output_name).stem}_cloudcompare.stl",
                                mime="application/octet-stream",
                                key="download_laz_cc_stl_btn",
                            )
                        if isinstance(laz_cc_report, dict):
                            controls_status_host.caption(
                                "CloudCompare: "
                                f"aristas abiertas {int(laz_cc_report.get('open_before', 0))} -> {int(laz_cc_report.get('open_after', 0))} | "
                                f"tiempo {float(laz_cc_report.get('elapsed_s', 0.0)):.2f}s"
                            )
            except Exception as exc:
                update_status(f"Error: {exc}", state="error")
                controls_status_host.exception(exc)

    if st.session_state.get("proc_local_preview_enabled", True) and not generate_requested:
        local_preview_view_mode = str(st.session_state.get("proc_local_preview_view_mode", "Malla"))
        if local_preview_view_mode not in ("Malla", "Nube"):
            local_preview_view_mode = "Malla"
            st.session_state.proc_local_preview_view_mode = local_preview_view_mode

        local_preview_center_mode = str(
            st.session_state.get("proc_local_preview_center_mode", center_mode_options[0])
        )
        if local_preview_center_mode not in center_mode_options:
            local_preview_center_mode = center_mode_options[0]
            st.session_state.proc_local_preview_center_mode = local_preview_center_mode

        if has_picked_center:
            (
                btn_view_mesh_col,
                btn_view_cloud_col,
                btn_center_auto_col,
                btn_center_manual_col,
                btn_center_pick_col,
                toolbar_info_col,
                toolbar_refresh_col,
            ) = preview_panel.columns([0.48, 0.48, 0.48, 0.48, 0.48, 1.9, 0.4], gap="small")
        else:
            (
                btn_view_mesh_col,
                btn_view_cloud_col,
                btn_center_auto_col,
                btn_center_manual_col,
                toolbar_info_col,
                toolbar_refresh_col,
            ) = preview_panel.columns([0.48, 0.48, 0.48, 0.48, 2.35, 0.4], gap="small")
            btn_center_pick_col = None

        if btn_view_mesh_col.button(
            "🧱",
            key="proc_preview_view_mesh_icon_btn",
            use_container_width=True,
            type="primary" if local_preview_view_mode == "Malla" else "secondary",
            help="Vista de malla reconstruida",
        ):
            st.session_state.proc_local_preview_view_mode = "Malla"
        if btn_view_cloud_col.button(
            "☁️",
            key="proc_preview_view_cloud_icon_btn",
            use_container_width=True,
            type="primary" if local_preview_view_mode == "Nube" else "secondary",
            help="Vista de nube de puntos",
        ):
            st.session_state.proc_local_preview_view_mode = "Nube"
        local_preview_view_mode = str(st.session_state.get("proc_local_preview_view_mode", "Malla"))

        if btn_center_auto_col.button(
            "🎯",
            key="proc_preview_center_auto_icon_btn",
            use_container_width=True,
            type="primary" if local_preview_center_mode == "Centro automático" else "secondary",
            help="Centro automático de la muestra",
        ):
            st.session_state.proc_local_preview_center_mode = "Centro automático"
        if btn_center_manual_col.button(
            "✋",
            key="proc_preview_center_manual_icon_btn",
            use_container_width=True,
            type="primary" if local_preview_center_mode == "Manual XY" else "secondary",
            help="Centro XY manual",
        ):
            st.session_state.proc_local_preview_center_mode = "Manual XY"
        if has_picked_center and btn_center_pick_col is not None:
            if btn_center_pick_col.button(
                "📍",
                key="proc_preview_center_pick_icon_btn",
                use_container_width=True,
                type="primary" if local_preview_center_mode == picked_center_label else "secondary",
                help="Usar punto elegido en nube (Paso 2)",
            ):
                st.session_state.proc_local_preview_center_mode = picked_center_label

        local_preview_center_mode = str(
            st.session_state.get("proc_local_preview_center_mode", center_mode_options[0])
        )
        picked_source = str(st.session_state.get("proc_local_preview_pick_source", "")).strip()
        if local_preview_center_mode == "Centro automático":
            local_preview_center_x = float(default_center_x)
            local_preview_center_y = float(default_center_y)
            st.session_state.proc_local_preview_center_x = local_preview_center_x
            st.session_state.proc_local_preview_center_y = local_preview_center_y
            toolbar_info_col.caption(f"X {local_preview_center_x:.2f} | Y {local_preview_center_y:.2f}")
        elif local_preview_center_mode == picked_center_label and has_picked_center:
            local_preview_center_x = float(
                st.session_state.get("proc_local_preview_center_x", default_center_x)
            )
            local_preview_center_y = float(
                st.session_state.get("proc_local_preview_center_y", default_center_y)
            )
            source_short = (picked_source or "Paso 2").strip()
            toolbar_info_col.caption(
                f"{source_short}: X {local_preview_center_x:.2f} | Y {local_preview_center_y:.2f}"
            )
        else:
            step_x = max((bounds_x_max - bounds_x_min) / 400.0, 0.01)
            step_y = max((bounds_y_max - bounds_y_min) / 400.0, 0.01)
            center_x_for_slider = snap_slider_value(
                float(st.session_state.get("proc_local_preview_center_x", default_center_x)),
                bounds_x_min,
                bounds_x_max,
                float(step_x),
            )
            center_y_for_slider = snap_slider_value(
                float(st.session_state.get("proc_local_preview_center_y", default_center_y)),
                bounds_y_min,
                bounds_y_max,
                float(step_y),
            )
            with toolbar_info_col.popover(
                "🎚️",
                help="Ajustar centro XY manual",
                use_container_width=False,
            ):
                local_preview_center_x = st.slider(
                    "Centro X",
                    bounds_x_min,
                    bounds_x_max,
                    float(center_x_for_slider),
                    float(step_x),
                    key="proc_local_preview_center_x_slider",
                )
                local_preview_center_y = st.slider(
                    "Centro Y",
                    bounds_y_min,
                    bounds_y_max,
                    float(center_y_for_slider),
                    float(step_y),
                    key="proc_local_preview_center_y_slider",
                )
            st.session_state.proc_local_preview_center_x = float(local_preview_center_x)
            st.session_state.proc_local_preview_center_y = float(local_preview_center_y)
            toolbar_info_col.caption(f"X {float(local_preview_center_x):.2f} | Y {float(local_preview_center_y):.2f}")
        local_view_anchor = (
            round(float(local_preview_center_x), 3),
            round(float(local_preview_center_y), 3),
            int(local_preview_sample_size),
        )
        previous_anchor = st.session_state.get("proc_local_preview_axis_anchor")
        if previous_anchor != local_view_anchor:
            st.session_state.proc_local_preview_axis_anchor = local_view_anchor
            st.session_state.proc_local_preview_axis_ranges = None
        refresh_preview_now = toolbar_refresh_col.button(
            "↻",
            key="proc_refresh_local_preview",
            help="Actualizar vista rápida ahora",
            use_container_width=True,
        )
        auto_preview = bool(st.session_state.get("proc_local_preview_auto", True))
        if not auto_preview and not refresh_preview_now:
            preview_panel.info("Activa la actualización automática o pulsa el botón para recalcular la muestra.")
        else:
            try:
                preview_t0 = time.time()
                enabled_preview_classes = [
                    cls for cls, flag in st.session_state.proc_class_filter.items() if flag
                ]
                if not enabled_preview_classes:
                    preview_panel.warning("No hay clases activas para generar la vista rápida.")
                else:
                    class_mask = np.isin(classification, enabled_preview_classes)
                    if percentile_low > 0 or percentile_high < 100:
                        z_low_preview, z_high_preview = np.percentile(
                            points[:, 2], [percentile_low, percentile_high]
                        )
                        class_mask &= (points[:, 2] >= z_low_preview) & (points[:, 2] <= z_high_preview)

                    candidate_idx = np.flatnonzero(class_mask)
                    candidate_count = int(candidate_idx.size)
                    if candidate_count < 3:
                        preview_panel.warning(
                            "La muestra local se queda sin puntos tras aplicar clases/percentiles."
                        )
                    else:
                        candidate_points = points[candidate_idx]
                        dist_sq = (
                            (candidate_points[:, 0] - float(local_preview_center_x)) ** 2
                            + (candidate_points[:, 1] - float(local_preview_center_y)) ** 2
                        )
                        k = min(int(local_preview_sample_size), candidate_count)
                        if k < candidate_count:
                            nearest_local = np.argpartition(dist_sq, k - 1)[:k]
                            nearest_order = np.argsort(dist_sq[nearest_local], kind="stable")
                            selected_idx = candidate_idx[nearest_local[nearest_order]]
                        else:
                            selected_idx = candidate_idx

                        sample_points = points[selected_idx].copy()
                        sample_class = classification[selected_idx].copy()
                        sample_return_number = (
                            st.session_state.return_number[selected_idx].copy()
                            if st.session_state.return_number is not None
                            else None
                        )
                        sample_num_returns = (
                            st.session_state.num_returns[selected_idx].copy()
                            if st.session_state.num_returns is not None
                            else None
                        )

                        if remove_outliers:
                            sample_points, sample_class, sample_keep = remove_statistical_outliers(
                                sample_points,
                                sample_class,
                                return_mask=True,
                            )
                            if sample_return_number is not None:
                                sample_return_number = sample_return_number[sample_keep]
                            if sample_num_returns is not None:
                                sample_num_returns = sample_num_returns[sample_keep]

                        if sample_points.shape[0] < 3:
                            preview_panel.warning(
                                "No hay puntos suficientes en la muestra despues de la limpieza."
                            )
                        else:
                            preview_axis_ranges = st.session_state.get("proc_local_preview_axis_ranges")
                            if preview_axis_ranges is None:
                                sx_min = float(np.min(sample_points[:, 0]))
                                sx_max = float(np.max(sample_points[:, 0]))
                                sy_min = float(np.min(sample_points[:, 1]))
                                sy_max = float(np.max(sample_points[:, 1]))
                                sz_min = float(np.min(sample_points[:, 2]))
                                sz_max = float(np.max(sample_points[:, 2]))

                                dx = max(sx_max - sx_min, 1.0)
                                dy = max(sy_max - sy_min, 1.0)
                                dz = max(sz_max - sz_min, 0.5)
                                pad_x = max(dx * 0.08, 0.6)
                                pad_y = max(dy * 0.08, 0.6)
                                pad_z = max(dz * 0.10, 0.4)

                                preview_axis_ranges = {
                                    "x": (sx_min - pad_x, sx_max + pad_x),
                                    "y": (sy_min - pad_y, sy_max + pad_y),
                                    "z": (sz_min - pad_z, sz_max + pad_z),
                                }
                                st.session_state.proc_local_preview_axis_ranges = preview_axis_ranges

                            preview_open_edges = None
                            if local_preview_view_mode == "Nube":
                                cloud_points = sample_points
                                cloud_classes = sample_class
                                cloud_limit = 25_000
                                if cloud_points.shape[0] > cloud_limit:
                                    cloud_idx = np.linspace(
                                        0,
                                        cloud_points.shape[0] - 1,
                                        cloud_limit,
                                        dtype=int,
                                    )
                                    cloud_points = cloud_points[cloud_idx]
                                    cloud_classes = cloud_classes[cloud_idx]

                                fig_cloud = go.Figure()
                                for cls in np.unique(cloud_classes):
                                    cls_int = int(cls)
                                    cls_mask = cloud_classes == cls
                                    cls_points = cloud_points[cls_mask]
                                    if cls_points.shape[0] == 0:
                                        continue
                                    fig_cloud.add_trace(
                                        go.Scatter3d(
                                            x=cls_points[:, 0],
                                            y=cls_points[:, 1],
                                            z=cls_points[:, 2],
                                            mode="markers",
                                            marker=dict(size=1.5, color=get_class_color(cls_int)),
                                            name=class_label(cls_int),
                                            legendgroup=f"local-{cls_int}",
                                        )
                                    )

                                fig_cloud.add_trace(
                                    go.Scatter3d(
                                        x=[float(local_preview_center_x)],
                                        y=[float(local_preview_center_y)],
                                        z=[float(np.median(cloud_points[:, 2]))],
                                        mode="markers",
                                        marker=dict(size=5, color="#ef4444", symbol="diamond"),
                                        name="Centro muestra",
                                        legendgroup="local-center",
                                    )
                                )
                                fig_cloud.update_layout(
                                    scene=dict(
                                        xaxis_title="X (m)",
                                        yaxis_title="Y (m)",
                                        zaxis_title="Z (m)",
                                        aspectmode="data",
                                        uirevision="local-cloud-preview",
                                        xaxis=dict(range=preview_axis_ranges["x"]),
                                        yaxis=dict(range=preview_axis_ranges["y"]),
                                        zaxis=dict(range=preview_axis_ranges["z"]),
                                    ),
                                    legend=dict(orientation="h"),
                                    height=local_plot_height,
                                    margin=dict(l=0, r=0, b=0, t=26),
                                )
                                preview_panel.plotly_chart(
                                    fig_cloud,
                                    use_container_width=True,
                                    key="plot_local_preview_switch_cloud",
                                    on_select="ignore",
                                )
                            else:
                                preview_components = None
                                if processing_mode == "Interpolación en rejilla":
                                    preview_mesh_points, preview_tri = build_mesh(
                                        sample_points,
                                        sample_class,
                                        mesh_resolution,
                                        interpolation_method,
                                        (smooth_ground, smooth_vegetation, smooth_buildings),
                                        noop_status_callback,
                                        reassign_overlap_class12=bool(reassign_overlap_class12),
                                        overlap_assign_to_buildings=bool(overlap_assign_to_buildings),
                                        overlap_building_density_hint=float(building_density_hint),
                                    )
                                    preview_triangles = np.asarray(preview_tri.simplices, dtype=np.int32)
                                    preview_hull_edges = preview_tri.convex_hull
                                else:
                                    preview_footprints = building_footprints
                                    if building_source == "footprints":
                                        local_margin = max(float(terrain_resolution) * 3.0, 5.0)
                                        sample_x_min, sample_x_max = (
                                            float(np.min(sample_points[:, 0])),
                                            float(np.max(sample_points[:, 0])),
                                        )
                                        sample_y_min, sample_y_max = (
                                            float(np.min(sample_points[:, 1])),
                                            float(np.max(sample_points[:, 1])),
                                        )
                                        preview_footprints = filter_footprints_to_bbox(
                                            building_footprints,
                                            sample_x_min,
                                            sample_x_max,
                                            sample_y_min,
                                            sample_y_max,
                                            margin=local_margin,
                                        )

                                    advanced_preview_result = call_build_mesh_advanced_compat(
                                        sample_points,
                                        sample_class,
                                        noop_status_callback,
                                        return_number=sample_return_number,
                                        num_returns=sample_num_returns,
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
                                        building_plane_tolerance=building_plane_tolerance,
                                        building_split_touching_clusters=building_split_touching_clusters,
                                        building_orthogonalize_edges=building_orthogonalize_edges,
                                        building_snap_roof_planes=building_snap_roof_planes,
                                        building_roof_model_mode=building_roof_model_mode,
                                        building_min_roof_plane_area_m2=building_min_roof_plane_area_m2,
                                        building_density_hint=building_density_hint,
                                        building_clean_class6_noise=building_clean_class6_noise,
                                        building_class6_noise_aggressiveness=building_class6_noise_aggressiveness,
                                        building_source=building_source,
                                        building_footprints=preview_footprints,
                                        building_footprint_height_mode=building_footprint_height_mode,
                                        building_footprint_height_field=building_footprint_height_field,
                                        building_footprint_floors_field=building_footprint_floors_field,
                                        building_footprint_fixed_height=building_footprint_fixed_height,
                                        building_footprint_fallback_height=building_footprint_fallback_height,
                                        building_footprint_floor_height=building_footprint_floor_height,
                                        building_footprint_min_height=building_footprint_min_height,
                                        building_footprint_max_height=building_footprint_max_height,
                                        building_footprint_min_area=building_footprint_min_area,
                                        building_footprint_simplify_tolerance=building_footprint_simplify_tolerance,
                                        building_footprint_max_vertices=building_footprint_max_vertices,
                                        building_footprint_source_epsg=building_footprint_source_epsg,
                                        building_footprint_target_epsg=building_footprint_target_epsg,
                                        building_footprint_shape_mode=building_footprint_shape_mode,
                                        building_footprint_outside_area_threshold=building_footprint_outside_area_threshold,
                                        vegetation_mode=vegetation_mode,
                                        vegetation_style_min_area=vegetation_style_min_area,
                                        vegetation_style_detail=vegetation_style_detail,
                                        vegetation_style_include_low=vegetation_style_include_low,
                                        vegetation_style_max_clusters=min(
                                            int(vegetation_style_max_clusters), 250
                                        ),
                                        vegetation_style_roughness=vegetation_style_roughness,
                                        vegetation_style_density_response=vegetation_style_density_response,
                                        vegetation_style_texture_pitch=vegetation_style_texture_pitch,
                                        vegetation_style_relief_cap=vegetation_style_relief_cap,
                                        vegetation_style_min_density=vegetation_style_min_density,
                                        vegetation_mass_cell_size=vegetation_mass_cell_size,
                                        vegetation_mass_include_low=vegetation_mass_include_low,
                                        vegetation_mass_min_height=vegetation_mass_min_height,
                                        vegetation_mass_min_density=vegetation_mass_min_density,
                                        vegetation_mass_min_ratio=vegetation_mass_min_ratio,
                                        vegetation_mass_min_patch_area=vegetation_mass_min_patch_area,
                                        vegetation_mass_close_radius=vegetation_mass_close_radius,
                                        vegetation_mass_open_radius=vegetation_mass_open_radius,
                                        vegetation_mass_texture_pitch=vegetation_mass_texture_pitch,
                                        vegetation_mass_roughness=vegetation_mass_roughness,
                                        vegetation_mass_density_response=vegetation_mass_density_response,
                                        vegetation_mass_relief_cap=vegetation_mass_relief_cap,
                                        vegetation_mass_relief_floor=vegetation_mass_relief_floor,
                                        vegetation_mass_base_embed=vegetation_mass_base_embed,
                                        vegetation_mass_edge_softness=vegetation_mass_edge_softness,
                                        vegetation_mass_organic_smooth=vegetation_mass_organic_smooth,
                                        vegetation_mass_micro_detail=vegetation_mass_micro_detail,
                                        vegetation_mass_max_cells=min(
                                            int(vegetation_mass_max_cells), 350_000
                                        ),
                                        vegetation_max_points=min(max(sample_points.shape[0], 5_000), 30_000),
                                        building_max_plane_iterations=40,
                                        reassign_overlap_class12=bool(reassign_overlap_class12),
                                        overlap_assign_to_buildings=bool(overlap_assign_to_buildings),
                                        overlap_building_density_hint=float(building_density_hint),
                                        return_components=bool(mesh_color_mode == "Por capas"),
                                    )
                                    if isinstance(advanced_preview_result, tuple) and len(advanced_preview_result) == 3:
                                        preview_mesh_points, preview_triangles, preview_components = advanced_preview_result
                                    else:
                                        preview_mesh_points, preview_triangles = advanced_preview_result
                                    preview_triangles = np.asarray(preview_triangles, dtype=np.int32)
                                    preview_hull_edges = None

                                preview_points_final = preview_mesh_points
                                preview_triangles_final = preview_triangles
                                if add_base and base_thickness is not None:
                                    preview_points_final, preview_triangles_final = add_base_to_mesh(
                                        preview_points_final,
                                        preview_triangles_final,
                                        float(base_thickness),
                                        hull_edges=preview_hull_edges,
                                    )
                                # En la vista rpida local evitamos simplificar por voxel para no distorsionar
                                # la lectura de edificios/capas mientras se ajustan parmetros.

                                preview_open_edges = count_boundary_edges(preview_triangles_final)
                                preview_payload_bytes = (
                                    preview_points_final.nbytes + preview_triangles_final.nbytes
                                )
                                preview_face_count = int(np.asarray(preview_triangles_final, dtype=np.int32).shape[0])
                                if preview_payload_bytes <= MAX_PREVIEW_BYTES:
                                    preview_plot_v, preview_plot_f = downsample_mesh_for_preview(
                                        np.asarray(preview_points_final, dtype=np.float64),
                                        np.asarray(preview_triangles_final, dtype=np.int32),
                                        max_faces=int(MAX_PREVIEW_FACES),
                                    )
                                    local_render_mode_note = "Render activo: uniforme segun acabado."
                                    fig_preview = None
                                    if mesh_color_mode == "Por capas" and preview_components is not None:
                                        fig_preview = build_layered_mesh_figure(
                                            preview_components,
                                            style_name=mesh_view_style,
                                            show_wireframe=bool(show_mesh_wireframe),
                                            uirevision_id="local-mesh-preview",
                                            height_px=local_plot_height,
                                            axis_ranges=preview_axis_ranges,
                                        )
                                        if fig_preview is not None:
                                            local_render_mode_note = "Render activo: por capas (terreno, vegetación, edificios)."
                                    if fig_preview is None:
                                        fig_preview = build_mesh_figure(
                                            preview_plot_v,
                                            preview_plot_f,
                                            style_name=mesh_view_style,
                                            show_wireframe=bool(show_mesh_wireframe),
                                            uirevision_id="local-mesh-preview",
                                            height_px=local_plot_height,
                                            axis_ranges=preview_axis_ranges,
                                        )
                                        if mesh_color_mode == "Por capas":
                                            if preview_components is None:
                                                local_render_mode_note = "Render activo: uniforme. Motivo: esta vista no dispone de capas separadas."
                                            else:
                                                local_render_mode_note = "Render activo: uniforme. Motivo: no se pudo construir la vista por capas."
                                    preview_panel.plotly_chart(
                                        fig_preview,
                                        use_container_width=True,
                                        key="plot_local_preview_switch_mesh",
                                    )
                                    preview_panel.caption(local_render_mode_note)
                                    if mesh_color_mode == "Por capas" and preview_components is not None and add_base:
                                        preview_panel.caption(
                                            "La vista por capas muestra las capas reconstruidas; la base slida no se colorea aparte."
                                        )
                                    if preview_plot_f.shape[0] < preview_face_count:
                                        preview_panel.caption(
                                            f"Vista rápida en modo seguro: {preview_plot_f.shape[0]:,} caras de "
                                            f"{preview_face_count:,}."
                                        )
                                else:
                                    preview_panel.warning(
                                        "La muestra gener demasiados datos para dibujar en 3D. "
                                        f"Lmite seguro: {MAX_PREVIEW_BYTES / (1024 * 1024):.0f} MB. "
                                        "Reduce puntos de muestra o simplificacin."
                                    )

                            preview_dt = time.time() - preview_t0
                            summary_tail = (
                                f"aristas abiertas={preview_open_edges:,} | " if preview_open_edges is not None else ""
                            )
                            preview_panel.caption(
                                "Muestra usada: "
                                f"{sample_points.shape[0]:,} / {candidate_count:,} puntos candidatos | "
                                f"centro XY=({local_preview_center_x:.2f}, {local_preview_center_y:.2f}) | "
                                f"{summary_tail}tiempo={preview_dt:.2f}s"
                            )
            except Exception as preview_exc:
                preview_panel.warning(f"No se pudo actualizar la vista rápida: {preview_exc}")
    elif not generate_requested:
        preview_panel.markdown("---")
        preview_panel.info('Activa "Vista rápida por muestra local" para usar la columna de previsualización.')
        if cloud_panel is not None:
            cloud_panel.info('Activa "Vista rápida por muestra local" para mostrar la nube local.')

# ============================================================================
# BOTÓN DE REINICIO
# ============================================================================

if st.session_state.file_loaded:
    st.sidebar.markdown("---")
    sidebar_new_project = st.sidebar.button("Cargar nuevo archivo")
    header_new_project = bool(st.session_state.get("proc_new_project_request", False))
    if sidebar_new_project or header_new_project:
        st.session_state.proc_new_project_request = False
        st.session_state.points = None
        st.session_state.classification = None
        st.session_state.file_loaded = False
        st.session_state.input_filename = ""
        st.session_state.catastro_footprints = None
        st.session_state.catastro_summary = None
        st.session_state.catastro_last_error = ""
        st.session_state.catastro_last_signature = None
        st.session_state.building_height_overrides = {}
        st.session_state.building_manual_height_enabled = False
        st.session_state.proc_generate_from_header = False
        st.session_state.step = 1
        st.rerun()

st.markdown("---")
st.caption("Aplicación experimental para la generación rápida de mallas a partir de nubes LiDAR.")
