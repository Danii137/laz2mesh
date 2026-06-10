"""
Utilities for automatic building footprint download from Spanish Cadastre INSPIRE WFS.
"""

from __future__ import annotations

import math
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

CAT_WFS_URL = "http://ovc.catastro.meh.es/INSPIRE/wfsBU.aspx"


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", ".")
    if not text:
        return None
    try:
        out = float(text)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _polygon_area_xy(poly_xy: np.ndarray) -> float:
    x = poly_xy[:, 0]
    y = poly_xy[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _epsg_from_srs_name(srs_name: str | None) -> int | None:
    if not srs_name:
        return None
    text = str(srs_name).strip().upper()
    if "EPSG::" in text:
        tail = text.split("EPSG::", 1)[1]
    elif "EPSG:" in text:
        tail = text.split("EPSG:", 1)[1]
    else:
        return None
    digits = "".join(ch for ch in tail if ch.isdigit())
    if not digits:
        return None
    try:
        code = int(digits)
    except Exception:
        return None
    return code if code > 0 else None


def _xml_local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def _build_bbox_query(
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    source_epsg: int,
    query_epsg: int,
) -> tuple[float, float, float, float]:
    min_x = float(min(x_min, x_max))
    max_x = float(max(x_min, x_max))
    min_y = float(min(y_min, y_max))
    max_y = float(max(y_min, y_max))
    if source_epsg == query_epsg:
        return min_x, min_y, max_x, max_y

    try:
        from pyproj import Transformer
    except Exception as exc:
        raise ImportError(
            "Para descargar Catastro con cambio de EPSG necesitas instalar 'pyproj'."
        ) from exc

    transformer = Transformer.from_crs(
        f"EPSG:{int(source_epsg)}", f"EPSG:{int(query_epsg)}", always_xy=True
    )
    xs = np.asarray([min_x, max_x, max_x, min_x], dtype=np.float64)
    ys = np.asarray([min_y, min_y, max_y, max_y], dtype=np.float64)
    tx, ty = transformer.transform(xs, ys)
    tx = np.asarray(tx, dtype=np.float64)
    ty = np.asarray(ty, dtype=np.float64)
    return float(np.min(tx)), float(np.min(ty)), float(np.max(tx)), float(np.max(ty))


def _is_out_of_limits_error(message: str) -> bool:
    text = (message or "").lower()
    return "out of limits" in text or "fuera de limites" in text or "extension out of limits" in text


def infer_point_cloud_epsg(points: np.ndarray) -> dict:
    """
    Heuristic CRS detection for Spanish georeferenced point clouds.
    Returns:
        {
          "is_georeferenced": bool,
          "source_epsg": int | None,
          "source_candidates": list[int],
          "query_candidates": list[int],
          "message": str,
        }
    """
    if points is None or points.shape[0] < 2:
        return {
            "is_georeferenced": False,
            "source_epsg": None,
            "source_candidates": [],
            "query_candidates": [],
            "message": "Nube insuficiente para inferir georreferenciacion.",
        }

    x = points[:, 0].astype(np.float64, copy=False)
    y = points[:, 1].astype(np.float64, copy=False)
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    x_span, y_span = x_max - x_min, y_max - y_min

    # WGS84 lon/lat (Spain)
    if -20.0 <= x_min <= 10.0 and -20.0 <= x_max <= 10.0 and 20.0 <= y_min <= 50.0 and 20.0 <= y_max <= 50.0:
        return {
            "is_georeferenced": True,
            "source_epsg": 4326,
            "source_candidates": [4326],
            "query_candidates": [25830, 25829, 25831, 4326],
            "message": "Detectado CRS geografico (EPSG:4326 aprox). Se probara Catastro en CRS proyectado para evitar problemas de orden de ejes.",
        }

    # UTM-like mainland Spain (ambiguous among 29/30/31)
    if 100_000.0 <= x_min <= 900_000.0 and 100_000.0 <= x_max <= 900_000.0 and 3_700_000.0 <= y_min <= 4_900_000.0 and 3_700_000.0 <= y_max <= 4_900_000.0:
        # Prefer zone 30 but test all.
        return {
            "is_georeferenced": True,
            "source_epsg": 25830,
            "source_candidates": [25830, 25829, 25831],
            "query_candidates": [25830, 25829, 25831],
            "message": (
                "Detectado CRS UTM ETRS89 (zona no totalmente univoca). "
                "Se probara 25830/25829/25831 automaticamente."
            ),
        }

    # WebMercator (Spain)
    if -2_000_000.0 <= x_min <= 1_500_000.0 and -2_000_000.0 <= x_max <= 1_500_000.0 and 3_800_000.0 <= y_min <= 6_000_000.0 and 3_800_000.0 <= y_max <= 6_000_000.0:
        return {
            "is_georeferenced": True,
            "source_epsg": 3857,
            "source_candidates": [3857],
            "query_candidates": [25830, 25829, 25831, 3857],
            "message": "Detectado CRS proyectado tipo WebMercator (EPSG:3857 aprox).",
        }

    # UTM-like Canary area (rough)
    if 100_000.0 <= x_min <= 900_000.0 and 100_000.0 <= x_max <= 900_000.0 and 2_700_000.0 <= y_min <= 3_500_000.0 and 2_700_000.0 <= y_max <= 3_500_000.0:
        return {
            "is_georeferenced": True,
            "source_epsg": 32628,
            "source_candidates": [32628],
            "query_candidates": [32628],
            "message": "Detectado CRS UTM zona 28 (Canarias, EPSG:32628 aprox).",
        }

    if max(abs(x_min), abs(x_max), abs(y_min), abs(y_max)) < 10_000.0 and max(x_span, y_span) < 5_000.0:
        return {
            "is_georeferenced": False,
            "source_epsg": None,
            "source_candidates": [],
            "query_candidates": [],
            "message": "La nube parece local/no georreferenciada (coordenadas pequenas).",
        }

    return {
        "is_georeferenced": False,
        "source_epsg": None,
        "source_candidates": [],
        "query_candidates": [],
        "message": "No se pudo inferir un CRS compatible con Catastro automaticamente.",
    }


def _catastro_get_feature_xml(
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    source_epsg: int,
    query_epsg: int,
    type_name: str,
    count: int,
    timeout_sec: int,
) -> tuple[str, dict]:
    qx_min, qy_min, qx_max, qy_max = _build_bbox_query(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        source_epsg=source_epsg,
        query_epsg=query_epsg,
    )
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeNames": str(type_name),
        "srsName": f"EPSG:{int(query_epsg)}",
        "count": str(max(int(count), 1)),
        "bbox": f"{qx_min},{qy_min},{qx_max},{qy_max},urn:ogc:def:crs:EPSG::{int(query_epsg)}",
    }
    url = CAT_WFS_URL + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=max(int(timeout_sec), 10)) as resp:
        xml_text = resp.read().decode("utf-8", errors="replace")
    return xml_text, {
        "url": url,
        "query_epsg": int(query_epsg),
        "bbox_query": (float(qx_min), float(qy_min), float(qx_max), float(qy_max)),
    }


def _parse_catastro_gml_to_footprints(xml_text: str) -> tuple[list[dict], dict]:
    root = ET.fromstring(xml_text)

    if _xml_local_name(root.tag).lower() == "exceptionreport":
        msg = root.findtext(".//{*}ExceptionText")
        raise ValueError(msg or "Catastro devolvio ExceptionReport.")

    footprints: list[dict] = []
    numeric_fields: dict[str, int] = {}
    feature_count = 0
    polygon_count = 0

    for member in root.findall(".//{*}featureMember"):
        if len(member) == 0:
            continue
        feature = member[0]
        feature_count += 1
        feature_type = _xml_local_name(feature.tag)

        local_id = feature.findtext(".//{*}localId")
        floors = _safe_float(feature.findtext(".//{*}numberOfFloorsAboveGround"))
        height_below = _safe_float(feature.findtext(".//{*}heightBelowGround"))

        if floors is not None:
            numeric_fields["num_floors_above_ground"] = numeric_fields.get("num_floors_above_ground", 0) + 1
        if height_below is not None:
            numeric_fields["height_below_ground_m"] = numeric_fields.get("height_below_ground_m", 0) + 1

        properties = {
            "source": "catastro_inspire",
            "feature_type": feature_type,
        }
        if local_id:
            properties["catastro_ref"] = str(local_id)
        if floors is not None:
            properties["num_floors_above_ground"] = float(floors)
        if height_below is not None:
            properties["height_below_ground_m"] = float(height_below)

        # Exterior rings only.
        for pos_list in feature.findall(".//{*}exterior//{*}posList"):
            text = (pos_list.text or "").strip()
            if not text:
                continue
            vals = np.fromstring(text, sep=" ", dtype=np.float64)
            if vals.size < 6:
                continue
            dim = int(pos_list.attrib.get("srsDimension", "2") or 2)
            dim = 2 if dim < 2 else dim
            usable = (vals.size // dim) * dim
            if usable < 6:
                continue
            coords = vals[:usable].reshape((-1, dim))[:, :2]
            if coords.shape[0] < 3:
                continue
            if np.linalg.norm(coords[0] - coords[-1]) <= 1e-9:
                coords = coords[:-1]
            if coords.shape[0] < 3:
                continue
            area = abs(_polygon_area_xy(coords))
            if area <= 1e-9:
                continue
            polygon_count += 1
            footprints.append(
                {
                    "xy": coords.astype(np.float64, copy=False),
                    "properties": dict(properties),
                    "area": float(area),
                }
            )

    summary = {
        "features": int(feature_count),
        "polygons": int(polygon_count),
        "accepted": int(len(footprints)),
        "discarded": int(max(polygon_count - len(footprints), 0)),
        "numeric_fields": dict(sorted(numeric_fields.items(), key=lambda kv: kv[0])),
        "response_epsg": _epsg_from_srs_name(root.attrib.get("srsName")),
    }
    return footprints, summary


def fetch_catastro_building_footprints(
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    source_epsg: int,
    query_candidates: list[int] | tuple[int, ...],
    type_name: str = "bu:BuildingPart",
    max_features: int = 4000,
    timeout_sec: int = 45,
    auto_tile_on_limits: bool = True,
    max_tile_size_m: float = 1200.0,
    max_tiles: int = 196,
) -> tuple[list[dict], dict]:
    """
    Download building footprints from Cadastre INSPIRE WFS in a bbox.
    """
    src = int(source_epsg)
    candidates = [int(e) for e in query_candidates if e is not None]
    if not candidates:
        candidates = [src]

    def _fetch_bbox_once(
        bx_min: float,
        bx_max: float,
        by_min: float,
        by_max: float,
    ) -> tuple[list[dict], dict]:
        errors: list[str] = []
        for epsg in candidates:
            try:
                xml_text, meta = _catastro_get_feature_xml(
                    x_min=bx_min,
                    x_max=bx_max,
                    y_min=by_min,
                    y_max=by_max,
                    source_epsg=src,
                    query_epsg=int(epsg),
                    type_name=str(type_name),
                    count=max(int(max_features), 1),
                    timeout_sec=int(timeout_sec),
                )
                footprints, summary = _parse_catastro_gml_to_footprints(xml_text)
                if footprints:
                    summary["query_epsg"] = int(epsg)
                    summary["source_epsg"] = int(src)
                    summary["request_meta"] = meta
                    return footprints, summary
                errors.append(f"EPSG:{epsg} sin resultados.")
            except Exception as exc:
                errors.append(f"EPSG:{epsg}: {exc}")
        raise ValueError(
            "No se pudieron descargar huellas catastrales para el bbox. "
            + " | ".join(errors[:4])
        )

    try:
        return _fetch_bbox_once(float(x_min), float(x_max), float(y_min), float(y_max))
    except Exception as first_exc:
        first_error = str(first_exc)
        if not auto_tile_on_limits or not _is_out_of_limits_error(first_error):
            raise

    # Fallback: divide bbox en teselas si Catastro limita el area maxima consultable.
    if src in (4326, 4258):
        raise ValueError(
            "Catastro rechazo el bbox por limites de area y el CRS origen es geografico. "
            "Reprojeta a un CRS metrico (UTM) o reduce area."
        )

    x_lo = float(min(x_min, x_max))
    x_hi = float(max(x_min, x_max))
    y_lo = float(min(y_min, y_max))
    y_hi = float(max(y_min, y_max))
    span_x = max(0.0, x_hi - x_lo)
    span_y = max(0.0, y_hi - y_lo)
    tile_side = max(50.0, float(max_tile_size_m))

    n_x = max(1, int(math.ceil(span_x / tile_side))) if span_x > 0 else 1
    n_y = max(1, int(math.ceil(span_y / tile_side))) if span_y > 0 else 1
    total_tiles = n_x * n_y
    if total_tiles > int(max_tiles):
        scale = math.sqrt(float(total_tiles) / float(max_tiles))
        tile_side = tile_side * scale
        n_x = max(1, int(math.ceil(span_x / tile_side))) if span_x > 0 else 1
        n_y = max(1, int(math.ceil(span_y / tile_side))) if span_y > 0 else 1
        total_tiles = n_x * n_y
        if total_tiles > int(max_tiles):
            raise ValueError(
                "Area demasiado grande para descarga automatica de Catastro. "
                f"Reduce extension o usa un recorte (teselas requeridas: {total_tiles})."
            )

    dx = (span_x / n_x) if n_x > 0 else 0.0
    dy = (span_y / n_y) if n_y > 0 else 0.0
    collected: list[dict] = []
    tiles_with_data = 0
    tiles_failed = 0
    agg_features = 0
    agg_polygons = 0
    agg_discarded = 0
    numeric_fields: dict[str, int] = {}
    response_epsg: int | None = None
    query_epsg: int | None = None
    tile_errors: list[str] = []

    for iy in range(n_y):
        by0 = y_lo + iy * dy
        by1 = y_hi if iy == (n_y - 1) else (y_lo + (iy + 1) * dy)
        for ix in range(n_x):
            bx0 = x_lo + ix * dx
            bx1 = x_hi if ix == (n_x - 1) else (x_lo + (ix + 1) * dx)
            try:
                tile_footprints, tile_summary = _fetch_bbox_once(bx0, bx1, by0, by1)
                if not tile_footprints:
                    continue
                collected.extend(tile_footprints)
                tiles_with_data += 1
                agg_features += int(tile_summary.get("features", 0))
                agg_polygons += int(tile_summary.get("polygons", 0))
                agg_discarded += int(tile_summary.get("discarded", 0))
                response_epsg = response_epsg or tile_summary.get("response_epsg")
                query_epsg = query_epsg or tile_summary.get("query_epsg")
                for key, val in (tile_summary.get("numeric_fields") or {}).items():
                    numeric_fields[str(key)] = numeric_fields.get(str(key), 0) + int(val)
            except Exception as tile_exc:
                tiles_failed += 1
                message = str(tile_exc)
                # "Sin resultados" no se considera fallo duro en teselado.
                if "sin resultados" in message.lower() or "no records founded" in message.lower():
                    continue
                tile_errors.append(f"tile({ix},{iy}): {message}")

    if not collected:
        raise ValueError(
            "No se obtuvieron huellas catastrales tras dividir el bbox en teselas. "
            + " | ".join(tile_errors[:4])
        )

    # Deduplicar huellas repetidas entre teselas contiguas.
    unique: list[dict] = []
    seen: set[tuple] = set()
    for item in collected:
        xy = np.asarray(item.get("xy"), dtype=np.float64)
        if xy.ndim != 2 or xy.shape[0] < 3:
            continue
        props = item.get("properties") if isinstance(item.get("properties"), dict) else {}
        ref = str(props.get("catastro_ref", "")).strip()
        area = float(item.get("area", 0.0))
        if not np.isfinite(area) or area <= 0:
            area = abs(_polygon_area_xy(xy))
        cx = float(np.mean(xy[:, 0]))
        cy = float(np.mean(xy[:, 1]))
        key = (ref, round(cx, 2), round(cy, 2), round(area, 2), int(xy.shape[0]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)

    summary = {
        "features": int(agg_features),
        "polygons": int(agg_polygons),
        "accepted": int(len(unique)),
        "discarded": int(agg_discarded),
        "numeric_fields": dict(sorted(numeric_fields.items(), key=lambda kv: kv[0])),
        "response_epsg": response_epsg,
        "query_epsg": query_epsg,
        "source_epsg": int(src),
        "tiled": True,
        "tiles_total": int(total_tiles),
        "tiles_with_data": int(tiles_with_data),
        "tiles_failed": int(tiles_failed),
        "tile_side_m": float(tile_side),
    }
    return unique, summary
