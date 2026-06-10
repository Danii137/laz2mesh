"""
Operaciones geométricas y de procesamiento de puntos.

Este módulo contiene funciones auxiliares para operaciones geométricas
como suavizado de rejillas y eliminación de outliers.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree


def smooth_grid(grid: np.ndarray, sigma: float) -> np.ndarray:
    """
    Aplica suavizado gaussiano a una rejilla 2D.
    
    Args:
        grid: Rejilla 2D a suavizar
        sigma: Desviación estándar del kernel gaussiano
        
    Returns:
        Rejilla suavizada
    """
    if sigma <= 0 or grid is None:
        return grid
    
    working = grid.copy()
    nan_mask = np.isnan(working)
    
    if np.all(nan_mask):
        return working
    
    # Rellenar NaNs con la media para el suavizado
    fill = np.nanmean(working[~nan_mask])
    working[nan_mask] = fill
    
    # Aplicar filtro gaussiano
    smoothed = gaussian_filter(working, sigma=sigma, mode="nearest")
    
    # Restaurar NaNs en las posiciones originales
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
    """
    Elimina outliers estadísticos basándose en la distancia a vecinos.
    
    Args:
        points: Array de puntos (N, 3)
        classification: Array de clasificación (N,) o None
        neighbors: Número de vecinos a considerar
        std_ratio: Ratio de desviación estándar para el umbral
        return_mask: Si True, retorna también la máscara de puntos válidos
        
    Returns:
        Si return_mask=False: (puntos_filtrados, clasificacion_filtrada)
        Si return_mask=True: (puntos_filtrados, clasificacion_filtrada, mascara)
    """
    # Si hay muy pocos puntos, no filtrar
    if points.shape[0] <= neighbors:
        mask_all = np.ones(points.shape[0], dtype=bool)
        if classification is None:
            if return_mask:
                return points, None, mask_all
            return points, None
        if return_mask:
            return points, classification, mask_all
        return points, classification
    
    # Construir árbol KD para búsqueda de vecinos
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=min(neighbors + 1, points.shape[0]))
    
    # Ignorar el primer vecino (el punto mismo)
    distances = distances[:, 1:]
    
    # Calcular distancia media a vecinos
    mean_dist = distances.mean(axis=1)
    
    # Calcular umbral estadístico
    threshold = mean_dist.mean() + std_ratio * mean_dist.std()
    
    # Crear máscara de puntos válidos
    mask = mean_dist <= threshold
    
    # Filtrar puntos
    filtered_points = points[mask]
    filtered_class = classification[mask] if classification is not None else None
    
    if return_mask:
        return filtered_points, filtered_class, mask
    
    return filtered_points, filtered_class


def compute_scale_preset(scale_ratio: int, nozzle_mm: float = 0.1) -> dict[str, float]:
    """
    Genera parámetros recomendados según la escala y el nozzle de impresión.
    
    Args:
        scale_ratio: Ratio de escala (ej: 1000 para 1:1000)
        nozzle_mm: Diámetro del nozzle en mm
        
    Returns:
        Diccionario con parámetros recomendados
    """
    nozzle_m = nozzle_mm / 1000.0
    min_feature_m = max(nozzle_m * scale_ratio, nozzle_m * 6.0)
    
    def _clip(value: float, min_value: float, max_value: float) -> float:
        return float(np.clip(value, min_value, max_value))
    
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
