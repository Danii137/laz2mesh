"""
Funciones de validación para parámetros y datos de entrada.

Este módulo contiene todas las validaciones necesarias para asegurar
que los datos y parámetros del usuario son válidos y coherentes.
"""

import numpy as np
import streamlit as st

from config.settings import MAX_POINTS_LOAD


def validate_height_parameters(
    ground_max: float, veg_low_max: float, veg_mid_max: float
) -> tuple[bool, str]:
    """
    Valida que los parámetros de altura sean coherentes.
    
    Args:
        ground_max: Altura máxima del terreno
        veg_low_max: Altura máxima de vegetación baja
        veg_mid_max: Altura máxima de vegetación media
        
    Returns:
        Tupla (es_valido, mensaje_error)
    """
    if ground_max >= veg_low_max:
        return False, "La altura máxima del terreno debe ser menor que la vegetación baja"
    
    if veg_low_max >= veg_mid_max:
        return False, "La altura de vegetación baja debe ser menor que la vegetación media"
    
    if ground_max < 0 or veg_low_max < 0 or veg_mid_max < 0:
        return False, "Las alturas no pueden ser negativas"
    
    return True, ""


def validate_point_cloud_size(points: np.ndarray) -> tuple[bool, str]:
    """
    Valida que el tamaño de la nube de puntos sea manejable.
    
    Args:
        points: Array de puntos (N, 3)
        
    Returns:
        Tupla (es_valido, mensaje_error)
    """
    if points.shape[0] == 0:
        return False, "La nube de puntos está vacía"
    
    if points.shape[0] > MAX_POINTS_LOAD:
        return (
            False,
            f"Archivo demasiado grande: {points.shape[0]:,} puntos. "
            f"Máximo permitido: {MAX_POINTS_LOAD:,} puntos"
        )
    
    if points.shape[0] < 3:
        return False, "Se necesitan al menos 3 puntos para generar una malla"
    
    return True, ""


def validate_class_selection(
    classification: np.ndarray, selected_classes: list[int]
) -> tuple[bool, str]:
    """
    Valida que la selección de clases sea válida.
    
    Args:
        classification: Array de clasificación de puntos
        selected_classes: Lista de clases seleccionadas
        
    Returns:
        Tupla (es_valido, mensaje_error)
    """
    if not selected_classes:
        return False, "Debes seleccionar al menos una clase para procesar"
    
    # Verificar que hay puntos en las clases seleccionadas
    mask = np.isin(classification, selected_classes)
    count = int(np.count_nonzero(mask))
    
    if count < 3:
        return False, "No hay suficientes puntos en las clases seleccionadas (mínimo 3)"
    
    return True, ""


def validate_percentiles(low: int, high: int) -> tuple[bool, str]:
    """
    Valida que los percentiles sean coherentes.
    
    Args:
        low: Percentil inferior
        high: Percentil superior
        
    Returns:
        Tupla (es_valido, mensaje_error)
    """
    if low < 0 or low > 100:
        return False, "El percentil inferior debe estar entre 0 y 100"
    
    if high < 0 or high > 100:
        return False, "El percentil superior debe estar entre 0 y 100"
    
    if low >= high:
        return False, "El percentil inferior debe ser menor que el superior"
    
    return True, ""


def validate_mesh_parameters(
    resolution: float, 
    voxel_simplify: float = 0.0,
    base_thickness: float = 0.0
) -> tuple[bool, str]:
    """
    Valida parámetros de generación de malla.
    
    Args:
        resolution: Resolución de la malla
        voxel_simplify: Tamaño de voxel para simplificación
        base_thickness: Grosor de la base
        
    Returns:
        Tupla (es_valido, mensaje_error)
    """
    if resolution <= 0:
        return False, "La resolución debe ser mayor que 0"
    
    if voxel_simplify < 0:
        return False, "El tamaño de voxel no puede ser negativo"
    
    if base_thickness < 0:
        return False, "El grosor de la base no puede ser negativo"
    
    return True, ""
