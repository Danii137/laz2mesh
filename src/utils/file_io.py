"""
Operaciones de entrada/salida de archivos.

Este mÃ³dulo maneja la lectura de archivos LAZ/LAS y la escritura de archivos STL,
con manejo mejorado de errores y archivos temporales.
"""

import os
import tempfile
from typing import Optional

import laspy
import numpy as np
import streamlit as st


def load_laz_file(uploaded_file) -> tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[str]
]:
    """
    Carga un archivo LAZ/LAS y extrae los datos de puntos.
    
    Args:
        uploaded_file: Archivo subido desde Streamlit
        
    Returns:
        Tupla (points, classification, return_number, num_returns, error_message)
        Si hay error, los arrays serÃ¡n None y error_message contendrÃ¡ el mensaje.
    """
    tmp_path = None
    try:
        # Crear archivo temporal
        suffix = os.path.splitext(uploaded_file.name)[1].lower() or ".laz"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        # Leer archivo LAZ
        las = laspy.read(tmp_path)
        
        # Extraer coordenadas
        points = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)
        
        # Extraer clasificaciÃ³n
        classification_attr = getattr(las, "classification", None)
        if classification_attr is None:
            classification = np.ones(points.shape[0], dtype=np.uint8)
        else:
            classification = np.asarray(classification_attr, dtype=np.uint8)
        
        # Extraer nÃºmero de retorno
        return_attr = getattr(las, "return_number", None)
        if return_attr is not None:
            return_number = np.asarray(return_attr, dtype=np.uint8)
        else:
            return_number = None
        
        # Extraer nÃºmero total de retornos
        total_return_attr = getattr(las, "num_returns", None)
        if total_return_attr is not None:
            num_returns = np.asarray(total_return_attr, dtype=np.uint8)
        else:
            num_returns = None
        
        return points, classification, return_number, num_returns, None
        
    except Exception as exc:
        return None, None, None, None, str(exc)
        
    finally:
        # Limpiar archivo temporal con mejor manejo de errores
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError as e:
                st.warning(f"No se pudo eliminar archivo temporal: {e}")


def write_stl(path: str, points: np.ndarray, triangles: np.ndarray) -> None:
    """
    Escribe una malla en formato STL binario.
    
    Args:
        path: Ruta del archivo de salida
        points: Array de vertices (N, 3)
        triangles: Array de triangulos (M, 3) con indices a points
    """
    triangles = np.asarray(triangles, dtype=np.int64)
    points = np.asarray(points, dtype=np.float32)

    with open(path, "wb") as fh:
        fh.write(b"\0" * 80)
        fh.write(np.uint32(triangles.shape[0]).tobytes())

        if triangles.size == 0:
            return

        verts = points[triangles]
        v0 = verts[:, 0, :]
        v1 = verts[:, 1, :]
        v2 = verts[:, 2, :]

        normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(normals, axis=1)
        valid = norms > 1e-12
        normals_out = np.zeros_like(normals, dtype=np.float32)
        normals_out[valid] = (normals[valid] / norms[valid, None]).astype(np.float32, copy=False)
        normals_out[~valid] = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        record_dtype = np.dtype(
            [
                ("normal", "<f4", (3,)),
                ("v0", "<f4", (3,)),
                ("v1", "<f4", (3,)),
                ("v2", "<f4", (3,)),
                ("attr", "<u2"),
            ]
        )
        records = np.empty(triangles.shape[0], dtype=record_dtype)
        records["normal"] = normals_out
        records["v0"] = v0
        records["v1"] = v1
        records["v2"] = v2
        records["attr"] = 0
        fh.write(records.tobytes())
def get_temp_output_path(prefix: str = "terrain_mesh", suffix: str = ".stl") -> str:
    """
    Genera una ruta para archivo temporal de salida.
    
    Args:
        prefix: Prefijo del nombre de archivo
        suffix: ExtensiÃ³n del archivo
        
    Returns:
        Ruta completa del archivo temporal
    """
    return os.path.join(tempfile.gettempdir(), f"{prefix}{suffix}")
