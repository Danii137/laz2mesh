"""
Componentes reutilizables de la interfaz de usuario.

Este módulo contiene funciones para renderizar componentes comunes de la UI.
"""

import numpy as np
import streamlit as st
from typing import Callable

from config.settings import CLASS_NAMES, CLASS_COLOR_SEQUENCE, CLASS_COLOR_MAP


def get_class_color(cls: int) -> str:
    """Obtiene el color asignado a una clase."""
    cls_int = int(cls)
    if cls_int not in CLASS_COLOR_MAP:
        color = CLASS_COLOR_SEQUENCE[len(CLASS_COLOR_MAP) % len(CLASS_COLOR_SEQUENCE)]
        CLASS_COLOR_MAP[cls_int] = color
    return CLASS_COLOR_MAP[cls_int]


def class_label(cls: int) -> str:
    """Genera una etiqueta legible para una clase."""
    cls_int = int(cls)
    name = CLASS_NAMES.get(cls_int, f"Clase {cls_int}")
    return f"{cls_int:02d} - {name}"


def render_class_toggle_buttons(
    state_dict: dict,
    state_prefix: str,
    available_classes: list[int],
    label_all: str = "Ver todas",
    label_none: str = "Ocultar todas",
) -> None:
    """
    Renderiza botones para activar/desactivar todas las clases.
    
    CÓDIGO DUPLICADO ELIMINADO: Función reutilizable.
    """
    col1, col2 = st.columns(2)
    if col1.button(label_all, key=f"{state_prefix}_all"):
        for cls in available_classes:
            state_dict[cls] = True
            st.session_state[f"{state_prefix}_{cls}"] = True
        st.session_state[f"{state_prefix}_visibility"] = state_dict
        st.rerun()
    if col2.button(label_none, key=f"{state_prefix}_none"):
        for cls in available_classes:
            state_dict[cls] = False
            st.session_state[f"{state_prefix}_{cls}"] = False
        st.session_state[f"{state_prefix}_visibility"] = state_dict
        st.rerun()


def render_class_checkboxes(
    state_dict: dict,
    state_prefix: str,
    available_classes: list[int],
    class_counts: dict[int, int],
) -> None:
    """Renderiza checkboxes para cada clase disponible."""
    for cls in available_classes:
        key = f"{state_prefix}_{cls}"
        if key not in st.session_state:
            st.session_state[key] = bool(state_dict.get(cls, True))
        new_val = st.checkbox(
            f"{class_label(cls)} ({class_counts.get(cls, 0):,})",
            key=key,
        )
        state_dict[cls] = bool(new_val)
    st.session_state[f"{state_prefix}_visibility"] = state_dict


def render_progress_tracker(
    status_callback: Callable[[str, int | None], None]
) -> tuple:
    """
    Crea componentes para seguimiento de progreso.
    
    Returns:
        Tupla (progress_bar, status_container, status_current, status_log)
    """
    progress_bar = st.progress(0)
    status_container = st.container()
    with status_container:
        status_current = st.empty()
        status_log = st.empty()
    
    return progress_bar, status_container, status_current, status_log


def ensure_class_states(classes: list[int], default_excluded: set[int]) -> None:
    """
    Asegura que los estados de clase estén inicializados.
    
    Args:
        classes: Lista de clases disponibles
        default_excluded: Conjunto de clases excluidas por defecto
    """
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
