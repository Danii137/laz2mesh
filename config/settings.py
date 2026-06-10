"""
Configuración centralizada de la aplicación LAZ to 3D Mesh Converter.

Este módulo contiene todas las constantes, parámetros por defecto y configuraciones
que se utilizan en toda la aplicación.
"""

# ============================================================================
# LÍMITES Y RESTRICCIONES
# ============================================================================

# Límite máximo de puntos que se pueden cargar (50 millones)
MAX_POINTS_LOAD = 50_000_000

# Límite máximo de bytes para vista previa 3D (~600 MB)
MAX_PREVIEW_BYTES = 160 * 1024 * 1024
MAX_PREVIEW_FACES = 320_000

# Tamaño de nozzle de impresora 3D en mm
FILAMENT_NOZZLE_MM = 0.1


# ============================================================================
# CLASIFICACIÓN DE PUNTOS
# ============================================================================

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

# Clases que se excluyen por defecto en el procesamiento
DEFAULT_EXCLUDED_CLASSES = {1, 7}

# Secuencia de colores armoniosa para visualización
CLASS_COLOR_SEQUENCE = [
    "#6366f1",  # Indigo
    "#10b981",  # Emerald
    "#f59e0b",  # Amber
    "#ef4444",  # Red
    "#8b5cf6",  # Violet
    "#ec4899",  # Pink
    "#06b6d4",  # Cyan
    "#84cc16",  # Lime
    "#f97316",  # Orange
    "#64748b",  # Slate
]

# Mapa dinámico de color por clase (se completa bajo demanda en UI)
CLASS_COLOR_MAP: dict[int, str] = {}


# ============================================================================
# TEMAS DE INTERFAZ
# ============================================================================

THEMES = {
    "oscuro": {
        "background": "#0b0f1a",
        "surface": "#161d2f",
        "text": "#f1f5f9",
        "accent": "#818cf8",
        "accent_alt": "#6366f1",
        "shadow": "rgba(0, 0, 0, 0.4)",
    },
    "claro": {
        "background": "#fdfdff",
        "surface": "#ffffff",
        "text": "#1e293b",
        "accent": "#4f46e5",
        "accent_alt": "#6366f1",
        "shadow": "rgba(15, 23, 42, 0.05)",
    },
}


# ============================================================================
# ESCALAS Y PRESETS
# ============================================================================

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


# ============================================================================
# PARÁMETROS POR DEFECTO
# ============================================================================

# Parámetros de suavizado
DEFAULT_SMOOTH_GROUND = 0.5
DEFAULT_SMOOTH_VEGETATION = 2.5
DEFAULT_SMOOTH_BUILDINGS = 0.3

# Parámetros de malla
DEFAULT_MESH_RESOLUTION = 1.5
DEFAULT_INTERPOLATION_METHOD = "linear"

# Parámetros de terreno avanzado
DEFAULT_TERRAIN_RESOLUTION = 5.0
DEFAULT_MAX_VERTICAL_ERROR = 1.0
DEFAULT_MAX_SLOPE_DEG = 40.0

# Parámetros de vegetación
DEFAULT_SPHERE_RADIUS = 0.5
DEFAULT_METABALL_GRID = 0.3
DEFAULT_METABALL_THRESHOLD = 0.5
DEFAULT_METABALL_MAX_DIM = 100

# Parámetros de edificios
DEFAULT_BUILDING_CLUSTER_RADIUS = 2.0
DEFAULT_BUILDING_MIN_POINTS = 12
DEFAULT_BUILDING_ROOF_PERCENTILE = 90.0
DEFAULT_BUILDING_BASE_PERCENTILE = 10.0
DEFAULT_BUILDING_PLANE_TOLERANCE = 0.18

# Parámetros de optimización
DEFAULT_VOXEL_SIMPLIFY = 0.0
DEFAULT_BASE_THICKNESS = 5.0

# Parámetros de limpieza
DEFAULT_REMOVE_OUTLIERS = True
DEFAULT_PERCENTILE_LOW = 0
DEFAULT_PERCENTILE_HIGH = 100
