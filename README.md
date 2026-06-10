# LAZ to 3D Mesh Converter

Aplicación profesional para convertir nubes de puntos LAZ/LAS a mallas 3D imprimibles.

## ✨ Características

- 🎯 **Carga de archivos LAZ/LAS** con validación automática
- 🎨 **Visualización 3D interactiva** de nubes de puntos
- 🔧 **Reclasificación automática** por altura
- 🏔️ **Dos modos de generación**:
  - **Interpolación en rejilla**: Rápido y eficiente
  - **Modo avanzado**: TIN progresivo profesional + Poisson + edificios extruidos
- 🎛️ **Presets de escala** para impresión 3D
- 📊 **Exportación a STL** lista para imprimir
- 🌓 **Temas claro/oscuro**

## 🏗️ Arquitectura

Proyecto refactorizado con arquitectura modular escalable:

```
laz to mesh/
├── backup/                      # Código original
├── config/                      # Configuración centralizada
│   └── settings.py
├── src/
│   ├── core/                    # Procesamiento principal
│   │   ├── mesh_builder.py
│   │   ├── terrain_processor.py
│   │   ├── vegetation_processor.py
│   │   └── building_processor.py
│   ├── utils/                   # Utilidades
│   │   ├── file_io.py
│   │   ├── geometry.py
│   │   └── validators.py
│   └── ui/                      # Interfaz de usuario
│       ├── theme.py
│       └── components.py
├── app.py                       # Aplicación principal
├── abrir_app.bat               # Lanzador Windows
└── requirements.txt            # Dependencias
```

## 📦 Instalación

### Requisitos

- Python 3.8+
- pip

### Pasos

1. **Clonar o descargar** el proyecto

2. **Instalar dependencias**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Ejecutar la aplicación**:
   - **Windows**: Doble clic en `abrir_app.bat`
   - **Linux/Mac**: `streamlit run app.py`

## 📚 Dependencias

### Obligatorias

- `streamlit` - Interfaz web
- `numpy` - Cálculos numéricos
- `scipy` - Interpolación y geometría
- `laspy` - Lectura de archivos LAZ/LAS
- `plotly` - Visualización 3D

### Opcionales (para modo avanzado)

- `open3d` - Poisson Surface Reconstruction
- `scikit-image` - Marching cubes para metaballs

## 🚀 Uso

1. **Cargar archivo**: Sube un archivo .laz o .las
2. **Revisar clasificación**: Visualiza y reclasifica puntos
3. **Configurar parámetros**: Selecciona escala y modo de generación
4. **Generar malla**: Procesa y descarga el archivo STL

## 🐛 Bugs Corregidos

Esta versión corrige todos los bugs identificados:

✅ Error de indentación en reclasificación  
✅ Mensajes de error corruptos  
✅ Validación de parámetros de usuario  
✅ Optimización de procesamiento de metaballs  
✅ Límites de memoria  
✅ Manejo mejorado de archivos temporales  
✅ `building_plane_tolerance` en modo avanzado  
✅ Código duplicado eliminado

## 🔧 Configuración

Edita `config/settings.py` para personalizar:

- Límites de memoria
- Colores de clases
- Parámetros por defecto
- Escalas de impresión

## 📝 Licencia

Aplicación experimental para generación rápida de mallas a partir de nubes LiDAR.

## 🤝 Contribuir

Para agregar nuevas funcionalidades:

1. **Nuevos procesadores**: Agrega en `src/core/`
2. **Nuevas validaciones**: Agrega en `src/utils/validators.py`
3. **Nuevos componentes UI**: Agrega en `src/ui/components.py`

La arquitectura modular facilita la extensión sin modificar código existente.
