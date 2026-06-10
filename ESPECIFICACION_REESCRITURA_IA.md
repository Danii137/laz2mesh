# Especificacion funcional para reconstruir LAZ2Mesh

## Instruccion principal para la IA

Construir una aplicacion nueva desde cero a partir de esta especificacion.

El proyecto actual solo debe utilizarse para comprender la idea general, los tipos de datos y el resultado buscado. No se debe copiar su arquitectura, sus algoritmos, su interfaz ni sus decisiones tecnicas. Parte de la implementacion existente es experimental, esta duplicada o no funciona correctamente.

La nueva solucion debe ser mas pequena, ordenada, comprobable y mantenible. Antes de incorporar una funcion avanzada, debe funcionar correctamente el recorrido principal completo.

## Objetivo del producto

LAZ2Mesh debe convertir una nube de puntos LiDAR en formato LAS o LAZ en una malla 3D limpia, visualmente comprensible y apta para exportar, especialmente como modelo topografico para impresion 3D.

La aplicacion debe ayudar a una persona no especialista a:

1. Cargar una nube de puntos.
2. Comprender su contenido mediante una vista previa.
3. Elegir que elementos quiere representar.
4. Generar una malla con una configuracion sencilla.
5. Revisar el resultado antes de descargarlo.
6. Exportar un archivo STL valido.

El valor principal no es ofrecer muchos controles, sino producir resultados predecibles con pocos pasos.

## Usuarios previstos

- Arquitectos, topografos y tecnicos que trabajan con datos LiDAR.
- Usuarios que desean crear maquetas fisicas de terrenos, edificios y vegetacion.
- Personas con conocimientos limitados de procesamiento de nubes de puntos.

La interfaz debe utilizar lenguaje claro y explicar las decisiones importantes sin exigir conocimientos sobre algoritmos de triangulacion.

## Entrada principal

La entrada principal es un archivo `.las` o `.laz` que puede contener:

- Coordenadas tridimensionales.
- Clasificacion de puntos LiDAR.
- Informacion de referencia espacial, cuando este disponible.
- Terreno, edificios, vegetacion, agua, ruido y otras clases.

La aplicacion debe validar el archivo, informar de problemas comprensibles y evitar bloquearse con archivos grandes. La vista previa puede utilizar una muestra representativa, pero la generacion final debe conservar el nivel de calidad seleccionado por el usuario.

Como entrada opcional, se pueden admitir huellas de edificios en un formato geografico estandar. Esta funcion no debe ser necesaria para completar el flujo principal.

## Flujo principal de usuario

### 1. Cargar y analizar

El usuario selecciona un archivo LAS o LAZ. La aplicacion lo analiza y muestra, como minimo:

- Numero de puntos.
- Extension horizontal y rango de alturas.
- Clases LiDAR encontradas y cantidad de puntos por clase.
- Sistema de coordenadas, si puede determinarse con seguridad.
- Avisos relevantes sobre calidad, tamaño o ausencia de clasificacion.

### 2. Revisar la nube

La aplicacion presenta una vista previa interactiva de la nube de puntos, coloreada por clase.

El usuario debe poder:

- Mostrar u ocultar clases.
- Excluir ruido y clases no deseadas.
- Corregir clasificaciones de forma sencilla cuando sea necesario.
- Limitar el area que se procesara.
- Restablecer facilmente cualquier cambio.

La edicion manual avanzada no es una prioridad inicial. Son preferibles herramientas simples, reversibles y fiables.

### 3. Configurar el resultado

Debe existir un modo sencillo como opcion principal. En este modo, el usuario elige:

- Escala o tamaño aproximado del modelo.
- Nivel de calidad: borrador, normal o alto.
- Elementos incluidos: terreno, edificios y vegetacion.
- Inclusion de una base inferior para impresion.

La aplicacion debe traducir estas decisiones a parametros internos coherentes. No debe exponer decenas de valores tecnicos sin necesidad.

Puede existir un modo avanzado separado para usuarios expertos, pero no debe complicar el modo sencillo ni ser necesario para obtener un buen resultado.

### 4. Generar la malla

La aplicacion procesa por separado los elementos seleccionados y los integra en un unico resultado:

- El terreno debe conservar la forma topografica relevante sin ruido excesivo.
- Los edificios deben reconocerse como volumenes diferenciados y legibles.
- La vegetacion debe representarse de forma simplificada y adecuada a la escala de una maqueta, no como una copia literal de cada punto.
- Las capas no deben deformarse ni destruirse entre ellas durante la combinacion.

Durante el proceso se debe mostrar progreso real, etapa actual y errores accionables.

### 5. Revisar el resultado

Antes de exportar, el usuario debe ver una previsualizacion de la malla final y un resumen que incluya:

- Dimensiones reales y dimensiones resultantes del modelo.
- Numero aproximado de vertices y caras.
- Elementos incluidos.
- Estado de integridad de la malla.
- Advertencias sobre detalle demasiado fino, huecos o complejidad excesiva.

El usuario debe poder volver a la configuracion, modificarla y regenerar sin tener que cargar de nuevo el archivo.

### 6. Exportar

La salida principal es un archivo STL preparado para abrirse en un laminador de impresion 3D.

La exportacion debe:

- Respetar la escala elegida.
- Usar unidades claramente indicadas.
- Evitar triangulos degenerados, vertices invalidos y geometria duplicada.
- Intentar producir una malla cerrada cuando se solicite una base imprimible.
- No presentarse como correcta si falla una validacion esencial.

Otros formatos de exportacion pueden añadirse mas adelante, pero no forman parte del nucleo inicial.

## Comportamiento esperado por tipo de elemento

### Terreno

El terreno es la capa imprescindible y debe ser la primera en funcionar bien. Debe generar una superficie continua, conservar pendientes y accidentes importantes, reducir puntos anormales y evitar ondulaciones artificiales.

### Edificios

Los edificios son opcionales. Deben aparecer como volumenes reconocibles sobre el terreno, con contornos y alturas coherentes. La primera version puede usar formas simplificadas. Es mas importante que los edificios esten bien situados y no dañen el terreno que reproducir tejados complejos.

### Vegetacion

La vegetacion es opcional y debe estar adaptada a la representacion en maqueta. Debe agruparse y simplificarse segun la escala. No es necesario reconstruir cada arbol de manera realista. La prioridad es que sea distinguible, imprimible y que no genere una malla descontroladamente pesada.

## Prioridades de desarrollo

Implementar y validar en este orden:

1. Lectura fiable de LAS y LAZ.
2. Analisis y vista previa eficiente.
3. Seleccion de clases y recorte del area.
4. Generacion correcta de terreno.
5. Escalado, base y exportacion STL valida.
6. Edificios simplificados.
7. Vegetacion simplificada.
8. Funciones geograficas y opciones avanzadas.

No avanzar a una etapa si la anterior no tiene pruebas y un resultado estable.

## Requisitos de experiencia de usuario

- Flujo guiado y corto, dividido en pasos claros.
- Valores predeterminados razonables.
- Una accion principal evidente en cada pantalla.
- Errores escritos en lenguaje comprensible.
- Posibilidad de volver atras sin perder el trabajo.
- Interfaz fluida incluso cuando el procesamiento final tarde.
- Vista previa separada del procesamiento de alta calidad.
- Controles avanzados ocultos por defecto.

## Requisitos tecnicos generales

La IA debe elegir una arquitectura nueva y modular, con responsabilidades separadas para:

- Lectura y validacion de datos.
- Estado del proyecto o sesion.
- Procesamiento de terreno.
- Procesamiento opcional de edificios y vegetacion.
- Construccion, reparacion y validacion de mallas.
- Visualizacion.
- Interfaz de usuario.
- Exportacion.

Los algoritmos de procesamiento deben poder ejecutarse y probarse sin iniciar la interfaz grafica. La logica principal no debe quedar mezclada con componentes visuales.

Se deben evitar archivos de codigo excesivamente grandes, estados globales dificiles de seguir, funciones duplicadas y dependencias opcionales obligatorias para arrancar la aplicacion.

## Calidad y pruebas

La nueva aplicacion debe incluir pruebas automatizadas sobre casos pequeños y conocidos:

- Archivo valido y archivo dañado.
- Nube con y sin clasificacion.
- Terreno plano, inclinado e irregular.
- Clases vacias o insuficientes.
- Recorte de una zona.
- Escalado a dimensiones de maqueta.
- Exportacion y lectura posterior del STL.
- Deteccion de caras invalidas y bordes abiertos.
- Procesamiento repetido con la misma entrada y configuracion.

Tambien debe existir un conjunto pequeño de datos de demostracion que permita comprobar el flujo completo sin usar archivos privados.

## Criterios minimos de aceptacion

La primera version se considera util cuando cumple todo lo siguiente:

1. Puede abrir un LAS o LAZ valido y mostrar un resumen sin errores.
2. Permite visualizar y seleccionar las clases que se procesaran.
3. Genera una malla de terreno coherente con una configuracion predeterminada.
4. Permite elegir escala y añadir una base.
5. Muestra una vista previa del resultado.
6. Exporta un STL que puede volver a abrirse correctamente.
7. Informa si la malla no es valida o no esta cerrada.
8. Mantiene la interfaz separada de la logica de procesamiento.
9. Incluye pruebas del recorrido principal.
10. Puede instalarse y ejecutarse siguiendo instrucciones breves y reproducibles.

## Funciones no prioritarias

Estas funciones pueden estudiarse despues de estabilizar el nucleo:

- Reconstruccion detallada de cubiertas de edificios.
- Diferentes estilos artisticos de vegetacion.
- Integracion automatica con servicios catastrales.
- Importacion y reparacion de mallas externas.
- Compatibilidad con herramientas externas de modelado o captura 3D.
- Gran cantidad de parametros manuales de triangulacion.
- Temas visuales y personalizacion estetica avanzada.

No deben condicionar la arquitectura ni retrasar el flujo principal.

## Decisiones que la nueva IA no debe heredar

- No asumir que una funcion existente es correcta porque ya esta implementada.
- No copiar archivos completos ni reconstruir el proyecto mediante parches sucesivos.
- No mantener funciones duplicadas por compatibilidad con versiones antiguas.
- No mezclar en un mismo modulo la interfaz, la lectura de archivos y los algoritmos 3D.
- No ocultar errores de geometria para permitir la descarga.
- No añadir opciones avanzadas antes de definir valores predeterminados fiables.
- No depender de programas externos para completar el flujo basico.
- No intentar resolver todos los tipos de nube LiDAR en la primera version.

## Resultado solicitado a la IA desarrolladora

Antes de programar, la IA debe proponer:

1. Una arquitectura breve y justificada.
2. Un alcance concreto para la primera version.
3. Los datos de prueba que utilizara.
4. Las reglas de validacion de la malla.
5. Un plan de implementacion por etapas con resultados demostrables.

Despues debe construir primero un recorrido vertical minimo: cargar una nube pequeña, generar terreno, previsualizarlo y exportar un STL validado. Las capas y funciones adicionales se incorporaran solo cuando ese recorrido sea estable.
