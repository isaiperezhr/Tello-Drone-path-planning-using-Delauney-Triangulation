# Proyecto de Seguimiento de Línea con Dron Tello

Este repositorio contiene un sistema de seguimiento de línea para el dron Tello utilizando **control PID** y **técnicas de visión por computadora**. El dron es capaz de detectar una línea en la imagen, calcular una trayectoria segura y corregir su movimiento en tiempo real.

## Características
- **Control PID** para mantener el dron alineado con la línea detectada.
- **Procesamiento de imagen** en tiempo real con OpenCV.
- **Planeación de trayectoria** mediante triangulación de Delaunay.
- **Análisis de errores y generación de gráficas** de control.

## Requisitos
Asegúrese de tener instaladas las siguientes dependencias antes de ejecutar el código:

```bash
pip install djitellopy opencv-python numpy matplotlib scipy scikit-image
```

## Instalación
1. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/tu-usuario/tu-repositorio.git
   cd tu-repositorio
   ```
2. **Ejecutar el script principal**:
   ```bash
   python seguimiento_tello.py
   ```

## Uso
- **Antes de iniciar**:
  - Asegúrese de que el dron Tello esté encendido y disponible.
  - Conéctese al Wi-Fi del dron.
- **Durante la ejecución**:
  - El dron despegará automáticamente y seguirá la línea detectada.
  - Para detener el vuelo y aterrizar, presione la tecla **'q'**.

## Explicación del Control PID
El sistema emplea tres controladores PID para ajustar la posición del dron:
- **Eje lateral (izquierda/derecha)**: Mantiene el dron centrado sobre la línea.
- **Eje frontal (adelante/atrás)**: Ajusta la velocidad para seguir la trayectoria.
- **Rotación (yaw)**: Corrige la orientación del dron según el ángulo de la línea detectada.

## Procesamiento de Imagen y Planeación de Trayectoria
1. **Segmentación de la línea**: Conversión a HSV y filtrado de color.
2. **Detección de bordes**: Aplicación de Sobel y umbralización con Otsu.
3. **Clasificación de líneas**: Detección de componentes conectados y clasificación en izquierda/derecha.
4. **Planeación de trayectoria**: Uso de triangulación de Delaunay para interpolar puntos intermedios.

## Contribuir
Si deseas contribuir a este proyecto:
1. **Haz un fork** del repositorio.
2. **Crea una nueva rama**:
   ```bash
   git checkout -b mi-nueva-funcionalidad
   ```
3. **Realiza tus cambios y sube la rama**:
   ```bash
   git commit -m "Agregada nueva funcionalidad X"
   git push origin mi-nueva-funcionalidad
   ```
4. **Abre un Pull Request** en GitHub.

## Licencia
Este proyecto está licenciado bajo los términos del archivo [LICENSE](LICENSE).

---

Si necesitas ajustes o quieres incluir más detalles, dime y lo modificamos. 🚀

