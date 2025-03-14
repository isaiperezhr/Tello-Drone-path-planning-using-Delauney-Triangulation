# ----------------------------------------------------------------------------------
# Este script implementa un sistema de planificación de trayectoria para un dron Tello
# basado en la detección de líneas en imágenes. Se utilizan técnicas de segmentación
# en el espacio de color HSV y la triangulación de Delaunay para determinar una ruta
# entre líneas izquierda y derecha, ofreciendo así una guía de navegación. El flujo
# básico es el siguiente:
#   1. Captura de la transmisión de video del dron Tello.
#   2. Procesamiento de la imagen para detección de líneas (segmentación HSV, operaciones
#      morfológicas, eliminación de objetos pequeños y detección de bordes).
#   3. Identificación de líneas izquierda/derecha mediante análisis de componentes
#      conectados (regiones) y clasificación basada en centroides.
#   4. Planificación de trayectoria utilizando triangulación de Delaunay entre los
#      puntos muestreados de ambas líneas y posterior generación de un camino intermedio.
#   5. Visualización en tiempo real o con matplotlib, dependiendo del parámetro
#      `realtime`.
#
# Uso y contexto:
#   - Este script requiere la librería djitellopy para conectarse con el dron Tello y
#     recibir la transmisión de video, así como las librerías de procesamiento de
#     imágenes (OpenCV, NumPy, scikit-image).
#   - Se asume que el dron tiene suficiente batería y se encuentra en un entorno con
#     líneas visibles que el algoritmo pueda detectar.
#   - Para iniciar el procesamiento en tiempo real, se ejecuta el script y se espera la
#     orden 'q' para terminar la sesión.
# ----------------------------------------------------------------------------------


import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.interpolate import interp1d
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops
from djitellopy import tello


def line_detect(RGB):
    """
    Función para la detección de líneas basadas en un umbral en el espacio de color HSV.

    Parámetros:
        RGB (np.ndarray): Imagen de entrada en formato RGB (dtype=uint8 o float).
                          Puede tener valores en el rango 0-255 o estar normalizada.

    Retorna:
        BW (np.ndarray): Máscara binaria (bool) que indica las regiones segmentadas.
        maskedRGBImage (np.ndarray): Imagen donde los píxeles que no pertenecen a la
                                     máscara se establecen a cero.
    """
    # Verificamos si la imagen está en formato float. Si no, la normalizamos dividiendo por 255.
    if not np.issubdtype(RGB.dtype, np.floating):
        RGB_norm = RGB.astype(np.float32) / 255.0
    else:
        RGB_norm = RGB.copy()

    # Convertimos la imagen a HSV utilizando OpenCV.
    # Importante: OpenCV asume Hue en [0, 180], Saturation en [0, 255] y Value en [0, 255].
    # Por ello, volvemos a escalar la imagen a 0-255 antes de la conversión y luego reescalamos.
    hsv = cv2.cvtColor((RGB_norm * 255).astype(np.uint8),
                       cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] /= 180.0  # Normalizamos el canal Hue a rango [0,1].
    hsv[..., 1] /= 255.0  # Normalizamos el canal Saturation a rango [0,1].
    hsv[..., 2] /= 255.0  # Normalizamos el canal Value a rango [0,1].

    # Definimos los umbrales para detectar las líneas (generado a partir de código de MATLAB).
    channel1Min, channel1Max = 0.004, 0.172
    channel2Min, channel2Max = 0.360, 1.000
    channel3Min, channel3Max = 0.000, 1.000

    # Creamos la máscara binaria basándonos en los rangos establecidos para HSV.
    BW = ((hsv[..., 0] >= channel1Min) & (hsv[..., 0] <= channel1Max) &
          (hsv[..., 1] >= channel2Min) & (hsv[..., 1] <= channel2Max) &
          (hsv[..., 2] >= channel3Min) & (hsv[..., 2] <= channel3Max))

    # Creamos la imagen enmascarada: establecemos en cero los píxeles que no están en la máscara.
    maskedRGBImage = RGB.copy()
    maskedRGBImage[~BW] = 0

    return BW, maskedRGBImage


def path_planning_delauney(I, realtime=False):
    """
    Función para el cálculo de la trayectoria planificada en una imagen dada,
    utilizando triangulación de Delaunay y procesamiento de líneas.

    Pasos principales:
      1. Preprocesamiento y detección de bordes (edge detection).
      2. Análisis de líneas: etiquetado, clasificación como izquierda/derecha y muestreo.
      3. Planificación de trayectoria con triangulación de Delaunay.
      4. Visualización de los resultados:
         - Si realtime==False, se muestra con matplotlib.
         - Si realtime==True, se dibuja directamente sobre la imagen con OpenCV.

    Parámetros:
        I (np.ndarray): Imagen de entrada en formato RGB.
        realtime (bool): Si es True, se hace la visualización en tiempo real usando cv2.imshow().
                         Si es False, se visualiza con matplotlib.

    Retorna:
        Pp (np.ndarray): Trayectoria planificada como un arreglo N x 2 con las coordenadas [x, y].
    """
    # === Paso 1: Preprocesamiento y Detección de Bordes ===
    # Obtenemos la máscara binaria de la imagen y la imagen enmascarada (no es usada aquí).
    I_mask, _ = line_detect(I)

    # Convertimos la máscara booleana a uint8 para operaciones morfológicas.
    I_mask_uint8 = I_mask.astype(np.uint8)

    # Creamos un elemento estructurante en forma de disco de tamaño 17x17
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))

    # Operación de apertura (elimina ruido aislado).
    I_open = cv2.morphologyEx(I_mask_uint8, cv2.MORPH_OPEN, kernel)
    # Operación de cierre (rellena huecos pequeños).
    I_closed = cv2.morphologyEx(I_open, cv2.MORPH_CLOSE, kernel)

    # Eliminamos objetos pequeños con área menor a 500 píxeles.
    I_final = remove_small_objects(
        I_closed.astype(bool), min_size=500).astype(np.uint8)

    # Detección de bordes con el método de Sobel.
    grad_x = cv2.Sobel(I_final, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(I_final, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(grad_x**2 + grad_y**2)

    # Normalizamos el gradiente a rango [0, 255] para un mejor manejo en OpenCV.
    gradient = cv2.normalize(gradient, None, 0, 255,
                             cv2.NORM_MINMAX).astype(np.uint8)

    # Umbralización con Otsu para obtener una imagen binaria de bordes.
    _, I_edge = cv2.threshold(
        gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    I_edge = I_edge.astype(bool)

    # === Paso 2: Análisis de Líneas (Detección, Clasificación y Muestreo) ===
    # Etiquetamos los componentes conectados en la imagen de bordes.
    labeled = label(I_edge)
    # Obtenemos las propiedades de cada región (cada línea detectada).
    regions = regionprops(labeled)
    # Ancho de la imagen para clasificar izquierda/derecha.
    imgWidth = I_edge.shape[1]

    # Verificamos si se han detectado regiones (líneas).
    if not regions:
        raise ValueError("No se han detectado líneas en la imagen.")
    elif len(regions) == 1:
        # Si solo se detecta una línea, clasificamos según su posición horizontal (bbox y centroide).
        region = regions[0]
        # Obtenemos la posición mínima y máxima en columnas del bounding box.
        min_col, max_col = region.bbox[1], region.bbox[3]
        # Obtenemos la coordenada x del centroide (recordar que centroid[1] es col, centroid[0] es row).
        centroid_x = region.centroid[1]
        if centroid_x < imgWidth / 2:
            # Clasificamos la línea como línea izquierda.
            left_line = np.column_stack(
                (region.coords[:, 1], region.coords[:, 0]))
            right_line = np.empty((0, 2))
            print(
                "Solo se detectó línea izquierda con base en el bounding box y centroide.")
        else:
            # Clasificamos la línea como línea derecha.
            left_line = np.empty((0, 2))
            right_line = np.column_stack(
                (region.coords[:, 1], region.coords[:, 0]))
            print(
                "Solo se detectó línea derecha con base en el bounding box y centroide.")
    else:
        # Si hay 2 o más líneas, tomamos las dos más grandes por área.
        print("Se detectaron dos o más líneas. Seleccionando las dos mayores por área...")
        regions_sorted = sorted(regions, key=lambda r: r.area, reverse=True)
        region1, region2 = regions_sorted[0], regions_sorted[1]

        # Obtenemos los centroides de las dos regiones.
        centroid1 = (region1.centroid[1], region1.centroid[0])
        centroid2 = (region2.centroid[1], region2.centroid[0])

        # Clasificamos la región con menor coordenada x del centroide como izquierda.
        if centroid1[0] < centroid2[0]:
            left_line = np.column_stack(
                (region1.coords[:, 1], region1.coords[:, 0]))
            right_line = np.column_stack(
                (region2.coords[:, 1], region2.coords[:, 0]))
            print("Región 1 clasificada como izquierda; Región 2 como derecha.")
        else:
            left_line = np.column_stack(
                (region2.coords[:, 1], region2.coords[:, 0]))
            right_line = np.column_stack(
                (region1.coords[:, 1], region1.coords[:, 0]))
            print("Región 2 clasificada como izquierda; Región 1 como derecha.")

    # Ordenamos las líneas detectadas según la coordenada y (vertical).
    left_line_sorted = left_line[np.argsort(
        left_line[:, 1])] if left_line.size else left_line
    right_line_sorted = right_line[np.argsort(
        right_line[:, 1])] if right_line.size else right_line

    # Muestreamos puntos cada 10 píxeles.
    sampling_interval = 10
    left_sampled = left_line_sorted[::sampling_interval] if left_line_sorted.size else left_line_sorted
    right_sampled = right_line_sorted[::sampling_interval] if right_line_sorted.size else right_line_sorted

    # === Paso 3: Planificación de Trayectoria con Triangulación de Delaunay ===
    inner_cone, outer_cone = right_sampled, left_sampled
    n_inner, n_outer = inner_cone.shape[0], outer_cone.shape[0]

    # Verificamos si ambas líneas tienen el mismo número de puntos. Si no, recortamos al mínimo.
    if n_inner != n_outer:
        print("Advertencia: el número de puntos de la línea interna y externa no coincide. Recortando al mínimo.")
        min_points = min(n_inner, n_outer)
        inner_cone, outer_cone = inner_cone[:
                                            min_points], outer_cone[:min_points]

    m = inner_cone.shape[0]

    # Combinamos las posiciones interna y externa de forma alternada para la triangulación.
    P = np.empty((2 * m, 2))
    P[0::2] = inner_cone
    P[1::2] = outer_cone

    xp, yp = []
    # Intervalo de segmentación para la triangulación.
    interv = sampling_interval
    num_pts = P.shape[0]

    xp, yp = [], []
    # Recorremos los puntos en segmentos, generando la triangulación de Delaunay por tramos.
    for i in range(interv, num_pts + 1, interv):
        # Definimos el índice inferior del segmento, asegurándonos de no ir por debajo de 0.
        lower_index = max(i - interv - 2, 0)
        P_seg = P[lower_index:i]

        # Si hay menos de 3 puntos en este segmento, no podemos triangular.
        if P_seg.shape[0] < 3:
            continue

        # Intentamos realizar la triangulación de Delaunay.
        try:
            tri = Delaunay(P_seg)
        except Exception:
            print(
                "Fallo en la triangulación de Delaunay en un segmento; se salta este segmento.")
            continue

        simplices = tri.simplices
        # Obtenemos los bordes (aristas) de cada triángulo y eliminamos duplicados.
        edges = np.vstack([
            np.sort(simplices[:, [0, 1]], axis=1),
            np.sort(simplices[:, [1, 2]], axis=1),
            np.sort(simplices[:, [2, 0]], axis=1)
        ])
        edges = np.unique(edges, axis=0)

        # Verificamos si no hay bordes.
        if edges.size == 0:
            continue

        # Filtramos los bordes para dejar solo aquellos que conectan un punto par (línea interna)
        # con uno impar (línea externa).
        mask = (((edges[:, 0] % 2 == 0) & (edges[:, 1] % 2 == 1)) |
                ((edges[:, 0] % 2 == 1) & (edges[:, 1] % 2 == 0)))
        filtered_edges = edges[mask]
        if filtered_edges.size == 0:
            continue

        # Calculamos puntos medios de dichos bordes (aristas).
        midpoints = (P_seg[filtered_edges[:, 0]] +
                     P_seg[filtered_edges[:, 1]]) / 2.0

        # Ordenamos los puntos medios por la coordenada y (vertical).
        mid_sorted = midpoints[np.argsort(midpoints[:, 1])]

        # Verificamos de nuevo que haya al menos 2 puntos medios.
        if mid_sorted.shape[0] < 2:
            continue

        # Calculamos las distancias acumuladas entre puntos consecutivos.
        dists = np.sqrt(np.sum(np.diff(mid_sorted, axis=0) ** 2, axis=1))
        total_distance = np.sum(dists)
        dist_bp = np.concatenate(([0], np.cumsum(dists)))

        # Generamos un conjunto de muestras uniforme a lo largo de la distancia total.
        t_interp = np.linspace(0, total_distance, 100)

        # Definimos funciones de interpolación cúbica para x e y.
        interp_func_x = interp1d(
            dist_bp, mid_sorted[:, 0], kind='cubic', fill_value="extrapolate")
        interp_func_y = interp1d(
            dist_bp, mid_sorted[:, 1], kind='cubic', fill_value="extrapolate")

        # Obtenemos las coordenadas x e y interpoladas.
        xq, yq = interp_func_x(t_interp), interp_func_y(t_interp)

        # Extendemos la lista de coordenadas finales con las interpolaciones del segmento.
        xp.extend(xq)
        yp.extend(yq)

    # Convertimos la lista de coordenadas finales a un arreglo N x 2.
    Pp = np.column_stack((xp, yp))

    # === Paso 4: Visualización ===
    if not realtime:
        # Visualización con matplotlib (no en tiempo real).
        plt.figure()
        plt.imshow(I)
        plt.title('Detección de Bordes y Trayectoria Planificada')
        if left_sampled.size:
            plt.plot(left_sampled[:, 0], left_sampled[:, 1],
                     'ro', markersize=6, linewidth=1.5, label='Puntos Muestrados Izquierda')
        if right_sampled.size:
            plt.plot(right_sampled[:, 0], right_sampled[:, 1],
                     'bo', markersize=6, linewidth=1.5, label='Puntos Muestrados Derecha')
        if Pp.size:
            plt.plot(Pp[:, 0], Pp[:, 1],
                     'g-', linewidth=2, label='Trayectoria Planificada')
        plt.legend()
        plt.show()
    else:
        # Visualización en tiempo real con OpenCV.
        result_img = I.copy()
        # Dibujamos los puntos muestreados en la línea izquierda (color rojo en OpenCV es BGR -> (0,0,255)).
        if left_sampled.size:
            for pt in left_sampled:
                cv2.circle(result_img, (int(pt[0]), int(
                    pt[1])), 3, (255, 0, 0), -1)
        # Dibujamos los puntos muestreados en la línea derecha (color azul -> (0,0,255) en BGR).
        if right_sampled.size:
            for pt in right_sampled:
                cv2.circle(result_img, (int(pt[0]), int(
                    pt[1])), 3, (0, 0, 255), -1)
        # Dibujamos la trayectoria planificada (línea verde -> (0,255,0) en BGR).
        if Pp.size:
            pts = np.int32(Pp.reshape((-1, 1, 2)))
            cv2.polylines(result_img, [pts], False, (0, 255, 0), thickness=2)
        # Convertimos de RGB a BGR para la correcta visualización en OpenCV.
        result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        cv2.imshow("Planificación de Trayectoria en Tiempo Real", result_img)

    return Pp


# === Inicialización del dron ===
me = tello.Tello()   # Creamos la instancia del dron Tello.
me.connect()         # Conectamos con el dron Tello.
me.streamon()        # Iniciamos la transmisión de video en tiempo real.
me.takeoff()         # Despegue del dron.
print(me.get_battery())  # Imprimimos el nivel actual de batería del dron.
# Ajustamos la altura inicial del dron moviéndolo 30 cm hacia abajo.
me.move_down(30)

# Definimos el ancho y alto de la ventana donde se mostrará el video.
width, height = 800, 500
imx, imy = width // 2, height // 2

if __name__ == '__main__':
    while True:
        # Leemos el cuadro actual transmitido por la cámara del dron.
        frame = me.get_frame_read().frame

        # Ajustamos la resolución de la imagen y la "volteamos" (flip) si fuera necesario.
        frame_rgb = cv2.flip(cv2.resize(frame, (width, height)))

        # NOTA: Si se quisiera convertir a RGB antes de procesar:
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            # Procesamos el cuadro en tiempo real (realtime=True para usar OpenCV).
            planned_path = path_planning_delauney(frame_rgb, realtime=True)
        except ValueError as e:
            # Si no se detectan líneas, mostramos un mensaje de error en la ventana.
            cv2.putText(frame, "No se detectaron lineas", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Planificación de Trayectoria en Tiempo Real", frame)

        # Si se presiona la tecla 'q', salimos del bucle principal.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cerramos todas las ventanas de OpenCV.
    cv2.destroyAllWindows()
