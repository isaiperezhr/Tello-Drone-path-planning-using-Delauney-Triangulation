"""
Este script implementa un sistema de seguimiento de línea para un dron Tello utilizando control PID 
y técnicas de visión por computadora. Se encarga de lo siguiente:

1. Conexión y gestión básica del dron (despegue, aterrizaje, transmisión de video, etc.).
2. Procesamiento de imagen en tiempo real para detectar líneas (segmentación en HSV y análisis de bordes).
3. Planeación de trayectoria empleando una triangulación de Delaunay para interpolar puntos entre líneas.
4. Cálculo de vectores de desplazamiento y corrección de trayectoria a través de controladores PID, 
   ajustando los ejes lateral, frontal y yaw del dron.
5. Generación de gráficas finales que muestran la evolución de los errores y las acciones de control 
   tras finalizar el vuelo (al presionar 'q').

Para ejecutar el script:
- Asegúrese de tener la librería djitellopy instalada y de que el dron Tello se encuentre disponible.
- Ejecute el script en un entorno donde estén disponibles las dependencias (OpenCV, Numpy, Matplotlib, etc.).
- Al iniciar, el dron despegará y comenzará el seguimiento de la línea detectada en la imagen.
- Presione 'q' en la ventana de video para finalizar la ejecución y aterrizar el dron.
"""


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from djitellopy import tello
import cv2
import time
from scipy.spatial import Delaunay
from scipy.interpolate import interp1d
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops


# Definición de la clase PIDController
class PIDController:
    """
    Controlador PID para el ajuste de la posición o de la trayectoria de un sistema.

    Este controlador se encarga de calcular la señal de control (output) a partir de la 
    diferencia entre un valor objetivo (setpoint) y un valor actual (current_value). 
    Usa tres términos principales: proporcional (kp), integral (ki) y derivativo (kd).

    Atributos:
        kp (float): Ganancia proporcional.
        ki (float): Ganancia integral.
        kd (float): Ganancia derivativa.
        setpoint (float): Valor objetivo que se desea alcanzar.
        prev_error (float): Error en el instante de tiempo anterior.
        integral (float): Suma acumulada del error para el término integral.
        last_time (float): Última marca de tiempo usada para calcular el derivativo.
    """

    def __init__(self, kp, ki, kd, setpoint=0):
        """
        Inicializa un controlador PID.

        Args:
            kp (float): Coeficiente proporcional.
            ki (float): Coeficiente integral.
            kd (float): Coeficiente derivativo.
            setpoint (float, opcional): Valor objetivo. Por defecto, 0.
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()

    def update(self, current_value):
        """
        Calcula la salida del PID dado un valor actual de la variable controlada.

        Args:
            current_value (float): Valor actual del sistema (por ejemplo, desplazamiento).

        Returns:
            float: Salida del PID para reducir la diferencia con respecto al setpoint.
        """
        # Tiempo actual
        current_time = time.time()
        # Tiempo transcurrido desde la última actualización
        elapsed_time = current_time - self.last_time

        # Error (diferencia entre setpoint y valor actual)
        error = self.setpoint - current_value
        # Acumular el error para el término integral
        self.integral += error * elapsed_time
        # Calcular el término derivativo (diferencia de error / tiempo)
        derivative = (error - self.prev_error) / \
            elapsed_time if elapsed_time > 0 else 0

        # Cálculo de la señal de control PID
        output = (self.kp * error) + (self.ki * self.integral) + \
            (self.kd * derivative)

        # Actualizar el error previo y la marca de tiempo
        self.prev_error = error
        self.last_time = current_time

        return output


# =====================================================================
#                     INICIALIZACIÓN DEL DRON
# =====================================================================
me = tello.Tello()         # Creación de la instancia del dron Tello.
me.connect()               # Conexión con el dron.
me.streamon()              # Activación de la transmisión de video.
me.takeoff()               # Orden de despegue.
print(me.get_battery())    # Imprime el nivel de batería en consola.
me.move_down(30)           # Ajuste de altura inicial (baja 30 cm).

# Dimensiones de la ventana de visualización
width, height = 800, 500
# Coordenadas del centro de la imagen (para referencias posteriores)
imx, imy = width // 2, height // 2

# =====================================================================
#                   CONTROLADORES PID PARA CADA EJE
# =====================================================================
# PID para movimiento lateral (izquierda/derecha)
pid_lr = PIDController(kp=0.45, ki=0.01, kd=0.15)

# PID para movimiento hacia adelante/atrás
pid_fb = PIDController(kp=0.7, ki=0.01, kd=0.3)

# PID para control de rotación (yaw)
pid_yaw = PIDController(kp=0.35, ki=0.04, kd=0.35)

# Listas para recopilar datos de error y salidas de control con fines de visualización
LR_Error, FB_Error, YAW_Error = [], [], []
Control_LR, Control_FB, Control_Yaw = [], [], []


def cntrl_plot():
    """
    Genera gráficas con Matplotlib para mostrar la evolución de los errores y las señales
    de control de cada uno de los controladores PID (lateral, frontal y yaw).

    No recibe ni retorna valores; actúa directamente sobre las listas globales de error 
    y señales de control.
    """
    # Cambiamos el backend de Matplotlib para ventana interactiva
    matplotlib.use('TkAgg')

    # Creamos una figura con 3 ejes (subplots) apilados verticalmente
    fig, (LRax, FBax, YAWax) = plt.subplots(
        3, 1, sharex=True, constrained_layout=True
    )

    # Gráfica del error y de la señal de control en eje lateral (Left/Right)
    LRax.plot(LR_Error, label="Error DX", color="blue")
    LRax.plot(Control_LR, label="Control Left/Right", color="cyan")
    LRax.set_title("Error DX & Control Lateral (Left/Right)",
                   loc='left', fontfamily="serif")
    LRax.legend()
    LRax.grid(True)

    # Gráfica del error y de la señal de control en eje frontal (Forward/Backward)
    FBax.plot(FB_Error, label="Error DY", color="green")
    FBax.plot(Control_FB, label="Control Forward/Backward", color="lime")
    FBax.set_title("Error DY & Control Forward/Backward",
                   loc='left', fontfamily="serif")
    FBax.legend()
    FBax.grid(True)

    # Gráfica del error y de la señal de control en yaw (rotación)
    YAWax.plot(YAW_Error, label="Error Ángulo", color="red")
    YAWax.plot(Control_Yaw, label="Control YAW", color="orange")
    YAWax.set_title("Error Ángulo & Control Yaw",
                    loc='left', fontfamily="serif")
    YAWax.legend()
    YAWax.grid(True)

    # Título general de la figura
    plt.suptitle("Drone Seguidor de Línea - Señales de Control & Error",
                 fontweight="bold", fontsize=18, fontfamily="serif")
    plt.show()


def line_detect(RGB):
    """
    Segmenta en la imagen las posibles regiones de la línea a seguir, 
    basándose en umbrales HSV predefinidos.

    Args:
        RGB (np.ndarray): Imagen de entrada en formato RGB. Puede ser uint8 [0–255]
                          o flotante normalizada [0–1].

    Returns:
        BW (np.ndarray): Máscara binaria (dtype=bool) que indica las regiones segmentadas.
        maskedRGBImage (np.ndarray): Imagen donde los píxeles que no pertenecen a la
                                     máscara se ponen en 0 (color negro).
    """
    # Normaliza la imagen si no es de tipo float
    if not np.issubdtype(RGB.dtype, np.floating):
        RGB_norm = RGB.astype(np.float32) / 255.0
    else:
        RGB_norm = RGB.copy()

    # Conversión de RGB a HSV (OpenCV usa H en [0,180], S y V en [0,255])
    hsv = cv2.cvtColor((RGB_norm * 255).astype(np.uint8),
                       cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] /= 180.0  # Se normaliza el hue a [0,1]
    hsv[..., 1] /= 255.0  # Se normaliza la saturación a [0,1]
    hsv[..., 2] /= 255.0  # Se normaliza el valor a [0,1]

    # Umbrales para filtrar la línea (calculados previamente)
    channel1Min, channel1Max = 0.004, 0.172
    channel2Min, channel2Max = 0.360, 1.000
    channel3Min, channel3Max = 0.000, 1.000

    # Crea la máscara binaria aplicando los umbrales en cada canal (H, S y V)
    BW = ((hsv[..., 0] >= channel1Min) & (hsv[..., 0] <= channel1Max) &
          (hsv[..., 1] >= channel2Min) & (hsv[..., 1] <= channel2Max) &
          (hsv[..., 2] >= channel3Min) & (hsv[..., 2] <= channel3Max))

    # Aplica la máscara a la imagen original (regiones fuera de la máscara se ponen a 0)
    maskedRGBImage = RGB.copy()
    maskedRGBImage[~BW] = 0

    return BW, maskedRGBImage


def path_planning_delauney(I, realtime=False):
    """
    Calcula la trayectoria planeada a partir de una imagen, usando Delaunay para 
    interpolar entre bordes izquierdo y derecho de la línea detectada.

    Pasos principales:
      1. Preprocesamiento y detección de bordes (apertura, cierre, eliminación de objetos pequeños).
      2. Análisis de líneas: detección y clasificación en izquierda/derecha, muestreo de puntos.
      3. Planeación de trayectoria usando triangulación de Delaunay para generar puntos intermedios.
      4. Visualización de resultados (si realtime=False, usa matplotlib; si realtime=True, usa OpenCV).

    Args:
        I (np.ndarray): Imagen de entrada en formato RGB.
        realtime (bool): Indica si se muestra el resultado en tiempo real con OpenCV 
                         (True) o con matplotlib (False).

    Returns:
        cx (float): Mediana de las coordenadas x de la trayectoria planeada.
        cy (float): Mediana de las coordenadas y de la trayectoria planeada.
        Pp (np.ndarray): Matriz de tamaño N x 2 con la trayectoria planeada (x, y).
    """
    # === Paso 1: Preprocesamiento y Detección de Bordes ===
    I_mask, _ = line_detect(I)                # Segmentación de la imagen
    # Conversión a uint8 para operaciones morfológicas
    I_mask_uint8 = I_mask.astype(np.uint8)

    # Creación de un elemento estructurante en forma de disco (17x17)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    # Operación de apertura (elimina pequeñas regiones blancas)
    I_open = cv2.morphologyEx(I_mask_uint8, cv2.MORPH_OPEN, kernel)
    # Operación de cierre (cierra pequeñas brechas en las regiones blancas)
    I_closed = cv2.morphologyEx(I_open, cv2.MORPH_CLOSE, kernel)

    # Elimina objetos pequeños con área menor a 500 píxeles
    I_final = remove_small_objects(
        I_closed.astype(bool), min_size=500).astype(np.uint8)

    # Detección de bordes usando el gradiente (Sobel)
    grad_x = cv2.Sobel(I_final, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(I_final, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(grad_x**2 + grad_y**2)
    # Normaliza el resultado a [0,255] para usar threshold
    gradient = cv2.normalize(gradient, None, 0, 255,
                             cv2.NORM_MINMAX).astype(np.uint8)
    # Umbral adaptativo (Otsu) para crear la imagen binaria de bordes
    _, I_edge = cv2.threshold(
        gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    I_edge = I_edge.astype(bool)

    # === Paso 2: Análisis de Líneas (Detección, Clasificación y Muestreo) ===
    # Etiquetado de componentes conectados
    labeled = label(I_edge)
    # Propiedades de las regiones encontradas
    regions = regionprops(labeled)
    imgWidth = I_edge.shape[1]           # Ancho de la imagen

    # Manejo de casos de líneas detectadas
    if not regions:
        # Error si no hay regiones en la imagen
        raise ValueError("No se detectaron líneas en la imagen.")
    elif len(regions) == 1:
        # Solo una línea detectada, se clasifica en izquierda o derecha según centroid
        region = regions[0]
        # min_col, max_col para el bounding box
        min_col, max_col = region.bbox[1], region.bbox[3]
        centroid_x = region.centroid[1]   # Centroid: (row, col) => col es x

        if centroid_x < imgWidth / 2:
            # Se asume que esta única línea es la línea izquierda
            left_line = np.column_stack(
                (region.coords[:, 1], region.coords[:, 0]))
            right_line = np.empty((0, 2))
            print(
                "Sólo se ha detectado la línea izquierda (según bounding box/centroid).")
        else:
            # Se asume que esta única línea es la línea derecha
            left_line = np.empty((0, 2))
            right_line = np.column_stack(
                (region.coords[:, 1], region.coords[:, 0]))
            print("Sólo se ha detectado la línea derecha (según bounding box/centroid).")
    else:
        # Dos o más líneas: se eligen las dos con mayor área y se las clasifica
        print("Se han detectado dos o más líneas. Se eligen las dos de mayor área...")
        regions_sorted = sorted(regions, key=lambda r: r.area, reverse=True)
        region1, region2 = regions_sorted[0], regions_sorted[1]

        # Centroides en (x, y) => (col, row)
        centroid1 = (region1.centroid[1], region1.centroid[0])
        centroid2 = (region2.centroid[1], region2.centroid[0])

        # El que tenga menor x se considera la línea izquierda
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

    # Ordenar los puntos por coordenada y (vertical)
    if left_line.size:
        left_line_sorted = left_line[np.argsort(left_line[:, 1])]
    else:
        left_line_sorted = left_line
    if right_line.size:
        right_line_sorted = right_line[np.argsort(right_line[:, 1])]
    else:
        right_line_sorted = right_line

    # Muestrear cada 10 puntos a lo largo de cada línea para simplificar
    sampling_interval = 10
    left_sampled = left_line_sorted[::sampling_interval] if left_line_sorted.size else left_line_sorted
    right_sampled = right_line_sorted[::sampling_interval] if right_line_sorted.size else right_line_sorted

    # === Paso 3: Planeación de Trayectoria usando Triangulación de Delaunay ===
    inner_cone, outer_cone = right_sampled, left_sampled
    n_inner, n_outer = inner_cone.shape[0], outer_cone.shape[0]
    if n_inner != n_outer:
        print("Aviso: Diferente número de puntos en línea interna y externa. Se recorta al mínimo.")
        min_points = min(n_inner, n_outer)
        inner_cone, outer_cone = inner_cone[:
                                            min_points], outer_cone[:min_points]

    m = inner_cone.shape[0]
    # Se combinan posiciones de línea interna y externa de forma alternada
    P = np.empty((2 * m, 2))
    P[0::2] = inner_cone
    P[1::2] = outer_cone

    xp, yp = [], []
    interv = sampling_interval  # Intervalo de segmentación
    num_pts = P.shape[0]

    # Se recorre la lista de puntos en bloques de tamaño "interv"
    for i in range(interv, num_pts + 1, interv):
        lower_index = max(i - interv - 2, 0)
        P_seg = P[lower_index:i]

        # Se requiere mínimo 3 puntos para hacer Delaunay
        if P_seg.shape[0] < 3:
            continue

        # Manejo de excepciones por si la triangulación falla
        try:
            tri = Delaunay(P_seg)
        except Exception:
            print("Fallo la triangulación Delaunay en un segmento. Segmento omitido.")
            continue

        simplices = tri.simplices
        # Se generan las aristas
        edges = np.vstack([
            np.sort(simplices[:, [0, 1]], axis=1),
            np.sort(simplices[:, [1, 2]], axis=1),
            np.sort(simplices[:, [2, 0]], axis=1)
        ])
        edges = np.unique(edges, axis=0)
        if edges.size == 0:
            continue

        # Filtra aristas que conectan un punto "par" (línea interna) con uno "impar" (línea externa)
        mask = (((edges[:, 0] % 2 == 0) & (edges[:, 1] % 2 == 1)) |
                ((edges[:, 0] % 2 == 1) & (edges[:, 1] % 2 == 0)))
        filtered_edges = edges[mask]
        if filtered_edges.size == 0:
            continue

        # Calcula puntos medios de las aristas
        midpoints = (P_seg[filtered_edges[:, 0]] +
                     P_seg[filtered_edges[:, 1]]) / 2.0
        # Ordena por la coordenada y
        mid_sorted = midpoints[np.argsort(midpoints[:, 1])]
        if mid_sorted.shape[0] < 2:
            continue

        # Interpolación de los puntos medios
        dists = np.sqrt(np.sum(np.diff(mid_sorted, axis=0) ** 2, axis=1))
        total_distance = np.sum(dists)
        dist_bp = np.concatenate(([0], np.cumsum(dists)))

        t_interp = np.linspace(0, total_distance, 100)
        interp_func_x = interp1d(
            dist_bp, mid_sorted[:, 0], kind='cubic', fill_value="extrapolate")
        interp_func_y = interp1d(
            dist_bp, mid_sorted[:, 1], kind='cubic', fill_value="extrapolate")
        xq, yq = interp_func_x(t_interp), interp_func_y(t_interp)

        xp.extend(xq)
        yp.extend(yq)

    # Pp contiene la trayectoria planeada
    Pp = np.column_stack((xp, yp))

    # === Paso 4: Visualización ===
    if not realtime:
        # Visualiza con matplotlib
        plt.figure()
        plt.imshow(I)
        plt.title('Bordes Detectados y Trayectoria Planeada')
        if left_sampled.size:
            plt.plot(left_sampled[:, 0], left_sampled[:, 1], 'ro',
                     markersize=6, linewidth=1.5, label='Puntos Izquierda')
        if right_sampled.size:
            plt.plot(right_sampled[:, 0], right_sampled[:, 1], 'bo',
                     markersize=6, linewidth=1.5, label='Puntos Derecha')
        if Pp.size:
            plt.plot(Pp[:, 0], Pp[:, 1], 'g-',
                     linewidth=2, label='Trayectoria')
        plt.legend()
        plt.show()
    else:
        # Visualiza en tiempo real con OpenCV
        result_img = I.copy()
        # Dibuja puntos muestreados de la izquierda en color azul (por ejemplo)
        if left_sampled.size:
            for pt in left_sampled:
                cv2.circle(result_img, (int(pt[0]), int(
                    pt[1])), 3, (255, 0, 0), -1)
        # Dibuja puntos muestreados de la derecha en color rojo
        if right_sampled.size:
            for pt in right_sampled:
                cv2.circle(result_img, (int(pt[0]), int(
                    pt[1])), 3, (0, 0, 255), -1)
        # Dibuja la trayectoria planeada en verde
        if Pp.size:
            pts = np.int32(Pp.reshape((-1, 1, 2)))
            cv2.polylines(result_img, [pts], False, (0, 255, 0), thickness=2)
        # Convierte de RGB a BGR para mostrar correctamente con OpenCV
        result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        cv2.imshow("Real-Time Path Planning", result_img)

    # Se toman las medianas de la trayectoria planeada como punto objetivo (cx, cy)
    cx, cy = np.median(Pp, axis=0)
    return cx, cy, Pp


def getVector(cx, cy, img, imx, imy):
    """
    Calcula el vector desde el centro de la imagen (imx, imy) hasta el centro (cx, cy)
    de la trayectoria, y dibuja dicho vector en la imagen.

    Args:
        cx (float): Coordenada X del punto de interés.
        cy (float): Coordenada Y del punto de interés.
        img (np.ndarray): Imagen donde se dibuja el vector.
        imx (int): Coordenada X del centro de la imagen.
        imy (int): Coordenada Y del centro de la imagen.

    Returns:
        dx (float): Diferencia en X entre el centro de la imagen y el centro detectado.
        dy (float): Diferencia en Y entre el centro de la imagen y el centro detectado.
        angulo (float): Ángulo en grados del vector, calculado con arctan2.
    """
    # Dibuja un círculo en el centro de la imagen como referencia (en rojo)
    cv2.circle(img, (imx, imy), 10, (0, 0, 255), cv2.FILLED)

    # Calcula la diferencia en cada eje
    dx, dy = imx - cx, imy - cy

    # Dibuja la flecha (vector) desde el centro de la imagen hasta (cx, cy)
    cv2.arrowedLine(img, (imx, imy), (int(cx), int(cy)), (0, 255, 0), 3)

    # Si el vector es cero en alguno de los ejes, retornamos ángulo = 0
    if dx == 0 or dy == 0:
        return dx, dy, 0
    else:
        # Calcula el ángulo en grados usando atan2(dx, dy)
        angulo = np.degrees(np.arctan2(dx, dy))
        return dx, dy, angulo


def sendCommands(dx, dy, angulo):
    """
    Envía los comandos de movimiento al dron basados en los controladores PID,
    ajustando la velocidad en los ejes lateral (left-right), frontal (forward-backward) 
    y la rotación (yaw).

    Args:
        dx (float): Diferencia en X entre el centro de la imagen y el punto objetivo.
        dy (float): Diferencia en Y entre el centro de la imagen y el punto objetivo.
        angulo (float): Ángulo calculado del vector entre el centro de la imagen y el objetivo.
    """
    # Cálculo de la señal de control PID para cada eje, restringiendo el rango con np.clip
    LR = int(np.clip(pid_lr.update(dx), -25, 25))       # Eje lateral
    # Eje frontal (no retroceder)
    fb_output = int(np.clip(pid_fb.update(dy), 0, 35))
    FB = fb_output + 35                                 # Offset para avanzar
    # Rotación, signo negativo para invertir
    YAW = -int(np.clip(pid_yaw.update(angulo), -45, 45))

    # Agregamos los errores y señales de control a las listas globales
    global LR_Error, FB_Error, YAW_Error, Control_LR, Control_FB, Control_Yaw
    LR_Error.append(0.1*dx)
    FB_Error.append(0.1*dy)
    YAW_Error.append(0.1*angulo)
    Control_LR.append(LR)
    Control_FB.append(FB)
    Control_Yaw.append(YAW)

    # Envía los comandos al dron (left-right, forward-backward, up-down, yaw)
    me.send_rc_control(LR, FB, 0, YAW)


# Código principal (loop de captura y control)
if __name__ == '__main__':
    while True:
        # Toma un frame del stream de video
        frame = me.get_frame_read().frame
        # Redimensiona el frame para mantener un tamaño constante
        frame_rgb = cv2.resize(frame, (width, height))

        try:
            # Se detecta la trayectoria en tiempo real (realtime=True)
            cx, cy, Pp = path_planning_delauney(frame_rgb, realtime=True)
            # Calcula el vector hacia el waypoint
            dx, dy, angulo = getVector(cx, cy, frame_rgb, imx, imy)
            # Envía los comandos al dron basados en los PID
            sendCommands(dx, dy, angulo)

        except ValueError as e:
            # Si no se detecta línea, muestra un mensaje en pantalla
            cv2.putText(frame, "No lines detected", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Real-Time Path Planning", frame)

        # Si se presiona 'q', se cierra el programa
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Detener movimiento
            me.send_rc_control(0, 0, 0, 0)
            # Apagar stream y ventanas
            me.streamoff()
            cv2.destroyAllWindows()
            # Aterrizar
            me.land()
            # Generar la gráfica de errores y señales de control
            cntrl_plot()
            break

    cv2.destroyAllWindows()
