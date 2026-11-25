import os
import cv2
import numpy as np


def hough_circunferencia(
    bordes,
    radio_R,
    umbral_votos=45,
    supresion_radio=35,
    max_circulos=6,
):
    alto, ancho = bordes.shape
    acumulador = np.zeros((alto, ancho), dtype=np.int32)

    ys, xs = np.nonzero(bordes)
    angulos = np.linspace(0.0, 2.0 * np.pi, 180, endpoint=False)
    dx = radio_R * np.cos(angulos)
    dy = radio_R * np.sin(angulos)

    for x, y in zip(xs, ys):
        x0_vals = np.round(x - dx).astype(np.int32)
        y0_vals = np.round(y - dy).astype(np.int32)

        validos = (x0_vals >= 0) & (x0_vals < ancho) & (y0_vals >= 0) & (y0_vals < alto)
        acumulador[y0_vals[validos], x0_vals[validos]] += 1

    acum_copia = acumulador.copy()
    centros_detectados = []

    while len(centros_detectados) < max_circulos:
        y_max, x_max = np.unravel_index(np.argmax(acum_copia), acum_copia.shape)
        votos = int(acum_copia[y_max, x_max])
        if votos < umbral_votos:
            break

        centros_detectados.append((int(x_max), int(y_max), int(radio_R), votos))

        y_min = max(0, y_max - supresion_radio)
        y_max_lim = min(alto, y_max + supresion_radio + 1)
        x_min = max(0, x_max - supresion_radio)
        x_max_lim = min(ancho, x_max + supresion_radio + 1)
        acum_copia[y_min:y_max_lim, x_min:x_max_lim] = 0

    return centros_detectados, acumulador


def main():
    ruta_imagen = "BloqueMotor.jpeg"
    if not os.path.exists(ruta_imagen):
        print("No se encontro la imagen del block de motor en la carpeta actual.")
        return

    imagen = cv2.imread(ruta_imagen)
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    suavizado = cv2.GaussianBlur(gris, (7, 7), 0)
    bordes = cv2.Canny(suavizado, 90, 200)

    radio_objetivo = 32
    detecciones, acumulador = hough_circunferencia(
        bordes,
        radio_R=radio_objetivo,
        umbral_votos=35,
        supresion_radio=35,
        max_circulos=6,
    )

    salida = imagen.copy()
    for idx, (cx, cy, r, votos) in enumerate(detecciones, start=1):
        cv2.circle(salida, (cx, cy), r, (255, 0, 0), 2)
        cv2.circle(salida, (cx, cy), 4, (0, 0, 255), -1)
        print(f"Candidato #{idx}: Centro A = (X={cx}, Y={cy}), R={r}, votos={votos}")

    print(f"Total de maximos encontrados: {len(detecciones)} | Voto maximo: {int(acumulador.max())}")
    cv2.imwrite("resultado_hough_circunferencias.png", salida)
    print("Imagen guardada como 'resultado_hough_circunferencias.png'.")


if __name__ == "__main__":
    main()