import os
import cv2
import numpy as np


def hough_rectas(
    bordes,
    paso_theta_grados=1.0,
    paso_rho=1.0,
    umbral_votos=120,
    supresion_vecindad=(25, 10),
    max_lineas=12,
):
    alto, ancho = bordes.shape
    diag = int(np.ceil(np.hypot(alto, ancho)))

    thetas_deg = np.arange(-90.0, 90.0, paso_theta_grados)
    thetas_rad = np.deg2rad(thetas_deg)
    rhos = np.arange(-diag, diag + paso_rho, paso_rho)

    acumulador = np.zeros((len(rhos), len(thetas_rad)), dtype=np.int32)

    ys, xs = np.nonzero(bordes)

    cos_t = np.cos(thetas_rad)
    sin_t = np.sin(thetas_rad)

    for x, y in zip(xs, ys):
        rho_vals = x * cos_t + y * sin_t
        idx_rho = np.round((rho_vals - rhos[0]) / paso_rho).astype(np.int32)
        validos = (idx_rho >= 0) & (idx_rho < len(rhos))
        idx_theta_validos = np.nonzero(validos)[0]
        idx_rho_validos = idx_rho[validos]
        acumulador[idx_rho_validos, idx_theta_validos] += 1

    acum_copia = acumulador.copy()
    picos = []
    vr, vt = supresion_vecindad

    while len(picos) < max_lineas:
        idx_max = np.unravel_index(np.argmax(acum_copia), acum_copia.shape)
        max_votos = int(acum_copia[idx_max])
        if max_votos < umbral_votos:
            break

        r_idx, t_idx = idx_max
        picos.append((float(rhos[r_idx]), float(thetas_deg[t_idx]), max_votos))

        r_min = max(0, r_idx - vr)
        r_max = min(acum_copia.shape[0], r_idx + vr + 1)
        t_min = max(0, t_idx - vt)
        t_max = min(acum_copia.shape[1], t_idx + vt + 1)
        acum_copia[r_min:r_max, t_min:t_max] = 0

    return picos, acumulador


def main():
    ruta_imagen = "BloqueMotor.jpeg"
    if not os.path.exists(ruta_imagen):
        print(f"No se encontro '{ruta_imagen}' en la carpeta actual.")
        return

    imagen = cv2.imread(ruta_imagen)
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    suavizado = cv2.GaussianBlur(gris, (7, 7), 0)
    bordes = cv2.Canny(suavizado, 90, 200)

    picos, acumulador = hough_rectas(
        bordes,
        paso_theta_grados=1.0,
        paso_rho=1.0,
        umbral_votos=95,
        supresion_vecindad=(35, 10),
        max_lineas=12,
    )

    salida = imagen.copy()
    alto, ancho = bordes.shape
    longitud_trazo = max(alto, ancho)

    for rho, theta_deg, votos in picos:
        theta_rad = np.deg2rad(theta_deg)
        a = np.cos(theta_rad)
        b = np.sin(theta_rad)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(round(x0 + longitud_trazo * (-b))), int(round(y0 + longitud_trazo * (a))))
        pt2 = (int(round(x0 - longitud_trazo * (-b))), int(round(y0 - longitud_trazo * (a))))
        cv2.line(salida, pt1, pt2, (0, 255, 0), 2)

    print(f"Maximos detectados en el acumulador (rho, theta): {len(picos)}")
    print(f"Valor maximo de votos alcanzado: {int(acumulador.max())}")
    for i, (rho, th, v) in enumerate(picos, start=1):
        print(f"  Recta #{i}: rho={rho:.1f}, theta={th:.1f}°, votos={v}")

    cv2.imwrite("resultado_hough_rectas.png", salida)
    print("Imagen guardada como 'resultado_hough_rectas.png'.")


if __name__ == "__main__":
    main()