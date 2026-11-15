import cv2
import numpy as np
import os

def procesar_hough_rectas():
    print("--- PROTOTIPO HOUGH RECTAS ---")
    
    nombre_archivo = "BloqueMotor.jpeg"
    
    if not os.path.exists(nombre_archivo):
        print(f"Error: No se encontro el archivo '{nombre_archivo}' en la carpeta actual.")
        print("Asegurate de guardar la imagen real con ese nombre exacto.")
        return

    imagen_original = cv2.imread(nombre_archivo)
    
    gris = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2GRAY)
    suavizado = cv2.GaussianBlur(gris, (5, 5), 0)
    
    bordes = cv2.Canny(suavizado, 50, 150, apertureSize=3)

    lineas = cv2.HoughLinesP(
        bordes, 
        rho=1, 
        theta=np.pi/180, 
        threshold=95,    
        minLineLength=55, 
        maxLineGap=15       
    )

    imagen_salida = imagen_original.copy()
    contador_lineas = 0
    if lineas is not None:
        for linea in lineas:
            coordenadas = np.ravel(linea)
            if len(coordenadas) >= 4:
                x1, y1, x2, y2 = coordenadas[:4]
                cv2.line(imagen_salida, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                contador_lineas += 1

    print(f"Se detectaron y marcaron {contador_lineas} rectas principales en la imagen.")
    
    archivo_salida = "resultado_hough_rectas.jpg"
    cv2.imwrite(archivo_salida, imagen_salida)
    print(f"Imagen de salida guardada exitosamente como '{archivo_salida}'.")

if __name__ == "__main__":
    procesar_hough_rectas()