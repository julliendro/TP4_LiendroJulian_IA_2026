import cv2
import numpy as np
import os

def procesar_hough_circunferencias():
    print("--- PROTOTIPO HOUGH CIRCUNFERENCIAS ---")
    
    nombre_archivo = "BloqueMotor.jpeg"
    
    if not os.path.exists(nombre_archivo):
        print(f"Error: No se encontro el archivo '{nombre_archivo}' en la carpeta actual.")
        print("Asegurate de guardar la imagen real con ese nombre exacto.")
        return

    imagen_original = cv2.imread(nombre_archivo)
    
    gris = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2GRAY)
    gris_suavizado = cv2.medianBlur(gris, 5)

    circulos = cv2.HoughCircles(
        gris_suavizado, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=20,   
        param1=120,     
        param2=20,       
        minRadius=40,    
        maxRadius=60      
    )

    imagen_salida = imagen_original.copy()
    contador_aros = 0
    
    if circulos is not None:
        circulos = np.uint16(np.around(circulos))
        for i in circulos[0, :]:
            centro_x, centro_y, r = i[0], i[1], i[2]
            cv2.circle(imagen_salida, (centro_x, centro_y), r, (255, 0, 0), 2)
            cv2.circle(imagen_salida, (centro_x, centro_y), 4, (0, 0, 255), -1)
            contador_aros += 1
            print(f"¡Aro detectado #{contador_aros}! Centro de montaje A: X={centro_x}, Y={centro_y}, Radio={r}")
    else:
        print("No se detectaron circunferencias con los parametros actuales. Prueba ajustando minRadius/maxRadius si es necesario.")

    print(f"Total de aros/circunferencias identificadas: {contador_aros}")

    archivo_salida = "resultado_hough_circunferencias.jpg"
    cv2.imwrite(archivo_salida, imagen_salida)
    print(f"Imagen de salida guardada exitosamente como '{archivo_salida}'.")

if __name__ == "__main__":
    procesar_hough_circunferencias()