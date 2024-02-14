import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import time

def contar_pixeles_verdes(imagen, contorno):
    # Crear una máscara del contorno
    mascara = np.zeros(imagen.shape[:2], dtype="uint8")
    cv2.drawContours(mascara, [contorno], -1, 255, -1)

    # Aplicar la máscara a la imagen original
    region_interior = cv2.bitwise_and(imagen, imagen, mask=mascara)

    # Convertir la región a HSV
    hsv = cv2.cvtColor(region_interior, cv2.COLOR_BGR2HSV)

    # Definir el rango de color verde en HSV
    verde_bajo = np.array([20, 40, 40])
    verde_alto = np.array([40, 255, 255])

    # Crear una máscara para los píxeles verdes
    mascara_verde = cv2.inRange(hsv, verde_bajo, verde_alto)

    # Contar la cantidad de píxeles verdes
    cantidad_pixeles_verdes = cv2.countNonZero(mascara_verde)

    return region_interior, cantidad_pixeles_verdes

def resaltar_contorno_y_contar_verde():
    # Inicializar la captura de video
    captura = cv2.VideoCapture(0)

    while True:
        # Leer el siguiente fotograma del video
        ret, frame = captura.read()

        # Convertir el fotograma a escala de grises
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Aplicar un desenfoque para reducir el ruido
        desenfocada = cv2.GaussianBlur(gris, (5, 5), 0)

        # Detectar bordes utilizando Canny
        bordes = cv2.Canny(desenfocada, 50, 150)

        # Aplicar una operación de cerrado para eliminar las venas internas
        kernel = np.ones((5, 5), np.uint8)
        closing = cv2.morphologyEx(bordes, cv2.MORPH_CLOSE, kernel)

        # Encontrar contornos en la imagen cerrada
        contornos, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Encontrar el contorno más grande (asumiendo que es la hoja)
        contorno_hoja = max(contornos, key=cv2.contourArea)

        # Dibujar un rectángulo alrededor del contorno más grande
        x, y, w, h = cv2.boundingRect(contorno_hoja)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Encontrar el contorno más largo
        contorno_largo = max(contornos, key=cv2.contourArea)

        # Dibujar un rectángulo alrededor del contorno más largo
        x, y, w, h = cv2.boundingRect(contorno_largo)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Contar píxeles verdes dentro del contorno
        region_verde, cantidad_pixeles_verdes = contar_pixeles_verdes(frame, contorno_hoja)

        # Mostrar el fotograma original, el fotograma con contornos resaltados y la región verde
        #cv2.imshow('Fotograma Original', frame)
        #cv2.imshow('Contornos Resaltados', imagen_contornos)
        #cv2.imshow('Región Verde', region_verde)

        model = torch.load('lettuce_npk.pth')
        model.eval()

        roi = frame[y:y+h, x:x+w]

        # Preprocesar la imagen
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_image = Image.fromarray(roi)
        input_tensor = transform(input_image)
        input_batch = input_tensor.unsqueeze(0)

        # Mover el tensor de entrada al dispositivo (GPU si está disponible)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_batch = input_batch.to(device)

        class_names = (["-K","-N","-P","FN"])

        def prediction(img):

            # Realizar la inferencia
            with torch.no_grad():
                pred = model(torch.stack([img]).to(device)).cpu().detach().numpy()[0]
            class_label = np.argmax(pred)
            return class_names[np.argmax(pred)], pred[class_label]

        # Obtener la clase predicha

        pred_class_name, confidence = prediction(input_tensor)

        # Imprimir la clase predicha al cuadrado junto con el objeto identificado en la transmisión en vivo
        cv2.putText(frame, str(pred_class_name + " " + str(confidence) + "%"), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Clase Predicha', frame)

        # Salir del bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar la captura de video y cerrar las ventanas
    captura.release()
    cv2.destroyAllWindows()

    # Imprimir la cantidad de píxeles verdes
    #print(f"Cantidad de píxeles verdes: {cantidad_pixeles_verdes}")

resaltar_contorno_y_contar_verde()
