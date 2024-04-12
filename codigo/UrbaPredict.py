import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import time

i=1

class_names = sorted(["-K","-N","-P","FN"])

# Use GPU if available
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

# Cargar tu modelo entrenado
model = torch.load('lettuce_npk.pth')
model.eval()

def preprocess_and_predict_image(img):
    if not isinstance(img, Image.Image):  # Comprobar si img es una instancia de PIL Image
        img = Image.fromarray(img)  # Convertir img a PIL Image si es una matriz NumPy
        
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Preprocesar la imagen
    preprocessed_img = transform(img).unsqueeze(0)
    
    # Realizar la predicción
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True)
    ])
    new_img = t(img)
    model.eval()
    with torch.no_grad():
        pred = model(torch.stack([new_img]).to(device)).cpu().detach().numpy()[0]
    class_label = np.argmax(pred)
    return class_names[np.argmax(pred)], pred[class_label]

# Función para capturar y procesar la imagen
def capture_and_process_image():
    cap = cv2.VideoCapture(0)  # Abrir la cámara
    temp_image_path = 'temp_image.jpg'  # Ruta para almacenar temporalmente la imagen

    while True:
        ret, frame = cap.read()  # Capturar un frame de la cámara

        cv2.imshow('Camera', frame)  # Mostrar el frame

        key = cv2.waitKey(1)
        
        # Si se presiona 'c' se captura la imagen temporalmente
        if key == ord('c'):
            cv2.imwrite(temp_image_path, frame)  # Guardar la imagen temporalmente
            image = Image.open(temp_image_path)
            cv2.imshow('Captured Image', np.array(image))  # Mostrar la imagen capturada
            pred_class_name, confidence = preprocess_and_predict_image(image)
            print("Predicted class:", pred_class_name)
        
        # Si se presiona 's' se guarda la imagen definitivamente
        elif key == ord('s'):
            global i
            if os.path.exists(temp_image_path):
                new_image_path = 'captured_image_' + str(i) + '.jpg'
                if not os.path.exists(new_image_path):
                    os.rename(temp_image_path, new_image_path)  # Renombrar la imagen temporal
                    print("Imagen guardada como:", new_image_path)
                    i += 1
                else:
                    print("La imagen ya existe. Presione 's' para sobrescribir o 'n' para no sobrescribir.")
                    overwrite = ''
                    while overwrite.lower() not in ['s', 'n']:
                        key = cv2.waitKey(0)
                        if key == ord('s'):
                            overwrite = 's'
                        elif key == ord('n'):
                            overwrite = 'n'
                        else:
                            print("Opción inválida. Presione 's' para sobrescribir o 'n' para no sobrescribir.")
                    if overwrite.lower() == 's':
                        os.remove(new_image_path)  # Borrar la imagen existente
                        os.rename(temp_image_path, new_image_path)  # Renombrar la imagen temporal
                        print("Imagen sobrescrita como:", new_image_path)
                        i += 1

                    else:
                        os.remove(temp_image_path)  # Borrar la imagen temporal
                        print("Imagen descartada.")
                    i += 1
                cv2.destroyAllWindows()  # Cerrar la ventana
            else:
                print("No hay imagen para guardar.")
        
        # Si se presiona 'd' se borra la imagen temporal
        elif key == ord('d'):
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)  # Borrar la imagen temporal
                print("Imagen borrada.")
                cv2.destroyAllWindows()  # Cerrar la ventana
            else:
                print("No hay imagen para borrar.")

        # Si se presiona 'q' se sale del bucle
        elif key == ord('q'):
            print("Saliendo...")
            time.sleep(2)
            break

    cap.release()  # Liberar la cámara
    cv2.destroyAllWindows()  # Cerrar la ventana

# Llamar a la función para capturar y procesar la imagen
capture_and_process_image()
