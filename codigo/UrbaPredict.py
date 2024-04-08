import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np

class_names = sorted(["-K","-N","-P","FN"])

# Use GPU if available
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

# Cargar tu modelo entrenado
model = torch.load('lettuce_npk.pth')
model.eval()

# Función para preprocesar la imagen antes de pasarla por el modelo
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)
    return image

# Función para hacer predicciones con el modelo
def prediction(img):
    t = transforms.Compose([
      transforms.ToTensor(),
      transforms.Resize(224, antialias=True)
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
            processed_image = preprocess_image(image)
            pred_class_name, confidence = prediction(processed_image)
            print("Predicted class:", pred_class_name)
        
        # Si se presiona 's' se guarda la imagen definitivamente
        elif key == ord('s'):
            if os.path.exists(temp_image_path):
                os.rename(temp_image_path, 'captured_image.jpg')  # Renombrar la imagen temporal
                print("Imagen guardada.")
            else:
                print("No hay imagen para guardar.")
        
        # Si se presiona 'd' se borra la imagen temporal
        elif key == ord('d'):
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)  # Borrar la imagen temporal
                print("Imagen borrada.")
            else:
                print("No hay imagen para borrar.")

        # Si se presiona 'q' se sale del bucle
        elif key == ord('q'):
            break

    cap.release()  # Liberar la cámara
    cv2.destroyAllWindows()  # Cerrar la ventana

# Llamar a la función para capturar y procesar la imagen
capture_and_process_image()
