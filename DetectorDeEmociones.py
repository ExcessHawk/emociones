# Importación de librerias
import serial
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import cv2
import time

# Variables para calcular FPS
time_actualframe = 0
time_prevframe = 0

# Tipos de emociones
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Modelo de detección de rostros
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Carga de la red neuronal entrenada
emotionModel = load_model("modelFEC.h5")

# Inicializar conexión con ESP32
ser = serial.Serial('COM3', 115200)  # Para Windows
time.sleep(2)

# Se crea la captura de video
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Toma la imagen, los modelos de detección de rostros y mascarillas 
# Retorna las localizaciones de los rostros y las predicciones de emociones de cada rostro
def predict_emotion(frame, faceNet, emotionModel):
    # Construye un blob de la imagen
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

    # Realiza las detecciones de rostros a partir de la imagen
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # Listas para guardar rostros, ubicaciones y predicciones
    faces = []
    locs = []
    preds = []

    # Recorre cada una de las detecciones
    for i in range(0, detections.shape[2]):
        # Fija un umbral para determinar que la detección es confiable
        # Tomando la probabilidad asociada en la detección
        if detections[0, 0, i, 2] > 0.4:
            # Toma el bounding box de la detección escalado
            # de acuerdo a las dimensiones de la imagen
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (Xi, Yi, Xf, Yf) = box.astype("int")

            # Valida las dimensiones del bounding box
            if Xi < 0: Xi = 0
            if Yi < 0: Yi = 0

            # Se extrae el rostro y se convierte BGR a GRAY
            # Finalmente se escala a 48x48
            face = frame[Yi:Yf, Xi:Xf]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (48, 48))
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)

            # Se agrega los rostros y las localizaciones a las listas
            faces.append(face)
            locs.append((Xi, Yi, Xf, Yf))

            pred = emotionModel.predict(face)
            preds.append(pred[0])

    return (locs, preds)

try:
    while True:
        # Se toma un frame de la cámara y se redimensiona
        ret, frame = cam.read()
        frame = imutils.resize(frame, width=640)

        (locs, preds) = predict_emotion(frame, faceNet, emotionModel)

        # Para cada hallazgo se dibuja en la imagen el bounding box y la clase
        for (box, pred) in zip(locs, preds):
            (Xi, Yi, Xf, Yf) = box
            emotion = classes[np.argmax(pred)]
            label = "{}: {:.0f}%".format(emotion, max(pred) * 100)

            # Enviar el resultado al ESP32
            ser.write((emotion + '\n').encode('utf-8'))
            print(f"Enviando: {emotion}")

            cv2.rectangle(frame, (Xi, Yi-40), (Xf, Yi), (255, 0, 0), -1)
            cv2.putText(frame, label, (Xi+5, Yi-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.rectangle(frame, (Xi, Yi), (Xf, Yf), (255, 0, 0), 3)

        time_actualframe = time.time()

        if time_actualframe > time_prevframe:
            fps = 1 / (time_actualframe - time_prevframe)

        time_prevframe = time_actualframe

        cv2.putText(frame, str(int(fps)) + " FPS", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass
finally:
    # Cerrar recursos
    cv2.destroyAllWindows()
    cam.release()
    ser.close()
