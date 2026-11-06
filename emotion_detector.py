"""Real-time facial emotion detector.

Pipeline: webcam frame -> SSD face detector (OpenCV DNN) -> 48x48 grayscale crop
-> Keras CNN (FER-2013 classes) -> on-screen overlay.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
PROTOTXT_PATH = BASE_DIR / "face_detector" / "deploy.prototxt"
WEIGHTS_PATH = BASE_DIR / "face_detector" / "res10_300x300_ssd_iter_140000.caffemodel"
MODEL_PATH = BASE_DIR / "modelFEC.h5"

# Order must match the training labels of modelFEC.h5 (FER-2013).
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

FRAME_WIDTH = 640
FACE_SIZE = 48
SSD_INPUT_SIZE = (300, 300)
SSD_MEAN = (104.0, 177.0, 123.0)


def parse_args():
    parser = argparse.ArgumentParser(description="Detector de emociones faciales en tiempo real.")
    parser.add_argument("--camera", type=int, default=0, help="Indice de la camara. Default: 0")
    parser.add_argument("--confidence", type=float, default=0.4, help="Umbral minimo de deteccion de rostro. Default: 0.4")
    return parser.parse_args()


def load_models():
    for path in (PROTOTXT_PATH, WEIGHTS_PATH, MODEL_PATH):
        if not path.exists():
            sys.exit(f"No se encontro el archivo requerido: {path}")

    # Imported lazily so `--help` works without TensorFlow's slow startup.
    from tensorflow.keras.models import load_model

    face_net = cv2.dnn.readNet(str(PROTOTXT_PATH), str(WEIGHTS_PATH))
    emotion_model = load_model(str(MODEL_PATH), compile=False)
    return face_net, emotion_model


def resize_to_width(frame, width):
    h, w = frame.shape[:2]
    if w == width:
        return frame
    return cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)


def detect_faces(frame, face_net, min_confidence):
    """Returns a list of (x1, y1, x2, y2) boxes clamped to the frame bounds."""
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, SSD_INPUT_SIZE, SSD_MEAN)
    face_net.setInput(blob)
    detections = face_net.forward()

    boxes = []
    for i in range(detections.shape[2]):
        if detections[0, 0, i, 2] < min_confidence:
            continue
        x1, y1, x2, y2 = (detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        # Skip degenerate boxes (face mostly outside the frame).
        if x2 - x1 < 10 or y2 - y1 < 10:
            continue
        boxes.append((x1, y1, x2, y2))
    return boxes


def predict_emotions(frame, boxes, emotion_model):
    """Classifies all faces in a single batch. Returns one probability vector per box."""
    if not boxes:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = np.stack([
        cv2.resize(gray[y1:y2, x1:x2], (FACE_SIZE, FACE_SIZE)).astype("float32")
        for (x1, y1, x2, y2) in boxes
    ])[..., np.newaxis]
    # Raw 0-255 pixels, same as the original pipeline the model was used with.
    return emotion_model.predict(faces, verbose=0)


def draw_face(frame, box, emotion, confidence):
    x1, y1, x2, y2 = box
    label = f"{emotion}: {confidence * 100:.0f}%"
    label_top = max(0, y1 - 40)
    cv2.rectangle(frame, (x1, label_top), (x2, y1), (255, 0, 0), -1)
    cv2.putText(frame, label, (x1 + 5, label_top + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)


def main():
    args = parse_args()
    face_net, emotion_model = load_models()

    capture_backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cam = cv2.VideoCapture(args.camera, capture_backend)
    if not cam.isOpened():
        sys.exit(f"No se pudo abrir la camara {args.camera}.")

    prev_time = time.time()
    fps = 0.0

    print("Presiona 'q' en la ventana de video para salir.")
    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                print("No se pudo leer un frame de la camara; saliendo.")
                break
            frame = resize_to_width(frame, FRAME_WIDTH)

            boxes = detect_faces(frame, face_net, args.confidence)
            preds = predict_emotions(frame, boxes, emotion_model)

            for box, pred in zip(boxes, preds):
                idx = int(np.argmax(pred))
                draw_face(frame, box, CLASSES[idx], float(pred[idx]))

            now = time.time()
            if now > prev_time:
                # Exponential smoothing keeps the FPS readout from jittering.
                fps = 0.9 * fps + 0.1 * (1.0 / (now - prev_time)) if fps else 1.0 / (now - prev_time)
            prev_time = now

            cv2.putText(frame, f"{int(fps)} FPS", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)
            cv2.imshow("Detector de emociones", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
