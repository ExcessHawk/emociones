# Detector de emociones

Detecta la emoción facial en tiempo real desde la webcam.

```
Webcam → detector de rostros SSD (OpenCV DNN) → recorte 48x48 en gris
       → CNN Keras (modelFEC.h5, clases FER-2013) → overlay en pantalla
```

Emociones: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`.

## Requisitos

- **Python 3.9 – 3.11** (TensorFlow < 2.16 no soporta versiones más nuevas, y el modelo es formato Keras 2).
- Webcam.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

## Uso

```bash
python emotion_detector.py
```

Presiona **q** en la ventana de video para salir.

| Opción | Default | Descripción |
|---|---|---|
| `--camera` | `0` | Índice de la cámara. |
| `--confidence` | `0.4` | Umbral mínimo para aceptar un rostro. |

## Archivos

| Archivo | Descripción |
|---|---|
| `emotion_detector.py` | Programa principal. |
| `modelFEC.h5` | CNN de clasificación de emociones (entrada 48x48x1). |
| `face_detector/` | Detector de rostros ResNet-10 SSD de OpenCV (`haarcascade_frontalface2.xml` no se usa). |
| `images/validation/` | Muestras del dataset FER-2013. |
