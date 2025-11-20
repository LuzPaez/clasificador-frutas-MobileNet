import numpy as np
import tensorflow as tf
from PIL import Image
import pathlib

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANNELS = 3

# Cargar modelo
MODEL_PATH = pathlib.Path(__file__).resolve().parent / "modelo_frutas_cnn.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Lista de clases originales del modelo
CLASSES = [
    "fresh_peaches_done",
    "fresh_pomegranates_done",
    "fresh_strawberries_done",
    "rotten_peaches_done",
    "rotten_pomegranates_done",
    "rotten_strawberries_done"
]

TRANSLATIONS = {
    "fresh_peaches_done": "Durazno fresco",
    "fresh_pomegranates_done": "Granada fresca",
    "fresh_strawberries_done": "Fresa fresca",
    "rotten_peaches_done": "Durazno podrido",
    "rotten_pomegranates_done": "Granada podrida",
    "rotten_strawberries_done": "Fresa podrida"
}

def predict_image(image_path):
    """Recibe ruta de imagen y retorna clase traducida + confianza."""
    
    # Cargar imagen
    image = Image.open(image_path).convert("RGB")
    image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))

    img_array = np.array(image).reshape(1, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS) / 255.0

    # Predicción
    pred = model.predict(img_array)
    pred_idx = np.argmax(pred)
    confidence = float(pred[0][pred_idx])

    # Obtener nombre original y traducido
    class_raw = CLASSES[pred_idx]
    class_translated = TRANSLATIONS.get(class_raw, class_raw)

    return class_translated, confidence
