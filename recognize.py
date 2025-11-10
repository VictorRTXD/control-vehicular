# recognize.py
import os, numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageOps, ImageEnhance

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
MODEL_PATH = os.path.join('model', 'plate_cnn.h5')

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}. Entrena primero.")

print("✅ Cargando modelo desde", MODEL_PATH)
model = load_model(MODEL_PATH)
print("✅ Modelo cargado.")

def preprocess_plate(path):
    img = Image.open(path).convert('L')
    w,h = img.size
    # recorta márgenes pequeños
    img = img.crop((int(0.02*w), int(0.05*h), int(0.98*w), int(0.95*h)))
    img = ImageOps.autocontrast(img)
    img = ImageEnhance.Contrast(img).enhance(1.6)
    # resize to (width,height) = (128,64) then ensure shape (64,128,1)
    img = img.resize((128,64))
    arr = img_to_array(img) / 255.0
    # ensure shape ordering (height,width,channels)
    if arr.shape[0] == 128 and arr.shape[1] == 64:
        arr = np.transpose(arr, (1,0,2))
    arr = arr.reshape((64,128,1))
    return np.expand_dims(arr, axis=0)

def recognize_plate(path, debug=False):
    x = preprocess_plate(path)
    preds = model.predict(x)
    # preds is list of 6 arrays
    if isinstance(preds, list):
        text = ''.join(CHARS[np.argmax(p)] for p in preds)
        if debug:
            for i,p in enumerate(preds):
                top = p[0].argsort()[-5:][::-1]
                print(f"char{i} top5:", [(CHARS[ii], float(p[0][ii])) for ii in top])
    else:
        # fallback
        arr = np.squeeze(preds)
        text = ''.join(CHARS[np.argmax(arr[i])] for i in range(arr.shape[0]))
    return text
