# app.py
# Flask app que guarda imagen + id (hash) y predicción de placa en CSV.
# Requisitos: flask, pillow, tensorflow (opcional), pytesseract (opcional)
# Si quieres usar pytesseract instala tesseract-ocr en el SO y pip install pytesseract

from flask import Flask, request, render_template_string, redirect, url_for, send_from_directory
from PIL import Image
import numpy as np, datetime, csv, os, io, base64, hashlib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)
MODEL_PATH = 'char_cnn.h5'   # opcional: modelo CNN si lo usas
CSV_PATH = 'plates_db.csv'
UPLOAD_DIR = 'uploads'
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------------------------------------------------
# Helpers CSV / I/O
# -------------------------------------------------------------------
def ensure_csv():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['id','predicted_plate','image_path','state','entry_time','exit_time'])
            writer.writeheader()

def read_csv():
    ensure_csv()
    with open(CSV_PATH, newline='') as f:
        return list(csv.DictReader(f))

def write_csv(rows):
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id','predicted_plate','image_path','state','entry_time','exit_time'])
        writer.writeheader()
        writer.writerows(rows)

# -------------------------------------------------------------------
# Reconocimiento (puedes adaptar: OCR con pytesseract o CNN)
# -------------------------------------------------------------------

# Intenta cargar modelo si existe (sino fallback a None)
_char_model = None
try:
    if os.path.exists(MODEL_PATH):
        _char_model = load_model(MODEL_PATH)
except Exception as e:
    print("No se cargó modelo CNN:", e)
    _char_model = None

def segment_characters(img, n_chars=6):
    """Simple segmentation vertical (demo)."""
    w, h = img.size
    new_h = 28
    new_w = max(28, int(w * (new_h / h)))
    gray = img.convert('L').resize((new_w, new_h))
    arr = np.array(gray)
    ch_w = new_w // n_chars
    chars = []
    for i in range(n_chars):
        crop = arr[:, i*ch_w:(i+1)*ch_w]
        chars.append(Image.fromarray(crop).resize((28,28)))
    return chars

def recognize_plate_cnn(img):
    """Intento de predicción por CNN (si está disponible)."""
    if _char_model is None:
        return ''
    labels = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    plate = ''
    for ch in segment_characters(img):
        a = img_to_array(ch).astype('float32') / 255.0
        if a.ndim == 3 and a.shape[2] == 3:
            a = a.mean(axis=2, keepdims=True)
        a = np.expand_dims(a, 0)
        pred = _char_model.predict(a, verbose=0)
        plate += labels[np.argmax(pred)]
    return plate

# Optional: if pytesseract available, prefer OCR
try:
    import pytesseract, re
    def recognize_plate_ocr(img):
        gray = img.convert('L')
        # psm 7 = treat image as a single text line (ajusta si hace falta)
        text = pytesseract.image_to_string(gray, config='--psm 7').upper()
        text = re.sub(r'[^A-Z0-9\-]','', text)
        # return cleaned text (first plausible match)
        m = re.search(r'[A-Z0-9]{2,3}\-?[A-Z0-9]{2,4}', text)
        return m.group(0) if m else text.strip()
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False
    def recognize_plate_ocr(img):
        return ''

def recognize_plate(img):
    """Primera opción: OCR si disponible, si no CNN, si no fallback string empty."""
    if OCR_AVAILABLE:
        res = recognize_plate_ocr(img)
        if res:
            return res
    # fallback to CNN prediction (if model present)
    if _char_model is not None:
        res = recognize_plate_cnn(img)
        return res
    return ''

# -------------------------------------------------------------------
# ID generation and saving images
# -------------------------------------------------------------------
def save_image_and_get_id(img):
    """
    Guarda la imagen y devuelve un id basado en hash de los bytes.
    Usamos hash SHA256 truncado a 12 chars para un ID legible.
    """
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=85)
    b = buf.getvalue()
    h = hashlib.sha256(b).hexdigest()[:12]
    filename = f"{h}.jpg"
    path = os.path.join(UPLOAD_DIR, filename)
    # Si no existe, guardar (esto hace que la misma imagen reuse el mismo archivo)
    if not os.path.exists(path):
        with open(path, 'wb') as f:
            f.write(b)
    return h, path

# -------------------------------------------------------------------
# Rutas web
# -------------------------------------------------------------------

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)

@app.route('/', methods=['GET'])
def index():
    rows = read_csv()
    # mostramos miniaturas (ruta relativa)
    html = '''
    <h2>🚘 Control Vehicular - Interfaz</h2>
    <p>Nota: Guardamos la imagen y un <strong>ID</strong> asociado (hash) para identificar la lectura.</p>
    <form action="/upload" method="post" enctype="multipart/form-data">
      <input type="file" name="image" accept="image/*" required>
      <button type="submit">Subir imagen</button>
    </form>
    <hr>
    <h3>Registros (CSV)</h3>
    <table border="1" cellpadding="4">
      <tr><th>ID</th><th>Predicción</th><th>Imagen</th><th>Estado</th><th>Entrada</th><th>Salida</th></tr>
      {% for r in rows %}
      <tr>
        <td>{{r['id']}}</td>
        <td>{{r['predicted_plate']}}</td>
        <td>
          {% if r['image_path'] %}
            <a href="{{ url_for('uploaded_file', filename=r['image_path'].split('/')[-1]) }}" target="_blank">
              <img src="{{ url_for('uploaded_file', filename=r['image_path'].split('/')[-1]) }}" width="140">
            </a>
          {% endif %}
        </td>
        <td>{{r['state']}}</td>
        <td>{{r['entry_time']}}</td>
        <td>{{r['exit_time']}}</td>
      </tr>
      {% endfor %}
    </table>
    '''
    return render_template_string(html, rows=rows)

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No se envió imagen", 400
    img = Image.open(request.files['image'].stream).convert('RGB')

    # obtener id y guardar imagen
    img_id, img_path = save_image_and_get_id(img)

    # predecir placa (texto) - puede ser vacío si no hay OCR/model
    predicted = recognize_plate(img)

    # guardar registro si no existe uno abierto para ese id
    now = datetime.datetime.now().isoformat()
    rows = read_csv()
    # determinamos si hay registro sin exit_time para mismo id
    existing_open = None
    for r in rows:
        if r['id'] == img_id and (r.get('exit_time') is None or r['exit_time']==''):
            existing_open = r
            break

    if existing_open:
        # si ya hay, no crear nueva entrada; mostramos opción para cerrar
        known = True
    else:
        known = False
        # añadimos un registro nuevo con entry_time y exit_time vacía
        rows.append({
            'id': img_id,
            'predicted_plate': predicted,
            'image_path': img_path,
            'state': '',
            'entry_time': now,
            'exit_time': ''
        })
        write_csv(rows)

    # preparar imagen base64 para mostrar en la página de confirmación
    buf = io.BytesIO()
    img.thumbnail((800,800))
    img.save(buf, format='JPEG')
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    html = '''
    <h2>🔍 Lectura</h2>
    <p><strong>ID:</strong> {{img_id}}</p>
    <p><strong>Predicción (texto):</strong> {{predicted}}</p>
    <img src="data:image/jpeg;base64,{{img_b64}}" width="420"><br><br>

    {% if known %}
      <form action="/action" method="post">
        <input type="hidden" name="id" value="{{img_id}}">
        <button type="submit" name="action" value="close">Registrar salida (Registro existente)</button>
      </form>
    {% else %}
      <form action="/action" method="post">
        <input type="hidden" name="id" value="{{img_id}}">
        <input type="hidden" name="predicted" value="{{predicted}}">
        <button type="submit" name="action" value="allow">✅ Aceptar acceso (registrar entrada)</button>
        <button type="submit" name="action" value="deny">❌ Denegar acceso (no registrar)</button>
      </form>
    {% endif %}
    <br><a href="/">⬅ Volver</a>
    '''
    return render_template_string(html, img_id=img_id, predicted=predicted, img_b64=img_b64, known=known)

@app.route('/action', methods=['POST'])
def action():
    img_id = request.form.get('id')
    action = request.form.get('action')
    now = datetime.datetime.now().isoformat()
    rows = read_csv()
    msg = ''
    # find row(s) by id
    if action == 'allow':
        # create entry if not exists open
        exists = any(r['id']==img_id and r.get('exit_time','')=='' for r in rows)
        if not exists:
            # create new
            rows.append({
                'id': img_id,
                'predicted_plate': request.form.get('predicted',''),
                'image_path': os.path.join(UPLOAD_DIR, f"{img_id}.jpg"),
                'state': '',
                'entry_time': now,
                'exit_time': ''
            })
            write_csv(rows)
        msg = f"✅ Entrada registrada para ID {img_id}"
    elif action == 'close':
        closed = False
        for r in rows:
            if r['id']==img_id and (r.get('exit_time') is None or r['exit_time']==''):
                r['exit_time'] = now
                closed = True
                break
        if closed:
            write_csv(rows)
            msg = f"✅ Salida registrada para ID {img_id}"
        else:
            msg = f"ℹ️ No se encontró registro abierto para ID {img_id}"
    elif action == 'deny':
        msg = f"❌ Acceso denegado para ID {img_id}"
    else:
        msg = "Acción desconocida"

    return f"<h2>{msg}</h2><a href='/'>⬅ Volver</a>"

# -------------------------------------------------------------------
if __name__ == '__main__':
    print("Arrancando app. Asegúrate de tener instalado pytesseract si usas OCR.")
    app.run(debug=True)
