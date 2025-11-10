# app.py
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os, hashlib, csv
from datetime import datetime
from recognize import recognize_plate

app = Flask(__name__)
STATIC_UPLOADS = os.path.join('static', 'uploads')
os.makedirs(STATIC_UPLOADS, exist_ok=True)
CSV_FILE = 'vehiculos.csv'

def ensure_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['id','predicted_plate','image_path','state','entry_time','exit_time'])
            writer.writeheader()

def read_csv():
    ensure_csv()
    with open(CSV_FILE, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def write_csv(rows):
    with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id','predicted_plate','image_path','state','entry_time','exit_time'])
        writer.writeheader()
        writer.writerows(rows)

def sha256_id_bytes(b):
    return hashlib.sha256(b).hexdigest()[:12]

@app.route('/')
def index():
    rows = read_csv()
    return render_template('index.html', vehiculos=rows)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No se envió ningún archivo", 400
    file = request.files['file']
    if file.filename == '':
        return "Nombre de archivo vacío", 400
    filename = secure_filename(file.filename)
    save_path = os.path.join(STATIC_UPLOADS, filename)
    file.save(save_path)

    # compute id by hashing bytes
    with open(save_path, 'rb') as f:
        b = f.read()
    img_id = sha256_id_bytes(b)
    img_filename = f"{img_id}.jpg"
    img_public_path = os.path.join(STATIC_UPLOADS, img_filename)
    # if not exists, save canonical jpg
    if not os.path.exists(img_public_path):
        from PIL import Image
        im = Image.open(save_path).convert('RGB')
        im.save(img_public_path, format='JPEG', quality=85)

    # predict plate
    plate = recognize_plate(img_public_path)

    # check CSV for open record with same id (no exit_time)
    rows = read_csv()
    found_open = None
    for r in rows:
        if r['id'] == img_id and r['exit_time'] == '':
            found_open = r
            break

    if found_open:
        known = True
    else:
        known = any(r['predicted_plate'] == plate and r['exit_time'] == '' for r in rows)

    # pass data to result template
    return render_template('resultado.html', placa=plate, image_filename=img_filename, known=known, img_id=img_id)

@app.route('/confirmar', methods=['POST'])
def confirmar():
    action = request.form.get('action')  # 'accept' or 'deny' or 'close'
    img_id = request.form.get('img_id')
    plate = request.form.get('plate')
    image_filename = request.form.get('image_filename')
    now = datetime.now().isoformat()
    rows = read_csv()
    if action == 'accept':
        # add entry if not exists open for this id
        exists_open = any(r['id']==img_id and r['exit_time']=='' for r in rows)
        if not exists_open:
            rows.append({
                'id': img_id,
                'predicted_plate': plate,
                'image_path': os.path.join('static','uploads', image_filename),
                'state': '',
                'entry_time': now,
                'exit_time': ''
            })
            write_csv(rows)
        message = f"✅ Entrada registrada: {plate}"
    elif action == 'close':
        closed = False
        for r in rows:
            if r['id'] == img_id and r['exit_time'] == '':
                r['exit_time'] = now
                closed = True
                break
        if closed:
            write_csv(rows)
            message = f"✅ Salida registrada para {plate}"
        else:
            message = f"ℹ️ No se encontró registro abierto para {plate}"
    else:
        message = f"🚫 Acceso denegado a {plate}"

    return render_template('confirmacion.html', mensaje=message)

# static file serving for uploads (Flask can serve from static/, but keep explicit route if needed)
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(STATIC_UPLOADS, filename)

if __name__ == "__main__":
    ensure_csv()
    app.run(debug=True)
