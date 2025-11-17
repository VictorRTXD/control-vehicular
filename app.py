from flask import Flask, request, jsonify, send_file, render_template_string, Response
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from inference import CarLicensePlateDetector
import os
import tempfile
import cv2
import csv
from datetime import datetime
from typing import Tuple, Union
import threading
import time
import base64
import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

# ==================== VARIABLES GLOBALES ====================
detector = CarLicensePlateDetector('models/best.pt')
camera = None
camera_active = False
camera_index = 0  # 0 = cámara por defecto, puede ser iRiunWebcam

# Diccionarios en memoria
known_vehicles = {}  # {plate: {'added_date': datetime, 'name': str}}
access_log = []  # [{plate, entry_time, exit_time, status}]
vehicle_status = {}  # {plate: 'inside' o 'outside'}
pending_validations = []  # [{plate, timestamp, frame_base64}]

# Control de detección
last_detections_by_plate = {}  # {plate: timestamp}
detection_cooldown = 5  # segundos entre detecciones de la misma placa

# CSV para persistencia (opcional)
CSV_VEHICLES = 'known_vehicles.csv'
CSV_LOG = 'access_log.csv'

# ==================== PERSISTENCIA CSV ====================
def load_data_from_csv():
    """Carga datos desde CSV al iniciar"""
    global known_vehicles, access_log
    
    # Cargar vehículos conocidos
    if os.path.exists(CSV_VEHICLES):
        try:
            with open(CSV_VEHICLES, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    known_vehicles[row['plate']] = {
                        'added_date': row['added_date'],
                        'name': row.get('name', 'N/A')
                    }
            print(f"✅ Cargados {len(known_vehicles)} vehículos conocidos")
        except Exception as e:
            print(f"⚠️ Error cargando vehículos: {e}")
    
    # Cargar log de accesos
    if os.path.exists(CSV_LOG):
        try:
            with open(CSV_LOG, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                access_log = list(reader)
            print(f"✅ Cargados {len(access_log)} registros de acceso")
        except Exception as e:
            print(f"⚠️ Error cargando log: {e}")

def save_vehicles_to_csv():
    """Guarda vehículos conocidos a CSV"""
    try:
        with open(CSV_VEHICLES, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['plate', 'added_date', 'name'])
            writer.writeheader()
            for plate, data in known_vehicles.items():
                writer.writerow({
                    'plate': plate,
                    'added_date': data['added_date'],
                    'name': data['name']
                })
    except Exception as e:
        print(f"⚠️ Error guardando vehículos: {e}")

def save_log_to_csv():
    """Guarda log de accesos a CSV"""
    try:
        with open(CSV_LOG, 'w', newline='', encoding='utf-8') as f:
            if access_log:
                writer = csv.DictWriter(f, fieldnames=access_log[0].keys())
                writer.writeheader()
                writer.writerows(access_log)
    except Exception as e:
        print(f"⚠️ Error guardando log: {e}")

# Cargar datos al iniciar
load_data_from_csv()

# ==================== FUNCIONES DE CÁMARA ====================
def get_available_cameras():
    """Detecta cámaras disponibles (incluye iRiunWebcam)"""
    available = []
    for i in range(10):  # Buscar hasta 10 cámaras
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available

def start_camera(index=0):
    """Inicia la cámara"""
    global camera, camera_active
    try:
        camera = cv2.VideoCapture(index)
        if camera.isOpened():
            camera_active = True
            print(f"✅ Cámara {index} iniciada correctamente")
            return True
        else:
            print(f"❌ No se pudo abrir la cámara {index}")
            return False
    except Exception as e:
        print(f"❌ Error iniciando cámara: {e}")
        return False

def stop_camera():
    """Detiene la cámara"""
    global camera, camera_active
    if camera:
        camera_active = False
        camera.release()
        camera = None
        print("⏹️ Cámara detenida")

def generate_frames():
    """Generador de frames para streaming de video"""
    global last_detections_by_plate, pending_validations
    
    last_displayed_plate = None
    last_displayed_time = None
    
    while camera_active and camera:
        success, frame = camera.read()
        if not success:
            break
        
        # Crear copia para procesamiento
        display_frame = frame.copy()
        current_time = datetime.now()
        
        # Guardar frame temporal
        temp_path = os.path.join(tempfile.gettempdir(), 'temp_frame.jpg')
        cv2.imwrite(temp_path, frame)
        
        # Detectar placa
        try:
            info, processed = detector.recognize_license_plate(temp_path)
            detected_plate = info.get('License')
            
            if detected_plate:
                # Verificar si ya fue detectada recientemente
                last_detection_time = last_detections_by_plate.get(detected_plate)
                
                if last_detection_time:
                    time_since_last = (current_time - datetime.fromisoformat(last_detection_time)).total_seconds()
                else:
                    time_since_last = 999  # Primera vez
                
                # Solo procesar si pasó el cooldown
                if time_since_last > detection_cooldown:
                    last_detections_by_plate[detected_plate] = current_time.isoformat()
                    
                    # Verificar si es vehículo conocido
                    if detected_plate in known_vehicles:
                        # Registrar acceso automáticamente (entrada o salida)
                        register_access(detected_plate)
                        print(f"✅ Vehículo conocido: {detected_plate}")
                    else:
                        # Verificar que no esté ya en validaciones pendientes
                        already_pending = any(v['plate'] == detected_plate for v in pending_validations)
                        
                        if not already_pending:
                            # Convertir frame a base64 para almacenar
                            _, buffer = cv2.imencode('.jpg', frame)
                            frame_base64 = base64.b64encode(buffer).decode('utf-8')
                            
                            # Agregar a validaciones pendientes
                            pending_validations.append({
                                'plate': detected_plate,
                                'timestamp': current_time.isoformat(),
                                'frame': frame_base64,
                                'datetime_obj': current_time
                            })
                            print(f"⚠️ Vehículo desconocido detectado: {detected_plate}")
                        else:
                            print(f"⏳ Vehículo {detected_plate} ya está pendiente de validación")
                
                # Actualizar para mostrar en video
                last_displayed_plate = detected_plate
                last_displayed_time = current_time
            
            # Usar el frame procesado con las anotaciones
            display_frame = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            print(f"Error en detección: {e}")
        
        # Agregar información en el frame
        cv2.putText(display_frame, f"Hora: {current_time.strftime('%H:%M:%S')}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if last_displayed_plate:
            status_text = f"Ultima placa: {last_displayed_plate}"
            is_known = last_displayed_plate in known_vehicles
            known_status = " [CONOCIDO]" if is_known else " [DESCONOCIDO]"
            color = (0, 255, 0) if is_known else (0, 165, 255)
            
            cv2.putText(display_frame, status_text + known_status, 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Mostrar estado (dentro/fuera)
            if is_known:
                vehicle_state = vehicle_status.get(last_displayed_plate, 'outside')
                state_text = f"Estado: {'DENTRO' if vehicle_state == 'inside' else 'FUERA'}"
                cv2.putText(display_frame, state_text, 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Codificar frame para streaming
        ret, buffer = cv2.imencode('.jpg', display_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.1)  # Reducir carga del CPU

def register_access(plate):
    """Registra entrada o salida de un vehículo"""
    global vehicle_status, access_log
    
    current_time = datetime.now().isoformat()
    current_status = vehicle_status.get(plate, 'outside')
    
    if current_status == 'outside':
        # Registrar ENTRADA
        access_log.append({
            'plate': plate,
            'entry_time': current_time,
            'exit_time': None,
            'status': 'inside'
        })
        vehicle_status[plate] = 'inside'
        print(f"✅ ENTRADA registrada: {plate}")
    else:
        # Registrar SALIDA (actualizar último registro)
        for log in reversed(access_log):
            if log['plate'] == plate and log['exit_time'] is None:
                log['exit_time'] = current_time
                log['status'] = 'completed'
                break
        vehicle_status[plate] = 'outside'
        print(f"✅ SALIDA registrada: {plate}")
    
    save_log_to_csv()

# ==================== PÁGINA PRINCIPAL ====================
@app.route('/')
def index():
    """Dashboard principal del sistema"""
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sistema de Control Vehicular</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: #1a1a2e;
                color: #eee;
            }
            
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                text-align: center;
                box-shadow: 0 2px 10px rgba(0,0,0,0.3);
            }
            
            .header h1 {
                margin: 0;
                font-size: 2em;
            }
            
            .container {
                display: grid;
                grid-template-columns: 2fr 1fr;
                gap: 20px;
                padding: 20px;
                max-width: 1800px;
                margin: 0 auto;
            }
            
            .panel {
                background: #16213e;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            }
            
            .panel h2 {
                margin-bottom: 15px;
                color: #667eea;
                border-bottom: 2px solid #667eea;
                padding-bottom: 10px;
            }
            
            #videoFeed {
                width: 100%;
                border-radius: 10px;
                background: #000;
                min-height: 400px;
            }
            
            .controls {
                display: flex;
                gap: 10px;
                margin-top: 15px;
                flex-wrap: wrap;
            }
            
            button {
                padding: 12px 24px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-size: 14px;
                font-weight: bold;
                transition: all 0.3s ease;
                flex: 1;
                min-width: 150px;
            }
            
            .btn-primary {
                background: #667eea;
                color: white;
            }
            
            .btn-primary:hover {
                background: #5568d3;
                transform: translateY(-2px);
            }
            
            .btn-danger {
                background: #e74c3c;
                color: white;
            }
            
            .btn-danger:hover {
                background: #c0392b;
            }
            
            .btn-success {
                background: #27ae60;
                color: white;
            }
            
            .btn-success:hover {
                background: #229954;
            }
            
            .btn-warning {
                background: #f39c12;
                color: white;
            }
            
            .btn-warning:hover {
                background: #e67e22;
            }
            
            button:disabled {
                background: #555;
                cursor: not-allowed;
                opacity: 0.5;
            }
            
            .status-badge {
                display: inline-block;
                padding: 5px 12px;
                border-radius: 20px;
                font-size: 0.85em;
                font-weight: bold;
            }
            
            .status-inside {
                background: #27ae60;
                color: white;
            }
            
            .status-outside {
                background: #95a5a6;
                color: white;
            }
            
            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
            }
            
            th, td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #34495e;
            }
            
            th {
                background: #0f3460;
                color: #667eea;
                font-weight: bold;
            }
            
            tr:hover {
                background: #1a2940;
            }
            
            .validation-card {
                background: #2c3e50;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 15px;
                border-left: 4px solid #f39c12;
            }
            
            .validation-card img {
                width: 100%;
                border-radius: 5px;
                margin: 10px 0;
            }
            
            .validation-plate {
                font-size: 1.5em;
                font-weight: bold;
                color: #f39c12;
                margin: 10px 0;
            }
            
            .validation-actions {
                display: flex;
                gap: 10px;
                margin-top: 10px;
            }
            
            .stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }
            
            .stat-card {
                background: #0f3460;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
            }
            
            .stat-number {
                font-size: 2.5em;
                font-weight: bold;
                color: #667eea;
            }
            
            .stat-label {
                color: #95a5a6;
                margin-top: 5px;
            }
            
            .empty-state {
                text-align: center;
                padding: 40px;
                color: #95a5a6;
            }
            
            .camera-select {
                padding: 10px;
                border-radius: 5px;
                background: #0f3460;
                color: white;
                border: 1px solid #667eea;
                margin-right: 10px;
            }
            
            .scroll-table {
                max-height: 400px;
                overflow-y: auto;
            }
            
            ::-webkit-scrollbar {
                width: 8px;
            }
            
            ::-webkit-scrollbar-track {
                background: #16213e;
            }
            
            ::-webkit-scrollbar-thumb {
                background: #667eea;
                border-radius: 4px;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🚗 Sistema de Control Vehicular</h1>
            <p>Vigilancia y registro de accesos en tiempo real</p>
        </div>
        
        <div class="container">
            <!-- Panel izquierdo: Video y controles -->
            <div>
                <div class="panel">
                    <h2>📹 Vigilancia en Tiempo Real</h2>
                    <img id="videoFeed" src="/video_feed" alt="Esperando cámara..." 
                         onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iODAwIiBoZWlnaHQ9IjYwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iODAwIiBoZWlnaHQ9IjYwMCIgZmlsbD0iIzFhMWEyZSIvPjx0ZXh0IHg9IjUwJSIgeT0iNTAlIiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtc2l6ZT0iMjQiIGZpbGw9IiM2NjdlZWEiIHRleHQtYW5jaG9yPSJtaWRkbGUiPkPDoW1hcmEgbm8gZGlzcG9uaWJsZTwvdGV4dD48L3N2Zz4='">
                    
                    <div class="controls">
                        <select id="cameraSelect" class="camera-select"></select>
                        <button class="btn-primary" onclick="startCamera()">▶️ Iniciar Cámara</button>
                        <button class="btn-danger" onclick="stopCamera()">⏹️ Detener</button>
                        <button class="btn-warning" onclick="refreshCameras()">🔄 Detectar Cámaras</button>
                    </div>
                </div>
                
                <div class="panel" style="margin-top: 20px;">
                    <h2>📊 Estadísticas</h2>
                    <div class="stats" id="stats">
                        <div class="stat-card">
                            <div class="stat-number" id="totalVehicles">0</div>
                            <div class="stat-label">Vehículos Conocidos</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number" id="vehiclesInside">0</div>
                            <div class="stat-label">Dentro</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number" id="totalAccess">0</div>
                            <div class="stat-label">Accesos Totales</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number" id="pendingValidations">0</div>
                            <div class="stat-label">Pendientes</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Panel derecho: Validaciones y registros -->
            <div>
                <div class="panel">
                    <h2>⚠️ Validaciones Pendientes</h2>
                    <div id="pendingValidations" class="scroll-table">
                        <div class="empty-state">
                            <p>No hay vehículos pendientes de validación</p>
                        </div>
                    </div>
                </div>
                
                <div class="panel" style="margin-top: 20px;">
                    <h2>📋 Registro de Accesos</h2>
                    <div class="scroll-table">
                        <table id="accessLog">
                            <thead>
                                <tr>
                                    <th>Placa</th>
                                    <th>Entrada</th>
                                    <th>Salida</th>
                                    <th>Estado</th>
                                </tr>
                            </thead>
                            <tbody id="accessLogBody">
                                <tr>
                                    <td colspan="4" style="text-align: center; color: #95a5a6;">
                                        Sin registros
                                    </td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let updateInterval;
            
            // Cargar cámaras disponibles al inicio
            refreshCameras();
            
            // Actualizar datos cada 2 segundos
            updateInterval = setInterval(updateData, 2000);
            
            async function refreshCameras() {
                try {
                    const response = await fetch('/api/cameras');
                    const cameras = await response.json();
                    const select = document.getElementById('cameraSelect');
                    select.innerHTML = '';
                    
                    if (cameras.length === 0) {
                        select.innerHTML = '<option>No se encontraron cámaras</option>';
                    } else {
                        cameras.forEach(cam => {
                            const option = document.createElement('option');
                            option.value = cam;
                            option.textContent = `Cámara ${cam}`;
                            select.appendChild(option);
                        });
                    }
                } catch (error) {
                    console.error('Error cargando cámaras:', error);
                }
            }
            
            async function startCamera() {
                const cameraIndex = document.getElementById('cameraSelect').value;
                try {
                    const response = await fetch('/api/start_camera', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({camera_index: parseInt(cameraIndex)})
                    });
                    const result = await response.json();
                    alert(result.message);
                    if (result.status === 'success') {
                        document.getElementById('videoFeed').src = '/video_feed?' + new Date().getTime();
                    }
                } catch (error) {
                    alert('Error iniciando cámara: ' + error);
                }
            }
            
            async function stopCamera() {
                try {
                    const response = await fetch('/api/stop_camera', {method: 'POST'});
                    const result = await response.json();
                    alert(result.message);
                } catch (error) {
                    alert('Error deteniendo cámara: ' + error);
                }
            }
            
            async function updateData() {
                try {
                    // Actualizar estadísticas
                    const statsResponse = await fetch('/api/stats');
                    const stats = await statsResponse.json();
                    
                    document.getElementById('totalVehicles').textContent = stats.total_vehicles;
                    document.getElementById('vehiclesInside').textContent = stats.vehicles_inside;
                    document.getElementById('totalAccess').textContent = stats.total_access;
                    document.getElementById('pendingValidations').textContent = stats.pending_validations;
                    
                    // Actualizar validaciones pendientes
                    const validationsResponse = await fetch('/api/pending_validations');
                    const validations = await validationsResponse.json();
                    updatePendingValidations(validations);
                    
                    // Actualizar log de accesos
                    const logResponse = await fetch('/api/access_log');
                    const log = await logResponse.json();
                    updateAccessLog(log);
                    
                } catch (error) {
                    console.error('Error actualizando datos:', error);
                }
            }
            
            function updatePendingValidations(validations) {
                const container = document.getElementById('pendingValidations');
                
                if (validations.length === 0) {
                    container.innerHTML = '<div class="empty-state"><p>No hay vehículos pendientes de validación</p></div>';
                    return;
                }
                
                container.innerHTML = validations.map((v, index) => `
                    <div class="validation-card">
                        <div class="validation-plate">🚗 ${v.plate}</div>
                        <img src="data:image/jpeg;base64,${v.frame}" alt="Frame">
                        <div style="font-size: 0.9em; color: #95a5a6;">
                            Detectado: ${new Date(v.timestamp).toLocaleString('es-MX')}
                        </div>
                        <div class="validation-actions">
                            <button class="btn-success" onclick="validateVehicle(${index}, true)" style="flex: 1;">
                                ✅ Reconocer
                            </button>
                            <button class="btn-danger" onclick="validateVehicle(${index}, false)" style="flex: 1;">
                                ❌ Denegar
                            </button>
                        </div>
                    </div>
                `).join('');
            }
            
            function updateAccessLog(log) {
                const tbody = document.getElementById('accessLogBody');
                
                if (log.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="4" style="text-align: center; color: #95a5a6;">Sin registros</td></tr>';
                    return;
                }
                
                tbody.innerHTML = log.slice().reverse().slice(0, 10).map(entry => `
                    <tr>
                        <td><strong>${entry.plate}</strong></td>
                        <td>${entry.entry_time ? new Date(entry.entry_time).toLocaleTimeString('es-MX') : '-'}</td>
                        <td>${entry.exit_time ? new Date(entry.exit_time).toLocaleTimeString('es-MX') : '-'}</td>
                        <td>
                            <span class="status-badge ${entry.status === 'inside' ? 'status-inside' : 'status-outside'}">
                                ${entry.status === 'inside' ? 'Dentro' : 'Completado'}
                            </span>
                        </td>
                    </tr>
                `).join('');
            }
            
            async function validateVehicle(index, approve) {
                try {
                    const response = await fetch('/api/validate_vehicle', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({index: index, approve: approve})
                    });
                    const result = await response.json();
                    
                    if (approve) {
                        alert(`✅ Vehículo ${result.plate} registrado como conocido`);
                    } else {
                        alert(`❌ Acceso denegado para ${result.plate}`);
                    }
                    
                    updateData();
                } catch (error) {
                    alert('Error: ' + error);
                }
            }
        </script>
    </body>
    </html>
    ''')

# ==================== API ENDPOINTS ====================

@app.route('/video_feed')
def video_feed():
    """Stream de video en tiempo real"""
    if not camera_active:
        return "Cámara no activa", 404
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/cameras')
def api_cameras():
    """Lista cámaras disponibles"""
    cameras = get_available_cameras()
    return jsonify(cameras)

@app.route('/api/start_camera', methods=['POST'])
def api_start_camera():
    """Inicia la cámara"""
    data = request.json
    camera_idx = data.get('camera_index', 0)
    
    if start_camera(camera_idx):
        return jsonify({'status': 'success', 'message': f'Cámara {camera_idx} iniciada'})
    else:
        return jsonify({'status': 'error', 'message': 'No se pudo iniciar la cámara'}), 400

@app.route('/api/stop_camera', methods=['POST'])
def api_stop_camera():
    """Detiene la cámara"""
    stop_camera()
    return jsonify({'status': 'success', 'message': 'Cámara detenida'})

@app.route('/api/stats')
def api_stats():
    """Estadísticas del sistema"""
    vehicles_inside = sum(1 for status in vehicle_status.values() if status == 'inside')
    
    return jsonify({
        'total_vehicles': len(known_vehicles),
        'vehicles_inside': vehicles_inside,
        'total_access': len(access_log),
        'pending_validations': len(pending_validations)
    })

@app.route('/api/pending_validations')
def api_pending_validations():
    """Lista de vehículos pendientes de validación"""
    return jsonify(pending_validations)

@app.route('/api/access_log')
def api_access_log():
    """Log de accesos"""
    return jsonify(access_log)

@app.route('/api/validate_vehicle', methods=['POST'])
def api_validate_vehicle():
    """Validar o denegar un vehículo desconocido"""
    global pending_validations
    
    data = request.json
    index = data.get('index')
    approve = data.get('approve', False)
    
    if index < 0 or index >= len(pending_validations):
        return jsonify({'error': 'Índice inválido'}), 400
    
    validation = pending_validations.pop(index)
    plate = validation['plate']
    
    if approve:
        # Agregar a vehículos conocidos
        known_vehicles[plate] = {
            'added_date': datetime.now().isoformat(),
            'name': 'N/A'
        }
        save_vehicles_to_csv()
        
        # Limpiar otras validaciones pendientes de la misma placa (por si hay duplicados)
        pending_validations = [v for v in pending_validations if v['plate'] != plate]
        
        # Registrar acceso (entrada)
        register_access(plate)
        
        print(f"✅ Vehículo {plate} aprobado y registrado")
        return jsonify({'status': 'approved', 'plate': plate, 'message': 'Vehículo registrado y acceso concedido'})
    else:
        print(f"❌ Acceso denegado para {plate}")
        return jsonify({'status': 'denied', 'plate': plate, 'message': 'Acceso denegado'})

# ==================== SUBIDA DE ARCHIVOS (ORIGINAL) ====================

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e: RequestEntityTooLarge) -> Tuple[str, int]:
    """Handle errors for files that exceed the maximum size limit."""
    return jsonify({"error": "File is too large. Maximum file size is 100MB."}), 413

@app.errorhandler(500)
def handle_internal_error(e: Exception) -> Tuple[str, int]:
    """Handle internal server errors."""
    return jsonify({"error": "Internal Server Error"}), 500

@app.route('/upload', methods=['POST'])
def upload_file() -> Union[str, Tuple[str, int]]:
    """Handle file upload requests."""
    file = request.files.get('file')
    if not file or file.filename == '':
        return jsonify({"error": "No file provided or file name is empty"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(tempfile.gettempdir(), filename)
    file.save(file_path)

    try:
        return process_file(filename, file_path)
    except Exception as e:
        app.logger.error(f"Error processing file: {str(e)}")
        return jsonify({"error": str(e)}), 500

def process_file(filename: str, file_path: str) -> Union[str, Tuple[str, int]]:
    """Process the uploaded file based on its format."""
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return process_image(file_path, filename)
    elif filename.lower().endswith(('.mp4', '.mov', '.avi')):
        return process_video(file_path, filename)
    else:
        return jsonify({"error": "Unsupported file format"}), 400

def process_image(file_path: str, filename: str) -> str:
    """Process an image file for license plate detection."""
    info, processed_image = detector.recognize_license_plate(file_path)
    output_path = os.path.join(tempfile.gettempdir(), 'processed_' + filename)
    cv2.imwrite(output_path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
    return send_file(output_path, as_attachment=True, download_name='processed_' + filename)

def process_video(file_path: str, filename: str) -> str:
    """Process a video file for license plate detection in each frame."""
    video_output_path = os.path.join(tempfile.gettempdir(), 'processed_' + filename)
    detector.process_video(file_path, video_output_path)
    return send_file(video_output_path, as_attachment=True, download_name='processed_' + filename)

# ==================== INICIAR SERVIDOR ====================
if __name__ == '__main__':
    print("="*60)
    print("🚀 SISTEMA DE CONTROL VEHICULAR")
    print("="*60)
    print("📍 Abre tu navegador en: http://127.0.0.1:5000")
    print("")
    print("📋 Funcionalidades:")
    print("   ✅ Detección de placas en tiempo real")
    print("   ✅ Registro automático de entradas/salidas")
    print("   ✅ Validación de vehículos desconocidos")
    print("   ✅ Compatible con cámara web e iRiunWebcam")
    print("")
    print("💾 Archivos de persistencia:")
    print(f"   - {CSV_VEHICLES} (vehículos conocidos)")
    print(f"   - {CSV_LOG} (registro de accesos)")
    print("")
    print("⚠️  Presiona CTRL+C para detener el servidor")
    print("="*60)
    app.run(debug=True, host='127.0.0.1', port=5000, threaded=True)