# Proyecto: Reconocimiento de Placas Vehiculares – Demo

## Objetivos específicos
- Implementar una red neuronal feedforward (MLP) — `mlp_demo.py`.
- Entrenar y evaluar el modelo con datos simulados.
- Analizar los resultados y ajustar hiperparámetros.
- Implementar un prototipo CNN + Flask que reconozca placas vehiculares desde imágenes.

---

## Metodología
1. **Obtención del dataset:**  
   Se generan caracteres sintéticos (0–9, A–Z) con Pillow en `data_chars/`.

2. **Preprocesamiento:**  
   Normalización (valores /255), imágenes 28×28 en escala de grises.

3. **Diseño de la arquitectura:**  
   - MLP: 2 capas ocultas (32, 16 neuronas).
   - CNN: 2 capas convolucionales + MaxPooling, capa densa 128.

4. **Entrenamiento:**
   - MLP: 20 épocas, Adam (lr=0.01).  
   - CNN: 3 épocas (demostrativo, aumentar para mejorar precisión).

5. **Evaluación:**  
   Gráficas de pérdida y precisión: `mlp_loss.png`, `mlp_acc.png`, `cnn_loss.png`, `cnn_acc.png`.

6. **Flujo del sistema (Flask):**  
   - Recibe imagen del coche.  
   - Segmenta la placa en 6 caracteres.  
   - Reconoce cada carácter con la CNN.  
   - Si la placa está en el CSV (`plates_db.csv`), marca salida.  
   - Si no, pregunta (en una versión extendida) si se debe registrar.

---

## Resultados esperados
- Gráficas de entrenamiento.  
- CSV actualizado con:
  - `plate`
  - `state` (opcional)
  - `entry_time`
  - `exit_time`

---

## Instrucciones de ejecución
```bash
# 1. Instalar dependencias
pip install tensorflow pillow flask matplotlib pandas

# 2. Generar dataset
python generate_dataset.py

# 3. Entrenar MLP
python mlp_demo.py

# 4. Entrenar CNN
python cnn_char_recognizer.py

# 5. Ejecutar servidor Flask
python app.py
