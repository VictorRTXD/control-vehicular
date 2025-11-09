# mlp_demo.py
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

np.random.seed(1)
X = np.random.randn(1000,4)
y = (X.sum(axis=1) + 0.5*np.random.randn(1000) > 0).astype(int)

split = 800
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(4,)),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer=keras.optimizers.Adam(0.01),
              loss='binary_crossentropy',
              metrics=['accuracy'])

hist = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                 epochs=20, batch_size=32, verbose=1)

plt.plot(hist.history['loss'], label='train_loss')
plt.plot(hist.history['val_loss'], label='val_loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('MLP Loss')
plt.savefig('mlp_loss.png'); plt.close()

plt.plot(hist.history['accuracy'], label='train_acc')
plt.plot(hist.history['val_accuracy'], label='val_acc')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.title('MLP Accuracy')
plt.savefig('mlp_acc.png'); plt.close()

model.save('mlp_model.h5')
print("Entrenamiento de MLP completado.")
