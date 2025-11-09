# cnn_char_recognizer.py
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

train_dir = 'data_chars/train'
test_dir = 'data_chars/test'
img_size = (28,28)
batch_size = 64

datagen = ImageDataGenerator(rescale=1./255)
train_gen = datagen.flow_from_directory(train_dir, target_size=img_size,
                                        color_mode='grayscale', class_mode='categorical')
test_gen = datagen.flow_from_directory(test_dir, target_size=img_size,
                                       color_mode='grayscale', class_mode='categorical')

model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_gen.num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

hist = model.fit(train_gen, validation_data=test_gen, epochs=3)

plt.plot(hist.history['loss'], label='train_loss')
plt.plot(hist.history['val_loss'], label='val_loss')
plt.legend(); plt.title('CNN Loss'); plt.savefig('cnn_loss.png'); plt.close()

plt.plot(hist.history['accuracy'], label='train_acc')
plt.plot(hist.history['val_accuracy'], label='val_acc')
plt.legend(); plt.title('CNN Accuracy'); plt.savefig('cnn_acc.png'); plt.close()

model.save('char_cnn.h5')
print("Entrenamiento de CNN completado.")
