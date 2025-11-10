# train_plate_cnn.py
import os, numpy as np, string
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

DATA_DIR = 'data/images'
MODEL_DIR = 'model'
os.makedirs(MODEL_DIR, exist_ok=True)

CHARS = string.ascii_uppercase + string.digits
NUM_CHARS = 6
NUM_CLASSES = len(CHARS)

def load_data():
    X, Y = [], [[] for _ in range(NUM_CHARS)]
    for f in os.listdir(DATA_DIR):
        if not f.lower().endswith('.png'):
            continue
        # filename format: <idx>_PLATE.png or PLATE.png
        label = f.split('_', 1)[-1].replace('.png','').replace('-','').upper()
        if len(label) != NUM_CHARS:
            continue
        img = load_img(os.path.join(DATA_DIR, f), color_mode='grayscale', target_size=(64,128))
        arr = img_to_array(img)/255.0
        X.append(arr)
        for i, ch in enumerate(label):
            vec = np.zeros(NUM_CLASSES)
            vec[CHARS.index(ch)] = 1
            Y[i].append(vec)
    X = np.array(X)
    Y = [np.array(y) for y in Y]
    return X, Y

print("🔎 Cargando datos...")
X, Y = load_data()
print(f"✅ Se cargaron {len(X)} imágenes correctamente.")
if len(X) == 0:
    raise RuntimeError("No se encontraron imágenes en data/images. Ejecuta generate_plates.py")

# split indices
idx = np.arange(len(X))
train_idx, test_idx = train_test_split(idx, test_size=0.1, random_state=42)
X_train, X_test = X[train_idx], X[test_idx]
Y_train = [y[train_idx] for y in Y]
Y_test  = [y[test_idx] for y in Y]

# model
inp = Input(shape=(64,128,1), name='input_layer')
x = Conv2D(32, (3,3), activation='relu', padding='same')(inp)
x = MaxPooling2D((2,2))(x)
x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2))(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
outputs = [Dense(NUM_CLASSES, activation='softmax', name=f'char_{i}')(x) for i in range(NUM_CHARS)]
model = Model(inputs=inp, outputs=outputs)

model.compile(
    optimizer=Adam(1e-3),
    loss=['categorical_crossentropy'] * NUM_CHARS,
    metrics=['accuracy'] * NUM_CHARS
)
model.summary()

# augmentation generator (we'll use custom generator with dictionaries)
datagen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.02,
    height_shift_range=0.02,
    zoom_range=0.05,
    shear_range=0.02
)

def gen(X_arr, Y_list, batch_size=32):
    N = len(X_arr)
    while True:
        idxs = np.random.permutation(N)
        for i in range(0, N, batch_size):
            batch_idxs = idxs[i:i+batch_size]
            batch_X = np.array([datagen.random_transform(X_arr[k]) for k in batch_idxs])
            batch_Y = {f'char_{j}': Y_list[j][batch_idxs] for j in range(NUM_CHARS)}
            yield batch_X, batch_Y

train_labels = {f'char_{j}': Y_train[j] for j in range(NUM_CHARS)}
test_labels  = {f'char_{j}': Y_test[j] for j in range(NUM_CHARS)}

ckpt = ModelCheckpoint(os.path.join(MODEL_DIR, 'plate_cnn.h5'), save_best_only=True, monitor='val_loss')
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

steps = max(1, len(X_train)//32)
history = model.fit(
    gen(X_train, Y_train, batch_size=32),
    steps_per_epoch=steps,
    validation_data=(X_test, test_labels),
    epochs=30,
    callbacks=[ckpt, es]
)

print("✅ Entrenamiento finalizado. Mejor modelo guardado en model/plate_cnn.h5")
