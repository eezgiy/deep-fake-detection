import argparse
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Argparse kullanarak komut satırından veri seti yolu alıyoruz
parser = argparse.ArgumentParser(description="Deep Fake Detection using CNN")
parser.add_argument('--dataset', type=str, required=True, help="Path to the dataset")
args = parser.parse_args()

# Veri seti yolunu alıyoruz
train_dir = args.dataset + '/cnn_train'  # Eğitim verisi yolu
test_dir = args.dataset + '/cnn_test'    # Test verisi yolu

print(f"Training data directory: {train_dir}")
print(f"Test data directory: {test_dir}")

# Veriyi hazırlamak için ImageDataGenerator kullanıyoruz
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Eğitim ve test verilerini oluşturuyoruz
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# CNN modelini oluşturuyoruz
model = Sequential()

# İlk Conv2D katmanı
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# İkinci Conv2D katmanı
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Üçüncü Conv2D katmanı
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten ve Dense katmanları
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1, activation='sigmoid'))

# Modeli derliyoruz
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Modeli eğitiyoruz
model.fit(
    train_generator,
    steps_per_epoch=2000 // 32,
    epochs=10,
    validation_data=test_generator,
    validation_steps=800 // 32
)

# Modeli kaydedebiliriz, örneğin:
# model.save("deep_fake_detector.h5")
