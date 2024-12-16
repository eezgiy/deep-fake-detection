import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

# 1. Veri Setinin Hazırlanması (örneğin, ImageDataGenerator ile)
def prepare_data(train_dir, test_dir, image_size=(224, 224)):
    train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, zoom_range=0.2)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(train_dir, target_size=image_size, batch_size=32, class_mode='binary')
    test_generator = test_datagen.flow_from_directory(test_dir, target_size=image_size, batch_size=32, class_mode='binary')

    return train_generator, test_generator

# 2. CNN Modelinin Tanımlanması
def build_model(input_shape=(224, 224, 3)):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # İki sınıf için sigmoid çıkışı
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    return model

# 3. Modelin Eğitilmesi
def train_model(model, train_generator, validation_generator, epochs=20):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator, callbacks=[early_stopping])

    return model, history

# 4. Modelin Değerlendirilmesi
def evaluate_model(model, test_generator):
    y_pred = (model.predict(test_generator) > 0.5).astype('int32')
    y_true = test_generator.classes  # Gerçek etiketler

    # Precision, Recall, F1 Score hesaplama
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    # Accuracy ve F1 Score'ı manuel olarak hesaplama
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

# 5. Ana Fonksiyon
if __name__ == '__main__':
    train_dir = 'cnn_datasets\cnn_test'   # Eğitim verisi yolu
    test_dir = 'cnn_datasets\cnn_train'    # Test verisi yolu

    # Veri hazırlığı
    train_generator, test_generator = prepare_data(train_dir, test_dir)

    # Modeli oluşturma
    model = build_model()

    # Modeli eğitme
    model, history = train_model(model, train_generator, test_generator)

    # Modeli değerlendirme
    evaluate_model(model, test_generator)

    # Modeli kaydetme
    model.save('deepfake_detection_model.h5')
