import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os
import matplotlib.pyplot as plt
import numpy as np

data_dir = "C:\\Users\\kerem\\OneDrive\\Masaüstü\\data"  
img_height, img_width = 128, 128 
batch_size = 32  

datagen = ImageDataGenerator(
    rescale=1.0/255,  
    validation_split=0.2  
)

train_data = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary', 
    subset='training'  
)

val_data = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary', 
    subset='validation' 
)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid') 
])

model.summary()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

epochs = 10  
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs
)

model.save("deepfake_detection_model.h5")
print("Model trained.")

def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")

    plt.show()

plot_training_history(history)

def visualize_prediction(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Test picture can not found: {image_path}")

    image = load_img(image_path, target_size=(img_height, img_width))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array)
    confidence = prediction[0][0]
    result = "Deep Fake" if confidence > 0.5 else "Real"

    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Prediction: {result}\nConfidence: {confidence:.2f}")
    plt.show()

test_image_path = "C:\\Users\\kerem\\OneDrive\\Masaüstü\\test_image.jpg"
try:
    visualize_prediction(test_image_path)
except FileNotFoundError as e:
    print(e)
