import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os

data_dir = "C:\Users\kerem\OneDrive\Masaüstü\data"  
img_height, img_width = 128, 128  
batch_size = 32

datagen = ImageDataGenerator(
    rescale=1.0/255,           
    validation_split=0.2    
)