# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers

# # Load MNIST dataset
# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# # Preprocess data
# x_train = x_train.astype("float32") / 255.0
# x_test = x_test.astype("float32") / 255.0
# x_train = x_train.reshape(-1, 28 * 28)
# x_test = x_test.reshape(-1, 28 * 28)

# # Define model architecture
# model = keras.Sequential([
#     layers.Dense(128, activation="relu", input_shape=(28 * 28,)),
#     layers.Dropout(0.5),
#     layers.Dense(10, activation="softmax")
# ])

# # Compile model
# model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# # Train model
# model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# # Save model
# model.save("my_model.h5")


import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import save_img

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess data
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Define function to create image directory and save images
def create_image_directory(images, labels, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for image, label in zip(images, labels):
        filename = os.path.join(directory, f"{label}_{np.random.randint(10000)}.png")
        save_img(filename, image)

# Create directory for test images
test_dir = "test_imag"
create_image_directory(x_test, y_test, test_dir)

