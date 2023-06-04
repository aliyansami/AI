import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow import keras
from tensorflow.keras import layers

# Mount the Google Drive to access files
# from google.colab import drive
# drive.mount('/content/drive')

# Set the path to the folder containing the test images
# path_to_folder = '/content/drive/My Drive/test_images'
path_to_folder = 'test_imag'

# Get a list of all image file names in the folder
file_names = os.listdir(path_to_folder)

# Create a list to store the image data and corresponding labels
images = []
labels = []

# Loop over each file in the folder
for file_name in file_names:
    # Extract the label from the file name (assuming the label is the first character)
    label = int(file_name[0])
    
    # Read the image file and resize it to 28x28 pixels
    img = cv2.imread(os.path.join(path_to_folder, file_name), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    
    # Add the image data and corresponding label to the lists
    images.append(img)
    labels.append(label)

# Convert the image data and labels to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Flatten the image data to 1D arrays
images = images.reshape(-1, 784)

# Normalize the pixel values to be between 0 and 1
scaler = StandardScaler()
images = scaler.fit_transform(images)

# Check the balance of the dataset
df1 = pd.Series(labels).value_counts()

oversampler = RandomOverSampler()
images, labels = oversampler.fit_resample(images, labels)
pd.Series(labels).value_counts()

# Split the dataset into a training set and a testing set
train_images, train_labels, test_images, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)


train_images = train_images.astype("float32") / 255.0
train_labels = train_labels.astype("float32") / 255.0
train_images = train_images.reshape(-1, 28 * 28)
test_labels = train_labels.reshape(-1, 28 * 28)

def create_model(units=128, optimizer='adam'):
    model = keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=(28 * 28,)),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")
])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
# Compile the model

model = create_model()
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
# history = model.fit(train_images, test_images, epochs=10, validation_data=(train_labels, test_labels))
history = model.fit(train_images, test_images, epochs=10, batch_size=32, validation_split=0.2)

model.save('mnist1.h5')
# # Evaluate the model on the testing data
# test_loss, test_acc = model.evaluate(train_images, test_images)
# print('Test accuracy:', test_acc)

# # Make predictions on the testing data
# y_pred = np.argmax(model.predict(train_images),axis=1)

# # Calculate evaluation metrics
# accuracy = accuracy_score(test_images, y_pred)
# precision = precision_score(test_images, y_pred, average='macro')
# recall = recall_score(test_images, y_pred, average='macro')
# f1 = f1_score(test_images, y_pred, average='macro')
# print('Accuracy:', accuracy)
# print('Precision:', precision)
# print('Recall:', recall)
# print('F1 score:', f1)

# clf = KerasClassifier(build_fn=create_model, epochs=10, batch_size=32, verbose=0)

# # Evaluate the model using cross-validation
# scores = cross_val_score(clf, train_images, test_images, cv=5)
# print("Cross-validation scores: ", scores)
# print("Mean score: ", np.mean(scores))

# # define the hyperparameters to search over
# param_grid = {
#     'units': [64, 128, 256],
#     'optimizer': ['adam', 'sgd']
# }

# # perform grid search with 3-fold cross-validation
# grid = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3)
# grid_result = grid.fit(train_images, test_images)

# # print the best parameters and their accuracy
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))