from tkinter import filedialog
from tkinter import *
import tensorflow as tf
from PIL import Image, ImageTk
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('mnist1.h5')

# Function to classify the input image
def classify_image():
    # Open the file dialog to select an image
    file_path = filedialog.askopenfilename()
    # Open the image using Pillow
    pil_image = Image.open(file_path).convert('L')
    # Resize the image to 28x28
    pil_image = pil_image.resize((28, 28))
    # Convert the image to a numpy array
    np_image = np.array(pil_image)
    # Reshape the image to a 1D array with 784 elements
    np_image = np_image.reshape(1, 784)
    # Normalize the pixel values
    np_image = np_image / 255.0
    # Use the model to predict the class of the input image
    prediction = model.predict(np_image)
    # Get the predicted class label
    predicted_class = np.argmax(prediction[0])
    # Display the predicted class label in the GUI
    label.config(text=f"Predicted class: {predicted_class}")

# Create the GUI
root = Tk()

# Add a button to select an image
button = Button(root, text="Select Image", command=classify_image)
button.pack()

# Add a label to display the predicted class label
label = Label(root, text="")
label.pack()

# Start the GUI event loop
root.mainloop()
