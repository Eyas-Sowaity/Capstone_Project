import streamlit as st
import os
import random
import numpy as np
import tensorflow as tf
from PIL import Image

# Path to the directory containing the celebrity images
IMAGE_DIR = "C:\\Users\\kingo\\OneDrive\\Documents\\Python Scripts\\Streamlit Dev\\Data_99_Classes\\train"

# List of classes (directory names)
CLASSES = os.listdir(IMAGE_DIR)


# Load the pre-trained face recognition model

MODEL_PATH = "VGG16.h5"

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model
with st.spinner('Model is being loaded..'):
    model=load_model()



# Function to preprocess the input image
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))  # Assuming the model input size is 224x224
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Function to get the name of the celebrity
def get_celebrity_name(image_path):
    # Preprocess the image
    img = preprocess_image(image_path)
    # Predict using the loaded model
    predictions = model.predict(img)
    # Get the predicted class index
    predicted_index = np.argmax(predictions)
    # Get the celebrity name from the class index
    celebrity_name = CLASSES[predicted_index]
    return celebrity_name

# Function to get a random image
def get_random_image():
    # Choose a random class
    random_class = random.choice(CLASSES)
    # Get list of images in the class directory
    images = os.listdir(os.path.join(IMAGE_DIR, random_class))
    # Choose a random image from the class
    random_image = random.choice(images)
    # Return the path to the random image
    return os.path.join(IMAGE_DIR, random_class, random_image)

def main():
    st.title("Celebrity Face Recognition")
    st.write("Click below to recognize a celebrity")

    # Get a random image
    random_image_path = get_random_image()

    # Display the image
    st.image(random_image_path, caption='Random Celebrity Image', use_column_width=True)

    if st.button("Recognize Celebrity"):
        # Get the name of the celebrity
        celebrity_name = get_celebrity_name(random_image_path)
        st.write(f"Predicted Celebrity:{celebrity_name}")

if __name__ == "__main__":
    main()
