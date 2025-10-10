import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from PIL import Image
import numpy as np
import os

st.title("MNIST Digit Classifier")

MODEL_PATH = "mnist_model.h5"

# Function to train and save the model
@st.cache_resource
def get_model():
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        st.info("Loaded existing model.")
    else:
        st.warning("No model found. Training a new MNIST model. This may take a few minutes...")
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(-1,28,28,1)/255.0
        x_test = x_test.reshape(-1,28,28,1)/255.0

        model = models.Sequential([
            layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(64, (3,3), activation='relu'),
            layers.MaxPooling2D((2,2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
        model.save(MODEL_PATH)
        st.success("Model trained and saved successfully!")

    return model

model = get_model()

# File uploader
uploaded = st.file_uploader("Upload a digit image", type=["png", "jpg"])
if uploaded:
    img = Image.open(uploaded).convert("L").resize((28,28))
    st.image(img, caption="Uploaded Image", width=150)

    data = np.array(img)/255.0
    data = np.expand_dims(data, axis=(0,-1))

    pred = np.argmax(model.predict(data))
    st.success(f"Prediction: {pred}")


