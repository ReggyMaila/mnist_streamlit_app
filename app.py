import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.title("🧠 MNIST Digit Classifier")
st.write("Upload a handwritten digit image to predict its class using a trained CNN model.")

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("digit_model.h5")
    return model

model = load_model()

uploaded = st.file_uploader("📤 Upload a digit image", type=["png", "jpg", "jpeg"])
if uploaded:
    img = Image.open(uploaded).convert("L").resize((28, 28))
    st.image(img, caption="Uploaded Image", use_container_width=True)
    data = np.expand_dims(np.array(img) / 255.0, axis=(0, -1))
    pred = np.argmax(model.predict(data))
    st.success(f"### 🟢 Prediction: {pred}")
