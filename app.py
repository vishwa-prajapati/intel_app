import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

st.title("Anomaly Detection App")
st.markdown("Upload an image to check for anomalies.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_container_width=True)


    # Preprocess
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # Define class names
    class_names = ["Normal", "Anomaly"]

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.write(f"### Prediction: **{predicted_class}**")


st.markdown("---")
st.header("🎥 Real-Time Anomaly Detection (Bonus)")

if st.button("Start Camera"):
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    class_names = ["Normal", "Anomaly"]  # ✅ Fix: Add this line here

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.resize(frame, (224, 224))
        img = np.expand_dims(img / 255.0, axis=0)

        prediction = model.predict(img)
        predicted_class = class_names[np.argmax(prediction)]

        # Draw label
        label = f"Prediction: {predicted_class}"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show frame in Streamlit
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
    
    cap.release()

