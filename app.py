import streamlit as st
import json
import logging
import PIL
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import time

logging.basicConfig(level=logging.DEBUG)

class ClassifyModel:
    def __init__(self):
        self.model = None
        self.class2tag = None
        self.tag2class = None

    def load(self, path="/model"):
        self.model = load_model('model_thalassemia_class.h5', compile=False)
        with open("class.json") as fin:
            self.tag2class = json.load(fin)
            self.class2tag = {v: k for k, v in self.tag2class.items()}
            logging.debug(f"class2tag: {self.class2tag}")

    def predict(self, image_array):
        logging.debug(f"Input shape: {image_array.shape}")
        # Ensure the input has the correct shape (None, 55, 55, 3)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = np.expand_dims(image_array, axis=-1)  # Add a single color channel
        image_array = np.repeat(image_array, 3, axis=-1)  # Repeat the single channel to make it 3 channels

        # Use the loaded model for prediction
        pred = self.model.predict(image_array)
        pred_class = np.argmax(pred, axis=1)[0]
        confidence = pred[0][pred_class]

        return pred_class, confidence

m = ClassifyModel()
m.load()


st.title('Thallasemia Classification')

st.header("Upload a blood image")

st.sidebar.title("About")

st.sidebar.info(
    "An application that utilizes artificial intelligence to accurately classify thalassemia disease through blood image analysis. Using advanced machine learning models, we are committed to providing innovative solutions that aid early diagnosis and effective management of this disease.")

st.sidebar.title("Creator")

st.sidebar.info(
    "Fikri Dwi Alpian - 120450022"
)
st.sidebar.info(
    "Anastasya Nurfitriyani Hidayat - 120450080	"
)
st.sidebar.info(
    "Muhammad Nabil Azizi - 120450090"
)

uploaded_file = st.file_uploader("", type=['jpeg', 'jpg', 'png'])

if uploaded_file is not None:
    # Preprocess the uploaded image
    image = PIL.Image.open(uploaded_file).resize((100, 100))
    img_array = np.array(image)
    img_array = img_array / 255  # Normalize pixel values to the range [0, 1]

    # Display the uploaded image
    st.image(image, use_column_width=True, caption=f'Uploaded Image: {uploaded_file.name}')

    # Button to initiate prediction
    if st.button('Predict'):
        # Record the start time
        start_time = time.time()

        # Use the model for prediction
        predicted_class, confidence = m.predict(img_array)

        # Record the end time
        end_time = time.time()

        # Get the predicted label
        predicted_label = m.class2tag[predicted_class]

        # Display the prediction result
        st.write(f"This is **{predicted_label}** (confidence: **{round(float(confidence), 4) * 100}%**)")
        
        # Display the execution time
        execution_time = end_time - start_time
        st.write(f"Time taken for prediction: {execution_time:.4f} seconds")
