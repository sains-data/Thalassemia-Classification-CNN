import streamlit as st
from PIL import Image
import numpy as np
import json
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.imagenet_utils import decode_predictions

# Load the model and class mapping outside the ClassifyModel class
model = load_model("model_thalassemia_classifier_80.h5")
with open("class.json") as fin:
    tag2class = json.load(fin)
    class2tag = {v: k for k, v in tag2class.items()}

# Create an instance of ClassifyModel
class ClassifyModel:
    def __init__(self):
        self.model = None
        self.class2tag = None
        self.tag2class = None

    def load(self, model_path="model_thalassemia_classifier_80.h5", class_path="class.json"):
        self.model = load_model(model_path)
        with open(class_path) as fin:
            self.tag2class = json.load(fin)
            self.class2tag = {v: k for k, v in self.tag2class.items()}

    def predict(self, image_array):
        pred = self.model.predict(image_array)
        pred_digits = np.argmax(pred, axis=1)

        return pred_digits, pred

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

uploaded_file = st.file_uploader("Choose a blood image...", type=['jpeg', 'jpg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((80, 80))
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1) 
    img_array = np.repeat(img_array, 3, axis=-1)

    st.image(image, use_column_width=True, caption=f'Uploaded Image: {uploaded_file.name}')

    if st.button('Predict'):
        pred_digits, pred_probabilities = m.predict(img_array)

        if len(pred_digits) > 0 and pred_digits[0] < len(class2tag):
            predicted_label = class2tag[pred_digits[0]]
            confidence = pred_probabilities[0][pred_digits[0]]
            st.write(f"Predicted Label: {predicted_label}")
            st.write(f"Confidence: {round(float(confidence), 4) * 100}%")
        else:
            st.write("Error: Predicted class index out of bounds.")
