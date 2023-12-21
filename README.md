# Deployment Machine Learning Model

This project is an implementation of **Convolutional Neural Network (CNN)** in the context of red blood cell classification to distinguish thalassemia variations.

It will be deployed to **Streamlit** to facilitate the use of the model to make it more interactive.
![image](https://github.com/sains-data/Thalassemia-Classification-CNN/assets/128737322/d85fb7f3-85a0-4e57-9572-7003740dd7d7)

### *Requirement*
**numpy, Pillow, scikit-learn, streamlit, tensorflow**

## Run Online from Streamlit
- Open link

```
https://thallasemia-project.streamlit.app/
```

- Upload image blood from "data uji" folder

- Click "Predict" button

- See the result
## Run Locally

Clone the project

```bash
  git clone -b Deploy https://github.com/sains-data/Thalassemia-Classification-CNN.git
```

Go to the project directory

```bash
  cd Thalassemia-Classification-CNN
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Start the server

```bash
  streamlit run app.py
```
