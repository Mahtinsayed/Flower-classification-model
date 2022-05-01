import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
from tensorflow.keras.models import load_model

#model = tf.keras.models.load_model("TensorFlow-Streamlit-main/saved_model/my_model2.hdf5")
model = load_model('saved_model/my_model2.hdf5')
### load file
uploaded_file = st.file_uploader("Choose a image file", type="jpg")

map_dict = {0: 'daisy',
            1: 'dandelion',
            2: 'roses',
            3: 'sunflowers',
            4: 'tulips'}


if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(180,180))
    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="RGB")

    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis,...]

    Genrate_pred = st.button("Generate Prediction")    
    if Genrate_pred:
        prediction = model.predict(img_reshape).argmax()
        st.title("Predicted Label for the image is {}".format(map_dict [prediction]))