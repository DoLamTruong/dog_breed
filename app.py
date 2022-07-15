import os
import numpy as np
import streamlit as st
from PIL import Image


from model_dog import *

st.markdown('<style>body{background-color: Blue;}</style>',unsafe_allow_html=True)
st.header('Data Scientist Capstone')
st.write("Dog Breed Classifier")

with st.spinner(text='Loading model, please wait a few second!'):
    model = resnet_model
st.success('Model loaded!')

image_file = st.file_uploader("Choose any image and get corresponding breed prediction", type=["jpg", "png", "jpeg"])
if image_file is not None:
    with st.spinner(text='Loading'):
        img = Image.open(image_file)
        st.text("Your image:")
        st.image(img.resize((256, 256)))
        img = img.convert('RGB')
        if os.path.exists("./images/input_img.jpg"):
            os.remove("./images/input_img.jpg")
        img.save("./images/input_img.jpg")

        class_id, breed = breed_algorithm("./images/input_img.jpg")
        if class_id == 0:
            st.text(str("This picture has at least a human, breed is " + breed))
        elif class_id == 1:
            st.text(str("This picture has at least a dog, its breed is " + breed))
        else:
            st.text("It wasn't possible to identify a human or dog in the image!")

