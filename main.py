import streamlit as st 
import tensorflow as tf
import numpy as np

# Tensorflow Model Predn 



def model_prediction(test_image):
    model = tf.keras.models.load_model('best_model.keras')
    image = tf.keras.preprosessing.image.load_image(test_image, target_size=(224,224))
    input_arr = tf.keras.preprosessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) 
    prediction = model.predict(input_arr)
    return prediction

st.sidebar.title("Dashboard")