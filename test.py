import streamlit as st
import tensorflow as tf
import numpy as np

# -------------------------------
# Model Prediction Function
# -------------------------------
def model_prediction(test_image):
    model = tf.keras.models.load_model('waste_classifier.h5')

    expected_size = model.input_shape[1:3]  # (224, 224)

    image = tf.keras.preprocessing.image.load_img(
        test_image,
        target_size=expected_size
    )

    img_array = tf.keras.preprocessing.image.img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    result_index = np.argmax(prediction)

    return result_index


# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("Waste Classification Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Waste Recognition"])

# -------------------------------
# HOME PAGE
# -------------------------------
if app_mode == "Home":
    st.header("Waste Classification System")
    st.markdown("""
    Welcome to the **Waste Classification System**!  
    This tool helps classify waste images into **Organic** or **Recyclable** categories.

    ###  How It Works
    1. Upload a waste image (plastic, paper, food waste, etc.)
    2. The system analyzes it using a trained CNN model
    3. You get instant classification results

    ###  Why Use This App?
    - Helps in automating **waste sorting**
    - Fast & accurate predictions
    - Built with TensorFlow/Keras and deep learning

    Go to the **Waste Recognition** page to get started!
    """)

# -------------------------------
# ABOUT PAGE
# -------------------------------
elif app_mode == "About":
    st.header("About the Project")
    st.markdown("""
    ###  Dataset Information
    This project uses the **Waste Classification Dataset** containing two classes:

    - **Organic Waste**
    - **Recyclable Waste**

    Images include:
    - Food scraps  
    - Leaves  
    - Plastic bottles  
    - Cardboard  
    - Glass  
    - Metals  

    ###  Model Used
    A Convolutional Neural Network (CNN) trained using:
    - Image augmentation  
    - 224×224 resized inputs  
    - Adam optimizer  
    - Softmax final layer  

    ###  Objective
    The goal is to enable easy and automated waste classification for better waste management.
    """)

# -------------------------------
# PREDICTION PAGE
# -------------------------------
elif app_mode == "Waste Recognition":
    st.header("Waste Image Recognition")

    test_image = st.file_uploader("Upload Waste Image:", type=["jpg", "jpeg", "png"])

    if st.button("Show Image"):
        if test_image is not None:
            st.image(test_image, use_column_width=True)
        else:
            st.warning("Please upload an image first.")

    # Predict Button
    if st.button("Predict"):
        if test_image is not None:
            with st.spinner("Analyzing Image..."):
                result_index = model_prediction(test_image)

                # Waste Classes
                class_names = ["Organic Waste", "Recyclable Waste"]

                st.success(f" Prediction: **{class_names[result_index]}**")
        else:
            st.error("Upload an image to classify!")
