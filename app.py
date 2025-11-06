import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from utils.preprocessing import prepare_input

# Load trained model
model = load_model("model/ann_model.h5")

st.title(" YouTube Video Popularity Prediction System")
st.write("This system predicts YouTube video popularity using engagement metrics and sentiment score.")

# User inputs
views = st.number_input("Enter video views:", min_value=0)
likes = st.number_input("Enter video likes:", min_value=0)
comments = st.number_input("Enter number of comments:", min_value=0)
sentiment = st.number_input("Enter average sentiment score (-1 to 2):", min_value=-1.0, max_value=2.0, step=0.1)

# Predict Button
if st.button("Predict Popularity"):
    data = prepare_input(views, likes, comments, sentiment)
    prediction = model.predict(data)

    # If model has 3 classes (Softmax)
    result = np.argmax(prediction)
    categories = ["Not Popular", "Average", "Popular"]

    st.success(f" Predicted Popularity: **{categories[result]}**")
    st.write(" Confidence Scores:")
    st.json({categories[i]: float(prediction[0][i]) for i in range(3)})
