import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load Model
MODEL_PATH = "model/ann_model.h5"
SCALER_PATH = "model/scaler.pkl"

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

st.title("🎬 YouTube Video Popularity Predictor (Single Video)")

# User input fields
views = st.number_input("Enter Views", min_value=0, step=1)
likes = st.number_input("Enter Likes", min_value=0, step=1)
comments = st.number_input("Enter Comments Count", min_value=0, step=1)
sentiment = st.number_input("Enter Sentiment Score (-1 to 1)", min_value=-1.0, max_value=1.0, step=0.01)

# Predict Button
if st.button("Predict Popularity"):
    user_data = np.array([[views, likes, comments, sentiment]])
    
    # Scale input data
    user_data_scaled = scaler.transform(user_data)

    # Predict
    prediction = model.predict(user_data_scaled)
    popularity_class = np.argmax(prediction, axis=1)[0]  # If multiclass

    if popularity_class == 0:
        result = "Low Popularity"
        tips = [
            "- Use attractive thumbnails & strong titles.",
            "- Increase engagement through polls or questions.",
            "- Improve video tags and description."
        ]
    elif popularity_class == 1:
        result = "Medium Popularity"
        tips = [
            "- Collaborate with similar creators.",
            "- Encourage sharing and audience interaction.",
            "- Post on community tab, IG stories, TikTok, Shorts."
        ]
    else:
        result = "High Popularity"
        tips = [
            "- Maintain content consistency.",
            "- Post follow-up or behind-the-scenes content.",
            "- Optimize monetization & brand deals."
        ]

    st.success(f"✅ Prediction Result: **{result}**")
    st.subheader("📌 Recommendations to Improve:")
    for t in tips:
        st.write(t)




