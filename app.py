import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from textblob import TextBlob  # for sentiment analysis

# Load Model and Scaler
MODEL_PATH = "model/ann_model.h5"
SCALER_PATH = "model/scaler.pkl"

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Streamlit UI
st.title("🎬 YouTube Video Popularity Predictor (with Comment Sentiment)")

st.subheader("📊 Enter Video Details")
views = st.number_input("Total Views", min_value=0, step=1)
likes = st.number_input("Total Likes", min_value=0, step=1)
comments_count = st.number_input("Total Comments Count", min_value=0, step=1)

st.subheader("💬 Paste Top 10 Comments")
user_comments = st.text_area("Enter the top 10 comments (each line = one comment)")

def get_avg_sentiment(comments_text):
    comments = comments_text.split("\n")
    sentiments = []
    for comment in comments:
        if comment.strip():
            polarity = TextBlob(comment).sentiment.polarity
            sentiments.append(polarity)
    if sentiments:
        return np.mean(sentiments)
    return 0.0  # if no comments provided

if st.button("🔮 Predict Popularity"):
    avg_sentiment = get_avg_sentiment(user_comments)

    # Prepare data
    user_data = np.array([[views, likes, comments_count, avg_sentiment]])
    user_data_scaled = scaler.transform(user_data)
    
    # Predict
    prediction = model.predict(user_data_scaled)
    popularity_class = np.argmax(prediction, axis=1)[0]

    # Map results
    if popularity_class == 0:
        result = "Low Popularity"
        tips = [
            "- Improve video SEO (titles, tags, description).",
            "- Ask viewers for feedback and engagement.",
            "- Post shorter clips or highlight reels."
        ]
    elif popularity_class == 1:
        result = "Medium Popularity"
        tips = [
            "- Increase audience interaction through polls.",
            "- Collaborate with creators.",
            "- Focus on thumbnails and storytelling."
        ]
    else:
        result = "High Popularity"
        tips = [
            "- Keep consistency in upload schedule.",
            "- Leverage your strong audience base.",
            "- Experiment with new formats (shorts, reels)."
        ]

    # Display output
    st.success(f"✅ Predicted Popularity: **{result}**")
    st.write(f"🧠 Average Sentiment Score: {avg_sentiment:.2f}")
    st.subheader("📌 Recommendations:")
    for t in tips:
        st.write(t)




