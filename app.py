import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from textblob import TextBlob
import plotly.express as px

# ================================
# Load Model and Scaler
# ================================
MODEL_PATH = "model/ann_model.h5"
SCALER_PATH = "model/scaler.pkl"

model = load_model(MODEL_PATH)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# ================================
# Streamlit Page Config
# ================================
st.set_page_config(page_title="🎬 YouTube Popularity Predictor", layout="centered")

# ================================
# CSS Styling
# ================================
def local_css(file_name="style.css"):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.markdown("""
            <style>
            body {
                background-image: linear-gradient(135deg, #F5F7FA 0%, #B8C6DB 100%);
            }
            .main {
                background-color: rgba(255,255,255,0.85);
                padding: 2rem;
                border-radius: 1rem;
            }
            </style>
        """, unsafe_allow_html=True)

local_css()

# ================================
# App Header
# ================================
st.title("🎬 YouTube Video Popularity Predictor")
st.write("Predict how popular your YouTube video will be based on engagement and comment sentiment.")

st.markdown("---")

# ================================
# Input Section
# ================================
st.subheader("📊 Enter Video Metrics")
col1, col2, col3 = st.columns(3)
with col1:
    views = st.number_input("Total Views", min_value=0, step=1, key="views")
with col2:
    likes = st.number_input("Total Likes", min_value=0, step=1, key="likes")
with col3:
    comments_count = st.number_input("Total Comments Count", min_value=0, step=1, key="comments_count")

st.markdown("---")

# ================================
# Comment Section (10 text boxes)
# ================================
st.subheader("💬 Enter Up to 10 Top Comments")
cols = st.columns(2)
comments = []

for i in range(10):
    with cols[i % 2]:
        comment = st.text_input(f"Comment {i+1}", key=f"comment_{i}")
        comments.append(comment)

# ================================
# Helper Functions
# ================================
def get_avg_sentiment(comment_list):
    sentiments = []
    for c in comment_list:
        if c.strip():
            polarity = TextBlob(c).sentiment.polarity
            sentiments.append(polarity)
    return np.mean(sentiments) if sentiments else 0.0

def normalize(value, max_value):
    if max_value == 0:
        return 0
    return min(value / max_value, 1.0)

# ================================
# Buttons
# ================================
colA, colB = st.columns(2)
with colA:
    predict_btn = st.button("🚀 Predict Popularity")
with colB:
    reset_btn = st.button("🔁 Reset")

if reset_btn:
    st.session_state.clear()
    st.rerun()

# ================================
# Prediction
# ================================
if predict_btn:
    avg_sentiment = get_avg_sentiment(comments)

    # Prepare user input
    user_data = np.array([[views, likes, comments_count, avg_sentiment]])
    user_data_scaled = scaler.transform(user_data)

    prediction = model.predict(user_data_scaled)
    popularity_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)

    # Weighted popularity score (same as training logic)
    max_views, max_likes, max_sentiment = 1000000, 50000, 1.0
    views_rank = normalize(views, max_views)
    likes_rank = normalize(likes, max_likes)
    sentiment_rank = normalize(avg_sentiment, max_sentiment)
    popularity_score = (0.5 * views_rank) + (0.3 * likes_rank) + (0.2 * sentiment_rank)

    # Determine popularity label
    if popularity_class == 0:
        result = "Low Popularity"
    elif popularity_class == 1:
        result = "Average Popularity"
    else:
        result = "High Popularity"

    # ================================
    # Display Results
    # ================================
    st.success(f"### 🎯 Predicted Popularity: **{result}**")
    st.write(f"🧠 **Average Sentiment Score:** {avg_sentiment:.2f}")
    st.write(f"📈 **Weighted Popularity Score:** {popularity_score:.2f}")
    st.write(f"🤖 **Model Confidence:** {confidence:.2%}")
    st.write(f"💬 **Comments Analyzed:** {len([c for c in comments if c.strip()])}")

    if avg_sentiment == 0:
        st.warning("⚠️ No comments or invalid text entered — sentiment not factored into the prediction.")

    # ================================
    # Recommendations
    # ================================
    st.subheader("📌 Personalized Recommendations")

    tips = []
    # Views (50%)
    if views_rank < 0.3:
        tips.append("📉 **Low Views (50%)** – Try better SEO, trending tags, and cross-platform promotion.")
    elif views_rank < 0.7:
        tips.append("👀 **Moderate Views** – Improve thumbnails and video titles for more clicks.")
    else:
        tips.append("🔥 **High Views** – Keep your momentum with consistent uploads!")

    # Likes (30%)
    if likes_rank < 0.3:
        tips.append("👍 **Low Likes (30%)** – Encourage engagement and add stronger CTAs.")
    elif likes_rank < 0.7:
        tips.append("💖 **Moderate Likes** – Good engagement; experiment with storytelling.")
    else:
        tips.append("🌟 **High Likes** – Your audience loves it! Keep that tone.")

    # Sentiment (20%)
    if sentiment_rank < 0.3:
        tips.append("😟 **Low Sentiment (20%)** – Address feedback or negative viewer comments.")
    elif sentiment_rank < 0.7:
        tips.append("🙂 **Mixed Sentiment** – Improve pacing or emotional delivery.")
    else:
        tips.append("🥰 **Positive Sentiment** – Great audience reception, keep it up!")

    for t in tips:
        st.write(t)

    # ================================
    # Visualization
    # ================================
    st.markdown("### 📊 Engagement Breakdown")
    chart_data = pd.DataFrame({
        "Metric": ["Views", "Likes", "Comments", "Sentiment"],
        "Value": [views_rank, likes_rank, comments_count, avg_sentiment]
    })
    fig = px.bar(chart_data, x="Metric", y="Value", text="Value",
                 color="Metric", title="Video Performance Overview", height=400)
    st.plotly_chart(fig)

# ================================
# Footer
# ================================
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | YouTube Popularity Predictor powered by ANN + Sentiment Analysis")



















