import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from textblob import TextBlob

# ======================
# Load Model and Scaler
# ======================
MODEL_PATH = "model/ann_model.h5"
SCALER_PATH = "model/scaler.pkl"

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ======================
# Page Setup
# ======================
st.set_page_config(page_title="YouTube Popularity Predictor", page_icon="🎬", layout="centered")
st.title("🎬 YouTube Video Popularity Predictor (with Smart Recommendations)")
st.markdown("---")

# ======================
# Optional CSS styling
# ======================
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("⚠️ style.css not found — continuing without custom theme.")

local_css("style.css")

# ======================
# Initialize Session State
# ======================
if "reset_flag" not in st.session_state:
    st.session_state.reset_flag = False

# ======================
# Input Section
# ======================
st.subheader("📊 Enter Video Metrics")

# Using session state for all inputs
views = st.number_input("Total Views", min_value=0, step=1, key="views")
likes = st.number_input("Total Likes", min_value=0, step=1, key="likes")
comments_count = st.number_input("Total Comments Count", min_value=0, step=1, key="comments_count")

st.markdown("---")

# ======================
# Comments Section (10 inputs)
# ======================
st.subheader("💬 Enter Up to 10 Top Comments")
cols = st.columns(2)
comment_inputs = []

for i in range(10):
    key = f"comment_{i}"
    with cols[i % 2]:
        comment = st.text_input(f"Comment {i+1}", key=key)
        comment_inputs.append(comment)

# ======================
# Helper Functions
# ======================
def get_avg_sentiment(comments_list):
    sentiments = []
    for comment in comments_list:
        if comment.strip():
            polarity = TextBlob(comment).sentiment.polarity
            sentiments.append(polarity)
    if sentiments:
        return np.mean(sentiments), len(sentiments)
    return 0.0, 0

def normalize(value, max_value):
    if max_value == 0:
        return 0
    return min(value / max_value, 1.0)

def reset_inputs():
    """Reset all user inputs"""
    for key in list(st.session_state.keys()):
        if key.startswith("comment_") or key in ["views", "likes", "comments_count"]:
            st.session_state[key] = ""
    st.session_state.reset_flag = True
    st.experimental_rerun()

# ======================
# Prediction Section
# ======================
st.markdown("---")

col1, col2 = st.columns([1, 1])
with col1:
    predict_btn = st.button("🔮 Predict Popularity")
with col2:
    reset_btn = st.button("🔁 Reset", on_click=reset_inputs)

if predict_btn:
    avg_sentiment, num_comments = get_avg_sentiment(comment_inputs)

    # Prepare data
    user_data = np.array([[views, likes, comments_count, avg_sentiment]])
    user_data_scaled = scaler.transform(user_data)

    prediction = model.predict(user_data_scaled)
    popularity_class = np.argmax(prediction, axis=1)[0]

    # Weighted popularity score (consistent with your ANN formula)
    max_views = 1000000
    max_likes = 50000
    max_sentiment = 1.0

    views_rank = normalize(views, max_views)
    likes_rank = normalize(likes, max_likes)
    sentiment_rank = normalize(avg_sentiment, max_sentiment)
    popularity_score = (0.5 * views_rank) + (0.3 * likes_rank) + (0.2 * sentiment_rank)

    # Popularity label
    if popularity_class == 0:
        result = "Low Popularity"
        emoji = "📉"
    elif popularity_class == 1:
        result = "Medium Popularity"
        emoji = "📊"
    else:
        result = "High Popularity"
        emoji = "🔥"

    # ======================
    # Display Results
    # ======================
    st.success(f"{emoji} Predicted Popularity: **{result}**")
    st.write(f"🧠 Average Sentiment Score: **{avg_sentiment:.2f}**")
    st.write(f"📈 Weighted Popularity Score: **{popularity_score:.2f}**")
    st.write(f"💬 Comments Analyzed: **{num_comments}**")

    if num_comments == 0:
        st.warning("⚠️ No comments entered — sentiment not factored into prediction.")

    # ======================
    # Recommendations
    # ======================
    st.subheader("📌 Personalized Recommendations")
    tips = []

    # Views (50%)
    if views_rank < 0.3:
        tips.append("📉 **Low Views (50%)** – Improve SEO, collaborate with creators, and cross-promote.")
    elif views_rank < 0.7:
        tips.append("👀 **Moderate Views** – Optimize titles, thumbnails, and tags.")
    else:
        tips.append("🔥 **High Views** – Maintain growth; explore related niches.")

    # Likes (30%)
    if likes_rank < 0.3:
        tips.append("👍 **Low Likes (30%)** – Use CTAs and more engaging intros.")
    elif likes_rank < 0.7:
        tips.append("💖 **Moderate Likes** – Try emotional storytelling or humor.")
    else:
        tips.append("🌟 **High Likes** – Excellent engagement! Keep audience tone.")

    # Sentiment (20%)
    if sentiment_rank < 0.3:
        tips.append("😟 **Low Sentiment (20%)** – Address feedback; maintain clarity.")
    elif sentiment_rank < 0.7:
        tips.append("🙂 **Mixed Sentiment** – Adjust pacing or message delivery.")
    else:
        tips.append("🥰 **Positive Sentiment** – Great reception! Keep your style.")

    # Comments-to-views ratio
    if comments_count < (0.01 * views):
        tips.append("💬 **Low Comment Ratio** – Ask engaging questions to increase discussion.")
    else:
        tips.append("💭 **Good Engagement** – Keep interacting via replies and polls.")

    for t in tips:
        st.write(t)

    st.info("💡 ANN model weighting: Views (50%) • Likes (30%) • Sentiment (20%)")


















