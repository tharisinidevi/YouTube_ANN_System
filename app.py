# app.py — FINAL VERSION (MATCHES rank(pct=True) TRAINING LOGIC)

import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from textblob import TextBlob
import re

# ======================
# Load ANN Model + Scaler
# ======================
MODEL_PATH = "model/ann_model.h5"
SCALER_PATH = "model/scaler.pkl"

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ======================
# TRAINING REFERENCE VALUES
# 🔴 MUST MATCH TRAINING DATA
# ======================
MAX_VIEWS = 5_000_000
MAX_LIKES = 120_000
MAX_COMMENTS = 30_000
MAX_AVG_SENTIMENT = 2.0

# ======================
# Sentiment Engine
# ======================
use_vader = False
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except:
        nltk.download("vader_lexicon")
    sia = SentimentIntensityAnalyzer()
    use_vader = True
except:
    use_vader = False


# ======================
# Streamlit Setup
# ======================
st.set_page_config(page_title="YouTube Popularity Predictor", page_icon="🎬", layout="centered")
st.title("🎬 YouTube Popularity Prediction")
st.markdown("---")


# ======================
# User Inputs
# ======================
st.subheader("📊 Enter Video Metrics")

views = st.number_input("Total Views", min_value=0, step=1)
likes = st.number_input("Total Likes", min_value=0, step=1)
comments_count = st.number_input("Total Comments", min_value=0, step=1)

st.markdown("---")
st.subheader("💬 Enter at least TWO comments")

cols = st.columns(2)
comment_inputs = []
for i in range(10):
    with cols[i % 2]:
        comment_inputs.append(st.text_input(f"Comment {i+1}"))


# ======================
# Sentiment Helpers
# ======================
def clean_comment(text):
    text = re.sub(r"http\S+|www\.\S+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def get_raw_sentiment(text):
    if use_vader:
        return sia.polarity_scores(text)["compound"]
    return TextBlob(text).sentiment.polarity


# ======================
# Prediction Logic
# ======================
if st.button("🔮 Predict Popularity"):

    non_empty = [c for c in comment_inputs if c.strip() != ""]
    if len(non_empty) < 2:
        st.error("⚠️ Please enter at least TWO comments.")
        st.stop()

    # ---- Compute avg sentiment ----
    sentiments = [get_raw_sentiment(clean_comment(c)) for c in non_empty]
    avg_sentiment = np.mean(sentiments)

    # ==================================================
    # 🔥 FEATURE RECONSTRUCTION (MATCHES TRAINING)
    # ==================================================

    # Approximate rank(pct=True)
    views_rank = min(views / MAX_VIEWS, 1.0)
    likes_rank = min(likes / MAX_LIKES, 1.0)
    comments_rank = min(comments_count / MAX_COMMENTS, 1.0)

    sentiment_rank = (
        avg_sentiment / MAX_AVG_SENTIMENT
        if MAX_AVG_SENTIMENT != 0 else 0
    )
    sentiment_rank = np.clip(sentiment_rank, 0, 1)

    # ANN INPUT (rank-based)
    X = np.array([[views_rank, likes_rank, comments_rank, sentiment_rank]])
    X_scaled = scaler.transform(X)

    # ---- Predict ----
    pred = model.predict(X_scaled)
    pred_class = np.argmax(pred)
    confidence = np.max(pred)

    labels = {
        0: ("Low Popularity", "📉"),
        1: ("Medium Popularity", "📊"),
        2: ("High Popularity", "🔥")
    }

    result, emoji = labels[pred_class]

    st.success(f"{emoji} **Predicted Popularity: {result}**")
    st.write(f"🤖 Model Confidence: **{confidence:.2%}**")

    # ---- Debug (for FYP validation) ----
    st.markdown("---")
    st.subheader("📌 Model Input Features (Ranks)")

    st.write(f"Views Rank: **{views_rank:.3f}**")
    st.write(f"Likes Rank: **{likes_rank:.3f}**")
    st.write(f"Comments Rank: **{comments_rank:.3f}**")
    st.write(f"Sentiment Rank: **{sentiment_rank:.3f}**")




















