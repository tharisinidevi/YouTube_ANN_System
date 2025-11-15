# app.py (updated sentiment handling; full file)
import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from textblob import TextBlob
import pandas as pd
import re

# Try to import VADER (nltk). If available, we'll use it (better for social text).
use_vader = False
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    # download lexicon if not present
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except Exception:
        nltk.download("vader_lexicon")
    sia = SentimentIntensityAnalyzer()
    use_vader = True
except Exception:
    use_vader = False

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
# Optional CSS loader
# ======================
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except:
        pass

local_css("style.css")

# ======================
# RESET BUTTON FUNCTION
# ======================
def reset_inputs():
    """Reset all user inputs"""
    for key in list(st.session_state.keys()):
        if key.startswith("comment_"):
            st.session_state[key] = ""
        elif key in ["views", "likes", "comments_count"]:
            st.session_state[key] = 0
    # trigger rerun to clear widgets
    st.rerun()

# ======================
# Input Section
# ======================
st.subheader("📊 Enter Video Metrics")

views = st.number_input("Total Views", min_value=0, step=1, key="views")
likes = st.number_input("Total Likes", min_value=0, step=1, key="likes")
comments_count = st.number_input("Total Comments Count", min_value=0, step=1, key="comments_count")

st.markdown("---")

# ======================
# Comments Section
# ======================
st.subheader("💬 Enter at Least TWO Comments (Required)")

cols = st.columns(2)
comment_inputs = []
for i in range(10):
    with cols[i % 2]:
        comment = st.text_input(f"Comment {i + 1}", "", key=f"comment_{i}")
        comment_inputs.append(comment)

# ======================
# Helper / Sentiment Functions
# ======================
def clean_comment(text):
    # remove URLs, extra whitespace, control chars
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def comment_sentiment_vader(text):
    """Return VADER compound score (-1..1)"""
    return sia.polarity_scores(text)["compound"]

def comment_sentiment_textblob(text):
    """Return TextBlob polarity (-1..1)"""
    return TextBlob(text).sentiment.polarity

def get_sentiment_scores(comments_list):
    """
    Returns:
      scores: list of (clean_text, score) for each non-empty comment
      avg_score: mean score (0.0 if none)
    Preference: VADER if available, else TextBlob.
    """
    scores = []
    for c in comments_list:
        c_clean = clean_comment(c)
        if not c_clean:
            continue
        if use_vader:
            try:
                s = comment_sentiment_vader(c_clean)
            except Exception:
                s = comment_sentiment_textblob(c_clean)
        else:
            s = comment_sentiment_textblob(c_clean)
        scores.append((c_clean, s))
    if scores:
        avg = np.mean([s for (_, s) in scores])
    else:
        avg = 0.0
    return scores, avg

def normalize(value, max_value):
    if max_value == 0:
        return 0
    return min(value / max_value, 1.0)

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

    # VALIDATION
    if views == 0 or likes == 0 or comments_count == 0:
        st.error("⚠️ Please fill in **Views**, **Likes**, and **Total Comments Count** before predicting.")
        st.stop()

    non_empty_comments = [c for c in comment_inputs if c.strip() != ""]
    if len(non_empty_comments) < 2:
        st.error("⚠️ Please enter **at least TWO comments** for sentiment analysis.")
        st.stop()

    # Get sentiment scores (per-comment)
    scores, avg_sentiment = get_sentiment_scores(non_empty_comments)
    num_comments = len(scores)

    # Show per-comment sentiment (debug/insight)
    with st.expander(f"🔎 Show {num_comments} comment sentiment scores"):
        st.write("Using:", "VADER" if use_vader else "TextBlob")
        for idx, (txt, s) in enumerate(scores, start=1):
            st.write(f"**Comment {idx}** (score={s:.3f}): {txt}")

    # Prepare ANN input
    user_data = np.array([[views, likes, comments_count, avg_sentiment]])
    user_data_scaled = scaler.transform(user_data)
    prediction = model.predict(user_data_scaled)
    popularity_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)

    # Weighted score (display-only)
    max_views, max_likes, max_sent = 1_000_000, 50_000, 1.0
    views_rank = normalize(views, max_views)
    likes_rank = normalize(likes, max_likes)
    sentiment_rank = (avg_sentiment + 1) / 2  # convert -1..1 to 0..1 for the weighted score
    popularity_score = (0.5 * views_rank) + (0.3 * likes_rank) + (0.2 * sentiment_rank)

    # Labels
    if popularity_class == 0:
        result, emoji = "Low Popularity", "📉"
    elif popularity_class == 1:
        result, emoji = "Medium Popularity", "📊"
    else:
        result, emoji = "High Popularity", "🔥"

    # DISPLAY
    st.success(f"{emoji} **Predicted Popularity: {result}**")
    st.write(f"🤖 Model confidence: **{confidence:.2%}**")
    st.subheader("📊 Video Performance Overview")
    st.write(f"👀 **Views:** {views:,} (normalized: {views_rank:.2f})")
    st.write(f"👍 **Likes:** {likes:,} (normalized: {likes_rank:.2f})")
    st.write(f"💬 **Total Comments:** {comments_count:,}")
    st.write(f"🧠 **Average Sentiment (raw -1..1):** {avg_sentiment:.3f}")
    st.write(f"📈 **Weighted Popularity Score (0..1):** {popularity_score:.3f}")
    st.write(f"💬 Comments Analyzed (used for sentiment): {num_comments}")

    if num_comments == 0:
        st.warning("⚠️ No valid comments after cleaning — sentiment not factored.")

    # Visualization
    df_plot = pd.DataFrame({
        "Metric": ["Views", "Likes", "Comments"],
        "Value": [views, likes, comments_count]
    })
    st.bar_chart(df_plot.set_index("Metric"))

    # Recommendations (same as before, but using sentiment_rank converted 0..1)
    st.subheader("📌 Personalized Recommendations")
    tips = []
    if views_rank < 0.3:
        tips.append("📉 Low Views — improve SEO, thumbnails, and promotion.")
    elif views_rank < 0.7:
        tips.append("👀 Moderate Views — optimize retention & titles.")
    else:
        tips.append("🔥 High Views — sustain your strategy and experiment.")

    if likes_rank < 0.3:
        tips.append("👍 Low Likes — stronger CTAs & hooks in first 10s.")
    else:
        tips.append("🌟 Likes OK — maintain engagement style.")

    if sentiment_rank < 0.4:
        tips.append("😟 Low Sentiment — address criticism; improve clarity and tone.")
    elif sentiment_rank < 0.7:
        tips.append("🙂 Mixed Sentiment — tweak pacing and clarity.")
    else:
        tips.append("🥰 Positive Sentiment — great reception!")

    for t in tips:
        st.write(t)















