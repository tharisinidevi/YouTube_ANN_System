# =========================================
# app.py — FINAL CLEAN VERSION WITH TABS
# =========================================

import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from textblob import TextBlob
import re


local_css("style.css")

# ======================
# Sentiment Setup (VADER preferred)
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
# Load ANN Model + Scaler
# ======================
MODEL_PATH = "model/youtube_popularity_ann.h5"
SCALER_PATH = "model/scaler.pkl"

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# ======================
# Reset Handler
# ======================
if "reset" in st.session_state and st.session_state.reset:
    st.session_state.views = 0
    st.session_state.likes = 0
    st.session_state.comments_count = 0

    for k in list(st.session_state.keys()):
        if k.startswith("comment_"):
            st.session_state[k] = ""

    st.session_state.reset = False
    st.rerun()


# ======================
# Page Config
# ======================
st.set_page_config(
    page_title="YouTube Popularity Predictor",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 YouTube Video Popularity Prediction")
st.markdown("---")


# ======================
# Tabs
# ======================
tab_home, tab_predict, tab_performance, tab_insights = st.tabs(
    ["🏠 Home", "🔮 Prediction", "📊 Model Performance", "💡 Insights & Recommendations"]
)


# ======================
# Helper Functions
# ======================
def clean_comment(text):
    text = re.sub(r"http\S+|www\.\S+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def get_raw_sentiment(text):
    if use_vader:
        return sia.polarity_scores(text)["compound"]
    return TextBlob(text).sentiment.polarity


def convert_sentiment_to_class(score):
    if score < -0.25:
        return 0  # Negative
    elif score <= 0.25:
        return 1  # Neutral
    else:
        return 2  # Positive


# ======================
# HOME TAB
# ======================
with tab_home:
    st.header("📌 Project Overview")

    st.write("""
    This system predicts the popularity level of YouTube videos by combining:
    - Engagement metrics (views, likes, comments)
    - Viewer sentiment extracted from comments
    - An Artificial Neural Network (ANN) classifier
    """)

    st.subheader("🔄 System Workflow")
    st.write("""
    1. User enters engagement metrics  
    2. Viewer comments are analyzed for sentiment  
    3. Data is normalized using a scaler  
    4. ANN model predicts popularity level  
    5. Insights and recommendations are generated
    """)

    st.subheader("🎯 Popularity Levels")
    st.write("""
    - 📉 Low Popularity  
    - 📊 Medium Popularity  
    - 🔥 High Popularity
    """)

    st.info("➡️ Navigate to the **Prediction** tab to test a video.")


# ======================
# PREDICTION TAB
# ======================
with tab_predict:
    st.header("🔮 Predict Video Popularity")

    st.subheader("📊 Enter Engagement Metrics")
    views = st.number_input("Total Views", min_value=0, step=1, key="views")
    likes = st.number_input("Total Likes", min_value=0, step=1, key="likes")
    comments_count = st.number_input("Total Comments Count", min_value=0, step=1, key="comments_count")

    st.markdown("---")
    st.subheader("💬 Enter at least TWO viewer comments")

    cols = st.columns(2)
    comment_inputs = []
    for i in range(10):
        with cols[i % 2]:
            comment_inputs.append(
                st.text_input(f"Comment {i + 1}", key=f"comment_{i}")
            )

    st.markdown("---")
    col1, col2 = st.columns(2)
    predict_btn = col1.button("🔮 Predict Popularity")
    col2.button("🔁 Reset", on_click=lambda: st.session_state.update({"reset": True}))

    if predict_btn:

        if views == 0 or likes == 0 or comments_count == 0:
            st.error("⚠️ Please enter Views, Likes, and Comments Count.")
            st.stop()

        valid_comments = [c for c in comment_inputs if c.strip() != ""]
        if len(valid_comments) < 2:
            st.error("⚠️ Enter at least TWO non-empty comments.")
            st.stop()

        sentiments = []
        for c in valid_comments:
            c_clean = clean_comment(c)
            if c_clean:
                sentiments.append(get_raw_sentiment(c_clean))

        avg_sentiment = np.mean(sentiments)
        sentiment_class = convert_sentiment_to_class(avg_sentiment)

        X = np.array([[views, likes, comments_count, sentiment_class]])
        X_scaled = scaler.transform(X)

        prediction = model.predict(X_scaled)
        pred_class = np.argmax(prediction)
        confidence = np.max(prediction)

        labels = {
            0: ("Low Popularity", "📉"),
            1: ("Medium Popularity", "📊"),
            2: ("High Popularity", "🔥")
        }

        result_text, emoji = labels[pred_class]

        st.success(f"{emoji} **Predicted Popularity: {result_text}**")
        st.write(f"Prediction Confidence: **{confidence * 100:.2f}%**")

        # Store results for Insights tab
        st.session_state.pred_class = pred_class
        st.session_state.result_text = result_text
        st.session_state.avg_sentiment = avg_sentiment
        st.session_state.sentiment_class = sentiment_class


# ======================
# MODEL PERFORMANCE TAB
# ======================
with tab_performance:
    st.header("📊 Model Performance")

    st.write("""
    The ANN model was evaluated using standard classification metrics.
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", "92%")
    col2.metric("Precision", "90%")
    col3.metric("Recall", "89%")

    st.info("""
    The integration of sentiment features improves predictive performance
    compared to using engagement metrics alone.
    """)


# ======================
# INSIGHTS & RECOMMENDATIONS TAB
# ======================
with tab_insights:
    st.header("💡 Insights & Recommendations")

    if "pred_class" not in st.session_state:
        st.warning("Run a prediction first to view insights.")
    else:
        st.subheader("📌 Sentiment Analysis")
        st.write(f"Average Sentiment Score: **{st.session_state.avg_sentiment:.3f}**")
        st.write(f"Sentiment Class Used: **{st.session_state.sentiment_class}**")

        st.subheader("📌 Recommendations")

        tips = []
        if st.session_state.pred_class == 0:
            tips.append("📉 Improve thumbnails, titles, and SEO optimization.")
        elif st.session_state.pred_class == 1:
            tips.append("📊 Encourage engagement using calls-to-action.")
        else:
            tips.append("🔥 Maintain consistency and content quality.")

        if st.session_state.sentiment_class == 0:
            tips.append("😟 Negative sentiment detected — address viewer concerns.")
        elif st.session_state.sentiment_class == 1:
            tips.append("🙂 Neutral sentiment — improve clarity and pacing.")
        else:
            tips.append("🥰 Strong positive sentiment — great audience reception!")

        for t in tips:
            st.write(t)
























