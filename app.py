# app.py — FINAL VERSION (MATCHES YOUR ANN MODEL EXACTLY)

import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from textblob import TextBlob
import pandas as pd
import re

# Try VADER sentiment
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
MODEL_PATH = "model/ann_model.h5"
SCALER_PATH = "model/scaler.pkl"

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ======================
# Streamlit Setup
# ======================
st.set_page_config(page_title="YouTube Popularity Predictor", page_icon="🎬", layout="centered")
st.title("🎬 YouTube Popularity Predictor (FYP ANN Version)")
st.markdown("---")

def reset_inputs():
    for key in list(st.session_state.keys()):
        st.session_state[key] = ""
    st.rerun()

# ======================
# User Inputs
# ======================
st.subheader("📊 Enter Video Metrics")

views = st.number_input("Total Views", min_value=0, step=1)
likes = st.number_input("Total Likes", min_value=0, step=1)
comments_count = st.number_input("Total Comments Count", min_value=0, step=1)

st.markdown("---")
st.subheader("💬 Enter at least TWO comments")

cols = st.columns(2)
comment_inputs = []
for i in range(10):
    with cols[i % 2]:
        comment_inputs.append(
            st.text_input(f"Comment {i + 1}", key=f"comment_{i}")
        )

# ======================
# Sentiment Helpers
# ======================
def clean_comment(text):
    text = re.sub(r"http\S+|www\.\S+", "", text)
    return re.sub(r"\s+", " ", text).strip()

def get_raw_sentiment(text):
    """Return sentiment score (-1 to 1) using VADER or TextBlob."""
    if use_vader:
        return sia.polarity_scores(text)["compound"]
    return TextBlob(text).sentiment.polarity

def convert_sentiment_to_class(s):
    """Convert -1..1 sentiment to model classes 0,1,2."""
    if s < -0.25:
        return 0        # negative
    elif s <= 0.25:
        return 1        # neutral
    else:
        return 2        # positive

# ======================
# Prediction Logic
# ======================
st.markdown("---")
col1, col2 = st.columns(2)
predict_btn = col1.button("🔮 Predict Popularity")
col2.button("🔁 Reset", on_click=reset_inputs)

if predict_btn:

    if views == 0 or likes == 0 or comments_count == 0:
        st.error("⚠️ Please enter Views, Likes, and Comments Count.")
        st.stop()

    non_empty = [c for c in comment_inputs if c.strip() != ""]
    if len(non_empty) < 2:
        st.error("⚠️ Enter at least TWO non-empty comments.")
        st.stop()

    # Compute sentiment of each comment
    sentiments = []
    for c in non_empty:
        c_clean = clean_comment(c)
        if c_clean:
            sentiments.append(get_raw_sentiment(c_clean))

    avg_sentiment = np.mean(sentiments)

    # Convert to ANN CLASS (0,1,2)
    sentiment_class = convert_sentiment_to_class(avg_sentiment)

    # ANN expects → [views, likes, comment_count, sentiment_class]
    X = np.array([[views, likes, comments_count, sentiment_class]])
    X_scaled = scaler.transform(X)

    # Predict
    pred = model.predict(X_scaled)
    pred_class = np.argmax(pred)
    confidence = np.max(pred)

    # Labels
    labels = {
        0: ("Low Popularity", "📉"),
        1: ("Medium Popularity", "📊"),
        2: ("High Popularity", "🔥")
    }

    result_text, emoji = labels[pred_class]

    # Display
    st.success(f"{emoji} **Predicted Popularity: {result_text}**")
    st.write(f"🤖 Model Confidence: **{confidence:.2%}**")

    st.markdown("---")
    st.subheader("📌 Sentiment Summary")
    st.write(f"Raw Avg Sentiment (-1..1): **{avg_sentiment:.3f}**")
    st.write(f"Sentiment Class Used for ANN: **{sentiment_class}**")

    with st.expander("View Analyzed Comments"):
        for i, s in enumerate(sentiments, start=1):
            st.write(f"Comment {i}: Sentiment = {s:.3f}")

    # Simple chart
    df_plot = pd.DataFrame({
        "Metric": ["Views", "Likes", "Comments"],
        "Value": [views, likes, comments_count]
    })
    st.bar_chart(df_plot.set_index("Metric"))

    # Recommendations
    st.subheader("📌 Recommendations")

    tips = []
    if pred_class == 0:
        tips.append("📉 Improve SEO, thumbnails, and title optimization.")
    elif pred_class == 1:
        tips.append("📊 Moderate performance — boost engagement with CTAs.")
    else:
        tips.append("🔥 Strong performance — keep consistency.")

    if sentiment_class == 0:
        tips.append("😟 Viewers feel negative — review feedback and adjust content.")
    elif sentiment_class == 1:
        tips.append("🙂 Balanced sentiment — improve clarity and pacing.")
    else:
        tips.append("🥰 Very positive comments — great audience reception!")

    for t in tips:
        st.write(t)

















