import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from textblob import TextBlob
import pandas as pd

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
# Optional CSS
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
    st.session_state.reset_flag = True
    st.rerun()  # works in latest Streamlit


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

# ======================
# Prediction Section
# ======================
st.markdown("---")

col1, col2 = st.columns([1, 1])
with col1:
    predict_btn = st.button("🔮 Predict Popularity")

with col2:
    reset_btn = st.button("🔁 Reset", on_click=reset_inputs)

# ======================
# RUN PREDICTION
# ======================
if predict_btn:

    # ------------------------------
    # VALIDATION CHECKS
    # ------------------------------
    if views == 0 or likes == 0 or comments_count == 0:
        st.error("⚠️ Please fill in **Views**, **Likes**, and **Total Comments Count** before predicting.")
        st.stop()

    # At least 2 comments required
    non_empty_comments = [c for c in comment_inputs if c.strip() != ""]
    if len(non_empty_comments) < 2:
        st.error("⚠️ Please enter **at least TWO comments** for sentiment analysis.")
        st.stop()

    # ------------------------------
    # Sentiment Calculation
    # ------------------------------
    avg_sentiment, num_comments = get_avg_sentiment(comment_inputs)

    # Prepare ANN input
    user_data = np.array([[views, likes, comments_count, avg_sentiment]])
    user_data_scaled = scaler.transform(user_data)
    prediction = model.predict(user_data_scaled)
    popularity_class = np.argmax(prediction, axis=1)[0]

    # Weighted score
    max_views, max_likes, max_sentiment = 1_000_000, 50_000, 1.0
    views_rank = normalize(views, max_views)
    likes_rank = normalize(likes, max_likes)
    sentiment_rank = normalize(avg_sentiment, max_sentiment)

    popularity_score = (0.5 * views_rank) + (0.3 * likes_rank) + (0.2 * sentiment_rank)

    # Result label
    if popularity_class == 0:
        result, emoji = "Low Popularity", "📉"
    elif popularity_class == 1:
        result, emoji = "Medium Popularity", "📊"
    else:
        result, emoji = "High Popularity", "🔥"

    # ======================
    # Show Results
    # ======================
    st.success(f"{emoji} **Predicted Popularity: {result}**")

    # --------------------------------
    # 📊 Visualization Section
    # --------------------------------
    st.subheader("📊 Video Metrics Visualization")

    df = pd.DataFrame({
        "Metric": ["Views", "Likes", "Total Comments"],
        "Value": [views, likes, comments_count]
    })

    st.bar_chart(df.set_index("Metric"))

    # --------------------------------
    # 📌 Performance Overview
    # --------------------------------
    st.subheader("📈 Performance Breakdown")

    st.write(f"👀 **Views:** {views:,}")
    st.write(f"👍 **Likes:** {likes:,}")
    st.write(f"💬 **Total Comments:** {comments_count:,}")
    st.write(f"🧠 **Average Sentiment Score:** {avg_sentiment:.2f}")
    st.write(f"📊 **Weighted Popularity Score:** {popularity_score:.2f}")
    st.write(f"💬 Comments Analyzed: {num_comments}")

    # ======================
    # Recommendations
    # ======================
    st.subheader("📌 Personalized Recommendations")
    tips = []

    # Views advice
    if views_rank < 0.3:
        tips.append("📉Low Views — improve SEO, use better thumbnails, or share more widely.")
    elif views_rank < 0.7:
        tips.append("👀Moderate Views — optimize titles and increase watch time.")
    else:
        tips.append("🔥High Views — great! Continue similar content.")

    # Likes advice
    if likes_rank < 0.3:
        tips.append("👍Low Likes — ask viewers to like your video and improve early hooks.")
    else:
        tips.append("🌟Strong Likes — your audience is engaged!")

    # Sentiment advice
    if sentiment_rank < 0.3:
        tips.append("😟Low Sentiment — review feedback and improve clarity.")
    else:
        tips.append("🥰Positive Sentiment — viewers enjoy your video tone!")

    for tip in tips:
        st.write(tip)














