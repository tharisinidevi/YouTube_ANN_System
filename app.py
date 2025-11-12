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
# Streamlit UI
# ======================
st.set_page_config(page_title="YouTube Popularity Predictor", page_icon="🎬", layout="centered")
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Apply your external CSS
local_css("style.css")


st.title("🎬 YouTube Video Popularity Predictor (Weight-Based Recommendations)")
st.write("Predict your video's popularity based on **views**, **likes**, **comments**, and **sentiment**!")

st.subheader("📊 Enter Video Metrics")
views = st.number_input("Total Views", min_value=0, step=1)
likes = st.number_input("Total Likes", min_value=0, step=1)
comments_count = st.number_input("Total Comments Count", min_value=0, step=1)

# ======================
# Comment Section (10 boxes)
# ======================
st.subheader("💬 Enter Top Comments (up to 10, optional)")
comments = []
for i in range(10):
    comment = st.text_input(f"Comment {i+1}", "")
    comments.append(comment.strip())

# Combine all non-empty comments
user_comments = "\n".join([c for c in comments if c])

# ======================
# Helper Functions
# ======================
def get_avg_sentiment(comments_text):
    comments = [c.strip() for c in comments_text.split("\n") if c.strip()]
    sentiments = []
    for comment in comments:
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
# Prediction Logic
# ======================
if st.button("🔮 Predict Popularity"):
    avg_sentiment, num_comments = get_avg_sentiment(user_comments)

    # Scale input
    user_data = np.array([[views, likes, comments_count, avg_sentiment]])
    user_data_scaled = scaler.transform(user_data)
    prediction = model.predict(user_data_scaled)
    popularity_class = np.argmax(prediction, axis=1)[0]

    # Weighted popularity score (your formula)
    max_views = 1_000_000
    max_likes = 50_000
    max_sentiment = 1.0

    views_rank = normalize(views, max_views)
    likes_rank = normalize(likes, max_likes)
    sentiment_rank = normalize(avg_sentiment, max_sentiment)

    popularity_score = (0.5 * views_rank) + (0.3 * likes_rank) + (0.2 * sentiment_rank)

    # Popularity Level
    if popularity_class == 0:
        result = "Low Popularity"
    elif popularity_class == 1:
        result = "Medium Popularity"
    else:
        result = "High Popularity"

    # ======================
    # Output Results
    # ======================
    st.success(f"✅ Predicted Popularity: **{result}**")
    st.write(f"🧠 Average Sentiment Score: {avg_sentiment:.2f}")
    st.write(f"📈 Weighted Popularity Score: **{popularity_score:.2f}**")
    st.write(f"💬 Comments Analyzed: **{num_comments}**")

    if num_comments == 0:
        st.warning("⚠️ No comments provided — sentiment analysis skipped (prediction may be less accurate).")

    # ======================
    # Personalized Recommendations
    # ======================
    st.subheader("📌 Personalized Recommendations:")

    tips = []

    # Views-based
    if views_rank < 0.3:
        tips.append("📉 **Low Views (50% weight)** – Focus on SEO optimization and collaboration to increase reach.")
    elif views_rank < 0.7:
        tips.append("👀 **Moderate Views** – Improve titles, tags, and watch time to boost visibility.")
    else:
        tips.append("🔥 **High Views** – Maintain consistency; create similar successful content.")

    # Likes-based
    if likes_rank < 0.3:
        tips.append("👍 **Low Likes (30% weight)** – Add stronger calls-to-action or engaging video hooks.")
    elif likes_rank < 0.7:
        tips.append("💖 **Moderate Likes** – Try emotional storytelling or better thumbnails.")
    else:
        tips.append("🌟 **High Likes** – Audience enjoys your content! Maintain your tone and energy.")

    # Sentiment-based
    if sentiment_rank < 0.3:
        tips.append("😟 **Low Sentiment (20% weight)** – Address criticism; improve tone and clarity.")
    elif sentiment_rank < 0.7:
        tips.append("🙂 **Mixed Sentiment** – Adjust pacing or explore lighter topics.")
    else:
        tips.append("🥰 **Positive Sentiment** – Viewers love your content! Build on this success.")

    # Comment ratio
    if comments_count < (0.01 * views):
        tips.append("💬 **Low Comment Ratio** – Ask questions or create polls to encourage more interaction.")
    else:
        tips.append("💭 **Good Engagement** – Keep responding to build loyalty.")

    for tip in tips:
        st.write(tip)

    st.info("💡 Insights are weighted by your model: Views (50%) > Likes (30%) > Sentiment (20%).")

# ======================
# Reset Button
# ======================
if st.button("🔁 Reset Form"):
    st.session_state.clear()
    st.experimental_rerun()












