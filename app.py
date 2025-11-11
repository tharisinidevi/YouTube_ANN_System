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
st.title("🎬 YouTube Video Popularity Predictor (Weight-Based Recommendations)")

st.subheader("📊 Enter Video Metrics")
views = st.number_input("Total Views", min_value=0, step=1)
likes = st.number_input("Total Likes", min_value=0, step=1)
comments_count = st.number_input("Total Comments Count", min_value=0, step=1)

st.subheader("💬 Paste Top Comments (any number, one per line)")
user_comments = st.text_area("Enter available comments (optional)")

# ======================
# Sentiment Calculation
# ======================
def get_avg_sentiment(comments_text):
    comments = [c.strip() for c in comments_text.split("\n") if c.strip()]
    sentiments = []
    for comment in comments:
        polarity = TextBlob(comment).sentiment.polarity
        sentiments.append(polarity)
    if sentiments:
        return np.mean(sentiments), len(sentiments)
    return 0.0, 0  # default if no comments

# ======================
# Normalize Helper Function
# ======================
def normalize(value, max_value):
    if max_value == 0:
        return 0
    return min(value / max_value, 1.0)

# ======================
# Prediction Button
# ======================
if st.button("🔮 Predict Popularity"):
    avg_sentiment, num_comments = get_avg_sentiment(user_comments)

    # Prepare input for ANN model
    user_data = np.array([[views, likes, comments_count, avg_sentiment]])
    user_data_scaled = scaler.transform(user_data)
    prediction = model.predict(user_data_scaled)
    popularity_class = np.argmax(prediction, axis=1)[0]

    # ======================
    # Weighted Popularity Score (from your model formula)
    # ======================
    max_views = 1000000   # normalize relative to 1M views
    max_likes = 50000     # normalize relative to 50k likes
    max_sentiment = 1.0   # sentiment polarity max

    views_rank = normalize(views, max_views)
    likes_rank = normalize(likes, max_likes)
    sentiment_rank = normalize(avg_sentiment, max_sentiment)

    popularity_score = (0.5 * views_rank) + (0.3 * likes_rank) + (0.2 * sentiment_rank)

    # ======================
    # Popularity Level Mapping
    # ======================
    if popularity_class == 0:
        result = "Low Popularity"
    elif popularity_class == 1:
        result = "Medium Popularity"
    else:
        result = "High Popularity"

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

    # Views-driven suggestions (50% weight)
    if views_rank < 0.3:
        tips.append("📉 **Low Views (50% weight)** – Focus on SEO optimization and collaboration to increase reach.")
    elif views_rank < 0.7:
        tips.append("👀 **Moderate Views** – Consider improving titles, tags, and watch time to boost visibility.")
    else:
        tips.append("🔥 **High Views** – Maintain consistency; experiment with similar successful topics.")

    # Likes-driven suggestions (30% weight)
    if likes_rank < 0.3:
        tips.append("👍 **Low Likes (30% weight)** – Add stronger calls-to-action or engaging video hooks.")
    elif likes_rank < 0.7:
        tips.append("💖 **Moderate Likes** – Try emotional storytelling or better thumbnail design.")
    else:
        tips.append("🌟 **High Likes** – Audience enjoys your content! Maintain tone and engagement style.")

    # Sentiment-driven suggestions (20% weight)
    if sentiment_rank < 0.3:
        tips.append("😟 **Low Sentiment (20% weight)** – Address viewer criticism; improve tone and clarity.")
    elif sentiment_rank < 0.7:
        tips.append("🙂 **Mixed Sentiment** – Experiment with pacing, topic depth, or tone.")
    else:
        tips.append("🥰 **Positive Sentiment** – Viewers love your content! Build on their feedback.")

    # Comment engagement check
    if comments_count < (0.01 * views):
        tips.append("💬 **Low Comment Ratio** – Ask questions or create polls to encourage more discussion.")
    else:
        tips.append("💭 **Good Engagement** – Keep interacting with your audience to boost loyalty.")

    for tip in tips:
        st.write(tip)

    st.info("💡 These insights align with your ANN model’s internal weighting: Views (50%) > Likes (30%) > Sentiment (20%).")






