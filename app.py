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

st.title("🎬 YouTube Video Popularity Predictor")
st.caption("Predict your video popularity based on engagement metrics and viewer sentiment.")

# Initialize session state for reset
if "reset_flag" not in st.session_state:
    st.session_state.reset_flag = False

def reset_inputs():
    st.session_state["views"] = 0
    st.session_state["likes"] = 0
    st.session_state["comments_count"] = 0
    st.session_state["user_comments"] = ""
    st.session_state.reset_flag = True

# ======================
# Input Section
# ======================
st.subheader("📊 Enter Video Metrics")

views = st.number_input("Total Views", min_value=0, step=1, key="views")
likes = st.number_input("Total Likes", min_value=0, step=1, key="likes")
comments_count = st.number_input("Total Comments Count", min_value=0, step=1, key="comments_count")

st.subheader("💬 Paste Top Comments (optional)")
user_comments = st.text_area("Enter available comments (one per line)", key="user_comments")

# ======================
# Sentiment Calculation
# ======================
def get_avg_sentiment(comments_text):
    comments = [c.strip() for c in comments_text.split("\n") if c.strip()]
    sentiments = [TextBlob(c).sentiment.polarity for c in comments]
    return (np.mean(sentiments), len(sentiments)) if sentiments else (0.0, 0)

# ======================
# Normalize Helper Function (for weighted score only)
# ======================
def normalize(value, max_value):
    if max_value == 0:
        return 0
    return min(value / max_value, 1.0)

# ======================
# Prediction Button
# ======================
col1, col2 = st.columns([1, 1])
with col1:
    predict_btn = st.button("🔮 Predict Popularity")
with col2:
    reset_btn = st.button("🔁 Reset", on_click=reset_inputs)

if predict_btn:
    avg_sentiment, num_comments = get_avg_sentiment(user_comments)

    # ======================
    # Model Input (aligned with training)
    # ======================
    user_data = np.array([[views, likes, comments_count, avg_sentiment]])
    user_data_scaled = scaler.transform(user_data)  # same scaling as during training

    # Predict
    prediction = model.predict(user_data_scaled)
    popularity_class = np.argmax(prediction, axis=1)[0]

    # Popularity class mapping
    if popularity_class == 0:
        result = "Not Popular"
        emoji = "📉"
    elif popularity_class == 1:
        result = "Average"
        emoji = "📊"
    else:
        result = "Popular"
        emoji = "🔥"

    # Display prediction
    st.success(f"{emoji} **Predicted Popularity:** {result}")
    st.write(f"🧠 Average Sentiment Score: `{avg_sentiment:.2f}`")
    st.write(f"💬 Comments Analyzed: `{num_comments}`")

    if num_comments == 0:
        st.warning("⚠️ No comments provided — sentiment analysis skipped (prediction may be less accurate).")

    # ======================
    # Weighted Popularity Score (for recommendations only)
    # ======================
    w_views, w_likes, w_sent = 0.5, 0.3, 0.2
    max_views, max_likes, max_sentiment = 1_000_000, 50_000, 1.0

    views_rank = normalize(views, max_views)
    likes_rank = normalize(likes, max_likes)
    sentiment_rank = normalize(avg_sentiment, max_sentiment)
    popularity_score = (w_views * views_rank) + (w_likes * likes_rank) + (w_sent * sentiment_rank)

    st.write(f"📈 Weighted Popularity Score (Display Only): **{popularity_score:.2f}**")

    # ======================
    # Personalized Recommendations
    # ======================
    st.subheader("📌 Personalized Recommendations")

    tips = []

    # Views-driven (50%)
    if views_rank < 0.3:
        tips.append("📉 **Low Views (50%)** – Improve SEO and collaborate with similar creators.")
    elif views_rank < 0.7:
        tips.append("👀 **Moderate Views** – Optimize video titles, tags, and improve watch time.")
    else:
        tips.append("🔥 **High Views** – Maintain upload consistency; expand into related niches.")

    # Likes-driven (30%)
    if likes_rank < 0.3:
        tips.append("👍 **Low Likes (30%)** – Use clear CTAs and engaging intros to increase reactions.")
    elif likes_rank < 0.7:
        tips.append("💖 **Moderate Likes** – Try emotional storytelling and attractive thumbnail designs.")
    else:
        tips.append("🌟 **High Likes** – Strong engagement! Continue leveraging audience preferences.")

    # Sentiment-driven (20%)
    if sentiment_rank < 0.3:
        tips.append("😟 **Low Sentiment (20%)** – Address criticism or clarify message tone.")
    elif sentiment_rank < 0.7:
        tips.append("🙂 **Mixed Sentiment** – Balance humor and clarity for better connection.")
    else:
        tips.append("🥰 **Positive Sentiment** – Viewers love your content! Keep your tone and style.")

    # Comments-to-views ratio
    if comments_count < (0.01 * views):
        tips.append("💬 **Low Comment Ratio** – Ask interactive questions to boost engagement.")
    else:
        tips.append("💭 **Good Engagement** – Maintain community interaction through replies and polls.")

    for tip in tips:
        st.write(tip)

    st.info("💡 Insights follow your ANN model weighting: Views (50%) > Likes (30%) > Sentiment (20%).")

elif reset_btn:
    st.experimental_rerun()








