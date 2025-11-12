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

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Apply your external CSS
local_css("style.css")
# ======================
# Input Section
# ======================
st.subheader("📊 Enter Video Metrics")
views = st.number_input("Total Views", min_value=0, step=1)
likes = st.number_input("Total Likes", min_value=0, step=1)
comments_count = st.number_input("Total Comments Count", min_value=0, step=1)

st.markdown("---")

# ======================
# Comments Section (10 inputs)
# ======================
st.subheader("💬 Enter Up to 10 Top Comments")
cols = st.columns(2)  # create 2 columns layout
comment_inputs = []

for i in range(10):
    with cols[i % 2]:
        comment = st.text_input(f"Comment {i+1}", "")
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
    return 0.0, 0  # no comments

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
    reset_btn = st.button("🔁 Reset")

if predict_btn:
    avg_sentiment, num_comments = get_avg_sentiment(comment_inputs)

    # Prepare data
    user_data = np.array([[views, likes, comments_count, avg_sentiment]])
    user_data_scaled = scaler.transform(user_data)

    prediction = model.predict(user_data_scaled)
    popularity_class = np.argmax(prediction, axis=1)[0]

    # Weighted popularity score (consistent with your training formula)
    max_views = 1000000   # normalization reference
    max_likes = 50000
    max_sentiment = 1.0

    views_rank = normalize(views, max_views)
    likes_rank = normalize(likes, max_likes)
    sentiment_rank = normalize(avg_sentiment, max_sentiment)

    popularity_score = (0.5 * views_rank) + (0.3 * likes_rank) + (0.2 * sentiment_rank)

    # Popularity level
    if popularity_class == 0:
        result = "Low Popularity"
    elif popularity_class == 1:
        result = "Medium Popularity"
    else:
        result = "High Popularity"

    # ======================
    # Display Results
    # ======================
    st.success(f"✅ Predicted Popularity: **{result}**")
    st.write(f"🧠 Average Sentiment Score: **{avg_sentiment:.2f}**")
    st.write(f"📈 Weighted Popularity Score: **{popularity_score:.2f}**")
    st.write(f"💬 Comments Analyzed: **{num_comments}**")

    if num_comments == 0:
        st.warning("⚠️ No comments entered — sentiment not factored into the prediction.")

    # ======================
    # Personalized Recommendations
    # ======================
    st.subheader("📌 Personalized Recommendations")

    tips = []

    # Views (50% weight)
    if views_rank < 0.3:
        tips.append("📉 **Low Views (50%)** – Improve SEO, collaborate with creators, and promote across platforms.")
    elif views_rank < 0.7:
        tips.append("👀 **Moderate Views** – Optimize video titles, tags, and thumbnails.")
    else:
        tips.append("🔥 **High Views** – Maintain your trend; experiment with related content themes.")

    # Likes (30% weight)
    if likes_rank < 0.3:
        tips.append("👍 **Low Likes (30%)** – Add interactive CTAs or improve engagement hooks.")
    elif likes_rank < 0.7:
        tips.append("💖 **Moderate Likes** – Try storytelling or stronger emotion-based messaging.")
    else:
        tips.append("🌟 **High Likes** – Great audience connection! Continue same tone and pacing.")

    # Sentiment (20% weight)
    if sentiment_rank < 0.3:
        tips.append("😟 **Low Sentiment (20%)** – Address criticism or improve clarity and positivity.")
    elif sentiment_rank < 0.7:
        tips.append("🙂 **Mixed Sentiment** – Refine delivery tone and pacing to boost viewer satisfaction.")
    else:
        tips.append("🥰 **Positive Sentiment** – Viewers love your content! Build on their enthusiasm.")

    # Comment engagement
    if comments_count < (0.01 * views):
        tips.append("💬 **Low Comment Ratio** – Ask questions or encourage discussion in video.")
    else:
        tips.append("💭 **Good Engagement** – Keep engaging with replies and pin top comments.")

    for t in tips:
        st.write(t)

    st.info("💡 Insights weighted: Views (50%) • Likes (30%) • Sentiment (20%).")

elif st.button("🔁 Reset Form"):
    st.session_state.clear()
    st.experimental_rerun()














