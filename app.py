# =========================================
# app.py — FINAL SUPERVISOR-APPROVED VERSION
# =========================================

import streamlit as st
import numpy as np
import joblib
import re
import os
import pandas as pd
from tensorflow.keras.models import load_model
from textblob import TextBlob
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# ======================
# Page Config
# ======================
st.set_page_config(
    page_title="YouTube Popularity Predictor",
    page_icon="🎬",
    layout="centered"
)

# ======================
# Load CSS
# ======================
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except:
        pass

local_css("style.css")

st.title("🎬 YouTube Video Popularity Prediction")
st.markdown("---")

# ======================
# Session Flags
# ======================
if "show_results" not in st.session_state:
    st.session_state.show_results = False

# ======================
# Sentiment Setup
# ======================
use_vader = False
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    nltk.download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()
    use_vader = True
except:
    use_vader = False

# ======================
# Load ANN Model + Scaler
# ======================
model = load_model("model/youtube_popularity_ann.h5")
scaler = joblib.load("model/scaler.pkl")

# ======================
# Reset Function
# ======================
def reset_all():
    st.session_state.views = 0
    st.session_state.likes = 0
    st.session_state.comments_count = 0

    for i in range(10):
        st.session_state[f"comment_{i}"] = ""

    st.session_state.show_results = False
    st.session_state.pred_class = None
    st.session_state.sentiment_class = None
    st.session_state.avg_sentiment = None
    st.session_state.sentiments = []

# ======================
# Tabs
# ======================
tab_home, tab_predict, tab_contact = st.tabs(
    ["🏠 Home", "🔮 Prediction", "📩 Contact & Feedback"]
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
        return 0
    elif score <= 0.25:
        return 1
    else:
        return 2

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
    st.write("📉 Low | 📊 Medium | 🔥 High")

    st.markdown("---")
    st.header("📊 Model Performance")

    X_test = joblib.load("model/X_test.pkl")
    y_test = joblib.load("model/y_test.pkl")

    y_pred = np.argmax(model.predict(scaler.transform(X_test)), axis=1)

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred)*100:.2f}%")
    col2.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted')*100:.2f}%")
    col3.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted')*100:.2f}%")

    cm = confusion_matrix(y_test, y_pred)
    fig_cm = ff.create_annotated_heatmap(
        z=cm,
        x=["Low", "Medium", "High"],
        y=["Low", "Medium", "High"],
        colorscale="Blues"
    )
    st.plotly_chart(fig_cm, use_container_width=True)

# ======================
# PREDICTION TAB
# ======================
with tab_predict:
    st.header("🔮 Predict Video Popularity")

    views = st.number_input("Total Views", 0, key="views")
    likes = st.number_input("Total Likes", 0, key="likes")
    comments_count = st.number_input("Total Comments Count", 0, key="comments_count")

    st.subheader("💬 Enter at least TWO comments")
    cols = st.columns(2)
    comments = []
    for i in range(10):
        with cols[i % 2]:
            comments.append(st.text_input(f"Comment {i+1}", key=f"comment_{i}"))

    col1, col2 = st.columns(2)
    predict_btn = col1.button("🔮 Predict")
    col2.button("🔁 Reset", on_click=reset_all)

    # ✅ FIXED INDENTATION
    if predict_btn:

        if views == 0 or likes == 0 or comments_count == 0:
            st.error("⚠️ Please enter Views, Likes, and Comments Count.")
            st.stop()

        valid_comments = [c for c in comments if c.strip() != ""]
        if len(valid_comments) < 2:
            st.error("⚠️ Enter at least TWO non-empty comments.")
            st.stop()

        sentiments = []
        for c in valid_comments:
            sentiments.append(get_raw_sentiment(clean_comment(c)))

        avg_sentiment = np.mean(sentiments)
        sentiment_class = convert_sentiment_to_class(avg_sentiment)

        X = scaler.transform([[views, likes, comments_count, sentiment_class]])
        prediction = model.predict(X)

        pred_class = np.argmax(prediction)

        labels = {
            0: ("Low Popularity", "📉"),
            1: ("Medium Popularity", "📊"),
            2: ("High Popularity", "🔥")
        }

        result_text, emoji = labels[pred_class]
        st.success(f"{emoji} **Predicted Popularity: {result_text}**")

        # Store results
        st.session_state.show_results = True
        st.session_state.pred_class = pred_class
        st.session_state.sentiments = sentiments
        st.session_state.avg_sentiment = avg_sentiment

    if st.session_state.show_results:
        st.subheader("📊 Sentiment Analysis")

        fig = go.Figure()
        fig.add_bar(
            x=[f"Comment {i+1}" for i in range(len(st.session_state.sentiments))],
            y=st.session_state.sentiments
        )
        fig.update_layout(
            title="Sentiment Score per Comment",
            yaxis_title="Sentiment Score (-1 to +1)"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📌 Actionable Recommendations")

        # Views
        if views < 1000:
            st.write("👀 **Low Views** — Improve SEO, thumbnails, and titles.")
        elif views < 10000:
            st.write("👀 **Moderate Views** — Promote content on social media.")
        else:
            st.write("👀 **High Views** — Maintain posting consistency.")

        # Likes
        like_ratio = likes / max(views, 1)
        if like_ratio < 0.02:
            st.write("👍 **Low Likes Engagement** — Encourage likes via CTA.")
        elif like_ratio < 0.05:
            st.write("👍 **Average Likes Engagement** — Improve content appeal.")
        else:
            st.write("👍 **High Likes Engagement** — Strong audience approval.")

        # Comments
        if comments_count < 50:
            st.write("💬 **Low Comments** — Ask questions to engage viewers.")
        elif comments_count < 200:
            st.write("💬 **Moderate Comments** — Reply to comments actively.")
        else:
            st.write("💬 **High Comments** — Strong community engagement.")

        # Sentiment
        if st.session_state.avg_sentiment < -0.25:
            st.write("😟 **Negative Sentiment** — Address viewer concerns.")
        elif st.session_state.avg_sentiment <= 0.25:
            st.write("🙂 **Neutral Sentiment** — Add emotional storytelling.")
        else:
            st.write("🥰 **Positive Sentiment** — Excellent audience satisfaction.")

# ======================
# CONTACT TAB
# ======================
with tab_contact:
    st.header("📩 Contact & Feedback")

    with st.form("feedback_form", clear_on_submit=True):
        name = st.text_input("Name")
        email = st.text_input("Email (optional)")
        feedback = st.text_area("Your Feedback")
        submit = st.form_submit_button("Submit")

    if submit and feedback.strip():
        os.makedirs("feedback", exist_ok=True)
        pd.DataFrame([[name, email, feedback]],
                     columns=["Name", "Email", "Feedback"]
                     ).to_csv(
            "feedback/feedback.csv",
            mode="a",
            header=not os.path.exists("feedback/feedback.csv"),
            index=False
        )
        st.success("✅ Feedback saved successfully!")

