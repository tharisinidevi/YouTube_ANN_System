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
# Reset Function (CORRECT)
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
    This system predicts YouTube video popularity by integrating:
    - Engagement metrics (views, likes, comments)
    - Viewer sentiment extracted from comments
    - An Artificial Neural Network (ANN) classifier
    """)

    st.subheader("🔄 System Workflow")
    st.write("""
    1. Engagement metrics are provided  
    2. Viewer comments are analyzed for sentiment  
    3. Features are normalized using a scaler  
    4. ANN predicts popularity level  
    """)


    st.subheader("🎯 Popularity Classes")
    st.write("📉 Low | 📊 Medium | 🔥 High")

    st.markdown("---")
    st.header("📊 Model Performance")

    try:
        X_test = joblib.load("model/X_test.pkl")
        y_test = joblib.load("model/y_test.pkl")
    except:
        st.error("Test dataset not found.")
        st.stop()

    X_test_scaled = scaler.transform(X_test)
    y_pred = np.argmax(model.predict(X_test_scaled), axis=1)

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred)*100:.2f}%")
    col2.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted')*100:.2f}%")
    col3.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted')*100:.2f}%")

    cm = confusion_matrix(y_test, y_pred)
    fig_cm = ff.create_annotated_heatmap(
        z=cm,
        x=["Low","Medium","High"],
        y=["Low","Medium","High"],
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

    if predict_btn:
        if views == 0 or likes == 0 or comments_count == 0:
            st.error("Please enter Views, Likes and Comments Count.")
            st.stop()

        valid_comments = [c for c in comments if c.strip()]
        if len(valid_comments) < 2:
            st.error("Enter at least TWO comments.")
            st.stop()

        sentiments = [get_raw_sentiment(clean_comment(c)) for c in valid_comments]
        avg_sentiment = np.mean(sentiments)
        sentiment_class = convert_sentiment_to_class(avg_sentiment)

        X = scaler.transform([[views, likes, comments_count, sentiment_class]])
        pred_class = np.argmax(model.predict(X))

        st.session_state.show_results = True
        st.session_state.pred_class = pred_class
        st.session_state.sentiment_class = sentiment_class
        st.session_state.avg_sentiment = avg_sentiment
        st.session_state.sentiments = sentiments

        labels = ["Low", "Medium", "High"]
        st.success(f"🔥 Predicted Popularity: **{labels[pred_class]}**")

    if st.session_state.show_results:
        st.subheader("💡 Recommendations")

        recs = []
        recs.append("📉 Improve thumbnails & SEO" if st.session_state.pred_class == 0 else
                    "📊 Increase engagement" if st.session_state.pred_class == 1 else
                    "🔥 Maintain consistency")

        recs.append("😟 Address negative feedback" if st.session_state.sentiment_class == 0 else
                    "🙂 Add emotional appeal" if st.session_state.sentiment_class == 1 else
                    "🥰 Audience loves it!")

        for r in recs:
            st.write(r)

# ======================
# CONTACT TAB (CSV STORAGE)
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
        file_path = "feedback/feedback.csv"

        df = pd.DataFrame([[name, email, feedback]],
                          columns=["Name", "Email", "Feedback"])

        if os.path.exists(file_path):
            df.to_csv(file_path, mode="a", header=False, index=False)
        else:
            df.to_csv(file_path, index=False)

        st.success("✅ Feedback saved successfully!")





