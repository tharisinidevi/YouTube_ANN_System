# =========================================
# app.py — FINAL SUPERVISOR-APPROVED VERSION
# =========================================

import streamlit as st
import numpy as np
import joblib
import re
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
model = load_model("model/youtube_popularity_ann.h5")
scaler = joblib.load("model/scaler.pkl")

# ======================
# RESET FUNCTION (IMPORTANT)
# ======================
def reset_all():
    keys_to_clear = [
        "views", "likes", "comments_count",
        "pred_class", "avg_sentiment", "sentiments"
    ]
    for i in range(10):
        keys_to_clear.append(f"comment_{i}")

    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    st.rerun()

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
# Tabs
# ======================
tab_home, tab_predict, tab_contact = st.tabs(
    ["🏠 Home", "🔮 Prediction", "📩 Contact & Feedback"]
)

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
    1. User provides engagement metrics  
    2. Viewer comments are analyzed for sentiment  
    3. Features are normalized using a scaler  
    4. ANN predicts popularity class  
    """)

    st.subheader("🎯 Popularity Classes")
    st.write("📉 Low | 📊 Medium | 🔥 High")

    st.markdown("---")
    st.header("📊 Model Performance Evaluation")

    try:
        X_test = joblib.load("model/X_test.pkl")
        y_test = joblib.load("model/y_test.pkl")
    except:
        st.error("❌ X_test.pkl or y_test.pkl not found in model folder.")
        st.stop()

    X_test_scaled = scaler.transform(X_test)
    y_pred = np.argmax(model.predict(X_test_scaled), axis=1)

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred)*100:.2f}%")
    col2.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted')*100:.2f}%")
    col3.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted')*100:.2f}%")

    st.subheader("📌 Confusion Matrix")

    fig_cm = ff.create_annotated_heatmap(
        z=confusion_matrix(y_test, y_pred),
        x=["Low", "Medium", "High"],
        y=["Low", "Medium", "High"],
        colorscale="Blues"
    )
    fig_cm.update_layout(xaxis_title="Predicted", yaxis_title="Actual")

    st.plotly_chart(fig_cm, use_container_width=True)

# ======================
# PREDICTION TAB
# ======================
with tab_predict:
    st.header("🔮 Predict Video Popularity")

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
                st.text_input(f"Comment {i+1}", key=f"comment_{i}")
            )

    st.markdown("---")
    col1, col2 = st.columns(2)
    predict_btn = col1.button("🔮 Predict Popularity")
    col2.button("🔁 Reset", on_click=reset_all)

    if predict_btn:
        if views == 0 or likes == 0 or comments_count == 0:
            st.error("⚠️ Please enter Views, Likes and Comments Count together.")
            st.stop()

        valid_comments = [c for c in comment_inputs if c.strip()]
        if len(valid_comments) < 2:
            st.error("⚠️ Please enter at least TWO comments.")
            st.stop()

        sentiments = [get_raw_sentiment(clean_comment(c)) for c in valid_comments]
        avg_sentiment = np.mean(sentiments)
        sentiment_class = convert_sentiment_to_class(avg_sentiment)

        X = scaler.transform([[views, likes, comments_count, sentiment_class]])
        pred_class = np.argmax(model.predict(X))

        labels = {
            0: "📉 Low Popularity",
            1: "📊 Medium Popularity",
            2: "🔥 High Popularity"
        }

        st.success(f"**Predicted Result:** {labels[pred_class]}")

        st.session_state.pred_class = pred_class
        st.session_state.avg_sentiment = avg_sentiment
        st.session_state.sentiments = sentiments

    if "pred_class" in st.session_state:
        st.markdown("---")
        st.header("💡 Insights & Recommendations")

        fig = go.Figure()
        fig.add_bar(
            x=[f"Comment {i+1}" for i in range(len(st.session_state.sentiments))],
            y=st.session_state.sentiments
        )
        fig.update_layout(yaxis_title="Sentiment Score (-1 to 1)")
        st.plotly_chart(fig, use_container_width=True)

        st.write(f"**Average Sentiment Score:** `{st.session_state.avg_sentiment:.3f}`")

# ======================
# CONTACT TAB
# ======================
with tab_contact:
    st.header("📩 Contact & Feedback")

    st.write("""
    This section allows users to submit feedback or suggestions
    to improve the prediction system.
    """)

    with st.form("feedback_form", clear_on_submit=True):
        name = st.text_input("Name")
        email = st.text_input("Email (optional)")
        feedback = st.text_area("Your Feedback / Suggestions", height=150)
        submit = st.form_submit_button("Submit Feedback")

    if submit:
        if feedback.strip() == "":
            st.warning("⚠️ Please enter feedback before submitting.")
        else:
            st.success("✅ Thank you! Your feedback has been received.")








