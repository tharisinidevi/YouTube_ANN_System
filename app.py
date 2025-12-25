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
    st.header("📊 Model Performance Evaluation")

    try:
        X_test = joblib.load("model/X_test.pkl")
        y_test = joblib.load("model/y_test.pkl")
    except:
        st.error("❌ X_test.pkl or y_test.pkl not found in model folder.")
        st.stop()

    X_test_scaled = scaler.transform(X_test)
    y_pred_prob = model.predict(X_test_scaled)
    y_pred = np.argmax(y_pred_prob, axis=1)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{acc*100:.2f}%")
    col2.metric("Precision", f"{prec*100:.2f}%")
    col3.metric("Recall", f"{rec*100:.2f}%")

    st.subheader("📌 Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)
    labels = ["Low", "Medium", "High"]

    fig_cm = ff.create_annotated_heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale="Blues",
        showscale=True
    )

    fig_cm.update_layout(
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        height=450
    )

    st.plotly_chart(fig_cm, use_container_width=True)

# ======================
# PREDICTION TAB
# ======================
with tab_predict:
    st.header("🔮 Predict Video Popularity")

    views = st.number_input("Total Views", min_value=0, step=1, key="views")
    likes = st.number_input("Total Likes", min_value=0, step=1, key="likes")
    comments_count = st.number_input("Total Comments Count", min_value=0, step=1, key="comments_count")

    st.subheader("💬 Enter at least TWO comments")
    cols = st.columns(2)
    comments = []
    for i in range(10):
        with cols[i % 2]:
            comments.append(st.text_input(f"Comment {i+1}", key=f"comment_{i}"))

    col1, col2 = st.columns(2)
    predict_btn = col1.button("🔮 Predict")
    col2.button("🔁 Reset", on_click=lambda: st.session_state.update({"reset": True}))

    if predict_btn:
        valid_comments = [c for c in comments if c.strip()]
        if views == 0 or likes == 0 or comments_count == 0 or len(valid_comments) < 2:
            st.error("⚠️ Please enter valid inputs and at least two comments.")
            st.stop()

        sentiments = [get_raw_sentiment(clean_comment(c)) for c in valid_comments]
        avg_sentiment = np.mean(sentiments)
        sentiment_class = convert_sentiment_to_class(avg_sentiment)

        X = np.array([[views, likes, comments_count, sentiment_class]])
        X_scaled = scaler.transform(X)

        prediction = model.predict(X_scaled)
        pred_class = np.argmax(prediction)

        labels_map = {
            0: ("Low Popularity", "📉"),
            1: ("Medium Popularity", "📊"),
            2: ("High Popularity", "🔥")
        }

        result, emoji = labels_map[pred_class]
        st.success(f"{emoji} **Predicted Popularity: {result}**")

        # Store for insights
        st.session_state.pred_class = pred_class
        st.session_state.avg_sentiment = avg_sentiment
        st.session_state.sentiments = sentiments

    # ======================
    # INSIGHTS & RECOMMENDATIONS
    # ======================
    if "pred_class" in st.session_state:
        st.markdown("---")
        st.header("💡 Insights & Recommendations")

        fig_sent = go.Figure()
        fig_sent.add_bar(
            x=[f"Comment {i+1}" for i in range(len(st.session_state.sentiments))],
            y=st.session_state.sentiments
        )

        fig_sent.update_layout(
            title="Sentiment Score per Comment",
            yaxis_title="Sentiment Score (-1 to +1)",
            height=400
        )

        st.plotly_chart(fig_sent, use_container_width=True)

        st.write(f"**Average Sentiment:** `{st.session_state.avg_sentiment:.3f}`")

        recs = []
        if st.session_state.pred_class == 0:
            recs.append("📉 Improve title, thumbnail, and SEO.")
        elif st.session_state.pred_class == 1:
            recs.append("📊 Increase engagement using CTAs.")
        else:
            recs.append("🔥 Maintain consistency and content quality.")

        if st.session_state.avg_sentiment < -0.25:
            recs.append("😟 Address negative feedback.")
        elif st.session_state.avg_sentiment <= 0.25:
            recs.append("🙂 Add emotional or storytelling elements.")
        else:
            recs.append("🥰 Strong positive reception.")

        for r in recs:
            st.write(r)

# ======================
# CONTACT TAB
# ======================
with tab_contact:
    st.header("📩 Contact & Feedback")

    st.write("Share suggestions to improve this prediction system.")

    with st.form("feedback_form"):
        name = st.text_input("Name")
        email = st.text_input("Email (optional)")
        feedback = st.text_area("Your Feedback / Suggestions")
        submit = st.form_submit_button("Submit")

    if submit:
        if feedback.strip() == "":
            st.warning("⚠️ Please enter feedback.")
        else:
            st.success("✅ Thank you! Your feedback has been received.")





