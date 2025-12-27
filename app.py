# =========================================
# app.py — FINAL SUBMISSION VERSION
# =========================================

import streamlit as st
import numpy as np
import joblib
import re
import smtplib
from email.mime.text import MIMEText
from tensorflow.keras.models import load_model
from textblob import TextBlob
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="YouTube Popularity Predictor",
    page_icon="🎬",
    layout="centered"
)
import streamlit as st

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        
st.set_page_config(
    page_title="YouTube Popularity Predictor",
    page_icon="🎬",
    layout="centered"
)

local_css("style.css")


# ======================
# LOAD MODEL & SCALER
# ======================
model = load_model("model/youtube_popularity_ann.h5")
scaler = joblib.load("model/scaler.pkl")


# ======================
# SENTIMENT SETUP
# ======================
def clean_comment(text):
    text = re.sub(r"http\S+|www\.\S+", "", text)
    return re.sub(r"\s+", " ", text).strip()

def get_raw_sentiment(text):
    return TextBlob(text).sentiment.polarity

def convert_sentiment_to_class(score):
    if score < -0.25:
        return 0
    elif score <= 0.25:
        return 1
    else:
        return 2

# ======================
# TABS
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
    This system predicts YouTube video popularity using:
    - Engagement metrics (views, likes, comments)
    - Viewer sentiment from comments
    - An Artificial Neural Network (ANN)
    """)

    st.subheader("📊 Model Performance")

    X_test = joblib.load("model/X_test.pkl")
    y_test = joblib.load("model/y_test.pkl")

    X_test_scaled = scaler.transform(X_test)
    y_pred = np.argmax(model.predict(X_test_scaled), axis=1)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{acc*100:.2f}%")
    c2.metric("Precision", f"{prec*100:.2f}%")
    c3.metric("Recall", f"{rec*100:.2f}%")

    st.subheader("📌 Confusion Matrix")

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

    views = st.number_input("Total Views", min_value=0, key="views")
    likes = st.number_input("Total Likes", min_value=0, key="likes")
    comments_count = st.number_input("Total Comments", min_value=0, key="comments_count")

    st.markdown("### 💬 Enter at least TWO comments")
    cols = st.columns(2)
    comment_inputs = []
    for i in range(10):
        with cols[i % 2]:
            comment_inputs.append(st.text_input(f"Comment {i+1}", key=f"comment_{i}"))

    col1, col2 = st.columns(2)
    predict_btn = col1.button("🔮 Predict")
    reset_btn = col2.button("🔁 Reset")

    # ---------- RESET ----------
    if reset_btn:
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    # ---------- PREDICT ----------
    if predict_btn:
        if views == 0 or likes == 0 or comments_count == 0:
            st.error("⚠️ Please enter Views, Likes, and Comments together.")
            st.stop()

        valid_comments = [c for c in comment_inputs if c.strip()]
        if len(valid_comments) < 2:
            st.error("⚠️ Please enter at least TWO comments.")
            st.stop()

        sentiments = [get_raw_sentiment(clean_comment(c)) for c in valid_comments]
        avg_sentiment = np.mean(sentiments)
        sentiment_class = convert_sentiment_to_class(avg_sentiment)

        X = scaler.transform([[views, likes, comments_count, sentiment_class]])
        pred = np.argmax(model.predict(X))

        labels = ["📉 Low", "📊 Medium", "🔥 High"]
        st.success(f"Predicted Popularity: **{labels[pred]}**")

        # ---------- INSIGHTS ----------
        st.subheader("📊 Sentiment Analysis")
        fig = go.Figure(go.Bar(
            x=[f"C{i+1}" for i in range(len(sentiments))],
            y=sentiments
        ))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("💡 Recommendations")

        if views < 1000:
            st.write("👀 Increase reach with SEO-friendly titles and thumbnails.")
        if likes / max(views, 1) < 0.03:
            st.write("👍 Encourage likes with call-to-actions.")
        if comments_count < 50:
            st.write("💬 Ask questions to boost comments.")
        if avg_sentiment < -0.25:
            st.write("😟 Address negative feedback in future videos.")
        elif avg_sentiment > 0.25:
            st.write("🥰 Strong positive audience reception!")

# ======================
# CONTACT TAB (EMAIL)
# ======================
with tab_contact:
    st.header("📩 Contact & Feedback")

    with st.form("feedback_form"):
        name = st.text_input("Name")
        email = st.text_input("Your Email (optional)")
        feedback = st.text_area("Your Feedback")
        submit = st.form_submit_button("Send")

    if submit:
        if feedback.strip() == "":
            st.warning("⚠️ Feedback cannot be empty.")
        else:
            msg = MIMEText(
                f"Name: {name}\nEmail: {email}\n\nFeedback:\n{feedback}"
            )
            msg["Subject"] = "YouTube Popularity App Feedback"
            msg["From"] = st.secrets["EMAIL_ADDRESS"]
            msg["To"] = st.secrets["EMAIL_ADDRESS"]

            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(
                    st.secrets["EMAIL_ADDRESS"],
                    st.secrets["EMAIL_PASSWORD"]
                )
                server.send_message(msg)

            st.success("✅ Feedback sent successfully!")











