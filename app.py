# =========================================
# app.py — FINAL CLEAN VERSION WITH TABS
# =========================================

import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from textblob import TextBlob
import re
import plotly.graph_objects as go


# ======================
# Load Local CSS
# ======================
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(
    page_title="YouTube Popularity Predictor",
    page_icon="🎬",
    layout="centered"
)

# Load custom CSS
local_css("style.css")


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
MODEL_PATH = "model/youtube_popularity_ann.h5"
SCALER_PATH = "model/scaler.pkl"

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


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
# Page Config
# ======================
st.set_page_config(
    page_title="YouTube Popularity Predictor",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 YouTube Video Popularity Prediction")
st.markdown("---")


# ======================
# Tabs
# ======================
tab_home, tab_predict, tab_performance, tab_insights = st.tabs(
    ["🏠 Home", "🔮 Prediction", "📊 Model Performance", "💡 Insights & Recommendations"]
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
        return 0  # Negative
    elif score <= 0.25:
        return 1  # Neutral
    else:
        return 2  # Positive


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
    st.write("""
    - 📉 Low Popularity  
    - 📊 Medium Popularity  
    - 🔥 High Popularity
    """)

    st.info("➡️ Navigate to the **Prediction** tab to test a video.")


# ======================
# PREDICTION TAB
# ======================
with tab_predict:
    st.header("🔮 Predict Video Popularity")

    st.subheader("📊 Enter Engagement Metrics")
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
                st.text_input(f"Comment {i + 1}", key=f"comment_{i}")
            )

    st.markdown("---")
    col1, col2 = st.columns(2)
    predict_btn = col1.button("🔮 Predict Popularity")
    col2.button("🔁 Reset", on_click=lambda: st.session_state.update({"reset": True}))

    if predict_btn:

        if views == 0 or likes == 0 or comments_count == 0:
            st.error("⚠️ Please enter Views, Likes, and Comments Count.")
            st.stop()

        valid_comments = [c for c in comment_inputs if c.strip() != ""]
        if len(valid_comments) < 2:
            st.error("⚠️ Enter at least TWO non-empty comments.")
            st.stop()

        sentiments = []
        for c in valid_comments:
            c_clean = clean_comment(c)
            if c_clean:
                sentiments.append(get_raw_sentiment(c_clean))

        avg_sentiment = np.mean(sentiments)
        sentiment_class = convert_sentiment_to_class(avg_sentiment)

        X = np.array([[views, likes, comments_count, sentiment_class]])
        X_scaled = scaler.transform(X)

        prediction = model.predict(X_scaled)
        pred_class = np.argmax(prediction)
        confidence = np.max(prediction)

        labels = {
            0: ("Low Popularity", "📉"),
            1: ("Medium Popularity", "📊"),
            2: ("High Popularity", "🔥")
        }

        result_text, emoji = labels[pred_class]

        st.success(f"{emoji} **Predicted Popularity: {result_text}**")
        #st.write(f"Prediction Confidence: **{confidence * 100:.2f}%**")

        # Store results for Insights tab
        st.session_state.pred_class = pred_class
        st.session_state.result_text = result_text
        st.session_state.avg_sentiment = avg_sentiment
        st.session_state.sentiment_class = sentiment_class


# ======================
# MODEL PERFORMANCE TAB
# ======================
with tab_performance:
    st.header("📊 Model Performance Evaluation")

    st.write("""
    This section evaluates the ANN model using a held-out test dataset.
    Performance metrics are calculated based on true labels and predicted outputs.
    """)

    # ======================
    # Load Test Data
    # ======================
    try:
        X_test = joblib.load("model/X_test.pkl")
        y_test = joblib.load("model/y_test.pkl")
    except FileNotFoundError:
        st.error("❌ Test dataset not found. Please ensure 'X_test.pkl' and 'y_test.pkl' exist in the 'model' folder.")
        st.stop()

    # ======================
    # Scale Test Data
    # ======================
    X_test_scaled = scaler.transform(X_test)

    # ======================
    # Predict
    # ======================
    y_pred_prob = model.predict(X_test_scaled)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # ======================
    # Compute Metrics
    # ======================
    from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{acc*100:.2f}%")
    col2.metric("Precision", f"{prec*100:.2f}%")
    col3.metric("Recall", f"{rec*100:.2f}%")

    st.markdown("---")

    # ======================
    # Confusion Matrix
    # ======================
    st.subheader("📌 Confusion Matrix")

    import plotly.figure_factory as ff

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

    st.info("""
    ✅ These metrics and the confusion matrix illustrate that incorporating sentiment features
    along with engagement metrics improves the ANN model’s ability to correctly classify video popularity levels.
    """)



# ======================
# INSIGHTS & RECOMMENDATIONS TAB
# ======================
with tab_insights:
    st.header("💡 Insights & Recommendations")

    if "pred_class" not in st.session_state:
        st.warning("⚠️ Run a prediction first to view insights and plots.")
        st.stop()

    # =========================
    # Retrieve stored values
    # =========================
    pred_class = st.session_state.pred_class
    avg_sentiment = st.session_state.avg_sentiment
    sentiment_class = st.session_state.sentiment_class

    # =========================
    # 1️⃣ SENTIMENT DISTRIBUTION PLOT
    # =========================
    st.subheader("📊 Sentiment Distribution of Viewer Comments")

    # Recalculate sentiments for visualization
    sentiment_scores = []
    for i in range(10):
        key = f"comment_{i}"
        if key in st.session_state and st.session_state[key].strip():
            cleaned = clean_comment(st.session_state[key])
            if cleaned:
                sentiment_scores.append(get_raw_sentiment(cleaned))

    if sentiment_scores:
        fig_sent = go.Figure()
        fig_sent.add_trace(go.Bar(
            x=[f"Comment {i+1}" for i in range(len(sentiment_scores))],
            y=sentiment_scores
        ))

        fig_sent.update_layout(
            title="Sentiment Score per Comment",
            yaxis_title="Sentiment Score (-1 to +1)",
            xaxis_title="Comments",
            height=400
        )

        st.plotly_chart(fig_sent, use_container_width=True)
    else:
        st.info("No sentiment data available for visualization.")

    st.write(f"**Average Sentiment Score:** `{avg_sentiment:.3f}`")

    # =========================
    # 2️⃣ PREDICTION PROBABILITY BAR CHART
    # =========================
    #st.subheader("📊 Popularity Prediction Confidence")

    #popularity_labels = ["Low Popularity", "Medium Popularity", "High Popularity"]
    #probabilities = st.session_state.get("prediction_probs", None)

    #if probabilities is not None:
        #fig_prob = go.Figure()
        #fig_prob.add_trace(go.Bar(
            #x=popularity_labels,
            #y=probabilities,
            #text=[f"{p*100:.1f}%" for p in probabilities],
            #textposition="auto"
        #))

        #fig_prob.update_layout(
            #title="ANN Prediction Probability Distribution",
            #yaxis_title="Probability",
            #xaxis_title="Popularity Level",
            #height=400
        #)

        #st.plotly_chart(fig_prob, use_container_width=True)
    #else:
        #st.info("Prediction probability data not found.")

    # =========================
    # 3️⃣ ENGAGEMENT METRICS COMPARISON
    # =========================
    st.subheader("📊 Engagement Metrics Comparison (Normalized)")

    # Normalization (simple max-based)
    views = st.session_state.views
    likes = st.session_state.likes
    comments_count = st.session_state.comments_count

    max_val = max(views, likes, comments_count, 1)

    engagement_vals = [
        views / max_val,
        likes / max_val,
        comments_count / max_val
    ]

    fig_eng = go.Figure()
    fig_eng.add_trace(go.Bar(
        x=["Views", "Likes", "Comments"],
        y=engagement_vals
    ))

    fig_eng.update_layout(
        title="Relative Strength of Engagement Metrics",
        yaxis_title="Normalized Value",
        height=400
    )

    st.plotly_chart(fig_eng, use_container_width=True)

    # =========================
    # 4️⃣ RECOMMENDATIONS
    # =========================
    st.subheader("📌 Actionable Recommendations")

    recommendations = []

    if pred_class == 0:
        recommendations.append("📉 Low popularity detected — improve thumbnails, titles, and SEO.")
    elif pred_class == 1:
        recommendations.append("📊 Moderate popularity — boost engagement using calls-to-action.")
    else:
        recommendations.append("🔥 High popularity — maintain consistency and content quality.")

    if sentiment_class == 0:
        recommendations.append("😟 Negative sentiment — review viewer feedback and improve content clarity.")
    elif sentiment_class == 1:
        recommendations.append("🙂 Neutral sentiment — add emotional appeal or storytelling.")
    else:
        recommendations.append("🥰 Positive sentiment — excellent audience reception, keep it up!")

    for rec in recommendations:
        st.write(rec)




