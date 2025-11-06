import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
import os

st.set_page_config(page_title="YouTube Popularity Predictor", layout="wide")

# --- CONFIG: change these to match how you trained the model ---
MODEL_PATH = "model/ann_model.h5"
SCALER_PATH = "model/scaler.pkl"   # if you saved it during training
# Define feature names in the exact order used to train the model:
EXPECTED_FEATURES = ["views", "likes", "comments", "shares", "sentiment_score"]
# If your model is multi-class, set class labels in the correct order used for training:
CLASS_LABELS = ["Low", "Medium", "High"]
# ----------------------------------------------------------------

# Helper: load model
if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found at {MODEL_PATH}. Please upload the model file.")
    st.stop()

model = load_model(MODEL_PATH)

# Try to load scaler if available
scaler = None
if os.path.exists(SCALER_PATH):
    try:
        scaler = joblib.load(SCALER_PATH)
        st.info("Loaded saved scaler from model/scaler.pkl")
    except Exception as e:
        st.warning("Could not load scaler.pkl (it may be incompatible). The app will scale using a newly fitted scaler (less ideal).")

st.title("🎬 YouTube Video Popularity Prediction")

uploaded_file = st.file_uploader("Upload your YouTube dataset (.csv)", type=["csv"])
if not uploaded_file:
    st.info("Please upload a CSV file that contains the features your model expects.")
    st.stop()

# Safely read CSV with fallback encodings
try:
    df = pd.read_csv(uploaded_file, encoding="utf-8")
except Exception:
    try:
        df = pd.read_csv(uploaded_file, encoding="latin-1", engine="python")
    except Exception as e:
        st.error(f"Could not read uploaded file: {e}")
        st.stop()

st.subheader("Preview of uploaded data")
st.dataframe(df.head())

st.write("Columns found in your file:", df.columns.tolist())

# Try to auto-rename common column variants to expected names (non-destructive)
rename_map = {
    "view_count": "views", "View": "views", "Views": "views",
    "like_count": "likes", "Like": "likes", "Likes": "likes",
    "comment_count": "comments", "comment": "comments", "Comment": "comments", "Comments": "comments",
    "share_count": "shares", "Share": "shares", "Shares": "shares",
    "sentiment": "sentiment_score", "sentiment_polarity": "sentiment_score", "compound": "sentiment_score"
}
# Only rename keys that actually exist in df to avoid creating new unexpected cols
to_rename = {k:v for k,v in rename_map.items() if k in df.columns}
if to_rename:
    df = df.rename(columns=to_rename)
    st.success(f"Auto-renamed columns: {to_rename}")

# Now check which expected features are present
present = [f for f in EXPECTED_FEATURES if f in df.columns]
missing = [f for f in EXPECTED_FEATURES if f not in df.columns]

st.write(f"Features expected by model (in order): {EXPECTED_FEATURES}")
st.write(f"Present columns matched: {present}")
if missing:
    st.warning(f"Missing features required by your model: {missing}")
    # Give user choice: either create defaults or stop
    create_defaults = st.checkbox("Create missing features with default values (not recommended unless you know what you're doing)", value=False)
    if create_defaults:
        for c in missing:
            # choose sensible default for numeric features
            df[c] = 0
        st.info(f"Default columns added for: {missing}")
        missing = []  # now considered present
    else:
        st.stop()

# Ensure the feature order is exactly as EXPECTED_FEATURES
X = df[EXPECTED_FEATURES].copy()

# If scaler was loaded, use it. Otherwise, fit a new scaler on X but warn user
if scaler is not None:
    try:
        X_scaled = scaler.transform(X)
    except Exception as e:
        st.warning("Saved scaler couldn't transform the uploaded data (shape mismatch). A new scaler will be fit on uploaded data (this may harm accuracy).")
        scaler = None

if scaler is None:
    st.warning("No compatible saved scaler found. Fitting a new StandardScaler on uploaded data (this is not recommended for final evaluation — better to save scaler used during training).")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

# Predict
y_pred = model.predict(X_scaled)

# Handle binary vs multi-class outputs
if y_pred.ndim == 1 or (y_pred.ndim == 2 and y_pred.shape[1] == 1):
    # Binary output (probability)
    probs = y_pred.flatten()
    predicted_numeric = (probs > 0.5).astype(int)
    # If you had class labels for 0/1 mapping, adjust here. We'll show 0/1:
    df["Predicted_Label"] = predicted_numeric
    df["Predicted_Probability"] = probs
else:
    # Multi-class probabilities
    idx = np.argmax(y_pred, axis=1)
    # If CLASS_LABELS length matches model output width, map to labels; otherwise map to indices
    if y_pred.shape[1] == len(CLASS_LABELS):
        df["Predicted_Label"] = [CLASS_LABELS[i] for i in idx]
    else:
        df["Predicted_Label"] = idx  # fallback: numeric class index
    # also store top probability
    df["Predicted_Prob"] = y_pred.max(axis=1)

st.success("✅ Prediction finished")
st.dataframe(df.head(20))

# Allow download of results
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download results (CSV)", csv, "predictions.csv", "text/csv")


