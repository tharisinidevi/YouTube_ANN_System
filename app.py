import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib   # To load the scaler

# Load trained model and scaler
model = load_model('model/ann_model.h5')
scaler = joblib.load('model/scaler.pkl')   # Make sure you saved scaler during training

st.title("🎬 YouTube Video Popularity Prediction (ANN System)")

uploaded_file = st.file_uploader("Upload your YouTube dataset (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Uploaded Data Preview")
    st.dataframe(df.head())

    try:
        if st.button("Predict Popularity"):
            # ✅ Match features used during training
            X = df[['views', 'likes', 'comments', 'sentiment_score']]  # Change if needed

            # ✅ Use SAME SCALER used in training
            X_scaled = scaler.transform(X)

            # ✅ Predict
            y_pred = model.predict(X_scaled)

            # ✅ If multi-class (3 outputs)
            df['Predicted_Popularity'] = np.argmax(y_pred, axis=1)

            # ✅ Show results
            st.success("✅ Prediction completed!")
            st.dataframe(df)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

