import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load model
model = load_model('model/ann_model.h5')

st.title("🎬 YouTube Video Popularity Prediction")

uploaded_file = st.file_uploader("Upload your YouTube dataset (CSV)", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    if st.button("Predict Popularity"):
        # ✅ Adjust column names according to your dataset
        df.rename(columns={
            'like_count': 'likes',
            'comment_count': 'comments',
            'share_count': 'shares'
        }, inplace=True)

        # ✅ If sentiment is missing
        if 'sentiment_score' not in df.columns:
            df['sentiment_score'] = 0.5

        X = df[['likes', 'comments', 'shares', 'sentiment_score']]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        y_pred = model.predict(X_scaled)

        df['Predicted_Popularity'] = (y_pred > 0.5).astype(int)
        st.success("✅ Prediction Completed!")
        st.dataframe(df)



