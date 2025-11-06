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
            'likes': 'likes',
            'comment_count': 'comments',
            'views': 'views'
        }, inplace=True)

        # ✅ If sentiment is missing
        if 'sentiment_rank' not in df.columns:
            df['sentiment_rank'] = 0.5

        X = df[['likes', 'comment_count', 'views', 'sentiment_rank']]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        y_pred = model.predict(X_scaled)

        df['Predicted_Popularity'] = (y_pred > 0.5).astype(int)
        st.success("✅ Prediction Completed!")
        st.dataframe(df)




