import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load Model
model = load_model('model/ann_model.h5')

st.title("🎬 YouTube Video Popularity Prediction System")

uploaded_file = st.file_uploader("Upload a YouTube CSV dataset", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("✅ File uploaded successfully!")
    st.write("📌 Columns detected in your dataset:", df.columns.tolist())

    # Rename dataset columns to match what model expects
    rename_columns = {
        'like_count': 'likes',
        'Likes': 'likes',
        'comment_count': 'comments',
        
        'view_count': 'views',
        'Views': 'views',
        'sentiment': 'sentiment_score',
        'sentiment_rank': 'sentiment_score',
        'compound': 'sentiment_score'
    }
    df.rename(columns=rename_columns, inplace=True)

    # Define the columns your model requires
    required_columns = ['likes', 'comments', 'views', 'sentiment_score']

    # Check missing columns
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        st.error(f"❌ Missing columns in your dataset: {missing}")
    else:
        if st.button("Predict Popularity"):
            scaler = StandardScaler()
            X = df[required_columns]
            X_scaled = scaler.fit_transform(X)

            # Predict
            y_pred = model.predict(X_scaled)
            df['Predicted_Popularity'] = (y_pred > 0.5).astype(int)

            st.success("✅ Prediction Completed!")
            st.dataframe(df)





