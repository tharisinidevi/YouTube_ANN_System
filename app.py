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
    st.write("✅ File uploaded!")
    st.write("Columns:", df.columns.tolist())

    # Rename or create missing columns
    rename_columns = {
        'like_count': 'likes',
        'comment_count': 'comments',
        'view_count': 'views',
        'sentiment_rank': 'sentiment_score'
    }
    df.rename(columns=rename_columns, inplace=True)

    if 'comments' not in df.columns:
        df['comments'] = 0  # Default

    required_columns = ['likes', 'comments', 'views', 'sentiment_score']
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        st.error(f"❌ Missing columns: {missing}")
    else:
        if st.button("Predict Popularity"):
            scaler = StandardScaler()
            X = df[required_columns]
            X_scaled = scaler.fit_transform(X)
            y_pred = model.predict(X_scaled)

            if y_pred.shape[1] == 1:
                df['Predicted_Popularity'] = (y_pred.flatten() > 0.5).astype(int)
            else:
                classes = ['Not Popular', 'Average', 'Popular']
                df['Predicted_Popularity'] = [classes[i] for i in y_pred.argmax(axis=1)]

            st.success("✅ Prediction Completed!")
            st.dataframe(df)

