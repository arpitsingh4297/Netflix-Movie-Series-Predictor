import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page configuration
st.set_page_config(page_title="Netflix Title Predictor", page_icon="🎬", layout="wide")

# Title and description
st.title("🎬 Netflix Title Predictor")
st.markdown("""
This app predicts whether a Netflix title is a **Movie** or **TV Show** based on its features.
Enter the details below and click **Predict** to see the result.
""")

# Load model and preprocessor
try:
    model = joblib.load('best_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
except FileNotFoundError:
    st.error("Model or preprocessor file not found. Please ensure 'best_model.pkl' and 'preprocessor.pkl' are in the app directory.")
    st.stop()

# Define feature input form
st.header("Enter Title Details")
with st.form(key='prediction_form'):
    col1, col2, col3 = st.columns(3)
    
    # Numerical features
    with col1:
        release_year = st.slider("Release Year", min_value=1925, max_value=2025, value=2015, step=1)
        year_added = st.slider("Year Added to Netflix", min_value=2008, max_value=2025, value=2019, step=1)
        month_added = st.slider("Month Added to Netflix", min_value=1, max_value=12, value=6, step=1)
        num_genres = st.slider("Number of Genres", min_value=1, max_value=5, value=2, step=1)
    
    # Categorical features
    with col2:
        rating = st.selectbox("Rating", options=['R', 'TV-14', 'TV-G', 'TV-MA', 'TV-PG', 'TV-Y', 'TV-Y7', 'TV-Y7-FV'], index=3)
        is_international = st.checkbox("International (Non-US)", value=False)
        is_drama = st.checkbox("Drama Genre", value=False)
        is_comedy = st.checkbox("Comedy Genre", value=False)
    
    with col3:
        is_documentary = st.checkbox("Documentary Genre", value=False)
        is_action = st.checkbox("Action Genre", value=False)
        is_horror = st.checkbox("Horror Genre", value=False)
        is_sci_fi = st.checkbox("Sci-Fi Genre", value=False)
        is_romance = st.checkbox("Romance Genre", value=False)
    
    # Submit button
    submit_button = st.form_submit_button(label="Predict")

# Process prediction
if submit_button:
    st.header("Prediction Result")
    
    # Create input DataFrame
    input_data = pd.DataFrame({
        'release_year': [release_year],
        'year_added': [year_added],
        'month_added': [month_added],
        'num_genres': [num_genres],
        'rating': [rating],
        'is_international': [1 if is_international else 0],
        'is_drama': [1 if is_drama else 0],
        'is_comedy': [1 if is_comedy else 0],
        'is_documentary': [1 if is_documentary else 0],
        'is_action': [1 if is_action else 0],
        'is_horror': [1 if is_horror else 0],
        'is_sci_fi': [1 if is_sci_fi else 0],
        'is_romance': [1 if is_romance else 0]
    })
    
    try:
        # Preprocess input
        X_processed = preprocessor.transform(input_data)
        
        # Predict
        prediction = model.predict(X_processed)[0]
        prediction_proba = model.predict_proba(X_processed)[0] if hasattr(model, 'predict_proba') else None
        
        # Display result
        result = "Movie" if prediction == 0 else "TV Show"
        st.success(f"Predicted Title Type: **{result}**")
        
        if prediction_proba is not None:
            st.write(f"Confidence (Movie): **{prediction_proba[0]:.2%}**")
            st.write(f"Confidence (TV Show): **{prediction_proba[1]:.2%}**")
        
        # Display input summary
        with st.expander("Input Summary"):
            st.write(input_data)
    
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Model trained on Netflix dataset")