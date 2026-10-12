
import os

import requests
import streamlit as st

# FastAPI backend URL
API_URL = os.getenv(
    "IRIS_API_URL",
    "http://127.0.0.1:8000"
).rstrip("/")

st.set_page_config(
    page_title="Iris Flower Predictor",
    page_icon="🌸",
    layout="centered",
)

# Styling
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eef7ff, #f4f0ff);
}
.main-title {
    text-align: center;
    color: #4338ca;
    font-size: 2.5rem;
    font-weight: 800;
}
.subtitle {
    text-align: center;
    color: #555;
    margin-bottom: 2rem;
}
.result-card {
    padding: 1.3rem;
    border-radius: 15px;
    background: white;
    border: 1px solid #ddd;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">🌸 Iris Flower Predictor</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">'
    'Predict an Iris flower species using your trained neural network.'
    '</div>',
    unsafe_allow_html=True,
)

st.subheader("Enter flower measurements")

with st.form("iris_prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        sepal_length = st.number_input(
            "Sepal length (cm)",
            min_value=0.1,
            max_value=10.0,
            value=5.1,
            step=0.1,
        )
        petal_length = st.number_input(
            "Petal length (cm)",
            min_value=0.1,
            max_value=10.0,
            value=1.4,
            step=0.1,
        )

    with col2:
        sepal_width = st.number_input(
            "Sepal width (cm)",
            min_value=0.1,
            max_value=10.0,
            value=3.5,
            step=0.1,
        )
        petal_width = st.number_input(
            "Petal width (cm)",
            min_value=0.1,
            max_value=10.0,
            value=0.2,
            step=0.1,
        )

    submitted = st.form_submit_button(
        "🔍 Predict Iris Species",
        use_container_width=True,
    )

if submitted:
    payload = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
    }

    try:
        with st.spinner("Analyzing flower measurements..."):
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()

        species = result["predicted_species"]
        confidence = result["confidence"]

        st.success("Prediction completed successfully!")

        st.markdown(
            f'<div class="result-card">'
            f'<h2>🌼 {species.title()}</h2>'
            f'<p>Model confidence</p>'
            f'<h1>{confidence:.2%}</h1>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.subheader("Prediction probabilities")

        probabilities = result.get("probabilities", {})
        if probabilities:
            st.bar_chart(probabilities)

    except requests.exceptions.RequestException:
        st.error(
            "Could not connect to the prediction API. "
            "Make sure your FastAPI backend is running."
        )
    except (KeyError, ValueError, TypeError):
        st.error("The API returned an unexpected prediction response.")

st.divider()
st.caption("Powered by Streamlit, FastAPI, and your trained neural network.")

