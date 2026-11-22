"""
AIPI 510: Module 3 Project: Data/ML Engineering
- frontend/app.py
Hasan Al-Quaid
=================================================
Streamlit front end for the diamond price model:
  1. The user describes a diamond with sliders
  2. The app sends it to the deployed API's /predict endpoint (on Cloud Run)
  3. The app shows the predicted price

The app never loads the model itself; all predictions come from the API.
"""
import os
import requests
import streamlit as st

# global values
# the deployed API on Cloud Run (set API_URL to use a different one, like http://localhost:8080)
API_URL = os.getenv("API_URL", "https://diamond-api-168452878253.us-east1.run.app")

# allowed values, ordered worst to best (same as config.yaml and the API)
CUTS = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLORS = ["J", "I", "H", "G", "F", "E", "D"]
CLARITIES = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]


def get_diamond():
    """
    Show the sliders and collect the diamond's details

    Returns:
        dict: carat, cut, color, clarity, depth, and table
    """
    carat = st.slider("Carat (weight)", 0.2, 5.0, 1.0, 0.01)

    col1, col2, col3 = st.columns(3)
    cut = col1.select_slider("Cut", CUTS, value = "Ideal")
    color = col2.select_slider("Color (J = worst, D = best)", COLORS, value = "G")
    clarity = col3.select_slider("Clarity (I1 = worst, IF = best)", CLARITIES, value = "VS2")

    # depth and table barely affect price, so they're hidden and start at typical values
    with st.expander("Advanced: proportions"):
        depth = st.slider("Depth %", 50.0, 75.0, 61.8, 0.1)
        table = st.slider("Table %", 45.0, 75.0, 57.0, 0.5)

    return {"carat": carat, "cut": cut, "color": color, "clarity": clarity, "depth": depth, "table": table}


def get_prediction(diamond):
    """
    Send the diamond to the API and return the predicted price

    Parameters:
        diamond (dict): The diamond's details from get_diamond()

    Returns:
        float: Predicted price in USD
    """
    # timeout is 60 seconds, in case Cloud Run needs time to wake up
    response = requests.post(f"{API_URL}/predict", json = diamond, timeout = 60)
    response.raise_for_status()   
    return response.json()["predicted_price"]


def main():
    st.title("💎 Diamond Price Predictor")
    st.write("Describe a diamond and the model will estimate its price down below.")

    diamond = get_diamond()

    if st.button("Predict price", type="primary"):
        try:
            with st.spinner("Asking the model..."):
                price = get_prediction(diamond)
            st.metric("Estimated price", f"${price:,.0f}")
        except Exception as e:
            st.error(f"Could not get a prediction: {e}")


if __name__ == "__main__":
    main()
