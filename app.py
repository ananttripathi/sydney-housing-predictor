"""
Sydney Housing Price Prediction - Decision Support App
--------------------------------------------------------
A simple Streamlit app that loads the trained Random Forest pipeline
(house_price_model.joblib, produced by the project notebook) and lets a
user enter property characteristics to get a predicted sale price for
Mosman, Parramatta, or Liverpool.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deploy for free on Streamlit Community Cloud:
    1. Push this folder (app.py, requirements.txt, house_price_model.joblib)
       to a public GitHub repository.
    2. Go to https://share.streamlit.io , sign in with GitHub, and click
       "New app".
    3. Select the repository/branch and set the main file to `app.py`.
    4. Click Deploy - Streamlit Cloud installs requirements.txt and hosts
       the app at a public https://<yourapp>.streamlit.app URL.
"""

import datetime as dt

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Sydney Housing Price Predictor", page_icon="\U0001F3E0", layout="centered")

MODEL_PATH = "house_price_model.joblib"
TRAIN_START_DATE = dt.date(2018, 5, 26)  # earliest sale date in the training data; used to compute days_since_start

SUBURB_STATION_DISTANCE = {
    # Approximate median distance-to-nearest-station (metres) observed in the training data,
    # used as a sensible default so a non-technical user doesn't need to know this figure.
    "Mosman": 6800,
    "Parramatta": 19400,
    "Liverpool": 27000,
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def main():
    st.title("Sydney Housing Price Predictor")
    st.caption(
        "Decision-support prototype for the 8D ML Mini Project - Random Forest model trained on "
        "201 completed Domain.com.au sales (2018-2020) across Mosman, Parramatta and Liverpool."
    )
    st.warning(
        "This model was trained on a small, historical (2018-2020) sample for a coursework project. "
        "Predictions are illustrative only and should not be used for real financial decisions.",
        icon="⚠️",
    )

    model = load_model()

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            suburb = st.selectbox("Suburb", ["Mosman", "Parramatta", "Liverpool"])
            property_type = st.selectbox("Property type", ["House", "Unit", "Townhouse"])
            bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3, step=1)
        with col2:
            bathrooms = st.number_input("Bathrooms", min_value=1, max_value=6, value=2, step=1)
            carspaces = st.number_input("Car spaces", min_value=0, max_value=8, value=1, step=1)
            sale_date = st.date_input("Sale / valuation date", value=dt.date(2020, 6, 1),
                                       min_value=TRAIN_START_DATE, max_value=dt.date(2020, 12, 31))

        distance_default = SUBURB_STATION_DISTANCE[suburb]
        distance_to_station = st.slider(
            "Distance to nearest train station (metres)",
            min_value=0, max_value=30000, value=int(distance_default), step=100,
            help="Defaults to the typical distance observed for this suburb in the training data.",
        )

        submitted = st.form_submit_button("Predict sale price")

    if submitted:
        days_since_start = (sale_date - TRAIN_START_DATE).days
        X = pd.DataFrame([{
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "carspaces": carspaces,
            "central_station": distance_to_station,
            "days_since_start": days_since_start,
            "suburb": suburb,
            "property_type_grouped": property_type,
        }])
        log_pred = model.predict(X)[0]
        pred_price = float(np.expm1(log_pred))

        st.success(f"### Predicted sale price: ${pred_price:,.0f}")
        st.caption(
            "Note: our evaluation found this model's predictions are far less reliable for atypical, "
            "premium, or renovated properties (see Part 4 of the project report) - treat this estimate "
            "as a rough, data-driven starting point, not a valuation."
        )

        with st.expander("See the exact features sent to the model"):
            st.dataframe(X)


if __name__ == "__main__":
    main()
