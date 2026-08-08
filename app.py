"""
8D Distinction Task -- Sydney Housing Price Prediction and Decision Support System
Streamlit app: loads the trained pipeline (gb_price_pipeline.joblib) produced by the
project notebook and returns a predicted sale price for user-entered property features.

Run with:  streamlit run app.py
"""
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Sydney Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- styling
st.markdown("""
<style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1100px;}
    .hero {
        background: linear-gradient(135deg, #0f2a4a 0%, #1a4a7a 55%, #2d6ba3 100%);
        border-radius: 14px; padding: 1.6rem 2rem; margin-bottom: 1.4rem;
        color: white;
    }
    .hero h1 {margin: 0; font-size: 1.9rem; color: white;}
    .hero p {margin: 0.35rem 0 0 0; color: #d8e6f4; font-size: 0.95rem;}
    div[data-testid="stMetric"] {
        background: #f4f8fc !important; border: 1px solid #dbe6f0; border-radius: 10px;
        padding: 0.8rem 0.9rem 0.5rem 0.9rem;
    }
    div[data-testid="stMetric"] * {color: #0f2a4a !important;}
    div[data-testid="stMetricLabel"] p {color: #4a6280 !important; font-weight: 600;}
    div[data-testid="stMetricValue"] {color: #0f2a4a !important;}
    div[data-testid="stMetricDeltaIcon-Up"] + div,
    div[data-testid="stMetricDelta"] {color: #0d7a3f !important;}
    div[data-testid="stMetricDelta"] svg {fill: #0d7a3f !important;}
    .price-card {
        background: linear-gradient(135deg, #103a63, #1c5a94);
        border-radius: 14px; padding: 1.4rem 1.8rem; color: white; text-align: center;
        margin-bottom: 0.8rem;
    }
    .price-card .label {font-size: 0.85rem; letter-spacing: 0.05em; text-transform: uppercase; color: #bcd6ef;}
    .price-card .value {font-size: 2.6rem; font-weight: 700; margin: 0.15rem 0;}
    .price-card .sub {font-size: 0.85rem; color: #cfe1f2;}
    .footer-note {color: #8a97a6; font-size: 0.8rem;}
    .stButton>button {
        background: #1c5a94; color: white; border-radius: 8px; border: none;
        font-weight: 600; padding: 0.55rem 1rem;
    }
    .stButton>button:hover {background: #124274; color: white;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    return joblib.load("gb_price_pipeline.joblib")


@st.cache_data
def load_context():
    return pd.read_csv("suburb_price_context.csv", index_col=0)


pipeline = load_pipeline()
suburb_context = load_context()

SUBURB_DIST = {"Mosman": 8, "Marrickville": 7, "Bankstown": 17}
SUBURB_BLURB = {
    "Mosman": "Premium harbourside, lower north shore",
    "Marrickville": "Gentrified inner-west, terraces & apartments",
    "Bankstown": "Affordable outer south-west, transport hub",
}
DWELLING_TYPES = ["House", "Townhouse/Semi", "Unit"]

# --------------------------------------------------------------------------- hero
st.markdown("""
<div class="hero">
    <h1>🏠 Sydney Housing Price Predictor</h1>
    <p>Decision-support prototype &middot; 8D Distinction Task &middot; trained on 103 real sold properties
    across Mosman, Marrickville and Bankstown (Domain.com.au, 2026)</p>
</div>
""", unsafe_allow_html=True)

tab_predict, tab_about = st.tabs(["🔮 Predict a price", "ℹ️ About this model"])

# --------------------------------------------------------------------------- PREDICT TAB
with tab_predict:
    col_form, col_result = st.columns([1, 1.3], gap="large")

    with col_form:
        st.subheader("Property details")
        with st.form("predict_form"):
            suburb = st.selectbox("Suburb", list(SUBURB_DIST.keys()))
            st.caption(SUBURB_BLURB[suburb])
            dwelling_type = st.selectbox("Dwelling type", DWELLING_TYPES)

            b1, b2, b3 = st.columns(3)
            beds = b1.number_input("Bedrooms", min_value=0, max_value=10, value=3, step=1)
            baths = b2.number_input("Bathrooms", min_value=0, max_value=10, value=2, step=1)
            parking = b3.number_input("Car spaces", min_value=0, max_value=10, value=1, step=1)

            has_land = st.checkbox("Land size known?", value=(dwelling_type != "Unit"))
            land_size = st.slider(
                "Land size (m²)", min_value=0, max_value=1000, value=250, step=10,
                disabled=not has_land,
            )
            submitted = st.form_submit_button("Predict sale price ➜", width="stretch")

    with col_result:
        st.subheader("Estimated sale price")
        if submitted:
            row = pd.DataFrame([{
                "beds": beds,
                "baths": baths,
                "parking": parking,
                "land_size_sqm": land_size if has_land else np.nan,
                "distance_to_cbd_km": SUBURB_DIST[suburb],
                "suburb": suburb,
                "dwelling_type": dwelling_type,
            }])
            pred_price = float(np.exp(pipeline.predict(row)[0]))
            ctx = suburb_context.loc[suburb]

            st.markdown(f"""
            <div class="price-card">
                <div class="label">Predicted sale price</div>
                <div class="value">${pred_price:,.0f}</div>
                <div class="sub">{beds}-bed {dwelling_type.lower()} &middot; {suburb}</div>
            </div>
            """, unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Suburb min", f"${ctx['min']:,.0f}")
            m2.metric("Suburb median", f"${ctx['median']:,.0f}", delta=f"{(pred_price/ctx['median']-1)*100:+.0f}% vs. this pred.")
            m3.metric("Suburb max", f"${ctx['max']:,.0f}")

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=["Suburb min", "Suburb median", "Your prediction", "Suburb max"],
                y=[ctx["min"], ctx["median"], pred_price, ctx["max"]],
                marker_color=["#a9c4de", "#5f8fbd", "#e8863c", "#a9c4de"],
                text=[f"${v:,.0f}" for v in [ctx["min"], ctx["median"], pred_price, ctx["max"]]],
                textposition="outside",
            ))
            fig.update_layout(
                height=320, margin=dict(l=10, r=10, t=20, b=10),
                yaxis_title="Price ($)", showlegend=False,
                plot_bgcolor="white", paper_bgcolor="white",
            )
            st.plotly_chart(fig, width="stretch")

            st.info(
                "⚠️ This is a statistical estimate from a small (103-property) training sample, "
                "not a professional valuation. It is least reliable for atypical properties "
                "(unusually large/small land, missing land size, or features outside the training "
                "range) -- see Part 4 of the accompanying report for where predictions should be "
                "treated cautiously."
            )
        else:
            st.markdown(
                "<div style='padding:2.5rem 1rem; text-align:center; color:#5a6b80; "
                "background:#f4f8fc; border:1px dashed #b9cadd; border-radius:12px;'>"
                "Fill in the property details and click <b>Predict sale price</b> to see an estimate here."
                "</div>", unsafe_allow_html=True
            )

# --------------------------------------------------------------------------- ABOUT TAB
with tab_about:
    st.subheader("About this model")
    st.write(
        "Trained on 103 real sold properties collected from Domain.com.au's sold-listings search "
        "(Mosman, Marrickville, Bankstown; August 2026 snapshot). Model: **Gradient Boosting Regressor** "
        "on log(price), selected ahead of Ridge Regression and Random Forest based on 5-fold "
        "cross-validated MAE, RMSE, MAPE and R² -- see Part 3 of the accompanying report."
    )
    display_context = suburb_context.rename(columns={"median": "Median", "min": "Min", "max": "Max"}).copy()
    for col in display_context.columns:
        display_context[col] = display_context[col].map(lambda v: f"${v:,.0f}")
    st.dataframe(display_context, width="stretch")
    st.caption(
        "Full methodology, error analysis, ML-vs-LLM-vs-human comparison and ethical reflection are in "
        "the accompanying PDF report and Jupyter notebook."
    )

st.markdown(
    "<p class='footer-note'>8D Distinction Task &middot; Sydney Housing Price Prediction and Decision "
    "Support System &middot; Anant Kumar Tripathi</p>",
    unsafe_allow_html=True,
)
