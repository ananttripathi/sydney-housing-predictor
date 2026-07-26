# Sydney Housing Price Prediction and Decision Support System

8D Distinction Task — ML Mini Project. Predicts residential sale prices in
Mosman, Parramatta and Liverpool (Sydney) from basic property
characteristics, using a Random Forest model trained on 201 real,
completed Domain.com.au auction/sale results (2018–2020).

## Contents

- `sydney_housing_3suburbs.csv` — the cleaned project dataset (201 rows).
- `Sydney-List-raw.csv` — the original Sydney-wide source data (~30,000 rows,
  from [briandorricott/SydneyHousePricesCovid](https://github.com/briandorricott/SydneyHousePricesCovid)),
  kept for full reproducibility of the filtering step.
- `Sydney_Housing_ML_Project.ipynb` — the full analysis notebook (Parts 1–5:
  data collection, EDA, feature engineering, model development/evaluation,
  prediction-failure analysis, and the ML vs LLM vs human comparison).
  Re-running it from top to bottom regenerates `house_price_model.joblib`
  and every figure in the report.
- `house_price_model.joblib` — the trained Random Forest pipeline (exported
  by the notebook), loaded directly by the app.
- `app.py` — the Streamlit decision-support app.
- `requirements.txt` — Python dependencies for the app.
- `fig_app_mockup.png` — a mockup of the app's interface (see note below).

## Reproducing the analysis

```bash
pip install -r requirements.txt jupyter nbclient nbformat imbalanced-learn
jupyter nbconvert --to notebook --execute --inplace Sydney_Housing_ML_Project.ipynb
```

This re-runs every step (data loading/filtering, EDA, feature engineering,
the three regression models with 5-fold CV, the holdout evaluation, the
prediction-failure analysis, and the ML/LLM/human comparison) and re-saves
`house_price_model.joblib`.

## Running the app locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`) in
your browser. Enter a suburb, property type, bedrooms, bathrooms, car
spaces, a sale/valuation date, and the distance to the nearest train
station (a sensible suburb-typical default is pre-filled), then click
**Predict sale price**.

## Live deployment

The app is deployed and publicly accessible at:
**https://sydney-housing-predictor.streamlit.app/**

## Deploying for free on GitHub + Streamlit Community Cloud

1. Create a new **public** GitHub repository and push this whole folder to
   it (`app.py`, `requirements.txt`, and `house_price_model.joblib` are the
   three files the app strictly needs at runtime).
2. Go to <https://share.streamlit.io>, sign in with your GitHub account,
   and click **New app**.
3. Select the repository and branch, set **Main file path** to `app.py`,
   and click **Deploy**.
4. Streamlit Cloud installs `requirements.txt` automatically and serves the
   app at a public `https://<your-app-name>.streamlit.app` URL within a
   couple of minutes.

## Note on the app screenshot in the report

This project was built and verified in a sandboxed, headless analysis
environment with no GUI browser available, so the screenshot originally
included in the report (`fig_app_mockup.png`) is an honest, clearly-labelled
mockup that reproduces `app.py`'s exact layout and widget labels, populated
with a **real prediction from the trained model** (not a made-up number).
The app has since been deployed live at the link above — visiting it and
taking a real screenshot there is recommended for final submission in
place of the mockup, if a literal screenshot is required.

## GenAI acknowledgement

This project (analysis notebook, app, and report) was developed with the
assistance of Claude (Anthropic) for planning, coding, execution and
drafting. Claude also produced the "LLM estimate" column in Part 5 as a
live, genuine model output (not a simulation). See the full acknowledgement
and methodology notes in the PDF report.
