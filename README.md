# Sydney Housing Price Predictor 

Decision-support prototype that predicts sale prices for properties in **Mosman**, **Marrickville**, and **Bankstown**, Sydney, trained on real sold-listing data scraped from Domain.com.au.

**🔗 Live app:** [sydney-housing-predictor.streamlit.app](https://sydney-housing-predictor.streamlit.app)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sydney-housing-predictor.streamlit.app)

---

## What it does

Enter a suburb, dwelling type, bedrooms, bathrooms, car spaces and (optionally) land size, and the app returns:

- A predicted sale price from a trained **Random Forest** regression model
- The suburb's min / median / max price range for context
- A bar chart comparing the prediction against that range
- An explicit disclaimer that the output is a statistical estimate, not a professional valuation

## Dataset

506 real sold properties collected directly from Domain.com.au's sold-listings search (August 2026 snapshot, sales spanning roughly January–August 2026):

| Suburb | Properties |
|---|---|
| Mosman | 169 |
| Marrickville | 167 |
| Bankstown | 170 |

Listings marked "Price Withheld" were excluded since price is the prediction target. Fields captured: address, sale date, sale method, price, bedrooms, bathrooms, car spaces, land size (where shown), and property type.

## Model

Three models were compared with 5-fold cross-validation on log(price): Ridge Regression, Random Forest, and Gradient Boosting. **Random Forest** was selected for deployment — it produced the best validation MAE, RMSE, MAPE and R² of the three, with a smaller overfitting gap than Gradient Boosting.

| Model | Val MAE | Val MAPE | Val R² |
|---|---|---|---|
| Ridge Regression | $415,001 | 19.5% | 0.879 |
| **Random Forest (deployed)** | **$330,567** | **14.6%** | **0.922** |
| Gradient Boosting | $344,735 | 15.4% | 0.918 |

On the held-out test set: MAE $677,167, RMSE $2,162,817, MAPE 16.7%, R²(log) 0.896. The largest remaining prediction errors are concentrated in ultra-high-value Mosman trophy homes (multi-million-dollar sales with limited comparable training examples) — full error analysis is in the notebook.

## Repo structure

```
.
├── app.py                                     # Streamlit app
├── price_pipeline.joblib                      # Trained Random Forest pipeline (preprocessing + model)
├── suburb_price_context.csv                   # Suburb min/median/max price lookup used by the app
├── mosman_marrickville_bankstown_sold.csv      # Real sold-property dataset (506 rows)
├── 8D_Distinction_Task_Sydney_Housing.ipynb    # Full analysis notebook (EDA, modelling, error analysis)
├── 8D_Distinction_Task_Report.pdf              # Written report summarising methodology and findings
├── requirements.txt                            # Python dependencies
└── runtime.txt                                 # Pinned Python version for Streamlit Community Cloud
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open `http://localhost:8501`.

## Reproduce the analysis

```bash
jupyter nbconvert --to notebook --execute 8D_Distinction_Task_Sydney_Housing.ipynb
```

This re-runs data loading, EDA, model training/evaluation, error analysis, and re-saves `price_pipeline.joblib` + `suburb_price_context.csv`.

## Notes & limitations

- ~half of Domain listings withhold sale price at the vendor's request, which likely under-represents the top end of each market (especially Mosman).
- Land size is missing for ~49% of rows (mostly units); treated as unreliable for units rather than trusted at face value.
- The model is a statistical estimate for decision-support purposes only — **not** a substitute for a licensed valuation.

## Acknowledgement
Generative AI tools were used to improve the clarity of the report, explain programming concepts, assist with
debugging Python code, and refine the presentation of the notebook. All data analysis, preprocessing,
visualisations, model development, interpretation of results, and final decisions were completed and verified by
me. AI was used as a learning aid, and the final submission reflects my own understanding while maintaining
academic integrity.
