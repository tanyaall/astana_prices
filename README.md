# Astana Apartment Price Model

Predicting apartment prices in Astana, Kazakhstan from krisha.kz listings, and showing where the estimate can and cannot be trusted.

The project is framed for a real estate agency that needs two things: a fast price estimate for the apartments clients bring in, and a clear picture of how reliable that estimate is.

## Results

| | |
|---|---|
| Final model | Random Forest on log(price) |
| Test MAPE | **9.7%** (95% CI 9.4-10.0%) |
| Median error | 7.1% |
| Apartments within ±10% | 64% |
| Baseline (district median price per m² × area) | 18.2% |

Model comparison on 5-fold cross-validation (same folds for all models):

| Model | MAPE, % | 95% CI |
|---|---|---|
| District price per m² × area (baseline) | 18.2 | 17.9-18.4 |
| Linear Regression | 14.1 | 13.8-14.5 |
| Ridge | 14.1 | 13.8-14.5 |
| Decision Tree | 14.0 | 13.8-14.3 |
| **Random Forest** | **10.2** | **10.0-10.4** |

Random Forest is significantly better than every other model on all 5 folds. The three simpler models are practically indistinguishable.

## What is in the notebook

1. **Data cleaning.** Test set split off before any analysis; target leakage column (`kzt_sq_m` = price / area) and repeated listings removed; documented rules for impossible values (wrong coordinates, 43-room flats, rooms under 15 m²); missing values kept as a separate category because they are not random.
2. **Metric choice.** MAPE, because a 3M tenge error means 12% on a 25M apartment but 2% on a 150M one, and agency commission is a percentage of price.
3. **Model comparison.** Four models of increasing complexity, hyperparameters tuned with `GridSearchCV`, confidence intervals and paired significance tests on out-of-fold errors.
4. **Market segments.** KMeans on apartment characteristics (without price) finds 5 segments: old housing stock, compact flats on the outskirts, mass-market family flats, new comfort+ buildings, large business/premium flats. PCA shows the market along two axes: housing level and size vs. age.
5. **Price drivers.** Ridge coefficients vs. Random Forest importances, with an explanation of where and why they disagree.
6. **Classification.** Predicting whether an apartment falls in the top 25% by price per m² (to route it to a premium sales channel). Logistic regression, ROC-AUC 0.93, decision threshold chosen by the cost of errors (0.30 instead of 0.5) which cuts the cost of errors by 22%.
7. **Market insight with a confidence interval.** Esil is the most expensive district: +92k tenge per m² compared to the rest of the city (95% CI 85-98k), about 5-6M tenge for a 60 m² apartment.
8. **Limits of the model.** Errors are much higher for apartments over 150 m² (17%), with 5+ rooms and in the premium class (13%). For these the model should not be used without a manual appraisal.

## Data

Dataset: [Astana estate (krisha.kz) on Kaggle](https://www.kaggle.com/datasets/zhanarbakenova/astana-estate-krisha-kz). The notebook uses `merged_df.csv` (listings merged with residential complex data and building coordinates, before any cleaning).

The data is not included in this repository. Download the dataset and place the file at:

```
data/merged_df.csv
```

## How to run

```
pip install -r requirements.txt
jupyter notebook astana_prices.ipynb
```

Run all cells from top to bottom. The full run takes several minutes, most of it in the Random Forest tuning and cross-validation.

## Repository structure

```
astana_prices.ipynb          notebook with the full analysis
presentation.pdf             summary for a non-technical audience
requirements.txt
data/                        put merged_df.csv here (not tracked)
```

## Limitations

- Prices are asking prices from listings, not transaction prices.
- The model does not see renovation quality, view or urgency of sale.
- The data covers one period; the model needs retraining when the market moves.

## Tools

Python, pandas, NumPy, scikit-learn, SciPy, Matplotlib
