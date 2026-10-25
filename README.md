# City-Day AQI Prediction

This project explores and compares regression models for estimating a city's daily Air Quality Index (AQI) from same-day pollutant readings, city, and calendar features.

## Files

- `city_day.csv` — the dataset used by the notebook.
- `city_day_aqi_prediction.ipynb` — data inspection, exploratory analysis, preprocessing, model comparison, and final evaluation.

## Requirements

Python 3.9 or newer is recommended. Install the notebook dependencies with:

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## Run the notebook

1. Open a terminal in the project directory.
2. Start Jupyter:

   ```bash
   jupyter notebook city_day_aqi_prediction.ipynb
   ```

3. Run the cells from top to bottom. Keep `city_day.csv` in the same directory as the notebook.

## Workflow

The notebook inspects the data, checks missing values and duplicates, and plots the target, pollutant distributions, correlations, city coverage, and AQI over time. It removes duplicate rows and rows without a usable AQI target or date. `AQI_Bucket` is excluded from the predictors because it is associated with the target.

The earliest 80% of unique dates are used for training; the latest 20% are held out for final evaluation. Training-period validation folds also keep records from the same date together. Numeric and categorical missing values are imputed, numeric features are scaled, and city names are one-hot encoded within each model pipeline.

Six regressors are compared: Linear Regression, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, and K-Nearest Neighbors. A median-prediction baseline is included. The model with the lowest average validation MAE is selected and evaluated on the later-date holdout using MAE, RMSE, and R².

## Notes and limitations

- This estimates AQI after same-day pollutant readings are available; it is not a forecast made before those readings exist.
- The chronological split evaluates later dates for cities represented in the training period. It does not measure performance on unseen cities.
- The dataset does not include a data dictionary, pollutant units, sensor details, or the AQI calculation method. Confirm these with the data provider before interpreting or deploying the results.
- Model metrics are generated when the notebook runs and are not prefilled in the notebook.
