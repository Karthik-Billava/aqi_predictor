# City-Day AQI Prediction

This project explores and compares regression models for estimating a city's daily Air Quality Index (AQI) from same-day pollutant readings, city, and calendar features.

## Files

- `city_day.csv` — the dataset used by the notebook.
- `city_day_aqi_prediction.ipynb` — data inspection, exploratory analysis, preprocessing, model comparison, and final evaluation.
- `app.py` — a simple form for entering readings and estimating AQI.
- `requirements.txt` — Python packages used by the app.

## Run the notebook

Install the notebook packages with:

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Open a terminal in the project directory and start Jupyter:

```bash
jupyter notebook city_day_aqi_prediction.ipynb
```

Run the cells from top to bottom. Keep `city_day.csv` in the same directory as the notebook.

## Run the prediction app locally

Install the app packages and start Streamlit:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Choose a city and date, enter at least one pollutant reading, and select **Estimate AQI**. Unknown readings can be left blank and are imputed by the model. The app fits and caches a small HistGradientBoosting model when a prediction is requested; it does not need a separate model file.

## Deploy for free with Streamlit Community Cloud

1. Push this repository to GitHub. The repository must include `app.py`, `requirements.txt`, and `city_day.csv`.
2. Create a free account at [Streamlit Community Cloud](https://share.streamlit.io/) and connect your GitHub account.
3. Choose **Create app**, then select this repository, the `main` branch, and `app.py` as the app file.
4. Select **Deploy**. Streamlit installs the packages from `requirements.txt` and gives you a public app link.

See the [official deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) for screenshots and current options.

The app loads the CSV on startup and trains the model the first time a prediction is requested. The dataset ends on 2020-07-01, so predictions for later dates are outside the training period and may be unreliable. The app does not fetch live pollutant readings.

## Workflow

The notebook inspects the data, checks missing values and duplicates, and plots the target, pollutant distributions, correlations, city coverage, and AQI over time. It removes duplicate rows and rows without a usable AQI target or date. `AQI_Bucket` is excluded from the predictors because it is associated with the target.

The earliest 80% of unique dates are used for training; the latest 20% are held out for final evaluation. Training-period validation folds also keep records from the same date together. Numeric and categorical missing values are imputed, numeric features are scaled, and city names are one-hot encoded within each model pipeline.

Six regressors are compared: Linear Regression, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, and K-Nearest Neighbors. A median-prediction baseline is included. The model with the lowest average validation MAE is selected and evaluated on the later-date holdout using MAE, RMSE, and R².

## Notes and limitations

- This estimates AQI after same-day pollutant readings are available; it is not a forecast made before those readings exist.
- The chronological split evaluates later dates for cities represented in the training period. It does not measure performance on unseen cities.
- The dataset does not include a data dictionary, pollutant units, sensor details, or the AQI calculation method. Confirm these with the data provider before interpreting or deploying the results.
- Model metrics are generated when the notebook runs and are not prefilled in the notebook.
