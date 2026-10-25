# City-Day AQI Predictor

This is a machine learning lab project using `city_day.csv`. The notebook compares regression models for estimating AQI from city, date, and pollutant readings. The small Streamlit app lets you enter those values and get an estimate.

## Run the app

From the project folder, install the app packages and start it:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Choose a city and date, then enter the pollutant readings you have. The fields start at `0.0` and change by `0.1` when you use the plus or minus buttons. Leave a reading at `0.0` if you do not have it; the model will fill it in. The app trains a small model from the CSV the first time you ask for a prediction.

## Deploy the app

The app can run on [Streamlit Community Cloud](https://share.streamlit.io/), which offers free hosting for Streamlit apps. The source repository needs `app.py`, `requirements.txt`, and `city_day.csv`.

1. Sign in to Streamlit Community Cloud with GitHub.
2. Choose **Create app**.
3. Select this repository, the `main` branch, and `app.py` as the main file.
4. Choose **Deploy**. Streamlit installs the packages and gives you an app link.

See Streamlit's [deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) for details.

## Run the notebook

Install Jupyter and the analysis packages if you need them:

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook city_day_aqi_prediction.ipynb
```

Run the notebook cells from top to bottom. It checks the data, makes a few plots, compares six regressors against a median baseline, and reports MAE, RMSE, and R². It trains on earlier dates and holds out the latest 20% of dates for final evaluation.

## A note about the estimates

The dataset ends on July 1, 2020, and does not include pollutant units or a detailed data dictionary. The app does not get live readings, and predictions for later dates may be unreliable. It estimates AQI after same-day pollutant readings are available; it is not a forecast made before those readings exist.
