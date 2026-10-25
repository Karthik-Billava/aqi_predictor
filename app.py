from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path(__file__).with_name("city_day.csv")
POLLUTANTS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO",
    "SO2", "O3", "Benzene", "Toluene", "Xylene",
]
DATE_FEATURES = ["Year", "Month", "Day", "DayOfWeek"]
FEATURES = ["City", *POLLUTANTS, *DATE_FEATURES]


@st.cache_data
def load_data():
    """Load the supplied training data and add simple date features."""
    if not DATA_PATH.exists():
        raise FileNotFoundError("city_day.csv must be in the project directory.")

    data = pd.read_csv(DATA_PATH)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.drop_duplicates().dropna(subset=["AQI", "Date"]).copy()

    data["Year"] = data["Date"].dt.year
    data["Month"] = data["Date"].dt.month
    data["Day"] = data["Date"].dt.day
    data["DayOfWeek"] = data["Date"].dt.dayofweek
    return data


@st.cache_resource
def train_model():
    """Fit one small model once and reuse it for app predictions."""
    data = load_data()
    numeric_features = [*POLLUTANTS, *DATE_FEATURES]

    numeric_steps = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    city_steps = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric_steps, numeric_features),
        ("city", city_steps, ["City"]),
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("regressor", HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.08,
            max_leaf_nodes=20,
            l2_regularization=1.0,
            random_state=42,
        )),
    ])
    model.fit(data[FEATURES], data["AQI"])
    return model


st.set_page_config(page_title="City-Day AQI Predictor", page_icon="🌫️")
st.title("City-Day AQI Predictor")
st.write("Enter a city, date, and any available same-day pollutant readings.")

training_data = load_data()
last_date = training_data["Date"].max().date()
st.caption("Enter as many measured pollutant values as you can; missing readings are imputed.")

with st.form("aqi_prediction_form"):
    city = st.selectbox("City", sorted(training_data["City"].dropna().unique()))
    selected_date = st.date_input("Date", value=last_date)

    st.subheader("Pollutant readings")
    st.caption("Enter values on the dataset's scale. Keep unknown readings at 0.0; each step changes by 0.1.")

    columns = st.columns(3)
    readings = {}
    for index, pollutant in enumerate(POLLUTANTS):
        with columns[index % 3]:
            readings[pollutant] = st.number_input(
                pollutant,
                min_value=0.0,
                value=0.0,
                step=0.1,
                format="%.1f",
                key=f"reading_{pollutant}",
            )

    submitted = st.form_submit_button("Estimate AQI")

if submitted:
    if all(value == 0.0 for value in readings.values()):
        st.error("Enter at least one pollutant reading above 0.0.")
        st.stop()

    record = {
        "City": city,
        **{
            pollutant: np.nan if readings[pollutant] == 0.0 else readings[pollutant]
            for pollutant in POLLUTANTS
        },
        "Year": selected_date.year,
        "Month": selected_date.month,
        "Day": selected_date.day,
        "DayOfWeek": selected_date.weekday(),
    }
    input_data = pd.DataFrame([record], columns=FEATURES)
    prediction = train_model().predict(input_data)[0]
    st.metric("Estimated AQI", f"{prediction:.1f}")
    st.caption("Missing readings are filled using medians learned from the training data.")
