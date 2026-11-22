
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.set_page_config(
    page_title="Indian AQI Prediction Dashboard",
    page_icon="🌍",
    layout="wide"
)

# Load model and dataset
model = joblib.load("best_aqi_model.joblib")
df = pd.read_csv("dashboard_data.csv")
df["DateTime"] = pd.to_datetime(df["DateTime"])

st.title("🌍 Indian Air Quality Prediction Dashboard")
st.write(
    "Explore air quality trends and predict AQI "
    "using a Machine Learning model."
)

# AQI category function
def classify_aqi(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"

# Sidebar
st.sidebar.header("Dashboard Filters")

cities = sorted(df["City"].dropna().unique())
selected_city = st.sidebar.selectbox("Select City", cities)

# City-wise data
city_data = df[df["City"] == selected_city].copy()

# Dashboard overview
st.subheader("AQI Overview")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Average AQI",
    round(city_data["AQI"].mean(), 2)
)

col2.metric(
    "Maximum AQI",
    round(city_data["AQI"].max(), 2)
)

col3.metric(
    "Minimum AQI",
    round(city_data["AQI"].min(), 2)
)

# AQI trend
st.subheader(f"AQI Trend: {selected_city}")

trend = (
    city_data.groupby(city_data["DateTime"].dt.date)["AQI"]
    .mean()
    .reset_index()
)

trend.columns = ["Date", "Average AQI"]

fig = px.line(
    trend,
    x="Date",
    y="Average AQI",
    title="Average AQI Trend"
)

st.plotly_chart(fig, use_container_width=True)

# City comparison
st.subheader("City-wise AQI Comparison")

city_comparison = (
    df.groupby("City")["AQI"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig2 = px.bar(
    city_comparison,
    x="City",
    y="AQI",
    title="Top 10 Cities by Average AQI"
)

st.plotly_chart(fig2, use_container_width=True)

# Prediction section
st.header("🔮 Predict AQI")

st.write("Enter the details below to predict the next hourly AQI.")

state_options = sorted(
    df[df["City"] == selected_city]["State"].dropna().unique()
)

if len(state_options) == 0:
    state_options = sorted(df["State"].dropna().unique())

state = st.selectbox("State", state_options)

previous_aqi = st.number_input(
    "Previous Hour AQI",
    min_value=0.0,
    max_value=500.0,
    value=100.0
)

hour = st.slider("Hour of Day", 0, 23, 12)
month = st.slider("Month", 1, 12, 6)

day_of_week = st.selectbox(
    "Day of Week",
    options=list(range(7)),
    format_func=lambda x: [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ][x]
)

season = st.selectbox(
    "Season",
    ["Winter", "Summer", "Monsoon", "Post-Monsoon"]
)

if st.button("Predict AQI"):

    input_data = pd.DataFrame([{
        "Previous_Hour_AQI": previous_aqi,
        "Hour_Number": hour,
        "Month": month,
        "Day_of_Week": day_of_week,
        "Hour_Sin": np.sin(2 * np.pi * hour / 24),
        "Hour_Cos": np.cos(2 * np.pi * hour / 24),
        "Month_Sin": np.sin(2 * np.pi * month / 12),
        "Month_Cos": np.cos(2 * np.pi * month / 12),
        "City": selected_city,
        "State": state,
        "Season": season
    }])

    prediction = float(model.predict(input_data)[0])
    prediction = max(0, min(500, prediction))

    category = classify_aqi(prediction)

    st.subheader("Prediction Result")

    st.metric("Predicted AQI", round(prediction, 2))

    st.success(f"Predicted Air Quality Category: {category}")

    st.caption(
        "This is an experimental one-hour-ahead model prediction, "
        "not an official CPCB reading or public-health advisory."
    )

st.subheader("AQI Category Reference")

category_table = pd.DataFrame({
    "AQI Range": [
        "0–50", "51–100", "101–200",
        "201–300", "301–400", "401–500"
    ],
    "Category": [
        "Good", "Satisfactory", "Moderate",
        "Poor", "Very Poor", "Severe"
    ]
})

st.table(category_table)

st.caption(
    "Dataset coverage: 2023. "
    "Source: CPCB-derived hourly AQI dataset."
)
