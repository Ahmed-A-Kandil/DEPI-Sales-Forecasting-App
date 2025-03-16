import joblib
import holidays

import numpy as np
import pandas as pd

import streamlit as st


def add_time_features(df):
    df = df.copy()

    df["Day of Week"] = df.index.dayofweek

    df["Month"] = df.index.month
    df["Month Sin"] = np.sin(df["Month"] * (2 * np.pi / 12))

    df["Day"] = df.index.day
    df["Day Sin"] = np.sin(df["Day"] * (2 * np.pi / 31))

    df["Day of Year"] = df.index.dayofyear
    df["Quarter"] = df.index.quarter
    df["Year"] = df.index.year

    if "Sales" in df.columns and len(df) > 1:
        df["Sales Day Before"] = df["Sales"].shift(1).fillna(0)
        df["Sales Day After"] = df["Sales"].shift(-1).fillna(0)
    else:
        df["Sales Day Before"] = 0
        df["Sales Day After"] = 0

    return df


def add_free_days(df):
    df = df.copy()

    years = df.index.year.unique()
    us_holidays = holidays.UnitedStates(years=years)
    holiday_dates = pd.to_datetime(list(us_holidays.keys()))

    df["Holiday"] = df.index.isin(holiday_dates).astype(int)
    df["Weekend"] = (df.index.dayofweek >= 5).astype(int)

    return df


model = joblib.load("model/model.joblib")


feature_cols = [
    "Day of Week",
    "Month",
    "Month Sin",
    "Day",
    "Day Sin",
    "Day of Year",
    "Quarter",
    "Year",
    "Sales Day Before",
    "Sales Day After",
    "Holiday",
    "Weekend",
]


st.title("Sales Forecasting App")

tabs = st.tabs(["Single Date Prediction", "Date Range Prediction"])

with tabs[0]:
    st.header("Single Date Prediction")

    selected_date = st.date_input("Select a date", value=pd.to_datetime("today"))

    df_single = pd.DataFrame(index=[pd.to_datetime(selected_date)])
    df_single = add_time_features(df_single)
    df_single = add_free_days(df_single)

    if st.button("Predict", key="single"):
        X_single = df_single[feature_cols]
        prediction = model.predict(X_single)[0]

        st.subheader("Predicted Sales")
        st.write(f"For {selected_date}, the predicted sales is: **{prediction:.2f}**")

with tabs[1]:
    st.header("Date Range Prediction")

    start_date = st.date_input("Start Date", value=pd.to_datetime("today"))
    end_date = st.date_input(
        "End Date", value=pd.to_datetime("today") + pd.DateOffset(days=30)
    )

    if start_date > end_date:
        st.error("Start date must be before or equal to End date.")
    else:
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")

        df_range = pd.DataFrame(index=date_range)
        df_range = add_time_features(df_range)
        df_range = add_free_days(df_range)

        if st.button("Predict", key="range"):
            X_range = df_range[feature_cols]
            predictions = model.predict(X_range)
            df_range["Prediction"] = predictions

            st.subheader("Predictions for Date Range")
            st.dataframe(df_range[["Prediction"]])
            st.line_chart(df_range["Prediction"])
