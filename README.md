# DEPI-Sales-Forecasting-App

A Streamlit-based sales forecasting application that uses an XGBoost model to predict sales based on time-related features. The app provides two functionalities:

- **Single Date Prediction:** Predict sales for one specific date.
- **Date Range Prediction:** Predict sales for every day in a specified date range.

## Features

- **Time Feature Engineering:**  
  Automatically generates features from the date (day of week, month, sine transformations, etc.).

- **Holiday and Weekend Flags:**  
  Uses the `holidays` package to flag US holidays and weekends.

- **XGBoost Prediction:**  
  Uses a pre-trained XGBoost model to forecast sales.

- **Interactive Interface:**  
  Built with Streamlit for quick and interactive forecasting.

## Prerequisites

- Python 3.10 or higher (recommended)
- Git (to clone the repository, optional)

## Setup Instructions

1. Clone the Repository

Clone this repository to your local machine:

```bash
git clone https://github.com/Ahmed-A-Kandil/DEPI-Sales-Forecasting-App.git
cd DEPI-Sales-Forecasting-App
```

2. Create a Virtual Environment

Create an isolated Python environment using venv (the directory is named .venv):

```bash
python -m venv .venv
```

3. Activate the Virtual Environment

- On macOS/Linux:

```bash
source .venv/bin/activate
```

- On Windows:

```bash
.\.venv\Scripts\activate
```

4. Install Dependencies

Ensure you have pip updated and then install the required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The requirements.txt file should include:

- streamlit
- pandas
- numpy
- holidays
- xgboost

5. Prepare Your Model

Ensure your XGBoost model file (e.g., xgb_model.pkl) is located in the project directory.

> Note: If you encounter warnings about loading models serialized with an older XGBoost version, consider re-saving your model using the current XGBoost methods.

6. Run the Application

Start the Streamlit app by running:

```
streamlit run app.py
```

Your browser will open the app with two tabs:

- Single Date Prediction: Select a single date for forecasting.
- Date Range Prediction: Provide a start and end date to view predictions over a period.
