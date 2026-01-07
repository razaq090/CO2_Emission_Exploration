# CO2 Emission Predictor

This project predicts **CO2 emissions of vehicles** using different machine learning models: **Linear Regression, Random Forest, and XGBoost**. It includes data exploration, model training, evaluation, and saved models for deployment.

---

## **Dataset**

The dataset `FuelConsumptionCo2.csv` contains vehicle fuel consumption and CO2 emissions data with the following columns:

- `MODELYEAR` – Year of the vehicle model
- `MAKE` – Vehicle manufacturer
- `MODEL` – Vehicle model name
- `VEHICLECLASS` – Vehicle type (e.g., Compact, SUV)
- `ENGINESIZE` – Engine size in liters
- `CYLINDERS` – Number of engine cylinders
- `TRANSMISSION` – Transmission type
- `FUELTYPE` – Fuel type (e.g., Petrol, Diesel)
- `FUELCONSUMPTION_CITY` – Fuel consumption in city (L/100 km)
- `FUELCONSUMPTION_HWY` – Fuel consumption on highway (L/100 km)
- `FUELCONSUMPTION_COMB` – Combined fuel consumption (L/100 km)
- `FUELCONSUMPTION_COMB_MPG` – Combined fuel consumption (MPG)
- `CO2EMISSIONS` – CO2 emissions in g/km (target variable)

---

## **Models and Performance**

| Model               | R² Score | RMSE (g/km) |
|--------------------|-----------|-------------|
| Linear Regression   | 0.876    | 22.62       |
| Random Forest       | 0.990    | 6.57        |
| XGBoost             | 0.993    | 5.84        |

> Metrics are calculated on the **test set (20% split)**.

---

## **Usage**

1. Install dependencies:

```bash
pip install -r requirements.txt


python scripts/linear_regression.py
python scripts/random_forest.py
python scripts/xgboost_model.py

import pickle
import pandas as pd

# Load model
with open("models/rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

# Prepare input dataframe (same features as training)
input_df = pd.DataFrame({
    'ENGINESIZE': [3.5],
    'CYLINDERS': [6],
    'FUELCONSUMPTION_CITY': [12.3],
    'FUELCONSUMPTION_HWY': [8.5],
    'VEHICLECLASS_SUV': [1],  # One-hot encoded column
    'FUELTYPE_P': [1]         # One-hot encoded column
})

# Predict CO2
co2_pred = rf_model.predict(input_df)
print(f"Predicted CO2 emission: {co2_pred[0]:.2f} g/km")


---

## **requirements.txt**

```text
pandas==2.1.0
numpy==1.26.0
scikit-learn==1.8.0
xgboost==1.7.6
