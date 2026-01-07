# xgboost_model.py
# Predict CO2 emissions using XGBoost

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import pickle

# Load dataset
df = pd.read_csv(r"D:/CO2 Emmision project/Data set/FuelConsumptionCo2.csv")

# Select features and target
features = [
    'ENGINESIZE',
    'CYLINDERS',
    'FUELCONSUMPTION_CITY',
    'FUELCONSUMPTION_HWY',
    'VEHICLECLASS',
    'FUELTYPE'
]
X = df[features]
y = df['CO2EMISSIONS']

# One-hot encode categorical features
X = pd.get_dummies(X, columns=['VEHICLECLASS', 'FUELTYPE'], drop_first=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train XGBoost model
xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)
xgb.fit(X_train, y_train)

# Predict on test set
y_pred = xgb.predict(X_test)

# Evaluate model
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"XGBoost R²: {r2:.3f}")
print(f"XGBoost RMSE: {rmse:.2f} g/km")

# Save trained model
with open("D:/CO2 EMMISION PROJECT/models/xgb_model.pkl", "wb") as f:
    pickle.dump(xgb, f)

print("XGBoost model saved to D:/CO2 EMMISION PROJECT/models/xgb_model.pkl")
