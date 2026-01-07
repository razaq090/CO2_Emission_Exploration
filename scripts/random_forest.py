# random_forest.py
# Predict CO2 emissions using Random Forest

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
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

# Train Random Forest model
rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# Predict on test set
y_pred = rf.predict(X_test)

# Evaluate model
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Random Forest R²: {r2:.3f}")
print(f"Random Forest RMSE: {rmse:.2f} g/km")

# Save trained model
with open("D:/CO2 EMMISION PROJECT/models/rf_model.pkl", "wb") as f:
    pickle.dump(rf, f)

print("Random Forest model saved to D:/CO2 EMMISION PROJECT/models/rf_model.pkl")
