# linear_regression.py
# Predict CO2 emissions using Linear Regression

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import pickle


# Load dataset
# Use raw string r"..." to handle backslashes
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


# Train Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)


#  Predict on test set
y_pred = lr.predict(X_test)


# Evaluate model
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Linear Regression R²: {r2:.3f}")
print(f"Linear Regression RMSE: {rmse:.2f} g/km")


# Save trained model
with open("D:/CO2 EMMISION PROJECT/models/lr_model.pkl", "wb") as f:
    pickle.dump(lr, f)

print("Model saved to D:/CO2 EMMISION PROJECT/models/lr_model.pkl")
