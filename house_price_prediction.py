import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load dataset
data = pd.read_csv("housing.csv")

# Features
X = data[['Size', 'Bedrooms', 'Bathrooms']]

# Target
y = data['Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# RMSE
rmse = np.sqrt(mean_squared_error(y_test, predictions))

# R²
r2 = r2_score(y_test, predictions)

print("RMSE:", rmse)
print("R² Score:", r2)

# Predict a new house price
new_house = [[180, 3, 2]]

predicted_price = model.predict(new_house)

print("\nPredicted Price:")
print(predicted_price[0])