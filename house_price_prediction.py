# House Price Prediction using Linear Regression
# Author: Bhargavi
# Description: This script trains a linear regression model to predict house prices
# using the Housing.csv dataset.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# Load Dataset
# -----------------------------
housing_data = pd.read_csv('Housing.csv')
print("Dataset Preview:")
print(housing_data.head())

# -----------------------------
# Check Missing Values
# -----------------------------
missing_values = housing_data.isnull().sum()
print("\nMissing Values:")
print(missing_values)

# -----------------------------
# Encode Categorical Columns
# -----------------------------
categorical_columns = ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']

for column in categorical_columns:
    housing_data[column] = housing_data[column].map({'yes': 1, 'no': 0})

housing_data['furnishingstatus'] = housing_data['furnishingstatus'].map({
    'furnished': 2,
    'semi-furnished': 1,
    'unfurnished': 0
})

print("\nEncoded Data:")
print(housing_data.head())

# -----------------------------
# Data Visualization
# -----------------------------
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.scatterplot(x='area', y='price', data=housing_data)
plt.title("Area vs Price")
plt.xlabel("Area")
plt.ylabel("Price")

plt.subplot(2, 2, 2)
sns.scatterplot(x='bedrooms', y='price', data=housing_data)
plt.title("Bedrooms vs Price")
plt.xlabel("Bedrooms")
plt.ylabel("Price")

plt.subplot(2, 2, 3)
sns.scatterplot(x='mainroad', y='price', data=housing_data)
plt.title("Mainroad vs Price")
plt.xlabel("Mainroad Access (1 = Yes, 0 = No)")
plt.ylabel("Price")

plt.tight_layout()
plt.savefig("visualizations.png")  # Save for GitHub preview
plt.close()

# -----------------------------
# Train-Test Split
# -----------------------------
X = housing_data.drop('price', axis=1)
y = housing_data['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -----------------------------
# Train Linear Regression Model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# Evaluate Model
# -----------------------------
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Squared Error:", mse)
print("R² Score:", r2)

# -----------------------------
# Predict Price for New House
# -----------------------------
new_house = pd.DataFrame({
    'area': [3039],
    'bedrooms': [3],
    'bathrooms': [3],
    'stories': [4],
    'mainroad': [1],
    'guestroom': [1],
    'basement': [0],
    'hotwaterheating': [0],
    'airconditioning': [1],
    'parking': [1],
    'prefarea': [0],
    'furnishingstatus': [2]
})

predicted_price = model.predict(new_house)
print("\nPredicted Price for New House:", predicted_price[0])
