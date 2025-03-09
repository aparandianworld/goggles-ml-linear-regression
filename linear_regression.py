import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
california_housing_dataset = fetch_california_housing()
df = pd.DataFrame(california_housing_dataset.data, columns = california_housing_dataset.feature_names)
df['MedHouseVal'] = california_housing_dataset.target

# Remove missing or impute values
print("original shape: ", df.shape)
print("number of missing values: \n", df.isnull().sum())
df = df.dropna(how = 'any')
print("shape after removing missing values: ", df.shape)

# Preview data
print("\nPreview of data: ")
print(df.head())

# Data statistics
print("\nData statistics: ")
print(df.describe())

# Feature matrix and target vector
feature_matrix = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"
]
target_vector = "MedHouseVal"
X = df.loc[:, feature_matrix].values
y = df.loc[:, target_vector].values

print("Feature matrix X shape: ", X.shape)
print("Target vector y shape: ", y.shape)

# Sprint data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model and fine tune hyper parameters
model = LinearRegression(fit_intercept=True)
model.fit(X_train, y_train)

# make predictions
y_train_prediction = model.predict(X_train)
y_test_prediction = model.predict(X_test)

# performance metrics
train_mse = mean_squared_error(y_train, y_train_prediction)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train, y_train_prediction)

test_mse = mean_squared_error(y_test, y_test_prediction)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_test_prediction)

# print performance metrics
print("\nTraining Performance Metrics:")
print(f"MSE: {train_mse:.2f}")
print(f"RMSE: {train_rmse:.2f}")
print(f"R2: {train_r2:.2f}")

# print performance metrics
print("\nTest Performance Metrics:")
print(f"MSE: {test_mse:.2f}")
print(f"RMSE: {test_rmse:.2f}")
print(f"R2: {test_r2:.2f}")

