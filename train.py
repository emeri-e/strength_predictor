import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
import joblib

# Data
data = {
    '% replacement': [0, 2, 5, 7, 10, 15, 20, 25, 30],
    '7_days': [8.671, 8.42, 9.11, 10.45, 4.89, 4.32, 3.78, 3.55, 2.98],
    '14_days': [8.94, 9.87, 12.45, 12.72, 5.78, 5.63, 4.2, 4.18, 3.23],
    '21_days': [12, 10.7, 16.78, 17.88, 9.87, 8.88, 8.34, 6.34, 5.94],
    '28_days': [12.65, 15.56, 18.34, 20.01, 16.43, 15.87, 15.22, 12.62, 11.04]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Reshape the Data
df_long = pd.melt(df, id_vars=['% replacement'], var_name='time', value_name='compressive_strength')
df_long['time'] = df_long['time'].str.extract('(\d+)').astype(int)

# Features and Target
X = df_long[['% replacement', 'time']]
y = df_long['compressive_strength']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model Definitions
models = {
    "linear_regression": LinearRegression(),
    "polynomial_regression": make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
    "ann": MLPRegressor(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
}

# Train and Save Models
for name, model in models.items():
    model.fit(X_scaled, y)
    joblib.dump(model, f'{name}.pkl')

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')
