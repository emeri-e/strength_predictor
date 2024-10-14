import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
import joblib

aggregate = 'lime'
# Data
# data = {
#     '% replacement': [0, 2, 5, 7, 10, 15, 20, 25, 30],
#     '7_days': [15.67, 15.03, 14.88, 14.45, 12.47, 10.32, 8.78, 7.55, 6.98],
#     '14_days': [17.01, 16.78, 15.97, 15.72, 13.43, 12.84, 10.2, 8.88, 7.23],
#     '21_days': [18.98, 17.67, 16.78, 15.88, 14.77, 13.22, 12.34, 9.88, 7.94],
#     '28_days': [22.45, 19.56, 18.34, 18.01, 16.44, 14.87, 13.22, 12.62, 11.04]
# }

# bebe
# data = {
#     '% replacement': [0, 2, 5, 7, 10, 15, 20, 25, 30],
#     '7_days': [4.14, 3.98, 3.22, 3.21, 3.02, 2.87, 2.57, 2.55, 2.02],
#     '14_days': [4.98, 4.66, 4.21, 3.65, 3.61, 3.31, 2.88, 2.68, 2.54],
#     '21_days': [5.66, 5.19, 4.96, 4.63, 4.25, 3.84, 3.21, 2.87, 2.64],
#     '28_days': [6.42, 6.02, 5.83, 5.43, 5.00, 4.67, 3.28, 3.20, 2.89]
# }

# ebuka
# data = {
#     '% replacement': [0, 2, 5, 7, 10, 15, 20, 25, 30],
#     '7_days': [14.85, 14.12, 13.89, 13.86, 13.22, 13.01, 12.9, 11.01, 9.89],
#     '14_days': [14.88, 14.78, 14.55, 14.52, 14.32, 13.87, 13.49, 12.84, 10.34],
#     '21_days': [17.54, 16.34, 16.31, 16.01, 15.99, 15.56, 14.77, 12.67, 11.36],
#     '28_days': [20.66, 19.88, 19.37, 19.24, 18.88, 18.44, 17.78, 16.86, 14.44]
# }

# robert
data = {
    'Lime Content (%)': [0, 2, 5, 7, 10, 15, 20, 25, 30],
    '10%-add water': [22.1, 24, 27.4, 29.3, 30.6, 29.8, 28.2, 26.6, 25.3],
    '12%-add water': [24.3, 26.3, 29.5, 31.1, 32.7, 32, 30.3, 28.7, 27.2],
    '14%-add water': [25.7, 27.8, 31.2, 33.2, 34.5, 33.6, 31.7, 29.9, 28.8],
    '16%-add water': [24.5, 26.1, 29.3, 31.4, 32.9, 32.1, 30.5, 28.3, 27.5],
    '18%-add water': [23.2, 24.9, 28, 29.8, 31.3, 30.5, 29, 27.1, 26.2],
    '20%-add water': [21.8, 23.4, 26.5, 28.2, 29.7, 28.9, 27.5, 25.8, 24.9]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# # Reshape the Data
# df_long = pd.melt(df, id_vars=['% replacement'], var_name='time', value_name='compressive_strength')
# df_long['time'] = df_long['time'].str.extract('(\d+)').astype(int)

# # Features and Target
# X = df_long[['% replacement', 'time']]
# y = df_long['compressive_strength']

# Reshape the Data
df_long = pd.melt(df, id_vars=['Lime Content (%)'], var_name='water_content', value_name='compressive_strength')
df_long['water_content'] = df_long['water_content'].str.extract('(\d+)').astype(int)

# Features and Target
X = df_long[['Lime Content (%)', 'water_content']]
y = df_long['compressive_strength']



# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model Definitions
models = {
    "linear_regression": LinearRegression(),
    "polynomial_regression": make_pipeline(PolynomialFeatures(degree=3), LinearRegression()),
    "ann": MLPRegressor(hidden_layer_sizes=(2960,), max_iter=1000, random_state=42)
}

# Train and Save Models
for name, model in models.items():
    model.fit(X_scaled, y)
    joblib.dump(model, f'models/{aggregate}/{name}.pkl')

# Save the scaler
joblib.dump(scaler, f'models/{aggregate}/scaler.pkl')
