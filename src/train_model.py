import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
import os
import numpy as np

# --- Load data ---
df = pd.read_csv('../data/Housing.csv')
print("✅ Dataset loaded:", df.shape)

# --- Basic cleanup (adjust column names if necessary) ---
# Example: use numeric columns commonly available - change to match your CSV
# If your dataset has different columns, update feature_cols and target_col accordingly
target_col = 'price'
feature_cols = ['area', 'bedrooms', 'bathrooms']

# Simple handling of missing values and categories
df = df.copy()
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype('category').cat.codes

df.fillna(df.median(numeric_only=True), inplace=True)

X = df[feature_cols]
y = df[target_col]

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ Split: train={X_train.shape}, test={X_test.shape}")

# --- Train RandomForest ---
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("✅ Model trained (RandomForest)")

# --- Predictions & metrics ---
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, preds)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2: {r2:.4f}")

# --- Ensure models dir exists ---
os.makedirs('../models', exist_ok=True)

# --- Save model and metrics ---
joblib.dump(model, '../models/model.joblib')
metrics = {'MAE': float(mae), 'MSE': float(mse), 'RMSE': float(rmse), 'R2': float(r2)}
with open('../models/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("✅ Model and metrics saved in ../models/")