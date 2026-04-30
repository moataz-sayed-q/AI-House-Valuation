import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from xgboost.core import XGBoostError
from sklearn.metrics import r2_score, mean_absolute_error

# 1. Load and preprocess data
df = pd.read_csv('BostonHousing.csv')
df = df.fillna(df.mean())

X = df.drop('medv', axis=1)
y = df['medv']

# 2. Split dataset and apply feature scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Configure model
gpu_model = XGBRegressor(
    n_estimators=3000,
    learning_rate=0.05,    # Step size for gradient updates
    max_depth=5,           # Maximum depth of each tree
    tree_method="hist",   # Histogram-based algorithm
    device="cuda",        # Try CUDA first
    random_state=42
)

cpu_model = XGBRegressor(
    n_estimators=3000,
    learning_rate=0.05,
    max_depth=5,
    tree_method="hist",
    device="cpu",
    random_state=42
)

# 4. Train model and generate predictions
trained_on_gpu = False
try:
    print("Training on GPU...")
    model = gpu_model
    model.fit(X_train_scaled, y_train)
    trained_on_gpu = True
except XGBoostError:
    print("GPU is not available for XGBoost. Falling back to CPU...")
    model = cpu_model
    model.fit(X_train_scaled, y_train)

# Move booster prediction to CPU when input data is NumPy/CPU to avoid device mismatch warnings.
if trained_on_gpu:
    model.set_params(device="cpu")

predictions = model.predict(X_test_scaled)

# 5. Evaluate model performance
print("\n=== XGBoost GPU Results ===")
print(f"R2 Score: {r2_score(y_test, predictions):.4f}")
print(f"MAE: ${mean_absolute_error(y_test, predictions):.2f}k")