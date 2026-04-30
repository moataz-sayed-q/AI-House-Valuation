import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# 1. Load and Prepare Data
df = pd.read_csv('BostonHousing.csv')
df = df.fillna(df.mean())

X = df.drop('medv', axis=1)
y = df['medv']

# 2. Split and Scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Train Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 4. Predict
predictions = model.predict(X_test_scaled)

# 5. Output Results
print("=== RANDOM FOREST RESULTS ===")
print(f"R2 Score: {r2_score(y_test, predictions):.4f}")
print(f"MAE: ${mean_absolute_error(y_test, predictions):.2f}k")

print("\n--- Sample Predictions ---")
comparison = pd.DataFrame({'Real Price': y_test.values[:5], 'AI Guess': predictions[:5]})
print(comparison)
