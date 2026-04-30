import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv('BostonHousing.csv')
df = df.fillna(df.mean())

X = df.drop('medv', axis=1)
y = df['medv']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)

print("=== LINEAR REGRESSION RESULTS ===")
print(f"R2 Score: {r2_score(y_test, predictions):.4f}")
print(f"MAE: ${mean_absolute_error(y_test, predictions):.2f}k")

comparison = pd.DataFrame({'Real': y_test.values[:5], 'Guess': predictions[:5]})
print("\n--- First 5 Comparison ---")
print(comparison)