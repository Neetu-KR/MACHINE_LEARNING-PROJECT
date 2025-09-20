import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 1.Dataset

df = pd.read_csv("students_dataset.csv")

# Features: exclude first_name & last_name (not numeric)
X = df[["standard", "attendance", "study_hours", "assignments", "semester"]]
y = df["final_grade"]

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train Models
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = (mse, r2)
    print(f"{name} -> MSE: {mse:.2f}, R²: {r2:.2f}")

# 5. Save Best Model
best_model_name = min(results, key=lambda k: results[k][0])  # lowest MSE
best_model = models[best_model_name]

joblib.dump(best_model, "best_student_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print(f"✅ Saved best model: {best_model_name}")

# 6. Predict for New Student
# Example: standard 10, attendance 85%, study hours 12, 7 assignments, semester 2
new_student = pd.DataFrame([[10, 85, 12, 7, 2]], columns=X.columns)

scaler = joblib.load("scaler.pkl")
best_model = joblib.load("best_student_model.pkl")

new_student_scaled = scaler.transform(new_student)
predicted_grade = best_model.predict(new_student_scaled)

print("📘 Predicted Final Grade:", round(predicted_grade[0], 2))