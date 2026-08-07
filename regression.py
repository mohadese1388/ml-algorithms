import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

df = pd.read_csv("health_regression_15_columns.csv")
X = df.drop(columns=["Health_Score"])
y = df["Health_Score"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)


model_lr = LinearRegression()
model_lr.fit(X_train_scaled, y_train)
y_pred_lr = model_lr.predict(X_test_scaled)


model_pca = LinearRegression()
model_pca.fit(X_train_pca, y_train)
y_pred_pca = model_pca.predict(X_test_pca)


print("--- نتایج رگرسیون خطی معمولی ---")
print(f"MAE (خطای مطلق): {mean_absolute_error(y_test, y_pred_lr):.2f}")
print(f"R2 Score (دقت مدل): {r2_score(y_test, y_pred_lr):.2f}")

print("\n--- نتایج رگرسیون خطی با PCA (۵ مؤلفه) ---")
print(f"MAE (خطای مطلق): {mean_absolute_error(y_test, y_pred_pca):.2f}")
print(f"R2 Score (دقت مدل): {r2_score(y_test, y_pred_pca):.2f}")

new_person = {
    'Age': 45,
    'Gender': 1,
    'BMI': 28.5,
    'Systolic_BP': 135.0,
    'Diastolic_BP': 85.0,
    'Cholesterol': 240.0,
    'Blood_Sugar': 110.0,
    'Physical_Activity_Hours': 4.0,
    'Sleep_Hours': 7.0,
    'Daily_Water_Intake': 2.5,
    'Stress_Level': 5,
    'Smoking_Status': 0,
    'Alcohol_Consumption': 2.0,
    'Family_History_Heart_Disease': 1
}

new_data = pd.DataFrame([new_person])


new_data_scaled = scaler.transform(new_data)


prediction_lr = model_lr.predict(new_data_scaled)


new_data_pca = pca.transform(new_data_scaled)


prediction_pca = model_pca.predict(new_data_pca)

print("=== پیش‌بینی برای داده جدید ===")
print(f"Health Score پیش‌بینی شده با مدل معمولی: {prediction_lr[0]:.2f}")
print(f"Health Score پیش‌بینی شده با مدل PCA: {prediction_pca[0]:.2f}")

