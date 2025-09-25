# salary_prediction.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.pipeline import Pipeline
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------
# 1️⃣ Load Dataset
# ------------------------
df = pd.read_csv("salary_data.csv")
print("Columns in dataset:", df.columns.tolist())
print(df.head())

# ------------------------
# 2️⃣ Convert categorical columns to numeric codes
# ------------------------
categorical_columns = ['education', 'job_title', 'location', 'company_size']

for col in categorical_columns:
    df[col] = df[col].astype('category').cat.codes

# Select features explicitly
X = df[['years_experience', 'education', 'job_title', 'location', 'company_size', 'skill_score']]
y = df['salary']

print("\nProcessed Features:")
print(X.head())

# ------------------------
# 3️⃣ Split data
# ------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------
# 4️⃣ Train Voting Regressor
# ------------------------
rf = RandomForestRegressor(n_estimators=100, random_state=42)
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)

ensemble = VotingRegressor(estimators=[('rf', rf), ('gb', gb)])
full_pipeline = Pipeline([('model', ensemble)])

full_pipeline.fit(X_train, y_train)
print("\n✅ Model trained successfully")

# ------------------------
# 5️⃣ Save model
# ------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(full_pipeline, "models/salary_predictor.pkl")
print("✅ Model saved at 'models/salary_predictor.pkl'")

# ------------------------
# 6️⃣ Test predictions
# ------------------------
predictions = full_pipeline.predict(X_test)

print("\nSample Predictions:")
for i in range(min(5, len(X_test))):
    print(f"Predicted: {predictions[i]:,.2f} | Actual: {y_test.iloc[i]:,.2f}")

# ------------------------
# 7️⃣ Graphical Visualization
# ------------------------
os.makedirs("graphs", exist_ok=True)

# 1. Salary distribution
plt.figure(figsize=(8,5))
sns.histplot(df['salary'], bins=20, kde=True)
plt.title("Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Frequency")
plt.savefig("graphs/salary_distribution.png")
plt.show()

# 2. Salary vs Years of Experience
plt.figure(figsize=(8,5))
sns.scatterplot(x=df['years_experience'], y=df['salary'], hue=df['education'], palette='viridis')
plt.title("Salary vs Years of Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.savefig("graphs/salary_vs_experience.png")
plt.show()

# 3. Predicted vs Actual Salary
plt.figure(figsize=(8,5))
sns.scatterplot(x=y_test, y=predictions)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title("Predicted vs Actual Salary")
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.savefig("graphs/predicted_vs_actual.png")
plt.show()
