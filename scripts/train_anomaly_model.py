import pandas as pd
import joblib
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ✅ Ensure model directory exists
os.makedirs("model", exist_ok=True)

# ✅ Load dataset
file_path = "data/IP_Based_Flows_Dataset.csv"
df = pd.read_csv(file_path, nrows=50000)

# ✅ Check label column exists
if "is_attack" not in df.columns:
    raise ValueError("❌ Dataset must include an 'is_attack' column with 0=normal and 1=anomaly.")

# ✅ Data Cleaning
df = df.drop_duplicates().fillna(0)

# ✅ Separate features and labels
y = df["is_attack"]

# ✅ Use only numeric features
X = df.drop(columns=["is_attack"])
X = X.select_dtypes(include=[np.number])  # ⛏️ Filter out strings (like IPs, protocols)

# ✅ Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ Train RandomForest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# ✅ Train SVM Model
svm_model = SVC(kernel="rbf", probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# ✅ Save models and scaler
joblib.dump(rf_model, "model/rf.pkl")
joblib.dump(svm_model, "model/svm.pkl")
joblib.dump(scaler, "model/sr.pkl")

print("✅ Models and scaler saved in 'model/' folder.")

# ✅ Evaluate Models
y_rf_pred = rf_model.predict(X_test)
y_svm_pred = svm_model.predict(X_test)

print("\n📈 RandomForest Classification Report:")
print(classification_report(y_test, y_rf_pred))

print("\n📈 SVM Classification Report:")
print(classification_report(y_test, y_svm_pred))

