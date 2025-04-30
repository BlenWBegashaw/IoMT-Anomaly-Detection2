import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

# ✅ Load dataset used in main dashboard
file_path = "data/IP_Based_Flows_Dataset.csv"
df = pd.read_csv(file_path, nrows=50000)

# ✅ Ensure label column exists
target_column = "is_attack"
if target_column not in df.columns:
    raise ValueError(f"❌ Target column '{target_column}' not found! Available columns: {df.columns.tolist()}")

# ✅ Load trained scaler
scaler_path = "model/sr.pkl"
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"❌ Scaler not found at '{scaler_path}'")
scaler = joblib.load(scaler_path)

# ✅ Load trained models
model_paths = {
    "RandomForest": "model/rf.pkl",
    "SVM": "model/svm.pkl"
}
models = {}
for name, path in model_paths.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ {name} model not found at '{path}'")
    models[name] = joblib.load(path)

# ✅ Split features and labels
X = df.drop(columns=[target_column])
y = df[target_column]

# ✅ Match training features with scaler
if hasattr(scaler, "feature_names_in_"):
    try:
        X = X[scaler.feature_names_in_]
    except KeyError as e:
        raise ValueError(f"❌ Feature mismatch: {e}")
else:
    raise ValueError("❌ Scaler does not retain feature names. Retrain it using a DataFrame input.")

# ✅ Scale features
X_scaled = scaler.transform(X)

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ✅ Evaluation helper
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"\n📊 {name} Classification Report")
    print(classification_report(y_test, y_pred, digits=4))
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0)
    }

# ✅ Evaluate models
results = {}
for name, model in models.items():
    results[name] = evaluate_model(name, model, X_test, y_test)

# ✅ Save results
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

metrics_df = pd.DataFrame.from_dict(results, orient="index")
metrics_df.to_csv(os.path.join(results_dir, "model_evaluation.csv"), index=True)

print(f"\n✅ Evaluation results saved to '{os.path.join(results_dir, 'model_evaluation.csv')}'")
