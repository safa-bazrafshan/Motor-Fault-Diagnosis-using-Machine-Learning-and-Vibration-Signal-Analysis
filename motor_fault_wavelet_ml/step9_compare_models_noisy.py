# step9_compare_models_noisy.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("motor_wavelet_dataset_multiple_noisy.csv")

# Split features and labels
X = df.drop("label", axis=1)
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models to compare
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "SVM": SVC(kernel="rbf", random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=42)
}

# Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.2f}")
    print(f"{name} Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Faulty"]))
    print("-" * 50)
    import joblib

    joblib.dump(models["RandomForest"], "random_forest_noisy.pkl")
    joblib.dump(models["SVM"], "svm_noisy.pkl")
    joblib.dump(models["XGBoost"], "xgboost_noisy.pkl")
    joblib.dump(scaler, "scaler_noisy.pkl")
    print("Models and scaler saved.")