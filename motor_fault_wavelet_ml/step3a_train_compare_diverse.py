import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("datasets/motor_signals_diverse.csv")

# Remove classes with fewer than 2 samples
class_counts = data["label"].value_counts()
data = data[data["label"].isin(class_counts[class_counts >= 2].index)]

X = data.drop(columns=["label"])
y = data["label"]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# RandomForest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("RandomForest Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred, zero_division=0))

# SVM
svm = SVC(kernel="rbf", probability=True)
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))
print(classification_report(y_test, svm_pred, zero_division=0))

# XGBoost
xgb = XGBClassifier(eval_metric='mlogloss')
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
print("XGBoost Accuracy:", accuracy_score(y_test, xgb_pred))
print(classification_report(y_test, xgb_pred, zero_division=0))

# Save models and encoder
joblib.dump(rf, "random_forest_diverse.pkl")
joblib.dump(svm, "svm_diverse.pkl")
joblib.dump(xgb, "xgboost_diverse.pkl")
joblib.dump(le, "label_encoder_diverse.pkl")

print("Models saved successfully.")