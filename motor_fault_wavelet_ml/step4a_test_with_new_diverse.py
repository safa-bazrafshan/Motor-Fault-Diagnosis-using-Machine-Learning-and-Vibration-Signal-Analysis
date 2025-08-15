import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("datasets/motor_signals_diverse.csv")
X = data.drop(columns=["label"])
y = data["label"]

X_train, X_test = X.iloc[:400], X.iloc[400:]
y_train, y_test = y.iloc[:400], y.iloc[400:]

rf_model = joblib.load("random_forest_diverse.pkl")
svm_model = joblib.load("svm_diverse.pkl")
xgb_model = joblib.load("xgboost_diverse.pkl")

rf_pred = rf_model.predict(X_test)
svm_pred = svm_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)

print("RandomForest Accuracy:", accuracy_score(y_test, rf_pred))
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))
print("XGBoost Accuracy:", accuracy_score(y_test, xgb_pred))

plt.figure(figsize=(6, 4))
accuracies = [
    accuracy_score(y_test, rf_pred),
    accuracy_score(y_test, svm_pred),
    accuracy_score(y_test, xgb_pred)
]
plt.bar(["RandomForest", "SVM", "XGBoost"], accuracies, color=["green", "orange", "blue"])
plt.ylim(0, 1.05)
plt.ylabel("Accuracy")
plt.title("Model Performance on Diverse Signals")
plt.show()