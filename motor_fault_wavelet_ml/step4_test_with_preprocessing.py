import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load test dataset
test_df = pd.read_csv("datasets/Motor_signals_diverse.csv")

# 2. Separate features and labels
X_test = test_df.drop(columns=["label"])
y_test = test_df["label"]

# 3. Encode labels to numeric
label_encoder = LabelEncoder()
y_test_encoded = label_encoder.fit_transform(y_test)

print("Shape of X_test:", X_test.shape)

# 4. Load trained models
rf_model = joblib.load("random_forest_diverse.pkl")
xgb_model = joblib.load("xgboost_diverse.pkl")

# 5. Evaluate Random Forest
rf_pred = rf_model.predict(X_test)
print("\nRandom Forest Accuracy:", accuracy_score(y_test_encoded, rf_pred))
print("Random Forest Classification Report:\n", classification_report(
    y_test_encoded, rf_pred, target_names=label_encoder.classes_
))

# 6. Evaluate XGBoost
xgb_pred = xgb_model.predict(X_test)
print("\nXGBoost Accuracy:", accuracy_score(y_test_encoded, xgb_pred))
print("XGBoost Classification Report:\n", classification_report(
    y_test_encoded, xgb_pred, target_names=label_encoder.classes_
))

# 7. Function to plot and save confusion matrix
def plot_confusion_matrix(y_true, y_pred, labels, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# 8. Plot and save confusion matrices
plot_confusion_matrix(y_test_encoded, rf_pred, label_encoder.classes_,
                      "Random Forest Confusion Matrix", "rf_confusion_matrix.png")

plot_confusion_matrix(y_test_encoded, xgb_pred, label_encoder.classes_,
                      "XGBoost Confusion Matrix", "xgb_confusion_matrix.png")

print("\nConfusion matrices saved as 'rf_confusion_matrix.png' and 'xgb_confusion_matrix.png'")