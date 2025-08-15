import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load noisy dataset
df = pd.read_csv("motor_wavelet_dataset_noisy.csv")

# Split features and label
X = df.drop("label", axis=1)
y = df["label"]

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Predict on training data
y_pred = model.predict(X)

# Accuracy
acc = accuracy_score(y, y_pred)
print(f"Accuracy (noisy data): {acc:.2f}")

# Confusion matrix
cm = confusion_matrix(y, y_pred)

# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Faulty"], yticklabels=["Normal", "Faulty"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Noisy Data")
plt.tight_layout()

# Show and save
plt.savefig("confusion_matrix_noisy.png", dpi=300)
plt.show()