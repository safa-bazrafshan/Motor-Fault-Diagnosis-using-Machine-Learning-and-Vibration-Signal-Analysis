import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("motor_wavelet_dataset_multiple_noisy.csv")

# Split features and labels
X = df.drop("label", axis=1)
y = df["label"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(model, "motor_fault_rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully.")