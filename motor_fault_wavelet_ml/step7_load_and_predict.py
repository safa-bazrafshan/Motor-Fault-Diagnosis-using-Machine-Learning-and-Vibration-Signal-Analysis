import joblib
import numpy as np

# Load saved model and scaler
model = joblib.load("motor_fault_rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# Example: new sample features (random example)
# Replace this array with extracted wavelet features from a new signal
new_sample = np.array([[65.0, 8.2, 420.0, 122.0, 55.0, 120.0,
                        48.0, 6.2, -915.0, -47.0, 48.0, 83.0]])

# Scale the new sample
new_sample_scaled = scaler.transform(new_sample)

# Predict
prediction = model.predict(new_sample_scaled)
label_map = {0: "Normal", 1: "Faulty"}

print(f"Predicted class: {label_map[prediction[0]]}")