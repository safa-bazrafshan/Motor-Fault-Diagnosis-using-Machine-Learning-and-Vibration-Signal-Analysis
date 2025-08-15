# step10_plot_model_comparison.py
import pandas as pd
import matplotlib.pyplot as plt

# Model results (from step9)
results = {
    "Model": ["RandomForest", "SVM", "XGBoost"],
    "Accuracy": [1.00, 1.00, 1.00]
}

df_results = pd.DataFrame(results)

# Save table
df_results.to_csv("model_comparison_noisy.csv", index=False)
print(df_results)

# Plot bar chart
plt.figure(figsize=(6, 4))
plt.bar(df_results["Model"], df_results["Accuracy"], color=["#4CAF50", "#2196F3", "#FF9800"])
plt.ylim(0, 1.05)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison (Noisy Dataset)")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Save plot
plt.tight_layout()
plt.savefig("model_comparison_noisy.png", dpi=300)
plt.show()