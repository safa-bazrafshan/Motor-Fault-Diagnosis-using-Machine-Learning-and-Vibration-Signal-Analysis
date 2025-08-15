# Fault Diagnosis of Electric Motors Using Signal Processing and Machine Learning

Author: Safa Bazrafshan  
Role: Independent Researcher  

---

## 📌 Project Overview
This project focuses on detecting and classifying electric motor faults using raw vibration signals and machine learning models.  
Two classifiers were implemented:
- Random Forest
- XGBoost

The models were trained and tested on a dataset containing multiple fault types under different operating conditions, noise levels, and frequencies.

---

## 📊 Key Results
- Random Forest Accuracy: 96.6%
- XGBoost Accuracy: 93.1%
- Both models performed well under varying noise and frequency levels.
- Confusion matrices and classification reports are provided for detailed evaluation.

---

## 📂 Repository Contents
- datasets/ → Contains raw and feature-extracted CSV files.
- step1_preprocessing.py → Data preprocessing and feature engineering.
- step2_train_models.py → Training Random Forest & XGBoost models.
- step3_evaluation.py → Model evaluation with classification reports.
- step4_test_with_preprocessing.py → Final testing script.
- rf_confusion_matrix.png / xgb_confusion_matrix.png → Confusion matrix plots.
- report.pdf → Full project report with methodology, results, and references.

---

## ⚙️ How to Run
### 1️⃣ Clone the repository
`bash
git clone https://github.com/YourUsername/motor-fault-detection.git
cd motor-fault-detection

---

2️⃣ Install dependencies

pip install -r requirements.txt

3️⃣ Run the evaluation

python step4_test_with_preprocessing.py


---

📈 Results Visualization

The confusion matrices and classification reports help to understand each model's performance across different fault categories.

Model Accuracy

Random Forest 96.6%
XGBoost 93.1%

---

📄 Report

The full project report (PDF) contains:

Abstract

Introduction

Methodology

Results

Conclusion & Future Work

---

## Author
Safa Bazrafshan  
Independent Researcher

---

📜 License

This project is released under the MIT License. Feel free to use and modify it for your own research or applications.
