# 💰 Loan Default Prediction using Machine Learning (XGBoost, RFE, GridSearch)

This project focuses on predicting **loan approval status** based on customer information using machine learning techniques. The dataset was sourced from [Kaggle](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data), and the final model achieved **93.2% test accuracy** with strong performance metrics.

---

## 🔍 Problem Statement

Financial institutions need a reliable way to **predict loan defaults** in order to minimize risk. Our goal is to build a robust classifier that can determine whether a loan should be approved or denied based on features like:

- Income
- Age
- Employment experience
- Credit score
- Loan amount
- Previous defaults

---

## 📊 Dataset Overview

- **Rows**: 45,000  
- **Target**: `loan_status` (1 = default, 0 = no default)  
- **Features**: Numerical + Categorical  
- **Source**: [Kaggle Loan Approval Dataset](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data)

---

## ✅ Steps & Techniques Used

### 📌 1. Data Cleaning & Preprocessing
- Removed/handled extreme outliers using IQR method  
- Type casting for float/integer mismatches  
- Label Encoding, OneHotEncoding, and OrdinalEncoding  
- Applied `RobustScaler` for numerical features

### 📌 2. Exploratory Data Analysis (EDA)
- Univariate analysis with histograms & boxplots  
- Multivariate analysis using `hue` with target feature  
- Insights drawn from education level, credit history, and defaults

### 📌 3. Feature Selection
- Used **RFE (Recursive Feature Elimination)** with multiple estimators  
- Final selected top 10 features led to reduced complexity and improved performance

### 📌 4. Model Training & Evaluation
- Compared multiple models:
  - Logistic Regression
  - Random Forest
  - SVM
  - **XGBoost (Winner 🏆)**

- Hyperparameter tuning with `GridSearchCV`  
- Evaluation Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix

---

## 📈 Final Results (XGBoost with GridSearch + RFE)

| Metric | Value |
|--------|--------|
| ✅ Accuracy | **93.2%** |
| 🎯 Precision (Class 1) | 0.89 |
| 🔄 Recall (Class 1) | 0.80 |
| ⚖️ F1-score (Class 1) | 0.84 |

- No SMOTE or oversampling needed  
- Train/Test performance well balanced — minimal overfitting  

---

## 🧠 Key Takeaways

- RFE significantly boosted performance by eliminating irrelevant features  
- XGBoost consistently outperformed other classifiers  
- Visual insights played a critical role in feature understanding  
- Handling outliers early improved the robustness of the pipeline

---

## 🛠️ Tech Stack

- Python 3.10  
- Pandas, NumPy  
- Seaborn, Matplotlib  
- Scikit-learn  
- XGBoost  
- GridSearchCV  
- RFE


---

## 📬 Contact

Feel free to connect or ask questions:  
**[https://www.linkedin.com/in/nour-ahmed-13677531b?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app]**

---

## 📌 License

This project is open-source and free to use under the [MIT License](LICENSE).
