# 🔮 Predictive Analytics Engine – Customer Churn

## 📖 Project Overview
This project is a **Predictive Analytics Engine** designed to identify customers at risk of churning for a telecom company. By analyzing historical customer data, the engine predicts which customers are likely to leave and provides actionable insights to improve retention.  

---

## 📂 Repository Structure
```
Predictive-Analytics-Engine/
│
├── data/
│   ├── raw/         # Original dataset
│   └── processed/   # Cleaned and preprocessed data
│
├── notebooks/       # EDA & modeling notebooks
├── src/             # Scripts for preprocessing, training, and evaluation
├── models/          # Saved trained models
├── README.md        # Project documentation
└── requirements.txt # Python dependencies
```

---

## 📊 Dataset
- **Source:** Telecom customer dataset  
- **Shape:** (7043, 21)  
- **Target Variable:** `Churn` (Yes/No)  
- **Key Features:**  
  - **Numerical:** `tenure`, `MonthlyCharges`, `TotalCharges`  
  - **Categorical:** `Contract`, `PaymentMethod`, `InternetService`, `OnlineSecurity`, `TechSupport`, `SeniorCitizen`, etc.

---

## 🎯 Problem Statement
Customer churn is a critical problem for subscription-based companies. Identifying potential churners in advance allows businesses to take proactive measures, reduce revenue loss, and improve customer satisfaction.  

---

## 🔍 Key Insights from EDA
- Customers with **month-to-month contracts** churn more frequently  
- High **MonthlyCharges** correlate with higher churn  
- Long-term customers have higher **TotalCharges** and lower churn  
- **Payment method, contract type, and value-added services** are strong predictors  

---

## 🛠️ Methodology
1. **Data Preprocessing**  
   - Handle missing values and correct data types  
   - Encode categorical variables using One-Hot Encoding  
   - Scale numerical features with StandardScaler  
   - Split dataset into train and test sets (80/20)  

2. **Baseline Model**  
   - Logistic Regression with `class_weight="balanced"` to handle imbalance  
   - Threshold tuning to optimize recall for churn  

3. **Advanced Models**  
   - Random Forest Classifier with class balancing and hyperparameter tuning  
   - Threshold optimization for business-focused recall improvement  

4. **Evaluation Metrics**  
   - Accuracy, Precision, Recall, F1-score  
   - ROC-AUC Score  
   - Confusion Matrix  

---

## 📈 Results
- **Logistic Regression (with threshold tuning)**:
  - Recall (churn) = 0.90  
  - ROC-AUC = 0.84  

- **Random Forest**:
  - Recall (churn) = 0.78  
  - ROC-AUC = 0.84  

> ✅ Business insight: Model identifies high-risk customers accurately for retention strategies  

---

## 🧠 Skills Learned
- Exploratory Data Analysis (EDA) 📊  
- Handling imbalanced datasets ⚖️  
- Preprocessing pipelines (`ColumnTransformer` & `Pipeline`) 🛠️  
- Logistic Regression & Random Forest modeling 🤖  
- Threshold tuning for business objectives 🎯  
- Model evaluation using precision, recall, F1-score, ROC-AUC 📈  
- Feature importance interpretation 📌  

---

## 🚀 Future Enhancements
- Implement **XGBoost** for improved predictive performance  
- Hyperparameter tuning using **GridSearchCV / RandomizedSearchCV**  
- Model explainability using **SHAP** or **LIME**  
- Deployment-ready model with `joblib` or `pickle` 💾  
---

