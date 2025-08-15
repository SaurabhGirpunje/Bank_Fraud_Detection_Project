# Bank Fraud Detection Project

## Project Overview

Bank fraud is a significant challenge for financial institutions worldwide. Fraudulent transactions can cause severe financial losses and harm customer trust. Detecting fraud early is critical to protecting both the bank and its customers.

This project focuses on building a **machine learning system to detect fraudulent bank transactions**. Using historical transaction data, the model predicts whether a given transaction is fraudulent (`isFraud = 1`) or normal (`isFraud = 0`).

The main goals of this project were:

* To explore patterns and relationships in transaction data that indicate fraud.
* To train machine learning models capable of accurately identifying fraud cases in a highly imbalanced dataset.
* To deploy the trained model in a **user-friendly Streamlit web application** for real-time fraud detection.

---

## Dataset Description

The dataset used in this project is sourced from Kaggle:
[Online Payment Fraud Detection Dataset](https://www.kaggle.com/datasets/jainilcoder/online-payment-fraud-detection/data?select=onlinefraud.csv)

It contains **6.36 million transactions** and the following key features:

| Column                  | Description                                                             |
| ----------------------- | ----------------------------------------------------------------------- |
| `step`                  | Represents a unit of time (in hours) since the start of the simulation. |
| `type`                  | Type of transaction (e.g., PAYMENT, TRANSFER).                          |
| `amount`                | Transaction amount.                                                     |
| `oldbalanceOrg`         | Account balance of the sender before the transaction.                   |
| `newbalanceOrig`        | Account balance of the sender after the transaction.                    |
| `oldbalanceDest`        | Account balance of the receiver before the transaction.                 |
| `isFraud`               | Target variable (1 = fraud, 0 = normal transaction).                    |
| `flag_dest_new_account` | Flag indicating if the destination account is newly opened.             |
| `transaction_hour`      | Derived feature: `step % 24` representing the hour of the day.          |

### Target Variable Distribution

The dataset is **highly imbalanced**, with most transactions being normal:

```
0 (Normal): 6,354,407
1 (Fraud): 8,213
```

This imbalance makes it critical to focus on metrics like **recall** for fraud detection, as missing even a small number of fraud cases could have significant financial consequences.

---

## Data Cleaning & Preprocessing

1. **Missing Values:**

   * The dataset contains no missing values, so no imputation was required.

2. **Feature Engineering:**

   * `transaction_hour` was derived from `step` using:

     ```python
     bankDf['transaction_hour'] = bankDf['step'] % 24
     ```

     This captures the time-of-day pattern of transactions, which can be helpful for detecting unusual activity.

3. **Column Selection:**

   * For modeling, the following cleaned columns were used:

     ```
     ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
      'oldbalanceDest', 'isFraud', 'flag_dest_new_account', 'transaction_hour']
     ```

4. **Categorical Encoding:**

   * The `type` feature was encoded using **one-hot encoding** to convert transaction types into numeric format for model input.

5. **Data Scaling:**

   * Continuous features like `amount`, `oldbalanceOrg`, and `newbalanceOrig` were scaled using **standardization** to improve model convergence and performance.

---

## Exploratory Data Analysis (EDA)

EDA was conducted to better understand the dataset and identify patterns related to fraud. Key insights include:

1. **Fraud Distribution:**

   * Fraud cases represent **only 0.13%** of total transactions, highlighting the extreme imbalance in the dataset.

2. **Transaction Amounts:**

   * Fraudulent transactions often involved **larger amounts** compared to normal transactions.

3. **Account Balances:**

   * Significant differences between `oldbalanceOrg` and `newbalanceOrig` were strong indicators of fraud.

4. **Time of Transaction:**

   * The `transaction_hour` feature revealed that fraud occurs **more frequently at unusual hours**, e.g., late night or early morning, compared to regular business hours.

5. **Destination Accounts:**

   * Fraudulent transactions were more likely to involve **newly created destination accounts**, indicated by the `flag_dest_new_account` feature.

6. **Correlations:**

   * Correlation analysis indicated that `newbalanceOrig`, `amount`, and `oldbalanceOrg` were highly associated with fraudulent activity.

---

## Feature Engineering

In addition to `transaction_hour`, several other transformations were applied to improve model performance:

* **Balance Change:**

  ```python
  bankDf['balance_change'] = bankDf['oldbalanceOrg'] - bankDf['newbalanceOrig']
  ```

  Large unexpected drops in balance often indicate fraud.

* **High-Value Transaction Flag:**

  * Transactions above a certain threshold were flagged to help the model detect extreme-value fraud patterns.

* **Transaction Type Encoding:**

  * One-hot encoding transformed categorical types into separate binary features for model training.

---

## Modeling

Three models were trained and evaluated:

1. **Logistic Regression:**

   * Baseline model to understand linear relationships.
   * Struggled with highly imbalanced data; low recall for fraud cases.

2. **Random Forest:**

   * Improved performance over Logistic Regression due to its ability to capture non-linear patterns.
   * Better recall but still generated some false negatives.

3. **XGBoost (Extreme Gradient Boosting):**

   * Outperformed other models significantly.
   * Handles imbalanced datasets well using scale\_pos\_weight or sample weighting.

### Model Training

* **Train-Test Split:** 80-20 split was used.
* **Hyperparameter Tuning:** Optuna was used to optimize XGBoost parameters such as `max_depth`, `learning_rate`, `n_estimators`, and `subsample`.
* **Metrics Monitored:**

  * **Recall:** Priority metric to catch as many fraud cases as possible.
  * **ROC-AUC Score:** Evaluates overall classification performance.
  * **Accuracy:** Secondary metric due to dataset imbalance.

### XGBoost Performance

| Metric      | Score  |
| ----------- | ------ |
| Test Recall | 0.9970 |
| ROC-AUC     | 0.9994 |
| Accuracy    | 0.9990 |

* Nearly all fraudulent transactions were detected.
* False positives were minimal, ensuring the model does not unnecessarily block legitimate transactions.

### Feature Importance

Using **SHAP analysis**, the most influential features for predicting fraud were:

1. `newbalanceOrig` – large balance drops are strong fraud indicators
2. `oldbalanceOrg` – initial account balance helps context
3. `amount` – higher amounts are riskier
4. `transaction_hour` – unusual hours increase fraud likelihood
5. `oldbalanceDest` – anomalies in recipient account balances

---

## Model Evaluation

Evaluation focused on **fraud detection effectiveness**, especially given the highly imbalanced dataset:

* **Recall (Fraud):** 0.997 – Ensures almost all fraud cases are caught.
* **Precision:** Slightly lower due to rare false positives.
* **ROC-AUC:** 0.9994 – Excellent discrimination between fraud and normal transactions.

The model demonstrates strong capability for real-world deployment, where **missing fraud cases can have severe financial consequences**.

---

## Deployment: Streamlit Web App

A **Streamlit web application** was developed to make the model accessible to non-technical users:

* Users can **input transaction details** such as type, amount, balances, and time.
* The app predicts whether the transaction is **fraudulent or normal**.
* Provides **visual explanations** using SHAP for transparency, showing which features influenced the prediction.

The Streamlit app enables **real-time fraud detection** and can be easily integrated with banking systems.

---

## Challenges & Solutions

1. **Class Imbalance:**

   * Fraud cases are extremely rare (0.13%).
   * **Solution:** Focused on recall, used sample weighting in XGBoost, and monitored false positives.

2. **Large Dataset (6.36M rows):**

   * Training models on millions of rows requires significant memory and compute.
   * **Solution:** Optimized data types, used batch processing, and employed XGBoost with efficient GPU support where possible.

3. **Feature Engineering:**

   * Capturing patterns like unusual balance changes or transaction times.
   * **Solution:** Derived features like `transaction_hour` and `balance_change`.

4. **Model Explainability:**

   * Needed to ensure the model decisions are understandable for stakeholders.
   * **Solution:** SHAP analysis was applied to identify key drivers of fraud predictions.

---

## Future Work

1. **Additional Features:**

   * Include customer behavior patterns, device information, and location data.

2. **Online Monitoring:**

   * Implement continuous learning to update the model as new fraud patterns emerge.

3. **Ensemble Methods:**

   * Combine XGBoost with other models (e.g., neural networks) for improved performance.

4. **Integration with Banking Systems:**

   * Real-time transaction monitoring with alerts for suspicious activity.

---

## Conclusion

This project demonstrates that **machine learning, particularly XGBoost**, is highly effective for detecting bank fraud:

* **High recall** ensures nearly all fraudulent transactions are caught.
* **SHAP analysis** provides interpretability for model decisions.
* The **Streamlit app** allows real-time predictions, making it practical for deployment in banking environments.

By combining **data-driven insights**, robust modeling, and user-friendly deployment, this project provides a scalable solution for **preventing financial fraud and protecting customer trust**.

---

## Streamlit App

The Streamlit app can be run locally or deployed on the web. It provides a **simple interface for real-time fraud prediction** and feature-level explanations using SHAP plots. \
Link: 