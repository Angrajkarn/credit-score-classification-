# Credit Score Classification: Machine Learning Pipeline & Enterprise Study

**System Objective**: Automated, High-Fidelity Classification of Creditworthiness (Poor, Standard, Good)
**Architecture**: LightGBM GBDT with Sliding Lags, Profile Summaries, and Auto-Regressive Inference

---

## 1. Executive Summary

This project implements a complete time-series panel classification pipeline to assign customer credit ratings. The dataset comprises **100,000 training records** and **50,000 test records** representing **12,500 distinct customers** tracked over a calendar year.

Utilizing LightGBM with advanced temporal indicators, structural double imputation, and sequential predictive tracking, the system achieves a validation **Accuracy of 83.13%** and a **Macro F1 Score of 0.8309**.

---

## 2. System Pipeline Architecture

The system consists of the following processing checkpoints:

```
[Raw Datasets] ──> [Cleaning & Outlier Capping] ──> [Double Imputation]
                                                             │
[Temporal Model] <── [Lag/Diff Generation] <── [History Age Reconstruction]
       │
[Auto-Regressive Predictions] ──> [submission.csv] & [Streamlit Dashboard]
```

---

## 3. Data Cleaning and Imputation Strategy

Raw banking records contain substantial noise and missing values. The preprocessor cleans the data using the following techniques:

### 3.1 Numeric Value Cleaning

- Removing trailing symbols (e.g. underscores in strings like `'23_'`) and converting to float.
- Capping outliers based on realistic domain limits:
  - **Customer Age**: $[18, 80]$
  - **Open Bank Accounts / Cards**: $[0, 15]$
  - **Interest Rate %**: $[1, 34\%]$
  - **Active Loans**: $[0, 15]$
  - **Delayed Payments**: $[0, 28]$
  - **Credit Inquiries**: Capped at $20$
  - **Monthly Balance**: Negative values or overflow bounds are set to `NaN`.

### 3.2 Double Imputation Matrix

- **Customer-Level Imputation**: Since each customer has multiple monthly records, missing fields are imputed with that specific customer's median (numerical) or mode (categorical) over their history.
- **Global Fallback**: If a customer does not have any historical values for a parameter, the engine falls back to the global median/mode.

### 3.3 Linear Credit History Age Reconstruction

Credit history age is parsed from strings (e.g. `'22 Years and 4 Months'`) to total months. Since credit history increases linearly by exactly $1$ month each calendar month:
$$\text{Reconstructed History}_{i, t} = \text{Base History}_{i} + (\text{Month Value}_{t} - 1)$$
This formula calculates the correct linear timeline for each month, filling in all missing historical age values.

---

## 4. Advanced Feature Engineering

We generate 86 features across three scopes:

1. **Dynamic Lags and Differences**: Calculates $Lag_1, Lag_2, Diff_1, Diff_2$ for balance sheet attributes (debt, balance, late payments) to capture client trajectory direction.
2. **Customer Summary Profiles**: Computes historical statistical profiles (mean, standard deviation, min, max) for each client to provide a stable financial fingerprint.
3. **Loan Categorical Flags**: Parses text arrays into 9 binary fields representing active loan items (e.g., student loan, mortgage, payday loan).

---

## 5. Model Selection & Auto-Regressive Predictor

- **Classifier**: LightGBM (configured with 230 estimators, 127 leaves, and native categorical handling).
- **Auto-Regressive Tracking**: Since the previous month's credit score is the single strongest indicator of the current score, the prediction loop feeds predictions forward:
  - **September (Month 9)** predictions use August true credit score.
  - **October (Month 10)** predictions use September predicted credit score.
  - **November (Month 11)** predictions use October predicted credit score.
  - **December (Month 12)** predictions use November predicted credit score.

---

## 6. Experimental Results & Benchmarks

| Configuration                                                      | Validation Accuracy | Validation Macro F1 Score | Key Findings                                                   |
| :----------------------------------------------------------------- | :-----------------: | :-----------------------: | :------------------------------------------------------------- |
| **Baseline** (Direct LightGBM, Label Encoded, No Lags)             |       72.76%        |          0.7200           | Average results. High overfitting gap.                         |
| **Advanced Profile** (With profiles, No Target Lag)                |       73.67%        |          0.7313           | Clean generalization. Profile stats add valuable traits.       |
| **Lags Included** (With Profile + Lags & Differences)              |       73.72%        |          0.7317           | Logloss improved significantly (0.6612 -> 0.6443).             |
| **Auto-Regressive Native Cats** (All Features + Native Categories) |     **83.13%**      |        **0.8309**         | **Top Performer**. Balanced performance across all categories. |

### Classification Report

```
              precision    recall  f1-score   support

        Poor       0.82      0.84      0.83      7216
    Standard       0.84      0.82      0.83     12960
         Good       0.83      0.84      0.83      4824

    accuracy                           0.83     25000
   macro avg       0.83      0.83      0.83     25000
weighted avg       0.83      0.83      0.83     25000
```

---

## 7. Interactive Streamlit App

The dashboard features an enterprise-grade dark UI:

- **Overview**: Metric summary statistics, dataset overview, and class distribution pie charts.
- **EDA**: Interactive distribution density plots, violin plots, and correlation heatmaps. Includes a **KDE-Boxplot Feature Explorer** and a **Customer Timeline Visualizer** for sequence tracking.
- **Model Validation**: Metrics, Confusion Matrix heatmaps, Feature Importance (gain), and ROC curves.
- **Live Sandbox**: Input fields to simulate any client file and return instant classification probabilities.
- **Prediction Center**: Direct download of generated predictions.
