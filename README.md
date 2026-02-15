# 📊 Adult Income Classification -- ML Project

## 🔍 Overview

This project builds and compares multiple Machine Learning models to
predict whether an individual earns **\>50K or ≤50K per year** using the
UCI Adult (Census Income) Dataset.

The project includes: - End-to-end preprocessing pipeline - Multiple
classification models - Evaluation using multiple performance metrics -
Confusion matrix visualization - Interactive Streamlit dashboard - Saved
trained models for deployment

------------------------------------------------------------------------

## 📁 Dataset

-   Source: UCI Adult Census Income Dataset
-   Training file: `adult.data`
-   Official test file: `adult.test`
-   Target variable: `income`
    -   `0` → ≤50K
    -   `1` → \>50K

------------------------------------------------------------------------

## 🛠️ Models Implemented

-   Logistic Regression
-   Decision Tree
-   K-Nearest Neighbors (KNN)
-   Naive Bayes
-   Random Forest
-   XGBoost

All models are trained using a Pipeline with: - StandardScaler (for
numerical features) - OneHotEncoder (for categorical features) -
ColumnTransformer - Dense encoding (to support GaussianNB)

------------------------------------------------------------------------

## 📊 Model Performance (80/20 Train-Test Split)

  -----------------------------------------------------------------------------------------
  Model         Accuracy     Precision      Recall      F1          MCC         AUC
  ------------- ------------ -------------- ----------- ----------- ----------- -----------
  Logistic      0.858        0.752          0.615       0.676       0.592       0.908
  Regression                                                                    

  Decision Tree 0.820        0.622          0.650       0.636       0.517       0.762

  KNN           0.836        0.678          0.613       0.644       0.539       0.864

  Naive Bayes   0.531        0.336          0.962       0.498       0.330       0.735

  Random Forest 0.861        0.747          0.640       0.690       0.604       0.907

  **XGBoost**   **0.879**    **0.790**      **0.680**   **0.731**   **0.657**   **0.931**
  -----------------------------------------------------------------------------------------

🏆 **Best Performing Model: XGBoost**
