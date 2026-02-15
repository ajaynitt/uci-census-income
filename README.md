Problem Statement

The goal is to predict whether an individual earns more than $50K annually based on demographic and employment attributes.

Dataset Description

The Adult Income dataset from UCI contains 48,842 records and 14 features. It is a binary classification problem where the target variable indicates income level (>50K or ≤50K).

| ML Model            | Accuracy | AUC   | Precision | Recall | F1 Score | MCC   |
| ------------------- | -------- | ----- | --------- | ------ | -------- | ----- |
| Logistic Regression | 0.823    | 0.860 | 0.744     | 0.461  | 0.569    | 0.487 |
| Decision Tree       | 0.809    | 0.749 | 0.623     | 0.626  | 0.625    | 0.497 |
| KNN                 | 0.826    | 0.858 | 0.677     | 0.599  | 0.636    | 0.523 |
| Naive Bayes         | 0.798    | 0.859 | 0.710     | 0.347  | 0.466    | 0.395 |
| Random Forest       | 0.855    | 0.908 | 0.748     | 0.645  | 0.693    | 0.601 |
| XGBoost             | 0.868    | 0.926 | 0.773     | 0.681  | 0.724    | 0.640 |


| Model               | Observation                                                                                                                      |
| ------------------- |----------------------------------------------------------------------------------------------------------------------------------|
| Logistic Regression | Provides strong baseline performance with good AUC (0.86). However, recall is lower due to class imbalance in the given dataset. |
| Decision Tree       | Shows moderate performance but slightly lower AUC, indicating potential overfitting and sensitivity to data variations.          |
| KNN                 | Performs better than Decision Tree due to feature scaling but is computationally expensive for large datasets.                   |
| Naive Bayes         | Lower recall and F1 score due to strong independence assumption among features.                                                  |
| Random Forest       | Significantly improves performance over Decision Tree by reducing variance and improving generalization.                         |
| XGBoost             | Achieves the best overall performance across all metrics due to boosting, regularization, and sequential learning.               |
