# Credit Card Fraud Detection Using Machine Learning

## Overview
This project aims to detect fraudulent credit card transactions using various machine learning algorithms. The dataset is highly imbalanced, with fraudulent transactions representing only 0.172% of the total data. To address this, we applied both undersampling and oversampling techniques, along with feature selection and model optimization strategies.

## Dataset
- **Source**: [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/joebeachcapital/credit-card-fraud)
- **Features**:
  - 30 numerical features: 28 anonymized via PCA, and 2 original (`Time`, `Amount`)
  - `Class` label: 0 for non-fraud, 1 for fraud

## Preprocessing Steps
1. **Data Scaling**: Applied `RobustScaler` to `Time` and `Amount` to reduce the effect of outliers.
2. **Class Imbalance Handling**:
   - **Undersampling**: Reduced majority class to match minority class.
   - **Oversampling (SMOTE)**: Generated synthetic fraud samples.
3. **Outlier Removal**: Applied to non-fraud transactions only.
4. **Feature Selection**:
   - Selected top 9 features based on feature importance.
5. **Data Splitting**: Stratified 80/20 train-test split.

## Models Used
We trained and compared the following models:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- XGBoost
- Support Vector Machine (SVM)

Each model was evaluated on:
- **Original Dataset**
- **Undersampled Dataset**
- **Oversampled Dataset**
- **Top Features Only (with and without sampling)**

## Evaluation Metrics
Due to the class imbalance, we focused on:
- **Precision**
- **Recall**
- **F1-Score**
- **ROC-AUC**

## Results Summary
- **XGBoost** and **KNN** showed the best balance between precision and recall.
- Feature selection improved computational efficiency with negligible performance loss.
- **Oversampling** improved recall significantly but reduced precision.
- **Undersampling** yielded realistic but low-precision models due to limited data.

## Folder Structure

```
.
├── Data/
├── Models/
├── analying_and_preprocessing.ipynb
├── models_comparison.ipynb
├── requirements.pdf
├── report.pdf
└── README.md
```
