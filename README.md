# Loan Default Prediction Model

This project predicts whether a loan applicant is likely to experience serious delinquency within the next two years. The goal is to identify at-risk borrowers early using historical credit data and apply this to real-world risk management.

---

## Dataset

- **Source**: [Give Me Some Credit (Kaggle)](https://www.kaggle.com/c/GiveMeSomeCredit)
- Two datasets used:
  - `cs-training.csv`: Full training data (150,000+ rows)
  - `cs-test.csv`: Unlabeled test set for scoring only
  
---

## Workflow

1. **Data Inspection**  
   - Identified missing values, duplicates, and invalid ages  
   - Visualized class imbalance and distributions

2. **Data Cleaning**  
   - Removed duplicates and ages < 18  
   - Imputed missing values with median  
   - Optionally capped extreme outliers (99th percentile)

3. **EDA (Exploratory Data Analysis)**  
   - Histograms and KDE plots by target class  
   - Boxplots for feature distribution  
   - Correlation heatmap for feature relationships

4. **Data Preprocessing**  
   - Feature scaling with `StandardScaler`  
   - SMOTE applied to balance classes (0 vs 1)

5. **Model Training**  
   - Trained and compared:
     - `RandomForestClassifier`
     - `LogisticRegression`  
   - Evaluated using classification report, ROC AUC, and confusion matrices

6. **Threshold Tuning**  
   - Plotted precision, recall, and F1 vs thresholds  
   - Optimized model to better balance false positives vs recall

---

## Key Insights

- The original data was highly imbalanced (only ~5% defaulters)
- SMOTE improved recall without sacrificing too much precision
- Random Forest performed best overall, with robust precision and recall
- Feature importance revealed that late payments and revolving credit utilization were most predictive

---

## Model Performance

| Metric        | Random Forest | Logistic Regression |
|---------------|----------------|---------------------|
| Accuracy      | 91%            | 80%                 |
| Recall (1)    | 44%            | 74%                 |
| F1 (1)        | 0.39           | 0.33                |

---

## Outputs

- Predictions: `output/random_forest_predictions.csv`
- Plots: Correlation heatmap, confusion matrix, ROC curve, etc.

---

## Future Work

- Try advanced models (XGBoost, LightGBM)
- Engineer risk-related features (e.g. late score index)
- Add explainability (SHAP) to justify individual predictions

---

## Author

**Thomas Stone-Wigg**  
Data Analyst in training, focused on finance, sustainability, and high-impact analytics.


