# Loan Default Prediction Model

This project predicts whether a loan applicant is likely to experience serious delinquency within the next two years. The aim is to identify high-risk borrowers early using structured credit data and apply practical modelling techniques suited to financial risk environments.

---

## Dataset

- **Source**: [Give Me Some Credit (Kaggle)](https://www.kaggle.com/c/GiveMeSomeCredit)
- Training data: `cs-training.csv` (150,000+ rows)
- Test data: `cs-test.csv` (unlabelled for prediction only)

---

## Project Workflow

1. **Data Inspection and Cleaning**
   - Removed duplicates and invalid age entries
   - Imputed missing values with median
   - Optionally capped outliers (99th percentile)

2. **Exploratory Data Analysis**
   - Distribution plots and boxplots grouped by default status
   - Correlation heatmap to identify predictive variables

3. **Preprocessing and Balancing**
   - Feature scaling using `StandardScaler`
   - Applied SMOTE to balance the target variable

4. **Modelling**
   - Trained and compared:
     - Random Forest
     - Logistic Regression
   - Evaluated using accuracy, F1 score, recall, and AUC

5. **Threshold Tuning and Evaluation**
   - Adjusted decision thresholds to improve recall
   - Analysed model performance using confusion matrices and ROC curves
   - Visualised feature importance (Random Forest)

---

## Model Summary

| Metric             | Random Forest  | Logistic Regression |
|--------------------|----------------|---------------------|
| Accuracy           | 91%            | 80%                 |
| Precision (Class 1)| 36%            | 22%                 |
| Recall (Class 1)   | 44%            | 74%                 |
| F1 Score (Class 1) | 0.39           | 0.33                |
| AUC                | 0.83           | 0.86                |

---

## Outputs

- Model predictions saved to: `output/random_forest_predictions.csv`
- Key visualizations: correlation heatmap, ROC curves, feature importance

---

## Future Improvements

- Explore more advanced models (e.g. XGBoost, LightGBM)
- Add interpretability tools (e.g. SHAP)
- Engineer domain-specific features such as late payment scores

---

## Author

**Thomas Stone-Wigg**  
Aspiring data analyst with an interest in finance, sustainability, and applied analytics.
