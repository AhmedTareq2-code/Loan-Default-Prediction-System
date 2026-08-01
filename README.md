# Loan Default Prediction System

Predicting whether a loan applicant will be approved or rejected, based on historical applicant data — built as a final project for an ML internship.

## Objective

Financial institutions need to assess loan applications quickly and consistently. This project builds and compares several classification models that predict loan approval outcomes from applicant attributes (income, credit history, marital status, property area, and related factors), with the goal of identifying risk before a loan is issued.

## Dataset

The classic **Loan Prediction dataset** (614 records, 12 raw features), including:

- Applicant demographics: `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`
- Financials: `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`
- Risk indicators: `Credit_History`, `Property_Area`
- Target: `Loan_Status` (Approved / Rejected)

## Project Workflow

1. **Data Preprocessing** — missing-value imputation (mode for categoricals, mean for `LoanAmount`), categorical encoding (binary mapping + one-hot encoding), train/test split (stratified), feature scaling
2. **Exploratory Data Analysis** — distribution, pairplot, count plots, scatter plots, and a correlation heatmap, each interpreted for what it implies about the modeling task
3. **Model Building** — four classifiers trained and evaluated on an identical held-out test set: Naive Bayes, Logistic Regression, Decision Tree, Random Forest
4. **Model Comparison** — accuracy, macro F1, and per-class precision/recall compared across all four models
5. **Findings & Insights** — synthesis of what the EDA and model results reveal together

## Results

| Model | Accuracy | Macro F1 | Reject (0) P/R | Approve (1) P/R |
|---|---|---|---|---|
| Naive Bayes | 78.9% | 0.70 | 0.83 / 0.39 | 0.78 / 0.96 |
| **Logistic Regression** | **79.7%** | 0.70 | 0.93 / 0.37 | 0.78 / 0.99 |
| Decision Tree | 66.7% | 0.62 | 0.46 / 0.50 | 0.77 / 0.74 |
| Random Forest | 75.6% | 0.65 | 0.72 / 0.34 | 0.76 / 0.94 |

**Logistic Regression** was selected as the primary model — it matches Naive Bayes on aggregate performance while offering directly interpretable feature coefficients, and both clearly outperform the tree-based models on this dataset size.

## Key Insights

- **Credit History is the dominant predictor** — applicants with a positive credit history are approved at a dramatically higher rate than any income-related feature suggests on its own.
- **The target is imbalanced (~2:1 approved to rejected)**, which explains why every model tested — regardless of algorithm family — is better at recognizing approvals than rejections (recall on "Reject" tops out at 50%).
- **Income and loan amount alone don't separate outcomes well** — the EDA pairplot shows heavy class overlap across all continuous features.
- **Property Area matters**: Semiurban properties have the highest approval rate among the three area types.

## Limitations & Future Work

- Single train/test split on a small dataset (614 rows) — k-fold cross-validation would give more reliable model comparisons.
- No hyperparameter tuning was applied; Decision Tree in particular is likely undertuned.
- Class imbalance was diagnosed but not directly corrected — class weighting or resampling is a natural next step to improve minority-class recall.
- Feature importance (e.g., via Random Forest's `.feature_importances_`) would let the "Credit History dominates" finding be confirmed numerically rather than only visually.

## Repository Structure

```
├── ML_Internship_Final_Project_Ahmed_Tareq.ipynb   # Full notebook: EDA, preprocessing, modeling, evaluation
├── loan_data.csv                                    # Dataset
├── Loan_Prediction_Summary.pdf                      # 1-page project summary
└── README.md
```

## Tech Stack

`pandas` · `numpy` · `seaborn` / `matplotlib` · `scikit-learn`

## Author

**Ahmed Tareq** — ML Internship Final Project
