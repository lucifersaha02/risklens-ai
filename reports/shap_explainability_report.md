# RiskLens AI SHAP Explainability Report

## Scope

- Model: `full_history_xgboost_calibrated`
- Dataset split: `validation`
- Explanation sample: 2,000 applicants
- Final holdout accessed: **No**

SHAP values explain the underlying XGBoost raw margin. The displayed final risk probability is subsequently transformed by sigmoid calibration, so SHAP values must not be interpreted as percentage-point probability changes.

## Leading global drivers

| Rank | Feature | Mean absolute SHAP | Mean signed SHAP |
|---:|---|---:|---:|
| 1 | EXT_SOURCE_MEAN | 0.442591 | -0.127403 |
| 2 | AMT_ANNUITY | 0.105212 | -0.004625 |
| 3 | GOODS_CREDIT_RATIO | 0.105046 | -0.014420 |
| 4 | CREDIT_ANNUITY_RATIO | 0.100527 | -0.026374 |
| 5 | INSTALLMENTS_DAYS_LATE_MEAN | 0.090126 | -0.013091 |
| 6 | EXT_SOURCE_MAX | 0.074304 | -0.001445 |
| 7 | NAME_EDUCATION_TYPE_Higher education | 0.063657 | -0.004865 |
| 8 | EXT_SOURCE_MIN | 0.063655 | -0.024045 |
| 9 | DAYS_EMPLOYED | 0.062599 | -0.008427 |
| 10 | INSTALLMENTS_AMT_PAYMENT_SUM | 0.060445 | -0.010655 |
| 11 | BUREAU_AMT_CREDIT_SUM_DEBT_MEAN | 0.054938 | -0.008010 |
| 12 | POS_CNT_INSTALMENT_FUTURE_MEAN | 0.053543 | -0.004309 |
| 13 | PREVIOUS_REFUSED_FLAG_MEAN | 0.050383 | -0.004923 |
| 14 | missingindicator_EXT_SOURCE_1 | 0.046103 | -0.000766 |
| 15 | BUREAU_DAYS_CREDIT_ENDDATE_MAX | 0.043366 | -0.002224 |

## Limitations

- SHAP describes this model's behavior; it does not establish causality.
- Correlated features can share or redistribute attribution.
- One-hot and engineered features require domain-aware interpretation.
- Reason codes support human review and are not autonomous lending decisions.
- Protected attributes require governance even when they are not top drivers.
