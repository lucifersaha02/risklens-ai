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
| 1 | EXT_SOURCE_MEAN | 0.452791 | -0.127191 |
| 2 | GOODS_CREDIT_RATIO | 0.100204 | -0.014103 |
| 3 | CREDIT_ANNUITY_RATIO | 0.093775 | -0.025832 |
| 4 | AMT_ANNUITY | 0.089130 | -0.003298 |
| 5 | INSTALLMENTS_DAYS_LATE_MEAN | 0.088863 | -0.013102 |
| 6 | EXT_SOURCE_MAX | 0.071082 | -0.001433 |
| 7 | CODE_GENDER_M | 0.066760 | -0.009218 |
| 8 | NAME_EDUCATION_TYPE_Higher education | 0.063122 | -0.004887 |
| 9 | INSTALLMENTS_AMT_PAYMENT_SUM | 0.063049 | -0.011175 |
| 10 | POS_CNT_INSTALMENT_FUTURE_MEAN | 0.055343 | -0.004946 |
| 11 | DAYS_EMPLOYED | 0.054374 | -0.008773 |
| 12 | CODE_GENDER_F | 0.053692 | -0.005011 |
| 13 | BUREAU_AMT_CREDIT_SUM_DEBT_MEAN | 0.053076 | -0.006796 |
| 14 | NAME_FAMILY_STATUS_Married | 0.051978 | -0.003935 |
| 15 | EXT_SOURCE_MIN | 0.050840 | -0.022612 |

## Limitations

- SHAP describes this model's behavior; it does not establish causality.
- Correlated features can share or redistribute attribution.
- One-hot and engineered features require domain-aware interpretation.
- Reason codes support human review and are not autonomous lending decisions.
- Protected attributes require governance even when they are not top drivers.
