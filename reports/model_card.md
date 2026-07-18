# RiskLens AI — Governed Full-History Model Card

## Release status

**Pre-holdout frozen research prototype. Not approved for production lending.**

This model estimates Home Credit default risk for portfolio demonstration and human decision-support research. It must not independently approve, decline, price, or otherwise determine access to credit.

## Model overview

| Item | Value |
|---|---|
| Algorithm | XGBoost binary classifier |
| Inputs | Application data plus aggregated credit history |
| Training applicants | 215,257 |
| Validation applicants | 30,751 |
| Calibration applicants | 30,751 |
| Calibration method | sigmoid |
| Locked policy threshold | 0.166667 |
| Final holdout | Sealed at model-card freeze |

## Intended use

- Portfolio demonstration of reproducible credit-risk modeling.
- Analyst decision support with documented probability and reason codes.
- Model-risk, calibration, drift, and subgroup diagnostic exercises.

## Prohibited use

- Autonomous credit approval, decline, pricing, or limit assignment.
- Use as a legally sufficient adverse-action notice.
- Deployment in a jurisdiction without legal, compliance, privacy, and model-risk review.
- Inferring causality from SHAP values or subgroup associations.
- Retraining or threshold tuning after viewing the final holdout.

## Data and evaluation design

- Source: Kaggle Home Credit Default Risk competition dataset.
- Applicant-level deterministic stratified split: 70% train, 10% validation, 10% calibration, and 10% final holdout.
- Relational history aggregates are target-free and joined by `SK_ID_CURR`.
- Preprocessing statistics are learned inside the training pipeline.
- Final holdout was not accessed during feature development, selection, calibration, threshold definition, fairness analysis, or SHAP analysis.

## Validation performance

| Metric | Value |
|---|---:|
| ROC-AUC | 0.7802 |
| PR-AUC | 0.2731 |
| Brier score | 0.06633 |
| Log loss | 0.23918 |
| Three-fold CV ROC-AUC | 0.7787 ± 0.0039 |
| Three-fold CV PR-AUC | 0.2688 ± 0.0096 |

## Calibration and decision policy

Sigmoid calibration was selected using 15,376 internally reserved calibration records and refitted on the entire calibration split.

The locked threshold `0.166667` follows a hypothetical false-negative:false-positive cost ratio of 5:1. These costs are portfolio assumptions, not lender estimates.

| Validation operating metric | Value |
|---|---:|
| Recall | 42.49% |
| Precision | 27.17% |
| Approval rate | 87.37% |
| Review/decline rate | 12.63% |
| Expected cost units/application | 0.3242 |

## Feature governance

Policy: `sensitive_attributes_audit_only_v1`

Direct gender, age, and family-status attributes are excluded from automated risk scoring and retained only for diagnostic governance and subgroup audits.

Excluded from all model preprocessing and scoring:

- `CODE_GENDER`
- `DAYS_BIRTH`
- `AGE_YEARS`
- `NAME_FAMILY_STATUS`

The excluded source attributes remain available only for subgroup auditing. Proxy effects may remain and require monitoring.

## Responsible-AI diagnostics

These validation diagnostics are not proof of fairness or legal compliance.

| Gap (max minus min) | Value |
|---|---:|
| Gender recall | 7.75% |
| Gender false-positive rate | 3.77% |
| Age-band recall | 36.69% |
| Age-band false-positive rate | 19.12% |

Age-band gaps remain material even after direct age exclusion, consistent with different base rates and possible proxy information. Human governance and ongoing subgroup monitoring are required.

## Explainability

SHAP analysis used 2,000 validation applicants and 688 transformed features. SHAP values explain the raw XGBoost margin before sigmoid calibration and are not probability percentage-point changes.

Leading global features:

- `EXT_SOURCE_MEAN`
- `AMT_ANNUITY`
- `GOODS_CREDIT_RATIO`
- `CREDIT_ANNUITY_RATIO`
- `INSTALLMENTS_DAYS_LATE_MEAN`
- `EXT_SOURCE_MAX`
- `NAME_EDUCATION_TYPE_Higher education`
- `EXT_SOURCE_MIN`
- `DAYS_EMPLOYED`
- `INSTALLMENTS_AMT_PAYMENT_SUM`

## Known limitations

- Competition data is historical, anonymized, and not representative of every market.
- Historical outcomes may encode structural inequities and policy effects.
- External-score fields are opaque and would require vendor governance in production.
- Missingness and dataset shift may materially affect predictions.
- SHAP attribution is descriptive, not causal.
- Subgroup diagnostics cover gender and age bands only.
- Hypothetical costs do not establish real business value.

## Human oversight and controls

- A qualified analyst must review model output and reason codes.
- Users must be able to escalate, override, and document decisions.
- Input validation, access control, audit logging, monitoring, and incident response are required before any production consideration.
- Performance, calibration, drift, and subgroup behavior must be monitored.
- Final holdout results must be reported once and must not trigger model tuning.

## Reproducibility freeze

The accompanying `model_governance_freeze.json` records SHA-256 hashes for the candidate model, calibrated model, and modeling configuration. Holdout evaluation must fail if those hashes no longer match.
