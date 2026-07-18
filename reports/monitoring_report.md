# RiskLens AI Monitoring Snapshot

Overall severity: **CRITICAL**

- Frozen model version: `697bafda2ed8`
- Reference: `validation` (30,751 rows)
- Current population: `home_credit_application_test_unlabeled` (48,744 rows)
- Prediction PSI: `0.0015` (stable)
- Feature alerts: 0 warning, 1 critical
- Labels available: No; performance drift was not measured.

## Highest feature PSI

| Feature | PSI | Severity |
|---|---:|---|
| CREDIT_ANNUITY_RATIO | 1.6419 | critical |
| GOODS_CREDIT_RATIO | 0.0962 | stable |
| missingindicator_EXT_SOURCE_1 | 0.0821 | stable |
| AMT_CREDIT | 0.0812 | stable |
| AMT_GOODS_PRICE | 0.0809 | stable |
| EXT_SOURCE_COUNT | 0.0700 | stable |
| CREDIT_CARD_RECORD_COUNT | 0.0553 | stable |
| POS_MONTHS_BALANCE_MIN | 0.0529 | stable |
| BUREAU_AMT_CREDIT_SUM_DEBT_MEAN | 0.0451 | stable |
| PREVIOUS_DAYS_DECISION_MIN | 0.0441 | stable |

## Interpretation

PSI thresholds are operational heuristics for investigation. They do not prove performance degradation, causality, fairness, or the need to retrain the frozen model.

Alerts require investigation and do not authorize post-holdout model tuning.
