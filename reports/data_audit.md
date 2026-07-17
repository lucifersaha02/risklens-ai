# RiskLens AI Data Audit

## Application table

- Rows: 307,511
- Columns: 122
- Positive target rate: 8.07%
- Duplicate applicant IDs: 0
- Columns above 60% missingness: 17
- DAYS_EMPLOYED sentinel records: 55,374

## Relational coverage

| Table | Rows | Applicant coverage |
|---|---:|---:|
| bureau.csv | 1,716,428 | 85.69% |
| previous_application.csv | 1,670,214 | 94.65% |
| installments_payments.csv | 13,605,401 | 94.84% |
| credit_card_balance.csv | 3,840,312 | 28.26% |
| POS_CASH_balance.csv | 10,001,358 | 94.12% |

## Interpretation

The TARGET is imbalanced, so accuracy alone is not an appropriate model metric. Missing values and the DAYS_EMPLOYED sentinel must be handled inside the training pipeline using transformations learned only from training data.
