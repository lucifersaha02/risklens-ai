# RiskLens AI Full-History Responsible-AI Diagnostic

Decision threshold: `0.1667`

This report evaluates model behavior across selected subgroups on validation data. It is diagnostic evidence, not proof of legal or ethical fairness.

## CODE_GENDER

| Group | Rows | Prevalence | ROC-AUC | Recall | FPR | Approval | Calibration gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| F | 20,287 | 7.05% | 0.781 | 37.81% | 8.05% | 89.86% | 0.05% |
| M | 10,462 | 10.06% | 0.776 | 50.38% | 14.34% | 82.04% | 0.05% |
| XNA | 2 | 0.00% | NA | 0.00% | 50.00% | 50.00% | 13.14% |

## AGE_BAND

| Group | Rows | Prevalence | ROC-AUC | Recall | FPR | Approval | Calibration gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| 30_to_39 | 8,084 | 9.67% | 0.790 | 48.98% | 12.76% | 83.73% | 0.27% |
| 40_to_49 | 7,741 | 7.57% | 0.792 | 41.98% | 8.82% | 88.67% | 0.11% |
| 50_to_59 | 6,901 | 5.80% | 0.761 | 31.75% | 5.92% | 92.58% | 0.37% |
| 60_plus | 3,477 | 4.86% | 0.731 | 17.16% | 2.84% | 96.46% | 0.12% |
| under_30 | 4,548 | 12.01% | 0.743 | 52.38% | 20.61% | 75.57% | 0.18% |

## Limitations

- The dataset is historical and may encode past structural inequalities.
- Gender and age diagnostics do not cover every protected or vulnerable group.
- Differences in base rates complicate direct parity comparisons.
- Subgroup metrics do not establish causality or legal compliance.
- A real lender would require jurisdiction-specific governance and review.
- Final holdout data was not accessed for this diagnostic.
