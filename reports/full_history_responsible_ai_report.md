# RiskLens AI Full-History Responsible-AI Diagnostic

Decision threshold: `0.1667`

This report evaluates model behavior across selected subgroups on validation data. It is diagnostic evidence, not proof of legal or ethical fairness.

## CODE_GENDER

| Group | Rows | Prevalence | ROC-AUC | Recall | FPR | Approval | Calibration gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| F | 20,287 | 7.05% | 0.780 | 39.20% | 8.75% | 89.11% | 0.46% |
| M | 10,462 | 10.06% | 0.773 | 46.96% | 12.52% | 84.02% | 0.78% |
| XNA | 2 | 0.00% | NA | 0.00% | 50.00% | 50.00% | 15.47% |

## AGE_BAND

| Group | Rows | Prevalence | ROC-AUC | Recall | FPR | Approval | Calibration gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| 30_to_39 | 8,084 | 9.67% | 0.792 | 46.42% | 11.74% | 84.91% | 0.64% |
| 40_to_49 | 7,741 | 7.57% | 0.788 | 40.44% | 8.26% | 89.30% | 0.08% |
| 50_to_59 | 6,901 | 5.80% | 0.754 | 33.00% | 6.23% | 92.22% | 0.58% |
| 60_plus | 3,477 | 4.86% | 0.725 | 17.16% | 2.87% | 96.43% | 0.04% |
| under_30 | 4,548 | 12.01% | 0.739 | 53.85% | 21.99% | 74.19% | 0.69% |

## Limitations

- The dataset is historical and may encode past structural inequalities.
- Gender and age diagnostics do not cover every protected or vulnerable group.
- Differences in base rates complicate direct parity comparisons.
- Subgroup metrics do not establish causality or legal compliance.
- A real lender would require jurisdiction-specific governance and review.
- Final holdout data was not accessed for this diagnostic.
