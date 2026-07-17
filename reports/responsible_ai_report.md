# RiskLens AI Responsible-AI Diagnostic

Decision threshold: `0.1667`

This report evaluates model behavior across selected subgroups on validation data. It is diagnostic evidence, not proof of legal or ethical fairness.

## CODE_GENDER

| Group | Rows | Prevalence | ROC-AUC | Recall | FPR | Approval | Calibration gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| F | 20,287 | 7.05% | 0.760 | 33.47% | 7.83% | 90.36% | 0.06% |
| M | 10,462 | 10.06% | 0.763 | 48.48% | 14.34% | 82.23% | 0.06% |
| XNA | 2 | 0.00% | NA | 0.00% | 50.00% | 50.00% | 12.49% |

## AGE_BAND

| Group | Rows | Prevalence | ROC-AUC | Recall | FPR | Approval | Calibration gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| 30_to_39 | 8,084 | 9.67% | 0.769 | 45.65% | 12.96% | 83.88% | 0.27% |
| 40_to_49 | 7,741 | 7.57% | 0.764 | 37.54% | 9.00% | 88.84% | 0.15% |
| 50_to_59 | 6,901 | 5.80% | 0.747 | 26.00% | 4.92% | 93.86% | 0.27% |
| 60_plus | 3,477 | 4.86% | 0.715 | 11.24% | 1.78% | 97.76% | 0.16% |
| under_30 | 4,548 | 12.01% | 0.738 | 52.93% | 21.44% | 74.78% | 0.35% |

## Limitations

- The dataset is historical and may encode past structural inequalities.
- Gender and age diagnostics do not cover every protected or vulnerable group.
- Differences in base rates complicate direct parity comparisons.
- Subgroup metrics do not establish causality or legal compliance.
- A real lender would require jurisdiction-specific governance and review.
- Final holdout data was not accessed for this diagnostic.
