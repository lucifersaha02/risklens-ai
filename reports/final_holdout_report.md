# RiskLens AI — One-Time Final Holdout Evaluation

**The final holdout has been accessed. Model development is permanently closed.**

Evaluation timestamp (UTC): `2026-07-18T06:32:57+00:00`

Applicants: 30,752

## Frozen model performance

| Metric | Point estimate | 95% bootstrap interval |
|---|---:|---:|
| ROC-AUC | 0.7840 | [0.7754, 0.7932] |
| PR-AUC | 0.2732 | [0.2571, 0.2914] |
| Brier score | 0.06634 | [0.06415, 0.06848] |
| Log loss | 0.23846 | [0.23193, 0.24489] |

## Locked policy performance

Threshold: `0.166667` using the frozen hypothetical 5:1 false-negative:false-positive cost assumption.

| Metric | Point estimate | 95% bootstrap interval |
|---|---:|---:|
| Recall | 42.57% | [40.46%, 44.61%] |
| Precision | 27.28% | [25.92%, 28.75%] |
| Approval rate | 87.40% | [87.04%, 87.77%] |
| Cost units/application | 0.3235 | [0.3120, 0.3350] |

## Governance statement

- Frozen artifact hashes were verified before loading holdout outcomes.
- Calibration and threshold were not refitted or selected on holdout data.
- Results must not trigger feature, hyperparameter, calibration, or threshold tuning.
- Subgroup results remain diagnostic and do not establish fairness or compliance.
- This remains a research prototype, not a production lending system.
