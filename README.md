# RiskLens AI

## Explainable Credit Risk Intelligence Platform

RiskLens AI is an end-to-end credit-risk decision-support prototype built using the Home Credit Default Risk dataset.

The platform will provide:

- Leakage-safe relational feature engineering
- Reproducible model training and comparison
- Calibrated probability-of-payment-difficulty predictions
- SHAP-based global and local explanations
- Business-oriented reason codes
- Fairness and subgroup evaluation
- Policy-grounded RAG explanations with citations
- Data and prediction drift monitoring
- FastAPI prediction services
- An interactive Streamlit underwriting dashboard

## Assessment modes

RiskLens exposes two clearly separated assessment workflows:

- **Existing applicant / full history:** looks up a Home Credit `SK_ID_CURR` and uses
  application data plus aggregated bureau, previous-application, instalment, credit-card,
  and POS history. This is the frozen primary research model.
- **New application simulator:** accepts manually entered application-time fields and uses
  a separately trained and calibrated application-only model. It never invents missing
  historical-credit records or sends manual inputs into the full-history model.

Both workflows estimate the probability of the Home Credit payment-difficulty target,
provide SHAP reason codes, and route the case to standard or enhanced human review. Neither
workflow automatically approves or declines a loan.

## New application data contract

The simulator accepts income, requested credit, annuity, goods price, employment duration,
three external credit signals, contract type, income type, education, housing, car ownership,
real-estate ownership, and number of children. External credit signals are assumed to come
from a lender or bureau on the Home Credit 0–1 scale; they are not ordinary self-reported
credit scores.

`TARGET`, applicant identifiers, gender, age/date of birth, and marital status are forbidden
decision inputs. The simulator was developed only from the original training partition using
its own internal train, validation, calibration, and test split. The frozen full-history model
and its final holdout were not modified or reused for simulator development.

## Intended use

This project is an educational and portfolio demonstration of industry-style data-science practices. It is not intended to make real lending decisions.

## Dataset

The project uses the Home Credit Default Risk competition dataset from Kaggle:

https://www.kaggle.com/competitions/home-credit-default-risk/data

Raw competition data is not included in this repository. Users must download it directly from Kaggle and accept the applicable competition rules.
