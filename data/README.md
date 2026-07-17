# Data

This project uses the Home Credit Default Risk competition dataset.

Official source:

https://www.kaggle.com/competitions/home-credit-default-risk/data

## Directory structure

- `raw/home_credit/`: Original, immutable Kaggle CSV files
- `interim/`: Intermediate relational aggregates
- `processed/`: Model-ready feature tables and split metadata

## Data policy

Raw and generated datasets are excluded from Git because of their size and Kaggle's competition terms. Users must download the dataset directly from Kaggle and accept the applicable rules.

Files under `raw/` must never be manually modified. All transformations should write to `interim/` or `processed/`.

## Expected raw files

- `application_train.csv`
- `application_test.csv`
- `bureau.csv`
- `bureau_balance.csv`
- `credit_card_balance.csv`
- `HomeCredit_columns_description.csv`
- `installments_payments.csv`
- `POS_CASH_balance.csv`
- `previous_application.csv`
- `sample_submission.csv`