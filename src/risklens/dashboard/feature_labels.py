"""Business-friendly labels for technical model features shown in the dashboard."""

from __future__ import annotations

FEATURE_LABELS = {
    "AMT_INCOME_TOTAL": "Total annual income",
    "AMT_CREDIT": "Requested credit amount",
    "AMT_ANNUITY": "Loan annuity amount",
    "AMT_GOODS_PRICE": "Goods price",
    "CNT_CHILDREN": "Number of children",
    "DAYS_EMPLOYED": "Employment timing",
    "EMPLOYMENT_YEARS": "Employment history",
    "CREDIT_INCOME_RATIO": "Credit-to-income ratio",
    "ANNUITY_INCOME_RATIO": "Annuity-to-income ratio",
    "CREDIT_ANNUITY_RATIO": "Credit-to-annuity ratio",
    "GOODS_CREDIT_RATIO": "Goods-price-to-credit ratio",
    "EXT_SOURCE_1": "External credit signal 1",
    "EXT_SOURCE_2": "External credit signal 2",
    "EXT_SOURCE_3": "External credit signal 3",
    "EXT_SOURCE_MEAN": "Average external credit signal",
    "EXT_SOURCE_MIN": "Lowest external credit signal",
    "EXT_SOURCE_MAX": "Highest external credit signal",
    "EXT_SOURCE_STD": "Variation across external credit signals",
    "EXT_SOURCE_COUNT": "Available external credit signals",
    "BUREAU_DAYS_CREDIT_MAX": "Most recent bureau credit timing",
    "BUREAU_DAYS_CREDIT_MIN": "Oldest bureau credit timing",
    "BUREAU_LOAN_COUNT": "Number of bureau credit records",
    "INSTALLMENTS_DAYS_LATE_MEAN": "Average instalment payment delay",
    "INSTALLMENTS_DAYS_LATE_MAX": "Longest instalment payment delay",
    "INSTALLMENTS_AMT_PAYMENT_SUM": "Total recorded instalment payments",
    "PREVIOUS_APPLICATION_COUNT": "Number of previous applications",
    "FLAG_OWN_CAR_N": "Does not own a car",
    "FLAG_OWN_CAR_Y": "Owns a car",
    "FLAG_OWN_REALTY_N": "Does not own real estate",
    "FLAG_OWN_REALTY_Y": "Owns real estate",
    "NAME_EDUCATION_TYPE_Higher education": "Higher-education category",
    "NAME_EDUCATION_TYPE_Secondary / secondary special": "Secondary-education category",
    "NAME_CONTRACT_TYPE_Cash loans": "Cash-loan application",
    "NAME_CONTRACT_TYPE_Revolving loans": "Revolving-loan application",
}


def business_feature_name(feature: str) -> str:
    """Return a concise business label while retaining deterministic fallback text."""
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    return feature.replace("_", " ").strip().title()


def display_feature_name(feature: str) -> str:
    """Combine the viewer-friendly label with the exact technical feature name."""
    return f"{business_feature_name(feature)} ({feature})"
