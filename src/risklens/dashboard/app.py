"""Streamlit dashboard consuming only the authenticated RiskLens AI API."""

from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from risklens.dashboard.client import RiskLensAPIClient, RiskLensAPIError

st.set_page_config(
    page_title="RiskLens AI",
    page_icon="🔍",
    layout="wide",
)


def risk_gauge(probability: float, threshold: float) -> go.Figure:
    """Build a calibrated-risk gauge with the frozen threshold marked."""
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "valueformat": ".2f"},
            title={"text": "Calibrated default probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#355C7D"},
                "steps": [
                    {"range": [0, threshold * 100], "color": "#D8F3DC"},
                    {"range": [threshold * 100, 100], "color": "#FFE5D9"},
                ],
                "threshold": {
                    "line": {"color": "#B02A37", "width": 4},
                    "value": threshold * 100,
                },
            },
        )
    )
    figure.update_layout(height=320, margin={"l": 20, "r": 20, "t": 60, "b": 20})
    return figure


def reason_chart(prediction: object) -> go.Figure:
    """Plot local raw-margin contributions in both directions."""
    records = [
        *prediction.reason_codes.risk_increasing,
        *prediction.reason_codes.risk_reducing,
    ]
    frame = pd.DataFrame([record.model_dump() for record in records])
    frame = frame.sort_values("shap_value")
    colors = ["#2E8B57" if value < 0 else "#B02A37" for value in frame["shap_value"]]
    figure = go.Figure(
        go.Bar(
            x=frame["shap_value"],
            y=frame["feature"],
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}<br>SHAP: %{x:.4f}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Applicant-level raw-margin drivers",
        xaxis_title="SHAP value",
        height=420,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    return figure


def render_applicant(client: RiskLensAPIClient) -> None:
    """Render applicant lookup, risk, workflow, and explanations."""
    st.subheader("Applicant decision support")
    st.caption("Research prototype. A qualified human must make and document every decision.")
    left, right = st.columns([2, 1])
    applicant_id = left.number_input("Applicant ID", min_value=1, value=100001, step=1)
    reason_count = right.slider("Reasons per direction", 1, 10, 5)
    if st.button("Score applicant", type="primary", width="stretch"):
        try:
            prediction = client.predict(int(applicant_id), reason_count)
        except RiskLensAPIError as error:
            st.error(str(error))
            return
        gauge_column, action_column = st.columns([2, 1])
        with gauge_column:
            st.plotly_chart(
                risk_gauge(
                    prediction.calibrated_default_probability,
                    prediction.decision_threshold,
                ),
                width="stretch",
            )
        with action_column:
            st.metric("Frozen threshold", f"{prediction.decision_threshold:.2%}")
            st.metric("Model version", prediction.model_version)
            if prediction.policy_action == "enhanced_manual_review_recommended":
                st.warning("Enhanced manual review recommended")
            else:
                st.success("Standard human review workflow")
            st.info("This output is not an autonomous decision or adverse-action notice.")
        st.plotly_chart(reason_chart(prediction), width="stretch")
        st.caption(
            "SHAP values explain the raw XGBoost margin before sigmoid calibration; "
            "they are not probability percentage-point changes and do not establish causality."
        )


def render_portfolio(client: RiskLensAPIClient) -> None:
    """Render immutable final performance and subgroup evidence."""
    st.subheader("Frozen portfolio evidence")
    try:
        summary = client.portfolio_summary()
    except RiskLensAPIError as error:
        st.error(str(error))
        return
    metrics = summary.metrics
    columns = st.columns(4)
    columns[0].metric("ROC-AUC", f"{metrics.roc_auc:.4f}")
    columns[1].metric("PR-AUC", f"{metrics.average_precision:.4f}")
    columns[2].metric("Brier score", f"{metrics.brier_score:.4f}")
    columns[3].metric("Holdout applicants", f"{summary.holdout_rows:,}")
    columns = st.columns(4)
    columns[0].metric("Recall", f"{metrics.recall:.2%}")
    columns[1].metric("Precision", f"{metrics.precision:.2%}")
    columns[2].metric("Approval rate", f"{metrics.approval_rate:.2%}")
    columns[3].metric("Cost/application", f"{metrics.cost_per_application:.4f}")

    gaps = summary.subgroup_gaps
    gap_frame = pd.DataFrame(
        {
            "Diagnostic": ["Gender recall", "Gender FPR", "Age recall", "Age FPR"],
            "Gap": [
                gaps.gender_recall,
                gaps.gender_false_positive_rate,
                gaps.age_band_recall,
                gaps.age_band_false_positive_rate,
            ],
        }
    )
    figure = go.Figure(
        go.Bar(x=gap_frame["Diagnostic"], y=gap_frame["Gap"] * 100, marker_color="#6C5B7B")
    )
    figure.update_layout(
        title="Final holdout subgroup max-minus-min gaps",
        yaxis_title="Percentage points",
        height=380,
    )
    st.plotly_chart(figure, width="stretch")
    st.warning(
        "Subgroup gaps are diagnostic—not proof of fairness or legal compliance. "
        "Age-band disparities remain material."
    )
    st.caption(
        f"One-time holdout evaluation: {summary.evaluated_at_utc}. "
        "Post-holdout tuning is prohibited."
    )


def render_governance(client: RiskLensAPIClient) -> None:
    """Render frozen-model status and sensitive-feature controls."""
    st.subheader("Model governance")
    try:
        info = client.model_info()
    except RiskLensAPIError as error:
        st.error(str(error))
        return
    st.json(info.model_dump())
    st.markdown("**Excluded from automated risk scoring:**")
    for feature in info.excluded_decision_features:
        st.markdown(f"- `{feature}`")
    st.error("Post-holdout model tuning is prohibited and technically recorded in the freeze.")


def render_monitoring(client: RiskLensAPIClient) -> None:
    """Render the latest unlabeled data-quality and drift snapshot."""
    st.subheader("Frozen-model monitoring")
    try:
        summary = client.monitoring_summary()
    except RiskLensAPIError as error:
        st.error(str(error))
        return

    if summary.overall_severity == "critical":
        st.error("Critical feature-drift investigation required")
    elif summary.overall_severity == "warning":
        st.warning("Monitoring warning requires investigation")
    else:
        st.success("Monitored population is stable")

    prediction = summary.prediction_drift
    columns = st.columns(4)
    columns[0].metric("Prediction PSI", f"{prediction.psi:.4f}")
    columns[1].metric("Prediction status", prediction.severity.title())
    columns[2].metric("Reference mean risk", f"{prediction.reference_mean_probability:.2%}")
    columns[3].metric("Current mean risk", f"{prediction.current_mean_probability:.2%}")

    counts = summary.feature_severity_counts
    st.caption(
        f"{counts.stable} stable features · {counts.warning} warnings · "
        f"{counts.critical} critical alerts"
    )
    drift_frame = pd.DataFrame([item.model_dump() for item in summary.top_feature_drift[:10]])
    color_map = {"stable": "#2E8B57", "warning": "#E0A800", "critical": "#B02A37"}
    colors = [color_map[level] for level in drift_frame["severity"]]
    figure = go.Figure(
        go.Bar(
            x=drift_frame["psi"],
            y=drift_frame["feature"],
            orientation="h",
            marker_color=colors,
        )
    )
    figure.update_layout(
        title="Highest transformed-feature PSI",
        xaxis_title="Population Stability Index",
        yaxis={"autorange": "reversed"},
        height=470,
    )
    st.plotly_chart(figure, width="stretch")

    quality = summary.data_quality
    st.markdown("**Data-quality checks**")
    st.write(
        {
            "duplicate_id_rate": quality.duplicate_id_rate,
            "target_present": quality.target_present,
            "alerts": quality.alerts,
        }
    )
    st.warning(
        "Labels are unavailable, so this snapshot measures population drift—not actual "
        "performance degradation. Alerts require investigation and do not authorize tuning."
    )
    st.caption(summary.interpretation)


st.title("RiskLens AI")
st.caption("Explainable Credit Risk Intelligence Platform")
with st.sidebar:
    st.header("API connection")
    api_url = st.text_input("API URL", value=os.getenv("RISKLENS_API_URL", "http://127.0.0.1:8000"))
    api_key = st.text_input(
        "API key",
        value=os.getenv("RISKLENS_API_KEY", ""),
        type="password",
    )
    st.caption("Credentials remain in this dashboard session and are not written to Git.")

if not api_key:
    st.info("Enter the API key in the sidebar to connect.")
    st.stop()

api_client = RiskLensAPIClient(api_url, api_key)
try:
    health = api_client.health()
    st.sidebar.success(f"Connected · model {health['model_version']}")
except RiskLensAPIError as error:
    st.error(str(error))
    st.stop()

applicant_tab, portfolio_tab, monitoring_tab, governance_tab = st.tabs(
    ["Applicant", "Portfolio evidence", "Monitoring", "Governance"]
)
with applicant_tab:
    render_applicant(api_client)
with portfolio_tab:
    render_portfolio(api_client)
with monitoring_tab:
    render_monitoring(api_client)
with governance_tab:
    render_governance(api_client)
