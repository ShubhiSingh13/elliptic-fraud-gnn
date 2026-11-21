"""Elliptic Sentinel: artifact-backed research dashboard."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

FILES = {
    "XGBoost": ARTIFACTS / "xgboost_baseline_metrics.json",
    "GCN": ARTIFACTS / "gcn" / "evaluation" / "five_seed_evaluation_summary.json",
    "GraphSAGE": ARTIFACTS / "gnn" / "five_seed_evaluation_summary.json",
}


st.set_page_config(
    page_title="Elliptic Sentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_artifacts() -> dict:
    results = {}
    for name, path in FILES.items():
        if not path.exists():
            st.error(f"Required evaluation artifact not found: {path.relative_to(ROOT)}")
            st.stop()
        try:
            results[name] = load_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            st.error(f"Could not read {name} evaluation results: {exc}")
            st.stop()
    return results


def metric_value(summary: dict, section: str, metric: str) -> tuple[float, float]:
    item = summary["test_evaluation"][section][metric]
    return float(item["mean"]), float(item.get("standard_deviation", 0.0))


def fmt_percent(value: float) -> str:
    return f"{value:.2%}"


def fmt_mean_std(mean: float, std: float) -> str:
    return f"{mean:.3f} ± {std:.3f}"


def show_metric_card(label: str, value: str, help_text: str | None = None):
    st.metric(label=label, value=value, help=help_text)


def overview(data: dict):
    st.title("🛡️ Elliptic Sentinel")
    st.caption(
        "Graph Neural Network-Based Cryptocurrency Fraud Detection System"
    )

    st.markdown(
        """
        An experimental transaction-risk research platform using the Elliptic
        Bitcoin transaction graph. It compares a conventional XGBoost baseline
        with graph-based GCN and directed GraphSAGE models.
        """
    )

    st.divider()
    st.subheader("Dataset at a glance")

    cols = st.columns(4)
    for col, label, value in zip(
        cols,
        ["Transactions", "Directed edges", "Time steps", "Features"],
        ["203,769", "234,355", "49", "165"],
    ):
        with col:
            show_metric_card(label, value)

    st.divider()
    st.subheader("Temporal evaluation split")

    split_data = pd.DataFrame(
        {
            "Split": ["Training", "Validation", "Test"],
            "Time steps": ["1–29", "30–34", "35–49"],
            "Transactions": [120804, 15461, 67504],
            "Labelled transactions": [26381, 3513, 16670],
            "Unknown labels": [94423, 11948, 50834],
        }
    )
    st.dataframe(
        split_data,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Unknown-label transactions remain part of the graph context but are "
        "excluded from supervised metric calculations."
    )

    st.divider()
    st.subheader("Models evaluated")

    model_cols = st.columns(3)
    descriptions = [
        (
            "XGBoost baseline",
            "A gradient-boosted tree baseline using transaction features. "
            "The reported baseline uses the selected first boosting tree.",
            "One evaluation run",
        ),
        (
            "GCN",
            "A two-layer Graph Convolutional Network evaluated on an "
            "undirected graph.",
            "Five random seeds",
        ),
        (
            "Directed GraphSAGE",
            "A directed message-passing model using incoming/outgoing "
            "neighbour information and degree features.",
            "Five random seeds",
        ),
    ]

    for col, (name, description, runs) in zip(model_cols, descriptions):
        with col:
            st.markdown(f"**{name}**")
            st.write(description)
            st.caption(runs)


def model_comparison(data: dict):
    st.title("Model performance")
    st.caption(
        "Test-period results for time steps 35–49. GCN and GraphSAGE values "
        "are mean ± standard deviation across five seeds."
    )

    xgb = data["XGBoost"]
    gcn = data["GCN"]
    sage = data["GraphSAGE"]

    st.subheader("Ranking performance")

    rows = []
    for name, summary in [("GCN", gcn), ("GraphSAGE", sage)]:
        section = "f1_threshold_metrics_mean_and_std"
        ap, ap_sd = metric_value(summary, section, "average_precision")
        auc, auc_sd = metric_value(summary, section, "roc_auc")
        rows.append(
            {
                "Model": name,
                "Average precision (AP)": fmt_mean_std(ap, ap_sd),
                "ROC-AUC": fmt_mean_std(auc, auc_sd),
                "Evaluation runs": "5 seeds",
            }
        )

    rows.append(
        {
            "Model": "XGBoost",
            "Average precision (AP)": f"{xgb['average_precision']:.3f}",
            "ROC-AUC": f"{xgb['roc_auc']:.3f}",
            "Evaluation runs": "1 run",
        }
    )

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "AP and ROC-AUC assess ranking quality. XGBoost is a single run, "
        "whereas GCN and GraphSAGE are five-seed summaries; this is not an "
        "equal-repetition comparison."
    )

    st.divider()
    st.subheader("Validation-selected F1 threshold")

    threshold_rows = [
        {
            "Model": "XGBoost",
            "Accuracy": fmt_percent(xgb["accuracy"]),
            "Precision": fmt_percent(xgb["precision"]),
            "Recall": fmt_percent(xgb["recall"]),
            "F1": fmt_percent(xgb["f1"]),
        }
    ]

    for name, summary in [("GCN", gcn), ("GraphSAGE", sage)]:
        section = "f1_threshold_metrics_mean_and_std"
        values = {}
        for metric in ["accuracy", "precision", "recall", "f1"]:
            mean, std = metric_value(summary, section, metric)
            values[metric] = fmt_mean_std(mean, std)
        threshold_rows.append(
            {
                "Model": name,
                "Accuracy": values["accuracy"],
                "Precision": values["precision"],
                "Recall": values["recall"],
                "F1": values["f1"],
            }
        )

    st.dataframe(
        pd.DataFrame(threshold_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Each model's F1 threshold was selected using labelled validation "
        "targets. The threshold was then frozen for test evaluation."
    )

    st.divider()
    st.subheader("Fixed-capacity investigation: top 1%")

    capacity_rows = []
    for name, summary in [("GCN", gcn), ("GraphSAGE", sage)]:
        section = "capacity_metrics_mean_and_std"
        precision, precision_sd = metric_value(summary, section, "precision")
        recall, recall_sd = metric_value(summary, section, "recall")
        f1, f1_sd = metric_value(summary, section, "f1")
        alerts, alerts_sd = metric_value(summary, section, "alert_count")

        capacity_rows.append(
            {
                "Model": name,
                "Labelled-alert precision": fmt_mean_std(
                    precision, precision_sd
                ),
                "Recall": fmt_mean_std(recall, recall_sd),
                "F1": fmt_mean_std(f1, f1_sd),
                "Labelled alerts": fmt_mean_std(alerts, alerts_sd),
                "Selected transactions": "676 / seed",
            }
        )

    capacity_rows.append(
        {
            "Model": "XGBoost",
            "Labelled-alert precision": "0.000",
            "Recall": "0.000",
            "F1": "0.000",
            "Labelled alerts": "42",
            "Selected transactions": "676",
        }
    )

    st.dataframe(
        pd.DataFrame(capacity_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.warning(
        "Capacity precision is calculated among selected transactions with "
        "known labels only. Transactions with unknown labels are excluded, "
        "not counted as false positives. Therefore, labelled-alert precision "
        "is not precision across all 676 selected transactions. The XGBoost "
        "top-1% result is its separately audited first-tree ranking result."
    )

    st.markdown(
        """
        **How to interpret the comparison**

        - XGBoost has the strongest reported AP and ROC-AUC in this experiment.
        - GCN has stronger mean precision and recall under the fixed top-1%
          capacity policy than the other reported models.
        - GraphSAGE has a different precision/recall trade-off and uses
          directed message passing.
        - There is no single winner independent of the operational objective.
        """
    )


def methodology(data: dict):
    st.title("Methodology & limitations")

    st.subheader("Evaluation design")
    st.markdown(
        """
        - **Dataset:** Elliptic Bitcoin transaction graph.
        - **Labels:** illicit = 1, licit = 0, unknown = -1.
        - **Temporal split:** training on time steps 1–29, validation on
          30–34, and test on 35–49.
        - **Feature scaling:** scaler fitted on training rows only.
        - **Unknown labels:** retained as graph context but excluded from
          supervised targets and reported classification metrics.
        - **Capacity policy:** deterministic top 1% ranking, selected without
          using labels, with ascending transaction ID as the tie-break.
        """
    )

    st.subheader("Important interpretation limits")
    st.warning(
        "Test-period results are exploratory. Test data was inspected during "
        "development, so these results are not from an untouched holdout."
    )

    st.markdown(
        """
        **Not a live transaction-scoring service**

        The current dashboard presents saved evaluation artifacts. The
        experiment evaluates transactions in period-level graph views; it
        should not be interpreted as a validated real-time or online
        transaction prediction service.

        **Source-level information leakage risk**

        Fitting the scaler on training rows does not, by itself, prove that
        every source feature is free from global or future information.
        Aggregated source features require careful interpretation.

        **Model comparison caveats**

        XGBoost is represented by one reported run, while GCN and GraphSAGE
        use five seeds. Model architecture, graph direction, degree features,
        and evaluation variability differ. The results should be read as
        experimental evidence, not a definitive production ranking.
        """
    )

    st.subheader("Saved evaluation status")
    status_rows = []
    for name, summary in data.items():
        status_rows.append(
            {
                "Model": name,
                "Evaluation status": summary.get(
                    "evaluation_status", "not specified"
                ),
                "Test status": summary.get(
                    "test_status",
                    summary.get("evaluation_status", "not specified"),
                ),
            }
        )
    st.dataframe(
        pd.DataFrame(status_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Capacity selection policy")
    for name in ["GCN", "GraphSAGE"]:
        policy = data[name].get("capacity_policy", {})
        with st.expander(f"{name} policy"):
            st.json(policy)

    st.caption(
        "This is a research prototype and should not be used to make "
        "financial, compliance, or law-enforcement decisions."
    )


def main():
    data = load_artifacts()

    with st.sidebar:
        st.title("🛡️ Elliptic Sentinel")
        st.caption("Research dashboard")
        page = st.radio(
            "Navigation",
            ["Overview", "Model performance", "Methodology & limitations"],
        )
        st.divider()
        st.caption("Artifact-backed view")
        st.caption("No model retraining or live inference is performed.")

    if page == "Overview":
        overview(data)
    elif page == "Model performance":
        model_comparison(data)
    else:
        methodology(data)

    st.divider()
    st.caption(
        "Elliptic Sentinel · Experimental research prototype · "
        "Metrics loaded from local evaluation artifacts"
    )


if __name__ == "__main__":
    main()
