# Elliptic Sentinel

Fraud detection on the Elliptic Bitcoin transaction graph, built as an end-to-end ML engineering project: temporal-leakage-safe evaluation, tabular and graph models (XGBoost, GCN, GraphSAGE), explainability, drift monitoring, a FastAPI service and a Streamlit dashboard.

> **Status:** Work in progress. Project scaffolding only; no model results yet.

## Planned Components

- Data pipeline with chronological train, validation and test splits, versioned with DVC
- Machine learning baselines using Logistic Regression and XGBoost
- Graph neural networks using GCN and GraphSAGE
- Explainability using SHAP and GNNExplainer
- Experiment tracking and model management with MLflow
- Simulated production monitoring using historical replay and drift detection
- FastAPI inference service and Streamlit dashboard
- Docker packaging and CI using GitHub Actions

## Repository Structure

- `src/elliptic_fraud/` — Core Python package
- `app/` — Streamlit dashboard
- `tests/` — Automated tests
- `data/` — Dataset storage
- `artifacts/` — Model and experiment artifacts
- `docs/` — Architecture, data and model documentation
- `docker/` — Container configuration

## Environment

- Python 3.11
- Apple Silicon development environment
- PyTorch and PyTorch Geometric
- XGBoost, Scikit-learn and SHAP
- FastAPI and Streamlit

Exact dependency versions and installation instructions will be documented during environment configuration.

## Important Limitations

- The project uses the Elliptic Bitcoin transaction dataset.
- The monitoring workflow is based on simulated chronological historical replay, not a live blockchain feed.
- Dataset features are anonymized and cannot directly score arbitrary raw Bitcoin transactions.

## Project Status

Currently setting up the project structure and development environment. Model training, evaluation and deployment are planned future stages.

## License

To be decided.
