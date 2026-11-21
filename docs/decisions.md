# Architecture and Experiment Decisions

## Project

Elliptic Sentinel — Graph Neural Network-Based Cryptocurrency Fraud Detection.

## Decision status

The core data pipeline, XGBoost baseline, GCN, and Directed GraphSAGE training and evaluation workflows have been implemented. The test suite has passed 103 tests.

The reported test results are exploratory because the test period was examined during project development.

## D-001: Temporal data split

- Training: time steps 1–29.
- Validation: time steps 30–34.
- Exploratory test: time steps 35–49.
- The test period has already been explored during project analysis.
- Test results must not be described as results from an untouched final holdout.

## D-002: Graph construction

- Preserve the original directed edge information.
- No cross-time-step edges were found in the dataset.
- Each time step is treated as an independent graph snapshot.
- Cumulative graph views provide chronological context but do not add historical message-passing neighbours across snapshots.
- The GCN uses an undirected graph representation.
- Directed GraphSAGE retains separate incoming- and outgoing-neighbour information.

## D-003: Primary GNN architecture

The primary directed graph model is GraphSAGE.

- Two message-passing layers.
- Hidden dimension: 64.
- Separate self, incoming-neighbour, and outgoing-neighbour transformations.
- Mean aggregation.
- ReLU activation.
- Dropout: 0.3.
- Single-logit binary classification head.
- Time step is not a model input.
- Log-transformed in-degree and out-degree features are included.

The GCN is implemented as a separate two-layer graph convolutional model with hidden dimension 64 and dropout 0.3.

## D-004: Degree features

- Include log1p(in-degree) and log1p(out-degree) in the Directed GraphSAGE model.
- Compute degrees using the graph available to the relevant graph view.
- Zero-degree aggregation must return a valid zero aggregate.
- Degree-feature and directional ablations remain possible future experiments; no unreported ablation result is claimed here.

### Supporting graph audit

- Nodes: 203,769.
- Directed edges: 234,355.
- Zero in-degree nodes: 55,322.
- Zero out-degree nodes: 37,424.
- Nodes with both zero in-degree and zero out-degree: 0.
- Maximum in-degree: 284.
- Maximum out-degree: 472.

## D-005: Training configuration

- Optimizer: Adam.
- Learning-rate pilot candidates: 0.001, 0.005, and 0.01.
- Weight decay: 0.0001.
- Maximum epochs: 500.
- Early-stopping patience: 50.
- Primary model-selection metric: validation average precision.
- CPU full-batch training.
- Final GNN experiments use random seeds 42–46.
- Learning-rate selection and checkpoint selection use validation results.

## D-006: Class imbalance

- Use weighted binary cross-entropy for the GNN training workflows.
- Calculate positive-class weight using labelled training nodes only.
- Unknown labels (-1) must never contribute to supervised loss or evaluation metrics.
- Unknown transactions may remain part of graph context.

## D-007: Validation and test discipline

- Validation data may control checkpoint selection, early stopping, and threshold selection.
- Test labels must not influence training, checkpoint selection, hyperparameter selection, or threshold selection.
- Report validation as development evaluation.
- Report test results as exploratory evaluation.
- Report threshold-free metrics alongside threshold-dependent metrics.
- Threshold-dependent classification thresholds are selected using labelled validation targets.
- Test-period ranking uses model scores, not labels, to select alerts.

## D-008: Feature preprocessing

- Fit feature scaling on training-period nodes only.
- Apply the fitted transformation to validation and test nodes.
- Preserve transaction ID and feature-row alignment.
- Dataset-level leakage potentially embedded in the original aggregated features remains an unresolved limitation.
- Training-only scaling does not establish that the original supplied features are free from temporal leakage.

## D-009: Implemented models and baselines

Implemented:
- XGBoost transaction-classification baseline.
- Two-layer GCN.
- Two-layer Directed GraphSAGE.
- Shared evaluation utilities for threshold-based and fixed-capacity ranking metrics.

Not claimed as completed experiments:
- Random Forest comparison.
- GraphSAGE degree-feature ablation.
- Directional aggregation ablations.
- 93-feature versus 165-feature comparison.

## D-010: Task framing

The system is a retrospective, period-level transaction-risk detection research system.

It is not presented as a live blockchain fraud-prevention system or as a validated per-transaction online inference service.

## D-011: Dataset and reporting limitations

- The test period was explored before final project reporting.
- The dataset's anonymised feature semantics are incompletely verified.
- Potential dataset-level leakage in precomputed aggregated features has not been independently established or ruled out.
- Results must not be described as leakage-free.
- The degree-feature experiment is a modelling choice, not evidence of causal fraud detection.
- The dataset contains a substantial unlabelled portion.
- Historical evaluation does not establish production performance.

## D-012: Fixed-capacity transaction-risk ranking

- The operational ranking analysis selects the top 1% of transactions by model score.
- Ranking is performed across the full relevant period transaction set, including transactions with unknown labels.
- Labels are not used to select alerts.
- The number of selected alerts is exactly 1% rounded up, subject to a minimum of one alert.
- Ties are resolved deterministically using ascending string representation of transaction IDs.
- Selected alerts are subsequently matched against labelled targets for metric calculation.
- Unlabelled alerts are not counted as labelled true positives or false positives.

## D-013: XGBoost selected baseline

- The selected XGBoost baseline uses the validation-selected best iteration.
- The saved model's best iteration is 0, while the booster contains 51 rounds.
- The exported first-tree score represents the selected baseline.
- The all-51-tree score is retained as a separate diagnostic and is not substituted for the selected baseline.
- XGBoost is a single-run baseline; its variability is not directly comparable to the five-seed GNN results.

## D-014: Exploratory test results

### Threshold-free metrics

| Model | Average Precision | ROC-AUC |
|---|---:|---:|
| XGBoost | 0.52089 | 0.88488 |
| GCN | 0.48234 ± 0.04760 | 0.86282 ± 0.01441 |
| GraphSAGE | 0.37515 ± 0.04129 | 0.87403 ± 0.00482 |

### Top-1% alert-capacity metrics

| Model | Precision | Recall | F1-score |
|---|---:|---:|---:|
| XGBoost selected baseline | 0.00000 | 0.00000 | 0.00000 |
| GCN | 0.78822 ± 0.06061 | 0.12558 ± 0.01362 | 0.21660 ± 0.02260 |
| GraphSAGE | 0.43356 ± 0.09308 | 0.03250 ± 0.00670 | 0.06034 ± 0.01215 |

- GNN results are mean ± standard deviation across five seeds (42–46).
- XGBoost is a single-run baseline.
- Under the fixed top-1% alert-capacity setting, GCN achieved the strongest reported precision and recall among these experiments.
- These results are exploratory and are not evidence of performance on an untouched holdout.

## D-015: Automated testing

- The automated test suite completed with 103 passed and 0 failed.
- Two PyTorch deprecation warnings were observed in the test run.
- The test suite covers core data, graph, model, training, and evaluation functionality.

## Evidence labels

- [V] Verified directly by project code, tests, or dataset audits.
- [U] Explicit project decision.
- [P] Preliminary claim from an external preprint; not independently verified.
- [H] Hypothesis requiring further testing.

## Current status

Core model implementations, training workflows, evaluation workflows, and automated tests are complete for the current experimental scope.

Further work may include ablation studies, additional baselines, explainability, monitoring, and deployment. These are not represented as completed components.
