
# Architecture and Experiment Decisions

## Project
Elliptic Sentinel — Graph Neural Network-Based Cryptocurrency Fraud Detection.

## Decision status
Architecture selected for implementation and verification. Model training has not started.

## D-001: Temporal data split
- Training: time steps 1–29.
- Validation: time steps 30–34.
- Exploratory test: time steps 35–49.
- The test period has already been explored during project analysis. Test results must therefore be described as exploratory, not as an untouched final holdout.

## D-002: Graph construction
- Preserve directed edges.
- No cross-time-step edges were found in the dataset.
- Each time step is treated as an independent graph snapshot.
- Cumulative graph views may be used for chronological context, but they do not add historical message-passing neighbors across snapshots.

## D-003: Primary GNN architecture
- Directed GraphSAGE.
- Two message-passing layers.
- Hidden dimension: 64.
- Separate self, incoming-neighbor and outgoing-neighbor transformations.
- Mean aggregation.
- ReLU activation.
- Dropout: 0.3.
- Single-logit binary classification head.
- Time step is not a model input.

## D-004: Degree features
- Include log1p(in-degree) and log1p(out-degree) in the primary model.
- Compute degrees using the graph available at the relevant time step.
- Run a degree-feature ablation without these two inputs.
- Zero-degree aggregation must return a valid zero aggregate.

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
- Learning-rate pilot: 0.001, 0.005 and 0.01.
- Weight decay: 0.0001.
- Maximum epochs: 500.
- Early-stopping patience: 50.
- Primary selection metric: validation average precision.
- CPU full-batch training initially.
- Random seed: 42 for initial verification; multiple seeds are planned for final reporting.

## D-006: Class imbalance
- Use weighted binary cross-entropy.
- Calculate positive-class weight using labelled training nodes only.
- Unknown labels (-1) must never contribute to supervised loss or evaluation metrics.

## D-007: Validation and test discipline
- Validation data may control checkpoint selection, early stopping and threshold selection.
- Test labels must not influence training, checkpoint selection, hyperparameter selection or threshold selection.
- Report validation as development evaluation.
- Report test results as exploratory evaluation.
- Report threshold-free metrics alongside threshold-dependent metrics.

## D-008: Feature preprocessing
- Fit feature scaling on training-period nodes only.
- Apply the fitted transformation to validation and test nodes.
- Preserve transaction ID and feature-row alignment.
- Dataset-level leakage potentially embedded in the original aggregated features remains an unresolved limitation.

## D-009: Baselines and ablations
Planned comparisons:
- XGBoost baseline under the unified split and evaluation protocol.
- Random Forest baseline.
- Directed GraphSAGE primary model.
- Directed GraphSAGE without degree inputs.
- Directional aggregation ablations.
- 93-feature versus 165-feature comparisons.

## D-010: Task framing
The initial system is a retrospective, snapshot-level transaction-risk detection system. It is not presented as a live blockchain fraud-prevention system.

## D-011: Dataset and reporting limitations
- The test period was explored before final model design.
- The dataset's anonymized feature semantics are incompletely verified.
- Potential dataset-level leakage in precomputed aggregate features has not been independently established or ruled out.
- Results must not be described as leakage-free.
- The degree-feature experiment is a modelling hypothesis, not evidence of causal fraud detection.

## Evidence labels
- [V] Verified directly by project code, tests or dataset audits.
- [U] Explicit project decision.
- [P] Preliminary claim from an external preprint; not independently verified.
- [H] Hypothesis requiring further testing.

## Current status
Architecture decisions recorded. Integrity tests and model implementation are pending.
