# Elliptic Sentinel

**Graph Neural Network-Based Cryptocurrency Fraud Detection**

Elliptic Sentinel is an end-to-end research and ML-engineering project
for identifying illicit Bitcoin transactions in the Elliptic transaction
graph. It compares a tabular XGBoost baseline with two graph-learning
approaches---GCN and directed GraphSAGE---and evaluates both
conventional classification and a fixed-capacity alerting workflow.

> **Research status:** Models have been trained and evaluated, the
> dashboard and evaluation pipeline are implemented, and the automated
> test suite passes. Test-set analyses have been used during
> development; the test set is therefore exploratory, not an untouched
> final holdout.

## Project highlights

-   Temporal train, validation, and test partitions across 49 time
    steps.
-   Three model families: XGBoost, Graph Convolutional Network (GCN),
    and directed GraphSAGE.
-   Validation-only threshold calibration and a deterministic top-1%
    alert-capacity evaluation.
-   Unknown-label transactions retained as graph context but excluded
    from supervised targets and labelled metrics.
-   Multi-seed evaluation for GCN and GraphSAGE (seeds 42--46).
-   Streamlit dashboard for project overview, model performance,
    methodology, and limitations.
-   Automated tests covering data splits, graph construction, training
    utilities, model behavior, and evaluation.

## Dataset

The Elliptic dataset contains:

  Item                              Count
  ----------------------------- ---------
  Transactions (nodes)            203,769
  Directed payment-flow edges     234,355
  Input features                      165
  Time steps                           49

The 165 features comprise 93 local features and 72 aggregated features.
Labels are encoded as:

-   `1` --- illicit
-   `0` --- licit
-   `-1` --- unknown

Unknown labels are not supervised targets and are not included in
labelled evaluation metrics. They remain available as graph context
where the evaluation protocol permits.

## Temporal evaluation protocol

  ----------------------------------------------------------------------------------
  Partition    Time steps   Transactions   Labelled    Illicit      Licit    Unknown
  ------------ ---------- -------------- ---------- ---------- ---------- ----------
  Train             1--29        120,804     26,381      2,871     23,510     94,423

  Validation       30--34         15,461      3,513        591      2,922     11,948

  Test             35--49         67,504     16,670      1,083     15,587     50,834
  ----------------------------------------------------------------------------------

The feature scaler is fitted on training rows only. Validation is used
for model and threshold selection. The test partition is evaluated after
selection and calibration; however, it has already been inspected during
development, so reported test results are exploratory.

**Important protocol limitation:** This is period-level transductive
evaluation, not per-transaction online prediction. A period's graph view
can include transactions from that same period, including transactions
later in the period. Train-only scaling does not remove any global or
future information that may already be encoded in the original dataset's
aggregated features.

## Models and current results

### Threshold-free test performance

  Model                  Average Precision (AP)           ROC-AUC
  -------------------- ------------------------ -----------------
  XGBoost                                0.5209            0.8849
  GCN                           0.4823 ± 0.0476   0.8628 ± 0.0144
  Directed GraphSAGE            0.3751 ± 0.0413   0.8740 ± 0.0048

GCN and GraphSAGE values are mean ± standard deviation across five seeds
(42--46). XGBoost is a single selected baseline, so its result does not
have a comparable multi-seed variability estimate.

### Fixed-capacity evaluation: top 1%

The alerting evaluation ranks all 67,504 transactions in the test period
and selects exactly 676 (1%). Selection is based on model scores and
transaction identifiers, not labels. Labelled metrics are then
calculated only for selected transactions with known labels.

  -----------------------------------------------------------------------
  Model            Precision         Recall             F1       Accuracy
  ----------- -------------- -------------- -------------- --------------
  XGBoost             0.0000         0.0000         0.0000         0.9325

  GCN               0.7882 ±       0.1256 ±       0.2166 ±       0.9410 ±
                      0.0606         0.0136         0.0226         0.0014

  Directed          0.4336 ±       0.0325 ±       0.0603 ±       0.9343 ±
  GraphSAGE           0.0931         0.0067         0.0121         0.0011
  -----------------------------------------------------------------------

Under this fixed 1% alert budget, GCN has the strongest reported
precision, recall, and F1 among these evaluated models. This result is
specific to the current dataset, split, graph construction, and
evaluation protocol; it should not be interpreted as evidence of
production performance.

The XGBoost row reports the selected baseline score output
(`first_tree_score`). A separate all-51-tree score export was examined
diagnostically and is not the selected baseline reported here.

### Threshold selection

The F1 decision threshold is selected using labelled validation targets
only and then frozen for test evaluation. The top-1% alert selector is a
separate capacity-based evaluation: it ranks all test-period
transactions by score, selects exactly `ceil(0.01 × N)` alerts, and
resolves score ties deterministically using ascending string
representation of transaction ID.

## Repository layout

``` text
elliptic-fraud-gnn/
├── app/
│   └── main.py
├── artifacts/
│   ├── gcn/
│   ├── gnn/
│   ├── xgboost_baseline_metadata.json
│   ├── xgboost_baseline_metrics.json
│   └── xgboost_transaction_scores.csv
├── docs/
│   └── decisions.md
├── src/
│   └── elliptic_fraud/
│       ├── evaluation/
│       ├── models/
│       └── training/
├── tests/
├── params.yaml
└── pyproject.toml
```

The exact contents of `artifacts/` depend on which training and
evaluation outputs have been generated locally.

## Run the project

Commands below assume the repository root and an activated Python
virtual environment.

Install the project dependencies using the repository's configured
environment workflow. Then run the automated test suite:

``` bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python -m pytest -v
```

**Current test status:** 103 passed, 0 failed. The run reported two
PyTorch deprecation warnings originating from `torch.jit.script`; they
did not cause test failures.

Launch the Streamlit dashboard:

``` bash
streamlit run app/main.py
```

The dashboard presents the project overview, model performance, and
methodology/limitations. It is a research interface, not a live
fraud-monitoring service.

## Reproducibility and interpretation

-   GCN and directed GraphSAGE final runs use seeds 42--46.
-   Model and threshold selection use validation data, not test metrics.
-   Unknown labels are excluded from supervised loss targets and
    labelled metric calculations.
-   The top-1% capacity is determined across all transactions, including
    those with unknown labels.
-   Reported GCN and GraphSAGE summary metrics are mean ± standard
    deviation across five seeds.
-   The XGBoost selected baseline is a single run and should not be
    treated as a five-seed comparison.
-   The test set has been inspected during development. Do not describe
    it as an untouched holdout.

## Limitations and responsible use

1.  **Exploratory test results:** Repeated development inspection means
    the test set is not an independent final holdout.
2.  **Transductive graph setting:** Evaluation uses period-level graph
    context and is not equivalent to strict real-time scoring at
    transaction arrival.
3.  **Source-feature leakage risk:** The original aggregated feature
    construction may contain global or future-derived information.
    Train-only scaling does not resolve this risk.
4.  **Label coverage:** Most transactions have unknown labels. Labelled
    metrics describe only the known-label subset.
5.  **Model comparison:** XGBoost is a single selected baseline, while
    GCN and GraphSAGE results aggregate five seeds.
6.  **No production claim:** The dashboard and evaluation workflow are
    research tooling; the project has not established live operational
    performance.
7.  **Dataset licensing:** Verify the original dataset and source-paper
    licence terms before redistributing data or derived files.

## Future work

-   Validate feature provenance and investigate source-level temporal
    leakage.
-   Add a genuinely untouched, prospectively defined evaluation period.
-   Extend robustness analysis across temporal windows and class
    prevalence.
-   Add explainability and drift-monitoring analyses with documented
    assumptions.
-   Strengthen deployment, API, and monitoring work before making any
    production-readiness claim.

## Disclaimer

This project is for academic and research purposes. Its predictions
should not be used as the sole basis for financial, legal, or
enforcement decisions.
