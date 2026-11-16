
# Elliptic Bitcoin Transaction Dataset — Data Card

## 1. Dataset Overview

**Dataset name:** Elliptic Bitcoin Transaction Dataset

**Domain:** Financial Technology / Cryptocurrency Fraud Detection

**Task:** Binary classification of Bitcoin transactions as licit or illicit, with an additional unlabelled category.

**Source:** Elliptic Data Set, published on Kaggle.

## 2. Dataset Files

The original dataset contains three CSV files:

- `elliptic_txs_features.csv` — Transaction IDs, time steps and 165 features.
- `elliptic_txs_edgelist.csv` — Directed relationships between transactions.
- `elliptic_txs_classes.csv` — Transaction class labels.

## 3. Dataset Statistics

| Property | Value |
|---|---:|
| Transactions | 203,769 |
| Directed edges | 234,355 |
| Features per transaction | 165 |
| Local features | 93 |
| Aggregated features | 72 |
| Time steps | 49 |
| Unlabelled transactions | 157,205 |
| Licit transactions | 42,019 |
| Illicit transactions | 4,545 |

## 4. Label Mapping

Original dataset labels are mapped as follows:

| Original label | Project label | Meaning |
|---|---:|---|
| 1 | 1 | Illicit |
| 2 | 0 | Licit |
| unknown | -1 | Unlabelled |

Unlabelled transactions may participate in graph context, but are excluded from supervised loss and evaluation.

## 5. Feature Description

The dataset provides 165 anonymised numerical features:

- First 93 features: Local transaction information.
- Last 72 features: Aggregated information derived from neighbouring transactions.

The original feature meanings are not fully available in the anonymised dataset.

## 6. Temporal Structure

The dataset contains 49 time steps.

The project uses chronological splitting:

- Training: time steps 1–29.
- Validation: time steps 30–34.
- Testing: time steps 35–49.

No transaction labels are used as model input.

## 7. Data Quality Checks

The following checks have been completed:

- No duplicate transaction IDs.
- No missing numerical feature values.
- No infinite feature values.
- No duplicate directed edges.
- Every edge references a valid transaction.
- No edges connect transactions from different time steps.
- All 49 time steps are represented.

## 8. Limitations

- Transaction features are anonymised.
- Raw Bitcoin transactions cannot be directly scored without matching the dataset's feature schema.
- The dataset contains a substantial unlabelled portion.
- Aggregated features are supplied by the original dataset; their underlying computation cannot be independently reconstructed from the provided files.
- Historical dataset evaluation does not establish real-world live fraud detection performance.

## 9. Intended Use

This dataset is used for academic research and development of a temporal graph-based cryptocurrency transaction fraud detection system.

The project is intended for experimentation and educational purposes, not as a production financial compliance or law-enforcement system.
