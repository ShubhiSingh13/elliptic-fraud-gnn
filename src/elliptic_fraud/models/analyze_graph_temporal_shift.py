
"""
Elliptic Sentinel
Graph Temporal Shift Diagnostic

Investigates graph connectivity across time steps 35-49,
with special attention to steps 42 and 43.

The graph is directed. In-degree and out-degree are
calculated separately.

This script does not train or modify any model.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[3]

GRAPH_PATH = (
    ROOT / "data/processed/elliptic/transaction_graph.pt"
)

OUTPUT_DIR = ROOT / "artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD GRAPH
# --------------------------------------------------

def load_graph():

    print("\nLoading transaction graph...")

    graph = torch.load(
        GRAPH_PATH,
        map_location="cpu",
        weights_only=True
    )

    edge_index = graph["edge_index"].numpy()
    time_steps = graph["time_step"].numpy()
    labels = graph["y"].numpy()
    tx_ids = graph["tx_id"].numpy()

    print("Graph loaded successfully.")
    print("Total nodes:", len(time_steps))
    print("Total directed edges:", edge_index.shape[1])

    return edge_index, time_steps, labels, tx_ids


# --------------------------------------------------
# 3. GRAPH STATISTICS BY TIME STEP
# --------------------------------------------------

def analyze_graph_statistics(
    edge_index,
    time_steps,
    labels
):

    print("\n" + "=" * 85)
    print("GRAPH STRUCTURE BY TIME STEP")
    print("=" * 85)

    source = edge_index[0]
    target = edge_index[1]

    records = []

    for step in range(35, 50):

        node_mask = time_steps == step
        node_indices = np.flatnonzero(node_mask)

        if len(node_indices) == 0:
            continue

        # Select edges belonging to this time step.
        edge_mask = (
            (time_steps[source] == step)
            & (time_steps[target] == step)
        )

        step_source = source[edge_mask]
        step_target = target[edge_mask]

        # Calculate degrees using global node indices.
        in_degree = np.bincount(
            step_target,
            minlength=len(time_steps)
        )[node_indices]

        out_degree = np.bincount(
            step_source,
            minlength=len(time_steps)
        )[node_indices]

        total_degree = in_degree + out_degree

        isolated = total_degree == 0

        # Build a local undirected graph for component
        # analysis. This is only for component counting;
        # original edge direction is preserved in degree
        # statistics above.
        local_index = np.full(
            len(time_steps),
            -1,
            dtype=np.int64
        )

        local_index[node_indices] = np.arange(
            len(node_indices)
        )

        local_source = local_index[step_source]
        local_target = local_index[step_target]

        adjacency = csr_matrix(
            (
                np.ones(2 * len(step_source)),
                (
                    np.concatenate([
                        local_source,
                        local_target
                    ]),
                    np.concatenate([
                        local_target,
                        local_source
                    ])
                )
            ),
            shape=(len(node_indices), len(node_indices))
        )

        component_count, component_labels = (
            connected_components(
                adjacency,
                directed=False,
                return_labels=True
            )
        )

        component_sizes = np.bincount(component_labels)

        largest_component = (
            int(component_sizes.max())
            if len(component_sizes) else 0
        )

        labelled = labels[node_indices] >= 0
        illicit = labels[node_indices] == 1
        licit = labels[node_indices] == 0

        records.append({

            "time_step": step,

            "nodes": len(node_indices),

            "edges": len(step_source),

            "mean_in_degree": float(np.mean(in_degree)),

            "mean_out_degree": float(np.mean(out_degree)),

            "mean_total_degree": float(np.mean(total_degree)),

            "median_total_degree": float(np.median(total_degree)),

            "isolated_nodes": int(np.sum(isolated)),

            "isolated_percentage": float(
                np.mean(isolated) * 100
            ),

            "connected_components": int(component_count),

            "largest_component_size": largest_component,

            "largest_component_percentage": float(
                largest_component / len(node_indices) * 100
            ),

            "labelled_nodes": int(np.sum(labelled)),

            "licit_nodes": int(np.sum(licit)),

            "illicit_nodes": int(np.sum(illicit))
        })

        print(
            f"Step {step:2d} | "
            f"Nodes: {len(node_indices):5d} | "
            f"Edges: {len(step_source):5d} | "
            f"Mean degree: {np.mean(total_degree):7.3f} | "
            f"Isolated: {np.mean(isolated) * 100:6.2f}% | "
            f"Components: {component_count:5d} | "
            f"Largest component: {largest_component:5d}"
        )

    result = pd.DataFrame(records)

    result.to_csv(
        OUTPUT_DIR / "temporal_graph_statistics.csv",
        index=False
    )

    return result


# --------------------------------------------------
# 4. DEGREE DISTRIBUTION BY CLASS
# --------------------------------------------------

def analyze_degree_by_class(
    edge_index,
    time_steps,
    labels
):

    print("\n" + "=" * 85)
    print("DEGREE DISTRIBUTION BY CLASS")
    print("=" * 85)

    source = edge_index[0]
    target = edge_index[1]

    in_degree = np.bincount(
        target,
        minlength=len(time_steps)
    )

    out_degree = np.bincount(
        source,
        minlength=len(time_steps)
    )

    total_degree = in_degree + out_degree

    records = []

    for step in range(35, 50):

        for class_value, class_name in [
            (0, "licit"),
            (1, "illicit")
        ]:

            mask = (
                (time_steps == step)
                & (labels == class_value)
            )

            degree = total_degree[mask]

            if len(degree) == 0:
                continue

            records.append({

                "time_step": step,

                "class": class_name,

                "transaction_count": len(degree),

                "mean_in_degree": float(
                    np.mean(in_degree[mask])
                ),

                "mean_out_degree": float(
                    np.mean(out_degree[mask])
                ),

                "mean_total_degree": float(
                    np.mean(degree)
                ),

                "median_total_degree": float(
                    np.median(degree)
                ),

                "isolated_percentage": float(
                    np.mean(degree == 0) * 100
                ),

                "degree_90th_percentile": float(
                    np.percentile(degree, 90)
                )
            })

    result = pd.DataFrame(records)

    result.to_csv(
        OUTPUT_DIR / "temporal_degree_by_class.csv",
        index=False
    )

    print("\nSteps 41-44:")

    print(
        result[
            result["time_step"].between(41, 44)
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"
        )
    )

    return result


# --------------------------------------------------
# 5. COMPARE STEPS 42 AND 43
# --------------------------------------------------

def compare_transition(
    graph_stats,
    degree_stats
):

    print("\n" + "=" * 85)
    print("DIRECT COMPARISON: STEP 42 VS STEP 43")
    print("=" * 85)

    comparison = graph_stats[
        graph_stats["time_step"].isin([42, 43])
    ]

    print("\nOverall graph statistics:")

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"
        )
    )

    class_comparison = degree_stats[
        degree_stats["time_step"].isin([42, 43])
    ]

    print("\nClass-wise degree statistics:")

    print(
        class_comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"
        )
    )


# --------------------------------------------------
# 6. MAIN
# --------------------------------------------------

def main():

    (
        edge_index,
        time_steps,
        labels,
        tx_ids
    ) = load_graph()

    graph_stats = analyze_graph_statistics(
        edge_index,
        time_steps,
        labels
    )

    degree_stats = analyze_degree_by_class(
        edge_index,
        time_steps,
        labels
    )

    compare_transition(
        graph_stats,
        degree_stats
    )

    print("\n" + "=" * 85)
    print("GRAPH TEMPORAL DIAGNOSTIC COMPLETED")
    print("=" * 85)

    print("\nReports saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
