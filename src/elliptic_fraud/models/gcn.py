"""Undirected Graph Convolutional Network baseline for Elliptic Sentinel."""

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected

from elliptic_fraud.models.gnn import (
    compute_log_degree_features,
    validate_edge_index,
)


class EllipticGCN(nn.Module):
    """Two-layer GCN for binary transaction classification.

    The input directed graph is converted to an undirected graph by
    adding reverse edges. Original directed in/out-degree features
    are optionally retained as node features.
    """

    def __init__(
        self,
        in_channels: int = 165,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_degree_features: bool = True,
    ):
        super().__init__()

        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")

        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive.")

        if num_layers != 2:
            raise ValueError("This project currently requires exactly 2 layers.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_degree_features = use_degree_features

        model_input_channels = in_channels + (
            2 if use_degree_features else 0
        )

        self.conv1 = GCNConv(
            model_input_channels,
            hidden_channels,
            add_self_loops=True,
            normalize=True,
        )

        self.conv2 = GCNConv(
            hidden_channels,
            hidden_channels,
            add_self_loops=True,
            normalize=True,
        )

        self.classifier = nn.Linear(hidden_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim != 2:
            raise ValueError(
                "x must have shape [num_nodes, num_features]."
            )

        if x.size(1) != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input features, "
                f"received {x.size(1)}."
            )

        if edge_index.device != x.device:
            raise ValueError(
                "x and edge_index must be on the same device."
            )

        validate_edge_index(
            edge_index=edge_index,
            num_nodes=x.size(0),
        )

        if self.use_degree_features:
            degree_features = compute_log_degree_features(
                edge_index=edge_index,
                num_nodes=x.size(0),
                dtype=x.dtype,
            )

            x = torch.cat([x, degree_features], dim=1)

        # Standard GCN baseline: symmetrize the directed graph.
        undirected_edge_index = to_undirected(
            edge_index,
            num_nodes=x.size(0),
        )

        x = self.conv1(x, undirected_edge_index)
        x = F.relu(x)
        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        x = self.conv2(x, undirected_edge_index)
        x = F.relu(x)
        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        logits = self.classifier(x).squeeze(-1)

        return logits
