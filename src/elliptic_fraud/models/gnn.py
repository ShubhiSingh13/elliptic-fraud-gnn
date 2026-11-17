
"""Directed GraphSAGE model for Elliptic Sentinel."""

import torch
from torch import nn
import torch.nn.functional as F


def validate_edge_index(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> None:
    """Validate edge_index independently of degree-feature usage."""

    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")

    if edge_index.dtype != torch.long:
        raise TypeError("edge_index must use torch.long indices.")

    if num_nodes < 0:
        raise ValueError("num_nodes cannot be negative.")

    if edge_index.numel() > 0:
        if edge_index.min().item() < 0:
            raise ValueError("edge_index contains a negative node index.")

        if edge_index.max().item() >= num_nodes:
            raise ValueError(
                "edge_index contains an out-of-range node index."
            )


def compute_log_degree_features(
    edge_index: torch.Tensor,
    num_nodes: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return [log1p(in_degree), log1p(out_degree)] for each node.

    Edges use the source-to-target convention:
        edge_index[0] = source
        edge_index[1] = target

    Degrees are calculated only from the supplied edge_index.
    """

    validate_edge_index(edge_index, num_nodes)

    source = edge_index[0]
    target = edge_index[1]

    in_degree = torch.zeros(
        num_nodes,
        device=edge_index.device,
        dtype=dtype,
    )
    out_degree = torch.zeros(
        num_nodes,
        device=edge_index.device,
        dtype=dtype,
    )

    if source.numel() > 0:
        in_degree.index_add_(
            0,
            target,
            torch.ones_like(target, dtype=dtype),
        )
        out_degree.index_add_(
            0,
            source,
            torch.ones_like(source, dtype=dtype),
        )

    return torch.stack(
        [
            torch.log1p(in_degree),
            torch.log1p(out_degree),
        ],
        dim=1,
    )


def mean_aggregate(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    direction: str,
) -> torch.Tensor:
    """Mean-aggregate neighbor features in one edge direction.

    direction='incoming': each target receives from its sources.
    direction='outgoing': each source receives from its targets.

    Nodes with no neighbors receive an all-zero aggregate.
    """

    if direction not in {"incoming", "outgoing"}:
        raise ValueError(
            "direction must be 'incoming' or 'outgoing'."
        )

    validate_edge_index(edge_index, x.size(0))

    num_nodes = x.size(0)

    source = edge_index[0]
    target = edge_index[1]

    aggregated = torch.zeros_like(x)
    counts = x.new_zeros((num_nodes, 1))

    if direction == "incoming":
        receiver, sender = target, source
    else:
        receiver, sender = source, target

    if receiver.numel() > 0:
        aggregated.index_add_(
            0,
            receiver,
            x[sender],
        )

        counts.index_add_(
            0,
            receiver,
            x.new_ones((receiver.numel(), 1)),
        )

    return aggregated / counts.clamp_min(1.0)


class DirectedSAGELayer(nn.Module):
    """GraphSAGE layer with separate self, incoming and outgoing transforms."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.lin_self = nn.Linear(in_channels, out_channels)
        self.lin_in = nn.Linear(in_channels, out_channels)
        self.lin_out = nn.Linear(in_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        self_features = self.lin_self(x)

        incoming = mean_aggregate(
            x,
            edge_index,
            direction="incoming",
        )
        incoming_features = self.lin_in(incoming)

        outgoing = mean_aggregate(
            x,
            edge_index,
            direction="outgoing",
        )
        outgoing_features = self.lin_out(outgoing)

        combined = (
            self_features
            + incoming_features
            + outgoing_features
        )

        return F.relu(combined)


class DirectedGraphSAGE(nn.Module):
    """Two-layer directed GraphSAGE for binary node classification."""

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
            raise ValueError(
                "This project currently requires exactly 2 layers."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in the range [0, 1)."
            )

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_degree_features = use_degree_features

        model_input_channels = in_channels + (
            2 if use_degree_features else 0
        )

        self.layers = nn.ModuleList(
            [
                DirectedSAGELayer(
                    model_input_channels,
                    hidden_channels,
                ),
                DirectedSAGELayer(
                    hidden_channels,
                    hidden_channels,
                ),
            ]
        )

        # One raw logit per node. Apply sigmoid only for probabilities.
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

        # Always validate edges, including when degree features are disabled.
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

            x = torch.cat(
                [x, degree_features],
                dim=1,
            )

        for layer in self.layers:
            x = layer(x, edge_index)

            x = F.dropout(
                x,
                p=self.dropout,
                training=self.training,
            )

        logits = self.classifier(x).squeeze(-1)

        return logits
