
import torch
from torch import nn
from torch_geometric.nn import GCNConv, MessagePassing


class MeanAggregator(MessagePassing):
    """Aggregate neighbour features using mean message passing."""

    def __init__(self, flow="source_to_target"):
        super().__init__(aggr="mean", flow=flow)

    def forward(self, x, edge_index):
        return self.propagate(
            edge_index,
            x=x,
            size=(x.size(0), x.size(0)),
        )


class DirectedSageLayer(nn.Module):
    """Toy layer with separate incoming and outgoing aggregations."""

    def __init__(self):
        super().__init__()
        self.lin_self = nn.Linear(1, 1, bias=False)
        self.lin_in = nn.Linear(1, 1, bias=False)
        self.lin_out = nn.Linear(1, 1, bias=False)

        self.incoming_aggregator = MeanAggregator(
            flow="source_to_target"
        )
        self.outgoing_aggregator = MeanAggregator(
            flow="target_to_source"
        )

    def forward(self, x, edge_index):
        self_features = self.lin_self(x)

        incoming = self.incoming_aggregator(x, edge_index)
        incoming = self.lin_in(incoming)

        outgoing = self.outgoing_aggregator(x, edge_index)
        outgoing = self.lin_out(outgoing)

        return self_features + incoming + outgoing


def make_toy_graph():
    # Edges: 0 -> 1, 0 -> 2, 3 -> 2
    edge_index = torch.tensor(
        [
            [0, 0, 3],
            [1, 2, 2],
        ],
        dtype=torch.long,
    )

    # One feature per node.
    x = torch.tensor(
        [[1.0], [10.0], [100.0], [1000.0]]
    )

    return x, edge_index


def test_default_message_passing_is_source_to_target():
    x, edge_index = make_toy_graph()

    aggregator = MeanAggregator(
        flow="source_to_target"
    )

    result = aggregator(x, edge_index)

    expected = torch.tensor(
        [[0.0], [1.0], [500.5], [0.0]]
    )

    torch.testing.assert_close(result, expected)


def test_reverse_message_passing_is_target_to_source():
    x, edge_index = make_toy_graph()

    aggregator = MeanAggregator(
        flow="target_to_source"
    )

    result = aggregator(x, edge_index)

    expected = torch.tensor(
        [[55.0], [0.0], [0.0], [100.0]]
    )

    torch.testing.assert_close(result, expected)


def test_gcn_conv_uses_source_to_target_direction():
    x = torch.tensor([[2.0], [7.0]])

    # A single directed edge: 0 -> 1
    edge_index = torch.tensor(
        [[0], [1]],
        dtype=torch.long,
    )

    conv = GCNConv(
        in_channels=1,
        out_channels=1,
        add_self_loops=False,
        normalize=False,
        bias=False,
    )

    with torch.no_grad():
        conv.lin.weight.fill_(1.0)

    result = conv(x, edge_index)

    expected = torch.tensor(
        [[0.0], [2.0]]
    )

    torch.testing.assert_close(result, expected)


def test_directed_sage_separates_incoming_and_outgoing():
    x, edge_index = make_toy_graph()

    layer = DirectedSageLayer()

    with torch.no_grad():
        layer.lin_self.weight.fill_(1.0)
        layer.lin_in.weight.fill_(2.0)
        layer.lin_out.weight.fill_(3.0)

    result = layer(x, edge_index)

    expected = torch.tensor(
        [[166.0], [12.0], [1101.0], [1300.0]]
    )

    torch.testing.assert_close(result, expected)


def test_snapshot_outputs_match_when_no_cross_step_edges_exist():
    # Two independent snapshots:
    # Step 1: 0 -> 1
    # Step 2: 2 -> 3
    x = torch.tensor(
        [[1.0], [10.0], [100.0], [1000.0]]
    )

    edge_index = torch.tensor(
        [
            [0, 2],
            [1, 3],
        ],
        dtype=torch.long,
    )

    time_step = torch.tensor([1, 1, 2, 2])

    aggregator = MeanAggregator()

    cumulative_output = aggregator(x, edge_index)

    for step in (1, 2):
        node_mask = time_step == step

        edge_mask = (
            (time_step[edge_index[0]] == step)
            & (time_step[edge_index[1]] == step)
        )

        snapshot_edge_index = edge_index[:, edge_mask]

        snapshot_output = aggregator(
            x,
            snapshot_edge_index,
        )

        torch.testing.assert_close(
            cumulative_output[node_mask],
            snapshot_output[node_mask],
        )


def test_cross_step_edge_changes_snapshot_aggregation():
    # Add an edge from step 1 to step 2.
    x = torch.tensor(
        [[1.0], [10.0], [100.0], [1000.0]]
    )

    edge_index = torch.tensor(
        [
            [0, 2, 1],
            [1, 3, 2],
        ],
        dtype=torch.long,
    )

    time_step = torch.tensor([1, 1, 2, 2])

    aggregator = MeanAggregator()

    cumulative_output = aggregator(x, edge_index)

    step_two_mask = time_step == 2

    step_two_edge_mask = (
        (time_step[edge_index[0]] == 2)
        & (time_step[edge_index[1]] == 2)
    )

    step_two_edges = edge_index[:, step_two_edge_mask]

    isolated_step_two_output = aggregator(
        x,
        step_two_edges,
    )

    # Node 2 receives an additional message from node 1
    # in the cumulative graph, so its result must differ.
    assert not torch.allclose(
        cumulative_output[step_two_mask],
        isolated_step_two_output[step_two_mask],
    )
