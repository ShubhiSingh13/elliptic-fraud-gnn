
import pytest
import torch

from elliptic_fraud.models.gnn import (
    DirectedGraphSAGE,
    DirectedSAGELayer,
    compute_log_degree_features,
    mean_aggregate,
)


def make_toy_graph():
    # Edges: 0 -> 1, 0 -> 2, 3 -> 2
    edge_index = torch.tensor(
        [[0, 0, 3], [1, 2, 2]],
        dtype=torch.long,
    )

    x = torch.tensor(
        [[1.0], [10.0], [100.0], [1000.0]]
    )

    return x, edge_index


def test_incoming_mean_aggregation():
    x, edge_index = make_toy_graph()

    result = mean_aggregate(x, edge_index, "incoming")

    expected = torch.tensor(
        [[0.0], [1.0], [500.5], [0.0]]
    )

    torch.testing.assert_close(result, expected)


def test_outgoing_mean_aggregation():
    x, edge_index = make_toy_graph()

    result = mean_aggregate(x, edge_index, "outgoing")

    expected = torch.tensor(
        [[55.0], [0.0], [0.0], [100.0]]
    )

    torch.testing.assert_close(result, expected)


def test_zero_neighbor_aggregation_returns_zero():
    x = torch.tensor([[2.0], [5.0], [8.0]])
    edge_index = torch.empty((2, 0), dtype=torch.long)

    incoming = mean_aggregate(x, edge_index, "incoming")
    outgoing = mean_aggregate(x, edge_index, "outgoing")

    torch.testing.assert_close(incoming, torch.zeros_like(x))
    torch.testing.assert_close(outgoing, torch.zeros_like(x))


def test_degree_features_are_log1p_of_directed_degrees():
    _, edge_index = make_toy_graph()

    result = compute_log_degree_features(
        edge_index=edge_index,
        num_nodes=4,
        dtype=torch.float32,
    )

    expected = torch.tensor(
        [
            [0.0, torch.log1p(torch.tensor(2.0)).item()],
            [torch.log1p(torch.tensor(1.0)).item(), 0.0],
            [torch.log1p(torch.tensor(2.0)).item(), 0.0],
            [0.0, torch.log1p(torch.tensor(1.0)).item()],
        ]
    )

    torch.testing.assert_close(result, expected)


def test_directed_layer_uses_separate_transformations():
    x, edge_index = make_toy_graph()

    layer = DirectedSAGELayer(1, 1)

    with torch.no_grad():
        for linear in (
            layer.lin_self,
            layer.lin_in,
            layer.lin_out,
        ):
            linear.bias.zero_()

        layer.lin_self.weight.fill_(1.0)
        layer.lin_in.weight.fill_(2.0)
        layer.lin_out.weight.fill_(3.0)

    result = layer(x, edge_index)

    expected = torch.tensor(
        [[166.0], [12.0], [1101.0], [1300.0]]
    )

    torch.testing.assert_close(result, expected)


def test_model_returns_one_logit_per_node():
    x, edge_index = make_toy_graph()

    model = DirectedGraphSAGE(
        in_channels=1,
        hidden_channels=8,
        dropout=0.0,
        use_degree_features=True,
    )

    model.eval()
    logits = model(x, edge_index)

    assert logits.shape == (4,)
    assert torch.isfinite(logits).all()


def test_degree_feature_ablation_changes_input_dimension():
    model_with_degrees = DirectedGraphSAGE(
        in_channels=165,
        hidden_channels=64,
        use_degree_features=True,
    )

    model_without_degrees = DirectedGraphSAGE(
        in_channels=165,
        hidden_channels=64,
        use_degree_features=False,
    )

    assert model_with_degrees.layers[0].lin_self.in_features == 167
    assert model_without_degrees.layers[0].lin_self.in_features == 165


def test_gradients_flow_through_both_layers_and_classifier():
    x, edge_index = make_toy_graph()

    model = DirectedGraphSAGE(
        in_channels=1,
        hidden_channels=8,
        dropout=0.0,
        use_degree_features=True,
    )

    model.train()

    logits = model(x, edge_index)
    loss = logits.square().mean()
    loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"Missing gradient: {name}"
        assert torch.isfinite(parameter.grad).all(), (
            f"Non-finite gradient: {name}"
        )


def test_invalid_aggregation_direction_is_rejected():
    x, edge_index = make_toy_graph()

    with pytest.raises(ValueError):
        mean_aggregate(x, edge_index, "sideways")
