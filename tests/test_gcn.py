import pytest
import torch

from elliptic_fraud.models.gcn import EllipticGCN


def make_toy_graph():
    x = torch.randn(4, 5)

    edge_index = torch.tensor(
        [[0, 0, 3], [1, 2, 2]],
        dtype=torch.long,
    )

    return x, edge_index


def test_model_returns_one_logit_per_node():
    x, edge_index = make_toy_graph()

    model = EllipticGCN(
        in_channels=5,
        hidden_channels=8,
        dropout=0.0,
    )

    model.eval()
    logits = model(x, edge_index)

    assert logits.shape == (4,)
    assert torch.isfinite(logits).all()


def test_degree_feature_ablation_changes_input_dimension():
    model_with_degrees = EllipticGCN(
        in_channels=165,
        hidden_channels=64,
        use_degree_features=True,
    )

    model_without_degrees = EllipticGCN(
        in_channels=165,
        hidden_channels=64,
        use_degree_features=False,
    )

    assert model_with_degrees.conv1.in_channels == 167
    assert model_without_degrees.conv1.in_channels == 165


def test_gradients_flow_through_both_layers_and_classifier():
    x, edge_index = make_toy_graph()

    model = EllipticGCN(
        in_channels=5,
        hidden_channels=8,
        dropout=0.0,
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


def test_empty_graph_is_supported():
    x = torch.randn(4, 5)
    edge_index = torch.empty((2, 0), dtype=torch.long)

    model = EllipticGCN(
        in_channels=5,
        hidden_channels=8,
        dropout=0.0,
    )

    model.eval()
    logits = model(x, edge_index)

    assert logits.shape == (4,)
    assert torch.isfinite(logits).all()


def test_invalid_feature_dimension_is_rejected():
    model = EllipticGCN(in_channels=5)

    x = torch.randn(4, 6)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)

    with pytest.raises(ValueError, match="Expected 5 input features"):
        model(x, edge_index)


def test_invalid_edge_index_is_rejected():
    model = EllipticGCN(in_channels=5)

    x = torch.randn(4, 5)
    edge_index = torch.tensor([[0], [5]], dtype=torch.long)

    with pytest.raises(ValueError, match="out-of-range"):
        model(x, edge_index)


def test_model_rejects_unsupported_layer_count():
    with pytest.raises(ValueError, match="exactly 2 layers"):
        EllipticGCN(num_layers=3)


def test_model_rejects_invalid_dropout():
    with pytest.raises(ValueError, match="dropout"):
        EllipticGCN(dropout=1.0)
