
import pytest
import torch

from elliptic_fraud.models.gnn import DirectedGraphSAGE


def test_invalid_negative_edge_rejected_without_degree_features():
    model = DirectedGraphSAGE(use_degree_features=False)
    x = torch.randn(3, 165)
    edge_index = torch.tensor([[0, -1], [1, 2]])

    with pytest.raises(ValueError, match="negative node index"):
        model(x, edge_index)


def test_out_of_range_edge_rejected_without_degree_features():
    model = DirectedGraphSAGE(use_degree_features=False)
    x = torch.randn(3, 165)
    edge_index = torch.tensor([[0, 3], [1, 2]])

    with pytest.raises(ValueError, match="out-of-range"):
        model(x, edge_index)


def test_invalid_edge_shape_rejected():
    model = DirectedGraphSAGE(use_degree_features=False)
    x = torch.randn(3, 165)
    edge_index = torch.tensor([[0, 1, 2]])

    with pytest.raises(ValueError, match="shape"):
        model(x, edge_index)


def test_valid_edges_work_without_degree_features():
    model = DirectedGraphSAGE(use_degree_features=False)
    model.eval()

    x = torch.randn(3, 165)
    edge_index = torch.tensor([[0, 1], [1, 2]])

    with torch.no_grad():
        logits = model(x, edge_index)

    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()
