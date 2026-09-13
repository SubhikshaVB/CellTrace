"""
Review-2 math/correctness tests (no data files needed).

These verify that the NEW code computes correctly (shapes, invariants,
loss behaviour) using small random tensors. They are software tests —
none of their numbers appear in any project report.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from modules.module3_gat import (
    ContrastiveGAT,
    build_frame_knn_graph,
    build_gt_pairs,
    contrastive_margin_loss,
)
from modules.module4_tracking import GroupWeightMLP


def test_knn_same_frame_only():
    pos = np.random.rand(12, 3).astype(np.float32) * 50
    times = np.array([0]*4 + [1]*4 + [2]*4)
    ei, stats = build_frame_knn_graph(pos, times, k=2)
    assert ei.shape[0] == 2
    assert (times[ei[0]] == times[ei[1]]).all()
    # 12 self-loops + 12 nodes * 2 neighbours
    assert stats["n_frame_edges"] == 24, stats
    print("PASS knn_same_frame_only", stats)


def test_gt_pairs_labels():
    node_ids = np.arange(6)
    times = np.array([0, 0, 1, 1, 2, 4])
    edges = [(0, 2), (1, 3), (2, 4), (4, 5)]  # last has dt=2 -> skipped
    pairs, labels, info = build_gt_pairs(node_ids, times, edges, neg_per_pos=2)
    assert info["n_positive"] == 3, info
    assert info["n_gt_skipped_non_consecutive"] == 1, info
    assert labels.sum() == 3 and len(labels) == 3 * (1 + 2)
    print("PASS gt_pairs_labels", info)


def test_gat_forward_normalized():
    torch.manual_seed(0)
    n, d = 10, 16
    x = torch.randn(n, d)
    ei = torch.tensor([list(range(n)), list(range(n))])  # self-loops
    model = ContrastiveGAT(d, hidden_dim=8, heads=2, out_dim=6)
    out = model(x, ei)
    assert out.shape == (n, 6)
    assert torch.allclose(out.norm(dim=1), torch.ones(n), atol=1e-5)
    print("PASS gat_forward_normalized", tuple(out.shape))


def test_contrastive_loss_direction():
    torch.manual_seed(0)
    emb = torch.randn(6, 8)
    emb = emb / emb.norm(dim=1, keepdim=True)
    pairs = torch.tensor([[0, 1], [2, 3]])
    labels = torch.tensor([1.0, 0.0])
    loss = contrastive_margin_loss(emb, pairs, labels, margin=1.0)
    assert loss.item() > 0
    # Identical positives + far negatives -> ~0 loss
    emb2 = torch.tensor([[1., 0.], [1., 0.], [1., 0.], [-1., 0.]])
    loss2 = contrastive_margin_loss(emb2, pairs, labels, margin=1.0)
    assert loss2.item() < 0.05, loss2.item()
    print("PASS contrastive_loss_direction", round(loss.item(), 4), round(loss2.item(), 4))


def test_group_mlp_softmax_sums_to_one():
    torch.manual_seed(0)
    mlp = GroupWeightMLP(hidden=8)
    w, b = mlp(torch.randn(10))
    assert w.shape == (5,)
    assert abs(float(w.detach().sum()) - 1.0) < 1e-5
    assert (w > 0).all()
    print("PASS group_mlp_softmax", [round(float(v), 3) for v in w.detach()])


def test_hungarian_available():
    from scipy.optimize import linear_sum_assignment
    C = np.array([[0.1, 0.9], [0.8, 0.2]])
    r, c = linear_sum_assignment(C)
    assert (r == [0, 1]).all() and (c == [0, 1]).all()
    print("PASS hungarian")


if __name__ == "__main__":
    test_knn_same_frame_only()
    test_gt_pairs_labels()
    test_gat_forward_normalized()
    test_contrastive_loss_direction()
    test_group_mlp_softmax_sums_to_one()
    test_hungarian_available()
    print("ALL REVIEW-2 MATH TESTS PASSED")
