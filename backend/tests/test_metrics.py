from app.evaluation.metrics import hit_rate_at_k, precision_at_k, reciprocal_rank


def test_retrieval_metrics():
    retrieved = ["a#p1", "b#p2", "c#p3"]
    expected = ["b#p2"]

    assert hit_rate_at_k(retrieved, expected) == 1.0
    assert precision_at_k(retrieved, expected) == 1 / 3
    assert reciprocal_rank(retrieved, expected) == 0.5
