def hit_rate_at_k(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    expected = set(expected_sources)
    return 1.0 if any(src in expected for src in retrieved_sources) else 0.0


def precision_at_k(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    if not retrieved_sources:
        return 0.0
    expected = set(expected_sources)
    hits = sum(1 for src in retrieved_sources if src in expected)
    return hits / len(retrieved_sources)


def reciprocal_rank(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    expected = set(expected_sources)
    for idx, src in enumerate(retrieved_sources, start=1):
        if src in expected:
            return 1.0 / idx
    return 0.0


def citation_correctness(citations: list[str], expected_sources: list[str]) -> float:
    if not citations:
        return 0.0
    expected = set(expected_sources)
    valid = sum(1 for c in citations if c in expected)
    return valid / len(citations)


def unanswerable_accuracy(answerable: bool, predicted_unanswerable: bool) -> float:
    expected_unanswerable = not answerable
    return 1.0 if expected_unanswerable == predicted_unanswerable else 0.0
