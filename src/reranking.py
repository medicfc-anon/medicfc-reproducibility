import re


def normalize_text(text):
    if text is None:
        return ""

    text = str(text).lower().strip()
    return re.sub(r"\s+", " ", text)


def term_in_sentence(term, sentence):
    term = normalize_text(term)
    sentence = normalize_text(sentence)

    if not term or not sentence:
        return False

    pattern = r"\b" + re.escape(term) + r"\b"

    return re.search(
        pattern,
        sentence,
        flags=re.IGNORECASE
    ) is not None


def calculate_concept_coverage(sentence, concept_mapping):
    if not concept_mapping:
        return 0.0, 0, 0

    total_concepts = len(concept_mapping)
    covered_concepts = 0

    for original_concept, expansions in concept_mapping.items():

        if expansions is None:
            expansions = []

        elif isinstance(expansions, str):
            expansions = [expansions]

        elif not isinstance(
            expansions,
            (list, tuple, set)
        ):
            expansions = [str(expansions)]

        candidates = [original_concept] + list(expansions)

        unique_candidates = []
        seen = set()

        for term in candidates:
            term = normalize_text(term)

            if not term:
                continue

            if term not in seen:
                seen.add(term)
                unique_candidates.append(term)

        concept_matched = any(
            term_in_sentence(term, sentence)
            for term in unique_candidates
        )

        if concept_matched:
            covered_concepts += 1

    coverage = (
        covered_concepts / total_concepts
        if total_concepts > 0
        else 0.0
    )

    return (
        coverage,
        covered_concepts,
        total_concepts
    )


def calculate_hybrid_score(
    similarity_score,
    concept_coverage,
    alpha=0.4,
    beta=0.6
):
    return (
        alpha * similarity_score
        + beta * concept_coverage
    )


def rerank_evidence(
    claim,
    evidence_texts,
    similarity_scores,
    concept_mapping
):
    results = []

    for evidence_text, similarity_score in zip(
        evidence_texts,
        similarity_scores
    ):
        (
            concept_coverage,
            covered_concepts,
            total_concepts
        ) = calculate_concept_coverage(
            evidence_text,
            concept_mapping
        )

        hybrid_score = calculate_hybrid_score(
            similarity_score,
            concept_coverage
        )

        results.append({
            "claim": claim,
            "evidence_text": evidence_text,
            "similarity_score": similarity_score,
            "concept_coverage": concept_coverage,
            "covered_concepts": covered_concepts,
            "total_concepts": total_concepts,
            "hybrid_score": hybrid_score
        })

    results.sort(
        key=lambda x: x["hybrid_score"],
        reverse=True
    )

    for rank, result in enumerate(
        results,
        start=1
    ):
        result["rank"] = rank

    return results