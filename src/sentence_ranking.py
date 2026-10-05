from sentence_transformers import util


def get_top_similar_sentences(
    claim,
    sentences,
    urls,
    sbert_model,
    threshold=0.5,
    batch_size=100
):
    if not sentences:
        return [], [], []

    unique_pairs = list(
        dict.fromkeys(
            zip(sentences, urls)
        )
    )

    unique_sentences = [
        sentence
        for sentence, _ in unique_pairs
    ]

    unique_urls = [
        url
        for _, url in unique_pairs
    ]

    if not unique_sentences:
        return [], [], []

    claim_embedding = sbert_model.encode(
        claim,
        convert_to_tensor=True
    )

    all_similarities = []

    for i in range(
        0,
        len(unique_sentences),
        batch_size
    ):
        batch = unique_sentences[
            i:i + batch_size
        ]

        sentence_embeddings = sbert_model.encode(
            batch,
            convert_to_tensor=True
        )

        similarities = util.cos_sim(
            claim_embedding,
            sentence_embeddings
        )[0].cpu().tolist()

        all_similarities.extend(similarities)

    sentence_scores = list(
        zip(
            unique_sentences,
            unique_urls,
            all_similarities
        )
    )

    filtered = [
        (sentence, url, score)
        for sentence, url, score in sentence_scores
        if score >= threshold
    ]

    filtered.sort(
        key=lambda x: x[2],
        reverse=True
    )

    if not filtered:
        return [], [], []

    all_sentences = [
        sentence
        for sentence, _, _ in filtered
    ]

    all_urls = [
        url
        for _, url, _ in filtered
    ]

    all_scores = [
        score
        for _, _, score in filtered
    ]

    return (
        all_sentences,
        all_scores,
        all_urls
    )