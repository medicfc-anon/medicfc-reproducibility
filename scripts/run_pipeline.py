import argparse
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import nltk
import pandas as pd
import spacy
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

from src.preprocessing import preprocess_claim
from src.phrase_extraction import (
    create_phrase_matcher,
    extract_biomedical_phrases
)
from src.umls import (
    initialize_umls,
    search_umls_contextual,
    clear_umls_cache
)
from src.query_builder import build_search_query
from src.google_search import google_search
from src.web_extraction import (
    fetch_article_content,
    extract_sentences
)
from src.sentence_ranking import get_top_similar_sentences
from src.reranking import rerank_evidence


def initialize_models():
    nlp = spacy.load("en_core_web_sm")

    nlp_sent = spacy.load(
        "en_core_web_sm",
        disable=["parser", "ner"]
    )

    nlp_sent.add_pipe("sentencizer")

    sbert_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    matcher = create_phrase_matcher(nlp)

    stop_words = set(
        stopwords.words("english")
    )

    return (
        nlp,
        nlp_sent,
        sbert_model,
        matcher,
        stop_words
    )


def filter_umls_terms(ranked_terms):
    filtered = []
    seen = set()

    for term, score in ranked_terms:
        term = str(term).strip()

        if not term:
            continue

        if not all(
            char.isalpha()
            or char.isspace()
            or char == "-"
            for char in term
        ):
            continue

        normalized = term.lower()

        if normalized in seen:
            continue

        seen.add(normalized)

        filtered.append(
            (term, score)
        )

    return filtered


def process_claim(
    claim,
    nlp,
    nlp_sent,
    sbert_model,
    matcher,
    stop_words,
    umls_tgt,
    google_api_key,
    google_cse_id
):
    original_claim = str(claim).strip()

    if not original_claim:
        return []

    processed_claim = preprocess_claim(
        original_claim
    )

    phrases = extract_biomedical_phrases(
        processed_claim,
        nlp,
        matcher,
        stop_words
    )

    concept_mapping = {}
    query_terms = []

    for phrase in phrases:
        ranked_terms = search_umls_contextual(
            phrase,
            original_claim,
            umls_tgt,
            sbert_model
        )

        filtered_terms = filter_umls_terms(
            ranked_terms
        )

        concept_mapping[phrase] = [
            term
            for term, _ in filtered_terms
        ]

        query_terms.extend(
            term
            for term, _ in filtered_terms
        )

    unique_terms = []
    seen_terms = set()

    for term in query_terms:
        normalized = term.lower()

        if normalized not in seen_terms:
            seen_terms.add(normalized)
            unique_terms.append(term)

    search_query = build_search_query(
        original_claim,
        unique_terms,
        max_terms=15
    )

    search_results = google_search(
        search_query,
        google_api_key,
        google_cse_id,
        max_results=10
    )

    sentences = []
    sentence_urls = []

    for result in search_results:
        url = result.get("link")

        if not url:
            continue

        article_text = fetch_article_content(
            url
        )

        article_sentences = extract_sentences(
            article_text,
            nlp_sent
        )

        sentences.extend(
            article_sentences
        )

        sentence_urls.extend(
            [url] * len(article_sentences)
        )

    (
        semantic_sentences,
        semantic_scores,
        semantic_urls
    ) = get_top_similar_sentences(
        original_claim,
        sentences,
        sentence_urls,
        sbert_model,
        threshold=0.5
    )

    if not semantic_sentences:
        return []

    reranked = rerank_evidence(
        original_claim,
        semantic_sentences,
        semantic_scores,
        concept_mapping
    )

    final_results = []

    for result in reranked:
        final_results.append({
            "claim": result["claim"],
            "evidence_text": result["evidence_text"],
            "similarity_score": result["similarity_score"],
            "concept_coverage": result["concept_coverage"],
            "covered_concepts": result["covered_concepts"],
            "total_concepts": result["total_concepts"],
            "hybrid_score": result["hybrid_score"],
            "rank": result["rank"]
        })

    return final_results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
    )

    args = parser.parse_args()

    google_api_key = os.getenv(
        "GOOGLE_API_KEY"
    )

    google_cse_id = os.getenv(
        "GOOGLE_CSE_ID"
    )

    umls_api_key = os.getenv(
        "UMLS_API_KEY"
    )

    if not google_api_key:
        raise ValueError(
            "GOOGLE_API_KEY is not configured."
        )

    if not google_cse_id:
        raise ValueError(
            "GOOGLE_CSE_ID is not configured."
        )

    if not umls_api_key:
        raise ValueError(
            "UMLS_API_KEY is not configured."
        )

    nltk.download(
        "stopwords",
        quiet=True
    )

    (
        nlp,
        nlp_sent,
        sbert_model,
        matcher,
        stop_words
    ) = initialize_models()

    umls_tgt = initialize_umls(
        umls_api_key
    )

    df = pd.read_csv(
        args.input
    )

    if "en_claim" in df.columns:
        claim_column = "en_claim"
    elif "claim" in df.columns:
        claim_column = "claim"
    else:
        raise ValueError(
            "Input CSV must contain 'en_claim' or 'claim'."
        )

    all_results = []

    total_claims = len(df)

    for index, row in df.iterrows():
        claim = row[claim_column]

        print(
            f"Processing claim "
            f"{index + 1}/{total_claims}"
        )

        try:
            results = process_claim(
                claim,
                nlp,
                nlp_sent,
                sbert_model,
                matcher,
                stop_words,
                umls_tgt,
                google_api_key,
                google_cse_id
            )

            all_results.extend(
                results
            )

        except Exception as exc:
            print(
                f"Claim {index + 1} failed: {exc}"
            )

        clear_umls_cache()

    output_columns = [
        "claim",
        "evidence_text",
        "similarity_score",
        "concept_coverage",
        "covered_concepts",
        "total_concepts",
        "hybrid_score",
        "rank"
    ]

    output_df = pd.DataFrame(
        all_results,
        columns=output_columns
    )

    output_df.to_csv(
        args.output,
        index=False
    )

    print(
        f"Saved {len(output_df)} evidence rows "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()