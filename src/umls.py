import os
import time

import requests
from sentence_transformers import util


UMLS_SIMILARITY_THRESHOLD = 0.5
UMLS_TOP_K = 10

umls_raw_cache = {}
umls_filtered_cache = {}


def get_tgt(api_key):
    response = requests.post(
        "https://utslogin.nlm.nih.gov/cas/v1/api-key",
        data={"apikey": api_key}
    )

    if response.status_code == 201:
        return response.headers["location"]

    raise RuntimeError(
        f"UMLS authentication failed: {response.status_code}"
    )


def get_service_ticket(tgt_url):
    response = requests.post(
        tgt_url,
        data={"service": "http://umlsks.nlm.nih.gov"}
    )

    if response.status_code == 200:
        return response.text.strip()

    raise RuntimeError(
        f"UMLS service ticket request failed: {response.status_code}"
    )


def get_cui(term, tgt_url):
    service_ticket = get_service_ticket(tgt_url)

    response = requests.get(
        "https://uts-ws.nlm.nih.gov/rest/search/current",
        params={
            "string": term,
            "pageNumber": 1,
            "ticket": service_ticket
        }
    )

    response.raise_for_status()

    results = response.json().get("result", {}).get("results", [])

    if not results:
        return None

    return results[0].get("ui")


def get_umls_synonyms(cui, tgt_url):
    service_ticket = get_service_ticket(tgt_url)

    response = requests.get(
        f"https://uts-ws.nlm.nih.gov/rest/content/current/CUI/{cui}/atoms",
        params={
            "ticket": service_ticket,
            "language": "ENG"
        }
    )

    response.raise_for_status()

    atoms = response.json().get("result", [])

    synonyms = []

    for atom in atoms:
        name = atom.get("name")

        if name:
            synonyms.append(name)

    return list(dict.fromkeys(synonyms))


def search_umls_contextual(
    phrase,
    context_claim,
    tgt_url,
    sbert_model
):
    cache_key = (
        phrase.lower(),
        context_claim.lower()
    )

    if cache_key not in umls_raw_cache:
        cui = get_cui(phrase, tgt_url)

        if cui:
            synonyms = get_umls_synonyms(cui, tgt_url)
        else:
            synonyms = []

        umls_raw_cache[cache_key] = synonyms

        time.sleep(1.0)

    all_related_terms = umls_raw_cache[cache_key]

    if cache_key not in umls_filtered_cache:
        if all_related_terms:
            context_embedding = sbert_model.encode(
                context_claim,
                convert_to_tensor=True
            )

            term_embeddings = sbert_model.encode(
                all_related_terms,
                convert_to_tensor=True
            )

            similarities = util.cos_sim(
                context_embedding,
                term_embeddings
            )[0].cpu().tolist()

            ranked_terms = sorted(
                zip(all_related_terms, similarities),
                key=lambda x: x[1],
                reverse=True
            )

            filtered_terms = [
                (term, score)
                for term, score in ranked_terms
                if score >= UMLS_SIMILARITY_THRESHOLD
            ]

            top_ranked = filtered_terms[:UMLS_TOP_K]
        else:
            top_ranked = []

        umls_filtered_cache[cache_key] = top_ranked

    return umls_filtered_cache[cache_key]


def clear_umls_cache():
    umls_raw_cache.clear()
    umls_filtered_cache.clear()


def initialize_umls(api_key):
    if not api_key:
        raise ValueError("UMLS_API_KEY is not configured.")

    return get_tgt(api_key)