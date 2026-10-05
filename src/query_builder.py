def build_search_query(claim, terms, max_terms=15):
    if not terms:
        return claim

    selected_terms = terms[:max_terms]
    terms_query = " OR ".join(
        f'"{term}"' for term in selected_terms
    )

    return f'{claim} ({terms_query})'