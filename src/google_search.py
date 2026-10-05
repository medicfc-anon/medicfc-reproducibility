from googleapiclient.discovery import build


def google_search(query, api_key, cse_id, max_results=10):
    try:
        service = build(
            "customsearch",
            "v1",
            developerKey=api_key
        )

        result = service.cse().list(
            q=query,
            cx=cse_id,
            num=min(max_results, 10)
        ).execute()

        return result.get("items", [])

    except Exception as e:
        print(f"Search error: {e}")
        return []