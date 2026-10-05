import requests
from bs4 import BeautifulSoup


def fetch_article_content(url, timeout=10):
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=timeout
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.content,
            "html.parser"
        )

        for element in soup(
            ["script", "style", "nav", "footer", "header", "aside", "form"]
        ):
            element.decompose()

        text = soup.get_text(
            separator=" ",
            strip=True
        )

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        return " ".join(lines)

    except Exception:
        return ""


def extract_sentences(text, nlp_sent):
    if not text or len(text) < 50:
        return []

    text = text[:100000]

    try:
        doc = nlp_sent(text)

        return [
            sent.text.strip()
            for sent in doc.sents
            if len(sent.text.strip()) > 30
        ]

    except Exception:
        return []