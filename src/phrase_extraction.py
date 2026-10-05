from spacy.matcher import Matcher


def create_phrase_matcher(nlp):
    matcher = Matcher(nlp.vocab)

    patterns = [
        [{"POS": "NOUN"}],
        [{"POS": "ADJ"}, {"POS": "NOUN"}],
        [{"POS": "NOUN"}, {"POS": "NOUN"}],
        [{"POS": "NOUN"}, {"POS": "NOUN"}, {"POS": "NOUN"}],
        [{"POS": "PROPN"}],
        [{"POS": "PROPN"}, {"POS": "PROPN"}],
    ]

    matcher.add("BIOMEDICAL_PHRASES", patterns)

    return matcher


def extract_biomedical_phrases(text, nlp, matcher, stop_words):
    doc = nlp(text)

    phrases = []
    seen = set()

    for _, start, end in matcher(doc):
        phrase = doc[start:end].text.lower().strip()

        if (
            phrase not in stop_words
            and len(phrase) > 2
            and phrase not in seen
        ):
            phrases.append(phrase)
            seen.add(phrase)

    return phrases