"""
src/pipeline.py
───────────────
Shared preprocessing pipeline for all notebooks.

Usage in any notebook:
    import sys; sys.path.append('..')
    from src.pipeline import (
        load_data, preprocess_text, build_capitalization_map,
        restore_capitalization, lemmatize_review,
        NEGATIONS, stop_words_filtered, nlp
    )
"""

import re
import warnings
import contractions
import spacy
from spacy.pipeline import EntityRuler
import nltk
from nltk.corpus import stopwords

warnings.filterwarnings("ignore")

# ── NLTK resources ─────────────────────────────────────────────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("vader_lexicon", quiet=True)

# ── spaCy model + custom EntityRuler ──────────────────────────────────────────
nlp = spacy.load("en_core_web_sm")

if "entity_ruler" not in nlp.pipe_names:
    ruler = nlp.add_pipe("entity_ruler", before="ner")
else:
    ruler = nlp.get_pipe("entity_ruler")

ruler.add_patterns([
    {"label": "ORG", "pattern": "Spotify"},
    {"label": "ORG", "pattern": "spotify"},
    {"label": "ORG", "pattern": "Apple Music"},
    {"label": "ORG", "pattern": "Google Play"},
])

# ── Negations (preserved during stopword removal) ─────────────────────────────
NEGATIONS = {
    "not", "no", "nor", "neither", "nobody", "nothing", "never", "cannot",
    "can't", "won't", "don't", "doesn't", "didn't",
    "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't",
}

# ── Stopwords with negations kept ─────────────────────────────────────────────
stop_words_filtered = set(stopwords.words("english")) - NEGATIONS


# ── Data loading ───────────────────────────────────────────────────────────────
def load_data(path: str = "../Data/reviews_spotify_kaggle.csv"):
    """Load and standardise the raw Spotify reviews CSV."""
    import pandas as pd
    df = pd.read_csv(path)
    df = df.rename(columns={"Review": "Content", "Time_submitted": "Date"})
    df = df.dropna(subset=["Content"]).reset_index(drop=True)
    return df


# ── Base cleaning ──────────────────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """
    Steps 1–5:
      1. Strip HTML tags
      2. Expand contractions + lowercase
      3. Remove URLs
      4. Remove @handles and #hashtags
      5. Normalise whitespace
    Punctuation and emojis are deliberately preserved as sentiment signals.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<.*?>", "", text)                      # 1. HTML
    text = contractions.fix(text).lower()                  # 2. contractions + lowercase
    text = re.sub(r"https?://\S+|www\.\S+", "", text)      # 3. URLs
    text = re.sub(r"@\w+|#\w+", "", text)                  # 4. handles / hashtags
    return " ".join(text.split())                          # 5. whitespace


# ── Capitalisation map (build once, reuse everywhere) ─────────────────────────
def build_capitalization_map(raw_texts, max_rows: int = None):
    """
    Scan raw review texts and build a lowercase→original-case lookup dict.
    Used to restore proper nouns / brand names before NER processing.

    Args:
        raw_texts: iterable of raw review strings (e.g. df['Content'])
        max_rows:  if set, only scan the first N rows (e.g. 500 for speed)
    """
    cap_map = {}
    texts = raw_texts if max_rows is None else list(raw_texts)[:max_rows]
    for raw_text in texts:
        for word in raw_text.split():
            cw = word.rstrip(",.!?;:")
            lw = cw.lower()
            if cw and cw[0].isupper() and len(cw) > 1:
                if lw not in cap_map:
                    cap_map[lw] = cw
    return cap_map


def restore_capitalization(text: str, capitalization_map: dict) -> str:
    """Restore capitalisation before spaCy NER processing."""
    words = text.split()
    restored = []
    for word in words:
        cw = word.rstrip(",.!?;:")
        punct = word[len(cw):]
        restored.append(capitalization_map.get(cw.lower(), cw) + punct)
    return " ".join(restored)


# ── Negation-preserving lemmatisation ─────────────────────────────────────────
def lemmatize_review(text: str, capitalization_map: dict) -> list:
    """
    Tokenise, remove stopwords (keeping negations), lemmatise.
    Named entities (ORG etc.) are excluded from the output.

    Returns a list of lemma strings.
    """
    doc = nlp(restore_capitalization(text, capitalization_map))
    named_entities = {ent.text.lower() for ent in doc.ents}
    lemmas = []
    for token in doc:
        if token.text.lower() in named_entities:
            continue
        if (
            token.text.lower() in stop_words_filtered
            and token.pos_ not in ("ADJ", "VERB")
        ):
            continue
        if token.pos_ in ("ADJ", "VERB", "NOUN"):
            lemmas.append(token.lemma_)
        elif token.text.lower() in NEGATIONS:
            lemmas.append(token.text.lower())
    return lemmas
