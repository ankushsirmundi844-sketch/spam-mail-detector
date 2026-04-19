"""
preprocessor.py
---------------
Text cleaning, feature engineering, and NLP utilities
for the Spam Mail Detector pipeline.
"""

import re
import string
import numpy as np
import pandas as pd

# ── Stopwords (curated minimal set for SMS context) ──────────────────────────
STOPWORDS = {
    "i","me","my","myself","we","our","ours","you","your","he","him","his",
    "she","her","it","its","they","them","their","what","which","who","this",
    "that","these","those","am","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","a","an","the","and","but","if",
    "or","as","at","by","for","with","about","into","through","to","from",
    "in","out","on","off","then","so","than","just","will","can","not","no",
    "it","its","the","to","of","a","in","that","have","i","it","for","on",
    "are","with","as","this","be","was","is","an","at","or","by","we","you",
}


def clean_text(text: str) -> str:
    """
    Full NLP preprocessing pipeline:
      1. Lowercase
      2. Replace URLs with URL token
      3. Replace phone numbers with PHONE token
      4. Replace currency amounts with MONEY token
      5. Replace other numbers with NUM token
      6. Remove punctuation
      7. Tokenize and remove stopwords (keeping length > 1)
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " URL ", text)
    text = re.sub(r"\b0[789]\d{9}\b|\b\d{11}\b", " PHONE ", text)
    text = re.sub(r"[£$€]\s?\d+[\d,]*", " MONEY ", text)
    text = re.sub(r"\b\d+\b", " NUM ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract hand-crafted features from raw message text.
    These are used alongside TF-IDF for an enriched feature set.
    """
    df = df.copy()
    msg = df["message"].astype(str)

    df["feat_length"]        = msg.apply(len)
    df["feat_word_count"]    = msg.apply(lambda x: len(x.split()))
    df["feat_caps_ratio"]    = msg.apply(lambda x: sum(c.isupper() for c in x) / max(len(x), 1))
    df["feat_digit_ratio"]   = msg.apply(lambda x: sum(c.isdigit() for c in x) / max(len(x), 1))
    df["feat_punct_count"]   = msg.apply(lambda x: sum(c in string.punctuation for c in x))
    df["feat_has_url"]       = msg.str.contains(r"http|www|bit\.ly", case=False, regex=True).astype(int)
    df["feat_has_phone"]     = msg.str.contains(r"\b\d{10,11}\b", regex=True).astype(int)
    df["feat_has_currency"]  = msg.str.contains(r"[£$€]|\bpound|\bdollar", case=False, regex=True).astype(int)
    df["feat_exclaim_count"] = msg.apply(lambda x: x.count("!"))
    df["feat_has_free"]      = msg.str.contains(r"\bfree\b", case=False, regex=True).astype(int)
    df["feat_has_win"]       = msg.str.contains(r"\bwin\b|\bwinner\b|\bwon\b", case=False, regex=True).astype(int)
    df["feat_has_urgent"]    = msg.str.contains(r"\burgent\b|\bimportant\b|\bnow\b", case=False, regex=True).astype(int)
    df["feat_has_call"]      = msg.str.contains(r"\bcall\b|\btxt\b|\btext\b", case=False, regex=True).astype(int)
    df["feat_avg_word_len"]  = msg.apply(lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0)

    return df


FEATURE_COLS = [
    "feat_length", "feat_word_count", "feat_caps_ratio", "feat_digit_ratio",
    "feat_punct_count", "feat_has_url", "feat_has_phone", "feat_has_currency",
    "feat_exclaim_count", "feat_has_free", "feat_has_win", "feat_has_urgent",
    "feat_has_call", "feat_avg_word_len",
]
