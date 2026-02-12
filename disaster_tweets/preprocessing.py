"""
Text preprocessing and feature engineering utilities
"""

import re
from collections import Counter
from typing import Dict, Literal, Optional, Union

import pandas as pd


def extract_url_features(text: Optional[str]) -> Dict[Literal["url_count", "top_domain", "has_url"], Union[int, Optional[str], bool]]:
    """
    Extract URL-related features from a text string.

    Parameters
    ----------
    text : Optional[str]
        Raw tweet text.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - "url_count" (int): Number of detected URLs.
        - "top_domain" (Optional[str]): Most frequent domain name
          extracted from URLs, or None if no domains found.
        - "has_url" (bool): Indicator whether at least one URL exists.
    """
    if not text:
        return {"url_count": 0, "top_domain": None, "has_url": False}

    urls = re.findall(r"(https?://[^\s]+|www\.[^\s]+)", text)
    url_count = len(urls)
    has_url = url_count > 0

    domains = []
    for url in urls:
        match = re.search(r"(?:www\.)?([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", url)
        if match:
            domains.append(match.group(1).lower())

    top_domain = None
    if domains:
        most_common_domain = Counter(domains).most_common(1)
        top_domain = most_common_domain[0][0] if most_common_domain else None

    return {"url_count": url_count, "top_domain": top_domain, "has_url": has_url}


def count_typos(text: Optional[str]) -> int:
    """
    Estimate the number of potential typos or stress-related patterns in text.

    Uses simple regex patterns such as:
    - Excessive repeated characters (e.g., "heeeelp")
    - Double letters
    - Very short standalone words
    - Repeated syllable fragments

    Parameters
    ----------
    text : Optional[str]
        Raw tweet text.

    Returns
    -------
    int
        Heuristic count of detected typo-like patterns.
    """
    if not text:
        return 0

    text = text.lower()

    typo_patterns = [
        r"(.)\1{2,}",  # excessive repeated characters
        r"([a-z])\1{1,}",  # double letters
        r"([a-z]+)([a-z]+)\1",  # this is for repeated syllable-like fragments
    ]

    typo_count = 0
    for pattern in typo_patterns:
        typo_count += len(re.findall(pattern, text))

    typo_count += len(re.findall(r"\b\w{1,2}\b", text))

    return typo_count


def full_preprocess(text: Optional[str]) -> str:
    """
    Perform full text preprocessing for future modeling.

    Steps:
    - Remove URLs
    - Remove mentions
    - Remove hashtags
    - Remove punctuation
    - Lowercase
    - Normalize selected disaster-related word forms
    - Remove basic English stopwords
    - Normalize whitespace

    Parameters
    ----------
    text : Optional[str]
        Raw tweet text.

    Returns
    -------
    str
        Cleaned and normalized text suitable for
        TF-IDF, n-grams, or other vectorizers.
    """
    if not text:
        return ""

    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)

    replacements = {
        "fires": "fire",
        "flooded": "flood",
        "earthquakes": "earthquake",
        "hurricanes": "hurricane",
        "injured": "injury",
    }

    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)

    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
    }

    text = " ".join(w for w in text.split() if w not in stop_words)

    return text.strip()


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply full feature engineering pipeline to a DataFrame.

    Adds the following columns:
    - url_count (int)
    - top_domain (Optional[str])
    - has_url (bool)
    - typo_count (int)
    - has_typos (int)
    - clean_text (str)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a 'text' column.

    Returns
    -------
    pd.DataFrame
        New DataFrame copy with additional engineered features.
    """
    df = df.copy()

    url_features = df["text"].apply(extract_url_features)

    df["url_count"] = url_features.map(lambda x: x["url_count"])
    df["top_domain"] = url_features.map(lambda x: x["top_domain"])
    df["has_url"] = url_features.map(lambda x: x["has_url"])

    df["typo_count"] = df["text"].apply(count_typos)
    df["has_typos"] = (df["typo_count"] > 2).astype(int)

    df["clean_text"] = df["text"].apply(full_preprocess)

    return df
