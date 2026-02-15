"""
Text preprocessing and feature engineering utilities
"""

import re
from collections import Counter
from typing import Dict, Literal, Optional, Union

import pandas as pd
import spacy
from spellchecker import SpellChecker

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# pre-compile regexes, added for reusability
URL_PATTERN = re.compile(r"http\S+|www\S+|https\S+", flags=re.MULTILINE)
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#\w+")
PUNCT_PATTERN = re.compile(r"[^\w\s]")

spell = SpellChecker(language="en")


def correct_spelling(text: str) -> str:
    """Fixes typos based on a dictionary."""
    if not text:
        return text
    
    words = text.split()
    corrected = []
    
    for word in words:
        if spell.unknown([word]):
            suggestions = spell.correction([word])  # type: ignore 
            if suggestions and suggestions[0] is not None:
                corrected.append(suggestions[0])
            else:
                corrected.append(word)
        else:
            corrected.append(word)
    
    return ' '.join(corrected)


def extract_url_features(
    text: Optional[str],
) -> Dict[Literal["url_count", "top_domain", "has_url"], Union[int, Optional[str], bool]]:
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
    if not text:
        return ""

    text = URL_PATTERN.sub("", text)
    text = MENTION_PATTERN.sub("", text)
    text = HASHTAG_PATTERN.sub("", text)
    text = PUNCT_PATTERN.sub("", text)

    # for cleaning html entities (for n-gram analysis)
    text = re.sub(r"&[a-zA-Z0-9#]+;", "", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")

    #how to clean all html tags?

    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)

    text = correct_spelling(text)

    replacements = {      #doesn't work well on new data
        "fires": "fire",
        "flooded": "flood",
        "earthquakes": "earthquake",
        "hurricanes": "hurricane",
        "injured": "injury",
        "usa": "united states",  # look for more country name issues / weird stuff
        "us": "united states",
        "u.s.": "united states",
        "unitedstates": "united states",
        "amp": "",
    }

    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)

    doc = nlp(text)
    lemmatized = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(lemmatized)


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
