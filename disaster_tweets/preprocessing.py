"""
Text preprocessing and feature engineering utilities
"""

import re
from collections import Counter
from typing import Dict, Literal, Optional, Union

import pandas as pd
import spacy
from spellchecker import SpellChecker
import html
import urllib.parse
from rapidfuzz import fuzz, process
import pycountry

try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    raise RuntimeError(
        "SpaCy model not found. Run: python -m spacy download en_core_web_sm"
    )

spell = SpellChecker(language="en")

# pre-compiled regex patterns for performance
URL_PATTERN = re.compile(r"http\S+|www\S+|https\S+", flags=re.MULTILINE)
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#(\w+)")  
PUNCT_PATTERN = re.compile(r"[^\w\s]")
HTML_TAG_PATTERN = re.compile(r"<.*?>")
HTTP_CODE_PATTERN = re.compile(r"\b\d{3}\b")  
REPEATED_CHARS_PATTERN = re.compile(r"(.)\1{2,}")

def correct_spelling(text: str) -> str:
    """
    Safe word-by-word spell correction.
    """
    if not text:
        return text

    words = text.split()
    corrected_words = []

    for word in words:
        if len(word) < 3:
            corrected_words.append(word)
            continue

        if word in spell:
            corrected_words.append(word)
            continue

        suggestion = spell.correction(word)
        corrected_words.append(suggestion if suggestion else word)

    return " ".join(corrected_words)

def normalize_locations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robust location normalization using pycountry + fuzzy fallback.
    - Countries are converted to ISO alpha_3 (lowercase, e.g. 'usa' → 'usa')
    - Cities/regions stay cleaned but unchanged
    No hard-coded lists, scalable.
    """
    df = df.copy()

    df["location"] = df["location"].fillna("").astype(str).str.lower().str.strip()

    def _clean_location_string(loc: str) -> str:
        loc = re.sub(r"[^\w\s]", "", loc)
        loc = re.sub(r"\s+", " ", loc).strip()
        return loc

    def _match_country(loc: str) -> str | None:
        if not loc:
            return None

        if len(loc) == 2:
            country = pycountry.countries.get(alpha_2=loc.upper())
            if country:
                return country.alpha_3.lower()

        if len(loc) == 3:
            country = pycountry.countries.get(alpha_3=loc.upper())
            if country:
                return country.alpha_3.lower()

        try:
            country = pycountry.countries.lookup(loc)
            return country.alpha_3.lower()
        except LookupError:
            pass

        country_names = [c.name.lower() for c in pycountry.countries]
        match = process.extractOne(
            loc,
            country_names,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=90
        )
        if match:
            matched_country = pycountry.countries.lookup(match[0])
            return matched_country.alpha_3.lower()

        return None

    cleaned_locations = df["location"].apply(_clean_location_string)
    normalized = cleaned_locations.apply(_match_country)
    df["location_normalized"] = normalized

    return df

def extract_url_features(text: Optional[str]) -> Dict[Literal["url_count", "top_domain", "has_url"], Union[int, Optional[str], bool]]:
    text = safe_text(text)

    if not text:
        return {"url_count": 0, "top_domain": None, "has_url": False}

    decoded_text = urllib.parse.unquote(text)
    urls = URL_PATTERN.findall(decoded_text)

    url_count = len(urls)
    has_url = url_count > 0

    domains = []
    for url in urls:
        match = re.search(r"(?:www\.)?([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", url)
        if match:
            domains.append(match.group(1).lower())

    top_domain = Counter(domains).most_common(1)[0][0] if domains else None

    return {"url_count": url_count, "top_domain": top_domain, "has_url": has_url}

def safe_text(text: Optional[str]) -> str:
    """
    Convert NaN/None to empty string and ensure text type.
    """
    if text is None:
        return ""
    if isinstance(text, float):
        # catches np.nan
        return ""
    return str(text)

def count_typos(text: Optional[str]) -> int:
    text = safe_text(text)

    if not text:
        return 0

    text = urllib.parse.unquote(text)
    text = html.unescape(text)
    text = text.lower()

    text = URL_PATTERN.sub("", text)
    text = HTML_TAG_PATTERN.sub("", text)

    typo_pattern = r"(.)\1{2,}|([a-z])\1{1,}|\b\w{1,2}\b"
    typo_count = len(re.findall(typo_pattern, text))

    words = re.findall(r"\b[a-z]{3,}\b", text)
    real_typos = sum(
        1 for w in words
        if spell.unknown([w]) and not REPEATED_CHARS_PATTERN.search(w)
    )

    return typo_count + real_typos

def full_preprocess(text: Optional[str]) -> str:
    text = safe_text(text)

    if not text:
        return ""

    text = urllib.parse.unquote(text)
    text = html.unescape(text)

    text = HTML_TAG_PATTERN.sub("", text)
    text = URL_PATTERN.sub("", text)
    text = MENTION_PATTERN.sub("", text)
    text = HASHTAG_PATTERN.sub(r"\1", text)
    text = PUNCT_PATTERN.sub("", text)
    text = HTTP_CODE_PATTERN.sub("", text)

    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)

    text = correct_spelling(text)

    doc = nlp(text)
    lemmatized = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct
    ]

    return " ".join(lemmatized)

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply full feature engineering pipeline to a DataFrame.

    Adds:
    - url_count, top_domain, has_url
    - hashtag_count, has_hashtag
    - typo_count, has_typos
    - clean_text
    - location_normalized (fuzzy matched)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with 'text' and optional 'location'.

    Returns
    -------
    pd.DataFrame
        Enhanced DataFrame.
    """
    df = df.copy()

    url_features = df["text"].apply(extract_url_features)
    df["url_count"] = url_features.map(lambda x: x["url_count"])
    df["top_domain"] = url_features.map(lambda x: x["top_domain"])
    df["has_url"] = url_features.map(lambda x: x["has_url"])

    df["hashtag_count"] = df["text"].apply(lambda x: len(HASHTAG_PATTERN.findall(str(x))))
    df["has_hashtag"] = (df["hashtag_count"] > 0).astype(int)

    df["typo_count"] = df["text"].apply(count_typos)
    df["has_typos"] = (df["typo_count"] > 2).astype(int)

    df["clean_text"] = df["text"].apply(full_preprocess)

    df = normalize_locations(df)

    return df