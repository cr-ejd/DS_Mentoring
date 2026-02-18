"""
Text preprocessing and feature engineering utilities
"""

import html
import re
import urllib.parse
from collections import Counter
from typing import Dict, List, Literal, Optional, Union

import pandas as pd
import pycountry
import spacy
from rapidfuzz import fuzz, process
from spellchecker import SpellChecker

try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    raise RuntimeError("SpaCy model not found. Run: python -m spacy download en_core_web_sm")

spell = SpellChecker(language="en")

URL_PATTERN = re.compile(r"http\S+|www\S+|https\S+", flags=re.MULTILINE)
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#(\w+)")
PUNCT_PATTERN = re.compile(r"[^\w\s]")
HTML_TAG_PATTERN = re.compile(r"<.*?>")
HTTP_CODE_PATTERN = re.compile(r"\b\d{3}\b")
REPEATED_CHARS_PATTERN = re.compile(r"(.)\1{2,}")


def correct_spelling(text: str) -> str:
    """Correct spelling of words in the given text using a dictionary-based spell checker.

    Only corrects words longer than 2 characters that are not in the known dictionary.

    Args:
        text (str): Input text to correct.

    Returns:
        str: Text with corrected spelling (or original if no corrections needed).
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
    """Normalize location strings to ISO alpha-3 country codes where possible.

    Improved version:
    - Prioritizes exact 2/3-letter country codes
    - Higher fuzzy thresholds, especially for short strings
    - partial matching only for very short strings and very high score
    - Checks parts strictly from the end (country is usually last)
    - Minimal aliases only for most common abbreviations
    - Falls back to cleaned original string if no reliable country match

    Returns alpha-3 code (lowercase) or cleaned original location.
    """
    df = df.copy()
    df["location"] = df["location"].fillna("").astype(str).str.lower().str.strip()

    country_names = []
    country_map: Dict[str, str] = {}
    for c in pycountry.countries:
        code = c.alpha_3.lower()  # type: ignore
        if hasattr(c, "name"):
            n = c.name.lower()  # type: ignore
            country_names.append(n)
            country_map[n] = code
        if hasattr(c, "official_name") and c.official_name:  # type: ignore
            o = c.official_name.lower()  # type: ignore
            if o not in country_map:
                country_names.append(o)
                country_map[o] = code

    minimal_aliases = {
        "uk": "gbr",
        "gb": "gbr",
        "us": "usa",
        "u.s.": "usa",
        "u.k.": "gbr",
    }
    for alias, code in minimal_aliases.items():
        if alias not in country_map:
            country_names.append(alias)
            country_map[alias] = code

    def _clean_location_string(loc: str) -> str:
        loc = re.sub(r"[^ \w,./-]", "", loc)
        loc = re.sub(r"\s+", " ", loc).strip()
        return loc

    def get_country_code(s: str) -> Optional[str]:
        s = s.strip()
        if not s:
            return None

        if len(s) == 2:
            country = pycountry.countries.get(alpha_2=s.upper())
            if country:
                return country.alpha_3.lower()

        if len(s) == 3:
            country = pycountry.countries.get(alpha_3=s.upper())
            if country:
                return country.alpha_3.lower()

        if s in country_map:
            return country_map[s]

        match = process.extractOne(s, country_names, scorer=fuzz.token_sort_ratio, score_cutoff=88)
        if match:
            return country_map[match[0]]

        if len(s) <= 5:
            match_partial = process.extractOne(
                s, country_names, scorer=fuzz.partial_token_sort_ratio, score_cutoff=90
            )
            if match_partial:
                return country_map[match_partial[0]]

        return None

    def _match_country(loc: str) -> str | None:
        if not loc:
            return None

        cleaned = _clean_location_string(loc)

        code = get_country_code(cleaned)
        if code:
            return code

        separators = [",", "/", "-", "–", "&", " and ", " or ", ";", " in ", " at ", " near "]
        parts: List[str] = [cleaned]
        for sep in separators:
            new_parts = []
            for p in parts:
                if sep in p:
                    split = [sp.strip() for sp in p.split(sep) if sp.strip()]
                    new_parts.extend(split)
                else:
                    new_parts.append(p)
            parts = new_parts

        parts = list(dict.fromkeys(p for p in parts if len(p) >= 2))

        for part in reversed(parts):
            code = get_country_code(part)
            if code:
                return code

        return cleaned if cleaned else None

    cleaned_locations = df["location"].apply(_clean_location_string)
    normalized = cleaned_locations.apply(_match_country)
    df["location_normalized"] = normalized

    return df


def extract_url_features(
    text: Optional[str],
) -> Dict[Literal["url_count", "top_domain", "has_url"], Union[int, Optional[str], bool]]:
    """Extract URL-related features from text.

    Counts URLs, detects their presence and extracts the most frequent domain.

    Args:
        text (Optional[str]): Input text that may contain URLs.

    Returns:
        dict: Dictionary with keys:
            - url_count (int): Number of URLs found
            - top_domain (str or None): Most common domain
            - has_url (bool): Whether any URL is present
    """
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
    """Safely convert any value to string, handling None and NaN.

    Args:
        text (Optional[str]): Input value (can be None, float nan, etc.).

    Returns:
        str: Empty string if input is None/nan, otherwise string representation.
    """
    if text is None:
        return ""
    if isinstance(text, float):
        return ""
    return str(text)


def count_typos(text: Optional[str]) -> int:
    """Count potential typos in text by checking unknown words (length >= 4).

    Removes URLs, HTML, mentions, hashtags and punctuation before analysis.

    Args:
        text (Optional[str]): Input text.

    Returns:
        int: Number of unknown words (considered typos).
    """
    text = safe_text(text)
    if not text:
        return 0
    text = urllib.parse.unquote(text)
    text = html.unescape(text)
    text = text.lower()
    text = URL_PATTERN.sub("", text)
    text = HTML_TAG_PATTERN.sub("", text)
    text = MENTION_PATTERN.sub("", text)
    text = HASHTAG_PATTERN.sub(r"\1", text)
    text = PUNCT_PATTERN.sub("", text)
    words = re.findall(r"\b[a-z]{4,}\b", text)
    real_typos = sum(1 for w in words if spell.unknown([w]))
    return real_typos


def full_preprocess(text: Optional[str]) -> str:
    """Fully preprocess tweet text: clean, lemmatize and optionally correct spelling.

    Steps:
        - Decode URL encoding and HTML entities
        - Remove HTML tags, URLs, mentions, hashtags symbols, punctuation
        - Lowercase and normalize whitespace
        - Lemmatize using spaCy (remove stop words and punctuation)
        - (Spelling correction is currently disabled)

    Args:
        text (Optional[str]): Raw tweet text.

    Returns:
        str: Cleaned and lemmatized text.
    """
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

    doc = nlp(text)
    lemmatized = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct and len(token.lemma_) > 1
    ]
    lemmatized_text = " ".join(lemmatized)

    # disabled for now to improve speed
    # corrected_text = correct_spelling(lemmatized_text)
    # return corrected_text

    return lemmatized_text


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply full feature extraction pipeline to a DataFrame of tweets.

    Adds the following columns:
        - url_count, top_domain, has_url
        - hashtag_count, has_hashtag
        - typo_count, has_typos
        - clean_text (lemmatized)
        - location_normalized (country code or cleaned string)

    Args:
        df (pd.DataFrame): Input DataFrame with at least 'text' column,
                           optionally 'location'.

    Returns:
        pd.DataFrame: DataFrame with added feature columns.
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
