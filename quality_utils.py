"""Utility helpers for quality filtering, dedup, and lightweight confidence scoring."""

import hashlib
import re


_LOW_SIGNAL_PATTERNS = [
    r"privacy policy",
    r"terms of (use|service)",
    r"all rights reserved",
    r"cookie(s)?",
    r"sign in",
    r"subscribe",
]


def normalize_text(text):
    return " ".join((text or "").split())


def text_quality_score(text):
    """Return a 0..1 quality score for short snippets."""
    t = normalize_text(text)
    if len(t) < 40:
        return 0.0

    alpha = sum(1 for ch in t if ch.isalpha())
    alpha_ratio = alpha / max(len(t), 1)
    symbol = sum(1 for ch in t if not (ch.isalnum() or ch.isspace() or ch in ".,!?;:'\"()-"))
    symbol_ratio = symbol / max(len(t), 1)

    tokens = re.findall(r"[a-z0-9]+", t.lower())
    unique_ratio = len(set(tokens)) / max(len(tokens), 1)

    score = 0.0
    score += 0.35 * min(1.0, len(t) / 320.0)
    score += 0.30 * min(1.0, alpha_ratio / 0.8)
    score += 0.20 * unique_ratio
    score += 0.15 * max(0.0, 1.0 - min(1.0, symbol_ratio / 0.2))

    lowered = t.lower()
    for pat in _LOW_SIGNAL_PATTERNS:
        if re.search(pat, lowered):
            score *= 0.6

    return max(0.0, min(1.0, score))


def is_high_signal_text(text, min_score=0.48):
    return text_quality_score(text) >= min_score


def content_hash(text, prefix_chars=280):
    base = normalize_text(text)[:prefix_chars].lower()
    return hashlib.sha1(base.encode("utf-8", errors="ignore")).hexdigest()


def lexical_overlap_score(query, text):
    q = {w for w in re.findall(r"[a-z0-9]+", (query or "").lower()) if len(w) > 2}
    t = {w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(w) > 2}
    if not q or not t:
        return 0.0
    return len(q.intersection(t)) / max(len(q), 1)


def source_weight(source):
    src = (source or "").lower()
    if src in {"bootstrap", "reward_positive", "reflection"}:
        return 1.1
    if src in {"local_ingest", "web_lookup", "lookup", "fallback_lookup"}:
        return 1.0
    return 0.9
