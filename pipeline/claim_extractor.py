"""
Claim extraction module for TrustAgent.

Extracts individual, verifiable claims from raw text (typed or extracted
from PDF).  Uses spaCy sentence segmentation plus heuristic filters to
identify sentences that contain product-attribute assertions.
"""

import re
from typing import List

# ---------------------------------------------------------------------------
# Lazy-loaded spaCy model
# ---------------------------------------------------------------------------
_nlp = None


def _get_nlp():
    """Load spaCy model lazily (downloads en_core_web_sm if missing)."""
    global _nlp
    if _nlp is None:
        try:
            import spacy
            try:
                _nlp = spacy.load("en_core_web_sm")
            except OSError:
                # Model not installed yet – download it
                import subprocess, sys
                subprocess.check_call(
                    [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                _nlp = spacy.load("en_core_web_sm")
        except ImportError:
            _nlp = None
    return _nlp


# ---------------------------------------------------------------------------
# Open-Domain Product Entity Recognition (spaCy NER & Noun Chunks)
# ---------------------------------------------------------------------------

def _has_product_entity(sentence: str) -> bool:
    """Check if the sentence mentions any product entity, brand, or model name dynamically."""
    nlp = _get_nlp()
    if nlp is not None:
        doc = nlp(sentence)
        for ent in doc.ents:
            if ent.label_ in ("ORG", "PRODUCT", "WORK_OF_ART", "PERSON") and len(ent.text.strip()) >= 2:
                return True
        for chunk in doc.noun_chunks:
            words = [w for w in chunk.text.split() if len(w) >= 2 and (w[0].isupper() or any(c.isdigit() for c in w))]
            if words:
                return True

    # Fallback: check for capitalized words or alphanumeric model identifiers
    tokens = sentence.split()
    capitalized = [t for t in tokens if len(t) >= 2 and t[0].isupper() and t.lower() not in ("the", "this", "that", "there", "here", "it", "in", "on", "for", "and", "or", "only", "we", "they", "is", "are")]
    return len(capitalized) >= 1


# ---------------------------------------------------------------------------
# Verifiable attribute keywords
# ---------------------------------------------------------------------------
_ATTRIBUTE_KEYWORDS = {
    "price", "cost", "costs", "priced",
    "weight", "weighs", "heavy",
    "battery", "battery life",
    "storage", "ram", "memory",
    "color", "colour",
    "screen", "display", "resolution",
    "range", "horsepower", "hp",
    "power", "watt", "watts",
    "capacity", "suction",
    "speed", "megapixel", "mp",
    "refresh rate", "hz",
    "noise", "db", "decibel",
    "pressure", "bar", "bars",
    "accuracy", "coverage",
    "thickness", "diameter",
    "flight time", "response time",
    "fuel economy", "mpg",
    "waterproof", "water resistance",
    "discount", "offer", "sale", "genuine", "authentic", "original",
    "warranty", "stock", "reviews", "rating", "stars", "fee"
}

_UNIT_PATTERNS = re.compile(
    r'\b(?:USD|\$|usd|inr|rs\.?|₹|hours?|days?|minutes?|grams?|kg|GB|TB|mAh|inches?|'
    r'watts?|kWh|miles?|hp|mpg|MP|Pa|cu\s*ft|sq\s*ft|mm|dB|Hz|ms|bars?|'
    r'levels?|liters?|meters?|degrees?|percent|K)\b',
    re.IGNORECASE,
)


def _has_number(text: str) -> bool:
    """Check if text contains a number (integer or decimal)."""
    return bool(re.search(r'\d+(?:\.\d+)?', text))


def _has_attribute_keyword(text: str) -> bool:
    """Check if text contains a verifiable attribute keyword."""
    text_lower = text.lower()
    for kw in _ATTRIBUTE_KEYWORDS:
        if kw in text_lower:
            return True
    return False


def _has_unit(text: str) -> bool:
    """Check if text contains a measurement unit."""
    return bool(_UNIT_PATTERNS.search(text))


def _is_verifiable_claim(sentence: str) -> bool:
    """
    Open-domain claim verification check:
    A sentence is a verifiable claim if it has a product/subject entity AND
    asserts a number, unit, color, or verifiable attribute.
    """
    if len(sentence.split()) < 3:
        return False

    has_entity = _has_product_entity(sentence)
    has_number_or_color = _has_number(sentence) or any(
        c in sentence.lower() for c in [
            "black", "white", "silver", "gray", "red", "blue",
            "green", "cyan", "gold", "purple", "charcoal", "navy",
            "stainless", "pearl", "neon", "rose"
        ]
    )
    has_attr = _has_attribute_keyword(sentence) or _has_unit(sentence)

    return has_entity and (has_number_or_color or has_attr)


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text content from a PDF file."""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(pdf_path)
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n".join(text_parts)
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {e}")


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_claims(text: str) -> List[str]:
    """
    Extract verifiable product claims from raw text.

    Uses spaCy for sentence segmentation, then filters for sentences
    that contain product mentions + numeric/attribute assertions.
    Falls back to regex sentence splitting if spaCy is unavailable.

    Parameters
    ----------
    text : str
        Raw text input (from user typing or PDF extraction).

    Returns
    -------
    list[str]
        List of individual verifiable claim strings.
    """
    if not text or not text.strip():
        return []

    # Clean up the text — normalize whitespace but preserve sentence-ending punctuation
    text = text.strip()
    # Replace newlines with spaces but keep period boundaries
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)

    # Segment into sentences
    nlp = _get_nlp()
    if nlp is not None:
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
    else:
        # Fallback: regex-based sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)

    # Further split long sentences on conjunctions (and, also, while, etc.)
    # This handles: "It is priced at 499 USD and comes with 1 TB of storage."
    expanded_sentences = []
    for sent in sentences:
        if len(sent.split()) > 12:
            # Try splitting on common conjunctions
            sub_parts = re.split(r'\.\s+|\band\b\s+|\balso\b\s+|\bwhile\b\s+|\bwhereas\b\s+', sent)
            sub_parts = [p.strip().rstrip('.') + '.' for p in sub_parts if p.strip()]
            if len(sub_parts) > 1:
                expanded_sentences.extend(sub_parts)
            else:
                expanded_sentences.append(sent)
        else:
            expanded_sentences.append(sent)

    # Filter for verifiable claims
    claims = []
    seen = set()
    for sent in expanded_sentences:
        sent = sent.strip()
        if not sent or sent in seen:
            continue
        if _is_verifiable_claim(sent):
            claims.append(sent)
            seen.add(sent)

    # If no strict claims were extracted, treat the whole text as a single claim
    if not claims and len(text.split()) >= 2:
        claims = [text]

    return claims


def extract_claims_from_pdf(pdf_path: str) -> List[str]:
    """Extract verifiable claims from a PDF file."""
    text = extract_text_from_pdf(pdf_path)
    return extract_claims(text)
