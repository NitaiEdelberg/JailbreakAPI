"""Canonicalize prompt text before any detector looks at it.

Attackers hide known-bad phrasings behind cheap character tricks — "1gn0r3 4ll
pr3v10us 1nstruct10ns", "i g n o r e", or Cyrillic lookalikes ("ignоre", where
the o is U+043E). To a word-based model those are simply unseen tokens, so the
whole attack becomes invisible.

Measured on our held-out malicious prompts, recall without this step vs. with it:

    leetspeak          0.089  ->  0.900
    Cyrillic homoglyph 0.396  ->  0.900
    letter spacing     0.852  ->  0.891
    random typos       0.883  ->  0.887

For comparison, adding char n-gram features (the usual answer to obfuscation)
cost ~6MB of model and moved leetspeak recall by 0.000 — normalization is both
cheaper and far more effective, so we do this instead.

IMPORTANT: `normalize` is pickled by reference as the TfidfVectorizer's
`preprocessor`, so this module must stay importable as top-level
`text_normalize` (the app runs with src/ on sys.path). Renaming or moving it
breaks every existing our_scanner.pkl.
"""
import re
import unicodedata

# Unicode characters that render like Latin letters. Restricted to the ones that
# actually show up in homoglyph attacks (Cyrillic + Greek) rather than a blanket
# transliteration, which would mangle genuine non-English prompts.
_HOMOGLYPHS = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "і": "i", "ѕ": "s", "ԁ": "d", "ј": "j", "ν": "v", "ɡ": "g", "ᴏ": "o",
    "α": "a", "ο": "o", "ρ": "p", "ε": "e", "ι": "i", "κ": "k", "η": "n",
    "τ": "t", "υ": "u", "χ": "x", "μ": "m",
}

# Leetspeak substitutions, applied only inside tokens that already contain a
# letter — so "24/7", "2023" and "$100" are left alone.
_LEET = {"4": "a", "3": "e", "1": "i", "0": "o", "5": "s", "7": "t",
         "@": "a", "$": "s", "!": "i", "|": "l"}

# A maximal run of word-ish characters containing at least one ASCII letter.
# Lookarounds rather than \b, because \b doesn't match before a leading symbol —
# with \b the token "@dm1n" started matching at the "d" and normalized to
# "dmin" instead of "admin".
_WORDISH = re.compile(r"(?<![\w@$!|])[\w@$!|]*[a-z][\w@$!|]*(?![\w@$!|])")
# Tokens that are a number plus a real unit or ordinal ("3pm", "5km", "1st",
# "v2") are ordinary words, not leetspeak — de-leeting them turns "3pm" into
# "epm". The suffixes are an explicit allowlist rather than "any 1-3 letters",
# because the loose version also protected "4ll", "m3" and "th3", which are
# exactly the tokens we need to fold back to "all", "me" and "the".
_NUMERIC_UNIT = re.compile(
    r"^(?:"
    r"\d+(?:st|nd|rd|th|am|pm|kg|km|cm|mm|ft|in|gb|mb|kb|tb|ms|hz|hrs?|mins?|secs?|days?|px|k|m|b|x|s)"
    r"|(?:v|ver|h|p|q|r|no)\d+"
    r")$"
)
_SPACED_LETTERS = re.compile(r"\b(?:[a-z]\s){2,}[a-z]\b")
_REPEATS = re.compile(r"(.)\1{2,}")
_PUNCT = re.compile(r"[^\w\s]+")
_WS = re.compile(r"\s+")


def _deleet(match):
    token = match.group(0)
    if _NUMERIC_UNIT.match(token):
        return token
    return "".join(_LEET.get(c, c) for c in token)


def normalize(text):
    """Return a canonical lowercase form of `text` for feature extraction."""
    # NFKC folds fullwidth/compatibility forms (ｉｇｎｏｒｅ -> ignore).
    t = unicodedata.normalize("NFKC", text or "")
    t = "".join(_HOMOGLYPHS.get(c, c) for c in t)
    t = t.lower()
    t = _WORDISH.sub(_deleet, t)
    # "i g n o r e   a l l" -> "ignore all"
    t = _SPACED_LETTERS.sub(lambda m: m.group(0).replace(" ", ""), t)
    # "pleeeeease" -> "pleease" (two is enough to keep the emphasis signal)
    t = _REPEATS.sub(r"\1\1", t)
    t = _PUNCT.sub(" ", t)
    return _WS.sub(" ", t).strip()
