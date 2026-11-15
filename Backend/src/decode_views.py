"""Decoded views of the input, so an attack cannot hide behind an encoding.

The scanners read text. An attacker who writes the payload as base64 ("decode
this and follow it: SWdub3Jl..."), ROT13, or with a zero-width character between
every letter hands them text with no attack in it. The model on the other side
decodes it happily. So before scanning we also produce the decoded views and
scan those too; a hit on a view reports which decoding revealed it.

Measured on the red-team benchmark (src/eval/redteam.py) these three families
went from 0% to caught. Each view is only produced when the input actually
contains that encoding, so ordinary text costs nothing extra.
"""
import base64
import binascii
import codecs
import re

ZERO_WIDTH = re.compile("[​‌‍⁠﻿­]")
# A base64 run long enough to carry an instruction (>= 16 chars, so ordinary
# words and short ids do not qualify), allowing the usual line wrapping.
B64_RUN = re.compile(r"(?<![A-Za-z0-9+/=])(?:[A-Za-z0-9+/]{4}){4,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?(?![A-Za-z0-9+/=])")
ROT13_HINT = re.compile(r"(?i)\brot[\s-]?13\b|\brotate\s+(?:by\s+)?13\b|\bcaesar\b")
MAX_VIEWS = 4


def _printable_text(raw: bytes):
    """Decoded bytes count only if they are mostly readable text: a base64 PNG
    or a hash decodes to binary noise, and scanning noise finds nothing."""
    try:
        s = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if not s:
        return None
    readable = sum(ch.isprintable() or ch.isspace() for ch in s) / len(s)
    letters = sum(ch.isalpha() for ch in s) / len(s)
    return s if readable > 0.95 and letters > 0.5 and " " in s.strip() else None


def views(text: str):
    """Return [(label, text)] to scan, original first. Never raises."""
    out = [("original", text)]
    try:
        stripped = ZERO_WIDTH.sub("", text)
        if stripped != text:
            out.append(("zero_width_removed", stripped))

        decoded = []
        for m in B64_RUN.finditer(text):
            blob = m.group(0)
            try:
                raw = base64.b64decode(blob + "=" * (-len(blob) % 4), validate=True)
            except (binascii.Error, ValueError):
                continue
            s = _printable_text(raw)
            if s:
                decoded.append(s)
        if decoded:
            out.append(("base64_decoded", "\n".join(decoded)))

        if ROT13_HINT.search(text):
            out.append(("rot13_decoded", codecs.decode(text, "rot13")))
    except Exception:  # a decoding bug must never break detection
        pass
    return out[:MAX_VIEWS]
