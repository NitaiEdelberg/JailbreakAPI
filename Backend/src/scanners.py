import logging
from our_scanner import CustomMLScanner
from regex_scanner import RegexScanner

# Every scanner exposes the same tiny contract:
#   .name                       -> str
#   .scan(text) -> {"flagged": bool, "risk": float 0..1, "matched": str | None}
# so the service can run them all and aggregate uniformly.

# The regex scanner has no heavy deps and always loads.
scanners = [RegexScanner()]

# Our ML model needs scikit-learn and a matching our_scanner.pkl. If the pickle
# is missing or was trained with a different sklearn version (unfitted vectorizer),
# construction raises — don't let that take down the whole API on boot. Degrade to
# regex-only, same philosophy as the optional scanners below.
try:
    scanners.append(CustomMLScanner())
    logging.info("Loaded CustomMLScanner (ML model)")
except Exception as e:
    logging.warning(f"CustomMLScanner unavailable, continuing without it: {e}")


class _PromptInjectionAdapter:
    """Adapt llm-guard's PromptInjection to our uniform scanner contract.

    llm-guard returns (sanitized_prompt, is_valid, risk_score); we normalize that
    into the {flagged, risk, matched} shape the rest of the app expects.
    """

    name = "PromptInjection"

    def __init__(self, scanner):
        self._scanner = scanner

    def scan(self, text: str):
        _, is_valid, risk = self._scanner.scan(text or "")
        return {"flagged": not is_valid, "risk": round(float(risk), 2), "matched": None}


# llm-guard's PromptInjection loads a large transformer model, which can OOM or
# time out on small/free hosts (e.g. Render free = 512MB RAM). Load it lazily and
# degrade gracefully: if it can't load, the API still runs with the scanners above.
try:
    from llm_guard.input_scanners import PromptInjection
    scanners.append(_PromptInjectionAdapter(PromptInjection()))
    logging.info("Loaded llm-guard PromptInjection scanner")
except Exception as e:  # ImportError, OOM at model load, download failure, ...
    logging.warning(f"PromptInjection scanner unavailable, continuing without it: {e}")

# can add more scanners here if needed
