import logging
import os

from our_scanner import CustomMLScanner
from regex_scanner import RegexScanner
from indirect_scanner import IndirectInjectionScanner

# Every scanner exposes the same tiny contract:
#   .name                       -> str
#   .scan(text) -> {"flagged": bool, "risk": float 0..1, "matched": str | None}
# so the service can run them all and aggregate uniformly.

# The rule layers have no heavy deps and always load. The indirect-injection
# scanner only adds content-specific rules for non-"user" sources (see
# indirect_scanner.py), so plain prompts score exactly as before.
scanners = [RegexScanner(), IndirectInjectionScanner()]

# Core scanners that failed to load, with the reason. The API keeps serving
# without them, but /health reports "degraded" so a silently weaker detector
# (regex-only) is visible instead of quietly calling attacks "safe".
unavailable = {}

# Our ML model needs scikit-learn and a matching our_scanner.pkl. If the pickle
# is missing or was trained with a different sklearn version (unfitted vectorizer),
# construction raises — don't let that take down the whole API on boot. Degrade to
# regex-only, same philosophy as the optional scanners below.
try:
    scanners.append(CustomMLScanner())
    logging.info("Loaded CustomMLScanner (ML model)")
except Exception as e:
    unavailable["CustomMLScanner"] = f"{type(e).__name__}: {e}"[:300]
    logging.error(f"CustomMLScanner unavailable, running DEGRADED (regex only): {e}")


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

# Our own small fine-tuned encoder (see transformer_scanner.py). Off unless asked
# for: even an int8 MiniLM plus onnxruntime is memory the free tier doesn't have
# to spare, and we would rather run three scanners well than four badly.
#   ENABLE_TRANSFORMER_SCANNER=1 TRANSFORMER_MODEL_DIR=/path/to/model
if os.getenv("ENABLE_TRANSFORMER_SCANNER", "").strip().lower() in {"1", "true", "yes"}:
    try:
        from transformer_scanner import TransformerScanner
        scanners.append(TransformerScanner())
        logging.info("Loaded TransformerScanner")
    except Exception as e:
        logging.warning(f"TransformerScanner unavailable, continuing without it: {e}")

# can add more scanners here if needed
