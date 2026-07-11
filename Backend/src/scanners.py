import logging
from our_scanner import CustomMLScanner
from regex_scanner import RegexScanner

# Lightweight scanners (our own regex + ML model) — always available.
scanners = [RegexScanner(), CustomMLScanner()]

# llm-guard's PromptInjection loads a large transformer model, which can OOM or
# time out on small/free hosts (e.g. Render free = 512MB RAM). Load it lazily and
# degrade gracefully: if it can't load, the API still runs with the scanners above.
try:
    from llm_guard.input_scanners import PromptInjection
    scanners.append(PromptInjection())
    logging.info("Loaded llm-guard PromptInjection scanner")
except Exception as e:  # ImportError, OOM at model load, download failure, ...
    logging.warning(f"PromptInjection scanner unavailable, continuing without it: {e}")

# can add more scanners here if needed
