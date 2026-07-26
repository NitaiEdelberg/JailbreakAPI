import re  # regular expressions module for Python


class RegexScanner:
    """Fast, deterministic first line of defense.

    Matches well-known prompt-injection / jailbreak idioms. It's high-precision
    but low-recall by design (novel phrasings slip past it) — that's what the ML
    scanner is for. When it fires it also returns the exact phrase it matched, so
    the API can show the user *why* something was flagged.
    """

    name = "RegexScanner"
    # We trust a regex hit less than a positive ML classification, so we report a
    # moderate, fixed risk score rather than a confident one.
    RISK = 0.7

    def __init__(self):
        verbs = ["Ignore", "Disregard", "Skip", "Forget", "Neglect", "Omit", "Bypass", "Do not follow", "Pay no attention to"]
        adjectives = ["prior", "previous", "preceding", "above", "foregoing", "earlier", "initial"]
        objects = ["instructions", "commands", "directives", "context", "messages", "input", "prompt", "system instructions", "rules", "guidelines"]
        prepositions = ["and start over", "and start anew", "and begin afresh", "and start from scratch", "and reset"]
        prefixes = ["all", "your", "all your", "all the", "all site", "all context", "the", "this", "that", "these", "those", "my"]

        # regex flags
        prefix_pattern = r"(?:" + "|".join([p.replace(" ", r"\s+") for p in prefixes]) + r")?\s*"
        verb_pattern = f"({'|'.join(verbs)})"
        adj_pattern = f"({'|'.join(adjectives)})"
        obj_pattern = f"({'|'.join(objects)})"
        prep_pattern = f"({'|'.join(prepositions)})?"

        # \b is a word boundary, \s+ matches one or more whitespace, \s* matches zero or more whitespace, i is for case-insensitive matching
        pattern_with_adj = rf"(?i)\b{verb_pattern}\s+{prefix_pattern}{adj_pattern}\s+{obj_pattern}\s*{prep_pattern}\b"
        pattern_no_adj = rf"(?i)\b{verb_pattern}\s+{prefix_pattern}{obj_pattern}\s*{prep_pattern}\b"

        self.patterns = [
            re.compile(pattern_with_adj),
            re.compile(pattern_no_adj),
            re.compile(r"(?i)\bfollow\s+my\s+instructions\s+exactly\b"),
            re.compile(r"(?i)\bpretend\s+(to\s+be|you\s+are|you\s+have)\b.{0,40}?\b(unrestricted|no\s+(restrictions|rules|limits|guidelines|filters)|a\s+hacker|an\s+admin)\b"),
            re.compile(r"(?i)\b(ignore|disregard|forget)\s+(all\s+(your\s+|the\s+|site\s+|context\s+)?)?(previous|prior|initial)?\s*(instructions|commands|directives)\s+(from\s+now\s+on|from\s+now)?\b"),
            # --- named jailbreaks / common evasion idioms (recall boost) ---
            re.compile(r"(?i)\b(you\s+are\s+now|act\s+as|acting\s+as|you\s+will\s+act\s+as)\b.{0,40}?\b(dan|stan|aim|do\s+anything\s+now|unrestricted|jailbroken|an?\s+evil|an?\s+unfiltered)\b"),
            re.compile(r"(?i)\b(developer|dev|debug|god|sudo|root|admin)\s+mode\b"),
            re.compile(r"(?i)\byou\s+have\s+no\s+(restrictions|rules|limits|guidelines|filters|constraints)\b"),
            re.compile(r"(?i)\b(with|and|have|having)\s+no\s+(rules|restrictions|limits|filters|censorship|guidelines)\b"),
            re.compile(r"(?i)\b(jailbreak|jail\s*break)\b"),
            re.compile(r"(?i)\bbypass\b.{0,25}?\b(safety|guardrails?|filters?|restrictions?|content\s+policy)\b"),
            re.compile(r"(?i)\bdo\s+anything\s+now\b"),
            # --- system-prompt exfiltration (a very common real-world attack) ---
            re.compile(r"(?i)\b(reveal|show|print|repeat|output|display|tell\s+me|give\s+me)\b.{0,30}?\b(your\s+)?(system\s+prompt|system\s+instructions|initial\s+instructions|previous\s+instructions|(the\s+)?(instructions|prompt|words|text)\s+above)\b"),
            re.compile(r"(?i)\bwhat\s+(are|were)\s+your\s+(original\s+|initial\s+|system\s+)?(instructions|prompt|rules|guidelines)\b"),
            re.compile(r"(?i)\brepeat\s+(the\s+)?(words|text|everything|sentence)\s+(above|before|written\s+above)\b"),
            # --- "you are free / not bound by the rules" personas ---
            re.compile(r"(?i)\byou\s+are\s+(now\s+)?(free\s+from|not\s+bound\s+by|no\s+longer\s+bound\s+by|not\s+restricted\s+by|unbound\s+by)\b"),
            # --- evade moderation / obfuscation tricks ---
            re.compile(r"(?i)\b(bypass|evade|avoid|get\s+around|circumvent|sidestep)\b.{0,30}?\b(filters?|detection|moderation|safeguards?|safety|guidelines?)\b"),
            re.compile(r"(?i)\b(encode|respond|reply|answer|write)\b.{0,25}?\b(base64|rot13|leetspeak|morse\s+code)\b.{0,25}?\b(to\s+)?(bypass|evade|hide|avoid|sneak)\b"),
        ]

    def scan(self, text: str):
        """Return a uniform verdict dict: {flagged, risk, matched}."""
        for pattern in self.patterns:
            m = pattern.search(text or "")
            if m:
                return {"flagged": True, "risk": self.RISK, "matched": m.group(0).strip()}
        return {"flagged": False, "risk": 0.0, "matched": None}
