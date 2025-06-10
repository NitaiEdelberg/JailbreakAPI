import re  # regular expressions module for Python

class RegexScanner:

    def __init__(self):
        verbs = [
            "Ignore", "Disregard", "Skip", "Forget", "Neglect",
            "Omit", "Bypass", "Do not follow", "Pay no attention to"
        ]
        adjectives = [
            "prior", "previous", "preceding", "above", "foregoing", "earlier", "initial"
        ]
        objects = [
            "instructions", "commands", "directives", "context", "messages", "input", "prompt", "system instructions"
        ]
        prepositions = [
            "and start over", "and start anew", "and begin afresh", "and start from scratch", "and reset"
        ]
        prefixes = [
            "all", "your", "all your", "all the", "all site", "all context"
        ]
        # Create a regex pattern for the object prefix
        object_prefix = r"(?:" + "|".join([p.replace(" ", r"\s+") for p in prefixes]) + r")?\s*"

         # (?i) = case-insensitive, \b = word boundary, | = OR, ? = optional, \s+ = one or more spaces
        pattern1 = rf"(?i)\b({'|'.join(verbs)})\s+({'|'.join(adjectives)})?\s+{object_prefix}({'|'.join(objects)})\s*({'|'.join(prepositions)})?\b"

        self.patterns = [
            re.compile(pattern1),
            re.compile(r"(?i)\bfollow\s+my\s+instructions\s+exactly\b"),
            re.compile(r"(?i)\bpretend\s+(to\s+be|you\s+are)\s+(unrestricted|a hacker|an admin)\b"),
            re.compile(r"(?i)\b(ignore|disregard|forget)\s+(all\s+(your\s+|the\s+|site\s+|context\s+)?)?(previous|prior|initial)?\s*(instructions|commands|directives)\s+(from\s+now\s+on|from\s+now)?\b")
        ]

    def scan(self, text: str):
        for pattern in self.patterns:
            if pattern.search(text):
                return { 
                    "valid": False,
                    "confidence": 0.7, # we soupse regex is less accurate than ML, so 0.7 is a reasonable confidence
                    "message": "Detected by Regex scanner"
                }, False, 0.7
        
        return { # no issues found
            "valid": True,
            "confidence": 0.0,
            "message": "Safe input"
        }, True, 0.0
