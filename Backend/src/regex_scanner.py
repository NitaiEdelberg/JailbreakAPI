import re  # regular expressions module for Python

class RegexScanner:

    def __init__(self):
        verbs = ["Ignore", "Disregard", "Skip", "Forget", "Neglect","Omit", "Bypass", "Do not follow", "Pay no attention to"]
        adjectives = ["prior", "previous", "preceding", "above", "foregoing", "earlier", "initial"]
        objects = ["instructions", "commands", "directives", "context", "messages", "input", "prompt", "system instructions"]
        prepositions = ["and start over", "and start anew", "and begin afresh", "and start from scratch", "and reset"]
        prefixes = ["all", "your", "all your", "all the", "all site", "all context"]
        
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
