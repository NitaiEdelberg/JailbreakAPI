from llm_guard.input_scanners import PromptInjection
from our_scanner import CustomMLScanner
from regex_scanner import RegexScanner

# can add more scanners here if needed

scanners = [RegexScanner(),CustomMLScanner(), PromptInjection()]


          