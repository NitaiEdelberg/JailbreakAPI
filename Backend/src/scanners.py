from llm_guard.input_scanners import PromptInjection
from src.our_scanner import CustomMLScanner

#PromptInjection() is removed for testing reasons

scanners = [CustomMLScanner(), PromptInjection()]


          