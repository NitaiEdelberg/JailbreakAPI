"""jailbreak-guard: screen prompts, documents and tool output with JailbreakAPI."""
from .client import Guard, GuardError, GuardUnavailable, Result
from .integrations import PromptInjectionBlocked, fastapi_guard, langchain_document_filter, langchain_guard

__all__ = ["Guard", "GuardError", "GuardUnavailable", "Result", "PromptInjectionBlocked",
           "fastapi_guard", "langchain_guard", "langchain_document_filter"]
__version__ = "0.1.0"
