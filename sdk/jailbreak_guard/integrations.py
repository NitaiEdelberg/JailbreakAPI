"""Drop-in guards for FastAPI and LangChain. Each import is lazy, so the core
client never needs either framework installed."""
from typing import Callable, Optional

from .client import Guard, Result


class PromptInjectionBlocked(Exception):
    """Raised by the LangChain guard when the input is malicious."""

    def __init__(self, result: Result):
        self.result = result
        super().__init__("blocked by JailbreakAPI: " + "; ".join(result.reasons or [result.verdict]))


def fastapi_guard(guard: Guard, field: str = "message", source: str = "user", block_suspicious: bool = False,
                  status_code: int = 400) -> Callable:
    """A FastAPI dependency that screens one string field of the JSON body.

        guard = Guard()
        @app.post("/chat", dependencies=[Depends(fastapi_guard(guard, field="message"))])
        def chat(body: ChatIn): ...

    Malicious input never reaches the handler; the client gets `status_code`
    with the verdict and reasons. If the guard is unreachable the Guard's own
    fail_open setting decides: closed returns 503, open lets the request through.
    """
    from fastapi import HTTPException, Request

    from .client import GuardUnavailable

    async def dependency(request: Request) -> Optional[Result]:
        try:
            body = await request.json()
        except Exception:
            return None
        text = body.get(field) if isinstance(body, dict) else None
        if not isinstance(text, str) or not text.strip():
            return None
        try:
            # The client is synchronous (stdlib only); keep it off the event loop.
            from starlette.concurrency import run_in_threadpool
            result = await run_in_threadpool(guard.check, text, source)
        except GuardUnavailable:
            raise HTTPException(503, detail={"error": "prompt-injection guard unavailable"})
        if result.is_malicious or (block_suspicious and result.is_suspicious):
            raise HTTPException(status_code, detail={"error": "prompt injection detected",
                                                     "verdict": result.verdict, "reasons": result.reasons})
        request.state.injection_check = result
        return result

    return dependency


def langchain_guard(guard: Guard, source: str = "user", block_suspicious: bool = False):
    """A LangChain Runnable that passes its input through unchanged, or raises
    PromptInjectionBlocked. Put it first in a chain:

        chain = langchain_guard(guard) | prompt | llm
    """
    from langchain_core.runnables import RunnableLambda

    def _check(value):
        text = value if isinstance(value, str) else (
            value.get("input") or value.get("question") or value.get("query") if isinstance(value, dict) else str(value))
        result = guard.check(str(text or ""), source=source)
        if result.is_malicious or (block_suspicious and result.is_suspicious):
            raise PromptInjectionBlocked(result)
        return value

    return RunnableLambda(_check, name="JailbreakGuard")


def langchain_document_filter(guard: Guard, source: str = "rag", drop_suspicious: bool = False):
    """A LangChain Runnable for retrieved Documents: drops the poisoned ones
    before they are stuffed into the prompt.

        chain = retriever | langchain_document_filter(guard) | format_docs | prompt | llm
    """
    from langchain_core.runnables import RunnableLambda

    def _filter(docs):
        kept, _ = guard.filter_documents(list(docs), source=source, drop_suspicious=drop_suspicious)
        return kept

    return RunnableLambda(_filter, name="JailbreakDocumentFilter")
